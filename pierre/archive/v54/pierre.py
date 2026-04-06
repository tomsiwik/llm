"""Pierre v5.4 — quantized_matmul side-path. Best of both worlds.

v5:   BitLinear side-path → 77 tok/s (fused kernel, but per-module eval overhead)
v5.3: bf16 lazy side-path → 61 tok/s (lazy eval, but unfused bf16 matmul slow)
v5.4: quantized_matmul lazy side-path → target ~120 tok/s

Key: mx.quantized_matmul does 2-bit packed matmul WITHOUT the BitLinear wrapper.
No per-module eval, no Python dispatch overhead, and 2x faster than bf16 matmul.

Benchmark:
  quantized_matmul (210 modules, 1 eval): 3.93ms
  bf16 matmul (210 modules, 1 eval):      7.79ms  ← 2x slower
  bf16 matmul (210 modules, per eval):     40.60ms ← 10x slower

Architecture: y = BitLinear_base(x) + scale * qmatmul_B(qmatmul_A(x))
  - Base: native BitLinear Metal kernel
  - Adapter A: 2-bit quantized via mx.quantize, run via mx.quantized_matmul
  - Adapter B: 2-bit quantized (padded to group_size=32), mx.quantized_matmul
  - Entire adapter graph is lazy — single eval at model output

Orthogonality: Grassmannian A_i ⊥ A_j (Finding #3, cos=0.0002).
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn

TARGET_MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

_mask_cache: dict[int, mx.array] = {}


# ---------------------------------------------------------------------------
# Quantized LoRA side-path — 2-bit via mx.quantized_matmul, fully lazy
# ---------------------------------------------------------------------------

class QuantizedLoRASideLayer(nn.Module):
    """Base layer + 2-bit quantized LoRA via mx.quantized_matmul.

    y = base(x) + scale * quantized_matmul_B(quantized_matmul_A(x))

    No BitLinear wrapper, no per-module eval. Entire adapter graph is lazy.
    2x faster than bf16 matmul, 10x faster than per-module-eval BitLinear.
    """

    def __init__(self, base: nn.Module,
                 A_q: mx.array, A_scales: mx.array, A_biases: mx.array,
                 B_q: mx.array, B_scales: mx.array, B_biases: mx.array,
                 B_padded: bool, lora_scale: float):
        super().__init__()
        self.base = base
        # Store quantized adapter matrices (frozen, not parameters)
        self._A_q = A_q
        self._A_scales = A_scales
        self._A_biases = A_biases
        self._B_q = B_q
        self._B_scales = B_scales
        self._B_biases = B_biases
        self._B_padded = B_padded
        self._lora_scale = lora_scale
        self.freeze()

    def __call__(self, x):
        y_base = self.base(x)

        # A path: x (B,T,in) @ A^T (in,rank) → h (B,T,rank)
        h = mx.quantized_matmul(x, self._A_q, scales=self._A_scales,
                                biases=self._A_biases, bits=2, group_size=64)

        # B path: h (B,T,rank) @ B^T (rank,out) → lora_out (B,T,out)
        # If B was padded (rank < 32), pad h to match
        if self._B_padded:
            h = mx.concatenate([h, mx.zeros_like(h)], axis=-1)

        lora_out = mx.quantized_matmul(h, self._B_q, scales=self._B_scales,
                                        biases=self._B_biases, bits=2, group_size=32)

        return y_base + (lora_out * self._lora_scale).astype(y_base.dtype)


# ---------------------------------------------------------------------------
# Inject quantized LoRA into model
# ---------------------------------------------------------------------------

def inject_quantized_lora(model, skeleton: dict, adapter_b: dict[str, mx.array],
                          domain_idx: int, scale: float) -> int:
    """Wrap target modules with QuantizedLoRASideLayer.

    Quantizes A (Grassmannian) and B (SFT adapter) to 2-bit packed format
    for use with mx.quantized_matmul.
    """
    from mlx.utils import tree_unflatten
    from mlx_lm.models.bitlinear_layers import BitLinear

    count = 0
    for li in range(len(model.model.layers)):
        updates = []
        for key in TARGET_MODULES:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue

            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None: break
            if m is None:
                continue

            # A matrix: (in_features, rank) → need (rank, in_features) for quantized_matmul
            A_fp = mx.array(skeleton[ak]).astype(mx.float32)  # (in, rank)
            A_weight = A_fp.T  # (rank, in)
            A_q, A_s, A_b = mx.quantize(A_weight, bits=2, group_size=64)

            # B matrix: (rank, out_features) → need (out_features, rank) for quantized_matmul
            B_fp = adapter_b[bk].astype(mx.float32)  # (rank, out)
            B_weight = B_fp.T  # (out, rank)

            # group_size minimum is 32. If rank < 32, pad B's input dimension
            rank = B_weight.shape[1]
            B_padded = False
            if rank < 32:
                pad_size = 32 - rank
                B_weight = mx.concatenate([B_weight, mx.zeros((B_weight.shape[0], pad_size))], axis=1)
                B_padded = True

            B_q, B_s, B_b = mx.quantize(B_weight, bits=2, group_size=32)

            wrapped = QuantizedLoRASideLayer(
                m, A_q, A_s, A_b, B_q, B_s, B_b, B_padded, scale
            )
            updates.append((key, wrapped))
            count += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return count


def strip_lora(model) -> int:
    from mlx.utils import tree_unflatten
    count = 0
    for layer in model.model.layers:
        updates = []
        for key in TARGET_MODULES:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None: break
            if isinstance(m, QuantizedLoRASideLayer):
                updates.append((key, m.base))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return count


# ---------------------------------------------------------------------------
# Hidden state + Router (identical to v5)
# ---------------------------------------------------------------------------

def extract_hidden(model, input_ids: mx.array) -> mx.array:
    T = input_ids.shape[1]
    if T not in _mask_cache:
        _mask_cache[T] = nn.MultiHeadAttention.create_additive_causal_mask(T)
    mask = _mask_cache[T].astype(mx.bfloat16)
    h = model.model.embed_tokens(input_ids)
    for layer in model.model.layers:
        h = layer(h, mask=mask)
    h = model.model.norm(h)
    mx.eval(h)
    return mx.mean(h, axis=1).astype(mx.float32)

def calibrate_router(model, tokenizer, domain_texts: dict[str, list[str]],
                     lam: float = 1.0, max_seq: int = 256) -> mx.array:
    domains = list(domain_texts.keys())
    D = len(domains)
    H = model.model.embed_tokens.weight.shape[1]
    XtX = mx.zeros((H, H))
    XtY = mx.zeros((H, D))
    for di, domain in enumerate(domains):
        for text in domain_texts[domain]:
            toks = tokenizer.encode(text)[:max_seq]
            if len(toks) < 4: continue
            h = extract_hidden(model, mx.array(toks)[None, :])
            XtX = XtX + h.T @ h
            XtY = XtY.at[:, di].add(h.squeeze(0))
    W = mx.linalg.solve(XtX + lam * mx.eye(H), XtY, stream=mx.cpu)
    W = W / mx.maximum(mx.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    mx.eval(W)
    return W

def route(model, tokenizer, text: str, W: mx.array, max_seq: int = 256) -> int:
    h = extract_hidden(model, mx.array(tokenizer.encode(text)[:max_seq])[None, :])
    return mx.argmax(h @ W, axis=-1).item()

def nre_merge(deltas, weights=None):
    if len(deltas) == 1: return deltas[0]
    if weights is None: weights = [1.0/len(deltas)] * len(deltas)
    w_sum = sum(weights)
    mean = sum(d.astype(mx.float32)*(w/w_sum) for d,w in zip(deltas, weights))
    sn = mx.mean(mx.stack([mx.linalg.norm(d.reshape(-1).astype(mx.float32)) for d in deltas]))
    mn = mx.linalg.norm(mean.reshape(-1)); mx.eval(sn, mn)
    return (mean*(sn/mn)).astype(mx.bfloat16) if mn.item() > 1e-8 else mean.astype(mx.bfloat16)

def null_space_projector(deltas, top_k=64):
    _, _, Vt = mx.linalg.svd(deltas, stream=mx.cpu)
    V = Vt[:top_k].T; P = V @ V.T; mx.eval(P); return P

def load_adapter(path): return dict(mx.load(path))
def load_skeleton(path): return dict(np.load(path))
