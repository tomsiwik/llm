"""Pierre v5.3 — Lazy bf16 LoRA side-path. Zero eval overhead.

v5 used BitLinear wrappers for the LoRA side-path. Each BitLinear.__call__
forces mx.eval() internally — 420 evals per forward pass → 45% overhead.

v5.3 uses raw bf16 matmul for the side-path. No wrapper, no per-module eval.
The entire adapter computation is lazy — eval happens once at model output.

Benchmark: 420 per-module evals = 36ms. Single batched eval = 2ms. 17x faster.

Architecture: y = BitLinear_base(x) + scale * (x @ A_bf16) @ B_bf16
  - Base: native ternary Metal kernel (untouched)
  - Adapter: bf16 matmul, lazy evaluation (tiny matrices at rank 16)
  - Orthogonality: Grassmannian A_i ⊥ A_j (Finding #3)

Why bf16 for adapters is fine:
  - Rank 16 adapter: A is (2560, 16), B is (16, 2560) → 164KB bf16 total
  - The 2-bit quantized version saves ~120KB but adds padding + scale overhead
  - At rank 16, memory is not the bottleneck — eval latency is
  - v5 proved ternary adapters give PPL 3-8% BETTER than bf16 (regularization)
  - v5.3 trades that small PPL gain for 17x less eval overhead
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
# Lazy LoRA side-path — NO eval inside, NO wrapper overhead
# ---------------------------------------------------------------------------

class LazyLoRASideLayer(nn.Module):
    """Base layer + bf16 LoRA as lazy side computation.

    y = base(x) + scale * (x @ A) @ B

    Key difference from v5 TernaryLoRASideLayer:
    - A and B are bf16, not BitLinear
    - No mx.eval() inside — entire adapter graph is lazy
    - Eval happens once at model output, not 420 times per forward
    """

    def __init__(self, base: nn.Module, A: mx.array, B: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.lora_a = A.astype(mx.bfloat16)   # (in, rank), frozen
        self.lora_b = B.astype(mx.bfloat16)   # (rank, out), frozen
        self.scale = scale
        self.freeze()

    def __call__(self, x):
        y_base = self.base(x)
        # Lazy bf16 side-path — no eval, no wrapper, just matmul
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return y_base + lora_out.astype(y_base.dtype)


# ---------------------------------------------------------------------------
# Inject lazy LoRA into model
# ---------------------------------------------------------------------------

def inject_lazy_lora(model, skeleton: dict, adapter_b: dict[str, mx.array],
                     domain_idx: int, scale: float) -> int:
    """Wrap target modules with LazyLoRASideLayer. Returns count."""
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

            A = mx.array(skeleton[ak]).astype(mx.bfloat16)   # (in, rank)
            B = adapter_b[bk].astype(mx.bfloat16)            # (rank, out)

            wrapped = LazyLoRASideLayer(m, A, B, scale)
            updates.append((key, wrapped))
            count += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return count


def strip_lora(model) -> int:
    """Remove all LoRA wrappers."""
    from mlx.utils import tree_unflatten
    count = 0
    for layer in model.model.layers:
        updates = []
        for key in TARGET_MODULES:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None: break
            if isinstance(m, LazyLoRASideLayer):
                updates.append((key, m.base))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return count


# ---------------------------------------------------------------------------
# Hidden state + Router (same as v5)
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


# ---------------------------------------------------------------------------
# Compose + Isolate (same as v5)
# ---------------------------------------------------------------------------

def nre_merge(deltas: list[mx.array], weights: list[float] | None = None) -> mx.array:
    if len(deltas) == 1: return deltas[0]
    if weights is None: weights = [1.0 / len(deltas)] * len(deltas)
    w_sum = sum(weights)
    mean = sum(d.astype(mx.float32) * (w / w_sum) for d, w in zip(deltas, weights))
    source_norm = mx.mean(mx.stack([mx.linalg.norm(d.reshape(-1).astype(mx.float32)) for d in deltas]))
    mean_norm = mx.linalg.norm(mean.reshape(-1))
    mx.eval(source_norm, mean_norm)
    if mean_norm.item() > 1e-8:
        return (mean * (source_norm / mean_norm)).astype(mx.bfloat16)
    return mean.astype(mx.bfloat16)

def null_space_projector(deltas: mx.array, top_k: int = 64) -> mx.array:
    _, _, Vt = mx.linalg.svd(deltas, stream=mx.cpu)
    V = Vt[:top_k].T; P = V @ V.T; mx.eval(P); return P


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_adapter(path: str) -> dict[str, mx.array]:
    return dict(mx.load(path))

def load_skeleton(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path))
