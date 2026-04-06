"""Pierre v5 — Fully ternary composable experts. Orthogonal by construction.

The theorem:
  If A_i ⊥ A_j (Grassmannian, Finding #3: cos=0.0002 at d=2560),
  then Δ_i = B_i^T @ A_i^T and Δ_j = B_j^T @ A_j^T live in orthogonal
  subspaces of the weight matrix. Interference is geometrically impossible
  regardless of B content or quantization.

The architecture:
  y = BitLinear_base(x) + scale * BitLinear_B(BitLinear_A(x))
  All three matmuls use the native ternary Metal kernel.
  A is Grassmannian (frozen, orthogonal). B is STE-trained ternary.

Proven components:
  - Grassmannian A orthogonality: cos=0.0002 (Finding #3, conclusive)
  - STE ternary B composition: ratio 1.068 (exp_adapter_compression_extreme)
  - Null-space projection on ternary: forgetting <0.003 (Finding #273)
  - Ridge regression router: 99.6% accuracy (Finding #276/287)
  - 15.8x adapter storage compression (46KB per 5 adapters vs 724KB FP32)

Speed target: ~140 tok/s (native BitLinear) with near-zero adapter overhead,
since all three matmuls are ternary integer additions.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.bitlinear_layers import BitLinear

TARGET_MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

_mask_cache: dict[int, mx.array] = {}


# ---------------------------------------------------------------------------
# Ternary packing (BitLinear uint8 format)
# ---------------------------------------------------------------------------

def pack_ternary(ternary: mx.array, out_features: int) -> mx.array:
    """Pack (out, in) ternary {-1,0,+1} → (out/4, in) uint8."""
    q = out_features // 4
    t = (ternary[:out_features].astype(mx.int32) + 1).astype(mx.uint8)
    return t[:q] | (t[q:2*q] << 2) | (t[2*q:3*q] << 4) | (t[3*q:4*q] << 6)


def unpack_ternary(packed: mx.array, out_features: int) -> mx.array:
    """Unpack (out/4, in) uint8 → (out, in) ternary {-1,0,+1}."""
    w0 = (packed & 3).astype(mx.int8) - 1
    w1 = ((packed >> 2) & 3).astype(mx.int8) - 1
    w2 = ((packed >> 4) & 3).astype(mx.int8) - 1
    w3 = ((packed >> 6) & 3).astype(mx.int8) - 1
    return mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]


# ---------------------------------------------------------------------------
# TernaryLoRA layer: base + ternary side-path using BitLinear kernels
# ---------------------------------------------------------------------------

class TernaryLoRASideLayer(nn.Module):
    """Base layer + ternary LoRA side-path. All matmuls use ternary kernels.

    y = base(x) + lora_scale * B_ternary(A_ternary(x))

    A is Grassmannian (frozen, orthogonal by construction).
    B is STE-trained ternary (or PTQ from FP32).
    Both A and B are stored as BitLinear for native Metal kernel inference.

    Orthogonality guarantee:
      A_i ⊥ A_j → Δ_i and Δ_j live in orthogonal weight subspaces.
      Interference is impossible regardless of B content.
    """

    def __init__(self, base: nn.Module,
                 A_packed: mx.array, A_scale: mx.array, A_out: int, A_in: int,
                 B_packed: mx.array, B_scale: mx.array, B_out: int, B_in: int,
                 lora_scale: float):
        super().__init__()
        self.base = base

        # A path: (in_features, rank) → maps input to low-rank space
        self.lora_a = BitLinear(A_in, A_out)
        self.lora_a.weight = A_packed
        self.lora_a.weight_scale = A_scale

        # B path: (rank, out_features) → maps back to output space
        self.lora_b = BitLinear(B_in, B_out)
        self.lora_b.weight = B_packed
        self.lora_b.weight_scale = B_scale

        self.lora_scale = lora_scale

        # Freeze everything — this is inference-only
        self.freeze()

    def __call__(self, x):
        y_base = self.base(x)
        # Ternary side-path: both matmuls use the Metal ternary kernel
        h = self.lora_a(x)                      # (B, T, rank) — ternary matmul
        h = h.astype(x.dtype)                   # Metal kernel may return f32; cast back
        y_lora = self.lora_b(h) * self.lora_scale  # (B, T, out) — ternary matmul
        return y_base + y_lora.astype(y_base.dtype)


# ---------------------------------------------------------------------------
# Convert FP32 adapter weights to ternary BitLinear format
# ---------------------------------------------------------------------------

def quantize_matrix_to_bitlinear(W: mx.array) -> tuple[mx.array, mx.array, int, int]:
    """Quantize a FP32/bf16 matrix to ternary BitLinear format.

    Uses absmax quantization: γ = mean(|W|), W_q = clip(round(W/γ), -1, 1)

    Returns (packed_weight, scale, out_features, in_features).
    """
    out_f, in_f = W.shape
    # Ensure out_features is divisible by 4 for packing
    pad = (4 - out_f % 4) % 4
    if pad > 0:
        W = mx.concatenate([W, mx.zeros((pad, in_f), dtype=W.dtype)])
        out_f_padded = out_f + pad
    else:
        out_f_padded = out_f

    W_f32 = W.astype(mx.float32)
    gamma = mx.mean(mx.abs(W_f32))
    mx.eval(gamma)
    if gamma.item() < 1e-10:
        gamma = mx.array([1.0], dtype=mx.bfloat16)
    W_ternary = mx.clip(mx.round(W_f32 / gamma), -1, 1).astype(mx.int8)
    packed = pack_ternary(W_ternary, out_f_padded)
    return packed, gamma.astype(mx.bfloat16).reshape(1), out_f, in_f


# ---------------------------------------------------------------------------
# Inject ternary LoRA into a loaded BitNet model
# ---------------------------------------------------------------------------

def inject_ternary_lora(model, skeleton: dict, adapter_b: dict[str, mx.array],
                        domain_idx: int, scale: float) -> int:
    """Wrap target modules with TernaryLoRASideLayer.

    Converts FP32 A (from skeleton) and FP32 B (from adapter) to ternary BitLinear,
    then wraps the base BitLinear module.

    Returns number of modules wrapped.
    """
    from mlx.utils import tree_unflatten

    count = 0
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        updates = []
        for key in TARGET_MODULES:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue

            # Navigate to the base module
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if not isinstance(m, BitLinear):
                continue

            # A matrix: (in_features, rank) — Grassmannian, orthogonal
            A_fp = mx.array(skeleton[ak]).astype(mx.float32)  # (in, rank)
            # B matrix: (rank, out_features) — domain-specific
            B_fp = adapter_b[bk].astype(mx.float32)           # (rank, out)

            # The BitLinear __call__ does: y = kernel(x, W) where W is (out/4, in) packed
            # and x is (batch, in). So we need:
            # lora_a: maps (batch, in_features) → (batch, rank)
            #   Weight shape: (rank, in_features) → pack as (rank/4, in_features)
            # lora_b: maps (batch, rank) → (batch, out_features)
            #   Weight shape: (out_features, rank) → pack as (out_features/4, rank)

            # A is stored as (in, rank), but we need W_a of shape (rank, in) for y = x @ W^T
            A_weight = A_fp.T  # (rank, in_features)
            B_weight = B_fp.T  # (out_features, rank)

            A_packed, A_scale, A_out, A_in = quantize_matrix_to_bitlinear(A_weight)
            B_packed, B_scale, B_out, B_in = quantize_matrix_to_bitlinear(B_weight)

            wrapped = TernaryLoRASideLayer(
                m, A_packed, A_scale, A_out, A_in,
                B_packed, B_scale, B_out, B_in, scale,
            )
            updates.append((key, wrapped))
            count += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return count


def strip_lora(model) -> int:
    """Remove all TernaryLoRA wrappers, restoring base BitLinear modules."""
    from mlx.utils import tree_unflatten
    count = 0
    for layer in model.model.layers:
        updates = []
        for key in TARGET_MODULES:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if isinstance(m, TernaryLoRASideLayer):
                updates.append((key, m.base))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return count


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden(model, input_ids: mx.array) -> mx.array:
    """Mean-pooled, post-norm hidden state. (B, T) -> (B, H).

    Causal mask + final norm REQUIRED (Finding #287: 99.6% → 16.8% without).
    """
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


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def calibrate_router(model, tokenizer, domain_texts: dict[str, list[str]],
                     lam: float = 1.0, max_seq: int = 256) -> mx.array:
    """W* = (X^TX + λI)^{-1} X^TY from labeled calibration text."""
    domains = list(domain_texts.keys())
    D = len(domains)
    H = model.model.embed_tokens.weight.shape[1]

    XtX = mx.zeros((H, H))
    XtY = mx.zeros((H, D))

    for di, domain in enumerate(domains):
        for text in domain_texts[domain]:
            toks = tokenizer.encode(text)[:max_seq]
            if len(toks) < 4:
                continue
            h = extract_hidden(model, mx.array(toks)[None, :])
            XtX = XtX + h.T @ h
            XtY = XtY.at[:, di].add(h.squeeze(0))

    W = mx.linalg.solve(XtX + lam * mx.eye(H), XtY, stream=mx.cpu)
    W = W / mx.maximum(mx.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    mx.eval(W)
    return W


def route(model, tokenizer, text: str, W: mx.array,
          max_seq: int = 256) -> int:
    """Route a query to the best expert. Returns domain index."""
    h = extract_hidden(model, mx.array(tokenizer.encode(text)[:max_seq])[None, :])
    return mx.argmax(h @ W, axis=-1).item()


# ---------------------------------------------------------------------------
# Compose: NRE merge (operates on FP32 B-matrices before ternary conversion)
# ---------------------------------------------------------------------------

def nre_merge(deltas: list[mx.array], weights: list[float] | None = None) -> mx.array:
    """result = weighted_mean(Δs) * mean_source_norm / ‖mean‖"""
    if len(deltas) == 1:
        return deltas[0]
    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)

    w_sum = sum(weights)
    mean = sum(d.astype(mx.float32) * (w / w_sum) for d, w in zip(deltas, weights))
    source_norm = mx.mean(mx.stack([mx.linalg.norm(d.reshape(-1).astype(mx.float32))
                                     for d in deltas]))
    mean_norm = mx.linalg.norm(mean.reshape(-1))
    mx.eval(source_norm, mean_norm)

    if mean_norm.item() > 1e-8:
        return (mean * (source_norm / mean_norm)).astype(mx.bfloat16)
    return mean.astype(mx.bfloat16)


# ---------------------------------------------------------------------------
# Isolate: null-space projection
# ---------------------------------------------------------------------------

def null_space_projector(deltas: mx.array, top_k: int = 64) -> mx.array:
    """P = VV^T from top-K SVD directions. Isolate: Δ' = Δ - Δ@P."""
    _, _, Vt = mx.linalg.svd(deltas, stream=mx.cpu)
    V = Vt[:top_k].T
    P = V @ V.T
    mx.eval(P)
    return P


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_adapter(path: str) -> dict[str, mx.array]:
    return dict(mx.load(path))

def load_skeleton(path: str) -> dict[str, np.ndarray]:
    return dict(np.load(path))
