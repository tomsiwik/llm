"""Pierre v4 — Fully ternary composable experts. Zero inference overhead.

v2: premerge into bfloat16 nn.Linear → 47 tok/s, 0% overhead, but slow base
v3: BitLinear + bf16 LoRA side-path → 140 tok/s base, 73 tok/s with adapter (48% overhead)
v4: ternary premerge — merge LoRA into BitLinear, re-quantize, repack → 140 tok/s, 0% overhead

The key insight (BitDelta, arXiv:2402.10193): fine-tune deltas are ~1 bit.
Re-quantizing W_base + LoRA_delta back to ternary loses almost nothing.

Pipeline:
  1. Unpack BitLinear weights to bf16
  2. Add LoRA delta: W_merged = W_base + scale * B^T @ A^T
  3. Re-quantize to ternary: clip(round(W / γ), -1, 1) where γ = mean(|W|)
  4. Repack to uint8 in BitLinear format
  5. Inference uses native Metal ternary kernel — zero overhead

Three algorithms, each one line of math:
  Router:    W* = (X^TX + λI)^{-1} X^TY           (DUME, Finding #276)
  Compose:   merged = mean(Δs) * ‖source‖/‖mean‖  (NRE, Finding #275)
  Isolate:   Δ' = Δ - Δ @ VV^T                     (Brainstacks, Finding #273)
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
# Ternary pack/unpack — BitLinear uint8 format
# ---------------------------------------------------------------------------

def unpack_ternary(packed: mx.array, out_features: int) -> mx.array:
    """Unpack (out/4, in) uint8 → (out, in) ternary {-1, 0, +1} as int8."""
    w0 = (packed & 3).astype(mx.int8) - 1
    w1 = ((packed >> 2) & 3).astype(mx.int8) - 1
    w2 = ((packed >> 4) & 3).astype(mx.int8) - 1
    w3 = ((packed >> 6) & 3).astype(mx.int8) - 1
    return mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]


def pack_ternary(ternary: mx.array, out_features: int) -> mx.array:
    """Pack (out, in) ternary {-1, 0, +1} → (out/4, in) uint8.

    Inverse of unpack_ternary. Striped layout: rows 0..q-1 in bits 0-1,
    q..2q-1 in bits 2-3, 2q..3q-1 in bits 4-5, 3q..4q-1 in bits 6-7.
    """
    q = out_features // 4
    t = (ternary[:out_features].astype(mx.int32) + 1).astype(mx.uint8)
    return t[:q] | (t[q:2*q] << 2) | (t[2*q:3*q] << 4) | (t[3*q:4*q] << 6)


def quantize_to_ternary(W: mx.array) -> tuple[mx.array, mx.array]:
    """Quantize bf16/f32 weight matrix to ternary {-1, 0, +1} + scale.

    Uses absmax quantization (BitNet b1.58):
      γ = mean(|W|)
      W_ternary = clip(round(W / γ), -1, 1)

    Returns (W_ternary as int8, scale as bf16 scalar).
    """
    W_f32 = W.astype(mx.float32)
    gamma = mx.mean(mx.abs(W_f32))
    mx.eval(gamma)
    W_ternary = mx.clip(mx.round(W_f32 / gamma), -1, 1).astype(mx.int8)
    return W_ternary, gamma.astype(mx.bfloat16).reshape(1)


# ---------------------------------------------------------------------------
# Ternary premerge: LoRA → merge → re-quantize → repack
# ---------------------------------------------------------------------------

def ternary_premerge(model, skeleton: dict, adapter_b: dict[str, mx.array],
                     domain_idx: int, scale: float) -> int:
    """Merge LoRA delta into BitLinear weights, staying in ternary.

    1. Unpack base weights: uint8 → ternary → bf16 (apply scale)
    2. Add LoRA delta: W_merged = W_base + scale * B^T @ A^T
    3. Re-quantize: W_merged → ternary {-1, 0, +1} + new scale
    4. Repack: ternary → uint8

    After this, the model is standard BitLinear. Native Metal kernel. Zero overhead.
    Returns number of modules modified.
    """
    from mlx_lm.models.bitlinear_layers import BitLinear

    count = 0
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_MODULES:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_b or ak not in skeleton:
                continue

            # Navigate to BitLinear module
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if not isinstance(m, BitLinear):
                continue

            # 1. Unpack base weights to bf16
            W_ternary = unpack_ternary(m.weight, m.out_features)
            old_scale = m.weight_scale.astype(mx.bfloat16)
            if m.invert_weight_scales:
                W_bf16 = W_ternary.astype(mx.bfloat16) / old_scale
            else:
                W_bf16 = W_ternary.astype(mx.bfloat16) * old_scale

            # 2. Add LoRA delta: scale * B^T @ A^T
            A = mx.array(skeleton[ak]).astype(mx.bfloat16)
            B = adapter_b[bk].astype(mx.bfloat16)
            delta = scale * (B.T @ A.T)
            W_merged = W_bf16 + delta

            # 3. Re-quantize to ternary
            W_new_ternary, new_scale = quantize_to_ternary(W_merged)

            # 4. Repack to uint8
            packed = pack_ternary(W_new_ternary, m.out_features)

            # 5. Update BitLinear in-place
            m.weight = packed
            m.weight_scale = new_scale
            # After re-quantization with our own scale, no inversion needed
            m.invert_weight_scales = False
            count += 1

    mx.eval(model.parameters())
    return count


def ternary_premerge_composed(model, skeleton: dict,
                              adapters: list[dict[str, mx.array]],
                              domain_indices: list[int],
                              weights: list[float] | None,
                              scale: float) -> int:
    """NRE-compose multiple adapters, then ternary-premerge the result."""
    if len(adapters) == 1:
        return ternary_premerge(model, skeleton, adapters[0], domain_indices[0], scale)

    # Build per-adapter deltas, NRE merge, then merge into model
    from mlx_lm.models.bitlinear_layers import BitLinear

    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)

    count = 0
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_MODULES:
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if not isinstance(m, BitLinear):
                continue

            # Compute per-adapter deltas for this module
            deltas = []
            for adapter_b, di in zip(adapters, domain_indices):
                bk = f"model.layers.{li}.{key}.lora_b"
                ak = f"layer_{li}_{key}_domain_{di}"
                if bk in adapter_b and ak in skeleton:
                    A = mx.array(skeleton[ak]).astype(mx.bfloat16)
                    B = adapter_b[bk].astype(mx.bfloat16)
                    deltas.append(scale * (B.T @ A.T))

            if not deltas:
                continue

            # NRE merge deltas
            merged_delta = nre_merge(deltas, weights[:len(deltas)])

            # Unpack, add, re-quantize, repack
            W_ternary = unpack_ternary(m.weight, m.out_features)
            old_scale = m.weight_scale.astype(mx.bfloat16)
            W_bf16 = W_ternary.astype(mx.bfloat16) * old_scale if not m.invert_weight_scales \
                else W_ternary.astype(mx.bfloat16) / old_scale
            W_merged = W_bf16 + merged_delta

            W_new_ternary, new_scale = quantize_to_ternary(W_merged)
            m.weight = pack_ternary(W_new_ternary, m.out_features)
            m.weight_scale = new_scale
            m.invert_weight_scales = False
            count += 1

    mx.eval(model.parameters())
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
# Compose: NRE merge
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
