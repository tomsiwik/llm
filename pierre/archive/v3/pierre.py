"""Pierre v3 — Composable Ternary Experts as a model, not middleware.

v1: 4 files, 552 lines, patched model from outside
v2: 1 file, 200 lines, still patched model from outside (premerge)
v3: 1 file, keeps BitLinear native kernels, LoRA as side-path

Key change: instead of unpacking BitLinear → nn.Linear → W += delta,
we keep BitLinear and add LoRA as a parallel computation:

    y = BitLinear(x) + scale * (x @ A) @ B

This preserves the native ternary Metal kernel (3x bandwidth savings)
and adds only a small bf16 matmul for the adapter.

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
# LoRA side-path layer: wraps any nn.Module (including BitLinear)
# ---------------------------------------------------------------------------

class LoRASideLayer(nn.Module):
    """Wraps a base layer and adds LoRA as a parallel path.

    y = base(x) + scale * (x @ A) @ B

    The base layer is NEVER modified. Works with BitLinear (packed ternary),
    nn.Linear, or anything with __call__(x) -> y.
    """

    def __init__(self, base: nn.Module, A: mx.array, B: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.lora_a = A      # (in, r), frozen
        self.lora_b = B      # (r, out), composed/routed
        self.scale = scale
        self.freeze(keys=["base", "lora_a"], strict=False)

    def __call__(self, x):
        y = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return y + lora_out.astype(y.dtype)


# ---------------------------------------------------------------------------
# Model wrapping: inject LoRA side-paths into a loaded model
# ---------------------------------------------------------------------------

def inject_lora(model, skeleton: dict, adapter_b: dict[str, mx.array],
                domain_idx: int, scale: float) -> int:
    """Inject LoRA side-paths into model, keeping BitLinear intact.

    Replaces target modules with LoRASideLayer wrappers.
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

            # Get the base module (BitLinear or nn.Linear)
            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            A = mx.array(skeleton[ak]).astype(mx.bfloat16)
            B = adapter_b[bk].astype(mx.bfloat16)
            wrapped = LoRASideLayer(m, A, B, scale)
            updates.append((key, wrapped))
            count += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return count


def inject_composed_lora(model, skeleton: dict,
                         adapters: list[dict[str, mx.array]],
                         domain_indices: list[int],
                         weights: list[float] | None,
                         scale: float) -> int:
    """Inject NRE-composed LoRA from multiple adapters.

    Composes B-matrices via NRE merge, then injects single LoRASideLayer.
    """
    if len(adapters) == 1:
        return inject_lora(model, skeleton, adapters[0], domain_indices[0], scale)

    # Compose B-matrices per (layer, key) via NRE
    composed_b = {}
    n_layers = len(model.model.layers)
    for li in range(n_layers):
        for key in TARGET_MODULES:
            bk = f"model.layers.{li}.{key}.lora_b"
            bs = [a[bk] for a in adapters if bk in a]
            if bs:
                composed_b[bk] = nre_merge(bs, weights)

    # Use first domain's A-matrix (they're from the same Grassmannian)
    return inject_lora(model, skeleton, composed_b, domain_indices[0], scale)


def strip_lora(model) -> int:
    """Remove all LoRA side-paths, restoring original base modules."""
    from mlx.utils import tree_unflatten

    count = 0
    for li, layer in enumerate(model.model.layers):
        updates = []
        for key in TARGET_MODULES:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if isinstance(m, LoRASideLayer):
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
    return mx.mean(h, axis=1).astype(mx.float32)   # (B, H)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def calibrate_router(model, tokenizer, domain_texts: dict[str, list[str]],
                     lam: float = 1.0, max_seq: int = 256) -> mx.array:
    """W* = (X^TX + λI)^{-1} X^TY from labeled calibration text."""
    domains = list(domain_texts.keys())
    D = len(domains)
    # Use in_features (not weight.shape — packed BitLinear has shape[0] = out/4)
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


def route_topk(model, tokenizer, text: str, W: mx.array,
               k: int = 1, max_seq: int = 256) -> tuple[list[int], list[float]]:
    """Route to top-k experts. Returns (indices, weights)."""
    h = extract_hidden(model, mx.array(tokenizer.encode(text)[:max_seq])[None, :])
    logits = (h @ W).squeeze(0)
    probs = mx.softmax(logits, axis=-1)
    if k >= W.shape[1]:
        idx = list(range(W.shape[1]))
        wts = probs.tolist()
    else:
        idx_arr = mx.argpartition(-probs, kth=k)[:k]
        mx.eval(idx_arr)
        idx = idx_arr.tolist()
        wts = [probs[i].item() for i in idx]
        w_sum = sum(wts)
        wts = [w / w_sum for w in wts]
    return idx, wts


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
