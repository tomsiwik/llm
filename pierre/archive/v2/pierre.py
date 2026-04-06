"""Pierre — Composable Ternary Experts in one file.

Three algorithms, each one line of math:
  Router:    W* = (X^TX + λI)^{-1} X^TY           (DUME, Finding #276)
  Compose:   merged = mean(Δs) * ‖source‖/‖mean‖  (NRE, Finding #275)
  Isolate:   Δ' = Δ - Δ @ VV^T                     (Brainstacks, Finding #273)

Pre-merge: W_new = W_base + Σ scale_i * B_i^T @ A_i^T
After merge, inference = standard nn.Linear. Zero overhead.
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_MODULES = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Cache: seq_len -> causal mask. Avoids re-creating for repeated lengths.
_mask_cache: dict[int, mx.array] = {}


# ---------------------------------------------------------------------------
# Hidden state extraction (shared by router and any future probe)
# ---------------------------------------------------------------------------

def extract_hidden(model, input_ids: mx.array) -> mx.array:
    """Mean-pooled, post-norm hidden state. (B, T) -> (1, H).

    Causal mask + final norm are REQUIRED for domain separation.
    Without them routing accuracy drops from 99.6% to 16.8% (Finding #287).
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
    pooled = mx.mean(h, axis=1).astype(mx.float32)            # (B, H)
    mx.eval(pooled)
    return pooled


# ---------------------------------------------------------------------------
# Router: closed-form ridge regression on mean-pooled hidden states
# ---------------------------------------------------------------------------

def calibrate_router(model, tokenizer, domain_texts: dict[str, list[str]],
                     lam: float = 1.0, max_seq: int = 256) -> mx.array:
    """Build router W* from labeled calibration text.

    Forward each sample, mean-pool the post-norm hidden state,
    accumulate X^TX and X^TY, solve in closed form.

    Returns W: (H, D) column-normalized router weight matrix.
    """
    domains = list(domain_texts.keys())
    D = len(domains)
    H = model.model.layers[0].self_attn.q_proj.weight.shape[0]

    XtX = mx.zeros((H, H))
    XtY = mx.zeros((H, D))

    for di, domain in enumerate(domains):
        for text in domain_texts[domain]:
            toks = tokenizer.encode(text)[:max_seq]
            if len(toks) < 4:
                continue
            h = extract_hidden(model, mx.array(toks)[None, :])  # (1, H)
            XtX = XtX + h.T @ h
            XtY = XtY.at[:, di].add(h.squeeze(0))

    # W* = (X^TX + λI)^{-1} X^TY
    W = mx.linalg.solve(XtX + lam * mx.eye(H), XtY, stream=mx.cpu)

    # Column-normalize (DUME §2.3)
    W = W / mx.maximum(mx.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    mx.eval(W)
    return W


def route(model, tokenizer, text: str, W: mx.array,
          max_seq: int = 256) -> int:
    """Route a query to the best expert. Returns domain index."""
    h = extract_hidden(model, mx.array(tokenizer.encode(text)[:max_seq])[None, :])
    return mx.argmax(h @ W, axis=-1).item()


# ---------------------------------------------------------------------------
# Compose: NRE merge of LoRA deltas with norm preservation
# ---------------------------------------------------------------------------

def lora_delta(B: mx.array, A: mx.array, scale: float) -> mx.array:
    """Single adapter delta: scale * B^T @ A^T.

    Shapes: B (r, out), A (in, r) -> delta (out, in).
    """
    return scale * (B.astype(mx.bfloat16).T @ A.astype(mx.bfloat16).T)


def nre_merge(deltas: list[mx.array], weights: list[float] | None = None) -> mx.array:
    """Norm-Rescaled Euclidean merge of tensors.

    result = weighted_mean(deltas) * mean_source_norm / ‖weighted_mean‖

    Equivalent to Fisher-Rao Karcher mean (Finding #275) at zero iteration cost.
    """
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
# Isolate: null-space projection via SVD
# ---------------------------------------------------------------------------

def null_space_projector(deltas: mx.array, top_k: int = 64) -> mx.array:
    """P = VV^T from top-K SVD directions. To isolate: Δ' = Δ - Δ@P."""
    _, _, Vt = mx.linalg.svd(deltas, stream=mx.cpu)
    V = Vt[:top_k].T
    P = V @ V.T
    mx.eval(P)
    return P


# ---------------------------------------------------------------------------
# Adapter I/O
# ---------------------------------------------------------------------------

def load_adapter(path: str) -> dict[str, mx.array]:
    """Load adapter B-matrices from .npz."""
    return dict(mx.load(path))


def load_skeleton(path: str) -> dict[str, np.ndarray]:
    """Load Grassmannian A-matrices from .npz."""
    return dict(np.load(path))


# ---------------------------------------------------------------------------
# Delta computation and pre-merge
# ---------------------------------------------------------------------------

def build_deltas(adapter_b: dict[str, mx.array], skeleton: dict,
                 domain_idx: int, scale: float, n_layers: int,
                 ) -> dict[tuple[int, str], mx.array]:
    """Compute {(layer, module): scale * B^T @ A^T} for one adapter.

    Uses the correct per-domain A-matrix from the Grassmannian skeleton.
    """
    deltas = {}
    for li in range(n_layers):
        for key in TARGET_MODULES:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk in adapter_b and ak in skeleton:
                deltas[(li, key)] = lora_delta(adapter_b[bk], mx.array(skeleton[ak]), scale)
    return deltas


def merge_deltas(delta_dicts: list[dict[tuple[int, str], mx.array]],
                 weights: list[float] | None = None,
                 ) -> dict[tuple[int, str], mx.array]:
    """NRE-merge multiple adapter delta dicts into one."""
    if len(delta_dicts) == 1:
        return delta_dicts[0]

    all_keys = set().union(*(d.keys() for d in delta_dicts))
    merged = {}
    for k in all_keys:
        tensors = [d[k] for d in delta_dicts if k in d]
        merged[k] = nre_merge(tensors, weights[:len(tensors)] if weights else None)
    mx.eval(*merged.values())
    return merged


def premerge(model, deltas: dict[tuple[int, str], mx.array]) -> int:
    """W_base += delta for each (layer, module). Returns count modified."""
    count = 0
    for (li, key), delta in deltas.items():
        m = model.model.layers[li]
        for part in key.split("."):
            m = getattr(m, part, None)
            if m is None:
                break
        if m is not None and isinstance(m, nn.Linear):
            m.weight = m.weight + delta
            count += 1
    mx.eval(model.parameters())
    return count
