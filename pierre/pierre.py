"""Pierre — Runtime LoRA on Ternary Language Models.

Composable domain adapters for BitNet b1.58 on Apple Silicon (MLX).

Architecture:
  y = BitLinear(x) + α · (x @ A) @ B

  - BitLinear: native 1.58-bit ternary kernel (140 tok/s, untouched)
  - A: frozen Grassmannian-orthogonal initialization (cos < 0.001 across adapters)
  - B: SFT-trained per-domain adapter weights
  - α: LoRA scaling factor

Composition:
  Multiple adapters compose via norm-rescaled averaging of B-matrices.
  Orthogonal A-matrices guarantee zero cross-adapter interference.

Components (literature references):
  Adapter:  Runtime LoRA (Hu et al. 2021, arXiv:2106.09685)
  Router:   Closed-form ridge regression (DUME, arXiv:2603.29765)
  Merge:    Norm-rescaled average (equivalent to Fisher-Rao Karcher mean)
  Isolate:  Null-space SVD projection (Brainstacks §3.5, arXiv:2604.01152)
  Init:     Grassmannian packing for orthogonal A-matrices

Verified results (Finding #288):
  Routing: 99.6% accuracy at N=5 domains
  Quality: 0.41 behavioral score (SFT adapters)
  Speed:   73 tok/s with adapters (native base: 140 tok/s)
  PPL:     0% degradation vs single-adapter (perfect routing)
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# Modules that receive LoRA adapters
ADAPTER_TARGETS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

# Causal mask cache (avoid re-creating for repeated sequence lengths)
_mask_cache: dict[int, mx.array] = {}


# ---------------------------------------------------------------------------
# Runtime LoRA: additive adapter on frozen base
# ---------------------------------------------------------------------------

class RuntimeLoRA(nn.Module):
    """Additive LoRA wrapper: y = base(x) + α · (x @ A) @ B

    The base module (BitLinear or nn.Linear) is never modified.
    A is frozen (Grassmannian orthogonal initialization).
    B is the trained domain-specific adapter.
    """

    def __init__(self, base: nn.Module, A: mx.array, B: mx.array, alpha: float):
        super().__init__()
        self.base = base
        self.lora_a = A.astype(mx.bfloat16)  # (in_features, rank)
        self.lora_b = B.astype(mx.bfloat16)  # (rank, out_features)
        self.alpha = alpha
        self.freeze(keys=["base", "lora_a"], strict=False)

    def __call__(self, x):
        y = self.base(x)
        return y + ((x @ self.lora_a) @ self.lora_b * self.alpha).astype(y.dtype)


# ---------------------------------------------------------------------------
# Adapter management: attach / detach / compose
# ---------------------------------------------------------------------------

def attach_adapter(model, frozen_A: dict, adapter_B: dict[str, mx.array],
                   domain_idx: int, alpha: float) -> int:
    """Attach a single domain adapter to the model.

    Wraps each target module with RuntimeLoRA. Base modules are untouched.
    Returns number of modules wrapped.
    """
    from mlx.utils import tree_unflatten

    count = 0
    for li in range(len(model.model.layers)):
        updates = []
        for key in ADAPTER_TARGETS:
            bk = f"model.layers.{li}.{key}.lora_b"
            ak = f"layer_{li}_{key}_domain_{domain_idx}"
            if bk not in adapter_B or ak not in frozen_A:
                continue

            m = model.model.layers[li]
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            A = mx.array(frozen_A[ak]).astype(mx.bfloat16)
            B = adapter_B[bk].astype(mx.bfloat16)
            updates.append((key, RuntimeLoRA(m, A, B, alpha)))
            count += 1

        if updates:
            model.model.layers[li].update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    return count


def detach_adapters(model) -> int:
    """Remove all adapters, restoring original base modules."""
    from mlx.utils import tree_unflatten

    count = 0
    for layer in model.model.layers:
        updates = []
        for key in ADAPTER_TARGETS:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if isinstance(m, RuntimeLoRA):
                updates.append((key, m.base))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    return count


def compose_adapters(adapter_Bs: list[dict[str, mx.array]],
                     weights: list[float] | None = None,
                     ) -> dict[str, mx.array]:
    """Compose multiple adapter B-matrices via norm-rescaled averaging.

    result[key] = weighted_mean(Bs) * mean_source_norm / ‖weighted_mean‖

    Equivalent to Fisher-Rao Karcher mean (Finding #275) at zero iteration cost.
    Preserves adapter norm to prevent 1/√N shrinkage from naive averaging.
    """
    if len(adapter_Bs) == 1:
        return adapter_Bs[0]
    if weights is None:
        weights = [1.0 / len(adapter_Bs)] * len(adapter_Bs)

    all_keys = set()
    for ab in adapter_Bs:
        all_keys.update(ab.keys())

    composed = {}
    for key in all_keys:
        tensors = [ab[key] for ab in adapter_Bs if key in ab]
        w = weights[:len(tensors)]
        composed[key] = _norm_rescaled_average(tensors, w)

    return composed


def _norm_rescaled_average(tensors: list[mx.array],
                           weights: list[float]) -> mx.array:
    """Weighted average with norm preservation."""
    if len(tensors) == 1:
        return tensors[0]
    w_sum = sum(weights)
    mean = sum(t.astype(mx.float32) * (w / w_sum) for t, w in zip(tensors, weights))
    source_norm = mx.mean(mx.stack([
        mx.linalg.norm(t.reshape(-1).astype(mx.float32)) for t in tensors
    ]))
    mean_norm = mx.linalg.norm(mean.reshape(-1))
    mx.eval(source_norm, mean_norm)
    if mean_norm.item() > 1e-8:
        return (mean * (source_norm / mean_norm)).astype(mx.bfloat16)
    return mean.astype(mx.bfloat16)


# ---------------------------------------------------------------------------
# Router: closed-form ridge regression (DUME)
# ---------------------------------------------------------------------------

def encode(model, input_ids: mx.array) -> mx.array:
    """Mean-pooled, post-norm hidden state. (B, T) → (B, H).

    Causal mask + final norm required for domain separation.
    Without them, routing accuracy drops from 99.6% to 16.8%.
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


def fit_router(model, tokenizer, domain_texts: dict[str, list[str]],
               lam: float = 1.0, max_seq: int = 256) -> mx.array:
    """Fit closed-form ridge regression router on calibration data.

    W* = (X^TX + λI)^{-1} X^TY, column-normalized.
    Returns W: (H, D) router weight matrix.
    """
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
            h = encode(model, mx.array(toks)[None, :])
            XtX = XtX + h.T @ h
            XtY = XtY.at[:, di].add(h.squeeze(0))

    W = mx.linalg.solve(XtX + lam * mx.eye(H), XtY, stream=mx.cpu)
    W = W / mx.maximum(mx.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    mx.eval(W)
    return W


def route(model, tokenizer, text: str, W: mx.array,
          max_seq: int = 256) -> int:
    """Route a query to the best domain. Returns domain index."""
    h = encode(model, mx.array(tokenizer.encode(text)[:max_seq])[None, :])
    return mx.argmax(h @ W, axis=-1).item()


# ---------------------------------------------------------------------------
# Null-space projection (Brainstacks §3.5)
# ---------------------------------------------------------------------------

def null_space_projector(deltas: mx.array, top_k: int = 64) -> mx.array:
    """SVD-based null-space projection: P = VV^T.

    To isolate a new adapter from existing ones: Δ' = Δ - Δ @ P
    """
    _, _, Vt = mx.linalg.svd(deltas, stream=mx.cpu)
    V = Vt[:top_k].T
    P = V @ V.T
    mx.eval(P)
    return P


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_adapter(path: str) -> dict[str, mx.array]:
    """Load adapter B-matrices from .npz file."""
    return dict(mx.load(path))


def load_frozen_A(path: str) -> dict[str, np.ndarray]:
    """Load Grassmannian-orthogonal frozen A-matrices from .npz file."""
    return dict(np.load(path))
