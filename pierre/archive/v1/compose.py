"""Composition and pre-merge — proven components from Finding #275 + E2E demo.

Two modes:
1. Single adapter: compute delta = scale * B^T @ A^T, merge into base weights.
2. Multi-adapter NRE: compute per-domain deltas, NRE-merge them, merge into base.

NRE (Norm-Rescaled Euclidean): average deltas, then rescale to preserve mean source
norm. Proven equivalent to Fisher-Rao Karcher mean (Finding #275).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def compute_delta(
    b_matrix: dict[str, mx.array],
    skeleton: dict[str, np.ndarray],
    domain_idx: int,
    lora_scale: float,
    n_layers: int,
) -> dict[tuple[int, str], mx.array]:
    """Compute LoRA delta W for a single adapter: scale * B^T @ A^T.

    Args:
        b_matrix: adapter state dict (key -> B matrix).
        skeleton: Grassmannian A matrices (numpy arrays).
        domain_idx: domain index for skeleton key lookup.
        lora_scale: LoRA scaling factor.
        n_layers: number of transformer layers.

    Returns:
        Dict of (layer_idx, key) -> delta weight matrix.
    """
    deltas = {}
    for li in range(n_layers):
        for key in TARGET_KEYS:
            b_key = f"model.layers.{li}.{key}.lora_b"
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if b_key not in b_matrix or skey not in skeleton:
                continue
            a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
            b_mx = b_matrix[b_key].astype(mx.bfloat16)
            deltas[(li, key)] = lora_scale * (b_mx.T @ a_mx.T)
    return deltas


def nre_merge_deltas(
    delta_list: list[dict[tuple[int, str], mx.array]],
    weights: list[float] | None = None,
) -> dict[tuple[int, str], mx.array]:
    """NRE-merge multiple per-domain delta dicts.

    Computes weighted average of deltas, then rescales to preserve mean source norm.

    Args:
        delta_list: list of N delta dicts from compute_delta().
        weights: optional per-adapter weights (default: uniform 1/N).

    Returns:
        Merged delta dict with norm preservation.
    """
    N = len(delta_list)
    if N == 1:
        return delta_list[0]

    if weights is None:
        weights = [1.0 / N] * N

    all_keys = set()
    for d in delta_list:
        all_keys.update(d.keys())

    merged = {}
    for key in all_keys:
        available = []
        w_available = []
        for i, d in enumerate(delta_list):
            if key in d:
                available.append(d[key].astype(mx.float32))
                w_available.append(weights[i])

        if not available:
            continue

        # Weighted mean
        w_sum = sum(w_available)
        mean = available[0] * (w_available[0] / w_sum)
        for v, w in zip(available[1:], w_available[1:]):
            mean = mean + v * (w / w_sum)

        # Mean source norm
        source_norms = []
        for v in available:
            flat = v.reshape(-1)
            source_norms.append(mx.sqrt(mx.sum(flat * flat)))
        mean_source_norm = mx.mean(mx.stack(source_norms))

        # Rescale to preserve norm
        mean_flat = mean.reshape(-1)
        mean_norm = mx.sqrt(mx.sum(mean_flat * mean_flat))

        mx.eval(mean_source_norm, mean_norm)
        if mean_norm.item() > 1e-8:
            merged[key] = (mean * (mean_source_norm / mean_norm)).astype(mx.bfloat16)
        else:
            merged[key] = mean.astype(mx.bfloat16)

    mx.eval(*merged.values())
    return merged


def premerge_deltas_into_model(
    model: nn.Module,
    deltas: dict[tuple[int, str], mx.array],
) -> int:
    """Apply pre-computed deltas to model base weights (0% per-token overhead).

    W_new = W_base + delta

    Args:
        model: model with nn.Linear layers (post BitLinear replacement).
        deltas: dict of (layer_idx, key) -> delta weight matrix.

    Returns:
        Number of layers modified.
    """
    merge_count = 0
    for (li, key), delta in deltas.items():
        parts = key.split(".")
        module = model.model.layers[li]
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and isinstance(module, nn.Linear):
            module.weight = module.weight + delta
            merge_count += 1

    mx.eval(model.parameters())
    return merge_count


# Convenience aliases for backward compat
nre_compose = nre_merge_deltas


def premerge_into_model(
    model: nn.Module,
    skeleton: dict[str, np.ndarray],
    adapter_b: dict[str, mx.array],
    domain_idx: int,
    lora_scale: float,
    n_layers: int,
) -> int:
    """Single-adapter convenience: compute delta and merge in one step.

    Args:
        model: model with nn.Linear layers.
        skeleton: Grassmannian A matrices.
        adapter_b: single adapter B matrices.
        domain_idx: domain index for A-matrix lookup.
        lora_scale: LoRA scaling factor.
        n_layers: number of transformer layers.

    Returns:
        Number of layers modified.
    """
    deltas = compute_delta(adapter_b, skeleton, domain_idx, lora_scale, n_layers)
    return premerge_deltas_into_model(model, deltas)
