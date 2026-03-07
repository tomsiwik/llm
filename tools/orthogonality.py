"""Orthogonality diagnostic for LoRA/capsule weight deltas.

Computes pairwise cosine similarity between weight deltas to predict
composition compatibility. Works in two modes:

- Micro: takes trained micro models + base model, computes per-layer cosine
- Macro: takes .npz capsule state files, computes per-layer cosine

Safety verdicts:
  cos < 0.1  → SAFE (orthogonal)
  cos 0.1-0.5 → CAUTION (calibration essential)
  cos > 0.5  → WARNING (test before deploying)

Usage:
  uv run python -m tools.orthogonality macro/capsule_states/python.npz macro/capsule_states/javascript.npz
"""

import argparse
import statistics

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened vectors."""
    dot = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))
    return float(dot / (norm_a * norm_b + 1e-12))


def verdict(cos: float) -> str:
    if cos < 0.1:
        return "SAFE"
    elif cos <= 0.5:
        return "CAUTION"
    else:
        return "WARNING"


def check_macro_compatibility(file_a: str, file_b: str) -> dict:
    """Compare two .npz capsule state files for orthogonality.

    Capsule states contain per-layer (A, B) weight matrices.
    Since surgery zero-initializes B, the trained weights ARE the deltas.
    """
    state_a = dict(np.load(file_a))
    state_b = dict(np.load(file_b))

    # Group keys by layer
    layers_a = {}
    layers_b = {}
    for key, val in state_a.items():
        # Keys like "layers.0.capsule_pool.groups.0.A.weight"
        parts = key.split(".")
        layer_idx = parts[1] if len(parts) > 1 else "0"
        layers_a.setdefault(layer_idx, []).append(val.flatten())
    for key, val in state_b.items():
        parts = key.split(".")
        layer_idx = parts[1] if len(parts) > 1 else "0"
        layers_b.setdefault(layer_idx, []).append(val.flatten())

    per_layer = {}
    all_sims = []
    for layer_key in sorted(layers_a.keys()):
        if layer_key not in layers_b:
            continue
        delta_a = np.concatenate(layers_a[layer_key])
        delta_b = np.concatenate(layers_b[layer_key])
        # Ensure same size (skip if mismatched architecture)
        min_len = min(len(delta_a), len(delta_b))
        cos = cosine_similarity(delta_a[:min_len], delta_b[:min_len])
        per_layer[layer_key] = cos
        all_sims.append(cos)

    aggregate = statistics.mean(all_sims) if all_sims else 0.0
    return {
        "per_layer": per_layer,
        "aggregate": aggregate,
        "verdict": verdict(aggregate),
        "all_sims": all_sims,
    }


def check_micro_compatibility(base_model, models: list) -> dict:
    """Compare micro models against a base for orthogonality.

    Extracts weight deltas (trained - base) and computes pairwise cosine.
    Works with any model that has .layers with nested weight parameters.
    """
    import mlx.core as mx
    import mlx.nn as nn

    def extract_delta(model, base) -> np.ndarray:
        """Flatten all weight deltas between model and base into one vector."""
        model_params = dict(nn.utils.tree_flatten(model.parameters()))
        base_params = dict(nn.utils.tree_flatten(base.parameters()))
        parts = []
        for key in sorted(model_params.keys()):
            if key in base_params:
                delta = model_params[key] - base_params[key]
                parts.append(np.array(delta.reshape(-1).tolist()))
        return np.concatenate(parts)

    n = len(models)
    deltas = [extract_delta(m, base_model) for m in models]

    all_sims = []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = cosine_similarity(deltas[i], deltas[j])
            all_sims.append(cos)
            pairs.append((i, j, cos))

    aggregate = statistics.mean(all_sims) if all_sims else 0.0
    return {
        "pairs": pairs,
        "aggregate": aggregate,
        "verdict": verdict(aggregate),
        "all_sims": all_sims,
    }


def print_report(result: dict, label_a: str = "A", label_b: str = "B"):
    """Print a human-readable orthogonality report."""
    print(f"\n{'=' * 50}")
    print("ORTHOGONALITY DIAGNOSTIC")
    print(f"{'=' * 50}")

    if "per_layer" in result:
        print(f"\n{'Layer':<10} {'Cosine':>10} {'Verdict':>10}")
        print("-" * 32)
        for layer, cos in sorted(result["per_layer"].items()):
            print(f"{layer:<10} {cos:>10.4f} {verdict(cos):>10}")

    if "pairs" in result:
        print(f"\n{'Pair':<10} {'Cosine':>10} {'Verdict':>10}")
        print("-" * 32)
        for i, j, cos in result["pairs"]:
            print(f"({i},{j}){'':<5} {cos:>10.4f} {verdict(cos):>10}")

    print(f"\n  Aggregate cosine: {result['aggregate']:.4f}")
    print(f"  Verdict: {result['verdict']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check orthogonality between capsule state files or models.",
    )
    parser.add_argument("files", nargs="+", help=".npz capsule state files to compare")
    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("Need at least 2 files to compare")

    # Pairwise comparison of all provided files
    all_sims = []
    for i in range(len(args.files)):
        for j in range(i + 1, len(args.files)):
            result = check_macro_compatibility(args.files[i], args.files[j])
            print(f"\n--- {args.files[i]} vs {args.files[j]} ---")
            print_report(result, args.files[i], args.files[j])
            all_sims.extend(result["all_sims"])

    if len(args.files) > 2 and all_sims:
        overall = statistics.mean(all_sims)
        print(f"\nOverall mean cosine: {overall:.4f} → {verdict(overall)}")


if __name__ == "__main__":
    main()
