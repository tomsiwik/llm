"""Self-Routing LoRA Expert Library.

Expert library with per-token routing to LoRA experts.
Each expert is a (A, B) pair with optional routing key K.

Without routing keys: score = ||x @ A||^2 (self-routing via A)
With routing keys:    score = ||x @ K||^2 (decoupled routing via K)

Routing keys are thin (d_in, d_key) matrices trained contrastively
at calibration time. They decouple routing from computation:
- K decides WHICH expert fires (trained for discrimination)
- A@B computes the expert delta (trained for reconstruction)

Usage:
    from tribe.lora_library import SelfRoutingLoRALibrary

    lib = SelfRoutingLoRALibrary(base_weight=W, base_bias=b, scale=16/16)
    lib.register_expert(A_py, B_py, label="python")
    lib.register_expert(A_js, B_js, label="javascript")
    lib.initialize_routing_keys(d_key=8)  # optional: add contrastive keys
    y = lib(x)  # base(x) + scale * top-k routed expert delta
"""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SelfRoutingLoRALibrary(nn.Module):
    """Expert library with per-token routing to LoRA experts.

    Each expert is a (A, B) pair with optional routing key K.
    Without K: score = ||x @ A||^2. With K: score = ||x @ K||^2.
    Top-k experts activated per token at full strength.
    """

    def __init__(self, base_weight, base_bias=None, top_k=1, scale=1.0):
        super().__init__()

        self.d_out, self.d_in = base_weight.shape
        self.weight = base_weight
        if base_bias is not None:
            self.bias = base_bias
        self._has_bias = base_bias is not None
        self._top_k = top_k
        self._scale = scale

        # Expert storage — lists grown by register_expert()
        self._expert_As = []  # list of mx.array (d_in, r_i)
        self._expert_Bs = []  # list of mx.array (r_i, d_out)
        self._labels = []
        self._n_experts = 0
        self._has_routing_keys = False
        self._d_key = 0

    @property
    def n_experts(self):
        return self._n_experts

    def register_expert(self, A, B, label=""):
        """Register a new expert (A, B) pair. No retraining needed.

        Args:
            A: (d_in, rank) projection/routing key matrix.
            B: (rank, d_out) knowledge matrix.
            label: human-readable label for diagnostics.
        """
        assert A.shape[0] == self.d_in, f"A.shape[0]={A.shape[0]} != d_in={self.d_in}"
        assert B.shape[1] == self.d_out, f"B.shape[1]={B.shape[1]} != d_out={self.d_out}"
        assert A.shape[1] == B.shape[0], "A rank != B rank"

        idx = self._n_experts
        # Store as named attributes so MLX sees them in parameter tree
        setattr(self, f"expert_A_{idx}", A)
        setattr(self, f"expert_B_{idx}", B)
        self._expert_As.append(A)
        self._expert_Bs.append(B)
        self._labels.append(label or f"expert_{idx}")
        self._n_experts += 1

    def initialize_routing_keys(self, d_key=8, init_from_A=True):
        """Add contrastive routing keys K_i for each expert.

        Args:
            d_key: routing key dimension (default 8, half of typical rank).
            init_from_A: warm-start from top SVD directions of A_i.
        """
        self._d_key = d_key
        for i in range(self._n_experts):
            A = getattr(self, f"expert_A_{i}")
            if init_from_A:
                # SVD warm-start: use top-d_key left singular vectors of A
                U, S, Vt = mx.linalg.svd(A, stream=mx.cpu)
                mx.eval(U, S)
                # U: (d_in, min(d_in, rank)), take first d_key columns
                k = min(d_key, U.shape[1])
                K = U[:, :k] * mx.sqrt(S[:k])[None, :]  # scale by singular values
                if k < d_key:
                    # Pad with small random if rank < d_key
                    pad = mx.random.normal((self.d_in, d_key - k)) * 0.01
                    K = mx.concatenate([K, pad], axis=1)
            else:
                K = mx.random.normal((self.d_in, d_key)) * (1.0 / math.sqrt(d_key))
            setattr(self, f"routing_key_{i}", K)
        self._has_routing_keys = True
        mx.eval(self.parameters())

    def _score_experts(self, x):
        """Compute routing scores for all experts.

        With routing keys: score_i = ||x @ K_i||^2 (decoupled routing).
        Without:           score_i = ||x @ A_i||^2 (self-routing fallback).

        Projections always use A (computation path unchanged).

        Args:
            x: (..., d_in) input tensor.

        Returns:
            scores: (..., n_experts) routing scores.
            projections: list of (..., r_i) per-expert projections.
        """
        scores = []
        projections = []
        for i in range(self._n_experts):
            A = getattr(self, f"expert_A_{i}")
            proj = x @ A  # (..., r_i) — always via A for computation
            projections.append(proj)

            if self._has_routing_keys:
                K = getattr(self, f"routing_key_{i}")
                route = x @ K  # (..., d_key)
                score = mx.sum(route * route, axis=-1)  # (...,)
            else:
                score = mx.sum(proj * proj, axis=-1)  # (...,) fallback
            scores.append(score)
        # Stack scores: (..., n_experts)
        scores = mx.stack(scores, axis=-1)
        return scores, projections

    def __call__(self, x):
        """Forward: base(x) + scale * top-k routed expert delta.

        Args:
            x: (..., d_in) input tensor.

        Returns:
            (..., d_out) output tensor.
        """
        # Base linear
        if self._has_bias:
            out = x @ self.weight.T + self.bias
        else:
            out = x @ self.weight.T

        if self._n_experts == 0:
            return out

        scores, projections = self._score_experts(x)

        k = min(self._top_k, self._n_experts)

        if k >= self._n_experts:
            # All experts active — soft weighting
            weights = mx.softmax(scores, axis=-1)  # (..., n_experts)
            delta = mx.zeros_like(out)
            for i in range(self._n_experts):
                B = getattr(self, f"expert_B_{i}")
                w_i = weights[..., i : i + 1]  # (..., 1)
                expert_out = projections[i] @ B  # (..., d_out)
                delta = delta + w_i * expert_out
        else:
            # Top-k selection
            # Get top-k indices
            if k == 1:
                # Fast path: argmax
                top_idx = mx.argmax(scores, axis=-1)  # (...,)
                top_scores = mx.max(scores, axis=-1, keepdims=True)
                # Gather the selected expert output
                delta = mx.zeros_like(out)
                for i in range(self._n_experts):
                    B = getattr(self, f"expert_B_{i}")
                    mask = (top_idx == i).astype(mx.float32)  # (...,)
                    mask = mx.expand_dims(mask, axis=-1)  # (..., 1)
                    expert_out = projections[i] @ B  # (..., d_out)
                    delta = delta + mask * expert_out
            else:
                # General top-k
                sorted_scores = mx.sort(scores, axis=-1)
                threshold = sorted_scores[..., -k : -k + 1]  # (..., 1)
                mask = (scores >= threshold).astype(mx.float32)  # (..., n_experts)
                # Softmax over selected
                masked_scores = scores * mask + (1 - mask) * (-1e9)
                weights = mx.softmax(masked_scores, axis=-1) * mask
                weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-8)

                delta = mx.zeros_like(out)
                for i in range(self._n_experts):
                    B = getattr(self, f"expert_B_{i}")
                    w_i = weights[..., i : i + 1]
                    expert_out = projections[i] @ B
                    delta = delta + w_i * expert_out

        return out + self._scale * delta

    def routing_stats(self, x):
        """Diagnostic: which experts activate for which tokens.

        Args:
            x: (..., d_in) input tensor.

        Returns:
            dict with:
                scores: (..., n_experts) raw routing scores
                selected: (...,) index of top-1 expert per token
                expert_load: (n_experts,) fraction of tokens routed to each
                labels: list of expert labels
        """
        if self._n_experts == 0:
            return {"scores": None, "selected": None, "expert_load": None,
                    "labels": []}

        scores, _ = self._score_experts(x)
        selected = mx.argmax(scores, axis=-1)  # (...,)
        mx.eval(scores, selected)

        # Expert load: fraction of tokens routed to each expert
        selected_np = np.array(selected.reshape(-1))
        n_tokens = len(selected_np)
        load = np.zeros(self._n_experts)
        for i in range(self._n_experts):
            load[i] = np.sum(selected_np == i) / max(n_tokens, 1)

        return {
            "scores": scores,
            "selected": selected,
            "expert_load": load,
            "labels": self._labels[:],
        }

    def per_expert_scores(self, x):
        """Per-expert mean activation score across all tokens.

        Args:
            x: (..., d_in) input tensor.

        Returns:
            (n_experts,) numpy array of mean scores.
        """
        scores, _ = self._score_experts(x)
        mx.eval(scores)
        # Mean over all dims except last
        scores_flat = scores.reshape(-1, self._n_experts)
        mean_scores = mx.mean(scores_flat, axis=0)
        mx.eval(mean_scores)
        return np.array(mean_scores)

    def __repr__(self):
        labels = ", ".join(self._labels) if self._labels else "empty"
        return (f"SelfRoutingLoRALibrary(d_in={self.d_in}, d_out={self.d_out}, "
                f"n_experts={self._n_experts}, top_k={self._top_k}, "
                f"experts=[{labels}])")


def collect_library_layers(model):
    """Find all SelfRoutingLoRALibrary layers in a model."""
    results = []

    def _search(module, prefix=""):
        if isinstance(module, SelfRoutingLoRALibrary):
            results.append((prefix, module))
            return
        if isinstance(module, nn.Module):
            children = module.children()
            for name, child in children.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Module):
                    _search(child, full_name)
                elif isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        _search(item, f"{full_name}.{i}")
                elif isinstance(child, dict):
                    for k, v in child.items():
                        _search(v, f"{full_name}.{k}")

    _search(model)
    return results
