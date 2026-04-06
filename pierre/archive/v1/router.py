"""Ridge regression router — DUME (arXiv:2603.29765, Finding #276).

Closed-form router: W* = (X^TX + λI)^{-1} X^TY
96% accuracy, zero training, 23s init, 12ms incremental add.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class RidgeRouter(nn.Module):
    """Top-k router with weights from ridge regression.

    Args:
        hidden_dim: H — transformer hidden size.
        num_experts: D — number of domain adapters.
        top_k: experts selected per query (default 1).
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = mx.zeros((hidden_dim, num_experts))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route tokens to experts.

        Args:
            x: (..., H) hidden states.

        Returns:
            expert_indices: (..., top_k) selected expert ids.
            expert_weights: (..., top_k) softmax gating weights.
        """
        logits = x @ self.weight
        probs = mx.softmax(logits, axis=-1)

        if self.top_k >= self.num_experts:
            indices = mx.broadcast_to(
                mx.arange(self.num_experts), (*probs.shape[:-1], self.num_experts)
            )
            weights = probs
        else:
            indices = mx.argpartition(-probs, kth=self.top_k, axis=-1)[
                ..., : self.top_k
            ]
            weights = mx.take_along_axis(probs, indices, axis=-1)
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)

        return indices, weights


class RouterStatistics:
    """Accumulates X^TX and X^TY incrementally for ridge regression (Eq. 7)."""

    def __init__(self, hidden_dim: int, num_experts: int, dtype=mx.float32):
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.A = mx.zeros((hidden_dim, hidden_dim), dtype=dtype)
        self.B = mx.zeros((hidden_dim, num_experts), dtype=dtype)
        self.n_tokens = 0

    def update(self, features: mx.array, domain_id: int) -> None:
        """Accumulate one batch of features for a domain.

        Args:
            features: (N, H) hidden states (flattened batch*seq).
            domain_id: integer domain label in [0, D).
        """
        features = features.astype(self.A.dtype)
        n = features.shape[0]
        self.A = self.A + features.T @ features
        col_update = mx.sum(features, axis=0)
        self.B = self.B.at[:, domain_id].add(col_update)
        self.n_tokens += n


def solve_ridge(
    stats: RouterStatistics,
    lam: float = 0.1,
    column_normalize: bool = True,
) -> mx.array:
    """Compute optimal router weights via ridge regression.

    W* = (A + λI)^{-1} B, then column-normalize.

    Returns:
        W*: (H, D) optimal router weight matrix.
    """
    H = stats.hidden_dim
    regularized = stats.A + lam * mx.eye(H, dtype=stats.A.dtype)
    W_star = mx.linalg.solve(regularized, stats.B, stream=mx.cpu)

    if column_normalize:
        col_norms = mx.linalg.norm(W_star, axis=0, keepdims=True)
        col_norms = mx.maximum(col_norms, 1e-8)
        W_star = W_star / col_norms

    mx.eval(W_star)
    return W_star
