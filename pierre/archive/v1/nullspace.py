"""Null-space projection — Brainstacks (arXiv:2604.01152, §3.5, Finding #273).

SVD-based orthogonal subspace isolation for zero forgetting.
Collects frozen adapter output deltas, extracts top-K principal directions,
projects new adapter updates into the null space of existing adapters.
"""

import mlx.core as mx


class NullSpaceProjector:
    """Compute and apply null-space projection from adapter output deltas.

    Args:
        top_k: number of principal directions to retain (default 64, §3.5).
    """

    def __init__(self, top_k: int = 64):
        self.top_k = top_k

    def compute_projector(self, deltas: mx.array) -> mx.array:
        """Compute null-space projection matrix via SVD.

        Args:
            deltas: (n_samples, h_dim) — output deltas from frozen adapter stack.

        Returns:
            P: (h_dim, h_dim) — projection matrix. To project into null space:
               delta_projected = delta - (delta @ P)
        """
        # SVD on CPU for numerical stability with large matrices
        _, _, Vt = mx.linalg.svd(deltas, stream=mx.cpu)
        V = Vt[: self.top_k].T  # (h_dim, top_k)
        P = V @ V.T  # (h_dim, h_dim)
        mx.eval(P)
        return P

    @staticmethod
    def project(delta: mx.array, P: mx.array) -> mx.array:
        """Project adapter delta into null space of existing adapters.

        Args:
            delta: (..., h_dim) adapter output to project.
            P: (h_dim, h_dim) projection matrix from compute_projector.

        Returns:
            Projected delta with interference removed.
        """
        return delta - (delta @ P)
