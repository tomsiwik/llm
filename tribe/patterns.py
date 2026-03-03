"""Pattern generation and identity helpers.

Patterns are (input, target) pairs that represent knowledge.
Clustered patterns simulate semantic domains.
"""

import mlx.core as mx
import numpy as np
from tribe.expert import DIM


def pattern_id(x, t):
    """Hashable identity for a pattern pair (for set membership tests).

    MLX arrays don't support Python's `in` operator, so we convert to tuples.
    """
    return (tuple(np.array(x).flat), tuple(np.array(t).flat))


def patterns_match(domain, pattern_ids_to_exclude):
    """Filter domain to patterns NOT in the given id set."""
    return [(x, t) for x, t in domain
            if pattern_id(x, t) not in pattern_ids_to_exclude]


def make_clustered_patterns(n_clusters, patterns_per_cluster, dim=DIM, seed=0):
    """Generate patterns organized in semantic clusters.

    Each cluster has a center and a fixed transform. Patterns are
    variations around the center with targets from the transform.

    Returns:
        (clusters, transforms) where:
        - clusters[i] = list of (mx.array, mx.array) pattern pairs
        - transforms[i] = (center_np, transform_matrix_np)
    """
    rng = np.random.RandomState(seed)
    clusters = []
    transforms = []
    for c in range(n_clusters):
        center = rng.randn(dim).astype(np.float32)
        target_transform = rng.randn(dim, dim).astype(np.float32) * 0.5
        transforms.append((center, target_transform))
        cluster = []
        for p in range(patterns_per_cluster):
            x = center + rng.randn(dim).astype(np.float32) * 0.3
            t = np.tanh(target_transform @ x)
            cluster.append((mx.array(x), mx.array(t.astype(np.float32))))
        clusters.append(cluster)
    return clusters, transforms
