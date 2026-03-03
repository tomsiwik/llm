"""System-level metrics for evaluating tribe health and knowledge organization."""

import mlx.core as mx
import numpy as np
from tribe.expert import forward_batch


def measure_knowledge_precision(tribe, clusters):
    """How precisely is knowledge located?

    For each cluster, measures how much better the best expert is
    compared to the second-best. Higher = more specialized.
    """
    fwd = getattr(tribe, 'fwd', None) or forward_batch
    precisions = []
    for cluster in clusters:
        X = mx.stack([x for x, _ in cluster])
        T = mx.stack([t for _, t in cluster])
        losses = []
        for m in tribe.routable_members():
            preds = fwd(m.weights, X)
            l = mx.mean((preds - T) ** 2).item()
            losses.append((m.id, l))
        if len(losses) < 2:
            continue
        losses.sort(key=lambda x: x[1])
        best_loss = losses[0][1]
        second_loss = losses[1][1]
        if best_loss > 0:
            precision = second_loss / (best_loss + 1e-8)
        else:
            precision = 1.0
        precisions.append(precision)
    return np.mean(precisions) if precisions else 1.0


def measure_system_loss(tribe, all_patterns):
    """System loss: for each pattern, use the best routable expert (oracle)."""
    fwd = getattr(tribe, 'fwd', None) or forward_batch
    routable = tribe.routable_members()
    if not routable:
        return float('inf')
    X = mx.stack([x for x, _ in all_patterns])
    T = mx.stack([t for _, t in all_patterns])
    expert_losses = []
    for m in routable:
        preds = fwd(m.weights, X)
        losses = mx.mean((preds - T) ** 2, axis=1)
        expert_losses.append(losses)
    all_losses = mx.stack(expert_losses)
    best_losses = mx.min(all_losses, axis=0)
    return mx.mean(best_losses).item()


def measure_representation_overlap(expert_weights_list, X, fwd_hidden):
    """Compute pairwise cosine similarity of expert hidden representations.

    Returns an E x E matrix where entry (i, j) is the mean absolute cosine
    similarity between expert i and expert j's hidden representations.
    Diagonal entries are 1.0. Off-diagonal values should decrease when
    orthogonality loss is applied.

    Args:
        expert_weights_list: list of expert weight dicts.
        X: (N, H, W, C) input batch (mx.array).
        fwd_hidden: function(weights, X) -> (N, hidden_dim) hidden representations.

    Returns:
        E x E numpy array of pairwise cosine similarity values.
    """
    hiddens = [fwd_hidden(ew, X) for ew in expert_weights_list]
    # Evaluate all hidden representations
    mx.eval(*hiddens)
    E = len(hiddens)
    overlap_matrix = np.zeros((E, E))
    for i in range(E):
        for j in range(E):
            hi = np.array(hiddens[i])
            hj = np.array(hiddens[j])
            hi_n = hi / (np.linalg.norm(hi, axis=-1, keepdims=True) + 1e-8)
            hj_n = hj / (np.linalg.norm(hj, axis=-1, keepdims=True) + 1e-8)
            cos = np.mean(np.abs(np.sum(hi_n * hj_n, axis=-1)))
            overlap_matrix[i, j] = cos
    return overlap_matrix


def measure_redundancy(tribe):
    """Average pairwise overlap across all routable member pairs."""
    routable = tribe.routable_members()
    if len(routable) < 2:
        return 0.0
    overlaps = []
    for i, m1 in enumerate(routable):
        for m2 in routable[i+1:]:
            if m1.domain and m2.domain:
                o = tribe.measure_overlap(m1, m2)
                overlaps.append(o)
    return np.mean(overlaps) if overlaps else 0.0
