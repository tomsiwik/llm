"""05 -- HC-SMoE: Hierarchical Clustering for Expert Merging

Paper: Retraining-Free Merging of Sparse MoE via Hierarchical Clustering
       (I Chun Chen et al., 2024)
URL:   https://arxiv.org/abs/2410.08589
Repo:  wazenmai/HC-SMoE

Extracted from: hcsmoe/merging/clustering.py, hcsmoe/merging/overlap.py

The core idea: flatten each expert's weight matrices into a single vector,
compute pairwise Euclidean distances, then run agglomerative (hierarchical)
clustering to group similar experts.  After grouping, experts in each cluster
are merged into a single expert (the one closest to the cluster centroid is
kept as the "dominant" expert, and non-dominant weights are averaged in).

This is a REFERENCE EXTRACT for study -- not runnable as-is.
Original PyTorch preserved; imports/model-specific wiring removed.
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# 1. PAIRWISE DISTANCE MATRIX
#    Flatten expert weight matrices -> one vector per expert, compute L2.
# ---------------------------------------------------------------------------

@torch.no_grad()
def pairwise_distances(X: torch.Tensor, method: str = "single") -> torch.Tensor:
    """Compute pairwise Euclidean distances between expert weight vectors.

    Args:
        X: (num_experts, flattened_param_dim) -- each row is one expert.
        method: linkage type; controls diagonal fill convention.

    Returns:
        (num_experts, num_experts) symmetric distance matrix.

    # TRIBE NOTE: We use a similar structure in measure_overlap(), but our
    # "distance" is loss-based (how differently two experts perform on each
    # pattern) rather than weight-space Euclidean distance.  Weight distance
    # is cheap but ignores functional equivalence -- two experts with
    # permuted neurons look distant in weight space but identical in function.
    """
    dot_product = torch.mm(X, X.t())
    square_norm = dot_product.diag()
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0).sqrt()
    if method == "single" or method == "average":
        distances.fill_diagonal_(float("inf"))  # prevent self-merge
    elif method == "complete":
        distances.fill_diagonal_(0.0)
    return distances


# ---------------------------------------------------------------------------
# 2. SINGLE LINKAGE STEP
#    Each step finds the closest pair of clusters and merges them.
# ---------------------------------------------------------------------------

@torch.no_grad()
def linkage_step(
    distances: torch.Tensor,
    pair_distances: torch.Tensor,
    clusters: torch.Tensor,
    method: str = "single",
    X: Optional[torch.Tensor] = None,
):
    """Perform one step of agglomerative clustering.

    Linkage variants:
      - single:   d(Ci, Cj) = min distance between any pair across clusters
      - complete: d(Ci, Cj) = max distance
      - average:  d(Ci, Cj) = mean distance across all cross-cluster pairs
      - ward:     d(Ci, Cj) = increase in total variance when merging

    # TRIBE NOTE: This is the merge-decision logic.  In our lifecycle,
    # bond() merges experts with high domain overlap (measured by loss
    # correlation on shared patterns).  HC-SMoE's linkage is purely
    # weight-geometric.  A hybrid could use functional overlap as the
    # distance metric inside hierarchical clustering -- getting HC-SMoE's
    # multi-level structure with our domain-aware similarity.
    """
    if method == "single":
        min_idx = torch.argmin(distances).item()
        i = min_idx // distances.shape[0]
        j = min_idx % distances.shape[0]
    elif method == "complete":
        max_idx = torch.argmax(distances).item()
        i = max_idx // distances.shape[0]
        j = max_idx % distances.shape[0]
    else:
        # average / ward: compute cluster-level distances
        i, j = _compute_cluster_distance(pair_distances, clusters, method, X)

    if i > j:
        i, j = j, i

    if method in ("average", "ward"):
        return i, j, distances

    # Update distance matrix: merge cluster j into cluster i
    for k in range(distances.shape[0]):
        if k != i and k != j:
            if method == "single":
                new_dist = torch.min(distances[i, k], distances[j, k])
            elif method == "complete":
                new_dist = torch.max(distances[i, k], distances[j, k])
            else:
                new_dist = distances[i, k]  # fallback
            distances[i, k] = new_dist
            distances[k, i] = new_dist

    # Invalidate cluster j
    if method == "single":
        distances[i, i] = float("inf")
        distances[j, :] = float("inf")
        distances[:, j] = float("inf")
    elif method == "complete":
        distances[i, i] = 0.0
        distances[j, :] = 0.0
        distances[:, j] = 0.0

    return i, j, distances


# ---------------------------------------------------------------------------
# 3. CLUSTER-LEVEL DISTANCE (for average / ward linkage)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_cluster_distance(
    pair_distances: torch.Tensor,
    clusters: torch.Tensor,
    method: str = "average",
    X: Optional[torch.Tensor] = None,
):
    """Compute inter-cluster distance for average or Ward linkage."""
    cluster_labels = torch.unique(clusters)

    if method == "average":
        distances = torch.zeros((len(cluster_labels), len(cluster_labels)))
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                members_i = torch.where(clusters == ci)[0]
                members_j = torch.where(clusters == cj)[0]
                # Average all pairwise distances between the two clusters
                total = sum(
                    pair_distances[vi, vj].item()
                    for vi in members_i
                    for vj in members_j
                )
                count = len(members_i) * len(members_j)
                distances[i, j] = total / count
                distances[j, i] = distances[i, j]
    elif method == "ward":
        # Ward: merge penalty = (ni * nj)/(ni+nj) * ||mu_i - mu_j||^2
        centers = torch.zeros((len(cluster_labels), X.shape[1]))
        for i, c in enumerate(cluster_labels):
            centers[i] = X[clusters == c].mean(dim=0)
        distances = torch.zeros((len(cluster_labels), len(cluster_labels)))
        for i, ci in enumerate(cluster_labels):
            for j, cj in enumerate(cluster_labels):
                if i >= j:
                    continue
                ni = (clusters == ci).sum()
                nj = (clusters == cj).sum()
                d = (ni * nj) / (ni + nj) * torch.cdist(
                    centers[i].unsqueeze(0), centers[j].unsqueeze(0), p=2
                )
                distances[i, j] = d
                distances[j, i] = d
    else:
        raise NotImplementedError(f"Unsupported linkage: {method}")

    distances.fill_diagonal_(float("inf"))
    idx = torch.argmin(distances)
    final_i = cluster_labels[idx // distances.shape[0]]
    final_j = cluster_labels[idx % distances.shape[0]]
    return final_i, final_j


# ---------------------------------------------------------------------------
# 4. FULL HIERARCHICAL CLUSTERING
#    Repeatedly merge until we reach the target number of clusters.
# ---------------------------------------------------------------------------

@torch.no_grad()
def hierarchical_clustering(
    X: torch.Tensor,
    n_clusters: int,
    method: str = "single",
):
    """Cluster expert weight vectors into n_clusters groups.

    Args:
        X: (num_experts, param_dim) flattened weight vectors.
        n_clusters: target number of expert groups after merging.
        method: linkage type ('single', 'complete', 'average', 'ward').

    Returns:
        clusters: (num_experts,) integer cluster labels.
        center_indices: list of expert indices closest to each cluster center
                        (these become the "dominant" experts that survive).

    # TRIBE NOTE: Their merge = our bond().  They decide HOW MANY to merge
    # (n_clusters) as a hyperparameter.  We decide WHEN to merge based on
    # overlap exceeding a threshold.  Their hierarchical approach could
    # inspire multi-level bonding: first merge highly-redundant pairs,
    # then optionally merge the merged experts in a second pass.
    """
    n_samples = X.shape[0]
    distances = pairwise_distances(X, method)
    pair_distances = distances.clone()
    clusters = torch.arange(n_samples)

    while len(torch.unique(clusters)) > n_clusters:
        i, j, distances = linkage_step(distances, pair_distances, clusters, method, X)
        cj = clusters[j]
        clusters[clusters == cj] = clusters[i]  # absorb cluster j into i

    # Reassign contiguous labels 0..n_clusters-1
    label_map = {}
    next_label = 0
    for idx_val in clusters.tolist():
        if idx_val not in label_map:
            label_map[idx_val] = next_label
            next_label += 1
    clusters = torch.tensor([label_map[v] for v in clusters.tolist()])

    # Find dominant expert: closest to cluster centroid
    # TRIBE NOTE: This is analogous to choosing which expert's weights
    # to keep as the "base" in our bond() operation.  We use the higher-
    # fitness expert; they use geometric centrality.
    center_indices = []
    for k in range(n_clusters):
        members = X[clusters == k]
        centroid = members.mean(dim=0)
        dists = torch.cdist(members, centroid.unsqueeze(0), p=2)
        closest = torch.argmin(dists, dim=0).item()
        center_indices.append(torch.where(clusters == k)[0][closest].item())

    return clusters, center_indices


# ---------------------------------------------------------------------------
# 5. OVERLAP / SIMILARITY METRICS
#    From hcsmoe/merging/overlap.py -- alternative ways to measure how
#    similar two experts are before deciding to merge.
# ---------------------------------------------------------------------------

def overlap_rate(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute overlap as intersection/union of value ranges per dimension.

    # TRIBE NOTE: This is a lightweight geometric proxy for functional
    # similarity.  Our measure_overlap() is more expensive (requires
    # evaluating both experts on shared data) but captures cases where
    # experts cover the same input domain with different weight geometry.
    """
    X_min, X_max = X.min(dim=0).values, X.max(dim=0).values
    Y_min, Y_max = Y.min(dim=0).values, Y.max(dim=0).values

    intersection_min = torch.max(X_min, Y_min)
    intersection_max = torch.min(X_max, Y_max)
    union_min = torch.min(X_min, Y_min)
    union_max = torch.max(X_max, Y_max)

    intersection_length = (intersection_max - intersection_min).clamp(min=0)
    union_length = union_max - union_min

    return (intersection_length.sum() / union_length.sum()).item()


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between expert output distributions.

    # TRIBE NOTE: Output distribution similarity could be a stronger
    # signal than weight-space distance for deciding bond().  If two
    # experts produce nearly identical output distributions on the
    # calibration set, they are functionally redundant regardless of
    # how different their weight matrices look.
    """
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return -(
        F.kl_div(p.log(), q, reduction="batchmean")
        + F.kl_div(q.log(), p, reduction="batchmean")
    ) / 2


# ---------------------------------------------------------------------------
# 6. SILHOUETTE SCORE
#    Used in the dynamic variant to automatically decide n_clusters.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_silhouette_score(
    tensor_list: torch.Tensor,
    cluster_labels: torch.Tensor,
) -> torch.Tensor:
    """Silhouette score: measures clustering quality.

    For each sample:
      a(i) = mean intra-cluster distance
      b(i) = mean nearest-cluster distance
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    # TRIBE NOTE: Silhouette score could help our lifecycle decide the
    # optimal number of experts.  If adding an expert doesn't improve
    # the silhouette of the competence clustering, the system is already
    # well-partitioned and the new expert would be redundant.
    """
    n = tensor_list.shape[0]
    # Pairwise distances
    pw_dist = torch.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            d = torch.norm(tensor_list[i] - tensor_list[j])
            pw_dist[i, j] = d
            pw_dist[j, i] = d

    unique_labels = torch.unique(cluster_labels)
    scores = torch.zeros(n)

    for i in range(n):
        same = [j for j in range(n) if cluster_labels[j] == cluster_labels[i] and j != i]
        a_i = torch.mean(pw_dist[i, same]) if same else 0

        b_i = float("inf")
        for label in unique_labels:
            if label == cluster_labels[i]:
                continue
            other = [j for j in range(n) if cluster_labels[j] == label]
            if other:
                b_i = min(b_i, torch.mean(pw_dist[i, other]).item())

        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    return torch.mean(scores)
