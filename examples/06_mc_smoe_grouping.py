"""06 -- MC-SMoE: Router-Logit Grouping and Frequency-Weighted Merging

Paper: Merge, Then Compress: Demystify Efficient SMoE with Hints
       from Its Routing Policy (Li et al., 2024)
URL:   https://arxiv.org/abs/2310.01334
Repo:  UNITES-Lab/MC-SMoE

Extracted from: mcsmoe/merging/grouping.py

MC-SMoE's key insight: the router already knows which experts are
redundant.  If two experts receive similar router logits across a
calibration set, they are functionally interchangeable.  After grouping
by router-logit correlation, experts are merged using usage-frequency
weighting: more-frequently-routed experts contribute more to the
merged result.

This is a REFERENCE EXTRACT for study -- not runnable as-is.
Original PyTorch preserved; model-specific wiring simplified.
"""

import torch
from torch.nn import functional as F
from typing import Dict, List, Optional
from copy import deepcopy

FP32_EPS = 1e-8

# ---------------------------------------------------------------------------
# SIMILARITY FUNCTIONS
# Used to measure how "close" two experts are in various feature spaces.
# ---------------------------------------------------------------------------

SIMILARITY_MAPPING_FUNCTION = {
    # TRIBE NOTE: cosine similarity shifted to [0,1].  Our measure_overlap()
    # uses loss correlation instead -- cosine on loss vectors rather than
    # weight vectors or router logits.  Both capture "do these two experts
    # respond similarly?" but from different angles.
    "cosine": lambda x, y: (F.cosine_similarity(x, y, dim=-1, eps=FP32_EPS) + 1) / 2,
    "mse": lambda x, y: 1 / (1 + 0.1 * torch.log(F.mse_loss(x, y, reduction="sum"))),
}

LEGAL_SIMILARITY_BASES = [
    "weight",           # flatten expert weights, compare directly
    "feature",          # compare mean hidden states routed to each expert
    "router-logits",    # compare columns of the router logit matrix
    "router-weight",    # compare rows of the router weight matrix
    "gradient",         # compare accumulated gradient vectors
    "weight-gradient",  # element-wise product of weight and gradient
    "random",           # random similarity (ablation baseline)
]


# ---------------------------------------------------------------------------
# 1. EXPERT GROUPER: SIMILARITY COMPUTATION
#    The core data structure that holds per-layer similarity matrices
#    and usage frequency counts.
# ---------------------------------------------------------------------------

class ExpertsGrouper:
    """Groups experts by similarity for subsequent merging.

    # TRIBE NOTE: This class is analogous to our measure_overlap() +
    # the grouping logic that precedes bond().  Key difference: MC-SMoE
    # computes a full N x N similarity matrix upfront on a calibration
    # set, then groups once.  Our lifecycle measures overlap continuously
    # during training, triggering bond() when overlap crosses a threshold.
    # Their approach is cheaper (one pass) but requires a good calibration
    # set; ours adapts online but costs more compute.
    """

    def __init__(
        self,
        num_experts: int,
        similarity_fn: str = "cosine",
        similarity_base: str = "router-logits",
    ):
        self.num_experts = num_experts
        self.similarity_fn = SIMILARITY_MAPPING_FUNCTION[similarity_fn]
        self.similarity_base = similarity_base
        # Per-layer state
        self._group_labels = {}       # layer_name -> (num_experts,) int tensor
        self._similarity = {}         # layer_name -> (num_experts, num_experts) float
        self._usage_frequency = {}    # layer_name -> (num_experts,) float

    def init_layer(self, layer_name: str):
        """Initialize tracking for one MoE layer."""
        self._group_labels[layer_name] = torch.arange(self.num_experts)
        self._similarity[layer_name] = torch.eye(self.num_experts)
        self._usage_frequency[layer_name] = torch.ones(self.num_experts) / self.num_experts


    # -------------------------------------------------------------------
    # 1a. ROUTER-LOGIT SIMILARITY
    #     The signature method of MC-SMoE.
    # -------------------------------------------------------------------

    def compute_similarity_by_router_logits(
        self,
        layer_name: str,
        router_logits: torch.Tensor,
    ):
        """Compute pairwise expert similarity from router logit columns.

        Args:
            router_logits: (total_tokens, num_experts) -- raw router outputs
                           across all tokens in the calibration set.

        The router logit column for expert i is a vector of length
        total_tokens.  If two experts have high cosine similarity between
        their logit columns, the router "thinks" they are interchangeable.

        # TRIBE NOTE: This is fundamentally different from our approach.
        # Router-logit correlation measures "which inputs WOULD be sent
        # to both experts" (routing intent).  Our loss-based overlap
        # measures "which inputs both experts ACTUALLY handle well"
        # (demonstrated competence).  Router correlation can detect
        # redundancy even before training converges; loss overlap
        # requires the experts to have learned something.
        """
        with torch.no_grad():
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    i_logits = router_logits[:, i].flatten()
                    j_logits = router_logits[:, j].flatten()
                    sim = self.similarity_fn(i_logits, j_logits)
                    self._similarity[layer_name][i, j] = sim
                    self._similarity[layer_name][j, i] = sim


    # -------------------------------------------------------------------
    # 1b. WEIGHT-BASED SIMILARITY
    # -------------------------------------------------------------------

    def compute_similarity_by_weight(
        self,
        layer_name: str,
        expert_weights: Dict[int, torch.Tensor],
    ):
        """Compare experts by flattened weight vectors.

        # TRIBE NOTE: Weight similarity is the simplest approach but
        # misses functional equivalence under neuron permutation.
        # HC-SMoE (file 05) uses this same metric inside hierarchical
        # clustering.  Our lifecycle avoids weight-space comparisons
        # entirely, relying on behavioral (loss-based) signals.
        """
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                sim = self.similarity_fn(
                    expert_weights[i].flatten(),
                    expert_weights[j].flatten(),
                )
                self._similarity[layer_name][i, j] = sim
                self._similarity[layer_name][j, i] = sim


    # -------------------------------------------------------------------
    # 2. USAGE FREQUENCY TRACKING
    #    Count how often each expert is selected by the router.
    # -------------------------------------------------------------------

    def compute_usage_frequencies(
        self,
        layer_name: str,
        router_indices: torch.Tensor,
    ):
        """Accumulate expert selection counts from router decisions.

        Args:
            router_indices: (total_tokens,) -- which expert was chosen
                            for each token.

        # TRIBE NOTE: Usage frequency is analogous to our expert.fitness.
        # An expert that is never routed to has zero fitness in our system
        # and becomes a candidate for recycle().  MC-SMoE uses frequency
        # to weight the merge: high-frequency experts dominate the merged
        # result.  Same intuition, different mechanism.
        """
        freq = torch.zeros(self.num_experts)
        for idx in router_indices:
            freq[idx] += 1
        self._usage_frequency[layer_name] = freq / freq.sum()


    # -------------------------------------------------------------------
    # 3. ROUTING-GUIDED GROUPING
    #    Use similarity + usage frequency to form expert groups.
    # -------------------------------------------------------------------

    def group_experts_by_routing(
        self,
        layer_name: str,
        num_groups: int,
    ) -> List[int]:
        """Group experts: top-K most-used become cluster cores, rest join
        the most similar core.

        Algorithm:
          1. Sort experts by usage frequency (descending).
          2. The top num_groups experts become "core" experts (cluster seeds).
          3. Each remaining expert joins the cluster whose core has the
             highest similarity to it.

        Returns:
            core_expert_indices: which experts are the cluster cores.

        # TRIBE NOTE: This is the grouping step before merge.  In our
        # lifecycle, we skip explicit grouping -- bond() acts on pairs
        # detected by measure_overlap().  MC-SMoE's approach could handle
        # many-to-one merges more gracefully: group 5 experts into 2
        # clusters, then merge each cluster.  Our pairwise bond() would
        # need 3 sequential operations to achieve the same.
        """
        sim_matrix = deepcopy(self._similarity[layer_name])
        usage = self._usage_frequency[layer_name]

        indices_by_usage = torch.argsort(usage, descending=True)

        # Step 1: Top-K become cores
        core_indices = indices_by_usage[:num_groups]
        for i in range(num_groups):
            self._group_labels[layer_name][indices_by_usage[i]] = i

        # Step 2: Assign remaining experts to most-similar core
        # TRIBE NOTE: This greedy assignment is simple but can lead to
        # imbalanced groups.  The globally-guided variant (below) adds
        # capacity constraints -- similar to how our lifecycle prevents
        # a single expert from accumulating too many domains.
        for i in range(num_groups, self.num_experts):
            expert_idx = indices_by_usage[i]
            similarities_to_cores = sim_matrix[expert_idx, core_indices]
            best_core = core_indices[torch.argmax(similarities_to_cores)]
            best_label = self._group_labels[layer_name][best_core]
            self._group_labels[layer_name][expert_idx] = best_label

        return core_indices.tolist()


    # -------------------------------------------------------------------
    # 4. GLOBALLY-GUIDED GROUPING (across layers)
    #    Assign different num_groups per layer based on usage patterns.
    # -------------------------------------------------------------------

    def assign_groups_globally(
        self,
        layer_names: List[str],
        average_num_groups: int,
    ) -> Dict[str, int]:
        """Assign per-layer group counts so the total across all layers
        matches the budget.

        Layers with more frequently-used experts get more groups (less
        merging); layers with idle experts get fewer groups (more merging).

        # TRIBE NOTE: This global budget allocation is something our
        # lifecycle does implicitly: layers/experts with low fitness get
        # recycled, effectively "merging" them into a fresh expert.  But
        # we don't have an explicit cross-layer coordination mechanism.
        # A global view could prevent over-pruning in critical layers.
        """
        total = average_num_groups * len(layer_names)
        all_freq = torch.cat([self._usage_frequency[n] for n in layer_names])
        sorted_freq, _ = torch.sort(all_freq, descending=True)

        threshold = sorted_freq[min(total, len(sorted_freq) - 1)]

        groups_per_layer = {}
        for name in layer_names:
            count = (self._usage_frequency[name] > threshold).sum().item()
            groups_per_layer[name] = max(1, count)  # at least 1 group
        return groups_per_layer


# ---------------------------------------------------------------------------
# 5. FREQUENCY-WEIGHTED MERGE
#    The actual parameter merging: weighted average by usage frequency.
# ---------------------------------------------------------------------------

def merge_experts_by_usage_frequency(
    expert_wi_weights: Dict[int, torch.Tensor],
    expert_wo_weights: Dict[int, torch.Tensor],
    group_labels: torch.Tensor,
    usage_frequencies: torch.Tensor,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Merge grouped experts using usage-frequency-weighted averaging.

    For each group:
      merged_W = sum(freq_i * W_i) / sum(freq_i)

    The dominant expert (first in the group) gets the merged weights;
    all other experts in the group become references to it.

    Args:
        expert_wi_weights: {expert_idx: weight_tensor} for input projection
        expert_wo_weights: {expert_idx: weight_tensor} for output projection
        group_labels: (num_experts,) cluster assignments
        usage_frequencies: (num_experts,) normalized routing frequencies

    Returns:
        merged_weights: {dominant_idx: {"wi": tensor, "wo": tensor}}

    # TRIBE NOTE: This is the merge execution -- analogous to our bond()
    # weight combination step.  Key difference in weighting:
    #   MC-SMoE: freq_i / sum(freq)  -- how often the router picks expert i
    #   TRIBE:   fitness-proportionate -- how well expert i actually performs
    #
    # Their frequency weighting assumes the router is well-calibrated.
    # Our fitness weighting is more robust when the router makes mistakes,
    # because fitness reflects actual loss, not just routing popularity.
    #
    # Another difference: after merging, MC-SMoE makes all group members
    # point to the same weight tensor (parameter sharing).  Our bond()
    # creates a genuinely new expert with blended weights and frees the
    # redundant one for reuse via recycle().
    """
    merged = {}
    for label in group_labels.unique():
        indices = torch.where(group_labels == label)[0]

        with torch.no_grad():
            # Weighted sum of input projection weights
            wi_stack = torch.stack([
                expert_wi_weights[idx.item()] * usage_frequencies[idx]
                for idx in indices
            ], dim=0)
            wo_stack = torch.stack([
                expert_wo_weights[idx.item()] * usage_frequencies[idx]
                for idx in indices
            ], dim=0)

            freq_sum = usage_frequencies[indices].sum() + FP32_EPS
            merged_wi = wi_stack.sum(dim=0) / freq_sum
            merged_wo = wo_stack.sum(dim=0) / freq_sum

        dominant_idx = indices[0].item()
        merged[dominant_idx] = {"wi": merged_wi, "wo": merged_wo}

    return merged


# ---------------------------------------------------------------------------
# 6. SPECTRAL CLUSTERING ALTERNATIVE
#    MC-SMoE also supports spectral clustering on the similarity matrix.
# ---------------------------------------------------------------------------

def group_by_spectral_clustering(
    similarity_matrix: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Alternative grouping via spectral clustering on the affinity matrix.

    # TRIBE NOTE: Spectral clustering treats the similarity matrix as a
    # graph and finds clusters by partitioning the graph.  This can find
    # non-convex clusters that k-means would miss.  Could be useful if
    # expert competence domains have complex, non-spherical boundaries
    # in feature space.  However, it requires the full similarity matrix,
    # which scales O(n^2) in the number of experts.
    """
    # In the original code, this calls sklearn.cluster.SpectralClustering
    # with affinity="precomputed".  Pseudocode:
    #
    # clustering = SpectralClustering(
    #     n_clusters=num_groups,
    #     affinity="precomputed",
    #     assign_labels="kmeans",
    # ).fit(similarity_matrix.cpu().numpy())
    # return torch.tensor(clustering.labels_)
    raise NotImplementedError("Requires sklearn; see original repo")
