"""07 -- REAP: Router-weighted Expert Activation Pruning

Paper: REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression
       (Lasby, Lazarevich, Sinnadurai, Lie, Ioannou, Thangarasa, 2025)
URL:   https://arxiv.org/abs/2510.13999
Repo:  CerebrasResearch/reap

Extracted from: src/reap/observer.py, src/reap/metrics.py, src/reap/prune.py

REAP's key contribution: expert importance = activation_norm * router_weight.
This is a principled saliency criterion that captures both how much an
expert changes the hidden state (activation norm) and how confidently the
router selects it (router weight).  Key finding: pruning > merging for
one-shot compression across 20B-1T parameter SMoE models.

This is a REFERENCE EXTRACT for study -- not runnable as-is.
Original PyTorch preserved; framework-specific hooks simplified.
"""

import torch
import torch.nn.functional as F
from typing import List, Callable, Optional
import math


# ---------------------------------------------------------------------------
# 1. DISTANCE METRICS
#    Used for both pruning saliency and merge-quality estimation.
# ---------------------------------------------------------------------------

def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity. Range: [0, 2].

    # TRIBE NOTE: We use a similar metric in measure_overlap() to compare
    # expert loss profiles.  REAP uses it to compare expert activation
    # patterns -- both are trying to answer "are these two experts doing
    # the same thing?" but from different observation points.
    """
    CHUNK_SIZE = 16
    chunks = max(1, int(x.shape[0] // CHUNK_SIZE))
    sims = []
    for xc, yc in zip(x.chunk(chunks, dim=0), y.chunk(chunks, dim=0)):
        sims.append(F.cosine_similarity(xc, yc, dim=-1))
    return 1.0 - torch.cat(sims, dim=0)


def angular_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Normalized angular distance: acos(cos_sim) / pi. Range: [0, 1]."""
    CHUNK_SIZE = 16
    chunks = max(1, int(x.shape[0] // CHUNK_SIZE))
    sims = []
    for xc, yc in zip(x.chunk(chunks, dim=0), y.chunk(chunks, dim=0)):
        sims.append(F.cosine_similarity(xc, yc, dim=-1))
    similarity = torch.cat(sims, dim=0)
    clamped = torch.clamp(similarity, -1.0, 1.0)
    return torch.acos(clamped) / math.pi


# ---------------------------------------------------------------------------
# 2. ONLINE STATISTICS TRACKER (Welford + Kahan)
#    Numerically stable running mean for accumulating saliency scores
#    across many calibration batches.
# ---------------------------------------------------------------------------

class OnlineStatsTracker:
    """Welford's algorithm with Kahan summation for stable online mean.

    # TRIBE NOTE: This is more sophisticated than our simple running
    # averages in health_check().  Kahan summation compensates for
    # floating-point drift over thousands of updates.  Worth adopting
    # if we ever run the lifecycle over very long training runs where
    # accumulated numerical error could corrupt fitness estimates.
    """

    def __init__(self, shape, count_shape=1, device="cpu", dtype=torch.float32):
        self.count = torch.zeros(count_shape, dtype=torch.long, device=device)
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self._compensation = torch.zeros(shape, dtype=dtype, device=device)

    def update(self, new_value: torch.Tensor, new_count):
        new_count = new_count.to(self.count.device, torch.long)
        new_value = new_value.to(self.mean.device, dtype=self.mean.dtype)

        updated_count = self.count + new_count
        delta = new_value - self.mean

        # Kahan-compensated mean update
        y = delta * new_count / updated_count
        y = y.nan_to_num(0)
        y = y - self._compensation
        t = self.mean + y
        self._compensation = (t - self.mean) - y
        self.mean = t
        self.count = updated_count


# ---------------------------------------------------------------------------
# 3. CORE SALIENCY METRICS
#    These are the expert importance scores that REAP computes.
# ---------------------------------------------------------------------------

def compute_expert_saliency(
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    renormalize: bool = False,
) -> dict:
    """Compute per-expert saliency scores from one forward pass.

    Args:
        activations: (num_experts, total_tokens, hidden_dim)
            Output of each expert for every token (even unrouted ones).
        selected_experts: (total_tokens, top_k)
            Which experts were selected per token.
        router_logits: (total_tokens, num_experts)
            Raw router output before softmax.
        num_experts: total number of experts in this layer.
        renormalize: if True, renormalize router weights to sum to 1
                     over only the selected top-k experts.

    Returns:
        Dictionary of saliency scores:
          - ean_sum: sum of activation norms for routed tokens
          - ean_mean: mean activation norm per expert
          - weighted_ean_sum: sum of (activation_norm * router_weight)
          - reap: mean of (activation_norm * router_weight) -- the REAP score

    # TRIBE NOTE: REAP = E[||expert_output|| * router_weight]
    # This could directly replace our unique_knowledge() for recycle
    # decisions.  An expert with low REAP score is both weakly activated
    # (low norm) and rarely/weakly selected (low router weight) -- a
    # clear candidate for recycling.
    #
    # Key insight from the paper: activation norm alone (EAN) misses
    # experts that fire strongly but rarely.  Router frequency alone
    # misses experts that are often selected but barely change the
    # hidden state.  REAP captures both failure modes.
    #
    # In our lifecycle terms:
    #   low REAP = low fitness + low unique_knowledge -> recycle()
    #   high REAP = high impact expert -> protect from recycling
    """
    device = activations.device
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(device)

    if renormalize:
        topk_weights = torch.gather(routing_weights, 1, selected_experts)
        routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True)
        routing_weights = torch.clamp(
            routing_weights, min=torch.finfo(routing_weights.dtype).eps
        )

    # Per-expert accumulators
    ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
    weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
    reap = torch.zeros(num_experts, device=device, dtype=torch.float32)

    expert_frequency = torch.bincount(
        selected_experts.view(-1), minlength=num_experts
    )

    for i in range(num_experts):
        # Find tokens actually routed to this expert
        active_mask = (selected_experts == i).any(dim=-1)
        if not active_mask.any():
            continue

        # Router confidence for this expert on its routed tokens
        active_router_weights = routing_weights[active_mask, i]

        # L2 norm of expert output (activation magnitude)
        # TRIBE NOTE: This is the "how much does this expert change
        # the hidden state?" signal.  An expert whose output is near-zero
        # is not contributing, regardless of how often it's selected.
        # Our health_check() uses gradient norm as a proxy for this --
        # if the gradient is small, the expert isn't learning, which
        # correlates with small activation norm at convergence.
        ean_norm = torch.linalg.norm(activations[i, active_mask, :], dim=-1)

        ean_sum[i] = ean_norm.sum()
        ean_mean[i] = ean_norm.mean()
        weighted_ean_sum[i] = (ean_norm * active_router_weights).sum()

        # THE REAP SCORE: activation_norm * router_weight, averaged
        # TRIBE NOTE: This is the paper's core contribution.  Simple
        # but effective.  The product captures both "does this expert
        # produce a meaningful output?" (norm) and "does the router
        # trust this expert?" (weight).  Experts that score low on
        # both dimensions are safe to prune.
        reap[i] = (ean_norm * active_router_weights).mean()

    return {
        "expert_frequency": expert_frequency,
        "ean_sum": ean_sum,
        "ean_mean": ean_mean,
        "weighted_ean_sum": weighted_ean_sum,
        "reap": reap,
    }


# ---------------------------------------------------------------------------
# 4. PRUNING DECISION
#    Given accumulated saliency scores, decide which experts to remove.
# ---------------------------------------------------------------------------

def prune_experts_by_saliency(
    saliency_scores: torch.Tensor,
    n_experts_to_prune: int,
    protected_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select experts with lowest saliency for removal.

    Args:
        saliency_scores: (num_experts,) per-expert importance score.
            Can be any of: reap, ean_sum, ean_mean, expert_frequency, etc.
        n_experts_to_prune: how many experts to remove.
        protected_indices: expert indices to never prune (e.g., "super experts"
            that handle outlier activations).

    Returns:
        experts_to_prune: (n_experts_to_prune,) indices of experts to remove.

    # TRIBE NOTE: This is a one-shot decision -- prune the bottom K.
    # Our lifecycle makes incremental decisions: recycle() replaces ONE
    # expert per health check, giving the system time to redistribute
    # load.  REAP's approach is better for post-training compression
    # where you want to shrink the model once.  Our approach is better
    # for online adaptation where expertise demands shift over time.
    #
    # Key paper finding: pruning consistently outperforms merging for
    # one-shot compression.  This suggests that when two experts are
    # redundant, it's better to just delete one than to average their
    # weights.  For our lifecycle, this means recycle() (which frees
    # the slot for reuse) may be better than bond() (which blends
    # weights) when the goal is compression rather than specialization.
    """
    scores = saliency_scores.clone()

    if protected_indices is not None:
        scores[protected_indices] = float("inf")

    _, experts_to_prune = torch.topk(scores, n_experts_to_prune, largest=False)
    return experts_to_prune


# ---------------------------------------------------------------------------
# 5. TOKEN-TO-TOKEN SIMILARITY MATRIX (TTM)
#    Pairwise expert distance based on how similarly they transform tokens.
# ---------------------------------------------------------------------------

def ttm_similarity_online(
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    distance_fn: Callable,
    num_experts: int,
    pairwise_expert_frequency: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise expert distance based on output activations.

    For each token routed to expert i, compute the distance between
    expert i's output and expert j's output on the same token.  Average
    over all such tokens.

    # TRIBE NOTE: This is a functional overlap metric -- exactly what
    # we want for measure_overlap().  If expert i and expert j produce
    # similar outputs on the tokens routed to either of them, they are
    # functionally redundant.  This is strictly more informative than
    # weight-space distance (HC-SMoE) or router-logit correlation
    # (MC-SMoE) because it measures actual behavior.
    #
    # The cost is that you need to run all experts on all tokens (or at
    # least on the union of their routed tokens).  For our small-scale
    # lifecycle this is cheap; for LLM-scale it requires the calibration
    # pass that REAP already does.
    """
    device = activations.device
    E, S, H = activations.shape
    K = selected_experts.shape[1]

    pairwise_distances = torch.zeros((num_experts, num_experts), device=device)

    # (S, E, H) -- rearrange for gather
    act_permuted = activations.permute(1, 0, 2)

    # Gather activations for selected experts: (S, K, H)
    selected_acts = torch.gather(
        act_permuted, 1, selected_experts.unsqueeze(-1).expand(-1, -1, H)
    )

    # Distance from each selected expert to ALL experts: (S, K, E)
    dist_matrix = distance_fn(
        selected_acts.unsqueeze(2),      # (S, K, 1, H)
        act_permuted.unsqueeze(1),       # (S, 1, E, H)
    )

    # Scatter-add distances into the pairwise matrix
    idx = selected_experts.view(S * K)
    flat_dists = dist_matrix.view(S * K, E)
    pairwise_distances.scatter_add_(
        0, idx.unsqueeze(-1).expand(-1, E), flat_dists
    )

    # Symmetrize and normalize
    pairwise_distances = pairwise_distances + pairwise_distances.T
    pairwise_distances = pairwise_distances / pairwise_expert_frequency
    pairwise_distances = pairwise_distances.nan_to_num(0)
    pairwise_distances.fill_diagonal_(0)

    return pairwise_distances


# ---------------------------------------------------------------------------
# 6. CHARACTERISTIC ACTIVATION (for merging -- contrast with pruning)
#    Per-expert "average output" used to measure merge quality.
# ---------------------------------------------------------------------------

def routed_characteristic_activation(
    activations: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_frequency: torch.Tensor,
    num_experts: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Compute per-expert mean activation over routed tokens only.

    # TRIBE NOTE: This is the "expert signature" -- what the expert
    # typically outputs.  Two experts with similar characteristic
    # activations are doing the same thing and can be safely merged.
    # This could augment our measure_overlap() with a functional
    # fingerprint rather than relying solely on loss correlation.
    """
    device = activations.device
    # (S, E, H)
    act_permuted = activations.permute(1, 0, 2)

    index_for_gather = selected_experts.unsqueeze(-1).expand(-1, -1, hidden_dim)
    gathered = act_permuted.gather(dim=1, index=index_for_gather)

    src = gathered.reshape(-1, hidden_dim)
    index_for_scatter = selected_experts.reshape(-1, 1).expand(-1, hidden_dim)

    ca = torch.zeros(num_experts, hidden_dim, dtype=torch.float64, device=device)
    ca.scatter_add_(dim=0, index=index_for_scatter, src=src.to(torch.float64))

    # Normalize by how many tokens were routed to each expert
    ca = ca / expert_frequency.unsqueeze(-1)
    return ca.nan_to_num(0)
