"""08 -- DynMoE: Dynamic Grow/Shrink via Gated Expert Selection

Paper: Dynamic Mixture of Experts: An Auto-Tuning Approach for Efficient
       Transformer Models (Guo, Cheng, Tang, Lin, 2024)
URL:   https://arxiv.org/abs/2405.14297
Repo:  LINs-lab/DynMoE

Extracted from: EMoE/tutel/tutel/gates/gated_multi_gate.py,
                EMoE/tutel/tutel/impls/moe_layer.py,
                EMoE/tutel/tutel/impls/losses.py

DynMoE's key innovation: instead of a fixed top-k routing, each token
dynamically selects HOW MANY experts to use via learnable per-expert
thresholds.  If similarity(token, expert) > threshold, the expert fires.
This naturally grows/shrinks the active expert set per input.

Additionally, DynMoE can add/remove experts from the pool between
training epochs based on routing statistics (which experts were used,
which were idle, and whether unrouted tokens need a new expert).

This is a REFERENCE EXTRACT for study -- not runnable as-is.
Original PyTorch preserved; distributed-computing wiring removed.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any


# ---------------------------------------------------------------------------
# 1. STRAIGHT-THROUGH SIGN ESTIMATOR
#    Enables gradient flow through the binary expert selection.
# ---------------------------------------------------------------------------

class SignStraightThrough(torch.autograd.Function):
    """Sign function with straight-through gradient estimator.

    Forward: sign(x) -- produces {-1, 0, +1}
    Backward: passes gradient through unchanged (identity)

    # TRIBE NOTE: This is how they make discrete expert selection
    # differentiable.  In our system, routing is external (round-robin
    # or loss-based), so we don't need this trick.  But if we ever
    # want to learn the routing jointly with the experts, this STE
    # approach is the standard way to handle discrete decisions in
    # backprop.
    """

    @staticmethod
    def forward(ctx: Any, scores: Tensor) -> Tensor:
        return torch.sign(scores)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        return grad_output


# ---------------------------------------------------------------------------
# 2. DYNAMIC GATING MODULE (GAMoE Gate)
#    The core of DynMoE: per-token adaptive expert count.
# ---------------------------------------------------------------------------

class DynamicExpertGate(torch.nn.Module):
    """Gating module that dynamically decides how many experts to activate.

    Instead of fixed top-k, each expert has a learnable threshold gate.
    A token activates expert i if:
        similarity(token, expert_i) > sigmoid(gate_i)

    The number of active experts varies per token based on input difficulty.

    # TRIBE NOTE: This is the per-input analogue of our per-generation
    # grow/shrink.  DynMoE adapts the ACTIVE expert count on every
    # forward pass; our lifecycle adapts the TOTAL expert count across
    # generations.  Both solve the same fundamental problem: "how many
    # experts does this problem need?"
    #
    # Key difference in granularity:
    #   DynMoE: easy tokens -> 1 expert, hard tokens -> 5 experts
    #   TRIBE:  easy tasks -> 3 total experts, hard tasks -> 8 total
    #
    # These are complementary: DynMoE's per-token routing could sit
    # inside our per-generation lifecycle.
    """

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        max_expert_pool: int = 64,
    ):
        super().__init__()
        self.num_active_experts = num_experts
        self.max_expert_pool = max_expert_pool

        # Similarity matrix: columns are expert "keys" in the input space.
        # Each expert is represented by a learned direction in model_dim.
        # TRIBE NOTE: This is essentially a learned router -- each expert
        # has a prototype vector, and tokens are matched by cosine
        # similarity.  Our loss-based routing doesn't need prototypes;
        # it evaluates each expert on each pattern directly.  Prototype
        # routing is O(1) per expert; loss routing is O(forward_pass).
        self.sim_matrix = torch.nn.Parameter(
            torch.nn.init.orthogonal_(
                torch.empty(max_expert_pool, model_dim)
            ).T.contiguous(),
            requires_grad=True,
        )

        # Per-expert threshold gates (learnable)
        # sigmoid(gate_i) = minimum similarity for expert i to activate.
        # TRIBE NOTE: This is the grow/shrink control.  High gate ->
        # expert rarely fires (shrink effect).  Low gate -> expert fires
        # often (grow effect).  The gradient naturally pushes gates to
        # balance load: if an expert is useful, its gate lowers; if it
        # wastes compute, its gate rises.
        self.gates = torch.nn.Parameter(
            torch.zeros(max_expert_pool), requires_grad=True
        )

        # Binary mask: which experts exist in the pool
        # 1.0 = active (available for routing), 0.0 = removed/not-yet-added
        # TRIBE NOTE: This is our expert.state in {ACTIVE, FROZEN, RETIRED}.
        # Their mask is simpler (binary), but the idea is the same:
        # not all expert slots need to be in use at all times.
        self.experts_mask = torch.nn.Parameter(
            torch.zeros(max_expert_pool), requires_grad=False
        )
        self.experts_mask[:num_experts] = 1.0

        self.capacity_factor = 0.0   # no token dropping
        self.gate_noise = 0.0        # no stochastic routing
        self.adaptive_top_k = True   # signal to the MoE layer
        self.enable_softmax_logits = False

    def forward(self, x: Tensor):
        """Route tokens to a dynamic number of experts.

        Args:
            x: (batch * seq_len, model_dim) input hidden states.

        Returns:
            logits: (batch * seq_len, max_pool) expert activation scores
            top_k: (batch * seq_len,) per-token number of active experts

        Algorithm:
          1. Compute cosine similarity between each token and each expert key
          2. Mask out inactive experts (experts_mask)
          3. Subtract per-expert threshold: ReLU(sim - sigmoid(gate))
          4. Apply straight-through sign to get binary decisions
          5. Count active experts per token = top_k
        """
        # Step 1: Cosine similarity between tokens and expert prototypes
        # TRIBE NOTE: The normalization makes this purely directional --
        # expert magnitude doesn't matter, only the "direction" of
        # expertise.  This is a nice property: it prevents experts from
        # gaming the router by simply scaling up their prototype vector.
        logits = torch.sigmoid(
            torch.matmul(
                F.normalize(x, dim=1),
                F.normalize(self.sim_matrix, dim=0),
            )
        )

        # Step 2: Mask inactive experts
        logits = logits * self.experts_mask

        # Step 3-4: Threshold and binarize
        thresholds = torch.sigmoid(self.gates)

        if self.training:
            # Training: hard threshold with STE gradient
            logits = F.relu(logits - thresholds)
            logits = SignStraightThrough.apply(logits)
            top_k = torch.sum(logits > 0, dim=1).to(torch.int)
        else:
            # Inference: same threshold, but ensure at least 1 expert
            # TRIBE NOTE: The fallback to "at least 1" prevents tokens
            # from being completely unprocessed.  In our system, the
            # round-robin router guarantees every pattern hits at least
            # one expert, so we don't need this safety net.
            active = F.relu(logits - thresholds)
            active = SignStraightThrough.apply(active)
            top_k = torch.sum(active > 0, dim=1).to(torch.int)

            # If a token activated zero experts, fall back to raw logits
            # (route to the best expert regardless of threshold)
            zero_mask = (torch.sum(active, dim=1) == 0).to(torch.int)
            zero_mask = zero_mask.unsqueeze(1).expand_as(logits)
            logits = zero_mask * logits + (1 - zero_mask) * active

            top_k = torch.max(top_k, torch.ones_like(top_k))

        return logits, top_k


# ---------------------------------------------------------------------------
# 3. EXPERT POOL MANAGEMENT: ADD / REMOVE
#    Adapt the expert pool between training epochs.
# ---------------------------------------------------------------------------

def remove_unused_experts(
    gate: DynamicExpertGate,
    routing_records: torch.Tensor,
):
    """Remove experts that received zero tokens during an epoch.

    # TRIBE NOTE: This is exactly our recycle() triggered by zero fitness.
    # Their implementation is simpler: just zero out the mask bit.  Our
    # recycle() additionally transfers knowledge (via weight blending with
    # the fittest expert) before freeing the slot.  The DynMoE approach
    # assumes the expert was truly useless; ours hedges by preserving
    # any knowledge the expert might have had.
    """
    # routing_records[i] = number of tokens routed to expert i
    usage_sign = torch.sign(routing_records)
    # Experts with zero usage get masked out
    gate.experts_mask.data *= usage_sign


def add_expert_for_unrouted_tokens(
    gate: DynamicExpertGate,
    sample_records: torch.Tensor,
):
    """Add a new expert to handle tokens that no expert claimed.

    If tokens exist that were not routed to any expert (because all
    thresholds were too high), create a new expert whose prototype
    vector points toward those unrouted tokens.

    # TRIBE NOTE: This is a form of demand-driven growth.  In our
    # lifecycle, growth happens via bond() (splitting a specialist) or
    # when the system detects patterns that no expert handles well.
    # DynMoE's approach is more direct: "these tokens have no home,
    # so build one."  The prototype is initialized as the mean of the
    # unrouted token embeddings -- a reasonable starting point.
    #
    # A key difference: DynMoE adds experts to a FIXED-SIZE pool
    # (max_expert_pool).  Once the pool is full, no more experts can
    # be added.  Our lifecycle has no hard cap -- bond() can always
    # create a new expert (though system resources are finite in practice).
    """
    if sample_records is None:
        return  # no unrouted tokens

    # Normalize the accumulated unrouted-token embeddings
    prototype = sample_records / torch.norm(sample_records)

    # Find a free slot in the expert pool
    inactive = (gate.experts_mask.data == 0).nonzero(as_tuple=True)[0]
    if len(inactive) == 0:
        return  # pool is full, can't add

    new_idx = inactive[0].item()
    gate.experts_mask.data[new_idx] = 1.0
    gate.sim_matrix.data[:, new_idx] = prototype
    gate.gates.data[new_idx] = 0.0  # neutral threshold


def adaptive_update(
    gate: DynamicExpertGate,
    routing_records: torch.Tensor,
    sample_records: torch.Tensor,
):
    """Full grow/shrink cycle: remove idle experts, add needed ones.

    # TRIBE NOTE: This is the equivalent of our generation-end lifecycle
    # pass (health_check -> recycle -> bond).  DynMoE runs it at epoch
    # boundaries; we run it after each training generation.  Same
    # structure: first clean up (remove/recycle), then grow (add/bond).
    """
    before = int(gate.experts_mask.sum())
    remove_unused_experts(gate, routing_records)
    add_expert_for_unrouted_tokens(gate, sample_records)
    after = int(gate.experts_mask.sum())
    return before, after


# ---------------------------------------------------------------------------
# 4. DIVERSITY LOSS
#    Encourages expert prototypes to be orthogonal (different expertise).
# ---------------------------------------------------------------------------

def diverse_and_simple_gate_loss(
    scores: torch.Tensor,
    top_ids: torch.Tensor,
    sim_matrix: torch.nn.Parameter,
    expert_mask: torch.nn.Parameter,
) -> torch.Tensor:
    """Loss that pushes expert prototypes apart + keeps gates simple.

    Two terms:
      1. Diversity: ||normalize(S)^T * normalize(S) - I|| over active experts
         Penalizes experts whose prototype vectors are too similar.
      2. Simplicity: mean(||gate||) -- L2 regularization on gate magnitudes
         Prevents gates from growing too large (which would suppress all routing).

    # TRIBE NOTE: The diversity loss is the training-time analogue of
    # our measure_overlap() check.  If two experts become too similar
    # during training, the loss pushes them apart BEFORE they need to
    # be merged.  Our lifecycle is reactive (merge after overlap is
    # detected); DynMoE is proactive (prevent overlap via loss).
    #
    # Both approaches have merit:
    #   Proactive (DynMoE): prevents redundancy, but limits the
    #     system's ability to have backup/redundant experts.
    #   Reactive (TRIBE): allows temporary redundancy (which may aid
    #     exploration), then cleans up via bond/recycle.
    """
    # Expert prototype similarity matrix
    normed = F.normalize(sim_matrix, dim=0)
    sims = torch.matmul(normed.T, normed)

    # Only penalize similarity among active expert pairs
    active_mask = torch.matmul(
        expert_mask.unsqueeze(0).T, expert_mask.unsqueeze(0)
    )
    target = torch.eye(sims.shape[0], device=sims.device)

    # Diversity loss: off-diagonal similarities should be zero
    sim_loss = torch.norm(sims * active_mask - target * active_mask)

    # Simplicity loss: keep gate magnitudes small
    simple_loss = torch.mean(torch.norm(sim_matrix, dim=0))

    return sim_loss + simple_loss


# ---------------------------------------------------------------------------
# 5. ONE-SCORE GATE UPDATE
#    Alternative routing where gate weights are derived from expert weights.
# ---------------------------------------------------------------------------

def update_gate_from_expert_weights(
    gate_weight: torch.Tensor,
    expert_fc1_weights: torch.Tensor,
    expert_fc2_weights: torch.Tensor,
    momentum: float = 0.0,
    normalize: bool = True,
    value_norm_weighted: bool = False,
) -> torch.Tensor:
    """Derive router weights from expert FFN weights (no learned router).

    If value_norm_weighted:
        gate = mean_over_hidden(fc1 * normalize(||fc2||))
    Else:
        gate = mean_over_hidden(fc1)

    The router weight for expert i is the mean of its first-layer weight
    matrix, optionally weighted by its second-layer norm.

    # TRIBE NOTE: This "one-score gate" removes the learned router entirely
    # and derives routing from the expert weights themselves.  This is
    # conceptually similar to our loss-based routing: instead of learning
    # a separate router, let the experts' own properties determine routing.
    # Their proxy (weight mean) is much cheaper than ours (actual loss
    # evaluation) but less accurate -- an expert's weight mean doesn't
    # tell you what inputs it handles well.
    """
    if value_norm_weighted:
        value_norm = torch.norm(expert_fc2_weights, p=2, dim=-1, keepdim=True)
        value_weight = F.normalize(value_norm, dim=-2, p=1)
        new_gate = (expert_fc1_weights * value_weight).sum(dim=1)
    else:
        new_gate = expert_fc1_weights.mean(dim=1)

    if normalize:
        new_gate = F.normalize(new_gate, dim=-1)

    if momentum > 0:
        new_gate = momentum * gate_weight + (1 - momentum) * new_gate

    return new_gate
