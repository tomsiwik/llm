"""01 -- Switch Transformer Top-1 Gating and Load Balancing Loss

Paper: Switch Transformers: Scaling to Trillion Parameter Models with Simple
       and Efficient Sparsity (Fedus, Zoph & Shazeer, 2022)
URL:   https://arxiv.org/abs/2101.03961
Repo:  labmlai/annotated_deep_learning_paper_implementations

Extracted from: labml_nn/transformers/switch/__init__.py
                labml_nn/transformers/switch/experiment.py

This file contains the core MoE routing logic from the Switch Transformer:
  1. SwitchFeedForward -- top-1 gating with capacity factor and token dropping
  2. load_balancing_loss -- the auxiliary loss that prevents expert collapse

These are the two pieces most relevant to TRIBE's routing design.
"""

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 1. Switch Feed-Forward: top-1 gating with capacity factor
# ---------------------------------------------------------------------------

class SwitchFeedForward(nn.Module):
    """Route each token to exactly one expert (top-1), with optional token
    dropping when an expert exceeds its capacity budget.

    # TRIBE NOTE: Our `route_by_loss()` is *oracle* routing -- we evaluate
    # every expert on every input and pick the one with lowest loss.  That is
    # O(n_experts) forward passes per sample.  Switch routing is O(1): a single
    # linear projection + argmax.  The trade-off is that Switch needs the load
    # balancing auxiliary loss (see below) to keep experts from collapsing,
    # whereas oracle routing has no collapse risk by construction.
    """

    def __init__(
        self,
        *,
        capacity_factor: float,
        drop_tokens: bool,
        is_scale_prob: bool,
        n_experts: int,
        expert: nn.Module,
        d_model: int,
    ):
        """
        Args:
            capacity_factor: multiplier on the ideal per-expert token budget.
                expert_capacity = (total_tokens / n_experts) * capacity_factor.
                Values > 1 leave headroom; < 1 forces dropping.
            drop_tokens: if True, tokens beyond capacity are passed through
                unchanged (identity).  If False, capacity is advisory only.
            is_scale_prob: if True, multiply expert output by the gating
                probability p_i(x).  This keeps gradient scale proportional
                to the router's confidence.
            n_experts: number of expert FFN copies.
            expert: a single FFN module (will be deep-copied n_experts times).
            d_model: token embedding dimension.
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # TRIBE NOTE: In TRIBE each expert is an independently-initialised
        # network.  Here they start as clones of a single template -- the
        # load-balancing loss is what drives them to specialise.
        self.experts = nn.ModuleList([_clone(expert) for _ in range(n_experts)])

        # Router: a single learned linear projection from token space to
        # expert-selection logits, followed by softmax.
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [seq_len, batch_size, d_model]

        Returns:
            final_output:     [seq_len, batch_size, d_model]
            counts:           [n_experts]  -- tokens routed to each expert
            route_prob_sum:   [n_experts]  -- sum of gating probs per expert
            n_dropped:        int          -- how many tokens were dropped
            route_prob_max:   [total_tokens] -- gating prob of chosen expert
        """
        seq_len, batch_size, d_model = x.shape
        # Flatten to [total_tokens, d_model] for routing
        x = x.view(-1, d_model)

        # --- Gating: softmax over expert logits ---
        # p_i(x) = softmax(h(x))_i  where h is the linear router
        route_prob = self.softmax(self.switch(x))

        # Top-1 selection: pick the expert with highest probability
        # TRIBE NOTE: This is the key difference from our oracle routing.
        # Switch picks based on a *learned prediction* of which expert will
        # be best.  We pick based on *measured loss* after running all experts.
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Build per-expert token index lists
        indexes_list = [
            torch.eq(routes, i).nonzero(as_tuple=True)[0]
            for i in range(self.n_experts)
        ]

        final_output = x.new_zeros(x.shape)

        # --- Capacity and token dropping ---
        # expert_capacity = (total_tokens / n_experts) * capacity_factor
        # TRIBE NOTE: We don't have an explicit capacity factor.  Instead our
        # `health_check()` detects overloaded experts *after the fact* via
        # gradient-norm monitoring and triggers lifecycle events (bond/freeze).
        # Switch prevents overload *proactively* by capping tokens per expert.
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) <= capacity:
                    continue
                # Randomly shuffle, then drop tokens beyond capacity
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                dropped.append(indexes_list[i][capacity:])
                indexes_list[i] = indexes_list[i][:capacity]

        # --- Expert forward passes ---
        # TRIBE NOTE: This loop is sequential over experts, same as our
        # `batch_route_by_loss`.  ScatterMoE (file 03) shows how to
        # parallelise this with scatter/gather ops.
        expert_output = [
            self.experts[i](x[indexes_list[i], :])
            for i in range(self.n_experts)
        ]

        # Scatter expert outputs back to token positions
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Dropped tokens get the identity (pass-through)
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        # --- Probability scaling ---
        if self.is_scale_prob:
            # Scale output by gating probability: y = p_i(x) * E_i(x)
            # This gives the router gradient signal proportional to how
            # useful each expert's output was.
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Straight-through estimator: multiply by p/stop_grad(p) = 1
            # so that the *value* is unchanged but gradients still flow
            # through the router.
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        final_output = final_output.view(seq_len, batch_size, d_model)

        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max


# ---------------------------------------------------------------------------
# 2. Load Balancing Auxiliary Loss
# ---------------------------------------------------------------------------

def load_balancing_loss(
    counts: torch.Tensor,
    route_prob_sum: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Compute the Switch Transformer load-balancing auxiliary loss for one layer.

    L = N * sum_i(f_i * P_i)

    where:
        f_i = fraction of tokens routed to expert i  (hard assignment)
        P_i = mean routing probability for expert i   (soft, differentiable)
        N   = number of experts

    Minimising this loss pushes the router toward uniform load distribution.
    The product f_i * P_i is minimised when both factors are equal to 1/N,
    i.e. perfectly balanced routing.

    # TRIBE NOTE: We have no analogue of this loss.  Our routing is oracle-
    # based (pick the expert with lowest actual loss), so there is no learned
    # router to regularise.  Instead, we handle load imbalance reactively via
    # the lifecycle: overloaded experts trigger bonding, idle experts get
    # recycled.  The load-balancing loss is the *proactive* alternative --
    # it steers the router away from collapse before it happens.

    Args:
        counts:         [n_layers, n_experts] -- token count per expert per layer
        route_prob_sum: [n_layers, n_experts] -- sum of routing probs per expert
        n_experts:      int

    Returns:
        Scalar loss (summed across layers).  Typically multiplied by a small
        coefficient alpha ~ 0.01 before adding to the main loss.
    """
    # Total tokens processed per layer
    total = counts.sum(dim=-1, keepdims=True)  # [n_layers, 1]

    # f_i: fraction of tokens assigned to expert i (hard, non-differentiable)
    route_frac = counts / total  # [n_layers, n_experts]

    # P_i: mean routing probability for expert i (soft, differentiable)
    route_prob = route_prob_sum / total  # [n_layers, n_experts]

    # L = N * sum_i(f_i * P_i), summed across layers
    # TRIBE NOTE: The factor N makes the minimum value equal to 1 (when
    # perfectly balanced), so alpha can be calibrated independently of N.
    loss = n_experts * (route_frac * route_prob).sum()

    return loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone(module: nn.Module) -> nn.Module:
    """Deep-copy a module (stand-in for labml_nn.utils.clone_module_list)."""
    import copy
    return copy.deepcopy(module)
