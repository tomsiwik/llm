"""03 -- ScatterMoE Parallel Expert Dispatch

Paper: ScatterMoE: Mixture-of-Experts with Scatter/Gather for Efficient
       Inference and Training (Tan, 2024)
URL:   https://arxiv.org/abs/2403.08245
Repo:  shawntan/scattermoe

Extracted from: scattermoe/parallel_experts.py, scattermoe/mlp.py

ScatterMoE replaces the sequential expert-by-expert loop in standard MoE
with a single batched matrix multiply using scatter/gather index operations.
All experts share one weight tensor of shape [n_experts, out_dim, in_dim],
and tokens are dispatched in parallel via sorted expert indices.

This file extracts:
  1. flatten_sort_count  -- the index-preparation step
  2. ParallelLinear      -- the core scatter2scatter autograd function
  3. ParallelExperts     -- the nn.Module wrapper
  4. MLP                 -- two-layer expert MoE using parallel dispatch
"""

import torch
import torch.nn as nn
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Index preparation: sort tokens by assigned expert
# ---------------------------------------------------------------------------

def flatten_sort_count(expert_idxs: torch.Tensor, num_experts: int):
    """Prepare dispatch indices by sorting tokens according to their expert.

    Given expert assignments for each token (possibly top-k, so each token
    may appear multiple times), this function:
      1. Flattens the [n_tokens, k] assignment tensor
      2. Sorts by expert index so tokens for the same expert are contiguous
      3. Counts how many tokens go to each expert (for offset computation)

    # TRIBE NOTE: This is the key enabler for parallel dispatch.  Our
    # `batch_route_by_loss` builds per-expert index lists with a Python
    # loop: `[torch.eq(routes, i).nonzero() for i in range(n_experts)]`.
    # ScatterMoE does the same thing but in one fused sort + bincount,
    # which is much faster on GPU because it avoids n_experts separate
    # kernel launches.

    Args:
        expert_idxs: [n_tokens, k] -- expert assignments (from top-k routing)
        num_experts: total number of experts

    Returns:
        sorted_expert_idxs:     [n_tokens * k] -- expert ids, sorted
        sorted_scattered_idxs:  [n_tokens * k] -- original positions, sorted
            to match.  Used to scatter results back to token order.
        expert_offsets:          [num_experts] -- cumulative token counts.
            expert_offsets[i] = number of tokens assigned to experts 0..i.
    """
    flattened_expert_idxs = expert_idxs.flatten()

    # Sort by expert id -- this groups tokens for the same expert together
    sorted_expert_idxs, sorted_scattered_idxs = torch.sort(flattened_expert_idxs)

    # Count tokens per expert (like np.bincount)
    expert_counts = torch.bincount(flattened_expert_idxs, minlength=num_experts)

    # Cumulative offsets for indexing into the sorted array
    expert_offsets = expert_counts.cumsum(-1)

    return sorted_expert_idxs, sorted_scattered_idxs, expert_offsets


# ---------------------------------------------------------------------------
# 2. ParallelLinear: the core scatter2scatter matmul
# ---------------------------------------------------------------------------

class ParallelLinear(torch.autograd.Function):
    """Batched linear layer that dispatches tokens to different expert weight
    slices in a single operation, using scatter/gather indexing.

    # TRIBE NOTE: This replaces the sequential loop:
    #
    #   for i in range(n_experts):
    #       output[idx[i]] = experts[i](x[idx[i]])
    #
    # with a single batched matmul over a [n_experts, out_dim, in_dim] weight
    # tensor, indexed by the sorted expert assignments.  The actual kernel
    # (scatter2scatter) is a Triton implementation that fuses the gather,
    # matmul, and scatter into one GPU kernel.  We represent it here as a
    # conceptual PyTorch equivalent.

    Forward:
        output[j] = W[expert_of(j)] @ x[token_of(j)] + b[expert_of(j)]
        (optionally scaled by gate values and gathered back to token order)

    The key insight: because tokens are sorted by expert, the weight-lookup
    pattern is contiguous per expert, enabling efficient blocked matmul.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_biases: Optional[torch.Tensor] = None,
        gates: Optional[torch.Tensor] = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ):
        """
        Args:
            x:                     [n_tokens, in_dim]
            expert_weights:        [n_experts, in_dim, out_dim]  (transposed)
            k:                     top-k value (how many experts per token)
            sorted_expert_idxs:    [n_tokens * k] -- which expert for each slot
            sorted_scattered_idxs: [n_tokens * k] -- which token for each slot
            expert_offsets:        [n_experts] -- cumulative count boundaries
            expert_biases:         [n_experts, out_dim] or None
            gates:                 [n_tokens, k] -- gating weights per expert
            grouped_in/out:        whether input/output are already in
                                   expert-grouped order

        # TRIBE NOTE: The `gates` parameter serves the same role as
        # `route_prob_max` in Switch (file 01).  It weights the contribution
        # of each expert to the final output.  In TRIBE we don't have gates;
        # each sample is routed to exactly one expert with weight 1.0.
        """
        # --- Core operation: scatter2scatter matmul ---
        # In the real implementation, this calls a Triton kernel that:
        #   1. Gathers x rows according to sorted_scattered_idxs
        #   2. Multiplies by the correct expert weight slice (indexed by
        #      sorted_expert_idxs)
        #   3. Scatters results to output positions
        #
        # Conceptual PyTorch equivalent (NOT how the Triton kernel works,
        # but logically equivalent):
        output = _scatter2scatter_pytorch(
            x, expert_weights, expert_biases, k,
            sorted_expert_idxs, sorted_scattered_idxs,
            grouped_in, grouped_out,
        )

        # If gating weights are provided, combine top-k expert outputs
        # via weighted sum: y = sum_j gate_j * expert_j(x)
        if gates is not None:
            output_expanded = output.view(gates.size(0), gates.size(1), output.size(-1))
            output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
        else:
            output_expanded = None

        ctx.save_for_backward(
            x, expert_weights, expert_biases,
            sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            gates, output_expanded,
        )
        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Backward pass mirrors forward: scatter2scatter with transposed weights.

        # TRIBE NOTE: The backward is symmetric -- it uses the same
        # scatter/gather machinery to route gradients back through the
        # correct expert weight slices.  Our sequential loop naturally
        # handles this via PyTorch autograd, but it launches n_experts
        # separate backward kernels instead of one fused kernel.
        """
        (x, expert_weights, expert_biases,
         sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
         gates, output_expanded) = ctx.saved_tensors
        k = ctx.k

        # Gate gradients
        if gates is not None:
            d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
        else:
            d_gates = None
            gates_flat = None
            gate_fan = 1

        # d_input: scatter2scatter with transposed expert weights
        # d_weights: grouped outer product of inputs and grad_out
        # (Details omitted -- the real code calls Triton kernels for
        # group_bwd_W and scatter2scatter with transposed weights.)

        # Placeholder: return None for all non-differentiable args
        return (
            grad_out,       # d_input (simplified)
            None,           # d_weights (computed by Triton kernel)
            None, None, None, None,  # k, sorted indices, offsets
            None, d_gates,           # biases, gates
            None, None,              # grouped_in, grouped_out
        )


# ---------------------------------------------------------------------------
# 3. ParallelExperts: nn.Module wrapping the batched weight tensor
# ---------------------------------------------------------------------------

class ParallelExperts(nn.Module):
    """All experts stored as a single [n_experts, out_dim, in_dim] tensor.

    # TRIBE NOTE: In TRIBE, each expert is a separate nn.Module with its own
    # parameter tensors.  This makes lifecycle operations (recycle, freeze)
    # simple -- just reinit or detach one expert's params.  But it means
    # the forward pass requires a Python loop over experts.
    #
    # ScatterMoE packs all expert weights into one tensor.  This enables
    # batched matmul but makes per-expert operations (like our recycle)
    # require index slicing into the shared tensor.
    """

    def __init__(self, num_experts: int, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        # Single tensor for all experts -- enables batched dispatch
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, output_size))
        else:
            self.bias = None

        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        gates: Optional[torch.Tensor] = None,
        grouped_in: bool = False,
        grouped_out: bool = False,
    ) -> torch.Tensor:
        """Dispatch inputs to experts via parallel scatter/gather.

        # TRIBE NOTE: Compare with our sequential dispatch:
        #   for i, expert in enumerate(self.experts):
        #       mask = (routes == i)
        #       output[mask] = expert(x[mask])
        #
        # This single call replaces that entire loop.
        """
        return ParallelLinear.apply(
            inputs, self.weight.permute(0, 2, 1), k,
            sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            self.bias, gates, grouped_in, grouped_out,
        )


# ---------------------------------------------------------------------------
# 4. MLP: two-layer expert MoE using parallel dispatch
# ---------------------------------------------------------------------------

class MoE_MLP(nn.Module):
    """Standard two-layer MoE FFN: h = act(xW1); y = hW2, with parallel dispatch.

    # TRIBE NOTE: This is functionally equivalent to our experts -- each
    # expert is a two-layer FFN.  The difference is purely in dispatch:
    # sequential loop (TRIBE) vs batched scatter/gather (ScatterMoE).
    # For small expert counts (4-8 in our experiments), the loop is fine.
    # For large-scale models with 64-128 experts, scatter/gather is essential.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        bias: bool = False,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.top_k = min(top_k, num_experts)
        self.activation = activation or nn.ReLU()

        # First layer: [n_experts, hidden_size, input_size]
        self.experts = ParallelExperts(num_experts, input_size, hidden_size, bias=bias)
        # Second layer: [n_experts, input_size, hidden_size]
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        expert_p: torch.Tensor,
        expert_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:           [batch, seq_len, input_size] -- input tokens
            expert_p:    [batch * seq_len, top_k] -- gating probabilities
            expert_idxs: [batch * seq_len, top_k] -- expert assignments

        Returns:
            y: [batch, seq_len, input_size]

        The routing decision (expert_p, expert_idxs) comes from an external
        router (not included here).  This module just handles dispatch.
        """
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])

        # Prepare sorted dispatch indices
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = \
            flatten_sort_count(expert_idxs, num_experts=self.num_experts)

        # First expert layer: x -> h (output in expert-grouped order)
        h = self.experts(
            x, self.top_k,
            sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            grouped_out=True,  # keep output in expert-sorted order
        )

        h = self.activation(h)

        # Second expert layer: h -> y (gate-weighted, back to token order)
        # TRIBE NOTE: `gates=expert_p` applies the routing weights here,
        # equivalent to Switch's `output * route_prob_max`.  The k=1 on
        # the second layer means each expert-grouped hidden state maps
        # through exactly one output expert slice, then the gating weights
        # combine the top-k results.
        y = self.output_experts(
            h, 1,
            sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            grouped_in=True,   # input is still in expert-sorted order
            gates=expert_p,    # weight outputs by routing probabilities
        )

        y = y.view(*x_shape[:-1], y.size(-1))
        return y


# ---------------------------------------------------------------------------
# Conceptual scatter2scatter (pure PyTorch, NOT the real Triton kernel)
# ---------------------------------------------------------------------------

def _scatter2scatter_pytorch(
    x: torch.Tensor,
    weights: torch.Tensor,
    biases: Optional[torch.Tensor],
    k: int,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    x_grouped: bool,
    y_grouped: bool,
) -> torch.Tensor:
    """Pure-PyTorch reference for what the Triton scatter2scatter kernel does.

    For each slot j in the sorted dispatch order:
      - Look up which expert:  e = sorted_expert_idxs[j]
      - Look up which token:   t = sorted_scattered_idxs[j] // k
      - Compute:  output[j] = weights[e] @ x[t] + biases[e]

    The real Triton kernel fuses this into blocked matmuls with coalesced
    memory access, handling the expert boundaries via the offset array.

    # TRIBE NOTE: This is the operation our sequential loop implements
    # implicitly.  Making it explicit as a single gather-matmul-scatter
    # is what enables GPU parallelism across all experts simultaneously.
    """
    n_slots = sorted_expert_idxs.size(0)
    out_dim = weights.size(-1)
    output = torch.zeros(n_slots, out_dim, device=x.device, dtype=x.dtype)

    for j in range(n_slots):
        e = sorted_expert_idxs[j].item()
        if x_grouped:
            t = j
        else:
            t = sorted_scattered_idxs[j].item() // k
        out = weights[e] @ x[t]
        if biases is not None:
            out = out + biases[e]
        output[j] = out

    return output
