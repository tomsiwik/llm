"""11 -- Drop-Upcycling: Expert Diversification via Partial Dropout + Reinit

Paper: Drop-Upcycling: Training Sparse Mixture of Experts with Partial
       Re-initialization (Taishi et al., 2024)
URL:   https://arxiv.org/abs/2408.06643
Repo:  Taishi-N324/Drop-Upcycling

Drop-Upcycling converts a dense model to a Mixture-of-Experts (MoE) model
by copying the FFN into N expert slots, then DIVERSIFYING each copy via:
  1. Random permutation of the intermediate dimension
  2. Partial re-initialization from weight statistics (mean, std)
  3. Random gate weight initialization

The key idea: simply copying weights (naive upcycling) creates identical
experts that the router cannot distinguish. Drop-upcycling forces diversity
by randomly replacing a fraction of each expert's weights with noise drawn
from the same distribution, ensuring experts start different but plausible.

Extracted from: conversions/drop_upcycling.py
"""

import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Weight initialization helpers
# ---------------------------------------------------------------------------
def initialize_weights(size: Tuple[int, ...], std: float = 0.02,
                       mean: float = 0.0) -> torch.Tensor:
    """Initialize weights from normal distribution with given statistics."""
    return torch.normal(mean=mean, std=std, size=size)


# TRIBE NOTE: Multiple gate initialization strategies. The choice matters
# because it determines the initial routing before any training. Compare
# with our system where initial routing is either round-robin or loss-based
# from the start -- we never need explicit gate initialization because our
# routing is non-parametric.
def initialize_gate_weights(size: Tuple[int, ...], method: str) -> torch.Tensor:
    """Initialize MoE gate (router) weights.

    Args:
        size: shape of the gate weight matrix [num_experts, hidden_size]
        method: initialization strategy

    Returns:
        Initialized gate weight tensor
    """
    if method == "torch_rand":
        # Uniform [0, 1] -- slightly favors positive routing
        return torch.rand(size)
    elif method == "torch_rand_mean0":
        # Uniform centered at 0 -- balanced initial routing
        weights = torch.rand(size)
        return weights - weights.mean()
    elif method == "torch_normal_002":
        # Small normal -- near-uniform softmax output initially
        return torch.normal(mean=0, std=0.02, size=size)
    elif method == "torch_normal_028":
        # Larger normal -- more variance in initial routing
        # TRIBE NOTE: std=0.2887 = 1/sqrt(12), the std of Uniform[0,1].
        # This creates initial gates with the same spread as uniform.
        return torch.normal(mean=0, std=0.2886751345948129, size=size)
    elif method == "torch_rand_002":
        # Uniform scaled to match small normal -- best of both worlds
        weights = torch.rand(size)
        weights_mean = weights.mean()
        return (weights - weights_mean) * 0.02 * (12 ** 0.5)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


# ---------------------------------------------------------------------------
# Core: Shuffle + Partial Re-initialization for Expert Diversification
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is the key algorithmic contribution. Compare with our
# recycle() operation:
#
#   Drop-Upcycling:
#     1. Copy dense FFN weights to each expert
#     2. Random permutation of intermediate dimension
#     3. Replace fraction of weights with noise from same distribution
#
#   Our recycle():
#     1. Copy weights from best-performing neighbor expert
#     2. Add calibrated Gaussian noise (no permutation)
#     3. Noise std = 10% of weight std
#
# Key differences:
#   - They permute dimensions (structural change); we don't
#   - They reinit from weight statistics; we add noise to existing weights
#   - They diversify ALL experts at init; we diversify ONE at recycle time
#   - Their ffn_init_ratio controls how much to reinit (e.g., 50%)
#   - Our noise ratio is fixed at 10%
#
# Their approach is more aggressive -- replacing 50% of weights breaks the
# original function significantly. Ours is more conservative -- 10% noise
# preserves most of the source expert's knowledge while creating enough
# diversity for the recycled expert to specialize differently.
def shuffle_and_partially_initialize(
    tensor: torch.Tensor,
    perm: torch.Tensor,
    target_size: int,
    is_down_proj: bool,
    ffn_init_ratio: float,
) -> torch.Tensor:
    """Diversify a single FFN weight matrix via permutation + partial reinit.

    Args:
        tensor: original weight matrix from dense model
        perm: random permutation of intermediate dimension indices
        target_size: target intermediate dimension for the MoE expert
        is_down_proj: True for down_proj (w2), False for gate_proj/up_proj
        ffn_init_ratio: fraction of weights to reinitialize (0.0 to 1.0)

    Returns:
        Diversified weight tensor for one expert
    """
    # Step 1: Permute the intermediate dimension
    # TRIBE NOTE: Permutation is a clever way to create structural diversity.
    # For up_proj/gate_proj [intermediate, hidden], we shuffle rows.
    # For down_proj [hidden, intermediate], we shuffle columns.
    # Same permutation is used for gate_proj, up_proj, and down_proj within
    # one expert to maintain internal consistency.
    if is_down_proj:
        # down_proj: [hidden_size, intermediate_size] -- shuffle columns
        shuffled = tensor.index_select(1, perm[:target_size])
    else:
        # gate_proj/up_proj: [intermediate_size, hidden_size] -- shuffle rows
        shuffled = tensor.index_select(0, perm[:target_size])

    # Step 2: Randomly select positions to reinitialize
    init_size = int(target_size * ffn_init_ratio)
    init_indices = torch.randperm(target_size)[:init_size]

    # Step 3: Compute statistics of selected positions, reinit from them
    # TRIBE NOTE: This is the "drop" part -- they measure the mean and std
    # of the selected weights, then replace them with fresh samples from
    # Normal(mean, std). This preserves the weight distribution while
    # destroying the specific learned values.
    #
    # Compare with our approach: we compute noise_std = 0.1 * weight.std()
    # and ADD noise rather than REPLACING. Their approach is more like
    # "forget and re-learn" while ours is "perturb and re-specialize".
    if is_down_proj:
        init_part = shuffled[:, init_indices]
        init_mean = init_part.mean().item()
        init_std = init_part.std().item()
        init_tensor = initialize_weights(
            (tensor.size(0), init_size), std=init_std, mean=init_mean
        ).to(dtype=tensor.dtype)
        shuffled[:, init_indices] = init_tensor
    else:
        init_part = shuffled[init_indices, :]
        init_mean = init_part.mean().item()
        init_std = init_part.std().item()
        init_tensor = initialize_weights(
            (init_size, tensor.size(1)), std=init_std, mean=init_mean
        ).to(dtype=tensor.dtype)
        shuffled[init_indices, :] = init_tensor

    return shuffled


# ---------------------------------------------------------------------------
# Full upcycling procedure: dense model -> MoE model
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is the complete conversion pipeline. It demonstrates
# a one-time "birth" event for all experts simultaneously, as opposed
# to our system where experts are born/recycled incrementally during
# training. The overall structure:
#
#   1. Copy all non-FFN parameters directly (embeddings, attention, norms)
#   2. For each layer, for each expert:
#      a. Generate a random permutation (shared across gate/up/down)
#      b. Apply shuffle_and_partially_initialize to each FFN matrix
#   3. Initialize gate (router) weights randomly
#
# After upcycling, the model needs continued training to:
#   - Teach the router to differentiate between experts
#   - Let each expert specialize away from others
#   - Recover any performance lost from partial reinitialization
def drop_upcycle(
    source_state_dict: dict,
    num_experts: int,
    num_layers: int,
    target_intermediate_size: int,
    ffn_init_ratio: float = 0.5,
    gate_init_method: str = "torch_rand",
) -> dict:
    """Convert a dense model's state dict to a MoE state dict.

    Args:
        source_state_dict: weights from the dense (non-MoE) model
        num_experts: number of experts per MoE layer
        num_layers: number of transformer layers
        target_intermediate_size: intermediate dim for each expert
        ffn_init_ratio: fraction of weights to reinitialize per expert
        gate_init_method: how to initialize router weights

    Returns:
        New state dict with MoE structure
    """
    target_state_dict = {}

    # Copy non-FFN parameters directly
    # TRIBE NOTE: Everything except the FFN (gate_proj, up_proj, down_proj)
    # is shared across experts. This is standard MoE -- only the FFN is
    # replicated as experts. In our system, each expert is a complete
    # sub-network, which gives more flexibility but uses more parameters.
    for name, param in source_state_dict.items():
        if not any(proj in name for proj in ['gate_proj', 'up_proj', 'down_proj']):
            target_state_dict[name] = param.clone()

    # Weight name mapping: MoE expert format -> dense format
    replace_mapping = {
        "block_sparse_moe.experts.{}.w1.weight": "mlp.gate_proj.weight",
        "block_sparse_moe.experts.{}.w2.weight": "mlp.down_proj.weight",
        "block_sparse_moe.experts.{}.w3.weight": "mlp.up_proj.weight",
    }

    for layer_idx in range(num_layers):
        # Initialize gate weights for this layer
        gate_name = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        gate_size = (num_experts, target_intermediate_size)
        target_state_dict[gate_name] = initialize_gate_weights(
            gate_size, gate_init_method
        )

        for expert_idx in range(num_experts):
            # TRIBE NOTE: Same permutation for all three projections within
            # one expert. This ensures that row i of gate_proj/up_proj
            # corresponds to column i of down_proj, maintaining the
            # internal structure of the FFN.
            perm = torch.randperm(target_intermediate_size)

            for moe_pattern, dense_pattern in replace_mapping.items():
                source_name = f"model.layers.{layer_idx}.{dense_pattern}"
                target_name = (
                    f"model.layers.{layer_idx}."
                    f"{moe_pattern.format(expert_idx)}"
                )

                if source_name in source_state_dict:
                    source_tensor = source_state_dict[source_name]
                    is_down_proj = "down_proj" in source_name

                    # TRIBE NOTE: Each expert gets a DIFFERENT permutation
                    # and DIFFERENT random reinit positions. This is what
                    # creates diversity. Expert 0 might keep neurons
                    # [3,7,1,5,2,...] while Expert 1 keeps [6,0,4,8,9,...].
                    # Combined with partial reinit, each expert starts with
                    # a unique "view" of the original dense FFN's knowledge.
                    target_state_dict[target_name] = (
                        shuffle_and_partially_initialize(
                            source_tensor,
                            perm,
                            target_intermediate_size,
                            is_down_proj,
                            ffn_init_ratio,
                        )
                    )

    return target_state_dict


# ---------------------------------------------------------------------------
# For comparison: Naive upcycling (no diversification)
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is the baseline they compare against. Simply copy the
# dense FFN to all experts without any modification. The problem: all
# experts start identical, so the router has no gradient signal to
# differentiate them. Training takes much longer to break symmetry.
#
# This is analogous to initializing all our experts with the same weights
# and hoping noise in SGD creates differentiation -- it works eventually
# but is slow and wasteful.
def naive_upcycle(
    source_state_dict: dict,
    num_experts: int,
    num_layers: int,
    gate_init_method: str = "torch_rand",
) -> dict:
    """Baseline: copy dense FFN to all experts without diversification."""
    target_state_dict = {}

    for name, param in source_state_dict.items():
        if not any(proj in name for proj in ['gate_proj', 'up_proj', 'down_proj']):
            target_state_dict[name] = param.clone()

    replace_mapping = {
        "block_sparse_moe.experts.{}.w1.weight": "mlp.gate_proj.weight",
        "block_sparse_moe.experts.{}.w2.weight": "mlp.down_proj.weight",
        "block_sparse_moe.experts.{}.w3.weight": "mlp.up_proj.weight",
    }

    for layer_idx in range(num_layers):
        gate_name = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        gate_size = (num_experts, 1)  # placeholder size
        target_state_dict[gate_name] = initialize_gate_weights(
            gate_size, gate_init_method
        )

        for expert_idx in range(num_experts):
            for moe_pattern, dense_pattern in replace_mapping.items():
                source_name = f"model.layers.{layer_idx}.{dense_pattern}"
                target_name = (
                    f"model.layers.{layer_idx}."
                    f"{moe_pattern.format(expert_idx)}"
                )
                if source_name in source_state_dict:
                    # Direct copy -- no diversification
                    target_state_dict[target_name] = (
                        source_state_dict[source_name].clone()
                    )

    return target_state_dict
