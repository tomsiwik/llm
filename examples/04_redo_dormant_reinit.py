"""04 -- ReDo: Dormant Neuron Detection and Reinitialization

Paper: ReDo: Recycling Dormant Neurons for Improved Generalization
       (Klein, Srinivas & Bojanowski, 2024)
URL:   https://arxiv.org/abs/2302.12902
Repo:  timoklein/redo

Extracted from: src/redo.py

ReDo periodically scans a network for dormant neurons -- neurons whose
activations have collapsed to near-zero -- and reinitialises their weights.
This is the closest published analogue to TRIBE's `recycle()` operation,
but operates at the *neuron* level rather than the *expert* level.

This file extracts:
  1. _get_redo_masks        -- dormant neuron detection via activation norms
  2. _kaiming_uniform_reinit -- partial Kaiming re-initialization
  3. _reset_dormant_neurons -- ingoing/outgoing weight reset strategy
  4. _reset_adam_moments    -- optimizer state cleanup after reinit
  5. run_redo               -- the top-level orchestrator
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# ---------------------------------------------------------------------------
# 1. Dormant neuron detection
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _get_redo_masks(activations: dict, tau: float) -> list:
    """Identify dormant neurons in each layer based on activation magnitude.

    A neuron is dormant if its normalised mean activation falls below
    threshold tau.  The normalisation divides by the layer mean, making
    tau independent of layer width.

    Detection formula (from the paper, Section 3):
        score_i = mean(|activation_i|)           across the batch
        normalised_score_i = score_i / mean(score)  across neurons in the layer
        dormant if normalised_score_i <= tau

    # TRIBE NOTE: Direct analogue to our `health_check()` in core.py, but
    # the detection signal differs fundamentally:
    #
    #   ReDo:  measures *activation magnitude* (output of the neuron)
    #   TRIBE: measures *gradient norm / param_count* (learning signal)
    #
    # Both detect "this unit is not contributing", but from different angles:
    # - Zero activation means the neuron is functionally dead (output=0)
    # - Zero gradient means the neuron is not being asked to change
    #   (could still be active but already converged)
    #
    # ReDo's threshold tau is a universal constant (paper uses 0.1).
    # Our threshold is calibrated per-architecture via normalisation by
    # parameter count, which lets the same threshold work across MLP and CNN.

    Args:
        activations: dict mapping layer_name -> activation tensor
            Collected via forward hooks during a diagnostic batch.
        tau: dormancy threshold.  0.0 means only truly dead neurons
            (numerically zero).  0.1 (paper default) catches near-dormant.

    Returns:
        List of boolean masks, one per layer (excluding the final output layer).
        True = dormant, False = active.
    """
    masks = []

    # Skip the last layer (output/Q-values) -- never reinitialise it
    for name, activation in list(activations.items())[:-1]:
        # Mean absolute activation, reduced across batch (and spatial dims for conv)
        if activation.ndim == 4:
            # Conv layer: [batch, channels, H, W] -> [channels]
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer: [batch, neurons] -> [neurons]
            score = activation.abs().mean(dim=0)

        # Normalise by layer mean to make tau architecture-independent
        # TRIBE NOTE: This normalisation is analogous to our division by
        # param_count in health_check.  Both serve the same purpose: making
        # the threshold transferable across layers of different sizes.
        normalised_score = score / (score.mean() + 1e-9)

        # Build binary mask
        layer_mask = torch.zeros_like(normalised_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalised_score <= tau] = True
        else:
            # tau=0: only catch numerically-zero neurons
            layer_mask[torch.isclose(normalised_score, torch.zeros_like(normalised_score))] = True

        masks.append(layer_mask)

    return masks


# ---------------------------------------------------------------------------
# 2. Partial Kaiming re-initialization
# ---------------------------------------------------------------------------

@torch.no_grad()
def _kaiming_uniform_reinit(layer: nn.Module, mask: torch.Tensor) -> None:
    """Re-initialise only the dormant neurons' weights using Kaiming uniform.

    Only the rows of layer.weight indexed by `mask` are overwritten.
    This preserves the non-dormant neurons' learned representations.

    # TRIBE NOTE: Our `recycle()` reinitialises the *entire* expert, not
    # individual neurons within it.  The granularity difference is significant:
    #
    #   ReDo:  fine-grained, reinits specific neurons (rows of weight matrix)
    #   TRIBE: coarse-grained, reinits entire expert (all parameters)
    #
    # ReDo preserves more knowledge but is more complex (must handle
    # ingoing + outgoing weights + optimizer state).  TRIBE's expert-level
    # recycle is simpler and appropriate for our small expert networks (~1K
    # params each).  For large experts, neuron-level recycling would preserve
    # more useful structure.

    Args:
        layer: the nn.Linear or nn.Conv2d whose dormant neurons to reinit.
        mask:  boolean tensor of shape [out_features].  True = reinit this neuron.
    """
    fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
    gain = nn.init.calculate_gain(nonlinearity="relu", param=math.sqrt(5))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std

    # Reinit only the masked rows
    layer.weight.data[mask, ...] = torch.empty_like(
        layer.weight.data[mask, ...]
    ).uniform_(-bound, bound)

    # Reinit bias for dormant neurons
    if layer.bias is not None:
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        layer.bias.data[mask, ...] = torch.empty_like(
            layer.bias.data[mask, ...]
        ).uniform_(-bound, bound)


# ---------------------------------------------------------------------------
# 3. Reset dormant neurons (ingoing + outgoing weights)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _reset_dormant_neurons(
    model: nn.Module,
    redo_masks: list,
    use_lecun_init: bool = False,
) -> nn.Module:
    """Re-initialise dormant neurons in a sequential network.

    Two-step reset per dormant neuron:
      1. **Ingoing weights**: reinitialised from the init distribution
         (Kaiming uniform or Lecun normal).
      2. **Outgoing weights**: set to zero, so the reinitialised neuron
         does not immediately disrupt the network's existing output.

    # TRIBE NOTE: The outgoing-weights-to-zero trick is elegant.  It means
    # the reborn neuron starts with *zero contribution* to the next layer,
    # then gradually grows in influence as it trains.  This is smoother
    # than our expert-level recycle, where the new expert immediately
    # participates in routing.
    #
    # We could adopt this idea: after recycle(), scale the new expert's
    # output by a small factor (or zero) and gradually ramp it up.  This
    # would reduce the transient disruption from expert replacement.

    Args:
        model:      the network containing layers to reset.
        redo_masks: list of boolean masks from _get_redo_masks.
        use_lecun_init: if True, use Lecun normal instead of Kaiming uniform.

    Returns:
        The model with dormant neurons reinitialised (modified in-place).
    """
    layers = [(name, layer) for name, layer in list(model.named_modules())[1:]]
    assert len(redo_masks) == len(layers) - 1

    for i in range(len(layers[:-1])):
        mask = redo_masks[i]
        layer = layers[i][1]
        next_layer = layers[i + 1][1]

        # Skip if no dormant neurons in this layer
        if torch.all(~mask):
            continue

        # Step 1: Reinit ingoing weights from init distribution
        if use_lecun_init:
            _lecun_normal_reinit(layer, mask)
        else:
            _kaiming_uniform_reinit(layer, mask)

        # Step 2: Zero out outgoing weights
        # TRIBE NOTE: This is the critical detail.  Setting outgoing weights
        # to zero means the network output is *unchanged* immediately after
        # reinit.  The dormant neuron has been "reborn" with fresh ingoing
        # weights but zero outgoing influence.  During subsequent training,
        # it gradually learns to contribute useful features.
        #
        # In TRIBE's recycle(), we reinit all weights and the expert
        # immediately participates.  Adding an outgoing-zero phase could
        # reduce the disruption of expert replacement.
        if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
            # Conv -> Linear transition: expand mask to account for
            # spatial flattening (each filter maps to multiple linear inputs)
            num_repetition = next_layer.weight.data.shape[1] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repetition)
            next_layer.weight.data[:, linear_mask] = 0.0
        else:
            next_layer.weight.data[:, mask, ...] = 0.0

    return model


# ---------------------------------------------------------------------------
# 4. Reset optimizer moments for reinitialised neurons
# ---------------------------------------------------------------------------

@torch.no_grad()
def _reset_adam_moments(optimizer: optim.Adam, reset_masks: list) -> optim.Adam:
    """Zero out Adam's running moment estimates for reinitialised neurons.

    Adam maintains per-parameter exponential moving averages:
      exp_avg    (first moment, ~gradient mean)
      exp_avg_sq (second moment, ~gradient variance)
      step       (timestep counter for bias correction)

    After reinit, these moments are stale (they reflect the *old* dead neuron's
    history).  Resetting them lets the optimizer treat the reborn neuron as
    if it were freshly initialised.

    # TRIBE NOTE: This is a detail we currently skip in TRIBE's recycle().
    # When we reinit an expert, the optimizer still holds old moments for
    # those parameters.  For SGD this doesn't matter (no moments), but for
    # Adam/AdamW it means the first few updates after recycle use stale
    # variance estimates, which can cause either too-large or too-small
    # steps.  Adding moment resets (as ReDo does) would make expert
    # recycling cleaner.
    #
    # Critically, ReDo also resets the *step counter* to zero.  This
    # reactivates Adam's bias correction (warmup), which prevents the
    # initial gradient estimates from being scaled up too aggressively.

    Args:
        optimizer: the Adam optimizer.
        reset_masks: list of boolean masks, one per layer.

    Returns:
        The optimizer with moments zeroed for reinitialised neurons.
    """
    state = optimizer.state_dict()["state"]

    for i, mask in enumerate(reset_masks):
        # Weight moments (even indices in state dict)
        state[i * 2]["exp_avg"][mask, ...] = 0.0
        state[i * 2]["exp_avg_sq"][mask, ...] = 0.0
        # TRIBE NOTE: Step counter reset is key.  Without it, Adam's bias
        # correction denominator (1 - beta^t) is already close to 1 for
        # large t, meaning the fresh gradient estimates get no warmup.
        state[i * 2]["step"].zero_()

        # Bias moments (odd indices)
        state[i * 2 + 1]["exp_avg"][mask] = 0.0
        state[i * 2 + 1]["exp_avg_sq"][mask] = 0.0
        state[i * 2 + 1]["step"].zero_()

        # Output weight moments (the outgoing connections we zeroed)
        if (
            len(state[i * 2]["exp_avg"].shape) == 4
            and len(state[i * 2 + 2]["exp_avg"].shape) == 2
        ):
            # Conv -> Linear transition
            num_repetition = state[i * 2 + 2]["exp_avg"].shape[1] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repetition)
            state[i * 2 + 2]["exp_avg"][:, linear_mask] = 0.0
            state[i * 2 + 2]["exp_avg_sq"][:, linear_mask] = 0.0
            state[i * 2 + 2]["step"].zero_()
        else:
            state[i * 2 + 2]["exp_avg"][:, mask, ...] = 0.0
            state[i * 2 + 2]["exp_avg_sq"][:, mask, ...] = 0.0
            state[i * 2 + 2]["step"].zero_()

    return optimizer


# ---------------------------------------------------------------------------
# 5. Top-level orchestrator
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_redo(
    obs: torch.Tensor,
    model: nn.Module,
    optimizer: optim.Adam,
    tau: float = 0.1,
    re_initialize: bool = True,
    use_lecun_init: bool = False,
) -> dict:
    """Run a full ReDo cycle: detect dormant neurons and optionally reinit them.

    The full algorithm (pseudocode from the paper):
        1. Forward pass a diagnostic batch to collect activations
        2. Compute normalised activation scores per neuron
        3. Mark neurons below threshold tau as dormant
        4. Reinit dormant neurons (ingoing from init dist, outgoing to zero)
        5. Reset optimizer moments for affected parameters

    # TRIBE NOTE: Compare with our lifecycle flow:
    #
    #   TRIBE                          ReDo
    #   -----                          ----
    #   health_check() per expert      forward hook on all layers
    #   grad_norm / param_count        mean |activation| / layer_mean
    #   threshold comparison           tau comparison
    #   recycle(expert)                reinit(neurons) + zero outgoing
    #   (no optimizer reset)           reset Adam moments + step counter
    #
    # Key differences:
    #   - Granularity:  TRIBE recycles whole experts; ReDo recycles neurons
    #   - Signal:       TRIBE uses gradient norm; ReDo uses activation norm
    #   - Frequency:    TRIBE checks every step; ReDo checks periodically
    #   - Optimizer:    TRIBE doesn't reset moments; ReDo does (important!)
    #   - Init:         TRIBE random init; ReDo Kaiming + outgoing=0

    Args:
        obs:           diagnostic batch to measure activations on.
        model:         the network to check.
        optimizer:     Adam optimizer (moments will be reset if reinitialising).
        tau:           dormancy threshold (paper default: 0.1).
        re_initialize: if True, actually reinit dormant neurons.
                       If False, just measure and report.
        use_lecun_init: use Lecun normal instead of Kaiming uniform.

    Returns:
        Dict with keys: model, optimizer, zero_fraction, zero_count,
        dormant_fraction, dormant_count.
    """
    activations = {}

    # --- Register forward hooks to capture activations ---
    # TRIBE NOTE: We measure health via grad_norm, which requires a backward
    # pass.  ReDo only needs a forward pass (cheaper), but the signal is
    # different: activation magnitude vs gradient signal.
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = _make_activation_hook(name, activations)
            handles.append(module.register_forward_hook(hook))

    # Forward pass to collect activations
    _ = model(obs)

    # --- Detect dormant neurons ---
    # Masks for tau=0 (truly dead) -- for logging only
    zero_masks = _get_redo_masks(activations, 0.0)
    total_neurons = sum(torch.numel(m) for m in zero_masks)
    zero_count = sum(torch.sum(m) for m in zero_masks)
    zero_fraction = (zero_count / total_neurons) * 100

    # Masks for actual tau threshold -- used for reinit
    masks = _get_redo_masks(activations, tau)
    dormant_count = sum(torch.sum(m) for m in masks)
    dormant_fraction = (dormant_count / total_neurons) * 100

    # --- Reinitialise ---
    if re_initialize:
        model = _reset_dormant_neurons(model, masks, use_lecun_init)
        optimizer = _reset_adam_moments(optimizer, masks)

    # Clean up hooks
    for handle in handles:
        handle.remove()

    return {
        "model": model,
        "optimizer": optimizer,
        "zero_fraction": zero_fraction,
        "zero_count": zero_count,
        "dormant_fraction": dormant_fraction,
        "dormant_count": dormant_count,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activation_hook(name: str, activations: dict):
    """Create a forward hook that stores ReLU-activated output."""
    def hook(layer, input, output):
        activations[name] = F.relu(output)
    return hook


@torch.no_grad()
def _lecun_normal_reinit(layer: nn.Module, mask: torch.Tensor) -> None:
    """Partial Lecun normal reinit (alternative to Kaiming uniform)."""
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    layer.weight[mask] = nn.init._no_grad_trunc_normal_(
        layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0
    )
    layer.weight[mask] *= stddev
    if layer.bias is not None:
        layer.bias.data[mask] = 0.0
