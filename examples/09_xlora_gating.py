"""09 -- X-LoRA: Hidden-State-Dependent Gating over LoRA Adapters

Paper: X-LoRA: Mixture of Low-Rank Adapter Experts (Buehler & Gelbukh, 2024)
URL:   https://arxiv.org/abs/2402.07148
Repo:  EricLBuehler/xlora

X-LoRA routes over multiple frozen LoRA adapters using a learned classifier
that maps hidden states to per-layer, per-adapter scaling coefficients. The
key insight: a single forward pass through the base model (with adapters
disabled) produces hidden states that a small MLP uses to predict how much
each LoRA expert should contribute -- independently at every layer.

Extracted from: src/xlora/xlora_classifier.py, src/xlora/xlora_insertion.py
"""

import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Temperature-scaled softmax for controlling sharpness of adapter selection
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is their "soft routing" -- temperature < 1 sharpens toward
# hard selection (like our top-k routing), temperature > 1 smooths toward
# uniform mixing. Our system uses argmin(loss) for hard routing; X-LoRA
# keeps it differentiable via this softmax.
class TemperatureScaledSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return self.softmax(scaled_logits)


# ---------------------------------------------------------------------------
# The xLoRA Classifier: predicts per-layer adapter scalings from hidden states
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is the core routing mechanism. It takes the LAST hidden
# state from a "scaling pass" (forward with adapters disabled) and predicts
# [batch, seq_len, n_layers, n_adapters] scaling coefficients.
#
# Key design choices:
#   1. Scalings are INPUT-DEPENDENT -- each token gets different adapter mix
#   2. Scalings are LAYER-WISE -- each transformer layer can use different mix
#   3. The classifier is the ONLY trained component; LoRA weights are frozen
#
# For TRIBE: This shows the path to "LoRA as knowledge entity". Each LoRA
# adapter is an expert. The classifier is the router. Our lifecycle could
# manage adapter birth (init new LoRA), death (prune unused), and bonding
# (merge two LoRAs with similar scalings).
class xLoRAClassifier(nn.Module):
    """Predicts adapter mixing coefficients from model hidden states."""

    def __init__(
        self,
        config_hidden_size: int,
        config_device: str,
        n_classes: int,       # number of LoRA adapters
        n_layers: int,        # number of swapped LoRA layers
        xlora_depth: int = 2,
        xlora_size: int = 2048,
        softmax_temperature: float = 1.0,
        enable_relu_and_dropout: bool = False,
        xlora_dropout_p: float = 0.2,
        layerwise_scalings: bool = True,
        enable_softmax: bool = True,
        use_bias: bool = True,
        top_k_lora: Optional[int] = None,
        enable_softmax_topk: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.softmax = TemperatureScaledSoftmax(temperature=softmax_temperature)
        self.enable_softmax = enable_softmax
        self.layerwise_scalings = layerwise_scalings
        self.top_k_lora = top_k_lora
        self.enable_softmax_topk = enable_softmax_topk

        self.log_scalings: List[torch.Tensor] = []
        self.scalings_logging = False

        # ---------------------------------------------------------------
        # Build the MLP classifier
        # ---------------------------------------------------------------
        # TRIBE NOTE: The classifier architecture is configurable depth.
        # Depth=1: single linear layer (hidden_size -> n_classes * n_layers)
        # Depth=2: hidden_size -> xlora_size -> n_classes * n_layers
        # This is surprisingly small -- just a few linear layers to route
        # over potentially dozens of LoRA adapters across all layers.
        self.inner: nn.ModuleList = nn.ModuleList([])
        output_dim = n_classes * n_layers if layerwise_scalings else n_classes

        if xlora_depth == 1:
            self.last = nn.Linear(config_hidden_size, output_dim, bias=use_bias)
        elif xlora_depth == 2:
            self.inner.append(nn.Linear(config_hidden_size, xlora_size, bias=use_bias))
            if enable_relu_and_dropout:
                self.inner.append(nn.ReLU())
                self.inner.append(nn.Dropout(p=xlora_dropout_p))
            self.last = nn.Linear(xlora_size, output_dim, bias=use_bias)
        else:
            self.inner.append(nn.Linear(config_hidden_size, xlora_size, bias=use_bias))
            if enable_relu_and_dropout:
                self.inner.append(nn.ReLU())
                self.inner.append(nn.Dropout(p=xlora_dropout_p))
            for _ in range(xlora_depth - 2):
                self.inner.append(nn.Linear(xlora_size, xlora_size, bias=use_bias))
                if enable_relu_and_dropout:
                    self.inner.append(nn.ReLU())
                    self.inner.append(nn.Dropout(p=xlora_dropout_p))
            self.last = nn.Linear(xlora_size, output_dim, bias=use_bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: Last hidden state from scaling pass [batch, seq_len, hidden_size]

        Returns:
            scalings: [batch_size, seq_len, n_layers, n_classes]
        """
        batch_size, seq_len, _ = hidden_state.shape

        # Pass through inner MLP layers
        for layer in self.inner:
            hidden_state = layer(hidden_state)
        logits = self.last(hidden_state)

        # TRIBE NOTE: If not layerwise, broadcast single prediction to all layers.
        # This is the simpler variant -- one routing decision shared across layers.
        # Layerwise scalings let each transformer layer pick different adapter mix.
        if not self.layerwise_scalings:
            logits = logits.unsqueeze(2)
            logits = logits.expand(-1, -1, self.n_layers, -1)

        scalings = logits.reshape(batch_size, seq_len, self.n_layers, self.n_classes)

        if self.enable_softmax:
            scalings = self.softmax(scalings)

        return scalings


# ---------------------------------------------------------------------------
# xLoRA Layer: applies per-adapter scalings during the actual forward pass
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is where the scaling coefficients actually modulate the
# LoRA outputs. For each adapter, the INPUT is scaled by the routing weight
# before passing through the LoRA matrices. This means:
#   output = base(x) + sum_i( scaling_i * lora_B_i(lora_A_i(x)) )
#
# Compare with MoLE (file 10): MoLE scales the OUTPUT of each LoRA, while
# X-LoRA scales the INPUT. Both achieve weighted mixture but X-LoRA's
# approach means the scaling affects what information flows into each adapter.
class xLoRALayer:
    """Wraps a LoRA layer to apply X-LoRA input-dependent scaling."""

    def __init__(self, layer_number: int, top_k_lora: Optional[int] = None,
                 enable_softmax_topk: bool = False,
                 global_scaling_weight: float = 1.0):
        self.layer_number = layer_number
        self.top_k_lora = top_k_lora
        self.enable_softmax_topk = enable_softmax_topk
        self.global_scaling_weight = global_scaling_weight

    @staticmethod
    def apply_scalings_to_x(
        x: torch.Tensor,
        scalings_layer: torch.Tensor,
        adapter: int,
    ) -> torch.Tensor:
        """Scale input x by the routing weight for a specific adapter.

        Args:
            x: input tensor [batch_size, seq_len, hidden_dim]
            scalings_layer: [batch_size, seq_len, n_classes]
            adapter: index of the current adapter

        Returns:
            x * scaling for this adapter [batch_size, seq_len, hidden_dim]
        """
        # TRIBE NOTE: This is the key operation. Each adapter gets a
        # per-token scaling factor. The unsqueeze(-1) broadcasts across
        # the hidden dimension so every feature is scaled equally.
        scalings = scalings_layer[:, :, adapter].unsqueeze(-1)
        return x * scalings

    def get_maybe_topk_scalings(
        self, xlora_scalings: torch.Tensor
    ) -> torch.Tensor:
        """Extract this layer's scalings, optionally with top-k sparsity.

        Args:
            xlora_scalings: [batch_size, seq_len, n_layers, n_classes]

        Returns:
            scalings for this layer: [batch_size, seq_len, n_classes]
        """
        scalings = xlora_scalings[:, :, self.layer_number, :]

        # TRIBE NOTE: Optional top-k sparsity -- zero out all but the
        # top k adapters. This converts soft routing to sparse routing,
        # similar to how Switch Transformer uses top-1 and our system
        # uses top-k routing. Sparsity helps both compute and prevents
        # adapter interference.
        if self.top_k_lora is not None:
            _, topk_indices = torch.topk(scalings, k=self.top_k_lora, dim=-1)
            mask = torch.zeros_like(scalings, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)
            scalings = scalings * mask.to(scalings.dtype)

        # Optional re-normalization after top-k masking
        if self.enable_softmax_topk:
            nonzero_mask = scalings != 0
            softmax_res_nonzero = torch.softmax(scalings[nonzero_mask], dim=-1)
            scalings[nonzero_mask] = softmax_res_nonzero

        return scalings


def xlora_linear_forward(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: Optional[torch.Tensor],
    lora_As: dict,
    lora_Bs: dict,
    lora_dropouts: dict,
    lora_scalings_dict: dict,
    active_adapters: list,
    xlora_scalings: torch.Tensor,
    layer_number: int,
    global_scaling_weight: float = 1.0,
    top_k_lora: Optional[int] = None,
    fan_in_fan_out: bool = False,
) -> torch.Tensor:
    """Core X-LoRA forward: base output + scaled sum of LoRA adapter outputs.

    TRIBE NOTE: The full forward computes:
        result = W @ x + bias
        for each adapter i:
            x_scaled = x * routing_weight[i]   # <-- input scaling
            result += B_i(A_i(dropout(x_scaled))) * lora_scaling * global_weight

    This is functionally a Mixture-of-Experts where each expert is a LoRA
    adapter and the gating is done by the xLoRAClassifier based on hidden
    states from a prior forward pass.
    """
    previous_dtype = x.dtype

    # TRIBE NOTE: We separate the "get scalings" logic here for clarity.
    # In the original code this is done via the xLoRALayer wrapper.
    xlora_layer = xLoRALayer(layer_number, top_k_lora)
    scalings_for_layer = xlora_layer.get_maybe_topk_scalings(xlora_scalings)

    # Base model forward (no LoRA)
    if fan_in_fan_out:
        result = torch.nn.functional.linear(x, base_weight.T, bias=base_bias)
    else:
        result = torch.nn.functional.linear(x, base_weight, bias=base_bias)

    # Sum contributions from each LoRA adapter, weighted by routing
    for adapter_n, active_adapter in enumerate(active_adapters):
        if active_adapter not in lora_As:
            continue
        lora_A = lora_As[active_adapter]
        lora_B = lora_Bs[active_adapter]
        dropout = lora_dropouts[active_adapter]
        scaling = lora_scalings_dict[active_adapter]

        x_cast = x.to(lora_A.weight.dtype)

        # TRIBE NOTE: Input scaling -- the adapter sees a version of x
        # that's amplified or attenuated based on the router's decision.
        # High scaling = this adapter is relevant for this input.
        # Low scaling = this adapter should contribute minimally.
        x_scaled = xLoRALayer.apply_scalings_to_x(
            x_cast, scalings_for_layer, adapter_n
        )
        result += lora_B(lora_A(dropout(x_scaled))) * scaling * global_scaling_weight

    return result.to(previous_dtype)
