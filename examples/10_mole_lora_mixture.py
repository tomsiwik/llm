"""10 -- MoLE: Mixture of LoRA Experts with Learned Gating

Paper: Mixture of LoRA Experts (Wu et al., 2024)
URL:   https://arxiv.org/abs/2404.13628
Repo:  yushuiwx/Mixture-of-LoRA-Experts

MoLE composes multiple LoRA adapters via a learned gating network that
produces soft mixing coefficients. Unlike X-LoRA which scales the INPUT
to each adapter, MoLE scales the OUTPUT of each adapter and sums them.
The gate is a simple linear layer per transformer block that maps the
flattened hidden state to per-adapter weights.

Extracted from: tools/peft/tuners/lora.py (Linear class),
                tools/transformers_mole/.../modeling_clip.py (gate computation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Utility: entropy of a probability distribution (for load balancing)
# ---------------------------------------------------------------------------
# TRIBE NOTE: They track gating entropy to monitor expert utilization.
# Low entropy = one expert dominates (specialist behavior).
# High entropy = uniform usage (generalist behavior).
# Our system measures this via "redundancy" metric instead -- overlap
# between expert competence regions.
def entropy(probs):
    """Compute entropy of probability distribution for monitoring."""
    logits = torch.distributions.utils.probs_to_logits(probs)
    p_log_p = probs * logits
    return -p_log_p.sum(-1)


def normalized_columns_initializer(weights, std=1.0):
    """Initialize weights with normalized columns (small init for gate)."""
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


# ---------------------------------------------------------------------------
# Per-block gating network: computes adapter mixing coefficients
# ---------------------------------------------------------------------------
# TRIBE NOTE: The gate is remarkably simple -- a single linear layer that
# maps flattened hidden states to N_experts logits, then softmax. This is
# computed ONCE per block and shared across all LoRA-replaced layers in
# that block (q_proj, v_proj, etc. all get the same gate values).
#
# Compare with X-LoRA (file 09): X-LoRA uses a separate MLP classifier
# that produces per-layer scalings. MoLE uses per-block gates that are
# simpler but less flexible.
#
# For TRIBE: This could be our default "bond" operation for LoRA experts.
# When two adapters consistently get similar gate values, they're redundant
# and could be merged (bonded). When one adapter's gate is consistently
# near zero, it's a candidate for recycling.
class MoLEGate(nn.Module):
    """Per-block gating for Mixture of LoRA Experts.

    Takes flattened hidden states and produces soft mixing weights
    over N LoRA adapters via a single linear layer + softmax.
    """

    def __init__(self, seq_len: int, hidden_size: int, num_experts: int = 3):
        super().__init__()
        # TRIBE NOTE: Gate input is the FULL flattened sequence
        # (seq_len * hidden_size). This means the routing decision
        # considers the entire input, not just per-token features.
        # Our loss-based routing also considers the full input but
        # evaluates each expert's actual loss rather than predicting it.
        self.wg = nn.Linear(seq_len * hidden_size, num_experts, bias=False)
        self.wg.weight.data.copy_(
            normalized_columns_initializer(self.wg.weight.data, 0.01)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute gate values from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            gates: [batch, num_experts] -- soft mixing weights
        """
        B = hidden_states.shape[0]
        # Flatten and normalize, then project to num_experts logits
        logits = self.wg(F.normalize(hidden_states.reshape(B, -1), dim=-1))
        gates = F.softmax(logits, dim=1)
        return gates


# ---------------------------------------------------------------------------
# MoLE Linear Layer: LoRA mixture with output-space gating
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is the core MoLE operation. Key differences from X-LoRA:
#   1. X-LoRA scales INPUTS to each adapter; MoLE scales OUTPUTS
#   2. X-LoRA has per-token, per-layer scalings; MoLE has per-sample, per-block
#   3. MoLE gate is computed externally and passed in; X-LoRA computes internally
#
# The output-scaling approach is simpler and more like standard MoE:
#   result = base(x) + sum_i( gate_i * lora_i(x) )
#
# For TRIBE: This "weighted sum of adapter outputs" is essentially what
# our bonding operation does when merging experts -- but MoLE does it
# dynamically at inference time rather than permanently.
class MoLELinear(nn.Module):
    """Linear layer with Mixture of LoRA Experts.

    Multiple LoRA adapters (A_i, B_i) are applied in parallel, their
    outputs weighted by externally-computed gate values, and summed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 3,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        # Base (frozen) linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # TRIBE NOTE: Multiple LoRA adapter pairs -- this is the "expert pool".
        # Each expert is a low-rank factorization: delta_W = B_i @ A_i.
        # In our system, each expert is a full sub-network, but the principle
        # is the same: a collection of specialized modules that can be mixed.
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout_layers = nn.ModuleDict()
        self.scaling = {}
        self.expert_names: List[str] = []

        for i in range(num_experts):
            name = f"expert_{i}"
            self.expert_names.append(name)
            self.lora_A[name] = nn.Linear(in_features, r, bias=False)
            self.lora_B[name] = nn.Linear(r, out_features, bias=False)
            self.scaling[name] = lora_alpha / r
            if lora_dropout > 0.0:
                self.lora_dropout_layers[name] = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout_layers[name] = nn.Identity()
            # Standard LoRA init: A ~ Kaiming, B = 0
            nn.init.kaiming_uniform_(self.lora_A[name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name].weight)

        # Per-layer temperature (learned)
        self.temperatures = nn.Parameter(torch.tensor([1.0]))

        # Monitoring: track gate usage and entropy
        # TRIBE NOTE: These track expert utilization over time -- exactly
        # the kind of signal our lifecycle uses to detect stale or
        # redundant experts. When gates_record shows one expert is never
        # selected, that's a recycling candidate.
        self.gates_record = nn.Parameter(
            torch.zeros([num_experts]), requires_grad=False
        )
        self.entropy_record = nn.Parameter(
            torch.zeros([1]), requires_grad=False
        )

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """Forward with gated LoRA mixture.

        Args:
            x: input [batch, seq_len, in_features]
            gates: mixing weights [batch, num_experts] from MoLEGate

        Returns:
            output [batch, seq_len, out_features]
        """
        B, L, D = x.shape
        previous_dtype = x.dtype

        # Base linear (frozen weights)
        result = F.linear(x, self.weight, bias=self.bias)

        # Compute each LoRA expert's output
        # TRIBE NOTE: All experts process the SAME input (no input scaling).
        # The differentiation comes entirely from the learned A_i, B_i
        # matrices and the output gating. This is simpler than X-LoRA
        # but means each expert must learn to ignore irrelevant inputs
        # on its own rather than having the router pre-filter.
        lora_results = []
        for name in self.expert_names:
            x_cast = x.to(self.lora_A[name].weight.dtype)
            sub_result = (
                self.lora_B[name](
                    self.lora_A[name](self.lora_dropout_layers[name](x_cast))
                )
                * self.scaling[name]
            )
            sub_result = sub_result.to(previous_dtype).unsqueeze(1)
            lora_results.append(sub_result)

        # Stack: [batch, num_experts, seq_len, out_features]
        lora_results = torch.cat(lora_results, dim=1)

        # Track entropy for monitoring (detached -- no gradient)
        entropy_gating = entropy(probs=gates).mean().detach()
        self.gates_record += gates.clone().detach().mean(dim=0)
        self.entropy_record += entropy_gating

        # TRIBE NOTE: Core MoLE operation -- weight each expert's output
        # by its gate value. gates is [B, num_experts], unsqueezed to
        # [B, num_experts, 1, 1] for broadcasting against
        # [B, num_experts, L, D]. Then sum across expert dimension.
        #
        # This is equivalent to: result += sum_i(gate_i * lora_i(x))
        # which is the standard MoE combination formula.
        assert gates.shape[0] == B and gates.shape[1] == len(self.expert_names)
        lora_results = lora_results * gates.unsqueeze(-1).unsqueeze(-1)
        lora_results = lora_results.sum(dim=1)

        assert lora_results.shape == result.shape
        return lora_results + result


# ---------------------------------------------------------------------------
# Full encoder block with MoLE gating (based on CLIP encoder layer)
# ---------------------------------------------------------------------------
# TRIBE NOTE: This shows how the gate is computed at the block level and
# shared across all LoRA-augmented layers in that block. The gate sees
# the hidden state BEFORE the attention/MLP operations, so it's routing
# based on the current representation, not the output.
class MoLEEncoderBlock(nn.Module):
    """Transformer block with Mixture of LoRA Experts.

    Demonstrates how a single gate per block routes all LoRA layers.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        seq_len: int = 77,
        num_experts: int = 3,
        num_heads: int = 12,
        r: int = 8,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Gate: one per block, shared by all LoRA layers
        self.gate = MoLEGate(seq_len, hidden_size, num_experts)

        # LoRA-augmented attention projections
        self.q_proj = MoLELinear(hidden_size, hidden_size, num_experts, r=r)
        self.v_proj = MoLELinear(hidden_size, hidden_size, num_experts, r=r)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with shared gating across LoRA layers.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        # TRIBE NOTE: Gate computed ONCE from normalized hidden states,
        # then passed to both q_proj and v_proj. This means the routing
        # decision is the same for all projections within a block --
        # a simplifying assumption that works well in practice.
        gates = self.gate(hidden_states)

        # Both projections use the same gate values
        q = self.q_proj(hidden_states, gates)
        v = self.v_proj(hidden_states, gates)

        # (Simplified -- real impl has full attention)
        hidden_states = residual + q + v

        return hidden_states
