"""Standard LoRA (Low-Rank Adaptation) for CL baselines.

True A@B LoRA (not self-routing atoms). Provides a fair baseline for
comparing against PEER lifecycle methods at matched parameter budgets.

Usage:
    from tribe.lora_standard import StandardLoRALinear, collect_standard_lora_layers

    layer = StandardLoRALinear(d_in=576, d_out=576, rank=16, scale=16.0)
    y = layer(x)  # base_linear(x) + (scale/rank) * (x @ A) @ B
"""

import math
import mlx.core as mx
import mlx.nn as nn


class StandardLoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with standard LoRA adaptation.

    Forward: base_linear(x) + (scale/rank) * (x @ lora_A) @ lora_B

    lora_A: (d_in, rank)  — init normal / sqrt(rank)
    lora_B: (rank, d_out) — init zeros (standard: start at identity)
    """

    def __init__(self, d_in, d_out, rank=16, scale=16.0,
                 base_weight=None, base_bias=None):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.scale = scale

        # Frozen base linear
        if base_weight is not None:
            self.weight = base_weight
        if base_bias is not None:
            self.bias = base_bias
        self._has_base = base_weight is not None
        self._has_bias = base_bias is not None

        # LoRA parameters
        self.lora_A = mx.random.normal((d_in, rank)) * (1.0 / math.sqrt(rank))
        self.lora_B = mx.zeros((rank, d_out))

    def __call__(self, x):
        # Base linear (frozen)
        if self._has_base:
            if self._has_bias:
                x_flat = mx.reshape(x, (-1, self.d_in))
                out = mx.addmm(self.bias, x_flat, self.weight.T)
                out = mx.reshape(out, (*x.shape[:-1], self.d_out))
            else:
                out = x @ self.weight.T
        else:
            out = mx.zeros((*x.shape[:-1], self.d_out))

        # LoRA delta: (scale/rank) * (x @ A) @ B
        lora_out = (x @ self.lora_A) @ self.lora_B
        return out + (self.scale / self.rank) * lora_out

    def __repr__(self):
        return (f"StandardLoRALinear(d_in={self.d_in}, d_out={self.d_out}, "
                f"rank={self.rank}, scale={self.scale})")


# ── Utilities ──────────────────────────────────────────────────

def collect_standard_lora_layers(model):
    """Find all StandardLoRALinear layers in a model."""
    results = []

    def _search(module, prefix=""):
        if isinstance(module, StandardLoRALinear):
            results.append((prefix, module))
            return
        if isinstance(module, nn.Module):
            children = module.children()
            for name, child in children.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Module):
                    _search(child, full_name)
                elif isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        _search(item, f"{full_name}.{i}")
                elif isinstance(child, dict):
                    for k, v in child.items():
                        _search(v, f"{full_name}.{k}")

    _search(model)
    return results


def total_standard_lora_params(model):
    """Count total LoRA parameters (A + B) across all patched layers."""
    total = 0
    for _, layer in collect_standard_lora_layers(model):
        total += layer.d_in * layer.rank   # lora_A
        total += layer.rank * layer.d_out  # lora_B
    return total
