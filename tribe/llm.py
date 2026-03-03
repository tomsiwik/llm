"""LLM backbone utilities for continual learning with self-routing LoRA atoms.

Load a frozen LLM, patch target layers with SelfRoutingLoRALinear atoms,
and evaluate via perplexity.

Usage:
    from tribe.llm import load_backbone, patch_with_atoms, freeze_base, compute_perplexity

    model, tokenizer = load_backbone("HuggingFaceTB/SmolLM-135M")
    patch_with_atoms(model, n_atoms=32, temperature=0.5)
    freeze_base(model)
    # Now only atom_A and atom_B params are trainable
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tribe.lora_atom import SelfRoutingLoRALinear, patch_linear_with_atoms, collect_atom_layers
from tribe.lora_standard import StandardLoRALinear, collect_standard_lora_layers
from tribe.peer_atom import PEERLifecycleLinear, ParallelPEERLayer, collect_peer_layers
from tribe.lora_library import SelfRoutingLoRALibrary, collect_library_layers


def load_backbone(model_name="HuggingFaceTB/SmolLM-135M"):
    """Load a pretrained LLM via mlx-lm.

    Returns:
        (model, tokenizer) tuple.
    """
    from mlx_lm import load
    model, tokenizer = load(model_name)
    return model, tokenizer


def patch_with_atoms(model, n_atoms=32, top_k=0, temperature=1.0, scale=1.0,
                     targets=("q_proj", "v_proj"), max_ghosts=0):
    """Patch target Linear layers in all transformer blocks with SelfRoutingLoRALinear.

    Args:
        model: mlx-lm model (e.g., LlamaForCausalLM).
        n_atoms: number of rank-1 LoRA atoms per layer.
        top_k: hard top-k selection (0 = soft routing).
        temperature: softmax temperature for routing.
        scale: LoRA output scaling.
        targets: which attention projections to patch (default: Q and V).
        max_ghosts: maximum ghost slots per layer (0 = no ghosts).

    Returns:
        list of (layer_name, SelfRoutingLoRALinear) for all patched layers.
    """
    patched = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for target in targets:
            if hasattr(attn, target):
                lora_layer = patch_linear_with_atoms(
                    attn, target,
                    n_atoms=n_atoms, top_k=top_k,
                    temperature=temperature, scale=scale,
                    max_ghosts=max_ghosts,
                )
                patched.append((f"layers.{i}.self_attn.{target}", lora_layer))
    return patched


def patch_with_peer(model, n_experts=1024, n_active=32, pk=8, d_key=None,
                    scale=1.0, targets=("q_proj", "v_proj")):
    """Patch target Linear layers with PEERLifecycleLinear.

    Args:
        model: mlx-lm model (e.g., LlamaForCausalLM).
        n_experts: total experts per layer (must be perfect square).
        n_active: experts selected per token.
        pk: top-k per sub-key half (Cartesian product gives pk² candidates).
        d_key: product key dimension (default: d_in // 2).
        scale: output scaling.
        targets: which attention projections to patch.

    Returns:
        list of (layer_name, PEERLifecycleLinear) for all patched layers.
    """
    patched = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for target in targets:
            linear = getattr(attn, target, None)
            if linear is None:
                continue
            d_out, d_in = linear.weight.shape
            bias = getattr(linear, 'bias', None)

            peer_layer = PEERLifecycleLinear(
                d_in=d_in, d_out=d_out,
                n_experts=n_experts, n_active=n_active, pk=pk,
                d_key=d_key, base_weight=linear.weight,
                base_bias=bias, scale=scale,
            )
            setattr(attn, target, peer_layer)
            patched.append((f"layers.{i}.self_attn.{target}", peer_layer))
    return patched


def patch_with_parallel_peer(model, n_branches=2, n_experts=529,
                              n_active=17, pk=8, d_key=None, scale=1.0,
                              use_streams=False, targets=("q_proj", "v_proj")):
    """Patch target Linear layers with ParallelPEERLayer.

    Args:
        model: mlx-lm model (e.g., LlamaForCausalLM).
        n_branches: number of competing PEER branches per layer.
        n_experts: experts per branch (must be perfect square).
        n_active: experts selected per token per branch.
        pk: top-k per sub-key half.
        d_key: product key dimension (default: d_in // 2).
        scale: output scaling.
        use_streams: enable mx.stream parallelism for branches.
        targets: which attention projections to patch.

    Returns:
        list of (layer_name, ParallelPEERLayer) for all patched layers.
    """
    patched = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for target in targets:
            linear = getattr(attn, target, None)
            if linear is None:
                continue
            d_out, d_in = linear.weight.shape
            bias = getattr(linear, 'bias', None)

            parallel_layer = ParallelPEERLayer(
                d_in=d_in, d_out=d_out,
                n_branches=n_branches, n_experts=n_experts,
                n_active=n_active, pk=pk, d_key=d_key,
                base_weight=linear.weight, base_bias=bias,
                scale=scale, use_streams=use_streams,
            )
            setattr(attn, target, parallel_layer)
            patched.append((f"layers.{i}.self_attn.{target}", parallel_layer))
    return patched


def patch_with_standard_lora(model, rank=16, scale=16.0,
                              targets=("q_proj", "v_proj")):
    """Patch target Linear layers with StandardLoRALinear.

    Args:
        model: mlx-lm model (e.g., LlamaForCausalLM).
        rank: LoRA rank.
        scale: LoRA output scaling.
        targets: which attention projections to patch.

    Returns:
        list of (layer_name, StandardLoRALinear) for all patched layers.
    """
    patched = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for target in targets:
            linear = getattr(attn, target, None)
            if linear is None:
                continue
            d_out, d_in = linear.weight.shape
            bias = getattr(linear, 'bias', None)

            lora_layer = StandardLoRALinear(
                d_in=d_in, d_out=d_out, rank=rank, scale=scale,
                base_weight=linear.weight, base_bias=bias,
            )
            setattr(attn, target, lora_layer)
            patched.append((f"layers.{i}.self_attn.{target}", lora_layer))
    return patched


def patch_with_library(model, lora_weights_list, labels=None, top_k=1, scale=1.0,
                       targets=("q_proj", "v_proj")):
    """Patch target Linear layers with SelfRoutingLoRALibrary containing multiple experts.

    Args:
        model: mlx-lm model (e.g., Qwen2.5-Coder).
        lora_weights_list: list of dicts, each mapping layer_path → (A, B).
            Layer paths follow the format from bench_stitch_toy.py:
            "model.layers.{i}.self_attn.{target}"
        labels: list of expert labels (e.g., ["python", "javascript"]).
        top_k: number of experts to activate per token.
        scale: LoRA output scaling (typically SCALE/RANK, e.g., 16/16=1.0).
        targets: which attention projections to patch.

    Returns:
        list of (layer_name, SelfRoutingLoRALibrary) for all patched layers.
    """
    if labels is None:
        labels = [f"expert_{i}" for i in range(len(lora_weights_list))]

    patched = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for target in targets:
            proj = getattr(attn, target, None)
            if proj is None:
                continue

            # Get base weight/bias from the linear layer
            base_weight = proj.weight
            base_bias = getattr(proj, 'bias', None)

            # Create library
            lib = SelfRoutingLoRALibrary(
                base_weight=base_weight,
                base_bias=base_bias,
                top_k=top_k,
                scale=scale,
            )

            # Register experts
            # Try both naming conventions
            name_v1 = f"model.layers.{i}.self_attn.{target}"
            name_v2 = f"layers.{i}.self_attn.{target}"
            for expert_idx, lora_w in enumerate(lora_weights_list):
                key = name_v1 if name_v1 in lora_w else name_v2
                if key in lora_w:
                    A, B = lora_w[key]
                    lib.register_expert(A, B, label=labels[expert_idx])

            setattr(attn, target, lib)
            patched.append((name_v2, lib))

    return patched


def freeze_base(model):
    """Freeze all model parameters, then unfreeze only adapter params.

    After this call, only atom/PEER parameters in patched layers
    are trainable. The entire LLM backbone is frozen.
    """
    model.freeze()
    # Unfreeze atom parameters in patched layers
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        for attr in ("q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"):
            proj = getattr(attn, attr, None)
            if proj is None:
                proj = getattr(layer.mlp, attr, None) if hasattr(layer, 'mlp') else None
            if isinstance(proj, SelfRoutingLoRALinear):
                keys = ["atom_A", "atom_B"]
                if proj._max_ghosts > 0:
                    keys.append("ghost_B")
                proj.unfreeze(keys=keys)
            elif isinstance(proj, StandardLoRALinear):
                proj.unfreeze(keys=["lora_A", "lora_B"])
            elif isinstance(proj, PEERLifecycleLinear):
                proj.unfreeze(keys=[
                    "weight_down", "weight_up", "sub_keys",
                    "query_proj", "gate_bias",
                ])
            elif isinstance(proj, ParallelPEERLayer):
                # Unfreeze gate params
                proj.unfreeze(keys=["gate_proj", "gate_bias_vec"])
                # Unfreeze all branch PEER params
                for branch in proj.branches:
                    branch.unfreeze(keys=[
                        "weight_down", "weight_up", "sub_keys",
                        "query_proj", "gate_bias",
                    ])
            elif isinstance(proj, SelfRoutingLoRALibrary):
                # Library experts (A, B) are frozen (pre-trained).
                # Unfreeze routing keys if they exist (for calibration).
                if proj._has_routing_keys:
                    keys = [f"routing_key_{i}" for i in range(proj.n_experts)]
                    proj.unfreeze(keys=keys)


def trainable_param_count(model):
    """Count trainable parameters (should be only atom params after freeze_base)."""
    total = 0
    for name, param in model.trainable_parameters().items() if hasattr(model.trainable_parameters, 'items') else []:
        total += param.size
    # MLX returns nested dict, so flatten
    def _count(params):
        n = 0
        if isinstance(params, dict):
            for v in params.values():
                n += _count(v)
        elif isinstance(params, list):
            for v in params:
                n += _count(v)
        elif isinstance(params, mx.array):
            n += params.size
        return n
    return _count(model.trainable_parameters())


def compute_perplexity(model, tokenizer, texts, max_length=256, batch_size=4):
    """Compute perplexity of model on a list of text strings.

    Args:
        model: patched LLM model.
        tokenizer: tokenizer from mlx-lm.
        texts: list of text strings.
        max_length: max tokens per sequence.
        batch_size: sequences per batch.

    Returns:
        float: perplexity (lower = better).
    """
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        all_tokens = []
        for text in batch_texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            all_tokens.append(tokens)

        if not all_tokens:
            continue

        # Pad to same length
        max_len = max(len(t) for t in all_tokens)
        if max_len < 2:
            continue

        padded = []
        masks = []
        for tokens in all_tokens:
            pad_len = max_len - len(tokens)
            padded.append(tokens + [0] * pad_len)
            masks.append([1.0] * len(tokens) + [0.0] * pad_len)

        input_ids = mx.array(padded)    # (B, L)
        mask = mx.array(masks)           # (B, L)

        # Forward pass
        logits = model(input_ids[:, :-1])  # (B, L-1, vocab)
        targets = input_ids[:, 1:]          # (B, L-1)
        target_mask = mask[:, 1:]           # (B, L-1)

        # Cross-entropy loss
        log_probs = nn.losses.cross_entropy(logits, targets, reduction='none')  # (B, L-1)
        masked_loss = log_probs * target_mask

        mx.eval(masked_loss, target_mask)
        total_loss += masked_loss.sum().item()
        total_tokens += target_mask.sum().item()

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))


def get_atom_stats(model):
    """Collect activation statistics from all patched layers.

    Returns:
        dict: layer_name → {
            'profile': (n_atoms,) activation profile,
            'effective_rank': float,
            'n_atoms': int,
        }
    """
    stats = {}
    for name, layer in collect_atom_layers(model):
        stats[name] = {
            'profile': layer.activation_profile(),
            'effective_rank': layer.effective_rank(),
            'n_atoms': layer.n_atoms,
        }
    return stats


def reset_all_stats(model):
    """Reset activation statistics on all patched layers."""
    for _, layer in collect_atom_layers(model):
        layer.reset_stats()


def update_all_stats(model, input_ids):
    """Update activation statistics on all patched layers for given input.

    This runs a forward pass and records which atoms fire.
    """
    # We need the hidden states at each layer to update stats.
    # For now, just run a forward pass — stats are updated inside __call__
    # if we modify the forward to track. Instead, we'll use a simpler approach:
    # extract hidden states at each attention layer and call update_stats.

    # Simple approach: just run forward, then manually compute stats
    # on a sample of the input
    pass  # Will be done in benchmark via explicit calls


def print_atom_summary(model):
    """Print a summary of atom parameters and statistics."""
    layers = collect_atom_layers(model)
    total_params = 0
    total_atoms = 0

    for name, layer in layers:
        n_params = layer.n_atoms * (layer.d_in + layer.d_out)
        total_params += n_params
        total_atoms += layer.n_atoms

    print(f"  Atom layers: {len(layers)}")
    print(f"  Total atoms: {total_atoms}")
    print(f"  Total atom params: {total_params:,}")

    # Activation stats if available
    has_stats = any(layer._forward_calls > 0 for _, layer in layers)
    if has_stats:
        eff_ranks = []
        for name, layer in layers:
            er = layer.effective_rank()
            eff_ranks.append(er)
        mean_er = np.mean(eff_ranks)
        print(f"  Mean effective rank: {mean_er:.1f} / {layers[0][1].n_atoms}")
