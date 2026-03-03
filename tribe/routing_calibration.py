"""Contrastive routing key calibration for LoRA expert libraries.

Train thin routing keys K_i per expert using InfoNCE loss on cached
hidden states. Decouples routing (K) from computation (A@B).

Usage:
    from tribe.routing_calibration import (
        extract_calibration_features, calibrate_routing_keys,
        evaluate_routing_accuracy,
    )

    features = extract_calibration_features(model, tokenizer, domain_texts)
    calibrate_routing_keys(patched_model, features, steps=50)
    acc = evaluate_routing_accuracy(patched_model, features)
"""

import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from tribe.lora_library import SelfRoutingLoRALibrary, collect_library_layers


def _parse_layer_idx(name):
    """Extract transformer layer index from a dotted name.

    Handles both "layers.5.self_attn.q_proj" and
    "model.layers.5.self_attn.q_proj" formats.
    """
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Cannot parse layer index from: {name}")


def extract_calibration_features(model, tokenizer, domain_texts, max_length=256):
    """Extract per-layer hidden states from base model for each domain.

    Runs one forward pass per sample, collecting the attention input
    (post-layernorm) at each transformer layer.

    Args:
        model: unpatched base model (for clean hidden states).
        tokenizer: tokenizer from mlx-lm.
        domain_texts: dict[str, list[str]] mapping domain label → text samples.
        max_length: max tokens per sample.

    Returns:
        dict[str, dict[int, mx.array]]: domain → layer_idx → (n_tokens, d_in).
    """
    features = {}

    for domain, texts in domain_texts.items():
        layer_tokens = {}  # layer_idx → list of (seq, d) arrays

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            if len(tokens) < 2:
                continue

            input_ids = mx.array([tokens])
            hs = _extract_attention_inputs(model, input_ids)

            for layer_idx, h in hs.items():
                h_flat = h.reshape(-1, h.shape[-1])  # (seq, d)
                mx.eval(h_flat)
                if layer_idx not in layer_tokens:
                    layer_tokens[layer_idx] = []
                layer_tokens[layer_idx].append(h_flat)

        # Concatenate all tokens per layer
        features[domain] = {}
        for layer_idx, arrays in layer_tokens.items():
            features[domain][layer_idx] = mx.concatenate(arrays, axis=0)
            mx.eval(features[domain][layer_idx])

    return features


def _extract_attention_inputs(model, input_ids):
    """Run one forward pass, return post-layernorm hidden states per layer."""
    h = model.model.embed_tokens(input_ids)
    T = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(h.dtype)

    hidden_states = {}
    for i, layer in enumerate(model.model.layers):
        hidden_states[i] = layer.input_layernorm(h)
        mx.eval(hidden_states[i])
        h = layer(h, mask, None)
        mx.eval(h)

    return hidden_states


def calibrate_routing_keys(model, features, steps=50, lr=1e-3,
                           temperature=0.1, tokens_per_domain=64,
                           verbose=True):
    """Train routing keys with InfoNCE loss on cached features.

    Freezes entire model, unfreezes only routing_key_* params.
    For each step, samples tokens from each domain, computes contrastive
    scores across all experts, and updates keys via cross-entropy.

    Args:
        model: patched model with SelfRoutingLoRALibrary layers
               (must have routing keys initialized).
        features: output of extract_calibration_features().
        steps: optimization steps.
        lr: learning rate for routing keys.
        temperature: contrastive temperature (lower = sharper).
        tokens_per_domain: tokens sampled per domain per step.
        verbose: print progress.

    Returns:
        list of per-step losses.
    """
    # Collect library layers and their layer indices
    lib_layers = collect_library_layers(model)
    if not lib_layers:
        raise ValueError("No SelfRoutingLoRALibrary layers found in model")

    # Use library's own label ordering (matches expert registration)
    first_lib = lib_layers[0][1]
    domains = list(first_lib._labels)
    n_domains = len(domains)
    domain_to_idx = {d: i for i, d in enumerate(domains)}
    # Verify all domains have features
    for d in domains:
        if d not in features:
            raise ValueError(f"No features for domain '{d}'. "
                             f"Available: {list(features.keys())}")

    # Parse layer index from name (e.g., "layers.5.self_attn.q_proj" → 5)
    lib_info = []
    for name, lib in lib_layers:
        layer_idx = _parse_layer_idx(name)
        if not lib._has_routing_keys:
            raise ValueError(f"Layer {name} has no routing keys. "
                             "Call initialize_routing_keys() first.")
        lib_info.append((name, lib, layer_idx))

    # Freeze everything, unfreeze only routing keys
    model.freeze()
    for name, lib, _ in lib_info:
        keys_to_unfreeze = [f"routing_key_{i}" for i in range(lib.n_experts)]
        lib.unfreeze(keys=keys_to_unfreeze)

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, packed_features, packed_labels):
        """InfoNCE loss over routing scores across all library layers."""
        total = mx.array(0.0)
        n_layers = 0
        for _, lib, layer_idx in lib_info:
            x = packed_features[layer_idx]  # (batch, d_in)
            scores = []
            for i in range(lib.n_experts):
                K = getattr(lib, f"routing_key_{i}")
                route = x @ K  # (batch, d_key)
                s = mx.sum(route * route, axis=-1)  # (batch,)
                scores.append(s)
            logits = mx.stack(scores, axis=-1) / temperature  # (batch, n_experts)
            total = total + nn.losses.cross_entropy(
                logits, packed_labels, reduction='mean'
            )
            n_layers += 1
        return total / n_layers

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Unique layer indices (q_proj and v_proj share the same hidden state)
    unique_layer_idxs = sorted(set(li for _, _, li in lib_info))

    losses = []
    rng = np.random.RandomState(42)

    for step in range(steps):
        # Sample tokens from each domain (once per unique transformer layer)
        sampled_features = {}  # layer_idx → (n_domains * tokens_per_domain, d_in)
        labels_list = []

        per_domain_arrays = {li: [] for li in unique_layer_idxs}

        for domain_idx, domain in enumerate(domains):
            domain_feats = features[domain]
            for layer_idx in unique_layer_idxs:
                f = domain_feats[layer_idx]
                n_avail = f.shape[0]
                indices = rng.randint(0, n_avail, size=tokens_per_domain)
                per_domain_arrays[layer_idx].append(f[mx.array(indices)])

            labels_list.extend([domain_idx] * tokens_per_domain)

        for layer_idx in per_domain_arrays:
            sampled_features[layer_idx] = mx.concatenate(
                per_domain_arrays[layer_idx], axis=0
            )
        packed_labels = mx.array(labels_list)

        loss, grads = loss_and_grad(model, sampled_features, packed_labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()
        losses.append(loss_val)

        if verbose and ((step + 1) % max(steps // 5, 1) == 0 or step == 0):
            print(f"    calibration step {step+1:4d}/{steps}: "
                  f"loss={loss_val:.4f}")

    return losses


def evaluate_routing_accuracy(model, features, verbose=True):
    """Measure per-layer and aggregate routing accuracy.

    For each layer and domain, checks what fraction of tokens are
    routed to the correct expert (argmax of routing scores).

    Args:
        model: patched model with calibrated routing keys.
        features: output of extract_calibration_features().
        verbose: print per-layer breakdown.

    Returns:
        dict with 'per_layer' (list of dicts) and 'mean_accuracy' (float).
    """
    lib_layers = collect_library_layers(model)
    if not lib_layers:
        raise ValueError("No SelfRoutingLoRALibrary layers found in model")

    # Use library's own label ordering (matches expert registration)
    first_lib = lib_layers[0][1]
    domains = list(first_lib._labels)
    domain_to_idx = {d: i for i, d in enumerate(domains)}

    lib_info = []
    for name, lib in lib_layers:
        layer_idx = _parse_layer_idx(name)
        lib_info.append((name, lib, layer_idx))

    per_layer = []
    all_correct = 0
    all_total = 0

    for name, lib, layer_idx in lib_info:
        layer_correct = 0
        layer_total = 0
        domain_accs = {}

        for domain_idx, domain in enumerate(domains):
            x = features[domain].get(layer_idx)
            if x is None:
                continue

            scores, _ = lib._score_experts(x)
            selected = mx.argmax(scores, axis=-1)
            mx.eval(selected)

            selected_np = np.array(selected.reshape(-1))
            correct = np.sum(selected_np == domain_idx)
            total = len(selected_np)

            domain_accs[domain] = correct / max(total, 1)
            layer_correct += correct
            layer_total += total

        acc = layer_correct / max(layer_total, 1)
        per_layer.append({
            "name": name,
            "accuracy": acc,
            "domain_accs": domain_accs,
        })
        all_correct += layer_correct
        all_total += layer_total

    mean_acc = all_correct / max(all_total, 1)

    if verbose:
        print(f"\n  {'Layer':<45s} | {'Acc':>6s} | "
              + " | ".join(f"{d:>8s}" for d in domains))
        print(f"  {'-'*45}-+-{'-'*6}-+-"
              + "-+-".join(f"{'-'*8}" for _ in domains))
        for r in per_layer:
            domain_str = " | ".join(
                f"{r['domain_accs'].get(d, 0):8.1%}" for d in domains
            )
            print(f"  {r['name']:<45s} | {r['accuracy']:6.1%} | {domain_str}")
        print(f"\n  Mean routing accuracy: {mean_acc:.1%}")

    return {"per_layer": per_layer, "mean_accuracy": mean_acc}
