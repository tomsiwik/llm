"""LoRA (Low-Rank Adaptation) adapters for ViT experts.

Each expert is a set of LoRA adapters applied to Q and V projections
in each transformer layer. Total params: ~295K per expert (rank=8).

The weight dict interface is compatible with tribe lifecycle operations
(clone, blend_weights, loss_on all work on LoRA weight dicts).
"""
import mlx.core as mx
import numpy as np


def make_lora_expert(rank=8, num_layers=12, d_model=768, num_classes=100, seed=0):
    """Create LoRA adapter weights for one expert.

    Adapts Q and V projections in each transformer layer.
    Total params: 2 * num_layers * 2 * rank * d_model + head params.

    LoRA: output = base_output + (x @ lora_A) @ lora_B
    - lora_A: (d_model, rank) initialized with random normal / sqrt(rank)
    - lora_B: (rank, d_model) initialized to zero (so initial LoRA output = 0)

    Args:
        rank: LoRA rank (default 8).
        num_layers: number of transformer layers to adapt (default 12).
        d_model: model hidden dimension (default 768 for ViT-B).
        num_classes: number of output classes (default 100 for CIFAR-100).
        seed: random seed for initialization.

    Returns:
        dict of LoRA weight arrays (all mx.arrays).
    """
    mx.random.seed(seed)
    weights = {}
    for layer in range(num_layers):
        for target in ['q', 'v']:
            weights[f'layer{layer}.{target}.lora_A'] = mx.random.normal((d_model, rank)) * (1.0 / np.sqrt(rank))
            weights[f'layer{layer}.{target}.lora_B'] = mx.zeros((rank, d_model))
    # Classification head: d_model -> num_classes
    weights['head.weight'] = mx.random.normal((num_classes, d_model)) * 0.01
    weights['head.bias'] = mx.zeros((num_classes,))
    return weights


def lora_param_count(weights):
    """Count total trainable parameters in a LoRA expert."""
    return sum(np.prod(weights[k].shape) for k in weights)


def vit_lora_forward(backbone_weights, lora_weights, X, return_hidden=False):
    """ViT forward pass with LoRA adapters applied.

    For each transformer layer, the attention Q and V projections are modified:
        Q = base_Q + (x @ lora_A_q) @ lora_B_q
        V = base_V + (x @ lora_A_v) @ lora_B_v

    Args:
        backbone_weights: frozen ViT-B/16 weights.
        lora_weights: LoRA adapter weights (from make_lora_expert).
        X: (N, 224, 224, 3) normalized images.
        return_hidden: if True, return (N, 768) CLS features before head.

    Returns:
        (N, 100) logits if return_hidden=False.
        (N, 768) CLS features if return_hidden=True.
    """
    from tribe.vit import _layer_norm, _gelu, VIT_CONFIG

    N = X.shape[0]
    D = VIT_CONFIG['hidden_dim']
    num_heads = VIT_CONFIG['num_heads']
    head_dim = D // num_heads

    # Patch embedding (frozen)
    patches = mx.conv2d(X, backbone_weights['patch_embed.weight'], stride=16, padding=0)
    if 'patch_embed.bias' in backbone_weights:
        patches = patches + backbone_weights['patch_embed.bias']
    patches = mx.reshape(patches, (N, -1, D))

    cls_token = mx.broadcast_to(backbone_weights['cls_token'], (N, 1, D))
    x = mx.concatenate([cls_token, patches], axis=1)
    x = x + backbone_weights['pos_embed']

    # Transformer blocks with LoRA
    for layer_idx in range(VIT_CONFIG['num_layers']):
        prefix = f'blocks.{layer_idx}'

        # Layer norm 1
        h = _layer_norm(x, backbone_weights[f'{prefix}.ln1.weight'],
                        backbone_weights[f'{prefix}.ln1.bias'])

        # Attention with LoRA on Q and V
        S = h.shape[1]
        qkv_w = backbone_weights[f'{prefix}.attn.qkv.weight']  # (3*D, D)
        qkv_b = backbone_weights[f'{prefix}.attn.qkv.bias']

        # Base QKV
        qkv = h @ qkv_w.T + qkv_b  # (N, S, 3*D)

        # LoRA modifications to Q and V
        q_lora_A = lora_weights[f'layer{layer_idx}.q.lora_A']
        q_lora_B = lora_weights[f'layer{layer_idx}.q.lora_B']
        v_lora_A = lora_weights[f'layer{layer_idx}.v.lora_A']
        v_lora_B = lora_weights[f'layer{layer_idx}.v.lora_B']

        # Q LoRA delta
        q_delta = (h @ q_lora_A) @ q_lora_B  # (N, S, D)
        # V LoRA delta
        v_delta = (h @ v_lora_A) @ v_lora_B  # (N, S, D)

        # Split QKV and apply LoRA deltas
        qkv_parts = mx.split(qkv, 3, axis=-1)  # [Q, K, V] each (N, S, D)
        q_adapted = qkv_parts[0] + q_delta
        k_base = qkv_parts[1]
        v_adapted = qkv_parts[2] + v_delta

        # Multi-head attention
        def reshape_heads(t):
            return mx.transpose(mx.reshape(t, (N, S, num_heads, head_dim)), (0, 2, 1, 3))

        q = reshape_heads(q_adapted)
        k = reshape_heads(k_base)
        v = reshape_heads(v_adapted)

        scale = head_dim ** -0.5
        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale
        attn = mx.softmax(attn, axis=-1)
        attn_out = attn @ v
        attn_out = mx.transpose(attn_out, (0, 2, 1, 3))
        attn_out = mx.reshape(attn_out, (N, S, D))

        proj_w = backbone_weights[f'{prefix}.attn.proj.weight']
        proj_b = backbone_weights[f'{prefix}.attn.proj.bias']
        attn_out = attn_out @ proj_w.T + proj_b

        x = x + attn_out

        # MLP (frozen, no LoRA)
        h2 = _layer_norm(x, backbone_weights[f'{prefix}.ln2.weight'],
                         backbone_weights[f'{prefix}.ln2.bias'])
        h2 = _gelu(h2 @ backbone_weights[f'{prefix}.mlp.fc1.weight'].T + backbone_weights[f'{prefix}.mlp.fc1.bias'])
        h2 = h2 @ backbone_weights[f'{prefix}.mlp.fc2.weight'].T + backbone_weights[f'{prefix}.mlp.fc2.bias']
        x = x + h2

    # Final layer norm
    x = _layer_norm(x, backbone_weights['norm.weight'], backbone_weights['norm.bias'])
    cls = x[:, 0, :]  # (N, 768)

    if return_hidden:
        return cls

    # Classification head (trainable, part of LoRA expert)
    logits = cls @ lora_weights['head.weight'].T + lora_weights['head.bias']
    return logits


# ── Sanity Check ─────────────────────────────────────────────

def verify_lora():
    """Verify LoRA expert creation and forward pass."""
    lora = make_lora_expert(rank=8)
    n_params = lora_param_count(lora)
    print(f"  LoRA expert: {n_params:,} params")
    print(f"  Keys: {list(lora.keys())[:6]}...")

    # Test forward pass (requires backbone)
    from tribe.vit import load_vit_backbone
    backbone = load_vit_backbone()
    X = mx.random.normal((2, 224, 224, 3))

    logits = vit_lora_forward(backbone, lora, X)
    mx.eval(logits)
    assert logits.shape == (2, 100), f"Expected (2, 100), got {logits.shape}"
    print(f"  Forward OK: {X.shape} -> {logits.shape}")

    hidden = vit_lora_forward(backbone, lora, X, return_hidden=True)
    mx.eval(hidden)
    assert hidden.shape == (2, 768), f"Expected (2, 768), got {hidden.shape}"
    print(f"  Hidden OK: {hidden.shape}")

    # Test gradient flow (only through LoRA weights, not backbone)
    def loss_fn(lora_w):
        logits = vit_lora_forward(backbone, lora_w, X)
        return mx.mean(logits ** 2)
    loss, grads = mx.value_and_grad(loss_fn)(lora)
    mx.eval(loss, *[grads[k] for k in grads])

    nonzero = sum(1 for k in grads if mx.sum(mx.abs(grads[k])).item() > 0)
    print(f"  Gradient OK: {nonzero}/{len(grads)} keys have nonzero grad")
    print("  LoRA verification passed!")


if __name__ == "__main__":
    verify_lora()
