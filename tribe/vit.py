"""ViT-B/16 backbone in pure MLX for LGME continual learning.

Loads pretrained ImageNet-21K weights from HuggingFace (google/vit-base-patch16-224-in21k).
All backbone weights are frozen -- only LoRA adapters are trainable.
"""
import mlx.core as mx
import numpy as np
import os

# ViT-B/16 config
VIT_CONFIG = {
    'image_size': 224,
    'patch_size': 16,
    'hidden_dim': 768,
    'num_heads': 12,
    'num_layers': 12,
    'mlp_dim': 3072,
    'num_patches': 196,  # (224/16)^2
}


def _gelu(x):
    """GELU activation (fast approximation)."""
    return x * mx.sigmoid(1.702 * x)


def _layer_norm(x, weight, bias, eps=1e-6):
    """Layer normalization."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return weight * (x - mean) / mx.sqrt(var + eps) + bias


def _attention(x, qkv_w, qkv_b, proj_w, proj_b, num_heads=12):
    """Multi-head self-attention.

    Args:
        x: (N, S, D) where S=num_patches+1 (cls), D=768
    """
    N, S, D = x.shape
    head_dim = D // num_heads

    # QKV projection
    qkv = x @ qkv_w.T + qkv_b  # (N, S, 3*D)
    qkv = mx.reshape(qkv, (N, S, 3, num_heads, head_dim))
    qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))  # (3, N, H, S, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale  # (N, H, S, S)
    attn = mx.softmax(attn, axis=-1)

    out = attn @ v  # (N, H, S, head_dim)
    out = mx.transpose(out, (0, 2, 1, 3))  # (N, S, H, head_dim)
    out = mx.reshape(out, (N, S, D))

    # Output projection
    out = out @ proj_w.T + proj_b
    return out


def _mlp(x, fc1_w, fc1_b, fc2_w, fc2_b):
    """MLP block: Linear -> GELU -> Linear."""
    h = x @ fc1_w.T + fc1_b
    h = _gelu(h)
    h = h @ fc2_w.T + fc2_b
    return h


def _transformer_block(x, weights, prefix):
    """Single transformer block: LN -> Attn -> Residual -> LN -> MLP -> Residual."""
    # Self-attention
    h = _layer_norm(x, weights[f'{prefix}.ln1.weight'], weights[f'{prefix}.ln1.bias'])
    h = _attention(h,
        weights[f'{prefix}.attn.qkv.weight'],
        weights[f'{prefix}.attn.qkv.bias'],
        weights[f'{prefix}.attn.proj.weight'],
        weights[f'{prefix}.attn.proj.bias'])
    x = x + h

    # MLP
    h = _layer_norm(x, weights[f'{prefix}.ln2.weight'], weights[f'{prefix}.ln2.bias'])
    h = _mlp(h,
        weights[f'{prefix}.mlp.fc1.weight'],
        weights[f'{prefix}.mlp.fc1.bias'],
        weights[f'{prefix}.mlp.fc2.weight'],
        weights[f'{prefix}.mlp.fc2.bias'])
    x = x + h
    return x


def vit_forward(weights, X, return_cls=True):
    """ViT-B/16 forward pass.

    Args:
        weights: dict of frozen ViT weights.
        X: (N, 224, 224, 3) images, float32 normalized.
        return_cls: if True, return (N, 768) CLS features.
                    If False, return (N, 197, 768) all tokens.

    Returns:
        (N, 768) CLS token features if return_cls=True.
    """
    N = X.shape[0]

    # Patch embedding: (N, 224, 224, 3) -> (N, 196, 768)
    # Conv2d with stride=16, kernel=16
    patches = mx.conv2d(X, weights['patch_embed.weight'], stride=16, padding=0)
    # Add patch embedding bias if present (ViT-B/16 has it)
    if 'patch_embed.bias' in weights:
        patches = patches + weights['patch_embed.bias']
    # patches shape: (N, 14, 14, 768)
    patches = mx.reshape(patches, (N, -1, VIT_CONFIG['hidden_dim']))  # (N, 196, 768)

    # Prepend CLS token
    cls_token = mx.broadcast_to(weights['cls_token'], (N, 1, VIT_CONFIG['hidden_dim']))
    x = mx.concatenate([cls_token, patches], axis=1)  # (N, 197, 768)

    # Add position embedding
    x = x + weights['pos_embed']  # (1, 197, 768) broadcasts

    # Transformer blocks
    for layer_idx in range(VIT_CONFIG['num_layers']):
        prefix = f'blocks.{layer_idx}'
        x = _transformer_block(x, weights, prefix)

    # Final layer norm
    x = _layer_norm(x, weights['norm.weight'], weights['norm.bias'])

    if return_cls:
        return x[:, 0, :]  # CLS token: (N, 768)
    return x


# ── Weight Loading ────────────────────────────────────────────

WEIGHTS_DIR = os.path.expanduser("~/.cache/vit-b16-in21k")


def download_vit_weights():
    """Download ViT-B/16 pretrained weights from HuggingFace.

    Tries safetensors first (no torch dependency), falls back to pytorch_model.bin.
    Converts to our flat dict format and caches as .npz.
    """
    weights_path = os.path.join(WEIGHTS_DIR, "vit_b16.npz")
    if os.path.exists(weights_path):
        return weights_path

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_path = None

    # Strategy 1: Use huggingface_hub to download safetensors
    try:
        from huggingface_hub import hf_hub_download
        print("  Downloading ViT-B/16 weights via huggingface_hub...")
        try:
            model_path = hf_hub_download(
                repo_id="google/vit-base-patch16-224-in21k",
                filename="model.safetensors",
                cache_dir=WEIGHTS_DIR
            )
        except Exception:
            # Some revisions may not have safetensors, try pytorch
            model_path = hf_hub_download(
                repo_id="google/vit-base-patch16-224-in21k",
                filename="pytorch_model.bin",
                cache_dir=WEIGHTS_DIR
            )
    except ImportError:
        pass

    # Strategy 2: Direct download of safetensors
    if model_path is None:
        import urllib.request
        safetensors_path = os.path.join(WEIGHTS_DIR, "model.safetensors")
        if not os.path.exists(safetensors_path):
            url = "https://huggingface.co/google/vit-base-patch16-224-in21k/resolve/main/model.safetensors"
            print("  Downloading ViT-B/16 weights (~330MB)...")
            urllib.request.urlretrieve(url, safetensors_path)
        model_path = safetensors_path

    # Convert to our weight format
    print("  Converting weights to MLX format...")
    weights = _convert_hf_weights(model_path)
    np.savez(weights_path, **{k: np.array(v) for k, v in weights.items()})
    print(f"  Saved converted weights to {weights_path}")
    return weights_path


def _convert_hf_weights(model_path):
    """Convert HuggingFace ViT weights to our flat dict format.

    Handles both safetensors and pytorch_model.bin formats.
    Safetensors keys omit the 'vit.' prefix; pytorch_model.bin keys include it.
    We auto-detect and handle both.
    """
    if model_path.endswith('.safetensors'):
        from safetensors.numpy import load_file
        hf_weights = load_file(model_path)
    else:
        import torch
        hf_weights = torch.load(model_path, map_location='cpu', weights_only=True)
        hf_weights = {k: v.numpy() for k, v in hf_weights.items()}

    # Auto-detect prefix: safetensors uses no prefix, pytorch uses 'vit.'
    sample_key = next(iter(hf_weights))
    prefix = 'vit.' if sample_key.startswith('vit.') else ''

    weights = {}

    # Patch embedding: (768, 3, 16, 16) PyTorch OIHW -> (768, 16, 16, 3) MLX OHWI
    pe_w = hf_weights[f'{prefix}embeddings.patch_embeddings.projection.weight']
    if pe_w.shape == (768, 3, 16, 16):
        pe_w = pe_w.transpose(0, 2, 3, 1)  # OIHW -> OHWI
    weights['patch_embed.weight'] = pe_w.astype(np.float32)

    # Patch embedding bias
    pe_b_key = f'{prefix}embeddings.patch_embeddings.projection.bias'
    if pe_b_key in hf_weights:
        weights['patch_embed.bias'] = hf_weights[pe_b_key].astype(np.float32)

    # CLS token: (1, 1, 768)
    weights['cls_token'] = hf_weights[f'{prefix}embeddings.cls_token'].astype(np.float32)

    # Position embedding: (1, 197, 768)
    weights['pos_embed'] = hf_weights[f'{prefix}embeddings.position_embeddings'].astype(np.float32)

    # Transformer blocks
    for i in range(12):
        hf_pfx = f'{prefix}encoder.layer.{i}'
        my_prefix = f'blocks.{i}'

        # Layer norms
        weights[f'{my_prefix}.ln1.weight'] = hf_weights[f'{hf_pfx}.layernorm_before.weight'].astype(np.float32)
        weights[f'{my_prefix}.ln1.bias'] = hf_weights[f'{hf_pfx}.layernorm_before.bias'].astype(np.float32)
        weights[f'{my_prefix}.ln2.weight'] = hf_weights[f'{hf_pfx}.layernorm_after.weight'].astype(np.float32)
        weights[f'{my_prefix}.ln2.bias'] = hf_weights[f'{hf_pfx}.layernorm_after.bias'].astype(np.float32)

        # Attention: separate Q, K, V -> concatenated QKV
        q_w = hf_weights[f'{hf_pfx}.attention.attention.query.weight'].astype(np.float32)
        k_w = hf_weights[f'{hf_pfx}.attention.attention.key.weight'].astype(np.float32)
        v_w = hf_weights[f'{hf_pfx}.attention.attention.value.weight'].astype(np.float32)
        q_b = hf_weights[f'{hf_pfx}.attention.attention.query.bias'].astype(np.float32)
        k_b = hf_weights[f'{hf_pfx}.attention.attention.key.bias'].astype(np.float32)
        v_b = hf_weights[f'{hf_pfx}.attention.attention.value.bias'].astype(np.float32)

        weights[f'{my_prefix}.attn.qkv.weight'] = np.concatenate([q_w, k_w, v_w], axis=0)  # (3*768, 768)
        weights[f'{my_prefix}.attn.qkv.bias'] = np.concatenate([q_b, k_b, v_b], axis=0)      # (3*768,)

        # Output projection
        weights[f'{my_prefix}.attn.proj.weight'] = hf_weights[f'{hf_pfx}.attention.output.dense.weight'].astype(np.float32)
        weights[f'{my_prefix}.attn.proj.bias'] = hf_weights[f'{hf_pfx}.attention.output.dense.bias'].astype(np.float32)

        # MLP
        weights[f'{my_prefix}.mlp.fc1.weight'] = hf_weights[f'{hf_pfx}.intermediate.dense.weight'].astype(np.float32)
        weights[f'{my_prefix}.mlp.fc1.bias'] = hf_weights[f'{hf_pfx}.intermediate.dense.bias'].astype(np.float32)
        weights[f'{my_prefix}.mlp.fc2.weight'] = hf_weights[f'{hf_pfx}.output.dense.weight'].astype(np.float32)
        weights[f'{my_prefix}.mlp.fc2.bias'] = hf_weights[f'{hf_pfx}.output.dense.bias'].astype(np.float32)

    # Final layer norm
    weights['norm.weight'] = hf_weights[f'{prefix}layernorm.weight'].astype(np.float32)
    weights['norm.bias'] = hf_weights[f'{prefix}layernorm.bias'].astype(np.float32)

    return weights


def load_vit_backbone():
    """Load pretrained ViT-B/16 backbone. Returns weight dict (all mx.arrays)."""
    weights_path = download_vit_weights()
    data = np.load(weights_path)
    return {k: mx.array(data[k]) for k in data.files}


# ── Image Preprocessing ──────────────────────────────────────

def resize_images_224(images_32, method='nearest'):
    """Resize (N, 32, 32, 3) images to (N, 224, 224, 3).

    Args:
        images_32: (N, 32, 32, 3) float32 images.
        method: 'nearest' for fast pixel repeat (default), 'bilinear' for scipy zoom.

    Applied once at data load time, not per-batch.
    """
    N = images_32.shape[0]
    if method == 'bilinear':
        try:
            from scipy.ndimage import zoom
            # zoom factor: 224/32 = 7.0 for spatial dims
            batch_size = 500
            result = np.empty((N, 224, 224, 3), dtype=np.float32)
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = images_32[start:end]
                result[start:end] = zoom(batch, (1, 7, 7, 1), order=1)
            return result
        except ImportError:
            pass  # fall through to nearest
    # Nearest-neighbor: repeat each pixel 7x7 (fast, works well with ViT)
    return np.repeat(np.repeat(images_32, 7, axis=1), 7, axis=2)


# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(images):
    """Apply ImageNet normalization. images: (N, H, W, 3) float32 [0,1]."""
    return (images - IMAGENET_MEAN) / IMAGENET_STD


# ── Sanity Check ─────────────────────────────────────────────

def verify_vit():
    """Verify ViT backbone loads and forward pass works."""
    print("  Loading ViT-B/16 backbone...")
    backbone = load_vit_backbone()
    n_tensors = len(backbone)
    n_params = sum(np.prod(backbone[k].shape) for k in backbone)
    print(f"  Backbone: {n_tensors} tensors, {n_params:,} params")

    X = mx.random.normal((2, 224, 224, 3))
    features = vit_forward(backbone, X)
    mx.eval(features)
    assert features.shape == (2, 768), f"Expected (2, 768), got {features.shape}"
    print(f"  Forward OK: {X.shape} -> {features.shape}")

    all_tokens = vit_forward(backbone, X, return_cls=False)
    mx.eval(all_tokens)
    assert all_tokens.shape == (2, 197, 768), f"Expected (2, 197, 768), got {all_tokens.shape}"
    print(f"  All tokens OK: {all_tokens.shape}")
    print("  ViT-B/16 verification passed!")


if __name__ == "__main__":
    verify_vit()
