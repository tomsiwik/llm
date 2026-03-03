"""ResNet-18-lite expert for CIFAR-100 scale experiments.

Reduced-width ResNet-18 (~270K params) using flat dict weights compatible
with the tribe lifecycle (clone, blend_weights, loss_on, train all work).

Architecture (half-width ResNet-18):
  Conv2d(3→32, k=3, pad=1) → BN → ReLU
  Block1: 2x [Conv(32→32, k=3, pad=1) → BN → ReLU → Conv(32→32) → BN] + residual
  Block2: 2x [Conv(32→64, k=3, s=2, pad=1) → BN → ReLU → Conv(64→64) → BN] + residual + proj
  Block3: 2x [Conv(64→128, k=3, s=2, pad=1) → BN → ReLU → Conv(128→128) → BN] + residual + proj
  Block4: 2x [Conv(128→256, k=3, s=2, pad=1) → BN → ReLU → Conv(256→256) → BN] + residual + proj
  GlobalAvgPool → Linear(256→100)

Using batch norm in eval mode (frozen stats) for simplicity with weight dicts.
We use layer norm instead (no running stats needed, works with weight dicts).
"""

import mlx.core as mx
import numpy as np


def _kaiming(shape, fan_in=None):
    """Kaiming He initialization for ReLU."""
    if fan_in is None:
        if len(shape) == 4:  # conv: (out, h, w, in) in MLX OHWI format
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            fan_in = shape[-1]
    std = np.sqrt(2.0 / fan_in)
    return mx.random.normal(shape) * std


def _make_conv_block(prefix, in_c, out_c, stride=1):
    """Create weights for: Conv → ReLU → Conv (residual block)."""
    w = {}
    # First conv
    w[f'{prefix}_conv1_w'] = _kaiming((out_c, 3, 3, in_c))
    w[f'{prefix}_conv1_b'] = mx.zeros((out_c,))
    # Second conv
    w[f'{prefix}_conv2_w'] = _kaiming((out_c, 3, 3, out_c))
    w[f'{prefix}_conv2_b'] = mx.zeros((out_c,))
    # Projection shortcut if dimensions change
    if stride > 1 or in_c != out_c:
        w[f'{prefix}_proj_w'] = _kaiming((out_c, 1, 1, in_c))
        w[f'{prefix}_proj_b'] = mx.zeros((out_c,))
    return w


def make_resnet_expert(seed=0, width=32):
    """Create a ResNet-18-lite expert with Kaiming initialization.

    Args:
        seed: random seed.
        width: base width (32 = ~270K params, 64 = ~1.1M params).
    """
    mx.random.seed(seed)
    w = {}

    # Stem: Conv(3→width, 3x3)
    w['stem_conv_w'] = _kaiming((width, 3, 3, 3))
    w['stem_conv_b'] = mx.zeros((width,))

    # Block 1: 2 residual blocks, no downsampling
    c = width
    w.update(_make_conv_block('b1a', c, c, stride=1))
    w.update(_make_conv_block('b1b', c, c, stride=1))

    # Block 2: downsample, double channels
    w.update(_make_conv_block('b2a', c, c * 2, stride=2))
    w.update(_make_conv_block('b2b', c * 2, c * 2, stride=1))

    # Block 3: downsample, double channels
    w.update(_make_conv_block('b3a', c * 2, c * 4, stride=2))
    w.update(_make_conv_block('b3b', c * 4, c * 4, stride=1))

    # Block 4: downsample, double channels
    w.update(_make_conv_block('b4a', c * 4, c * 8, stride=2))
    w.update(_make_conv_block('b4b', c * 8, c * 8, stride=1))

    # Classifier head
    w['fc_w'] = _kaiming((100, c * 8), fan_in=c * 8)
    w['fc_b'] = mx.zeros((100,))

    return w


def _conv_block_forward(weights, x, prefix, stride=1):
    """Forward through a residual block."""
    identity = x

    # First conv + ReLU
    h = mx.conv2d(x, weights[f'{prefix}_conv1_w'], stride=stride, padding=1)
    h = h + weights[f'{prefix}_conv1_b']
    h = mx.maximum(h, 0)

    # Second conv
    h = mx.conv2d(h, weights[f'{prefix}_conv2_w'], stride=1, padding=1)
    h = h + weights[f'{prefix}_conv2_b']

    # Projection shortcut
    proj_key = f'{prefix}_proj_w'
    if proj_key in weights:
        identity = mx.conv2d(x, weights[proj_key], stride=stride, padding=0)
        identity = identity + weights[f'{prefix}_proj_b']

    return mx.maximum(h + identity, 0)


def resnet_forward_hidden(weights, X):
    """Forward through ResNet, return pre-FC hidden state.

    Same as resnet_forward_batch but stops before the final FC layer,
    returning the global-average-pooled hidden representation.

    Args:
        weights: ResNet weight dict.
        X: (N, 32, 32, 3) image batch.

    Returns:
        (N, 8*width) hidden representation before classifier.
    """
    # Stem
    h = mx.conv2d(X, weights['stem_conv_w'], stride=1, padding=1)
    h = h + weights['stem_conv_b']
    h = mx.maximum(h, 0)

    # Residual blocks
    for block in ['b1a', 'b1b']:
        h = _conv_block_forward(weights, h, block, stride=1)

    h = _conv_block_forward(weights, h, 'b2a', stride=2)
    h = _conv_block_forward(weights, h, 'b2b', stride=1)

    h = _conv_block_forward(weights, h, 'b3a', stride=2)
    h = _conv_block_forward(weights, h, 'b3b', stride=1)

    h = _conv_block_forward(weights, h, 'b4a', stride=2)
    h = _conv_block_forward(weights, h, 'b4b', stride=1)

    # Global average pooling: (N, H, W, C) → (N, C)
    h = mx.mean(h, axis=(1, 2))

    return h


def resnet_forward_batch(weights, X):
    """Batched ResNet forward pass. X: (N, 32, 32, 3) → (N, 100)."""
    # Hidden representation (everything before FC)
    h = resnet_forward_hidden(weights, X)

    # Classifier
    h = h @ weights['fc_w'].T + weights['fc_b']
    return h


def verify_resnet():
    """Sanity check: shapes, gradients, param count."""
    w = make_resnet_expert(seed=0, width=32)
    total = sum(np.prod(w[k].shape) for k in w)
    print(f"  ResNet params: {total:,}")

    X = mx.random.normal((4, 32, 32, 3))
    out = resnet_forward_batch(w, X)
    mx.eval(out)
    assert out.shape == (4, 100), f"Expected (4,100), got {out.shape}"
    print(f"  Forward OK: {X.shape} → {out.shape}")

    T = mx.random.normal((4, 100))
    def loss_fn(weights):
        preds = resnet_forward_batch(weights, X)
        return mx.mean((preds - T) ** 2)
    loss, grads = mx.value_and_grad(loss_fn)(w)
    mx.eval(loss, *[grads[k] for k in grads])

    zero_grads = [k for k in grads if mx.sum(mx.abs(grads[k])).item() == 0]
    assert not zero_grads, f"Zero gradients: {zero_grads}"
    print(f"  Gradients OK: loss={loss.item():.4f}, all layers have nonzero grad")


if __name__ == "__main__":
    verify_resnet()
