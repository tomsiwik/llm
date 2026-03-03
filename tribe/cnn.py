"""CNN expert for MNIST-scale experiments.

Tiny CNN (~64K params) using flat dict weights compatible with
the tribe lifecycle (clone, blend_weights, loss_on, train all work).

Architecture:
  Conv2d(1→16, k=5, stride=2, pad=2) → ReLU → (14,14,16)
  Conv2d(16→32, k=5, stride=2, pad=2) → ReLU → (7,7,32)
  Flatten → 1568
  Linear(1568→32) → ReLU
  Linear(32→10)
"""

import mlx.core as mx
import numpy as np


def make_cnn_expert(seed=0):
    """Create a CNN expert with Kaiming-initialized weights."""
    mx.random.seed(seed)
    return {
        'conv1_w': mx.random.normal((16, 5, 5, 1)) * np.sqrt(2.0 / 25),
        'conv1_b': mx.zeros((16,)),
        'conv2_w': mx.random.normal((32, 5, 5, 16)) * np.sqrt(2.0 / (25 * 16)),
        'conv2_b': mx.zeros((32,)),
        'fc1_w': mx.random.normal((32, 1568)) * np.sqrt(2.0 / 1568),
        'fc1_b': mx.zeros((32,)),
        'fc2_w': mx.random.normal((10, 32)) * np.sqrt(2.0 / 32),
        'fc2_b': mx.zeros((10,)),
    }


def cnn_forward_batch(weights, X):
    """Batched CNN forward pass. X: (N, 28, 28, 1) → (N, 10)."""
    # Conv1: (N,28,28,1) → (N,14,14,16)
    h = mx.conv2d(X, weights['conv1_w'], stride=2, padding=2)
    h = h + weights['conv1_b']
    h = mx.maximum(h, 0)

    # Conv2: (N,14,14,16) → (N,7,7,32)
    h = mx.conv2d(h, weights['conv2_w'], stride=2, padding=2)
    h = h + weights['conv2_b']
    h = mx.maximum(h, 0)

    # Flatten: (N,7,7,32) → (N,1568)
    N = h.shape[0]
    h = mx.reshape(h, (N, -1))

    # FC1: (N,1568) → (N,32)
    h = h @ weights['fc1_w'].T + weights['fc1_b']
    h = mx.maximum(h, 0)

    # FC2: (N,32) → (N,10)
    h = h @ weights['fc2_w'].T + weights['fc2_b']
    return h


def verify_cnn_shapes():
    """Sanity check: forward pass produces correct output shape."""
    w = make_cnn_expert(seed=0)
    X = mx.random.normal((4, 28, 28, 1))
    out = cnn_forward_batch(w, X)
    mx.eval(out)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"
    total = sum(np.prod(w[k].shape) for k in w)
    print(f"  CNN shapes OK: input (4,28,28,1) → output {out.shape}, params={total}")


def verify_cnn_gradients():
    """Sanity check: gradients flow through all layers."""
    w = make_cnn_expert(seed=42)
    X = mx.random.normal((4, 28, 28, 1))
    T = mx.random.normal((4, 10))

    def loss_fn(weights):
        preds = cnn_forward_batch(weights, X)
        return mx.mean((preds - T) ** 2)

    loss, grads = mx.value_and_grad(loss_fn)(w)
    mx.eval(loss, *[grads[k] for k in grads])

    for k in w:
        g_norm = mx.sum(grads[k] ** 2).item()
        assert g_norm > 0, f"Zero gradient for {k}"
        print(f"  {k:10s} shape={str(w[k].shape):16s} grad_norm={g_norm:.6f}")
    print(f"  CNN gradients OK: loss={loss.item():.4f}")
