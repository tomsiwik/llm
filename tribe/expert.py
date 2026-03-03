"""Expert operations: create, forward, train, clone, blend.

Framework-agnostic interface — currently uses MLX, but the operations
(forward pass, gradient step, weight blending) map directly to any
GPU framework (PyTorch, JAX).
"""

import mlx.core as mx
import numpy as np


# Default architecture for toy experiments
DIM = 4
HIDDEN = 6


def make_expert(dim=DIM, hidden=HIDDEN, seed=0):
    """Create a small MLP expert with random weights."""
    mx.random.seed(seed)
    return {
        'W1': mx.random.normal((hidden, dim)) * 0.3,
        'b1': mx.zeros((hidden,)),
        'W2': mx.random.normal((dim, hidden)) * 0.3,
        'b2': mx.zeros((dim,)),
    }


def forward(weights, x):
    """Forward pass: x → ReLU(W1 x + b1) → W2 h + b2."""
    h = mx.maximum(weights['W1'] @ x + weights['b1'], 0)
    return weights['W2'] @ h + weights['b2']


def forward_batch(weights, X):
    """Batched forward pass: X (batch, dim) → (batch, dim)."""
    H = mx.maximum(X @ weights['W1'].T + weights['b1'], 0)
    return H @ weights['W2'].T + weights['b2']


def loss_on(weights, patterns, fwd=None):
    """Mean MSE across (input, target) pattern pairs."""
    if fwd is None:
        fwd = forward_batch
    if not patterns:
        return float('inf')
    X = mx.stack([x for x, _ in patterns])
    T = mx.stack([t for _, t in patterns])
    preds = fwd(weights, X)
    return mx.mean((preds - T) ** 2).item()


def train(weights, patterns, steps=300, lr=0.02, fwd=None, member=None):
    """Train expert on patterns via gradient descent. Returns final loss.

    Args:
        member: optional TribeMember. If provided and has warmup_remaining > 0,
                scales loss by warmup_scale (0→1 ramp) and decrements counter.
    """
    if fwd is None:
        fwd = forward_batch
    X = mx.stack([x for x, _ in patterns])
    T = mx.stack([t for _, t in patterns])
    for _ in range(steps):
        scale = 1.0
        if member is not None and member.warmup_remaining > 0:
            scale = member.warmup_scale
        def loss_fn(w, _scale=scale):
            preds = fwd(w, X)
            return _scale * mx.mean((preds - T) ** 2)
        _, grads = mx.value_and_grad(loss_fn)(weights)
        for k in weights:
            weights[k] = weights[k] - lr * grads[k]
        mx.eval(*[weights[k] for k in weights])
        if member is not None:
            if member.warmup_remaining > 0:
                member.warmup_remaining -= 1
            member.age += 1
    return loss_on(weights, patterns, fwd=fwd)


def orthogonality_loss(expert_hiddens):
    """Penalize similarity between expert hidden representations.

    For each pair of experts (i, j), computes mean squared cosine similarity
    of their hidden representations. Returns mean across all pairs.

    This encourages experts to develop independent internal representations,
    reducing redundancy in the mixture.

    Args:
        expert_hiddens: list of (N, hidden_dim) mx.arrays, one per expert.

    Returns:
        scalar loss: mean squared cosine similarity across all expert pairs.
    """
    E = len(expert_hiddens)
    if E < 2:
        return mx.array(0.0)

    loss = mx.array(0.0)
    count = 0
    for i in range(E):
        for j in range(i + 1, E):
            # L2-normalize each expert's hidden representations
            hi_norm = expert_hiddens[i] / (mx.linalg.norm(expert_hiddens[i], axis=-1, keepdims=True) + 1e-8)
            hj_norm = expert_hiddens[j] / (mx.linalg.norm(expert_hiddens[j], axis=-1, keepdims=True) + 1e-8)
            # Cosine similarity per sample, then mean of squared values
            cos_sim = mx.sum(hi_norm * hj_norm, axis=-1)  # (N,)
            loss = loss + mx.mean(cos_sim ** 2)
            count += 1

    return loss / count


def clone(weights):
    """Deep copy of weight dict."""
    return {k: mx.array(weights[k]) for k in weights}


def blend_weights(weight_list, contributions):
    """Weighted blend of multiple experts' weights.

    Args:
        weight_list: list of weight dicts
        contributions: list of floats summing to ~1.0

    Returns:
        New weight dict with blended values.
    """
    result = {}
    for k in weight_list[0]:
        blended = sum(c * np.array(w[k]) for w, c in zip(weight_list, contributions))
        result[k] = mx.array(blended.astype(np.float32))
    return result


def param_count(weights):
    """Total number of parameters in a weight dict."""
    return sum(np.prod(weights[k].shape) for k in weights)


def reset_optimizer_state(optimizer, weight_keys=None):
    """Zero Adam moments for recycled expert. See examples/04_redo_dormant_reinit.py.

    MLX Adam state layout:
        state['step']  — global step counter (uint64)
        state[key]     — {'m': first moment, 'v': second moment} per weight key

    Args:
        optimizer: mlx.optimizers.Adam instance (already init'd).
        weight_keys: which weight keys to reset. If None, resets all param entries.
    """
    state = optimizer.state
    keys = weight_keys or [k for k in state if k not in ('step', 'learning_rate')]
    for k in keys:
        if k in state and isinstance(state[k], dict):
            state[k]['m'] = mx.zeros_like(state[k]['m'])
            state[k]['v'] = mx.zeros_like(state[k]['v'])
    # Reset step counter so Adam bias correction treats params as fresh
    state['step'] = mx.array(0, dtype=state['step'].dtype)
