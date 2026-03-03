# Phase 3: Joint Router-Expert Training (P1)

## Goal
Train router and experts jointly via straight-through estimation. Single biggest accuracy improvement for learned routing.

## Background

Current state: `SwitchRouter.train_step()` only optimizes `sum(P_i^2)` for load balance. It never sees expert training loss. The router is blind to expert quality.

Target: Router loss = expert_task_loss + alpha * load_balance_loss, backpropagated through both router weights and expert weights jointly.

## Changes

### 3a. Differentiable Soft MoE Forward Pass

**`tribe/router.py`** — add `soft_forward()`:
```python
def soft_forward(self, X_flat, expert_weights_list, fwd, labels=None):
    """Differentiable MoE forward with straight-through routing.

    Returns (mixed_logits, aux_loss, stats) where mixed_logits has gradients
    flowing through both router and expert parameters.
    """
    probs = self._probs(X_flat)  # (N, E) — differentiable

    # Hard assignments (stop gradient) for dispatch
    hard_assignments = mx.stop_gradient(
        mx.one_hot(mx.argmax(probs, axis=-1), self.num_experts)
    )  # (N, E) one-hot

    # Use soft probs for gradient, hard for dispatch (STE)
    # Trick: hard - stop_gradient(soft) + soft
    routing_weights = hard_assignments - mx.stop_gradient(probs) + probs  # (N, E)

    # Run each expert on ALL inputs (we'll weight by routing)
    expert_outputs = []
    for i, ew in enumerate(expert_weights_list):
        # Need to convert flat input back to image shape for CNN/ResNet
        out = fwd(ew, X_images)  # (N, num_classes)
        expert_outputs.append(out)
    expert_stack = mx.stack(expert_outputs, axis=1)  # (N, E, C)

    # Mix by routing weights
    routing_expanded = mx.expand_dims(routing_weights, axis=-1)  # (N, E, 1)
    mixed = mx.sum(expert_stack * routing_expanded, axis=1)  # (N, C)

    # Aux loss
    aux = self.load_balancing_loss(X_flat)

    return mixed, aux, {'probs': probs}
```

**Note**: Running ALL experts on ALL inputs is expensive (E forward passes). For top-k=1 with 10 experts, this is 10x the compute. Optimization: only run top-2 experts per sample and zero the rest.

### 3b. Efficient Top-K Dispatch with STE

**`tribe/router.py`** — optimized version:
```python
def soft_forward_topk(self, X_flat, X_images, expert_weights_list, fwd, top_k=2):
    """Top-k STE routing — only runs top-k experts per sample."""
    probs = self._probs(X_flat)  # (N, E)
    N, E = probs.shape

    # Get top-k expert indices per sample
    top_k_indices = mx.argpartition(-np.array(probs), top_k, axis=1)[:, :top_k]

    # For each expert, compute outputs only for samples routed to it
    mixed_output = mx.zeros((N, expert_weights_list[0]['fc_w'].shape[0]))

    for i, ew in enumerate(expert_weights_list):
        # Which samples have this expert in their top-k?
        mask = np.any(np.array(top_k_indices) == i, axis=1)  # (N,) bool
        if not np.any(mask):
            continue
        sample_idx = np.where(mask)[0]
        X_sub = X_images[mx.array(sample_idx)]
        out = fwd(ew, X_sub)  # (n_sub, C)

        # Weight by routing probability (differentiable)
        weight = probs[mx.array(sample_idx), i:i+1]  # (n_sub, 1)
        # Scatter weighted output back
        for j, si in enumerate(sample_idx):
            mixed_output = mixed_output.at[si].add(weight[j] * out[j])

    # Normalize by sum of routing weights per sample
    weight_sum = mx.sum(probs * (hard_mask), axis=1, keepdims=True)
    mixed_output = mixed_output / (weight_sum + 1e-8)

    aux = self.load_balancing_loss(X_flat)
    return mixed_output, aux
```

### 3c. Joint Training Loop

**`bench_cifar100.py`** — replace separate router/expert training:
```python
def train_step_joint(router, expert_weights_list, X_images, labels, fwd,
                     optimizer_router, optimizers_expert, alpha=0.01):
    """Joint router + expert training step."""

    X_flat = mx.reshape(X_images, (X_images.shape[0], -1))

    # All params in one dict for value_and_grad
    all_params = {'router': router.weights}
    for i, ew in enumerate(expert_weights_list):
        all_params[f'expert_{i}'] = ew

    def joint_loss(params):
        # Router forward
        logits = X_flat @ params['router']['router_w'].T + params['router']['router_b']
        probs = mx.softmax(logits, axis=-1)

        # STE: hard assignment with soft gradient
        hard = mx.one_hot(mx.argmax(mx.stop_gradient(probs), axis=-1), router.num_experts)
        weights = hard - mx.stop_gradient(probs) + probs

        # Expert outputs
        expert_outs = []
        for i in range(len(expert_weights_list)):
            out = fwd(params[f'expert_{i}'], X_images)
            expert_outs.append(out)
        stacked = mx.stack(expert_outs, axis=1)  # (N, E, C)

        # Weighted mix
        mixed = mx.sum(stacked * mx.expand_dims(weights, -1), axis=1)  # (N, C)

        # Task loss
        task_loss = cross_entropy_loss(mixed, labels)

        # Load balance loss
        f = mx.mean(hard, axis=0)
        P = mx.mean(probs, axis=0)
        aux = router.num_experts * mx.sum(f * P)

        return task_loss + alpha * aux

    loss, grads = mx.value_and_grad(joint_loss)(all_params)

    # Update router
    for k in router.weights:
        router.weights[k] = router.weights[k] - lr * grads['router'][k]

    # Update experts
    for i, ew in enumerate(expert_weights_list):
        g = grads[f'expert_{i}']
        for k in ew:
            ew[k] = ew[k] - lr * clip_grad(g[k])

    return loss
```

### 3d. Router Lifecycle Integration

When an expert is recycled, the router must adapt:
```python
def on_expert_recycled(router, expert_idx):
    """Reset router's bias for recycled expert to encourage routing to it."""
    router.weights['router_b'] = router.weights['router_b'].at[expert_idx].add(0.5)
    # Small positive bias = router gives recycled expert a chance
```

When an expert is frozen, reduce its routing probability:
```python
def on_expert_frozen(router, expert_idx):
    """Frozen experts still route but shouldn't dominate."""
    # No change needed — frozen expert continues to serve with current weights
    # Router naturally adapts via joint training
    pass
```

## Verification
```bash
# Smoke test: joint training converges
uv run --with mlx python -c "
from tribe.router import SwitchRouter
import mlx.core as mx
r = SwitchRouter(3072, 10, top_k=2)
# ... create dummy experts, run joint_loss, check gradients flow
"

# Full benchmark
uv run --with mlx python bench_cifar100.py
# Compare: Learned+LC (joint) vs Learned+LC (separate)
# Expected: joint training improves learned router FA by 5-10%
```

## Files Modified
| File | Change | Lines |
|------|--------|-------|
| tribe/router.py | `soft_forward()`, `soft_forward_topk()` | ~60 |
| bench_cifar100.py | `train_step_joint()`, joint training loop for method 6 | ~80 |

## Success Criteria
- Learned router FA improves from 24.9% → 30-35% (closing gap to oracle)
- Router entropy converges toward non-trivial value (not collapsed)
- Joint training adds < 50% overhead vs separate training
- Gradient norms are stable (no explosion from STE)
