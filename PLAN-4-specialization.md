# Phase 4: Expert Specialization Losses (P6)

## Goal
Add orthogonality + variance losses to actively push experts apart. Published result: 23.79% relative improvement, 45% overlap reduction.

## Background

Paper: "Advancing Expert Specialization for Better MoE" (May 2025, arxiv:2505.22323)

Two losses that combat expert overlap:
1. **Orthogonality loss**: Minimize cross-expert representation similarity
2. **Variance loss**: Maximize routing score variance (discourage uniform routing)

These are *proactive* defenses against overlap (prevent it from forming), complementing the lifecycle's *reactive* cleanup (remove overlap after it forms).

## Changes

### 4a. Orthogonality Loss Between Expert Representations

**`tribe/expert.py`** — add `orthogonality_loss()`:
```python
def orthogonality_loss(expert_weights_list, X, fwd):
    """Encourage independent expert representations.

    For each pair of experts, compute cosine similarity of their
    hidden representations on the same input batch. Penalize high similarity.

    L_orth = (1/E^2) * sum_{i!=j} cos_sim(h_i, h_j)^2
    """
    # Get hidden representations (before final FC layer)
    hiddens = []
    for ew in expert_weights_list:
        h = fwd_hidden(ew, X)  # Need a forward function that returns hidden state
        hiddens.append(h)  # (N, hidden_dim)

    E = len(hiddens)
    loss = mx.array(0.0)
    count = 0
    for i in range(E):
        for j in range(i+1, E):
            # Cosine similarity per sample, then mean
            hi_norm = hiddens[i] / (mx.linalg.norm(hiddens[i], axis=-1, keepdims=True) + 1e-8)
            hj_norm = hiddens[j] / (mx.linalg.norm(hiddens[j], axis=-1, keepdims=True) + 1e-8)
            cos_sim = mx.sum(hi_norm * hj_norm, axis=-1)  # (N,)
            loss = loss + mx.mean(cos_sim ** 2)
            count += 1

    return loss / max(count, 1)
```

Need to add `fwd_hidden()` functions that return intermediate representations:

**`tribe/resnet.py`** — add `resnet_forward_hidden()`:
```python
def resnet_forward_hidden(weights, X):
    """Forward through ResNet, return pre-FC hidden state."""
    # Same as resnet_forward_batch but stop before FC layer
    h = _stem_forward(weights, X)
    for block in range(4):
        for layer in range(2):
            h = _conv_block_forward(weights, h, f'block{block}_{layer}', ...)
    # Global average pool
    h = mx.mean(h, axis=(1, 2))  # (N, 8*width)
    return h  # Return BEFORE fc_w @ h + fc_b
```

### 4b. Routing Variance Loss

**`tribe/router.py`** — add `variance_loss()`:
```python
def variance_loss(self, X_flat):
    """Maximize variance of routing scores.

    Low variance = router assigns similar probability to all experts (undecided).
    High variance = router is confident about which expert to use.

    L_var = -Var(routing_scores) = -(1/E) * sum_i (P_i - mean(P))^2
    """
    probs = self._probs(X_flat)  # (N, E)
    mean_probs = mx.mean(probs, axis=0)  # (E,) — average routing prob per expert
    variance = mx.mean((mean_probs - mx.mean(mean_probs)) ** 2)
    return -variance  # Negative because we MAXIMIZE variance (minimize this loss)
```

### 4c. Integration Into Training Loop

**`bench_cifar100.py`** — add specialization losses:
```python
def train_step_with_specialization(
    expert_weights_list, X, labels, fwd, fwd_hidden,
    router=None, alpha_orth=0.1, alpha_var=0.01, alpha_bal=0.01
):
    """Training step with specialization losses."""

    def total_loss(all_params):
        # 1. Task loss (standard CE per expert)
        task_loss = ... # as before

        # 2. Orthogonality loss
        hiddens = [fwd_hidden(all_params[f'expert_{i}'], X)
                   for i in range(n_experts)]
        orth_loss = compute_orthogonality(hiddens)

        # 3. Variance loss (if router exists)
        var_loss = 0.0
        if router:
            X_flat = mx.reshape(X, (X.shape[0], -1))
            probs = mx.softmax(X_flat @ all_params['router']['router_w'].T
                              + all_params['router']['router_b'], axis=-1)
            mean_p = mx.mean(probs, axis=0)
            var_loss = -mx.mean((mean_p - mx.mean(mean_p)) ** 2)

        return task_loss + alpha_orth * orth_loss + alpha_var * var_loss

    loss, grads = mx.value_and_grad(total_loss)(all_params)
    return loss
```

### 4d. Monitoring: Expert Overlap Metric

**`tribe/metrics.py`** — add `measure_representation_overlap()`:
```python
def measure_representation_overlap(expert_weights_list, X, fwd_hidden):
    """Compute pairwise cosine similarity of expert hidden representations.

    Returns E×E matrix where entry (i,j) is mean |cos_sim(h_i, h_j)|.
    Diagonal = 1.0. Off-diagonal should decrease with orthogonality loss.
    """
    hiddens = [fwd_hidden(ew, X) for ew in expert_weights_list]
    E = len(hiddens)
    overlap_matrix = np.zeros((E, E))
    for i in range(E):
        for j in range(E):
            hi = np.array(hiddens[i])
            hj = np.array(hiddens[j])
            hi_n = hi / (np.linalg.norm(hi, axis=-1, keepdims=True) + 1e-8)
            hj_n = hj / (np.linalg.norm(hj, axis=-1, keepdims=True) + 1e-8)
            cos = np.mean(np.abs(np.sum(hi_n * hj_n, axis=-1)))
            overlap_matrix[i, j] = cos
    return overlap_matrix
```

## Verification
```bash
# Test orthogonality loss gradient flow
uv run --with mlx python -c "
from tribe.resnet import make_resnet_expert, resnet_forward_hidden
import mlx.core as mx
experts = [make_resnet_expert(seed=i) for i in range(3)]
X = mx.random.normal((4, 32, 32, 3))
# Compute orth loss and verify gradients
"

# Run benchmark with specialization losses
uv run --with mlx python bench_cifar100.py
# Compare: with vs without orthogonality loss
# Expected: overlap matrix off-diagonal decreases 30-45%
```

## Files Modified
| File | Change | Lines |
|------|--------|-------|
| tribe/expert.py | `orthogonality_loss()` | ~25 |
| tribe/resnet.py | `resnet_forward_hidden()` | ~15 |
| tribe/router.py | `variance_loss()` | ~10 |
| tribe/metrics.py | `measure_representation_overlap()` | ~20 |
| bench_cifar100.py | Integrate orth + var losses into training | ~30 |

## Success Criteria
- Expert representation overlap decreases 30%+ (measured via cosine similarity matrix)
- FA improves 2-5% from better specialization
- Routing entropy increases (router becomes more decisive)
- Lifecycle fires LESS (proactive prevention reduces need for reactive cleanup)
