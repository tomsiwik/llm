# Phase 1: Foundation Fixes (P0 + P3)

## Goal
Make results honest and validate ReDo contribution. Two quick fixes that unlock everything else.

## Changes

### 1a. Max-Confidence Routing at Test Time (P0)

**Problem**: `measure_accuracy_tribe()` in `bench_cifar100.py` uses oracle routing — picks expert with lowest CE loss on the *correct label*. This is cheating in class-incremental evaluation.

**Fix**: Add max-confidence routing — pick expert whose softmax output has the highest max probability. No ground truth needed.

**`bench_cifar100.py`** — add `measure_accuracy_tribe_maxconf()`:
```python
def measure_accuracy_tribe_maxconf(tribe_members, test_data, fwd):
    """Route by max softmax confidence — no oracle labels needed."""
    X, labels = test_data.as_mx_labels()
    all_confs = []
    all_preds = []
    for mid, w in tribe_members:
        logits = fwd(w, X)
        probs = mx.softmax(logits, axis=-1)
        max_conf = mx.max(probs, axis=-1)          # (N,)
        pred_class = mx.argmax(logits, axis=-1)     # (N,)
        all_confs.append(max_conf)
        all_preds.append(pred_class)
    # For each sample, pick expert with highest confidence
    confs = mx.stack(all_confs, axis=0)             # (E, N)
    preds = mx.stack(all_preds, axis=0)             # (E, N)
    best_expert = mx.argmax(confs, axis=0)          # (N,)
    final_preds = mx.take_along_axis(preds, mx.expand_dims(best_expert, 0), axis=0)[0]
    correct = mx.sum(final_preds == mx.array(labels))
    mx.eval(correct)
    return correct.item() / len(labels)
```

**`bench_cifar100.py`** — also add learned router evaluation:
```python
def measure_accuracy_tribe_learned(router, tribe_members, test_data, fwd):
    """Route using the learned SwitchRouter at test time."""
    X, labels = test_data.as_mx_labels()
    X_flat = mx.reshape(X, (X.shape[0], -1))
    assignments, _, _ = router.route(X_flat)
    # For each sample, use assigned expert's prediction
    preds = mx.zeros((len(labels),), dtype=mx.int32)
    for expert_idx, sample_indices in assignments.items():
        if not sample_indices:
            continue
        idx = mx.array(sample_indices)
        X_sub = X[idx]
        mid, w = tribe_members[expert_idx]
        logits = fwd(w, X_sub)
        expert_preds = mx.argmax(logits, axis=-1)
        # scatter into preds
        for i, si in enumerate(sample_indices):
            preds = preds.at[si].add(expert_preds[i] - preds[si])
    correct = mx.sum(preds == mx.array(labels))
    mx.eval(correct)
    return correct.item() / len(labels)
```

**Report both**: Oracle, Max-Confidence, and Learned Router accuracy for all tribe methods.

### 1b. Integrate warmup_scale Into Training (P3)

**Problem**: `TribeMember.warmup_scale` computes 0→1 ramp but no training loop reads it.

**Fix**: The training function should scale the loss by `warmup_scale` for recycled experts.

**`tribe/expert.py`** — modify `train()` to accept optional `member`:
```python
def train(weights, patterns, steps=300, lr=0.02, fwd=None, member=None):
    fwd = fwd or forward_batch
    X = mx.stack([x for x, _ in patterns])
    T = mx.stack([t for _, t in patterns])
    for _ in range(steps):
        scale = member.warmup_scale if member and member.warmup_remaining > 0 else 1.0
        def loss_fn(w):
            preds = fwd(w, X)
            return scale * mx.mean((preds - T) ** 2)
        loss, grads = mx.value_and_grad(loss_fn)(weights)
        for k in weights:
            weights[k] = weights[k] - lr * grads[k]
        mx.eval(*weights.values())
        if member and member.warmup_remaining > 0:
            member.warmup_remaining -= 1
            member.age += 1
    return weights
```

**`bench_cifar100.py`** — in lifecycle tribe training loop, pass member to training:
```python
# During training step for each expert
scale = member.warmup_scale if member.warmup_remaining > 0 else 1.0
# Scale loss contribution
loss = scale * cross_entropy_loss(logits, labels_batch)
```

## Verification
```bash
# Run bench with both oracle and max-conf reporting
uv run --with mlx python bench_cifar100.py

# Expected: max-conf accuracy < oracle accuracy
# Expected: learned router accuracy ≈ max-conf accuracy
# Warmup: recycled experts should show gradual output ramp
```

## Files Modified
| File | Change | Lines |
|------|--------|-------|
| bench_cifar100.py | Add max-conf + learned router eval, report 3 routing modes | ~60 |
| tribe/expert.py | Add `member` param to `train()`, apply warmup_scale | ~10 |

## Success Criteria
- All 3 routing modes reported for tribe methods
- Max-conf accuracy is lower than oracle but > fine-tune baseline
- warmup_scale decrements correctly during recycled expert training
