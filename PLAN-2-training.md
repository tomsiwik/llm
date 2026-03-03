# Phase 2: Training Infrastructure (P2 + P4)

## Goal
Boost absolute accuracy from ~29% to 45-55% FA with proper training. Calibrate lifecycle for CE loss.

## Changes

### 2a. Adam Optimizer + Cosine LR Schedule

**`bench_cifar100.py`** — replace manual SGD with MLX Adam:
```python
import mlx.optimizers as optim

def make_optimizer(weights, base_lr=0.001):
    return optim.Adam(learning_rate=base_lr)

def cosine_lr(step, total_steps, base_lr=0.001, min_lr=1e-5):
    """Cosine annealing with warm restart."""
    progress = step / total_steps
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

For each expert, maintain its own Adam optimizer. On `recycle()`, call `reset_optimizer_state()`.

### 2b. Data Augmentation

**`tribe/cifar100.py`** — add `augment_batch()`:
```python
def augment_batch(images, rng=None):
    """Random crop (pad 4, crop 32x32) + horizontal flip."""
    if rng is None:
        rng = np.random
    N, H, W, C = images.shape
    # Pad 4 pixels on each side
    padded = np.pad(images, ((0,0), (4,4), (4,4), (0,0)), mode='reflect')
    # Random crop back to 32x32
    crops = np.empty_like(images)
    for i in range(N):
        y = rng.randint(0, 9)  # 0 to 8 (padded is 40x40)
        x = rng.randint(0, 9)
        crops[i] = padded[i, y:y+32, x:x+32, :]
    # Random horizontal flip (50%)
    flip_mask = rng.random(N) > 0.5
    crops[flip_mask] = crops[flip_mask, :, ::-1, :]
    return crops
```

**`bench_cifar100.py`** — use augmentation in training loop:
```python
# Each training step:
idx = rng.choice(len(train_data), batch_size, replace=False)
X_batch = augment_batch(train_data.images[idx])
X_batch = mx.array(X_batch)
labels_batch = mx.array(train_data.labels[idx])
```

### 2c. Increase Training Budget

- Steps per task: 500 → 2000
- Batch size: 64 (unchanged, fits in memory)
- Total training time estimate: ~50min → ~3.5hrs on M-series Mac

For faster iteration, add a `--quick` flag:
```python
STEPS_PER_TASK = 500 if '--quick' in sys.argv else 2000
```

### 2d. Calibrate Lifecycle Thresholds for CE Loss (P4)

**Problem**: health_check uses `grad_norm < 1e-5` calibrated for MSE. CE loss has different gradient magnitudes.

**Fix**: Use relative thresholds instead of absolute:

**`tribe/core.py`** — update `health_check()`:
```python
def health_check(self, overlap_threshold=0.5, min_active=2,
                 freeze_grad_threshold=None, competence_threshold=None,
                 loss_fn=None):
    """If freeze_grad_threshold is None, use relative: grad < 0.01 * mean_grad."""
    all_grad_norms = []
    for m in self.active_members():
        if m.domain and len(m.domain) >= 3:
            # ... compute grad_norm as before ...
            all_grad_norms.append((m, grad_norm))

    if freeze_grad_threshold is None and all_grad_norms:
        # Relative threshold: freeze if grad < 1% of mean
        mean_grad = sum(g for _, g in all_grad_norms) / len(all_grad_norms)
        freeze_grad_threshold = mean_grad * 0.01

    for m, grad_norm in all_grad_norms:
        if grad_norm < freeze_grad_threshold and n_active > min_active:
            recommendations.append(('freeze', m.id, ...))
```

**`bench_cifar100.py`** — pass CE-appropriate thresholds:
```python
def _lifecycle(tribe, members, fwd, generation):
    recs = tribe.health_check(
        overlap_threshold=0.3,           # lower for CE (more sensitive)
        freeze_grad_threshold=None,      # use relative (auto-calibrate)
        competence_threshold=0.5,        # CE loss < 0.5 = competent
    )
```

Also add a recycle trigger based on relative domain size AND loss:
```python
# Recycle: expert has < 10% of avg domain AND worst loss
avg_domain = mean(len(m.domain) for m in active)
for m in active:
    if len(m.domain) < 0.1 * avg_domain:
        # Only recycle if also underperforming
        if m.fitness() < median_fitness:
            recycle(m.id, optimizer=optimizers[m.id])
```

## Verification
```bash
# Quick mode (~50 min)
uv run --with mlx python bench_cifar100.py --quick

# Full mode (~3.5 hrs)
uv run --with mlx python bench_cifar100.py

# Expected: FA 45-55% for tribe methods with Adam+aug
# Expected: Lifecycle fires more frequently (5-8 events vs 4)
```

## Files Modified
| File | Change | Lines |
|------|--------|-------|
| tribe/cifar100.py | `augment_batch()` | ~20 |
| tribe/core.py | Relative threshold in `health_check()` | ~15 |
| bench_cifar100.py | Adam optimizer, cosine LR, augmentation, 2000 steps | ~80 |

## Success Criteria
- FA > 40% for lifecycle tribe (up from 29%)
- Lifecycle fires 5+ events across 10 tasks
- Fine-tune baseline also improves (to ~15-20%) confirming infrastructure improvement
- Total training time < 4 hours on M-series Mac
