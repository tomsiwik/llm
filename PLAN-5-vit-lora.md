# Phase 5: ViT-B/16 + LoRA Expert Scaling (P5)

## Goal
Scale LGME to pretrained backbone. Target: match CODA-Prompt (~86%) on CIFAR-100 with better forgetting. This is where we become SOTA-competitive.

## Background

Current SOTA on class-incremental CIFAR-100:
- CODA-Prompt: 86.11% (ViT-B/16 pretrained)
- SEMA: 86.98% (ViT-B/16 pretrained, CVPR 2025)
- L2P: 77.87% (ViT-B/16 pretrained)
- DualPrompt: 80.43% (ViT-B/16 pretrained)

All use pretrained ViT-B/16 (ImageNet-21K) + lightweight task-specific adaptation. Our lifecycle operates on adapter weights — same dict interface, much fewer params per expert.

## Architecture

### 5a. Pretrained ViT Backbone (Frozen)

Use `mlx-community/vit-base-patch16-224` or download ImageNet-pretrained ViT-B/16 weights.

**`tribe/vit.py`** — new file:
```python
class ViTBackbone:
    """Frozen ViT-B/16 feature extractor."""

    def __init__(self, weights_path):
        self.weights = load_pretrained_vit(weights_path)
        # Freeze all backbone weights
        # ViT-B/16: 12 layers, d=768, 86M params

    def extract_features(self, X):
        """(N, 224, 224, 3) → (N, 768) CLS token features."""
        # Resize CIFAR 32x32 → 224x224 (bilinear interpolation)
        # Or use ViT-B/16-224 with 224x224 input
        # Patch embedding → 12 transformer blocks → CLS token
        return cls_features  # (N, 768)

    def extract_patch_features(self, X):
        """(N, 224, 224, 3) → (N, 197, 768) all patch tokens."""
        return patch_features
```

**CIFAR-100 preprocessing**: Resize 32×32 → 224×224 (bicubic), normalize with ImageNet stats.

### 5b. LoRA Adapters as Experts

Each expert is a set of LoRA adapters applied to the ViT's attention layers.

**`tribe/lora.py`** — new file:
```python
def make_lora_expert(rank=8, num_layers=12, d_model=768, seed=0):
    """Create LoRA adapter weights for one expert.

    Adapts Q and V projections in each transformer layer.
    Total params per expert: 2 * num_layers * 2 * rank * d_model
    = 2 * 12 * 2 * 8 * 768 = 294,912 (~295K params per expert)
    """
    mx.random.seed(seed)
    weights = {}
    for layer in range(num_layers):
        for target in ['q_proj', 'v_proj']:
            # LoRA: W = W_base + A @ B where A: (d, r), B: (r, d)
            weights[f'layer{layer}.{target}.lora_A'] = mx.random.normal((d_model, rank)) * (1.0 / rank)
            weights[f'layer{layer}.{target}.lora_B'] = mx.zeros((rank, d_model))  # zero-init
    return weights

def lora_forward(base_weights, lora_weights, X, scale=1.0):
    """Forward pass: base_output + scale * (X @ A @ B)."""
    # For each adapted layer:
    # output = base_linear(X) + scale * (X @ lora_A @ lora_B)
    pass

def param_count_lora(weights):
    return sum(w.size for w in weights.values())
    # ~295K per expert, 10 experts = ~2.95M adapter params
    # + 86M frozen backbone = ~89M total but only 2.95M trainable
```

### 5c. Lifecycle on LoRA Adapters

The lifecycle operates identically — `recycle()`, `freeze()`, `bond()`, `distill()` all work on the adapter weight dict:

```python
# bond() — blend two LoRA adapter sets
child_lora = blend_weights([parent_a_lora, parent_b_lora], [0.5, 0.5])
# recycle() — reinitialize LoRA adapters
new_lora = make_lora_expert(rank=8, seed=...)
# freeze() — stop adapter training, continue routing
# distill() — match teacher adapter outputs on probe inputs
```

Key difference: `warmup_scale` now scales the LoRA output:
```python
# During forward:
lora_output = scale * warmup_scale * (X @ lora_A @ lora_B)
# warmup_scale ramps 0→1, so recycled expert starts contributing nothing
# This is EXACTLY ReDo's "outgoing weights to zero" principle
```

### 5d. Training Loop

```python
def train_expert_lora(backbone, lora_weights, train_data, optimizer,
                      steps=200, batch_size=32):
    """Train one LoRA expert on its routed data."""
    for step in range(steps):
        X_batch, labels = sample_batch(train_data, batch_size)
        features = backbone.extract_features(X_batch)  # frozen, no grad

        def loss_fn(lora_w):
            # Apply LoRA to get adapted features
            adapted = apply_lora(backbone.weights, lora_w, X_batch)
            logits = classifier(adapted)  # shared or per-expert classifier
            return cross_entropy(logits, labels)

        loss, grads = mx.value_and_grad(loss_fn)(lora_weights)
        optimizer.apply_gradients(grads, lora_weights)
```

### 5e. Evaluation

Report all 3 routing modes:
1. Oracle (loss-based, for comparison with our prior results)
2. Max-confidence (softmax argmax, label-free)
3. Learned router (SwitchRouter on CLS features)

### 5f. Implementation Options for ViT

**Option A: Pure MLX** — Port ViT-B/16 to MLX from scratch using `mlx.nn`.
- Pro: Full control, consistent with codebase
- Con: Must handle weight loading from PyTorch/HF checkpoints

**Option B: Use `mlx-lm` or `mlx-image`** — If available, use existing MLX ViT.
- Pro: Fast to get running
- Con: May not expose internals needed for LoRA injection

**Option C: Hybrid** — Use HF `timm` for feature extraction, MLX for LoRA + lifecycle.
- Pro: Proven ViT implementation
- Con: Mixing frameworks

**Recommendation**: Option A. Write a minimal ViT-B/16 in MLX (~200 lines), load HF weights. We need full control for LoRA injection and lifecycle integration.

## Verification
```bash
# 1. ViT feature extraction works
uv run --with mlx python -c "
from tribe.vit import ViTBackbone
backbone = ViTBackbone('weights/vit-b16.npz')
X = mx.random.normal((2, 224, 224, 3))
features = backbone.extract_features(X)
assert features.shape == (2, 768)
"

# 2. LoRA training converges
uv run --with mlx python -c "
from tribe.lora import make_lora_expert, param_count_lora
lora = make_lora_expert(rank=8)
print(f'Params per expert: {param_count_lora(lora):,}')
# Train on 1 task, verify loss decreases
"

# 3. Full benchmark
uv run --with mlx python bench_cifar100_vit.py
# Target: FA > 80% with lifecycle, forgetting < 5%
```

## Files Created/Modified
| File | Change | Lines |
|------|--------|-------|
| tribe/vit.py | **New**: ViT-B/16 backbone in MLX | ~250 |
| tribe/lora.py | **New**: LoRA adapter creation, forward, param counting | ~100 |
| bench_cifar100_vit.py | **New**: ViT+LoRA benchmark runner | ~300 |
| tribe/cifar100.py | Add 224×224 resize + ImageNet normalization | ~15 |

## Success Criteria
- FA > 80% with lifecycle (competitive with L2P 77.87%)
- FA > 85% target (competitive with CODA-Prompt 86.11%)
- Forgetting < 5% (lifecycle advantage over static)
- Total trainable params: ~3M (10 LoRA experts) + 86M frozen backbone
- Training time: < 2 hours on M-series Mac (LoRA is fast)
