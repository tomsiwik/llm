# The Room Model: Geometric Routing Through Orthogonal Adapter Walls

## The Idea (plain language)

The base model is the floor of a room (ternary lattice).
Each domain adapter is a wall, orthogonal to the floor and to each other.
Each base weight casts a shadow onto each wall — the shadow IS the adaptation.

Routing is not a separate step. A token's hidden state has a DIRECTION in this room.
Its dot product with each wall = how much that domain contributes.
One matmul does both routing AND adaptation simultaneously.

```
         ceiling (science)
            │
   medical ─┼── code
            │
          floor (base model)

Token "dissolve the sodium bicarbonate" points toward:
  cooking: 0.8  (recipe context)
  science: 0.5  (chemistry)
  medical: 0.1  (biochem overlap)
  code:    0.0  (irrelevant)

These weights fall out of the geometry. No router needed.
```

## The Math

### Setup (one-time, when adapters are installed)

**Step 1: Each adapter lives on an orthogonal wall.**

The Grassmannian initialization gives us orthogonal A-matrices:
  A_i ∈ R^{d × r}, where A_i^T A_j = 0 for i ≠ j

Each A_i defines a wall in d-dimensional space.
The B_i (trained adapter) is the shadow painted on that wall.

**Step 2: Stack all walls into one matrix.**

Instead of separate (A_i, B_i) pairs applied sequentially, pre-compute:

  ΔW_i = α · B_i^T @ A_i^T          (per-adapter delta, d_out × d_in)

Stack all deltas:

  W_room = Σ_i ΔW_i = α · Σ_i B_i^T @ A_i^T

Because A_i ⊥ A_j, the deltas live in orthogonal subspaces.
Their sum is non-interfering by construction.

**Step 3: Pre-merge into one matrix.**

  W_combined = W_room                 (one d_out × d_in matrix)

This is computed ONCE. The cost is N matrix multiplications at setup time.

### Inference (every token, fast)

  y = BitLinear(x) + x @ W_combined   ← ONE extra matmul for ALL domains

The matmul x @ W_combined simultaneously:
1. Projects x onto each adapter wall (routing)
2. Applies each wall's shadow (adaptation)
3. Sums the contributions (composition)

All in one operation. No routing step. No per-adapter dispatch.

### Why routing is automatic

The hidden state x has a direction in d-dimensional space.
For a medical token, x aligns with the medical wall (A_medical).
The projection x @ W_combined picks up mostly ΔW_medical because:

  x @ W_combined = x @ Σ_i α · B_i^T @ A_i^T
                 = Σ_i α · (x @ A_i^T)^T @ B_i^T      (by linearity, conceptually)
                 ≈ α · (x @ A_medical^T)^T @ B_medical^T  (dominant term)

Because x is most aligned with A_medical, the medical wall's contribution
dominates. Other walls contribute proportionally to their alignment with x.

This IS soft routing by cosine similarity — but it falls out of the matmul
for free, not as a separate computation.

## Complexity

### Setup cost (one-time)
- Per adapter: B_i^T @ A_i^T = O(d² · r) matmul
- Sum N adapters: O(N · d² · r)
- For d=2560, r=16, N=24: ~2.5 GFLOP = milliseconds on M5 Pro

### Per-token inference cost
- Base: BitLinear(x) = one ternary matmul (native Metal kernel)
- Adapter: x @ W_combined = ONE bf16 matmul
- Total: 2 matmuls per module instead of 2N

### Dispatches
- Current v3: 420 dispatches (210 modules × 2 per module)
- Room model: 210 dispatches (210 modules × 1 combined matmul)
- With v6 QKV concat: ~60 dispatches (4 groups × 30 layers × ~0.5)

### Memory
- W_combined: d_out × d_in per module = same as one full-rank delta
- For all 210 modules: ~210 × 2560 × 2560 × 2 bytes = ~2.7 GB bf16
- Fits in 48GB with room to spare

## Key Properties

### 1. Zero routing overhead
Routing IS the matmul. No separate router, no calibration data, no ridge regression.
The token's direction in hidden space automatically picks the right domain mix.

### 2. Orthogonality guarantees non-interference
Because A_i ⊥ A_j, each wall's contribution is independent.
Adding a new wall (domain 25) doesn't affect existing walls.
Proof: ΔW_new ⊥ ΔW_existing by Grassmannian construction.

### 3. Continuous composition in one matrix
W_combined = Σ ΔW_i lives in continuous bf16 space.
No ternary merging (which we proved impossible).
The base stays ternary. The combined adapter is one continuous matrix.

### 4. Adding a new domain = one matrix addition
To add domain N+1:
1. Compute ΔW_{N+1} = α · B_{N+1}^T @ A_{N+1}^T
2. W_combined += ΔW_{N+1}
3. Done. Milliseconds. No retraining.

### 5. Removing a domain = one matrix subtraction
W_combined -= ΔW_i
Exact reversal. No degradation.

## Connection to Game Engine Algorithms

| Game engine | Room model | GPU primitive |
|-------------|-----------|---------------|
| Pre-baked lightmap | Pre-computed W_combined | Static texture |
| Lightmap = Σ light_i contributions | W_combined = Σ ΔW_i | Matrix addition |
| Real-time shadow = dot(normal, light_dir) | Routing = dot(x, adapter_wall) | Dot product |
| Texture atlas (pack all textures) | W_combined (pack all adapters) | Single memory block |
| Terrain splatting blend weights | Cosine similarity routing weights | Same dot product |
| Deferred rendering (compute once) | Pre-compute combined delta | Same batch pattern |

The pre-baked lightmap analogy is exact:
- Game: compute lighting offline, bake into one texture, sample at runtime
- Pierre: compute adapter deltas offline, sum into one matrix, matmul at runtime
- Both: the expensive work is done once, runtime is a single lookup/matmul

## What This Changes About Pierre

### Current architecture (v3):
```python
# Per query:
domain_idx = route(model, tok, text, W)    # separate routing step
attach_adapter(model, frozen_A, B_i, ...)  # attach ONE adapter
y = model(x)                               # inference with one adapter
detach_adapters(model)                     # cleanup
```

### Room model:
```python
# One-time setup:
W_combined = sum(alpha * B_i.T @ A_i.T for i in range(N))
inject_combined(model, W_combined)

# Per query:
y = model(x)    # that's it. routing + adaptation + composition = one matmul
```

No router. No per-query adapter selection. No dispatch overhead per adapter.
The geometry of the orthogonal walls does everything.

## Theoretical Capacity

At d=2560 with rank r=16:
- Maximum orthogonal walls: d/r = 160 domains
- Each wall's shadow: 16 dimensions of domain knowledge
- W_combined: 2560×2560 = ~13MB per module, ~2.7GB total

At d=2560 with rank r=8 (half rank):
- Maximum orthogonal walls: 320 domains
- Each wall's shadow: 8 dimensions (less expressive per domain)
- Trade-off: more domains vs less per-domain capacity

## Proven Components (what we already have)

- Grassmannian A-matrices: proven orthogonal at cos < 0.001 (Finding #3)
- SFT adapter B-matrices: proven quality at 0.41 behavioral (Finding #288)
- NRE composition: proven equivalent to Karcher mean (Finding #275)
- Ridge router: 99.6% accuracy (Finding #287) — but Room model doesn't need it

## What Needs to Be Proven (the experiment)

1. Does W_combined = Σ ΔW_i actually produce correct multi-domain output?
   (Theory says yes — orthogonal deltas sum without interference)

2. Does the automatic routing (token direction → wall alignment) match
   the explicit router's accuracy?
   (Theory says it should — same cosine similarity, different computation path)

3. What's the actual tok/s with the Room model?
   (Should be ~130+ tok/s: native base + one extra bf16 matmul per module)

4. Does this hold at N=24? N=50? N=160?
   (Grassmannian guarantees up to d/r = 160, but need empirical verification)

## Open Questions

1. The automatic routing is SOFT (proportional to cosine). Current router
   is HARD (top-1 selection). Does soft routing match or beat hard routing?

2. W_combined is a dense d×d matrix even though each ΔW_i is rank-r.
   The sum of N rank-r matrices has rank ≤ N·r. At N=24, r=16: rank 384.
   Can we exploit this low-rank structure for speed?

3. When does the orthogonal assumption break down? At what N do the
   Grassmannian walls become "too crowded" and start interfering?

4. Can we borrow fast inverse sqrt or other game engine approximations
   for the projection computation?
