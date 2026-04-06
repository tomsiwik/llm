# Parallel Experiment Plan: Room Model + SHINE Integration

## The Four Independent Pieces

Each piece proves one thing. All can run in parallel. Then they consolidate.

```
Piece A: Room Model works         Piece B: Geometric routing works
  (sum adapters → one matmul)       (token direction = domain selection)
         ↘                                    ↙
          ↘                                  ↙
           Piece AB: Room Model is complete
                        ↓
Piece C: SHINE replication         Piece D: SHINE→Room bridge
  (understand & port to MLX)         (context-aware wall modulation)
         ↘                                    ↙
          ↘                                  ↙
           Piece CD: Dynamic Room Model
                        ↓
              FINAL: Unified system
```

---

## Piece A: Room Model W_combined (can run NOW)

**What to prove:** Summing N orthogonal adapter deltas into one matrix
preserves per-domain quality.

**Uses:** Existing SFT adapters (5 domains), existing Grassmannian skeleton.
No new training. No new code beyond ~50 lines.

**Test:**
```
W_combined = Σ α · B_i^T @ A_i^T     (precompute, milliseconds)
inject W_combined as one matrix per module
for each domain:
  feed domain text → measure PPL
  compare against single-adapter PPL
```

**Pass criteria:**
- Each domain's PPL within 20% of its single-adapter PPL
- Speed: only 210 dispatches (one per module), target >100 tok/s

**Why this works (theory):**
- A_i ⊥ A_j (Grassmannian, cos < 0.001)
- Therefore ΔW_i ⊥ ΔW_j in weight space
- Sum of orthogonal vectors = no interference
- Each domain's signal passes through independently

**Estimated time:** 30 minutes (mostly model loading)

---

## Piece B: Geometric Routing (can run NOW, parallel with A)

**What to prove:** The token's hidden state direction naturally aligns
with the correct adapter wall, matching explicit routing.

**Uses:** Same SFT adapters, same skeleton. Plus the ridge regression
router W from earlier experiments for comparison.

**Test:**
```
for each test text with known domain:
  h = encode(model, text)                    # hidden state
  # Explicit routing (what we do now):
  explicit_domain = argmax(h @ router_W)
  # Geometric routing (what Room Model does):
  projection_per_wall = [‖h @ A_i^T‖ for each domain i]
  geometric_domain = argmax(projections)
  # Compare
```

**Pass criteria:**
- Geometric routing agrees with explicit routing >80% of the time
- OR geometric routing achieves >80% accuracy directly

**Why this works (theory):**
- The ridge router already uses h @ W where W = (X^TX+λI)^{-1}X^TY
- The geometric projection uses h @ A_i^T
- If the router's learned W is close to the Grassmannian A structure,
  they should agree

**Estimated time:** 20 minutes

---

## Piece C: SHINE Architecture Study (parallel with A+B)

**What to understand:** SHINE's core mechanism — how it maps context
to adapter weights in one pass.

**Approach:** 
1. Clone repo, read metanetwork_family.py and LoraQwen.py
2. Extract the M2P Transformer architecture (the lightweight transformer
   that converts memory states to LoRA weights)
3. Identify which parts are Qwen3-specific vs architecture-general
4. Port the M2P concept to MLX (not the full SHINE — just the generator)

**Key components to extract:**
- Memory extraction: hidden states from all L layers → stacked tensor (L, M, H)
- M2P Transformer: alternating row/column attention on (L, M, H)
- Parameter generation: reshape M2P output → LoRA A and B matrices

**What we DON'T need from SHINE:**
- Their pretraining pipeline (6B tokens)
- Their specific model (Qwen3-8B)
- Their evaluation setup

**What we DO need:**
- The M2P Transformer architecture (portable, ~100 lines)
- The memory extraction pattern (collect hidden states from all layers)
- How they reshape outputs to LoRA weights

**Estimated time:** 2-3 hours (reading + porting)

---

## Piece D: SHINE→Room Bridge Design (after A+B+C)

**What to design:** Use SHINE's context-aware mechanism to dynamically
modulate the Room Model's W_combined.

**The idea:**
```
STATIC Room Model (Piece A):
  W_combined = Σ ΔW_i                    (fixed, pre-baked)
  y = base(x) + x @ W_combined           (same mix for every token)

DYNAMIC Room Model (Piece D):
  W_combined(context) = Σ α_i(context) · ΔW_i   (context-aware weights)
  y = base(x) + x @ W_combined(context)          (adapted per query)
```

The α_i(context) weights come from a SHINE-like mechanism:
1. Feed the context through the base model
2. Extract memory states from all layers
3. A lightweight M2P network outputs mixing weights α_i per domain
4. W_combined is re-weighted (milliseconds — just scalar × matrix)

**This differs from full SHINE:**
- SHINE generates entire A and B matrices from scratch (expensive, fragile)
- We generate only N scalar weights α_i (cheap, robust)
- Our adapters are pre-trained and proven (quality guaranteed)
- SHINE's adapters are generated on-the-fly (quality varies)

**Estimated time:** Design only (no code until A+B+C prove out)

---

## How They Consolidate

### Phase 1: Static Room Model (Pieces A + B)
```
W_combined = Σ ΔW_i
y = base(x) + x @ W_combined
```
If A passes: multi-domain quality from one matmul.
If B passes: routing is free (no separate router).
Result: fast, simple, but all domains always active.

### Phase 2: SHINE-Aware Room Model (Phase 1 + C + D)
```
α = SHINE_lite(context)        # lightweight: just N scalars
W_context = Σ α_i · ΔW_i       # context-weighted sum
y = base(x) + x @ W_context    # adapted per query
```
The SHINE-lite module tells the Room Model how to mix walls
for this specific query. Only adds one lightweight forward pass
at the START of generation (not per token).

### Phase 3: Training-Free Adaptation (future)
Full SHINE-style: generate entirely new adapter walls from
novel contexts without any training. But built on top of the
Room Model's proven orthogonal geometry.

---

## What Can Be Tested on Toy Model vs Real Model

| Piece | Toy model? | Real model? | Why |
|-------|-----------|-------------|-----|
| A: W_combined sum | ✓ (micro/models/gpt/) | ✓ (BitNet-2B) | Both work. Real model gives real numbers. |
| B: Geometric routing | ✗ (need real domain data) | ✓ | Need actual domain-separated text. |
| C: SHINE port | ✓ (test on small transformer) | Later | Architecture port, not quality eval. |
| D: Bridge design | Paper only | Later | Design first, build after A+B+C. |

**Recommendation:** Run A and B on the real BitNet-2B model with existing
SFT adapters. They use what we already have. Run C as a code study of the
SHINE repo in parallel. No toy model needed — we have real adapters.

---

## Experiment Registration

### exp_room_model_wcombined (Piece A)
Priority: P0
Kill: any domain PPL > 2x single-adapter | speed < 100 tok/s
Est: 30 min

### exp_room_model_geometric_routing (Piece B)  
Priority: P0 (parallel with A)
Kill: geometric routing accuracy < 60% | disagrees with ridge router > 50%
Est: 20 min

### exp_shine_architecture_study (Piece C)
Priority: P1 (parallel with A+B)
Kill: core architecture not portable to MLX | depends on Qwen-specific features
Est: 2-3 hours

### exp_room_model_dynamic (Piece D)
Priority: P2 (after A+B+C)
Kill: context-weighted W_combined worse than static W_combined
Est: TBD
