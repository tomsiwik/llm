# Pierre Architecture: Self-Growing Model Through Adapter Promotion

## The Vision

A frozen pre-trained base model grows through accumulated adapter promotions.
Domain adapters — generated via M2P or SFT — compose without interference
thanks to Grassmannian orthogonality, and the best get promoted into the base.

```
pre-trained base (Qwen3-4B or similar)
  → users generate sessions
    → M2P generates session adapters from context (1.15ms)
      → sessions crystallize into domain adapters (SFT, 300 steps)
        → best adapters get promoted into base (scale=5, 0pp MMLU)
          → base grows stronger
            → new adapters train on stronger base
              → repeat
```

## The Architecture

### Four Tiers of Knowledge

```
┌─────────────────────────────────────────────────────────┐
│ Tier 4: Session Adapters (ephemeral, per-conversation)  │
│   Generated from context in one M2P forward pass.       │
│   Lives only during the session. ~1KB-10KB.             │
│   66.6% of SFT quality (Finding #339).                  │
├─────────────────────────────────────────────────────────┤
│ Tier 3: User Adapters (persistent, per-user)            │
│   Distilled from accumulated sessions via M2P.          │
│   "This user prefers functional React, avoids useEffect"│
│   Updated after each session. ~100KB.                   │
├─────────────────────────────────────────────────────────┤
│ Tier 2: Domain Adapters (shared, SFT-trained)           │
│   "Modern React", "Medical Billing", "Rust Async"       │
│   Crystallized from 100s of users. ~1-10MB.             │
│   Candidates for PROMOTION to base.                     │
├─────────────────────────────────────────────────────────┤
│ Tier 1: Base (frozen + promoted adapters)                │
│   Started as pre-trained Qwen3-4B.                      │
│   Grows through promotion at scale=5 (0pp MMLU, #333).  │
│   Each promotion adds a solidified expert permanently.  │
└─────────────────────────────────────────────────────────┘
```

### The Promotion Cycle

```
Train adapter via SFT (LoRA, scale=5, 300 steps)
  ↓
Deploy as composable expert (PHATGOOSE-style routing)
  per-token softmax routing (Finding #28: matches oracle at N=24)
  Grassmannian A-matrices prevent interference (Finding #3: cos=0.0002)
  ↓
Monitor usage patterns
  universal expert? → promote to base
  rarely used? → prune (LOO removes 20.8% safely, Finding #39)
  conflicting? → retrain
  ↓
Promote: base_new = base + scale*adapter (scale=5)
  0pp MMLU degradation for single promotion (Finding #333)
  ↓
Repeat (HYPOTHESIZED — multi-cycle untested, see below)
```

**Important:** SVD solidification before promotion is KILLED (#329) — it breaks
Grassmannian structure. Simple scale reduction (scale=5) is the correct approach.

**Important:** Self-growing from random init is KILLED (#331) — needs pre-trained seed.
Promotion works ON TOP of a pre-trained base (#333).

**Honest status of promotion cycle:**
- Single-cycle promotion: DEMONSTRATED (#333, 0pp MMLU, scale=5)
- Multi-cycle promotion: UNTESTED (Level 4A in PoC roadmap)
- Promotion from random init: KILLED (#331, catastrophic at step 3)
- Bulk dissolve of 10 adapters: DESTRUCTIVE (#353, parity 6.3x regression)
- Safe dissolve strategies: DESIGNED, awaiting experiment (Level 1A)

### The Decoupled Guarantee

Three independent properties, with different evidence levels:

| Property | Mechanism | Evidence Level |
|----------|-----------|---------------|
| Parameter-space orthogonality | Frozen Grassmannian A (QR) | **Mathematical proof:** A_i^T A_j = 0 |
| Activation-space interference | Depends on B-matrix alignment | **Empirical:** max\|cos\|=0.29 at N=5, no bound |
| Domain quality | M2P generates B from context | **Empirical:** 97.7-100.6% at toy scale |
| Scale | Preservation loss or fixed scale=5 | **Empirical:** 0pp MMLU at scale=5 |

```
⟨Δ_i, Δ_j⟩_F = trace(B_j (A_j A_i^T) B_i^T) = 0  because A_j A_i^T = 0
```

This guarantees zero **parameter-space** interference. Activation-space interference
(B_i(A_i·x) vs B_j(A_j·x) in the output) is empirically small but unbounded.

### M2P Distillation Paths

**Path A: Context → Adapter (instant domain expertise)**
```
context text → base model encodes → M2P reads hidden states → generates B
Result: domain adapter in 1.15ms (Finding #339: 66.6% of SFT quality)
```

**Path B: Teacher → Adapter (model distillation)**
```
Teacher (Qwen3-8B) processes domain → hidden states
M2P reads teacher states → generates B that makes student ≈ teacher
```

**Path C: Sessions → User Adapter (personalization)**
```
Accumulated session adapters → M2P distills into persistent user adapter
Each session refines the user's expertise profile
```

## Mathematical Foundation

### Why Promotion Works (proven)

1. **Adapter promotion ≡ continued pre-training.**
   ReLoRA (Finding #86): merge LoRA → reset → retrain → converges to same quality.

2. **Promoted adapters don't interfere with future adapters.**
   Grassmannian orthogonality (Finding #3, cos=0.0002). Each expert occupies
   its own subspace. Promotion at scale=5 preserves this (#333).

3. **Per-token routing matches oracle quality.**
   Softmax router at N=24: 0% gap to oracle, 0% fallback (#28).
   Hidden states linearly separable at 98.3% (#310).

4. **Scale=5 is the safe operating point.**
   0pp MMLU degradation under composition (#330).
   Preservation loss can teach M2P to self-calibrate.

### Capacity

At d=3584 (Qwen3-4B) with rank r=16:
  - Max orthogonal adapters: 224 simultaneous
  - Memory: N_max=853 on M5 Pro 48GB
  - At N=500: 23.86 GB (59.6% of budget)

### Open Problem: M2P Multi-Domain Training

M2P B-matrices collapse to centroid when trained on multiple domains simultaneously
(Finding #341). Domains with low base loss get adapters calibrated for hard tasks.
Additive domain conditioning improves median to 47.3% but doesn't break the
centroid for easy domains (#342).

Fixes under investigation:
- Multiplicative gating (not additive)
- Per-domain loss normalization
- Separate M2P heads
- Single-domain training + composition evaluation

## Supporting Research

| Paper | What it proves | How we use it |
|-------|---------------|---------------|
| **SHINE** (arXiv:2602.06358) | Context → adapter in one forward pass | Session adapter generation |
| **ReLoRA** (arXiv:2307.05695) | Repeated LoRA merge = full pre-training | Promotion is mathematically sound |
| **PHATGOOSE** (post-hoc gating) | Independent experts compose zero-shot | Our routing architecture |
| **CAMEL** (autonomous expert tuner) | Expert freezing + drift detection + pruning | Lifecycle management |
| **ScaleZero** (DPS strategy) | Progressive adapter → permanent expert | Promotion mechanism |
| **MINGLE** (null-space gating) | SVD experts compose with null-space constraint | Related approach (we use Grassmannian) |
| **FuseChat** (token alignment) | Fuse different architectures | Future: cross-architecture fusion |

## Two Products

### Pierre Tiny (edge, free tier)
- Base: BitNet-2B or small Qwen
- 5-25 SFT domain adapters
- 73-97 tok/s on M5 Pro
- 1.7-3GB memory
- Per-token softmax routing

### Pierre Pro (cloud, pro tier)
- Base: Qwen3-4B (or promoted)
- 25-200+ domain adapters
- Session adapters (M2P-generated, ephemeral)
- User profile adapters (M2P-distilled)
- Full promotion lifecycle at scale=5
- Per-token routing + block-diagonal attention
