# Vision: The Living Composable Model

## Core Thesis

A language model grows by acquiring domain experts, improves by evolving them,
and scales by adding capacity — all without retraining the base.

## Status: Composition Crisis → Dual-Track Resolution

**What works:**
- Distillation pipeline ($0.44/expert, HumanEval +9.1pp, reasoning +10.6pp)
- Structural orthogonality (cos scaling law validated d=64 to d=1024)
- Grassmannian skeleton (AP packing, frozen-A, capacity bounds)
- Individual adapter quality (mean -0.95pp on MMLU — nearly neutral)
- Routing latency solved (<21us at N=1000)

**What's broken:**
- ~~Equal-weight composition catastrophic~~ **RESOLVED** by 1/N scaling (PPL=2.36)
- Trained adapter orthogonality inflated (cos=0.142 at d=896, not 0.0002)
- Orthogonality alone insufficient for semantic composability (arxiv 2510.03262)
- Evolve phase has no validated scoring metric at macro

**Two parallel resolution tracks:**

### Track A: BitNet-SOLE (Ternary Composition) — $0, local
Hypothesis: ternary {-1, 0, 1} base weights bound adapter magnitudes,
preventing the logit-scale mismatch that causes composition catastrophe.

- Base: microsoft/bitnet-b1.58-2B-4T (or scale to 30B)
- Experts: LoRA (FP16) or BitLoRA (ternary) adapters
- Serving: bitnet.cpp on CPU ($50/mo vs $115/mo GPU)
- Architecture inspired by MoTE (2506.14435): frozen shared base + ternary routed experts

### Track B: FP16 + Smart Routing — ~$5, GPU
Hypothesis: PPL-probe weighted composition (r=0.990 oracle correlation at micro)
fixes the N=5 regression on Qwen2.5-7B.

- If <20% degradation with routing: FP16 SOLE works with mandatory routing
- If still >50% degradation: FP16 equal-weight is dead, pivot to Track A

## What We Proved

### Conclusive (survives peer review)

| Finding | Result | Scale |
|---------|--------|-------|
| LoRA orthogonality scales as 1/sqrt(D) | cos 17-69x below sqrt(r/d) | Micro + Macro |
| Distillation pipeline produces functional adapters | HumanEval +9.1pp, $0.44/expert | Macro (7B) |
| Reasoning distillation works (K1) | +10.6pp on MATH-500 (67.6% vs 57.0%) | Macro (7B) |
| Individual adapters don't harm general knowledge | Mean -0.95pp on 57 MMLU subjects | Macro (7B) |
| Composition regression is from interference | Diagnosis: COMPOSITION_INTERFERENCE | Macro (7B) |
| Routing latency is not a bottleneck | All strategies <21us at N=1000 | Micro |
| Grassmannian AP packing is optimal | 1.2-1.5x beyond orthonormality, zero drift | Micro |
| Expert removal is safe | Naive subtraction <0.2% error at SOLE cosines | Micro |
| Safety bound complete | 5 micro experiments unified, transfers to production | Micro |

### Killed (important negative results)

| Finding | Result | Implication |
|---------|--------|-------------|
| Equal-weight composition is catastrophic | **RESOLVED by 1/N scaling** (PPL trillions→2.36) | 1/N scaling mandatory; unscaled is dead |
| Trained adapter orthogonality inflated | cos=0.142 (35.6x micro prediction) | Geometric orthogonality ≠ functional independence |
| Composition fragile to expert dropout | **RESOLVED by 1/N scaling** — all 5 adapters net positive | CV=112.2% was unscaled artifact |
| FFN-only adapters | PPL +66.7% | All-modules adapters required |
| Answer-conditioned PPL at macro | r=-0.63 (vs r=0.811 micro) | Evolve scoring needs different approach |

### Supported (directional evidence, needs more validation)

| Finding | Result | Caveat |
|---------|--------|--------|
| Pre-merge preserves gains over base | K3 PASS at N=5 | Only 5 adapters, contaminated eval |
| ReLoRA base supports composition | cos_ratio=0.875x | GPT-2-124M only, not 7B |
| PPL-probe weighting (micro) | r=0.990 oracle correlation | Never tested at macro |
| 1/N scaling resolves composition catastrophe | PPL trillions→2.36 (10^12x improvement) | N=5 only; scaling to N=50+ untested |
| LOO PPL ranks adapter contribution | sql < python < bash < math < medical | N=5, calibration from training tails |
| SOLE vs monolithic: +32.7% gap (confounded) | Union bf16 vs SOLE NF4 QLoRA | Per-domain PPL unmeasured (blocking) |

## Architecture

### Current: SOLE (Structurally Orthogonal Latent Experts)

```
Frozen Base (Qwen2.5-7B or BitNet-30B)
    │
    ├── LoRA Expert 1 (rank-16, all-modules, ~6MB)
    ├── LoRA Expert 2
    ├── ...
    └── LoRA Expert N

Routing: PPL-probe weighted (per-query top-k selection)
Serving: vLLM (GPU) or bitnet.cpp (CPU)
Evolution: clone, correct, compete, prune
```

### Proposed: MoTE-SOLE (Mixture of Ternary Experts)

```
Frozen FP16 Base (shared expert, always active)
    │
    ├── Ternary Expert 1 ({-1,0,1} weights, ~0.6MB)
    ├── Ternary Expert 2 (routed via top-k gating)
    ├── ...
    └── Ternary Expert N

Router: trained with load-balancing loss
Memory: 8 ternary experts fit in 1 FP16 expert's memory
Serving: bitnet.cpp on CPU, $50/mo
```

### The Grassmannian Skeleton

Pre-computed optimal subspace slots on Gr(r, d). Architecture-dependent, not
weight-dependent. Experts hot-swappable. Clones share slots.

```
Skeleton: Gr(16, 4096) with N slots
    ├── Slot 1: python_v2.safetensors
    ├── Slot 2: math_v1.safetensors
    ├── Slot 3: medical_v1.safetensors
    │           └── challenger: medical_v2_delta
    ├── Slot 4: (free)
    ⋮
    └── Slot N: ...
```

Capacity: N_max = d²/r² (609K at d=4096 r=16, 147K at d=6144 r=16).

## The Three Phases

### Phase 1: Distill — Create Experts ($0.25-0.44 each)
Teacher (70B via Groq) → training data → student (base + QLoRA rank-16).
Validated: 98% PPL win rate, HumanEval +9.1pp, reasoning +10.6pp on MATH-500.

### Phase 2: Compose — Serve with Routing
**Status: CRITICAL BLOCKER.** Equal-weight pre-merge is broken.
Two fixes being tested: (A) BitNet ternary base, (B) PPL-probe weighted routing.

### Phase 3: Evolve — Learn Without Retraining
Clone-and-compete mechanism. Never tested at macro.
Shadow scoring metric needs replacement (answer-conditioned PPL killed at macro).

## Readiness Assessment (Honest)

| Phase | Readiness | Blocker |
|-------|-----------|---------|
| Distill | **60%** | Quality validation on held-out benchmarks |
| Compose | **20%** | Equal-weight broken; routing and ternary untested at macro |
| Evolve | **10%** | No validated scoring metric; mechanism never tested |
| Serving | **30%** | No production code; vLLM incompatible with RTX 5090 LoRA |
| **Overall** | **~25-30%** | Composition is the existential blocker |

## Competitive Positioning (Honest)

SOLE does NOT compete with frontier models (GPT-4, Claude, DeepSeek-V3) on
general reasoning or instruction following. The 7B base is the quality ceiling.

**Where SOLE competes:**
- Domain-specific Q&A (medical, legal, code) at 7B-class quality
- Cost efficiency: $0.25/expert vs $1000+ for full fine-tune
- Modularity: update one domain without touching others
- Privacy: on-premise deployment (30B BitNet on $2000 workstation)
- Edge: domain expert on smartphone (BitNet-2B + LoRA)

**Where SOLE cannot compete:**
- General reasoning (needs >100B dense parameters)
- Long-context synthesis (limited by base attention capacity)
- Tasks requiring frontier model depth

## Priority Queue (What to Run Next)

### P1: Must-run experiments ($0-5)
1. ~~**BitNet N=5 composition** (Track A, $0, local)~~ — **SUPPORTED** (2026-03-19): non-catastrophic, but mechanism is quantization recovery (denominator effect), not interference reduction. FP16 wins 3/5 domains in absolute PPL. Gate-pass for downstream BitNet experiments.
   - **Real BitNet-2B-4T** (2026-03-20): **SUPPORTED**. First LoRA fine-tuning on real BitNet-2B-4T via MLX. K1-K4 PASS. Composition ratio 3.59x. |cos|=0.001. $0 compute, 12 min. Caveats: train/val contamination, under-training, single seed.
2. **PPL-probe macro composition** (Track B, ~$5, GPU) — make-or-break for FP16
3. **Poisoned adapter detection** (~$3, GPU) — operationally critical

### P2: Run if P1 succeeds ($5-10)
4. BitLoRA (ternary adapters) composition — if Track A passes
5. ~~SOLE vs monolithic LoRA baseline~~ — **SUPPORTED** (4/5 domains, -5.8% avg PPL)
6. Reasoning adapter K2/K3 — the killer demo

### P3: Defer
- Base-freedom (ReLoRA at 7B) — park until compose works
- Scale to 500 experts — blocked on composition fix
- New micro experiments — diminishing returns
- Grassmannian refinements — complete

## Key References

- BitNet b1.58 (arxiv 2402.17764) — ternary weight architecture
- MoTE (arxiv 2506.14435) — Mixture of Ternary Experts
- LoTA-QAF (arxiv 2505.18724) — lossless ternary adaptation
- Rethinking Inter-LoRA Orthogonality (arxiv 2510.03262) — orthogonality ≠ composability
- QVAC BitNet LoRA (HuggingFace blog) — cross-platform LoRA on BitNet
- BitLoRA (ScienceDirect 2026) — ternary adapters for BitNet
- SOLE Adversarial Review (SOLE_ADVERSARIAL_REVIEW.md) — full project audit
- BitNet-SOLE Research (references/BITNET_SOLE_RESEARCH.md) — integration analysis
