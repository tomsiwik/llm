# Pierre P0: M2P Distillation Pipeline

## Current State (updated 2026-04-07)

The decoupled architecture — proven at toy scale, untested on real language:
- **Grassmannian A orthogonality:** PROVEN (cos < 1e-8, #341 K848)
- **M2P quality scaling:** 97.6% (d=256) → 100.6% (d=512) → 99.6% (d=1024) — #359, #361, #362
- **Layer depth:** 99.7% (L=2) → 86.4% (L=16) — #363. L=36 untested.
- **TF-IDF routing:** 95% accuracy — #354
- **Cross-domain transfer:** 8/10 pairs useful — #353
- **Safe dissolve:** script ready, NOT YET RUN
- **Single-cycle promotion:** DEMONSTRATED (#333). Multi-cycle: UNTESTED.
- **Natural language:** ZERO evidence. All results on synthetic domains.

## The Three Stages

```
Stage 1: FIX M2P (toy)           → break the centroid collapse
Stage 2: COMPOSE (toy→small)     → 5 M2P adapters compose with Grassmannian
Stage 3: SCALE (Qwen3-4B)        → M2P on production model, full benchmarks
```

## Stage 1: Fix M2P Multi-Domain Training (P0, current)

**Problem:** M2P converges B-matrices to centroid when trained on 5 domains
simultaneously. Domains with low base loss get catastrophically bad adapters.

**Model:** ToyGPT (d=64, 4 layers, ~100K params)
**Data:** 5 synthetic domains, 500 examples each

| Experiment | Approach | Status |
|-----------|----------|--------|
| exp_m2p_distillation_toy | Baseline M2P + Grassmannian A | KILLED (#341) — centroid collapse |
| exp_m2p_domain_conditioned | Additive domain embedding | KILLED (#342) — insufficient |
| exp_m2p_scale_calibrated | Preservation loss teaches scale | Active (P0) |

**Remaining fix candidates:**
1. Multiplicative gating on domain signal
2. Per-domain loss normalization (gradient magnitude equalization)
3. Separate M2P output heads per domain
4. Train single-domain M2P, compose at eval time

**Go/no-go:** If median M2P quality reaches ≥25% of SFT across all 5 domains
with no domain below -10%, the architecture works. If all fixes fail,
fall back to SFT-only adapters (proven to work: #319, #332).

## Stage 2: M2P Composition (P0, next)

**Model:** ToyGPT → validate on Qwen3-0.6B
**What we learn:** Do 5 independently M2P-generated adapters compose
without interference under Grassmannian guarantee?

| Experiment | What It Tests | Status |
|-----------|---------------|--------|
| exp_m2p_composition_n5 | 5 M2P adapters compose with frozen A-slots | Open (P0) |

**Metrics:**
- Mean quality ≥25% of 5 independent SFT adapters
- No domain degrades below -10%
- Grassmannian cos ≤ 1e-5 maintained
- Composition PPL ≤ 1.1x best-single

**Go/no-go:** If composition degrades despite Grassmannian A → the problem
is in activation space (B_i write to same outputs). Add output-space
orthogonality loss to M2P training.

## Stage 3: Production Scale (P1)

**Model:** Qwen3-4B-4bit (already validated: #317, #318, #320, #332)
**What we learn:** Does M2P + Grassmannian work at production scale?

| Experiment | What It Tests | Status |
|-----------|---------------|--------|
| exp_m2p_teacher_distillation | Qwen3-8B → 4B knowledge transfer via M2P | Open (P1) |
| exp_shine_architecture_study | Full SHINE architecture on Qwen3-4B | Active |
| exp_multi_tenant_serving | Per-user adapter stacks | Active |

**Metrics:**
- Full benchmark suite (MMLU, GSM8K, HumanEval via lm-eval harness)
- M2P session adapters generate in <5ms
- >50 tok/s with composed adapters on M5 Pro
- Promotion lifecycle works end-to-end

## Resource Budget

Everything runs on M5 Pro 48GB. No cloud needed.

| Stage | Compute | Memory | Time |
|-------|---------|--------|------|
| 1. Fix M2P | Minutes | <1GB | Days (iteration) |
| 2. Compose | Minutes-hours | <5GB | 1-2 weeks |
| 3. Scale | Hours-days | ~10GB | Weeks |

## Decision Tree

```
Stage 1: Fix centroid collapse?
├── YES → Stage 2 (M2P composition)
│         Composition works?
│         ├── YES → Stage 3 (Qwen3-4B, benchmarks)
│         │         Competitive?
│         │         ├── YES → SHIP IT
│         │         └── NO → M2P capacity insufficient at scale
│         │                   → Fall back to SFT adapters + Grassmannian
│         └── NO → Activation-space interference
│                   → Add output-space orthogonality loss
│                   → Or: compose in output space (not parameter space)
└── NO → M2P multi-domain fundamentally broken
         → Fall back: SFT-only adapters (proven to work)
         → Grassmannian A + SFT B + scale=5 + softmax routing
         → This path already works (#332) — just slower to create adapters
```

## SFT Fallback (already proven)

If M2P distillation doesn't pan out, the SFT path is fully validated:
- SFT adapters converge on Qwen3-4B (#319)
- Grassmannian orthogonality holds on GQA (#318)
- Composition at scale=5 preserves MMLU (#330)
- Full integrated pipeline works (#332)
- Per-token routing matches oracle (#28, #310, #313)
- Expert promotion works at scale=5 (#333)

The only thing lost is instant adapter generation (1.15ms M2P vs 300-step SFT).
The composition guarantee and routing are independent of how B is created.
