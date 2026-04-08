# Current Direction: Level 3 — Statistical Closure (Active)

## PoC Progress

| Level | Status | Key Result |
|-------|--------|-----------|
| 0 (0%) | DONE | Math hardened, claims reframed |
| 1 (20%) | DONE | L=36: 89.1% PASS. Safe dissolve: S3 wins. |
| 2 (40%) | DONE | Activation α=0.38 PASS. Cipher 99.7%. |
| 3 (60%) | **ACTIVE** | v4 SUPPORTED; SFT n=500 measurement in progress |
| 4 (80%) | WAITING | Needs Level 3 closure |
| 5 (100%) | WAITING | Needs Level 4 |

## Level 3: Statistical Closure (v4 SUPPORTED, n=500 measurement active)

### v4 result (exp_m2p_qwen06b_gsm8k_v4) — SUPPORTED

All kill criteria PASS:
- K916 PASS: grad_norm=1.506 at step 0 (Theorem 5 confirmed)
- K917 PASS: M2P loss=0.907 at step 1000
- K918 PASS: quality_ratio=1.433, CI_lower=0.773 >= 0.60 at n=500

Key finding: M2P accuracy = 28.6% (n=500) > SFT accuracy = 26.0% (fixed from v2, n=200).
Gap is NOT statistically significant (p=0.36, two-proportion z-test).
CI_lower=0.773 is OPTIMISTIC — SFT was measured at n=200, not 500.
Reviewer correctly flagged: propagate SFT uncertainty before claiming closure.

### Active: exp_m2p_sft_n500_baseline

**Purpose:** Statistical closure — remeasure SFT at n=500, propagate uncertainty.

**Design:** MATH.md + run_experiment.py complete.
- Location: micro/models/m2p_sft_n500_baseline/
- Loads v2 SFT adapter (lora_a_matrices.npz + sft_b_matrices.npz)
- Evaluates on n=500 GSM8K test examples (SEED=42, same as v4)
- Runs Wilson CI, two-proportion z-test, Fieller delta method

Kill criteria:
- K919: SFT accuracy at n=500 measured with Wilson 95% CI (unconditional)
- K920: Two-proportion z-test M2P(28.6%, n=500) vs SFT(n=500) — PASS if p<0.05
- K921: quality_ratio CI_lower recomputed with propagated SFT uncertainty (Fieller)

**Prediction from MATH.md Theorem 2:**
If SFT≈26% at n=500: Fieller CI_lower ≈ 0.30 (vs reported 0.773, bias ≈ 0.47).
The optimism in v4's CI_lower will be quantified.

**Expected runtime:** ~17 min (500 inferences × ~2s each).

### Ready to run: exp_m2p_vera_bottleneck

**Purpose:** Parameter overhead fix — VeRA-style shared matrices to reduce M2P
parameter count while maintaining accuracy. Currently M2P has 357M params (larger
than Qwen3-0.6B itself). VeRA uses shared random basis + per-layer 2r scale scalars.

**Status:** DESIGNED AND READY TO RUN.
- Location: micro/models/m2p_vera_bottleneck/
- MATH.md: Theorem 1 (parameter reduction: 357M → 4.7M, 76x), Theorem 2 (Theorem 5
  inherited), Theorem 3 (VeRA rank-r expressiveness via JL lemma)
- run_experiment.py: VeRAM2PNetwork with single Linear(1024, 576) output head +
  frozen W_q/W_v, functional forward identical to v4, 500 train steps

**Theorem 1 prediction (verified by direct arithmetic):**
  Trainable params = 4,656,576 ≈ 4.7M (encoder 4.2M + scale_head 459K)
  Reduction from v4 = ~76x (K922 requires ≥35x — PASS with margin)

**Kill criteria:**
  K922: params ≤ 10M — predicted PASS (4.7M)
  K923: quality_ratio ≥ 0.70 at n=500 — empirical (VeRA Table 2 extrapolation)
  K924: grad_norm > 0 at step 0 — predicted PASS (Theorem 2, verified in smoke test)

**To run:** experiment run exp_m2p_vera_bottleneck
**Estimated runtime:** ~45-60 min (500 steps + 500 GSM8K eval)

## What's NOT Affected

All composition/routing/scaling math is independent of how B is generated:
- Grassmannian A-matrices, TF-IDF routing, safe dissolve, activation bounds — all hold.
- Statistical closure is about CREDIBILITY of the M2P-vs-SFT comparison,
  not about the M2P mechanism itself (which is proven by Theorem 5 + K916-K918 PASS).

---

## Level 3 (composition): exp_m2p_composition_n5_qwen3 — DESIGNED, READY TO RUN

**Purpose:** Verify that two real-LLM M2P adapters (math + sort) compose without mutual
interference on Qwen3-0.6B. This is the composition extension of v4.

**Design:** MATH.md + run_experiment.py complete.
- Location: micro/models/m2p_composition_n5_qwen3/
- Phase 1: Train math M2P (warm-start from v4, 300 steps on 500 GSM8K examples)
- Phase 2: Train sort M2P (fresh, 300 steps on 500 synthetic word-sort examples)
- Phase 3: Train TF-IDF router (200 math + 200 sort train prompts, 100+100 val)
- Phase 4: K925 — grad_norm > 0 under composed adapter (Theorem 5 + Theorem 1)
- Phase 5: Single-adapter baselines (100 math + 100 sort examples)
- Phase 6: Composed adapter eval + TF-IDF routing (100 math + 100 sort)

**Kill criteria:**
- K925: grad_norm > 0 under composed B = 0.5*B_math + 0.5*B_sort (Theorem 5)
- K926: TF-IDF routing >= 80% on both tasks (predicted: >= 95% — highly separable)
- K927: quality_ratio >= 0.75 on both tasks (Theorem 3 lower bound: 1.15)

**Key fix vs. exp_m2p_composition_n5 (KILLED, 36.6% routing):**
Router trained on raw input text only (TF-IDF), never calls model forward.
By construction, routing is invariant to adapter composition state (Theorem 2).

**Expected runtime:** ~35-45 min on M5 Pro 48GB.

**To run:** experiment run exp_m2p_composition_n5_qwen3
