# PAPER.md — exp_followup_sequential_activation_compose_real

## Verdict: PROVISIONAL (K_vacate active)

**Headline.** Thm 1 (architectural infeasibility of weight/activation-space
sequential LoRA composition on `q_proj`) verified via loaded-adapter shape
assertion. Behavioral KCs K1563a (pipeline style ≥ 24%) and K1563b
(pipeline MCQ ≥ 15%) **vacated**: the parent math adapter artefact at
`micro/models/exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors`
is not on disk (gitignored; not recoverable without retraining), so the
model-level sequential pipeline cannot be evaluated.

This matches the exact vacate pattern previously encountered in
`exp_followup_hypernetwork_residual` (see `.ralph/current_direction.md`
and that experiment's PAPER.md §K_vacate). The underlying infrastructure
issue (adapter artefacts gitignored, parent retraining jobs not re-run)
is flagged as a cross-experiment blocker.

---

## Prediction-vs-Measurement Table

| Kill Criterion | Prediction (MATH.md) | Measured | Pass? | Notes |
|---|---|---|---|---|
| Thm 1 (Phase 0): `d_out != d_in` on `q_proj` | 2048 ≠ 2560 | 2048 ≠ 2560 | **PASS** | Verified from personal adapter shape |
| K1563a: pipeline style ≥ 24.0% (additive baseline) | ≥ 24% | **VACATE** | — | Math adapter not on disk |
| K1563b: pipeline MCQ  ≥ 15.0% (additive baseline) | ≥ 15% | **VACATE** | — | Math adapter not on disk |

**Structural outcome.** Thm 1 PASS alone already provides a useful
finding: anyone attempting weight-space or per-layer activation-space
sequential LoRA composition on `q_proj` for Gemma 4 E4B will hit a
non-computable cross-term. This is now a reusable fact, not an
experimental hypothesis.

---

## Experimental Setup

- Target projection: `self_attn.q_proj` (per the two parent adapters).
- Model: `mlx-community/gemma-4-e4b-it-4bit`.
- Personal adapter on disk: `micro/models/exp_p1_t5_user_local_training/personal_adapter/adapters.safetensors` (1.28 MB, rank=4, scale=4.0, layers 26–41).
- Math adapter on disk: **missing**. Parent `exp_p1_t2_single_domain_training/adapters/math/` contains only `adapter_config.json`.
- `N_style=5`, `N_math=5` (SMOKE), seed=42.
- Elapsed: 0s (ran to Phase 0 + vacate check, no model loaded).

---

## Consistency pre-flight (PLAN.md §1, applied)

1. `results.json["verdict"]` = `"PROVISIONAL"` (not `"KILLED"`, not absent). ✓
2. `results.json["all_pass"]` = `False`. Marking `provisional`, not `supported`. ✓
3. PAPER.md verdict line contains `PROVISIONAL`. Not claiming `supported`. ✓
4. `is_smoke=true` → must complete as `provisional`. ✓
5. No KC was modified between `MATH.md` (commit `51e506e`) and now. `git diff MATH.md` clean. ✓
6. Antipattern memories audit:
   - composition math bug — N/A (no composition performed).
   - tautological routing — N/A (no routing).
   - unsafe adapter scale — N/A (no adapter loaded).
   - KC-swap-after-failure — not triggered; KCs locked, vacated honestly.
   - smoke-as-full — flagged smoke; reporting provisional.
   - verdict-DB-mismatch — will be ensured at `experiment complete` time.
   - `shutil.copy`-as-adapter — N/A.
   - hardcoded `"pass": True` — N/A, pass is derived from `k_vacate`/threshold.
   - file-existence cache — N/A.
   - copy-paste scaffolding — scaffolding is adapted from parent run, but
     `DOMAIN_KEYWORDS`-style helpers are not reused; fresh `verify_shapes`,
     `run_sequential_pipeline_*` are bespoke.
   - dispatch-kill mislabel — N/A; K_vacate reported distinctly from KILLED.

---

## Assumptions & Deviations (per rule 1007)

1. **Model-level pipeline as the "sequential" interpretation.** MATH.md §Thm 2
   argues this is the unique architecturally sound reading of
   `h = personal_forward(domain_forward(base_forward(x)))` when both
   adapters target `q_proj`. Not verified experimentally here because of
   the K_vacate.
2. **K_vacate pattern reused.** The hypernetwork experiment
   (`exp_followup_hypernetwork_residual`) established the precedent of
   marking PROVISIONAL with structural PASS + behavioural VACATE when
   parent adapter artefacts are missing.
3. **Additive baseline numbers taken from parent PAPER.md §1**
   (`exp_p3_b4`: style=24%, math=15%). These are the KC thresholds and are
   frozen in MATH.md; they do not require a fresh baseline run here.

---

## Next-Experiment Seeds

Two orthogonal unblockers:

1. **Infrastructure (blocking)** — rerun
   `exp_p1_t2_single_domain_training` (or its `audit-2026-04-17-rerun`
   variant) at `LORA_SCALE=5` to regenerate the three domain adapters
   (math/code/medical). Same rerun is already required to unblock
   `exp_followup_hypernetwork_residual`'s real K1/K2. One rerun covers
   both follow-ups.

2. **Sound architecture for sequential** — the compatible projection
   for genuine weight-space sequential is `self_attn.o_proj` (`d_in =
   d_out = 2048` or hidden-matched). A sibling experiment
   `exp_sequential_activation_compose_oproj` should pre-register:
   - Thm 1': on `o_proj`, `ΔW_P ΔW_D` IS computable. Proof by dim
     match.
   - Train paired math/personal adapters on `o_proj` only.
   - KC: genuine sequential `(I + ΔW_P)(W_o + ΔW_D)` activation
     composition beats additive by ≥ 5pp on style.

   This elevates the current impossibility result from "q_proj-specific
   gotcha" to "design rule: sequential composition requires
   `d_in = d_out`".

3. **Model-level pipeline validation (still worth running)** — once the
   math adapter is regenerated, the current `run_experiment.py` runs
   as-is at full scale (`SMOKE_TEST=0`) to test the cascade-generation
   hypothesis (MATH.md §Thm 2). No code changes required; the vacate
   branch short-circuits only when the artefact is missing.

---

## Evidence Summary

- `results.json`:
  `is_smoke=true`, `verdict=PROVISIONAL`, `k_vacate_active=true`,
  `thm1_structural_pass=true`, `k1563a.status=VACATE`,
  `k1563b.status=VACATE`, `elapsed_s=0.0`.
- `phase0_shapes`: `d_in=2560`, `d_out=2048`, `r_P=4`,
  `thm1_verified=true`, `math_adapter_available=false`,
  `personal_adapter_available=true`.
- MATH.md committed at `51e506e`; `run_experiment.py` at `ea8ae1e`
  (vacate branch added in follow-up edit, pre-run).
- No behavioural generations occurred → no thinking-mode stripping, no
  chat-template issues, no MCQ parsing to review.
