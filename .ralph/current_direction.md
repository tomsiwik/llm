# Current direction (2026-04-19)

## Last completed
- `exp_prod_openai_api_compat` → **KILLED_PREEMPTIVE** (researcher
  iter 40). 36th preempt in audit-2026-04-17 cohort; **first
  KILLED-parent-source preempt** (novel T5-K variant); 8th F#502/F#646
  schema-incomplete; 27th composition-bug (software-infrastructure-
  unbuilt; inherits F#652).
  Defense-in-depth = 4 of {T1, T2, T3, T5-K}. Over-determined.
  - T1: shortfall = **4/4** (no `/v1/chat/completions` endpoint,
    no `pierre serve` CLI, no `X-Pierre-Adapters` header handler,
    no SSE streaming harness in `pierre/`/`macro/`/`composer/`).
  - T2: 138.0 min vs 120 min ceiling (16 × 3 × 3 × 45 s + 1800 s
    cold-start). Independent block.
  - T3: DB literal `success_criteria: []` + `⚠ INCOMPLETE:
    success_criteria, references, kill_results (all untested)`.
  - T5-K (**novel sub-axis**): parent `exp_prod_mlxlm_integration`
    has `verdict=KILLED` with **5** failed preflight keys (T1B, T1C,
    T2, T3, DEP). Child inherits all 5; none independently resolved.
  - T4: pin_ratio = 0.00 (`.audit` directory absent). Reinforcer
    did not fire; does not block either (reinforce-only by design).
  Runner: pure stdlib, 1.52 s wall, 0 MLX, 0 HTTP bind, 0 model load.
- `exp_prod_adapter_attach_ux` → **KILLED_PREEMPTIVE** (researcher
  iter 39). 35th preempt in audit-2026-04-17 cohort; 16th
  SUPPORTED-source preempt; 7th F#502/F#646 schema-incomplete
  instance; 26th composition-bug (software-infrastructure-unbuilt
  variant, same as iter 37 `version_resolution`).
  Defense-in-depth = 3 of {T1, T3, T5}.
  - T1: shortfall=2/4 (`pierre_cli_entry_point` absent per
    pyproject.toml scripts = only `compose`; logit-cosine harness
    absent per 0 hits repo-wide). Server + p99 sub-probes produced
    grep false-positives; transparency logged in MATH A7.
  - T2: 1.23 min ≤ 120 ceiling. Reinforces only.
  - T3: DB literal `success_criteria: []` + `⚠ INCOMPLETE`.
  - T4: pin_ratio 0.70 > 0.20 floor; reinforces only.
  - T5: 4/5 automated literal breaches vs `exp_p1_t4_serving_v2`:
    (A) CLI-scope, (B) detach-scope, (C) p99-scope (source
    results.json pins neither p50 nor p99; MATH A8 correction),
    (D) process-restart-scope, (E) state-consistency-scope.
  Runner: pure stdlib, 2.60 s wall, 0 MLX.
- `exp_prod_differential_privacy_training` → **KILLED_PREEMPTIVE**
  (researcher iter 38). 34th preempt in audit-2026-04-17 cohort;
  15th SUPPORTED-source preempt; 6th F#502/F#646 schema-incomplete
  instance; 25th composition-bug, with new (s3)
  platform-library-absent-from-target-ecosystem sub-axis.
  **First 4-theorem block in the drain** (T1 ∧ T2 ∧ T3 ∧ T5 all fire).
  - T1: shortfall=3/4 (per_sample_gradient_mlx, rdp_accountant,
    non_dp_lora_baseline_on_same_data all absent).
  - T2: 726 min vs 120 min ceiling (6.05× overshoot) — first T2
    independent block in the drain. DP-SGD 10× Opacus floor × 3
    seeds (K1666) = 660 min training + 66 min non-DP baseline pair.
  - T3: DB literal `success_criteria: []` + `⚠ INCOMPLETE`
    (6th F#502/F#646 hit — heuristic earned at 6×).
  - T5: 5/5 literal breaches vs `exp_p1_t5_user_local_training`
    (source MATH.md DP-vocab count = 0):
    (A) privacy-mechanism-scope, (B) library-scope (<200 lines +
    HF PEFT), (C) comparator-scope (base not DP-vs-nonDP),
    (D) reproducibility-scope (N=1 vs 3 seeds), (E) platform-library.
  Dir: `micro/models/exp_prod_differential_privacy_training/`.
  Tags: privacy, product, ap-017.

## Queue state
- `experiment list --status open` P≤2 = **10** (down from 11).
- `experiment list --status active` = empty ✓.
- LEARNINGS debt: 12 entries (analyst-owed; cap remains 50/50).
- Drain-forward candidates for iter 39: attach_ux, openai_compat,
  update_mech, mistral_nemo, quantization, arena_hard, proprietary,
  n100_macro, multi_seed (likely F#571 Room Model-superseded).
