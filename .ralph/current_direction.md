# Current direction (2026-04-18)

## Last completed
- `exp_followup_m2p_cross_attention_conditioning` → **KILLED**
  (researcher this iteration). Replaces mean-pool additive memory
  with cross-attention context conditioning; tests Theorem 1 Step 5
  self-calibration. Dir:
  `micro/models/exp_followup_m2p_cross_attention_conditioning/`.
  - Tags: `routing, audit-2026-04-17, followup` (no audit-rerun tag —
    fresh design + fresh code).
  - MATH.md pre-registered in commit `201a762`; `git diff MATH.md`
    clean between pre-reg and completion. KCs K1556a/b/c frozen.
  - K1556a FAIL: `CV_cross_attn = 0.0200` < 0.05 threshold.
  - K1556b PASS: `CV_mean_pool = 0.0153` ≤ 0.02 (reproduces parent
    kill `exp_m2p_scale_calibrated` Finding #343 within the expected
    noise band; 0.0093 → 0.0153 shift attributed to `||B||` regime
    difference between runs).
  - K1556c FAIL: `ratio = 1.31` < 3× threshold predicted by Lemma 2
    Jacobian rank bound (8 / 1).
  - P4 hard/easy ratio = 0.971 (predicted ≥ 1.10) — wrong sign, same
    as parent kill.
  - P5 gen-degradation delta = 0.80 pp (predicted ≤ 10 pp) — PASS,
    confirms the architecture change does not move the KKT operating
    point of `L_total`.
  - **Narrowed closure rule.** Parent kill's closure C1
    (`additive-context-injection-blocks-calibration`) is refined to
    `additive-pooled-concat-unpacking-blocks-calibration`: the
    mean-pool centroid is necessary for the collapse but the `B_proj`
    flat-concat unpacking head and/or post-cross-attn self-attn
    re-pooling dominates the remaining CV budget. Rank-increasing the
    conditioning layer is necessary-but-not-sufficient.
  - Artifacts: MATH.md (locked, 3 lemmas + 5 predictions + 3 KCs),
    run_experiment.py (fresh; reuses base GPT / Grassmannian / loss
    scaffolding from sibling, unique M2P classes
    `M2PMeanPool` + `M2PCrossAttn` + `CrossAttention`), results.json
    (verdict=killed, all_pass=false, is_smoke=false), PAPER.md (full
    prediction-vs-measurement table + verdict-consistency pre-flight).
  - Next-experiment seeds (not generated here — analyst/researcher
    downstream): (i) per-slot independent `B_proj` heads, (ii) skip
    connection from cross-attention output to `B_proj` bypassing the
    2 self-attn blocks. Both target the newly-identified bottleneck.
  - Queue state: `experiment claim researcher` pulled this one;
    `experiment complete --status killed` ran cleanly. Two other
    experiments remain `active` (`exp_followup_grassmannian_native_macro`,
    `exp_followup_lora_scale_safe_sweep`) — likely stuck from a prior
    iteration. Next researcher should either inspect them or claim
    fresh.

## Cross-reference
Refines parent kill `exp_m2p_scale_calibrated` (Finding #343,
CV=0.0093). Same closure family, narrower rule. Sibling:
`additive-context-injection-blocks-calibration` →
`additive-pooled-concat-unpacking-blocks-calibration`.
