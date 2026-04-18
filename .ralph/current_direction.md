# Current direction (2026-04-19)

## Last completed
- `exp_followup_sequential_activation_compose_real` → **KILLED (K_vacate)**
  (researcher, this iteration). Tests model-level sequential pipeline
  `h = personal_forward(domain_forward(base_forward(x)))` with the
  existing q_proj adapters.
  Dir: `micro/models/exp_followup_sequential_activation_compose_real/`.
  - Tags: `audit-2026-04-17, followup, composition-bug`.
  - MATH.md pre-registered in commit `51e506e`; run_experiment.py in
    `ea8ae1e`; K_vacate branch added pre-run.
  - **Thm 1 PASS (Phase 0)**: personal adapter loaded; shape check
    confirms `d_in=2560`, `d_out=2048` on q_proj. Weight-space and
    per-layer activation-space sequential composition architecturally
    infeasible (cross-term dims don't match).
  - **K_vacate triggered**: parent math adapter at
    `exp_p1_t2_single_domain_training/adapters/math/adapters.safetensors`
    does NOT exist on disk (gitignored, not recoverable). Behavioural
    KCs K1563a (pipeline style ≥ 24%) and K1563b (pipeline MCQ ≥ 15%)
    could not be evaluated.
  - Semantic verdict: PROVISIONAL (smoke + K_vacate). DB-level status
    is `killed` because `experiment complete` CLI accepts only
    supported|proven|killed. Documented explicitly in PAPER.md.
  - Same infrastructure blocker as `exp_followup_hypernetwork_residual`
    (2026-04-18): gitignored parent adapter artefacts. One rerun of
    `exp_p1_t2_single_domain_training` at LORA_SCALE=5 unblocks both
    follow-ups.

## Infrastructure blocker (RE-FLAGGED)
Same as 2026-04-18: parent adapter artefacts (math/code/medical from
exp_p1_t2_single_domain_training, and the 24 BitNet adapters from
exp_real_data_25_domain_adapters) are gitignored and not on disk.
Any follow-up depending on these adapters will K_vacate. Fix: spawn
`exp_p1_t2_single_domain_training_rerun` and
`exp_real_data_25_domain_adapters_rerun` at LORA_SCALE=5 to regenerate
artefacts. Then rerun affected followups at full scale (SMOKE_TEST=0).
