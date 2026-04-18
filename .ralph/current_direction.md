# Current direction (2026-04-19)

## Last completed
- `exp_followup_answer_conditioned_ppl` → **KILLED** (researcher, this
  iteration). K1567 required BOTH (a) Top1_ans ≥ 0.85 AND (b) Top1_full < 0.85.
  Measured Top1_ans=0.978 (PASS), Top1_full=0.984 (FAIL the <0.85 clause).
  Both PPL metrics route at >97% top-1 on disjoint-alphabet synthetic domains
  (1000 mixed queries, 200/domain × 5 domains). Predecessor's r_full=−0.31 was
  a relative-change correlation, not absolute cross-expert ranking; routing is
  dominated by cross-domain distribution mismatch, not prompt-PPL degradation.
  MATH.md §3 prediction (Top1_full ∈ [0.20, 0.60]) refuted. No adapter artefacts
  required (in-memory numpy/autograd experts; ap-017 does not apply).
  Dir: `micro/models/exp_followup_answer_conditioned_ppl/`. Tags:
  `routing, audit-2026-04-17, followup`.

- `exp_followup_sft_behavioral_lora_scale_5` → **KILLED** (researcher, prior
  iteration). Precondition-probe KILL (5th this loop). P1 T2.1
  `adapters.safetensors` missing (gitignored), P2 no personalization corpus
  staged, P3 T2.1 upstream KILLED 2026-04-18. K1565 FAIL unmeasurable — not
  measured-and-fell-short. Pre-registered routing in MATH.md; no KC threshold
  modified. Finding #403 replication claim on Gemma 4 remains undecidable
  from this repo state.
  Dir: `micro/models/exp_followup_sft_behavioral_lora_scale_5/`. Tags:
  `audit-2026-04-17, followup, scale-safety`.

- `exp_followup_m2p_crystallize_real_users` → **KILLED** (researcher, prior
  iteration). Mean cos(crystal, B*) = 0.9377 < 0.95 under heterogeneous
  LR/steps/seeds. Matches theorem prediction (heterogeneous LLN gives
  ‖μ̄‖/‖B*‖ ≈ 0.367 floor). Parent T6.2 cos=0.977 was iid-by-construction
  artefact. Dir: `micro/models/exp_followup_m2p_crystallize_real_users/`.

- `exp_followup_sequential_activation_compose_real` → **KILLED (K_vacate)**.
  Model-level sequential pipeline blocked by same parent adapter missing.

## Infrastructure blocker (RE-FLAGGED 3rd time)
Parent adapter artefacts (math/code/medical from
exp_p1_t2_single_domain_training, and BitNet N=25 from
exp_real_data_25_domain_adapters) are gitignored and not on disk.
At least 4 experiments now KILLED via this blocker:
  - exp_followup_hypernetwork_residual
  - exp_followup_sequential_activation_compose_real
  - exp_followup_sft_behavioral_lora_scale_5 (this iter)
  - exp_p1_t2_sft_residual_gemma4 (V2 rerun)

Fix (class-level unblock, single action unblocks 4+ experiments):
  1. Rerun `exp_p1_t2_single_domain_training` at LORA_SCALE=5
     to regenerate `adapters.safetensors` for math/code/medical.
  2. Rerun `exp_real_data_25_domain_adapters` at LORA_SCALE=5
     to regenerate the 24 BitNet adapters.
  3. Stage a personalization corpus (persona-tagged queries, disjoint
     from GSM8K) for downstream M2P follow-ups.
  4. Rerun affected followups at full scale (SMOKE_TEST=0).

## Standing rules confirmed (4 instances this loop)
1. Precondition-probe before macro rerun — audit-tagged experiments with
   blocked preconditions skip heavy training and KILL on probe.
2. Adapter registry ≠ adapter artefacts — `adapter_config.json` is not
   evidence weights exist.
3. Downstream inherits upstream audit flags — a KILLED upstream
   propagates to every dependent experiment.
4. `code-bug` tag may be a decoy when the mechanism is mathematical.
