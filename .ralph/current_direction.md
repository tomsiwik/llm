# Current direction (2026-04-19)

## Last completed
- `exp_followup_format_compat_peft_required` → **SUPPORTED**
  (researcher, this iteration). K1576 PASS all three gates:
  K1576.a hard-required `import peft / transformers / torch`
  (peft 0.18.1, transformers 5.5.0, torch 2.10.0) eliminates
  silent-bypass antipattern structurally; K1576.b real
  `peft.PeftModel.from_pretrained(tiny_llama, dir)` + forward pass
  on a tiny `LlamaForCausalLM` (0.034 s load + 0.008 s forward,
  12/12 LoraLayer wraps, logits `[1,4,256]`); K1576.c separate
  q/k/v_proj keys at all 4 layers, `fused_keys_present=false`,
  distinct A matrices (max|A_q−A_k|=0.589, max|A_q−A_v|=0.640),
  fused-QKV stack shapes `[3r=18, d_in=64]` and `[3·d_q=192, 3r=18]`
  match Theorem 2 block-diagonal fusion. Closes the honest gap left
  by `exp_p1_t4_adapter_format_compat` (KILLED, Finding #585 —
  silent-ImportError-bypass + subset-direction-fallacy). Runtime
  0.065 s CPU; no HF-Hub fetch. Grassmannian max_dev=2.38e-7 under
  1e-6 tol. Dir: `micro/models/exp_followup_format_compat_peft_required/`.
  Tags: `routing, audit-2026-04-17, followup`.

- `exp_followup_orthogonal_projection_scale_control` → **KILLED**
  (researcher, prior iteration). KC #1573 FAIL via pre-registered
  R-struct + R-pareto. Theoretical-refutation probe (no retraining).
  Parent Thm C1 (spectral-gap vacuity, σ_k/σ_{k+1}=1.005) is
  scale-invariant; parent Thm C3 (capacity-interference dominance,
  80% rank-level floor) makes K3 fail at every s∈{4,6,8,10} under
  linear-delta scaling. Disproves the "LORA_SCALE=20 confounder"
  hypothesis for the orthogonal_adapter_training KILL — the kill
  is structural, not scale-sensitive. Dir:
  `micro/models/exp_followup_orthogonal_projection_scale_control/`.
  Tags: `audit-2026-04-17, followup, scale-safety`.

- `exp_followup_ss_rn_path_valid_sft` → **KILLED** (researcher, prior iteration).
  6th precondition-probe KILL this loop. K1572 (|acc_final − 74.4%| ≤ 5pp)
  routed FAIL-unmeasurable: P1 T2.1 `adapters/math/adapters.safetensors`
  missing (gitignored), P2 no personalization corpus staged, P3 upstream T2.1
  verdict=KILLED. No training executed; pre-registered routing in MATH.md
  applied cleanly. Duplicate-scope with Finding #600
  (`exp_followup_sft_behavioral_lora_scale_5`); only the KC band differs
  (≤5pp here vs QR≥0.90 there). Finding #602 registered. Dir:
  `micro/models/exp_followup_ss_rn_path_valid_sft/`. Tags:
  `audit-2026-04-17, followup, scale-safety`.

- `exp_followup_answer_conditioned_ppl` → **KILLED** (researcher, prior
  iteration). K1567 required BOTH (a) Top1_ans ≥ 0.85 AND (b) Top1_full < 0.85.
  Measured Top1_ans=0.978 (PASS), Top1_full=0.984 (FAIL the <0.85 clause).
  Predecessor's r_full=−0.31 was a relative-change correlation, not absolute
  cross-expert ranking; routing is dominated by cross-domain distribution
  mismatch, not prompt-PPL degradation. MATH.md §3 prediction refuted.

- `exp_followup_sft_behavioral_lora_scale_5` → **KILLED** (5th precondition-probe
  KILL, Finding #600). Same blocker class as this iteration.

- `exp_followup_m2p_crystallize_real_users` → **KILLED** (mean cos(crystal, B*)
  = 0.9377 < 0.95 under heterogeneous LR/steps/seeds).

- `exp_followup_sequential_activation_compose_real` → **KILLED (K_vacate)**
  by same parent adapter missing.

## Infrastructure blocker (RE-FLAGGED 4th time — BLOCKING AT LEAST 5 EXPERIMENTS)
Parent adapter artefacts (math/code/medical from
`exp_p1_t2_single_domain_training`, and BitNet N=25 from
`exp_real_data_25_domain_adapters`) are gitignored and not on disk.
Experiments now KILLED via this blocker:
  - exp_followup_hypernetwork_residual
  - exp_followup_sequential_activation_compose_real
  - exp_followup_sft_behavioral_lora_scale_5 (Finding #600)
  - exp_p1_t2_sft_residual_gemma4 (V2 rerun)
  - exp_followup_ss_rn_path_valid_sft (this iter, Finding #602)

Fix (class-level unblock, single action unblocks 5+ experiments):
  1. Rerun `exp_p1_t2_single_domain_training` at LORA_SCALE=5 to regenerate
     `adapters.safetensors` for math/code/medical.
  2. Rerun `exp_real_data_25_domain_adapters` at LORA_SCALE=5 to regenerate
     the 24 BitNet adapters.
  3. Stage a personalization corpus (persona-tagged queries, disjoint from
     GSM8K) for downstream M2P follow-ups.
  4. Rerun affected followups at full scale (SMOKE_TEST=0).

## Standing rules confirmed (5 instances this loop)
1. Precondition-probe before macro rerun — audit-tagged experiments with
   blocked preconditions skip heavy training and KILL on probe.
2. Adapter registry ≠ adapter artefacts — `adapter_config.json` is not
   evidence weights exist.
3. Downstream inherits upstream audit flags — a KILLED upstream propagates
   to every dependent experiment.
4. `code-bug` tag may be a decoy when the mechanism is mathematical.
5. Duplicate-scope followups (same claim, different KC band) share one blocker
   and should be collapsed to a single survivor after unblock.
