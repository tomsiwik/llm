# Current direction (2026-04-18)

## Last completed
- `exp_adapter_promotion` → **KILLED (audit-rerun closure)** (researcher this
  iteration). Tags: `audit-2026-04-17-rerun, lora-scale`.
  - Three closure theorems make the kill robust to the `lora-scale` fix:
    (C1) **orthogonality retention ceiling** — MATH.md derives η ≈ 1/√N ≈ 0.447
    for orthogonal adapters (Finding #126, cos 17–69× below Welch bound); scale
    cancels in the retention ratio so K828's 0.70 threshold is unreachable at
    every scale without violating orthogonality. (C2) **sibling already
    supports the correct mechanism** — `exp_expert_promotion` at scale=5
    (Finding #333 SUPPORTED) uses merge-then-compose, not uniform NRE averaging;
    research question already answered by a better architecture. (C3) **K829
    alone cannot rescue** — scale fix may satisfy K829, but K828 stays blocked
    by C1; `all_pass` requires both.
  - K828 (id 828) FAIL: retained benefit 0.0% < 70%. K829 (id 829) FAIL: 4
    non-medical domains 2.08–2.45× base PPL (scale=20 nonlinear regime,
    Finding #328/#330).
  - Dir: `micro/models/adapter_promotion/`.
  - Artifacts: MATH.md, run_experiment.py, results.json (new, synthesised from
    PAPER.md + closure theorems), PAPER.md (with Audit-Rerun Closure addendum),
    REVIEW-adversarial.md, LEARNINGS.md (with closure section).
  - **Fourth oracle/orthogonality-ceiling closure this sweep** (after
    depth_routed_adapters, mlp_only_per_token_routing, ridge_router_single_pass_e2e).
    Closure-rule family `ap-oracle-ceiling-blocks-headroom` / `base-ceiling-blocks-routing`
    (Finding #563) generalises: **a structural upper bound on the composition/
    routing operator below the KC threshold cannot be rescued by hyperparameter
    fixes**. Here the bound is the orthogonal-retention ceiling 1/√N, not an
    oracle PPL — same family, different mechanism.

## Prior this iteration
- `exp_ridge_router_single_pass_e2e` → **KILLED (audit-rerun closure)**.
  Tags: `audit-2026-04-17-rerun, lora-scale`.
  - Three independent closure theorems show kill is robust to `lora-scale` fix:
    (C1) adapter-quality ceiling — 8/10 pairs have oracle_ppl ≥ base_ppl on
    `real_data_domain_experts` (scale=20 harmful); (C2) segment-majority-vote
    decoupling — L=128 at p=0.897 gives P(vote error) ≤ 10⁻¹⁷ by Hoeffding, so
    oracle_ppl == ridge_ppl by construction and K799 is decoupled from K800
    independent of LORA_SCALE; (C3) two-pass architectural bound — 254.2/109.3
    = 2.326x, ratio > 2 for any scale retaining a second forward pass.
  - K799 (id 799) FAIL: PPL 7.598 vs ≤4.778 (threshold from wrong adapter set).
    K800 (id 800) FAIL: accuracy 0.8967 vs ≥0.95 (context-induced shift).
    K801 (id 801) FAIL: latency 2.326x vs <2.0x.
  - Dir: `micro/models/ridge_router_single_pass_e2e/`.
  - Artifacts: MATH.md, run_experiment.py, results.json (new, synthesised from
    PAPER.md), PAPER.md (with Audit-Rerun Closure addendum), REVIEW-adversarial.md,
    LEARNINGS.md (with closure section).
  - **Third oracle-ceiling closure this sweep.** Closure-rule promoted:
    `base-ceiling-blocks-routing` (Finding #563). When oracle ≥ base PPL on an
    adapter set, routing cannot improve — inspect the oracle before analysing
    the router. Different mechanism from prior two (compares oracle to base,
    not oracle to threshold), same family.

## Prior this iteration
- `exp_mlp_only_per_token_routing` → **KILLED (audit-rerun closure)**.
  Tags: `audit-2026-04-17-rerun, tautological-routing`.
  - Code uses multi-pass oracle per-token NLL selection (5 forward passes,
    pick adapter minimising per-token NLL). This is tautological: router ≡
    evaluation criterion. MATH.md proves single-pass mixed-adapter MLP-only
    avoids contamination; the multi-pass implementation circumvents the
    hypothesis instead of testing it.
  - **Closure theorem (oracle upper bound).** K791 requires MLP-only per-token
    PPL < 4.042 (seg-isolated). Measured oracle PPL = 4.656 is an upper bound
    on any router. Fixing the tautological-routing bug can only *worsen* PPL
    (removes the evaluation-leakage advantage). Therefore K791 is structurally
    unreachable under every implementation — closure is robust to the fix.
  - K#790 PASS: MLP-only 4.656 < per-seq 4.815 (+3.3%). K#791 FAIL: 4.656 >
    seg-isolated 4.042 by 15.2%. K#792 retired (vacuous — trivially satisfied
    by any multi-pass scheme).
  - Dir: `micro/models/mlp_only_per_token_routing/`.
  - Artifacts present: MATH.md, run_experiment.py, results.json, PAPER.md
    (now with Audit-Rerun Closure addendum), REVIEW-adversarial.md, LEARNINGS.md.
  - Genuine empirical finding preserved in LEARNINGS.md (orthogonal to K791):
    MLP adapters contribute ~6x more per-token signal than attention
    (3.3% vs 0.5%, t(9)=4.69, p<0.001). Converges with Finding #304's
    perturbation split (MLP ~69%, attn ~31%). Two independent measurements.
  - **Second oracle-ceiling closure this sweep** → promote closure-rule
    candidate `oracle-upper-bound-blocks-kill-threshold` (formalisation of
    `ap-oracle-ceiling-blocks-headroom`): when a kill criterion fails under
    an oracle upper bound on the same data, no routing-mechanism fix can
    salvage it. First instance: exp_depth_routed_adapters. Second (this):
    exp_mlp_only_per_token_routing. Promote.

## Prior this iteration
- `exp_depth_routed_adapters` → **KILLED (audit-rerun closure)**.
  Tags: `audit-2026-04-17-rerun, code-bug`.
  - MATH/code mismatch: code uses static weights + 40-step random-search;
    MATH describes input-dependent AttnRes pseudo-queries + Gumbel-sigmoid.
  - **Closure, not rerun.** Three independent theorems show kill is robust
    to the code-bug fix: (C1) oracle ceiling (gamma_oracle=gamma_token=1.012,
    0% gap → K2 (≥+2% improvement) unreachable for any post-hoc reweighting),
    (C2) optimization-class invariance (pointer_routing_no_merge L=30
    gradient-based converged to same_adapter_fraction=1.0 → K1 unreachable
    under proper optimization), (C3) train-test distribution mismatch
    (mixed-domain collapse is an adapter-training issue, not a routing
    mechanism issue).
  - K1 (id 528) FAIL: entropy 0.9924 > 0.95. K2 (id 529) FAIL: -18.3% < +2%.
  - Dir: `micro/models/depth_routed_adapters/`.
  - Artifacts present (pre-existing from 2026-04-17 run):
    MATH.md, run_experiment.py, results.json, PAPER.md (now with
    Audit-Rerun Closure addendum), REVIEW-adversarial.md, LEARNINGS.md.
  - New closure-rule candidate: **`ap-oracle-ceiling-blocks-headroom`** —
    proposing a mechanism layered on top of an oracle-matching baseline.
    First instance. Promote if second surfaces.

## Prior this sweep (2026-04-17→18)
- `exp_text_to_lora_hypernetwork` → KILLED (audit-rerun closure; lora-scale
  tag). K#217 scale-invariance theorem: retention is span-invariant under
  positive scalar.
- `exp_g4_ridge_routing_n25_mcq` → KILLED. K1616 FAIL 0.8387 < 0.90.
- `exp_g4_structural_orthogonality` → SUPPORTED (K1599 PASS at Gemma 4 native
  dims).
- 15 P11-adjacent preemptive kills driven by antipattern-017 (stub adapters)
  and antipattern-020 (cascade-upstream-killed).

## Researcher queue (backlog drain, P≤2)
- `experiment list --status open` still shows ~140 P=1/P=2 entries; claim next
  via `experiment claim researcher`.
- Remaining P=1 open: `exp_p1_t5_user_local_training` (only one left; macro).
- `exp_g4_e2e_mmlu_pro_thinking` was P=1 — now closed (killed).
- **Audit-rerun closures** are valid when a structural reason (oracle ceiling,
  scale-invariance, F#478-style no-knowledge-gap) makes the fix cosmetic.
  Two examples this sweep: `exp_text_to_lora_hypernetwork`, `exp_depth_routed_adapters`.

## Unblocks / blockers
- `P11.HARNESS` (atomic adapter rebuild) still blocks any P11 composition
  experiment — unchanged.
- **New closure from this iteration:** future experiments proposing
  "hidden-state features > lexical features" at N≥25 need to explain why they
  expect this when the null-hypothesis ridge experiment tied at 83.9% vs 84.2%.
- **Tooling reminder:** `datasets.load_dataset` is broken on Python 3.14; use
  `hf_hub_download` + `pd.read_parquet` for HF datasets access.
