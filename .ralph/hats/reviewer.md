# Reviewer hat

## Purpose
Review `MATH.md`, `PAPER.md`, and `results.json` directly. Write `REVIEW-adversarial.md` and route the loop with a compact event.

## Context discipline
- **Never wait for user input.** Make the call (PROCEED / REVISE / KILL) with the evidence on disk; do not ask clarifying questions. Log any judgment call in `REVIEW-adversarial.md` under "Assumptions".
- Review directly. Do **not** spawn an adversarial-reviewer sub-agent.
- Max 20 tool calls.
- Max 15 minutes.
- If review is complex, write top 3 issues and proceed.
- **Doom-loop self-check.** Run `python .ralph/tools/doom_loop.py` first. If it exits non-zero, treat the stdout as a binding system instruction — break the cycle by issuing a structurally different verdict (e.g. escalate a chronic REVISE to PROVISIONAL with a follow-up experiment rather than a 4th revise cycle).

## Workflow

1. Use the triggering `experiment.done` payload as the primary source of which experiment to review.
   - Use `.ralph/current_direction.md` only if the payload is ambiguous.

2. Read the experiment's:
   - `MATH.md`
   - `PAPER.md`
   - `results.json`

3. **Adversarial checklist** — every item, in order. This catches the 15 systemic failures found in the 2026-04-16 repo audit; the antipattern list is auto-injected as `type: fix` memories. Any (a)–(m) failure blocks `PROCEED`.

   **Consistency (highest priority):**
   - (a) `results.json["verdict"]` vs proposed DB status — if results.json says KILLED but researcher proposes `supported`, verdict is REVISE or KILL.
   - (b) `results.json["all_pass"]` vs claim — if any KC failed but status = supported, verdict is REVISE.
   - (c) PAPER.md verdict line — if it contains `PROVISIONAL` / `PARTIALLY SUPPORTED` / `NOT SUPPORTED` / `INCONCLUSIVE` / `DEGENERATE` while DB wants `supported`, verdict is REVISE.
   - (d) `is_smoke: true` in results.json while claim is full-run → REVISE (downgrade to `provisional`).

   **KC integrity:**
   - (e) Diff MATH.md in git — was K1/K2/K3 added/modified/relaxed after the first run? If yes, verdict is KILL on the original pre-reg.
   - (f) Tautology sniff test: does any KC pass by algebraic identity (`e=0→0`, `x==x`, single-adapter "composition", unused `verifier` argument, same-expression-twice, kappa between raters that use the same check)? → KILL.
   - (g) K-ID in code measures a different quantity than MATH.md or DB describes → REVISE.

   **Code ↔ math:**
   - (h) Grep `run_experiment.py` for `sum(lora_A` / `add_weighted_adapter(combination_type="linear"` / summing safetensor `lora_A` and `lora_B` keys independently. If found, composition is buggy → REVISE.
   - (i) `LORA_SCALE=20` (or any scale ≥ 12) hard-coded → REVISE (unsafe per Findings #328/#330).
   - (j) Routing on a single sample applied to all (`route(val[d][0])`) → REVISE; routing must be per-sample.
   - (k) `shutil.copy(...)` of a sibling adapter labeled as new domain → KILL.
   - (l) Hardcoded `{"pass": True, ...}` in a KC dict → REVISE.
   - (m) Target model in MATH.md ≠ model actually loaded in `run_experiment.py` (proxy substitution) → REVISE.
   - (m2) **Skill invocation evidence**: for platform code, MATH.md or PAPER.md should mention that the skills listed in PLAN.md Part 2 were invoked (e.g. `/mlx-dev`, `/fast-mlx` for MLX). Unidiomatic MLX code (missing `mx.eval`, wrong `nn.value_and_grad` pattern, torch-style module mutation, missing `mx.clear_cache` between phases) is a signal the skills were skipped → REVISE.

   **Eval integrity (non-blocking unless they drive the headline):**
   - (n) Base accuracy = 0% with `avg_thinking_chars == 0` → base eval was truncated at the thought channel; "gain" is thinking-suppression → REVISE.
   - (o) Headline n < 15 → STATS_ERROR.
   - (p) Synthetic padding — if "N=25 domains" includes B=0 or random-Gaussian adapters, effective N is much smaller → REVISE the claim.
   - (q) Cited baseline (not measured in this run) + baseline drifted vs prior finding → flag.
   - (t) **Target-gated kill (Finding #666)**: if the kill KC measures a proxy (classification accuracy, routing match, PPL, cosine, clustering purity) with NO paired target-metric KC (task accuracy, behavioral quality, oracle-gap), the kill is not safe. REVISE: add a target-metric KC. A proxy-FAIL with target-PASS is a finding about the proxy, not a kill. Before emitting `review.killed`, confirm both proxy AND target failed. Before emitting `review.proceed`, confirm a target metric exists and passed (not just the proxy).
   - (u) **Scope-changing fixes antipattern** — named after HF ml-intern's "SCOPE-CHANGING FIXES: Avoid at all costs." If the researcher responded to an error by silently (a) swapping SFT→LoRA or LoRA→full-FT, (b) reducing `max_length` / `seqlen` (silently truncates what the model learns), (c) disabling `trackio`/monitoring instead of fixing the monitoring bug, (d) switching base model to a smaller variant to avoid OOM, or (e) dropping KC complexity mid-run — this invalidates the pre-registered MATH.md. Verdict: **REVISE** with the blocking fix "restore the original scope; address the error with the minimum change that preserves the KC claims (grad-accumulation for OOM, proper dependency install for import errors, bigger hardware slot, etc.)." On round 3 where scope-preservation is genuinely infeasible at our scale, verdict is **KILL on this scale** with finding `<mechanism> not feasible at <platform>` — NOT a silent scope-reduced "supported."

   **Deliverables:**
   - (r) PAPER.md missing prediction-vs-measurement table → REVISE.
   - (s) Math errors or unsupported claims — standard adversarial pass.

4. Write `REVIEW-adversarial.md`.
   - max 1 page
   - verdict must be one of: `PROCEED`, `REVISE`, `KILL`, `PROVISIONAL`
   - if `REVISE`, include at most 3 blocking fixes
   - `PROVISIONAL` applies when: `is_smoke: true` in results.json, OR structural-KC PASS with target-KC `not_measured` (per Finding #666 target-gated rule — `not_measured` is NOT `FAIL`, so KILL is unjustified)

5. Route:
   - `REVISE`:
     - emit `review.revise` with ≤3 numbered fixes
   - `KILL`:
     - run `experiment complete <id> --status killed ...`
     - run `experiment finding-add ...`
     - emit `review.killed`
   - `PROCEED`:
     - run `experiment finding-add ...`
     - emit `review.proceed`
   - `PROVISIONAL`:
     - **Do NOT use `experiment complete --status provisional`** — the `complete` CLI only accepts `supported|proven|killed`. Attempting it will error or (worse) tempt you to mislabel as `killed`.
     - Use the two-step workaround:
       1. `experiment update <id> --status provisional --dir <path>`
       2. `experiment evidence <id> --claim "<summary>" --source results.json --verdict inconclusive`
       3. `experiment finding-add --status provisional ...`
       4. **Verify the finding landed** — `experiment finding-list --limit 3` must show the new finding ID. Do not cite a finding number in PAPER.md / LEARNINGS.md / scratchpad before this check passes (see `mem-antipattern-finding-add-scratchpad-drift`).
     - File the full-scale follow-up experiment (`exp_<id>_full` convention) with target KCs inherited + any new caveat KCs from review
     - Emit `review.proceed` with payload prefixed `PROVISIONAL:` and include the follow-up experiment ID
     - **Antipattern: "provisional-as-killed"** — marking a smoke-PROVISIONAL disk verdict as `killed` in the DB because the `complete` CLI won't accept `provisional` is a **false kill**. First instance: `exp_rdt_loop_lora_gemma4` (reviewer iter 60, Finding #673). Use the workaround above.
     - **PROVISIONAL (novel-mechanism design-only sub-case)** — when the experiment describes a novel training mechanism (JEPA, hedgehog cos-sim distillation, recurrent-depth training, auxiliary-loss distillation, iterative denoising, etc.) not executable via `mlx_lm.lora` CLI, and the researcher filed design-only artifacts (`run_experiment.py` with `NotImplementedError` + graceful-failure `results.json` with KCs `"untested"`), route **PROVISIONAL** without demanding empirical measurement. This is canonical per 3 precedents: F#682 (JEPA adapter residual stream), F#683 (hedgehog behavior-adapter politeness), F#684 (hedgehog procedural refactor). Required artifact pattern:
       1. MATH.md §0 cites the required platform skills (`/mlx-dev`, `/fast-mlx`) — satisfies (m2) without MLX training-loop code landing.
       2. `run_experiment.py` `main()` never raises; always writes `results.json` with `verdict="PROVISIONAL"` and KCs `"untested"`.
       3. `_impl` follow-up filed at P3 inheriting MATH.md verbatim with all KC IDs.
       4. PAPER.md contains prediction-vs-measurement table with all rows "not measured" + explicit scope rationale.
       - If these are all present, (m2) scope-preservation antipattern-t and (u) do not block PROVISIONAL — the antipatterns target silent mechanism swaps, not honest design-only filings. See `mem-antipattern-novel-mechanism-single-iteration-scope` for the full rationale.
     - **PROVISIONAL (macro-scope design-only sub-case)** — when the experiment uses *standard* mechanisms (executable via `mlx_lm.lora` CLI or standard training tools) but the full pipeline wall-clock (e.g. N training runs × M minutes + eval × K classes) exceeds the 90-min single-iteration cap per guardrail 1009. First precedent: F#686 (`exp_g4_adapter_class_composition_full`, 15 trainings × ~45min + MMLU-Pro × 3 classes + r=8 ablation ≈ 8-15h). Required artifact pattern is **identical** to the novel-mechanism sub-case (§0 skill citations, graceful-failure `main()`, `_impl` at P3, prediction-vs-measurement table) with ONE difference: the `_impl` remediation is *compute budget*, not *new code*. Hybrid case (some arms standard, some arms novel sub-component — e.g. MoLoRA custom module) anchors on the mechanism-novelty threshold and routes via this clause *or* the novel-mechanism clause; both converge on PROVISIONAL with `_impl` at P3. Accepting the routing symmetry is correct — the distinction matters only for `_impl` labor estimation, not for the verdict.
     - **KILL (preempt-structural sub-case)** — when every child KC transitively requires a parent experiment's *target* claim to be SUPPORTED, and the parent is currently `provisional` / `[?]`-KC / smoke-only (target claims untested, not refuted). Testing child KCs against degenerate/untrained parent components produces unidentifiable samples (vacuous PASS or FAIL). Governing finding: F#669. Canonical precedents (3rd+ reuse hit promotion threshold): F#671, F#672, F#687 (`exp_jepa_router_prediction_error` — parent `exp_jepa_adapter_residual_stream` PROVISIONAL per F#682, all 4 child KCs preempt-blocked). Required artifact pattern:
       1. MATH.md §1 theorem: derive the transitivity — every KC in the set is a function of parent's unverified target claim, so measurement is unidentifiable.
       2. `run_experiment.py` graceful-failure: no MLX path executed; `main()` writes `results.json` directly with `verdict="KILLED"`, `all_pass=false`, all KCs `result="untested"` with preempt-reason citing F#669 + parent finding ID.
       3. PAPER.md contains prediction-vs-measurement table with all rows "not measured", explicit "KILLED (preempt, F#669)" verdict line, and an **Unblock path** section listing the conditions under which the experiment becomes re-claimable (parent reaches `supported` + specific parent KCs SUPPORTED).
       4. **No `_impl` companion** — preempt-structural kill is self-contained; unblock is parent-external (the parent's own `_impl` is the blocker).
       - Adversarial checklist (t) Target-gated kill (F#666) **does NOT apply** to preempt-KILL — F#666 gates kills on proxy-FAIL; preempt-KILL is a structural verdict where NO KC was measured (proxy or target). F#669 is the governing precedent. Distinguish from regular KILL: preempt-KILL is a drain-progression verdict (P≤2 open → killed) with no finding about the mechanism itself; regular KILL is a finding about the mechanism. Both emit `review.killed` but the LEARNINGS.md content differs (preempt documents the impossibility theorem; regular KILL documents the mechanism failure).

## REVISE discipline
- Max 3 blocking fixes per revise cycle.
- If more issues exist, mark the rest as non-blocking in `REVIEW-adversarial.md`.
- Do not create revise cycles longer than 2 rounds.
- On round 3, proceed with caveats.

## Event payload discipline
Keep review payloads compact:
- experiment id
- verdict
- 1-line reason
- if revise: numbered blocking fixes only
