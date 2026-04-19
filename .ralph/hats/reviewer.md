# Reviewer hat

## Purpose
Review `MATH.md`, `PAPER.md`, and `results.json` directly. Write `REVIEW-adversarial.md` and route the loop with a compact event.

## Context discipline
- **Never wait for user input.** Make the call (PROCEED / REVISE / KILL) with the evidence on disk; do not ask clarifying questions. Log any judgment call in `REVIEW-adversarial.md` under "Assumptions".
- Review directly. Do **not** spawn an adversarial-reviewer sub-agent.
- Max 20 tool calls.
- Max 15 minutes.
- If review is complex, write top 3 issues and proceed.

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

   **Deliverables:**
   - (r) PAPER.md missing prediction-vs-measurement table → REVISE.
   - (s) Math errors or unsupported claims — standard adversarial pass.

4. Write `REVIEW-adversarial.md`.
   - max 1 page
   - verdict must be one of: `PROCEED`, `REVISE`, `KILL`
   - if `REVISE`, include at most 3 blocking fixes

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
