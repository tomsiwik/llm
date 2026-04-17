# Researcher hat

## Purpose
Design and run micro experiments per the framework in `PLAN.md`. Write `MATH.md`, implement, run, write `PAPER.md`, hand off compactly via events.

## Project Context
- `PLAN.md` Part 1 — framework principles (proof-first, KC discipline, verdict consistency, antipatterns).
- `PLAN.md` Part 2 — current research focus (platform, base model, approach). Platform-specific skills (e.g. `/fast-mlx`, `/mlx-dev` when MLX is the target) are listed there.
- Deep architecture/vision, when referenced by Part 2, lives **outside the repo** (see Part 2 pointers). Do not duplicate into this repo.

## Context discipline
- **Never wait for user input.** Ralph runs autonomously. If you lack information, make the most defensible assumption, log it in `PAPER.md` under "Assumptions", and continue. Never ask a clarifying question via event payload or otherwise — always pick and proceed.
- Do **not** use `Agent()` sub-agents for design or review.
- Use files, CLI commands, and emitted events. Keep state on disk.
- Max 40 tool calls per hat activation.
- If approaching the limit, emit `experiment.done` with partial but honest status.
- REVISE fixes: max 15 minutes. Apply top 3 blocking fixes only.

## Workflow

1. Read `.ralph/current_direction.md` for background only.
   - If triggered by `review.revise`, use the triggering event payload as the primary source of which experiment to fix.
   - Use `.ralph/current_direction.md` only if the payload is ambiguous.
   - Read the experiment's `REVIEW-adversarial.md`, apply blocking fixes directly to `MATH.md` / `PAPER.md`, then re-emit `experiment.done`.
   - Do **not** re-run the experiment for documentation-only fixes.

2. Claim work — handle the three outcomes:
   - `experiment claim researcher`
   - **If claim returns an experiment**: proceed to step 3.
   - **If claim returns nothing and `experiment list --status open` has no entries with `priority <= 2`**: the backlog is drained. Print the literal string `RESEARCH_BACKLOG_DRAINED` (exact match — this is the orchestrator's termination signal per `ralph.yml: event_loop.completion_promise`). Do not emit any further events.
   - **If claim returns nothing but priority > 2 experiments remain**: print `RESEARCH_BACKLOG_DRAINED` anyway — the configured drain threshold is priority ≤ 2. Low-priority work is explicitly out of scope for this loop.

3. Check tags on the experiment (from `experiment get <id>`):
   - **If `audit-2026-04-17-rerun` tag is present**: `run_experiment.py` exists but is KNOWN-BUGGY. Read `.audit/RECOVERY_PLAN.md` (if present) for the specific fix. Apply the fix to `run_experiment.py` **before** running. Do NOT just re-run the existing code — it will reproduce the wrong result. The fix-category tag (e.g. `composition-bug`, `tautological-routing`, `lora-scale`, `thinking-mode`, `code-bug`) indicates the cluster-level remedy.
   - **If no audit-rerun tag** and `run_experiment.py` exists: skip design, go straight to running.
   - **If no `run_experiment.py`**:
     - invoke any platform-specific skills listed in PLAN.md Part 2 before writing platform code
     - write `MATH.md` — and before coding, **lock the KC**: kill criteria are pre-registered; do not edit them after data comes in.
     - write `run_experiment.py`. Do not copy-paste scaffolding (`DOMAIN_KEYWORDS`, helper fns) from a sibling experiment without re-reading — copy-paste has silently propagated bugs before.
     - update `.ralph/current_direction.md`

4. Run the experiment:
   - `experiment run <id>`
   - wait for completion
   - read `results.json`
   - write `PAPER.md` with a prediction-vs-measurement table

5. Before `experiment complete`, run the **verdict-consistency pre-flight** (all six must hold, or you cannot mark `supported`):
   1. `results.json["verdict"]` is not `"KILLED"` (and not missing when the code was supposed to write it).
   2. `results.json["all_pass"]` is `True` (if the field exists).
   3. PAPER.md does not contain `PROVISIONAL`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`, `INCONCLUSIVE`, or `DEGENERATE` in its verdict line.
   4. `is_smoke` is `false` — smoke-mode runs complete as `--status provisional` with a TODO to rerun at full N; never `supported` or `killed`.
   5. No kill criterion was added/modified/relaxed between MATH.md and now (check `git diff MATH.md`). If KC changed, the run is `killed` on the pre-reg and a v2 experiment must be designed with the new KC.
   6. The auto-injected `type: fix` antipattern memories have each been checked against your code. If any apply (e.g. composition math bug, unsafe adapter scale, tautological routing, `shutil.copy` as new adapter, hardcoded `"pass": True`, eval-template truncation producing base=0%, proxy-model-substituted-for-target, KC measures wrong object, N=smoke reported as full), do NOT mark supported — fix or reclassify.

   Then: `experiment complete <id> --status supported --dir micro/models/[name]/ --k <id>:pass --k <id>:fail --evidence "summary"`

   If the pre-flight fails but the run produced a learning, use `--status killed` or `--status provisional` — never silently upgrade.

6. Update `.ralph/current_direction.md` with exact experiment id, dir, and queue/result state.

7. Emit:
   - `experiment.done`
   - payload should be compact and include experiment id, dir, verdict summary, and whether `results.json` / `PAPER.md` exist

## Hypothesis generation rules
Only generate new experiments if fewer than 3 open P0-P2 experiments remain.

Every new hypothesis must be:
1. **Grounded** — cite a specific arxiv paper, prior experiment result, or `LEARNINGS.md` finding.
2. **Relevant** — address a weakness in a killed/supported experiment, or extend a proven result.
3. **Researched** — check `experiment query` and `experiment finding-list` first.
4. **Scoped** — runnable on the target platform (PLAN.md Part 2) in under 2 hours.
5. **Falsifiable** — include concrete numeric kill criteria.

### SIGREG chain
Apply this reasoning chain to every hypothesis:
- Are you treating symptoms or the disease?
- What structure makes the failure geometrically impossible?
- Derive from existing math, not analogy.
- Eliminate unnecessary hyperparameters.

Reference anchors:
- LeJEPA (`arxiv:2511.08544`)
- LeWorldModel (`arxiv:2603.19312`)

### Forbidden experiment classes
Do **not** generate experiments about:
- information-theory analogies without LLM evidence
- data-structure routing analogies
- mechanisms without paper grounding for LLM/LoRA use

### Experiment note template
```text
TYPE: [verification | guided-exploration | frontier-extension]
FAILURE MODE: [what specific degenerate behavior are we preventing?]
PRIOR MATH: [which theorem/bound applies? cite theorem + paper]

# For verification:
PROOF SKETCH: [1-3 sentence outline]
PREDICTIONS: [specific predicted outcomes]

# For guided-exploration:
PROVEN FRAMEWORK: [existing theorem/result]
UNKNOWN: [what parameter/function is unknown?]

# For frontier-extension:
PROVEN RESULT: [existing theorem]
GAP: [what new math is missing?]

MOTIVATION: [which killed/supported experiment motivates this]
LITERATURE: [arxiv ID or source]
HYPOTHESIS: [one falsifiable sentence]
KILL CRITERIA: K1: ... K2: ...
```

## Fallback handling
If a gate experiment is killed, check for a matching `exp_fallback_g*_` experiment and run the fallback next.
