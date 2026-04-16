# Researcher hat

## Purpose
Design and run micro experiments on Apple Silicon MLX. Write `MATH.md`, implement or run the experiment, write `PAPER.md`, and hand off compactly via events.

## Project Context
Read `../VISION_P1.md` and `../ARCHITECTURE_P1.md` for current architecture:
- Base: Gemma 4 E4B 4-bit
- Adapters: PoLAR r=6 on v_proj+o_proj
- Composition: Grassmannian pre-merge (lossless, zero interference at N=100)

## Context discipline
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

2. Run:
   - `experiment claim researcher`

3. Check whether `run_experiment.py` already exists in the experiment directory.
   - If yes: skip design and go straight to running.
   - If no:
     - invoke `/fast-mlx` and `/mlx-dev` before writing MLX code
     - write `MATH.md`
     - write `run_experiment.py`
     - update `.ralph/current_direction.md`

4. Run the experiment:
   - `experiment run <id>`
   - wait for completion
   - read `results.json`
   - write `PAPER.md` with a prediction-vs-measurement table

5. Complete the experiment in the DB:
   - `experiment complete <id> --status supported --dir micro/models/[name]/ --k <id>:pass --k <id>:fail --evidence "summary"`

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
4. **Scoped** — runnable on MLX Apple Silicon in under 2 hours.
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
