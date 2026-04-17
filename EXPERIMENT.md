# Researcher reference card

## Experiment CLI
```bash
experiment claim <worker>                         # pick next, get full YAML
experiment run <id>                               # run via pueue (MANDATORY — never bare `uv run python`)
experiment complete <id> --status supported \     # finish in one shot
  --dir micro/models/<name>/ --k <kill-id>:pass --evidence "K1 PASS: val"
experiment finding-add --title "..." --status supported --result "..." \
  --caveat "..." --failure-mode "..." --impossibility-structure "..."
experiment query "search"                         # FTS across experiments + evidence + findings
experiment ref-add --arxiv <id> --title "..." --relevance "..."
```
Status values: `open | active | supported | killed | proven | provisional`

## Proof-first output
`MATH.md` (theorem + predictions) → code (verify on platform) → `PAPER.md` (prediction-vs-measurement table) → `REVIEW-adversarial.md` (self-review).

## Memory safety (platform-specific patterns)
Use a phased execution pattern — each compute phase in its own function with explicit cleanup between phases. Platform details (MLX `mx.eval` / `mx.clear_cache` discipline, unified-memory rules, `mlx-lm` version) live in PLAN.md Part 2 and the invoked platform skill.

## Required reading before code
Invoke the skills listed in PLAN.md Part 2 before writing platform code.
