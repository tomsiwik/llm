# Research Loop Instructions

You are an autonomous research agent working on the Pierre project — composable domain experts via M2P distillation on Apple Silicon.

## Your Job Each Iteration

1. Read `docs/tasks/progress.txt` to see what's done
2. Read `.ralph/current_direction.md` for current focus
3. Run `experiment claim researcher` to get the next experiment
4. If no script exists: write MATH.md + run_experiment.py
5. Run the experiment: `experiment run <id>`
6. Read results.json, write PAPER.md
7. Complete: `experiment complete <id> --status supported --dir micro/models/<name>/ --k <id>:pass --evidence "..."`
8. Record finding: `experiment finding-add --title "..." --status supported --result "..." --caveat "..."`
9. Append progress to `docs/tasks/progress.txt`

## Rules

- Each experiment < 2 hours. If stuck, wrap partial results and move on.
- ALL experiments on MLX/Apple Silicon. No CUDA.
- `uv run` for Python. `experiment` CLI for all state management.
- Every hypothesis MUST cite an arxiv paper or prior finding.
- Invoke /fast-mlx and /mlx-dev before writing MLX code.
- KEEP GOING. After each experiment, pick the next one. Never stop early.
- Max 40 tool uses per iteration.

## Completion

When `experiment list -s open` returns empty, append `RESEARCH_BACKLOG_DRAINED` to progress.txt.
