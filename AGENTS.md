- Use `experiment` CLI for tracking experiments

## Research Loop

1. Load `experiment` skill to familiarize yourself with the cli
2. Check `experiment list -s open` for open experiments
3. Run `experiment claim` to get next experiment
4. If no `run_experiment.py` exists: write MATH.md to evaluate the math on the experiment + script (invoke /fast-mlx /mlx-dev skills first)
5. Run: `experiment run <id>` to execute script (uses `pueue` and it will be queued)
6. Read `results.json` after experiment ran (can take a while). Write PAPER.md with prediction-vs-measurement table.
7. Review own work: write REVIEW-adversarial.md (max 3 blocking issues)
8. Complete: `experiment complete <including all props>`
9. Record: `experiment finding-add <including all props>`
10. Loop to step 1