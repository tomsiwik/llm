# Pierre Research — Agent Rules

## Project

Composable domain experts via M2P distillation on Apple Silicon (M5 Pro 48GB, MLX only).
Frozen base model + Grassmannian A-matrices (QR orthogonality) + M2P-generated B-matrices.

## Platform

- **Hardware:** Apple M5 Pro, 48GB unified memory
- **Framework:** MLX only. No CUDA. No PyTorch GPU.
- **Python:** `uv run` for all Python execution
- **Experiments:** `experiment` CLI for tracking (Turso remote DB)
- **Queue:** `pueue` for experiment execution

## Research Loop

Each iteration:

1. Read `docs/tasks/progress.txt` for what's done
2. Read `.ralph/current_direction.md` for current focus
3. Run `experiment claim researcher` — get next experiment
4. If no `run_experiment.py`: write MATH.md + script (invoke /fast-mlx /mlx-dev first)
5. Run: `experiment run <id>`
6. Read results.json. Write PAPER.md with prediction-vs-measurement table.
7. Review own work: write REVIEW-adversarial.md (max 3 blocking issues)
8. Complete: `experiment complete <id> --status supported --dir micro/models/<name>/ --k <id>:pass --evidence "..."`
9. Record: `experiment finding-add --title "..." --status supported --result "..." --caveat "..."`
10. Append progress to `docs/tasks/progress.txt`
11. Loop to step 1

When `experiment list -s open` returns empty: append `RESEARCH_BACKLOG_DRAINED` to progress.txt.

## Hats (role-based behavior)

### @researcher
Design and run experiments. MATH.md before code. Max 40 tool uses.
If stuck >15 min, wrap partial results and move on.

### @reviewer
Read MATH.md + PAPER.md + results.json. Write REVIEW-adversarial.md.
Max 3 blocking fixes. Max 2 review rounds. Then PROCEED with caveats.

### @analyst
Read PAPER.md + REVIEW-adversarial.md. Write LEARNINGS.md (max 30 lines).
Max 10 tool uses. No heavy research in the loop.

## Code Rules

- Proof-first: MATH.md before code, Theorem/Proof/QED
- Every experiment needs kill criteria with specific numbers
- Every hypothesis MUST cite an arxiv paper or prior finding
- Memory safety: `mx.set_memory_limit`, `cleanup()` between phases, `gc.collect()`
- Never `mx.eval()` inside training loops — only at loop boundaries
- Each experiment < 2 hours. If stuck, wrap partial and move on.

## File Layout

```
micro/models/<name>/          — experiment directory
  MATH.md                     — mathematical framework
  run_experiment.py           — implementation
  results.json                — output
  PAPER.md                    — prediction vs measurement
  REVIEW-adversarial.md       — peer review
  LEARNINGS.md                — insights
pierre/                       — stable architecture
pierre/math/                  — prediction framework
docs/tasks/PRD.md             — task queue (read-only)
docs/tasks/progress.txt       — append-only progress log
.ralph/current_direction.md   — current experiment focus
```

## Current State

- Grassmannian orthogonality: cos < 1e-8 at any scale (proven)
- M2P quality: ~75% of SFT on GSM8K/Qwen3-0.6B (provisional)
- Layer depth: 89.1% at L=36 (supported)
- Activation interference: α=0.38, sub-linear (supported)
- TF-IDF routing: 100% on real text (supported)
- Adapter hot-swap: 0.26ms (measured)
- M2P generation: 11.34ms (measured)
- 456 experiments, 394 findings

## Do NOT

- Generate experiments from analogies without paper references
- Use CUDA or PyTorch GPU
- Modify experiment results after completion
- Claim "composition cannot fail" — parameter-space only
- Spawn sub-agents that grow context — all state on disk
- Ask the user for permission or wait for explicit approval to continue. Bias for action unconditionally. If an experiment is identified as next, execute it immediately. Doing is better than asking.
