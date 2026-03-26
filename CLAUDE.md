# Project: Composable Ternary Experts

## Target Platform
Apple M5 Pro, 48GB unified memory. MLX 0.31.1. All work must run here. No CUDA. No RunPod.

## Required Skill Invocations

Before writing ANY MLX code, you MUST invoke these skills:
- `/fast-mlx` — performance optimization (lazy eval, compilation, memory, fast ops)
- `/mlx-dev` — correct idiomatic MLX patterns (API gotchas, indexing, NHWC, `__call__`)

Before managing experiments, invoke:
- `/experiment` — use `experiment get <id> --yaml` to see full structured state with completeness warnings

## Coding Rules
- Read `CODING_GUIDELINES.md` before writing ANY experiment script
- Every script: function-scoped phases, cleanup between phases, `mx.eval()` at loop boundaries
- Memory limits: `mx.set_memory_limit(total - 8 * 1024**3)`, `mx.set_cache_limit(2 * 1024**3)`
- Use `uv run` for all Python execution

## Project Context
- `PROMPT.md` — full research loop instructions
- `VISION.md` — architecture and proven results
- `FINDINGS.md` — all experimental evidence
- `.ralph/current_direction.md` — what's being worked on now
