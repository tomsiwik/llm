# Project: Composable Ternary Experts

## Platform
Apple M5 Pro, 48GB. MLX only. No CUDA.

## Before Writing MLX Code
Invoke `/fast-mlx` and `/mlx-dev` skills first.

## Before/After Experiments
```bash
experiment claim <worker-id>                    # get next experiment + full details
# ... run experiment ...
experiment complete <id> --status supported \   # finish in one shot
  --dir micro/models/<name>/ \
  --k 183:pass --k 184:fail \
  --evidence "K1 PASS: metric=value"
```

## Key Files
- `PROMPT.md` — research loop instructions
- `VISION.md` — architecture, proven results, readiness
- `FINDINGS.md` — all evidence
- `CODING_GUIDELINES.md` — mandatory MLX memory/cleanup patterns
- `.ralph/current_direction.md` — what's in progress
