# Agent Reference

## Platform
Apple M5 Pro 48GB. MLX only. No CUDA.

## Project Context
- `../VISION_P1.md` — current architecture (Gemma 4 + PoLAR + Grassmannian)
- `../ARCHITECTURE_P1.md` — mathematical reference (every mechanism formalized)
- `docs/MLX_GEMMA4_GUIDE.md` — HOW TO: load, train, adapt Gemma 4 on MLX

## Skills
`/fast-mlx` `/mlx-dev` before MLX code | `/experiment` before CLI commands | `/paper2code` to implement papers | `/notebooklm` for literature (manual sessions only)

## Experiment CLI
```bash
experiment claim <worker>                         # pick next, get full YAML
experiment run <id>                               # run via pueue (MANDATORY)
experiment complete <id> --status supported \     # finish in one shot
  --dir micro/models/<name>/ --k <kill-id>:pass --evidence "K1 PASS: val"
experiment finding-add --title "..." --status supported --result "..." \
  --caveat "..." --failure-mode "..." --impossibility-structure "..."
experiment query "search"                         # FTS everything
experiment ref-add --arxiv <id> --title "..." --relevance "..."
```
Status: conclusive | supported | provisional | killed

## Proof-First
MATH.md (theorem+predictions) → code (verify on MLX) → PAPER.md (prediction vs measurement table)

## MLX Memory Safety
Use phased execution pattern — each compute phase in its own function:
```python
def phase_train(...):
    model = load(...)
    # train
    cleanup(model)
    return results

def phase_eval(...):
    model = load(...)
    # eval
    cleanup(model)
    return metrics

def cleanup(*objects):
    for obj in objects: del obj
    gc.collect()
    mx.clear_cache()
```
See `/mlx-dev` skill for full memory management reference.
