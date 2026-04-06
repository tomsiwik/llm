# Agent Reference

## Platform
Apple M5 Pro 48GB. MLX only.

## Skills
`/fast-mlx` `/mlx-dev` before MLX code | `/experiment` before CLI commands | `/huggingface-papers` for papers | `/paper2code` to implement papers | `/notebooklm` for literature

## Experiment CLI
```bash
experiment claim <worker>                         # pick next, get full YAML
experiment complete <id> --status supported \      # finish in one shot
  --dir micro/models/<name>/ --k <kill-id>:pass --evidence "K1 PASS: val"
experiment finding-add --title "..." --status supported --result "..." \
  --caveat "..." --failure-mode "..." --impossibility-structure "..."
experiment query "search"                         # FTS everything
experiment ref-add --arxiv <id> --title "..." --relevance "..."
```
Status: conclusive | supported | provisional | killed

## Proof-First
MATH.md (theorem+predictions) → code (verify on MLX) → PAPER.md (prediction vs measurement table)
