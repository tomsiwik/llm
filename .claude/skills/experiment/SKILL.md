---
name: experiment
description: |
  Research experiment tracking (Turso remote DB). Use for:
  - Claiming and completing experiments
  - Querying experiment details, status, and evidence
  - Adding experiments, kill/success criteria, evidence, references, tags, dependencies
  - Full-text search across all experiment data
argument-hint: <command> [options]
---

# Experiment CLI

## Core Workflow (2 commands per experiment)

```bash
# 1. CLAIM — atomically pick next experiment, get full YAML details
experiment claim <worker-id>
experiment claim <worker-id> --id <specific-id>     # claim specific
experiment claim <worker-id> --tag optimization      # filter by tag
experiment claim <worker-id> --max-priority 1        # only P0-P1

# 2. COMPLETE — finish in one shot: status + kill results + evidence + dir
experiment complete <id> --status supported \
  --dir micro/models/<name>/ \
  --k 183:pass --k 184:pass --k 185:fail \
  --evidence "K1 PASS: PPL=1.05, K2 PASS: ratio=1.02, K3 FAIL: cos=0.08" \
  --source results.json
```

`claim` outputs full YAML: kill criteria with IDs, success criteria, deps, blocks, tags, refs, notes.
`complete` updates kill criteria, adds evidence, sets status+dir, clears claim — all in one call.

## Other Commands

```bash
experiment list --status open,active          # see available work (shows WORKER column)
experiment list --blocking                    # critical-path experiments
experiment get <id> --yaml                    # full structured details
experiment stats                              # dashboard

# Add new experiment
experiment add <id> --title "..." --scale micro --priority 0 --platform local-apple

# Add structured metadata
experiment kill-add <id> --text "K1: metric > threshold -> KILL"
experiment success-add <id> --condition "..." --unlocks "..."
experiment dep-add <id> --on <depends-on-id>
experiment tag <id> --add routing --add mlx

# Add evidence (when not using complete)
experiment evidence <id> --claim "..." --source "..." --verdict pass

# Search
experiment query "ternary routing"

# References
experiment refs --tag composition
experiment ref-add --arxiv 2505.22934 --title "OSRM" --relevance "..." --tag composition

# Release stale claim
experiment claim <worker> --release --release-id <id>
```

## Status Flow

`open` → `active` (via claim) → `supported` | `proven` | `killed` (via complete)
