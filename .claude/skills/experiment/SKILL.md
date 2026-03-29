---
name: experiment
description: |
  Research experiment tracking CLI (Turso remote DB). Invoke this skill before
  using ANY experiment command to get exact flag names and syntax.
argument-hint: <command> [options]
---

# Experiment CLI — Complete Reference

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

---

## All Commands — Exact Signatures

### Experiments: Create & Update

```bash
# ADD — one-shot: tags, kill criteria, and deps inline (no follow-up commands needed)
experiment add <id> --title "..." [--scale micro|macro] [--priority 0-99] \
  [--platform local|local-apple|runpod-flash] [--dir <path>] \
  [--notes "..."] [--grounded-by <arxiv-id>] \
  [--tag composition --tag novel] \
  [--kill "AUC > 0.75" --kill "beats random baseline"] \
  [--dep exp_other_experiment]

# UPDATE — any field
experiment update <id> [--status open|active|proven|supported|killed] \
  [--priority <n>] [--platform <p>] [--dir <path>] [--notes "..."] [--title "..."]
```

### Experiments: Query & List

```bash
experiment get <id>                           # full details (human-readable)
experiment get <id> --yaml                    # full details (structured YAML)
experiment list [-s open,active] [-t <tag>] [--blocking]
experiment query "search text"                # FTS across experiments + evidence + findings
experiment stats                              # dashboard with counts
```

### Kill Criteria

```bash
# ADD — use --text (NOT --description)
experiment kill-add <id> --text "metric > threshold" [--reason "why"]

# UPDATE — use --criterion <kill-id> and --result
experiment kill-update <id> --criterion <kill-id> --result pass|fail|inconclusive
```

### Success Criteria

```bash
experiment success-add <id> --condition "..." --unlocks "..." \
  [--reason "..."] [--max-followup 1]
```

### Tags & Dependencies

```bash
# TAGS — positional args OR --add flags both work
experiment tag <id> routing mlx critical-path
experiment tag <id> --add routing --add mlx --add critical-path

# DEPENDENCIES — use --on
experiment dep-add <id> --on <depends-on-id>
```

### Evidence

```bash
experiment evidence <id> --claim "K1 PASS: value=X" --source results.json \
  [--verdict pass|fail|inconclusive] [--date 2026-03-29]
```

### References

```bash
experiment ref-add --title "..." --relevance "..." \
  [--arxiv <id>] [--url <url>] [--local-path <path>] \
  [--repro-steps "..."] [--tag <t>]...

experiment refs [--unused] [-t <tag>]
```

### Findings

```bash
# ADD — all flags, no positional args
experiment finding-add \
  --title "..." \
  --status conclusive|supported|killed|provisional \
  --result "..." \
  [--caveat "..."] \
  [--experiment <exp-id>] \
  [--scale micro|macro] \
  [--failure-mode "what degenerate behavior?"] \
  [--impossibility-structure "what math makes it impossible?"] \
  [--date 2026-03-29]

# LIST — filter by status and/or scale
experiment finding-list [--status conclusive|supported|killed|provisional] [--scale micro|macro]

# GET — positional finding ID (numeric)
experiment finding-get <finding-id>

# UPDATE — any field by finding ID (numeric)
experiment finding-update <finding-id> \
  [--title "..."] [--status conclusive|supported|killed|provisional] \
  [--result "..."] [--caveat "..."] [--experiment <exp-id>] \
  [--failure-mode "..."] [--impossibility-structure "..."]
```

### Claim Management

```bash
experiment claim <worker>                             # auto-pick next
experiment claim <worker> --id <exp-id>               # claim specific
experiment claim <worker> --tag <tag>                  # filter by tag
experiment claim <worker> --max-priority 1             # only high-priority
experiment claim <worker> --release --release-id <id>  # release stale claim
```

---

## Common Mistakes to Avoid

| Wrong | Right | Why |
|-------|-------|-----|
| `--hypothesis "..."` | `--notes "..."` | `add` has no --hypothesis flag |
| `--description "..."` | `--text "..."` | `kill-add` uses --text |
| `experiment dep <a> <b>` | `experiment dep-add <a> --on <b>` | command is dep-add with --on |
| `--note "..."` | `--notes "..."` | plural form |
| separate add + tag + kill-add | `add --tag x --kill "..." --dep y` | one-shot add does it all |

### Method Bank

```bash
# ADD — technique with problem context, applicability conditions, and source
experiment method-add \
  --name "Method Name" \
  --description "What it does" \
  --solves "What problem it addresses" \
  [--proven-in "Where it was proven to work"] \
  [--use-when "Conditions to reach for this method"] \
  [--not-now-because "Why we are not using it right now"] \
  [--source "arxiv ID, repo URL, paper"] \
  [--status parked|exploring|applied|rejected] \
  [--tag routing --tag scaling]

# LIST — browse by status, tag, or problem (FTS)
experiment method-list [--status parked] [--tag routing] [--problem "interference"]

# GET — full details
experiment method-get <method-id>

# UPDATE — change status, conditions, etc.
experiment method-update <method-id> [--status exploring] [--use-when "..."] [--not-now-because "..."]
```

Method statuses: `parked` (interesting, not now) → `exploring` (active experiment) → `applied` (in production) | `rejected` (proven not to work)

---

## Status Flow

`open` → `active` (via claim) → `supported` | `proven` | `killed` (via complete)

## Finding Status Definitions

- `conclusive` = formal proof verified, all predictions match measurements
- `supported` = proof mostly verified, or exploration narrowed an unknown
- `provisional` = frontier extension, or empirical observation awaiting proof
- `killed` = proof's predictions refuted, or proof found incorrect
