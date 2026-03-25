---
name: experiment
description: |
  Query and manage the research experiment tracking database (Turso remote). Use when:
  - Listing, filtering, or searching experiments by status/tag/dependency
  - Getting full details of an experiment (kill criteria, evidence, references, dependencies)
  - Adding new experiments, evidence, or references
  - Updating experiment status (open → active → proven/supported/killed)
  - Querying what we know about a topic (full-text search across experiments + evidence)
  - Checking experiment stats (status distribution, kill rate, tag frequency)
  - Finding which references ground which experiments
  - Picking the next experiment to work on
argument-hint: <command> [options]
---

# Experiment Tracking CLI

Remote Turso-backed research experiment tracker. Requires `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` in `.env`.

**Base command**: `experiment <command>`

## Workflow: Pick Next Experiment

```bash
# 1. Find open work at your scale
experiment list --status open,active

# 2. Find what unblocks the most work
experiment list --blocking

# 3. Get full details of chosen experiment
experiment get <experiment-id>

# 4. Mark as active before starting
experiment update <experiment-id> --status active

# 5. After completing, update status + add evidence
experiment update <experiment-id> --status supported
experiment evidence <experiment-id> --claim "K1 PASS: metric=value" --source "results.json" --verdict pass
```

## Commands

### List experiments
```bash
experiment list
experiment list --status open,active
experiment list --tag bitnet
experiment list --blocking
```
Flags: `--status` (comma-separated), `--tag`, `--blocking` (show experiments that block others).

### Get experiment details
```bash
experiment get <experiment-id>
```
Shows: title, status, scale, priority, platform, kill criteria (with pass/fail), success criteria, evidence, tags, references, dependencies, and blocks.

### Full-text search
```bash
experiment query "routing"
experiment query "ternary adapter composition"
```
FTS5 ranked search across experiment titles, notes, and evidence claims.

### Stats dashboard
```bash
experiment stats
```
Shows: status distribution, kill rate, success rate, evidence count, top tags, scale distribution.

### List references
```bash
experiment refs
experiment refs --unused
experiment refs --tag composition
```

### Add experiment
```bash
experiment add <id> --title "My experiment" --scale micro --priority 2 --grounded-by 2505.22934
```
Creates a new experiment. `--grounded-by` links to a reference by arXiv ID.

### Update experiment
```bash
experiment update <id> --status active
experiment update <id> --status killed --notes "K1 failed: metric below threshold"
experiment update <id> --priority 1 --platform local-apple
```
Partial update. Auto-sets updated_at.

### Add evidence
```bash
experiment evidence <id> --claim "K1 PASS: rho=1.0" --source "results.json" --verdict pass
experiment evidence <id> --claim "K2 FAIL: latency 3x baseline" --source "benchmark.json" --verdict fail
```
Appends evidence to an experiment. Auto-sets date to today.

### Add reference
```bash
experiment ref-add --arxiv 2505.22934 --title "OSRM" --relevance "Data-aware orthogonality" --local-path references/osrm/ --tag composition
```

### Import from YAML (one-time migration)
```bash
experiment import --hypotheses HYPOTHESES.yml --references references/REFERENCES.yml
```
Idempotent — skips existing IDs on re-run.

## Database

Remote Turso (libSQL) at the URL in `.env`. Schema: experiments, kill_criteria, success_criteria, evidence, tags, references, plus junction tables.

## Tips

- Suppress oclif warnings: `2>&1 | grep -v "Warning"`
- The `query` command uses FTS5 — use natural language, not exact IDs
- Always `--grounded-by` when adding experiments to link supporting literature
- Check `list --blocking` to find critical-path work
- Platforms: `local`, `local-apple`, `runpod-flash`
- Status flow: `open` → `active` → `proven` | `supported` | `killed`
