---
name: experiment
description: |
  Query and manage the research experiment tracking database (Turso remote). Use when:
  - Listing, filtering, or searching experiments by status/tag/dependency
  - Getting full details of an experiment (kill criteria, evidence, references, dependencies)
  - Adding new experiments, evidence, references, kill/success criteria, dependencies, tags
  - Updating experiment status (open → active → proven/supported/killed)
  - Updating kill criteria results (pass/fail/inconclusive)
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

# 3. Get full details of chosen experiment (YAML for structured view)
experiment get <experiment-id> --yaml

# 4. Mark as active before starting
experiment update <experiment-id> --status active

# 5. After completing, update kill criteria results + status + evidence
experiment kill-update <experiment-id> --criterion <id> --result pass
experiment update <experiment-id> --status supported
experiment evidence <experiment-id> --claim "K1 PASS: metric=value" --source "results.json" --verdict pass
```

## Commands

### Get experiment details
```bash
experiment get <experiment-id>
experiment get <experiment-id> --yaml
```
Shows: title, status, scale, priority, platform, kill criteria (with pass/fail and IDs), success criteria, evidence, tags, references, dependencies, blocks, and FULL notes.

The `--yaml` flag outputs structured YAML with all fields, including completeness warnings for missing data. **Use --yaml when you need the full picture** — it shows everything, never truncates, and flags what needs filling.

Completeness warnings flag: missing kill_criteria, success_criteria, tags, references, platform, experiment_dir, and untested kill results.

### List experiments
```bash
experiment list
experiment list --status open,active
experiment list --tag bitnet
experiment list --blocking
```
Flags: `--status` (comma-separated), `--tag`, `--blocking` (show experiments that block others).

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

### Add kill criterion
```bash
experiment kill-add <experiment-id> --text "K1: metric > threshold -> KILL" --reason "Based on prior work"
```
Adds a structured kill criterion (result defaults to `untested`). The `--reason` flag defaults to "pre-registered".

### Update kill criterion result
```bash
experiment kill-update <experiment-id> --criterion <kill-criterion-id> --result pass
experiment kill-update <experiment-id> --criterion <kill-criterion-id> --result fail
```
Updates result to: `pass`, `fail`, or `inconclusive`. Get the criterion ID from `experiment get --yaml`.

### Add success criterion
```bash
experiment success-add <experiment-id> --condition "Metric passes threshold" --unlocks "Next experiment or capability" --reason "Why this matters"
```
`--max-followup` (default 1) limits how many follow-up experiments this success enables.

### Add dependency
```bash
experiment dep-add <experiment-id> --on <depends-on-id>
```
Declares that experiment depends on another. Shows in `get` output and affects `list --blocking`.

### Add tags
```bash
experiment tag <experiment-id> --add routing --add mlx --add novel
```
Tags are created if they don't exist. Multiple `--add` flags supported.

### Add evidence
```bash
experiment evidence <id> --claim "K1 PASS: rho=1.0" --source "results.json" --verdict pass
experiment evidence <id> --claim "K2 FAIL: latency 3x baseline" --source "benchmark.json" --verdict fail
```
Appends evidence to an experiment. Auto-sets date to today.

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

### Add reference
```bash
experiment ref-add --arxiv 2505.22934 --title "OSRM" --relevance "Data-aware orthogonality" --local-path references/osrm/ --tag composition
```

### Import from YAML (one-time migration)
```bash
experiment import --hypotheses HYPOTHESES.yml --references references/REFERENCES.yml
```
Idempotent — skips existing IDs on re-run.

## Workflow: After Running an Experiment

```bash
# 1. Update kill criteria results (get IDs from --yaml output)
experiment kill-update <id> --criterion 183 --result pass
experiment kill-update <id> --criterion 184 --result fail

# 2. Add evidence for each finding
experiment evidence <id> --claim "K1 PASS: val_loss=1.2 < 2.0 threshold" --source "results.json" --verdict pass
experiment evidence <id> --claim "K2 FAIL: PPL 52.3 > 3x baseline (48.0)" --source "results.json" --verdict fail

# 3. Set final status
experiment update <id> --status killed --notes "K2 failed: PPL too high"

# 4. Set experiment directory
experiment update <id> --dir "micro/models/my_experiment/"
```

## Database

Remote Turso (libSQL) at the URL in `.env`. Schema:
- `experiments` — core entity (id, title, status, scale, priority, platform, notes, dates)
- `kill_criteria` — per-experiment kill criteria with result tracking (untested/pass/fail/inconclusive)
- `success_criteria` — per-experiment success conditions with unlocks and followup limits
- `evidence` — timestamped claims with source and verdict
- `experiment_dependencies` — directed dependency graph between experiments
- `tags` + `experiment_tags` — tagging system
- `references` + `experiment_references` — literature links with arxiv IDs

## Tips

- **Always use `--yaml` for the full picture** — it shows all fields and flags what's missing
- Suppress oclif warnings: `2>&1 | grep -v "Warning"`
- The `query` command uses FTS5 — use natural language, not exact IDs
- Always `--grounded-by` when adding experiments to link supporting literature
- Check `list --blocking` to find critical-path work
- After adding an experiment, immediately add kill criteria, success criteria, and tags
- Platforms: `local`, `local-apple`, `runpod-flash`
- Status flow: `open` → `active` → `proven` | `supported` | `killed`
