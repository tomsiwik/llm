---
name: experiment
description: |
  Query and manage the research experiment tracking database. Use when:
  - Listing, filtering, or searching experiments by status/tag/dependency
  - Getting full details of an experiment (kill criteria, evidence, references, dependencies)
  - Adding new experiments, evidence, or references
  - Querying what we know about a topic (full-text search across experiments + evidence)
  - Checking experiment stats (status distribution, kill rate, tag frequency)
  - Finding which references ground which experiments
  - Checking for unused references or blocking experiments
argument-hint: <command> [options]
---

# Experiment Tracking CLI

A SQLite-backed research experiment tracker. All commands run via bun from the project root.

**Base command**: `cd packages/cli && bun run src/index.ts <command>`

## Commands

### List experiments
```bash
cd packages/cli && bun run src/index.ts list
cd packages/cli && bun run src/index.ts list --status open,active
cd packages/cli && bun run src/index.ts list --tag bitnet
cd packages/cli && bun run src/index.ts list --blocking
```
Flags: `--status` (comma-separated), `--tag`, `--blocking` (show experiments that block others).

### Get experiment details
```bash
cd packages/cli && bun run src/index.ts get <experiment-id>
```
Shows: title, status, scale, priority, platform, kill criteria (with pass/fail), success criteria, evidence (with dates and verdicts), tags, grounding references (with local code paths), dependencies, and blocks.

### Full-text search
```bash
cd packages/cli && bun run src/index.ts query "routing"
cd packages/cli && bun run src/index.ts query "ternary adapter composition"
```
FTS5 ranked search across experiment titles, notes, and evidence claims. Use this to answer questions like "what do we know about X?".

### Stats dashboard
```bash
cd packages/cli && bun run src/index.ts stats
```
Shows: status distribution with visual bars, kill rate, success rate, evidence count, top tags, scale distribution.

### List references
```bash
cd packages/cli && bun run src/index.ts refs
cd packages/cli && bun run src/index.ts refs --unused
cd packages/cli && bun run src/index.ts refs --tag composition
```
Shows literature references. `--unused` finds refs not linked to any experiment.

### Add experiment
```bash
cd packages/cli && bun run src/index.ts add <id> --title "My experiment" --scale micro --priority 2 --grounded-by 2505.22934
```
Creates a new experiment. `--grounded-by` links to a reference by arXiv ID.

### Update experiment
```bash
cd packages/cli && bun run src/index.ts update <id> --status killed
cd packages/cli && bun run src/index.ts update <id> --priority 1 --platform local-apple
```
Partial update. Auto-sets updated_at.

### Add evidence
```bash
cd packages/cli && bun run src/index.ts evidence <id> --claim "K1 PASS: rho=1.0" --source "results.json" --verdict pass
```
Appends evidence to an experiment. Auto-sets date to today.

### Add reference
```bash
cd packages/cli && bun run src/index.ts ref-add --arxiv 2505.22934 --title "OSRM" --relevance "Data-aware orthogonality" --local-path references/osrm/ --tag composition
```
Adds a literature reference. URL defaults to HuggingFace paper page if arxiv provided.

### Import from YAML (one-time migration)
```bash
cd packages/cli && bun run src/index.ts import --hypotheses ../../HYPOTHESES.yml --references ../../references/REFERENCES.yml
```
Idempotent — skips existing IDs on re-run.

## Database

SQLite at `packages/db/data/experiments.db`. If missing, run import to recreate from YAML source files.

Schema: experiments, kill_criteria, success_criteria, evidence, tags, references, plus junction tables for M2M relationships.

## Tips

- Suppress oclif warnings in output by piping through `2>&1 | grep -v "Warning"`
- The `query` command uses FTS5 for ranked results — use natural language terms, not exact IDs
- When creating new experiments, always provide `--grounded-by` to link to supporting literature
- Check `list --blocking` to find which completed experiments unblock pending work
- Platforms: `local`, `local-apple`, `runpod-flash`
- Statuses: `open` → `active` → `proven` | `supported` | `killed`
