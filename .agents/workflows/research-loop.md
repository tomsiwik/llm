# Research Loop Workflow

## Trigger
`/startresearch` or manually invoke this workflow

## Steps

### Step 0: Read Context (MANDATORY — do this first)
Read these files to understand current state and available tooling:
1. `.ralph/current_direction.md` — what's in progress
2. `.ralph/memories.md` — accumulated knowledge (includes P1 strategy, MLX guide pointers)
3. `CODING_GUIDELINES.md` — mandatory MLX memory/cleanup patterns

If the experiment is tagged `p1` or involves Gemma 4:
4. `docs/MLX_GEMMA4_GUIDE.md` — verified commands, model IDs, GrassmannianLoRA code, dimensions
5. `../ARCHITECTURE_P1.md` — full mathematical reference for Gemma 4 building blocks

If writing MLX code, also read the relevant sections of `CODING_GUIDELINES.md`
(function scoping, cleanup between phases, `mx.eval()` in training loops).

### Step 1: Check State
Read `docs/tasks/progress.txt`.
Run `experiment list -s open,active` to see available work.
If no open experiments: append `RESEARCH_BACKLOG_DRAINED` to progress.txt and stop.

### Step 2: Claim Experiment
Run `experiment claim researcher` to get next highest-priority experiment.
Note the experiment ID, directory, kill criteria, and notes.

Read the experiment notes carefully — they contain:
- Impossibility structure (why the math guarantees the result)
- Dependencies (what must be proven first)
- Grounding references (arxiv papers)

### Step 3: Design (if needed)
Check if `micro/models/<name>/run_experiment.py` exists.
If NO:
- Write MATH.md with theorem/proof/prediction
- Write run_experiment.py following `CODING_GUIDELINES.md` patterns
- For MLX code: follow the template in `CODING_GUIDELINES.md` section 6
- For Gemma 4 models: use model IDs from `docs/MLX_GEMMA4_GUIDE.md` section 1
- For LoRA training: use commands from `docs/MLX_GEMMA4_GUIDE.md` section 3
- For Grassmannian adapters: use the `GrassmannianLoRALinear` from `docs/MLX_GEMMA4_GUIDE.md` section 4
- Python: always use `uv run python` (project uses uv, not system python)
- Update `.ralph/current_direction.md`

### Step 4: Run
Execute: `experiment run <id>`
Wait for completion. Read results.json.

### Step 5: Document
Write PAPER.md with prediction-vs-measurement table.
Write brief REVIEW-adversarial.md (max 3 issues).

### Step 6: Complete
```bash
experiment complete <id> --status supported \
  --dir micro/models/<name>/ \
  --k <kill_id>:pass \
  --evidence "summary of results"

experiment finding-add \
  --title "what was tested" \
  --status supported \
  --result "key numbers" \
  --caveat "limitations" \
  --experiment <id> \
  --scale micro
```

### Step 7: Log Progress
Append to `docs/tasks/progress.txt`:
```
## <date>: <experiment_name> — <STATUS>
<1-line summary of result>
```

### Step 8: Loop
Go to Step 1.
