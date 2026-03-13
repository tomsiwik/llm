# Living Composable Model — Execution Plan

## Repo Structure

```
llm/
  micro/          # Research proofs (MLX, char-level, 92 experiments)
  macro/          # Macro validations (RunPod results, PAPER.md files)
  composer/       # Product code (compose CLI, distill, evolve)
  adapters/       # Trained PEFT adapters (the actual experts)
  eval/           # Benchmark configs and results
  references/     # Literature
```

## Prerequisites

```bash
cd /Users/tom/Code/tomsiwik/llm
source .venv/bin/activate
pip install -e .                    # core (pyyaml, python-dotenv)
pip install -e '.[train]'          # Unsloth + TRL + PEFT (RunPod)
pip install -e '.[serve]'          # vLLM (RunPod)
```

## RunPod Setup

```bash
python composer/runpod_exec.py test     # verify SSH
python composer/runpod_exec.py setup    # install deps, cache Qwen2.5-7B
python composer/runpod_exec.py sync     # sync repo
```

---

## Execution Steps (Sequential)

Each step produces a concrete artifact. Ralph loop runs steps 2-6 on RunPod.
Steps are designed so ralph can execute each one as a self-contained script.

### Step 0: Restructure repo (local, no GPU)

Move `tools/` to `composer/`. Fix pyproject.toml dependencies.
Create `composer/distill.py` and `composer/evolve.py` scaffolds.

**Artifact:** `composer/` directory with working `compose` CLI entry point.
**Verify:** `compose --help` works.

### Step 1: Build distillation data generator (local, no GPU)

Build `composer/distill.py generate` — calls Groq batch API to produce
training JSONL for N domains. Uses chat completion format compatible with
TRL SFTTrainer.

**Input:** domain list (hardcoded 5 domains for first run)
**Output:** `data/distillation/{domain}/train.jsonl` (1000 examples each)
**Format:** `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
**Verify:** JSONL files exist, parseable, ~500 tokens/example avg.

Domains for first proof:
- python (code generation)
- sql (database queries)
- bash (shell scripting)
- medical (clinical Q&A)
- math (reasoning)

### Step 2: Train 5 LoRA adapters (RunPod, GPU required)

Script: `composer/distill.py train`
Uses Unsloth + TRL SFTTrainer. Trains QLoRA rank-16 on Qwen2.5-7B.

```bash
python composer/runpod_exec.py run composer/distill.py train \
  --data data/distillation/ \
  --base Qwen/Qwen2.5-7B \
  --output adapters/ \
  --rank 16 --steps 300 --lr 2e-4
```

**Input:** `data/distillation/{domain}/train.jsonl`
**Output:** `adapters/{domain}/adapter_config.json` + `adapter_model.safetensors`
**Format:** Standard HuggingFace PEFT (compatible with vLLM, transformers, Unsloth)
**Verify:** Each adapter dir has adapter_config.json + adapter_model.safetensors.
**Time:** ~15 min/adapter = ~75 min total. Cost: ~$0.20.

### Step 3: Verify orthogonality (RunPod, GPU required)

```bash
python composer/runpod_exec.py run composer/orthogonality.py \
  adapters/python/ adapters/sql/ adapters/bash/ adapters/medical/ adapters/math/
```

**Expected:** pairwise cosine ~0.0001-0.001 (same as 0.5B finding, should be even lower at d=4096)
**Kill criterion:** any pair > 0.01 → investigate.
**Artifact:** orthogonality_results.json

### Step 4: Compose and serve (RunPod, GPU required)

```bash
# Register all adapters
compose init --base Qwen/Qwen2.5-7B
for d in adapters/*/; do
  name=$(basename "$d")
  compose add "$d" --name "$name" --rank 16
done

# Test routing
compose route "Write a Python function to sort a list"
compose route "SELECT * FROM users WHERE age > 30"
compose route "What are the symptoms of diabetes?"

# Serve
compose serve --port 8080
```

**Verify:** routing sends python queries to python expert, SQL to SQL, etc.
**Verify:** vLLM starts and accepts requests with `--enable-lora`.
**Artifact:** routing_test.json (prompt → expert mapping for 30 test prompts)

### Step 5: Benchmark vs base (RunPod, GPU required)

Script: `composer/bench.py`

For each domain, compare:
- Base model (no adapter) on domain-specific eval set
- Expert adapter on its own domain eval set
- Composed model (hash ring routing) on mixed eval set

```bash
python composer/runpod_exec.py run composer/bench.py \
  --base Qwen/Qwen2.5-7B \
  --adapters adapters/ \
  --eval-data eval/ \
  --output results/e2e_benchmark.json
```

**Metrics:** per-domain perplexity, routing accuracy (did it pick the right expert?)
**Kill criterion:** composed model worse than base on >20% of domains.
**Artifact:** results/e2e_benchmark.json + results/E2E_PAPER.md

### Step 6: Clone-and-compete proof (RunPod, GPU required)

Script: `composer/evolve.py`

1. Take the python adapter
2. Inject 50 deliberately wrong training examples (bad code patterns)
3. Clone the original, fine-tune clone with corrections (50 steps)
4. Register both on hash ring
5. Shadow score: run 1000 eval prompts, compare perplexity of original vs clone
6. Verify clone wins (lower perplexity on corrected examples)
7. Verify clone doesn't regress on non-corrected examples

```bash
python composer/runpod_exec.py run composer/evolve.py tournament \
  --expert adapters/python/ \
  --corrections data/corrections/python_fixes.jsonl \
  --eval-queries 1000 \
  --output results/tournament.json
```

**Kill criterion:** clone doesn't win >70% of corrected queries.
**Kill criterion:** clone regresses >2% on uncorrected queries.
**Artifact:** results/tournament.json + results/TOURNAMENT_PAPER.md

---

## Ralph Loop Execution

The ralph loop runs steps 2-6 sequentially on RunPod. Each step is a
self-contained Python script that reads inputs, produces outputs, and
writes a PAPER.md with results and kill criteria assessment.

### Ralph prompt for the full pipeline:

```
Run the end-to-end validation pipeline for the Living Composable Model.
Execute these scripts in sequence on RunPod. Each step depends on the
previous step's output. Stop if any kill criterion triggers.

Step 2: python composer/distill.py train --data data/distillation/ --base Qwen/Qwen2.5-7B --output adapters/
  → adapters/{domain}/adapter_config.json must exist for all 5 domains

Step 3: python composer/orthogonality.py adapters/python/ adapters/sql/ adapters/bash/ adapters/medical/ adapters/math/
  → all pairwise cosine < 0.01

Step 4: Test compose CLI routing on 30 prompts, verify correct expert selection

Step 5: python composer/bench.py --base Qwen/Qwen2.5-7B --adapters adapters/ --output results/e2e_benchmark.json
  → composed model beats base on >80% of domains

Step 6: python composer/evolve.py tournament --expert adapters/python/ --output results/tournament.json
  → clone wins >70% of corrected queries, regresses <2% on uncorrected

Pull all results back: adapters/, results/, *.json, *PAPER.md
Write results/PIPELINE_PAPER.md summarizing all steps.
```

---

## What Must Be Built Before RunPod (Step 0 + 1)

| File | Purpose | Lines (est) |
|------|---------|-------------|
| `composer/__init__.py` | Package init | 1 |
| `composer/compose.py` | Moved from tools/ (existing) | existing |
| `composer/orthogonality.py` | Moved from tools/ (existing) | existing |
| `composer/runpod_exec.py` | Moved from tools/ (existing) | existing |
| `composer/distill.py` | Data gen (Groq API) + train (Unsloth/TRL) | ~300 |
| `composer/bench.py` | Benchmark composed vs base | ~200 |
| `composer/evolve.py` | Clone, fine-tune, shadow score, tournament | ~250 |
| `pyproject.toml` | Fix deps: core, train, serve, micro | edit |
| `data/distillation/` | Generated training JSONL | generated |
| `eval/` | Domain-specific eval sets | generated |

Step 0 + 1 are local work (no GPU). Steps 2-6 are RunPod work (ralph loop).

---

## Budget

| Step | Platform | Cost |
|------|----------|------|
| Step 1: Data gen (5 domains) | Groq batch API | ~$1 |
| Step 2: Training (5 adapters) | RunPod A5000 | ~$0.20 |
| Step 3-6: Eval + tournament | RunPod A5000 | ~$2 |
| **Total first proof** | | **~$3** |

After proof works, scale to 50 domains (~$15), then 500 (~$130).
