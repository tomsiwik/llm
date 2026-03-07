# Living Composable Model — Running Guide

## Prerequisites

```bash
cd /Users/tom/Code/tomsiwik/llm
source .venv/bin/activate

# Local (micro experiments): MLX
pip install -e .

# GPU serving: vLLM + PyTorch
pip install -e '.[serve]'
```

---

## Phase 1: Distill Experts

### Generate Training Data (Groq/Cerebras)

```bash
# Generate domain taxonomy
python tools/distill.py taxonomy --count 50 --output domains.json

# Generate training data via Groq batch API (70B teacher)
python tools/distill.py generate \
  --domains domains.json \
  --teacher llama-3.3-70b-versatile \
  --examples-per-domain 1000 \
  --output data/distillation/

# Or via Cerebras
python tools/distill.py generate \
  --domains domains.json \
  --teacher llama3.1-70b \
  --provider cerebras \
  --examples-per-domain 1000 \
  --output data/distillation/
```

### Train LoRA Experts (RunPod)

```bash
# Train all experts from generated data
python tools/distill.py train \
  --data data/distillation/ \
  --base Qwen/Qwen2.5-7B \
  --rank 16 \
  --steps 300 \
  --output adapters/

# Or on RunPod:
python tools/runpod_exec.py run tools/distill.py train ...
```

---

## Phase 2: Compose

```bash
# Initialize registry
compose init --base Qwen/Qwen2.5-7B

# Register all trained adapters
for d in adapters/*/; do
  name=$(basename "$d")
  compose add "$d" --name "$name" --rank 16
done

# Check orthogonality (should be ~0.0002)
python -m tools.orthogonality adapters/python/ adapters/javascript/

# Show routing for a prompt
compose route "def fibonacci(n):"

# Launch vLLM multi-LoRA server
compose serve --port 8080

# Generate with auto-routing
compose generate "def fibonacci(n):" --max-tokens 100

# Remove expert (instant, no retraining)
compose remove medical
```

### API Usage

```bash
curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "python", "prompt": "def fib(n):", "max_tokens": 100}'
```

---

## Phase 3: Evolve

```bash
# Evaluate expert quality on domain-specific test sets
python tools/evolve.py evaluate --experts adapters/ --eval-data eval/

# Generate corrections using teacher model
python tools/evolve.py correct \
  --expert adapters/python/ \
  --wrong-outputs logs/python_errors.jsonl \
  --teacher llama-3.3-70b-versatile

# Clone and fix expert
python tools/evolve.py clone \
  --expert adapters/python/ \
  --corrections corrections/python.jsonl \
  --output adapters/python-v2/ \
  --steps 100

# Register clone for tournament
compose add adapters/python-v2/ --name python-v2 --rank 16

# Run tournament (shadow scoring)
python tools/evolve.py tournament \
  --expert-a python \
  --expert-b python-v2 \
  --queries 5000

# Prune loser
compose remove python  # or python-v2, based on tournament result
```

---

## RunPod Setup

```bash
python tools/runpod_exec.py test     # verify connection
python tools/runpod_exec.py setup    # install deps, cache model
python tools/runpod_exec.py sync     # sync repo
python tools/runpod_exec.py run macro/script.py  # sync + run
python tools/runpod_exec.py pull macro/results.json  # pull back
```

| GPU | VRAM | On-demand $/hr |
|-----|------|----------------|
| **RTX A5000** | **24 GB** | **$0.16** |
| RTX 3090 | 24 GB | $0.22 |
| RTX 4090 | 24 GB | $0.34 |

---

## Proven Research (Foundation)

| Finding | Result | Status |
|---------|--------|--------|
| LoRA orthogonality | cos=0.0002 at d=896 | PROVEN |
| MoE beats joint training | -0.70% vs joint | PROVEN |
| Hash routing N=20 | 5.3% displacement | PROVEN |
| Prune-then-compose | +0.012% gap | PROVEN |
| L2 norm stability | 0/25 failures | PROVEN |
| Batched LoRA k=1 | -4% overhead | PROVEN |

See `FINDINGS.md` for details, `ARCHIVE.yml` for all 84 experiments.
