# Running the Research Loops

## Prerequisites

```bash
cd /Users/tom/Code/tomsiwik/llm
source .venv/bin/activate

# .env contains RUNPOD_API_KEY (gitignored)
# SSH config: ~/.ssh/config has Host runpod alias
# SSH key: ~/.ssh/runpod.id_ed25519 (from 1Password "SSH runpod")
```

---

## Current Phase: LoRA MoE Benchmark + Compose CLI

All 7 macro hypotheses are validated. Now building the system.

**Loop 1 (RunPod)**: 5-domain LoRA MoE benchmark — the headline result.
**Loop 2 (RunPod)**: Compose CLI end-to-end test — the usable tool.

Both use RunPod but work in separate directories and don't interfere.

### Loop 1: 5-Domain LoRA MoE Benchmark (RunPod, ~3h)

```bash
ralph run -a -p "You are running the definitive LoRA MoE benchmark on RunPod. The script is ready at macro/lora_moe_benchmark.py.

SSH is configured: 'ssh runpod' works. Use tools/runpod_exec.py for all RunPod interaction. Setup is done (deps installed, Qwen cached).

YOUR TASK:
1. Sync repo: python tools/runpod_exec.py sync
2. Run: python tools/runpod_exec.py run macro/lora_moe_benchmark.py --timeout 10800
3. If it fails, read the error, fix the script locally, sync again, retry
4. Pull results: python tools/runpod_exec.py pull macro/lora_moe_benchmark/results.json
5. Pull PAPER.md: python tools/runpod_exec.py pull macro/lora_moe_benchmark/PAPER.md
6. Update FINDINGS.md with the headline numbers
7. Update HYPOTHESES.yml — add exp_lora_moe_benchmark node with results

WHAT THIS MEASURES:
- 5 domain LoRA adapters (Python, JS, Medical, Legal, Math) on Qwen2.5-0.5B
- Learned softmax router (top-2)
- Benchmark: MoE vs joint training vs simple average vs TIES vs DARE
- Inference latency: MoE overhead vs monolithic
- 3 seeds for statistical confidence

THE KEY QUESTION: Does composed LoRA MoE match joint training quality?

All scripts use: HF_HOME=/workspace/hf_cache, Qwen/Qwen2.5-0.5B, PyTorch+CUDA, peft.

CONSTRAINTS: Never sleep > 600s. Never create/terminate pods. Never ask for confirmation. Write .ralph/current_direction.md at task start."
```

### Loop 2: Compose CLI End-to-End Test (RunPod, ~2h)

```bash
ralph run -a -p "You are testing the compose CLI end-to-end on RunPod. The scripts are ready:
- tools/compose.py — the compose CLI tool
- macro/compose_e2e_test.py — the end-to-end test script

SSH is configured: 'ssh runpod' works. Use tools/runpod_exec.py for all RunPod interaction. Setup is done.

IMPORTANT: Wait 5 minutes before starting to let Loop 1 sync first, then:

YOUR TASK:
1. Sync repo: python tools/runpod_exec.py sync
2. Run: python tools/runpod_exec.py run macro/compose_e2e_test.py --timeout 7200
3. If it fails, read the error, fix the script locally, sync again, retry
4. Pull results: python tools/runpod_exec.py pull macro/compose_e2e/results.json
5. Write macro/compose_e2e/PAPER.md with findings
6. Update VISION.md 'What Remains' section — check off compose CLI prototype

WHAT THIS TESTS:
- Train 5 LoRA adapters, save as .pt files
- Register with compose CLI (init, add, list)
- Benchmark: routing quality, merge latency, cache hit rates
- Generate text with routed experts
- Test add/remove expert workflow (plug-and-play)
- Detailed latency: merge time, forward time, generate time

THE KEY QUESTION: Is the compose workflow practical? What's the latency overhead?

CONSTRAINTS: Never sleep > 600s. Never create/terminate pods. Never ask for confirmation. Write .ralph/current_direction.md at task start.

NOTE: Loop 1 is running the MoE benchmark simultaneously. Your scripts use different output directories (macro/compose_e2e/ vs macro/lora_moe_benchmark/) so there's no conflict. But be aware the GPU is shared — if you hit OOM, reduce batch size."
```

### Running Both

```bash
# Tab 1 — LoRA MoE Benchmark
cd /Users/tom/Code/tomsiwik/llm && ralph run -a -p "<loop 1 prompt>"

# Tab 2 — Compose CLI E2E (start ~5 min after Tab 1)
cd /Users/tom/Code/tomsiwik/llm && ralph run -a -p "<loop 2 prompt>"
```

Both use RunPod but write to separate directories. GPU is shared —
if OOM occurs, the second loop should reduce batch size or wait.

---

## Previous Phase: Macro Hypothesis Validation (COMPLETED)

All 7 tasks from `macro/RUNPOD_QUEUE.md` completed successfully:

| Task | Result | Status |
|------|--------|--------|
| Gap-as-Signal bridge (d=256, 20 seeds) | r²=0.865 at N=4 | PROVEN |
| Gap-as-Signal real LoRA (d=896) | cos=0.0002, r²=0.22 | PROVEN (self-defeating) |
| SwiGLU gate pruning | +196% quality loss | KILLED |
| LoRA orthogonality scaling | cos=7e-05 across ranks | PROVEN |
| Hash routing N=20 | 5.3% displacement | PROVEN |
| Prune-compose pipeline | +0.012% gap | PROVEN |
| L2 norm stability | 0/25 failures | PROVEN |

---

## RunPod Setup (you do this manually)

### 1. Create Pod

Go to RunPod Dashboard -> Pods -> Create.

| Setting | Value |
|---------|-------|
| GPU | RTX A5000 (24 GB, $0.16/hr on-demand) |
| Image | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Volume | Attach your 50 GB network volume |
| SSH | Enable (your keys are already configured) |

### 2. SSH is configured

SSH config in `~/.ssh/config` (Host `runpod` alias).
Key from 1Password "SSH runpod" at `~/.ssh/runpod.id_ed25519`.
Direct TCP connection (not the RunPod proxy which requires PTY).

Pod IP/port may change on restart — update `~/.ssh/config` with new
values from `runpod.get_pods()` -> `runtime.ports[type=tcp]`.

### 3. Test + Setup

```bash
python tools/runpod_exec.py test    # verify connection
python tools/runpod_exec.py setup   # install deps, cache model
python tools/runpod_exec.py sync    # sync repo
```

---

## GPU Selection

Queried 2026-03-07. On-demand pricing (no spot):

| GPU | VRAM | On-demand $/hr | Available | Notes |
|-----|------|----------------|-----------|-------|
| **RTX A5000** | **24 GB** | **$0.16** | **yes** | **Cheapest 24 GB** |
| RTX A4000 | 16 GB | $0.17 | yes | 16 GB, fine for 0.5B |
| RTX 3090 | 24 GB | $0.22 | yes | Fallback |
| RTX 4090 | 24 GB | $0.34 | yes | 2x faster |

**Pick: RTX A5000 at $0.16/hr.** 24 GB fits Qwen2.5-0.5B + LoRA easily.

---

## `tools/compose.py` Reference

The compose CLI for plug-and-play LoRA composition:

```bash
# Initialize registry with base model
python tools/compose.py init --base Qwen/Qwen2.5-0.5B

# Register LoRA adapters
python tools/compose.py add adapter.pt --name python --domain code --rank 16

# List registered experts
python tools/compose.py list

# Benchmark composition
python tools/compose.py bench --prompts 20

# Generate with routing
python tools/compose.py generate "def fibonacci(n):" --max-tokens 100 --top-k 2

# Remove an expert (no retraining needed)
python tools/compose.py remove python

# Serve via HTTP
python tools/compose.py serve --port 8080
```

---

## `tools/runpod_exec.py` Reference

```bash
python tools/runpod_exec.py test                    # verify SSH
python tools/runpod_exec.py setup                   # install deps + cache model
python tools/runpod_exec.py sync                    # rsync repo to pod
python tools/runpod_exec.py run macro/script.py     # sync + run script
python tools/runpod_exec.py exec "nvidia-smi"       # arbitrary command
python tools/runpod_exec.py pull macro/results.json  # pull file back
```

---

## Cost Estimates

| Phase | GPU | Hours | Cost |
|-------|-----|-------|------|
| Macro hypothesis validation (done) | RTX A5000 | ~10h | ~$1.60 |
| LoRA MoE benchmark | RTX A5000 | ~3h | ~$0.48 |
| Compose CLI E2E test | RTX A5000 | ~2h | ~$0.32 |
| **Total estimated** | | | **~$2.40** |

---

## Troubleshooting

- **SSH fails**: Pod may have restarted with new IP. Update `~/.ssh/config`.
- **CUDA OOM**: Reduce batch size. Two loops sharing GPU may cause this.
- **Two loops on RunPod**: They use separate output directories. If GPU contention, stagger start by 5 min.
- **Pod idle cost**: $0.16/hr = $3.84/day idle. Stop pod when not in use.
