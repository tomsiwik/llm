# Coding Guidelines

Rules for writing experiment scripts in this project. Every script MUST follow these.

Incident context: on 2026-03-25 a local MLX experiment (`bitnet_evolve_multi_cycle`)
caused 1803 macOS memory-pressure kills and a crash loop because it ran all phases in
one function with zero cleanup. These rules exist to prevent that.

---

## 1. Function Scoping (MANDATORY)

Each compute-intensive phase MUST run inside its own function. Never chain model
loading, training, evaluation, and composition in a single scope. Hidden references
(optimizer state, computation graphs, closures) survive `del` and cause OOM.

```python
import gc

def phase_train(model_id, data_path):
    model = load_model(model_id)
    # ... train ...
    save_results(results)
    cleanup(model)
    return results

def phase_evaluate(model_id, adapter_path):
    model = load_model(model_id)
    # ... evaluate ...
    cleanup(model)
    return metrics

def phase_compose(adapter_paths):
    model = load_model(model_id)
    # ... compose and measure ...
    cleanup(model)
    return composition_results

# main() is a thin orchestrator — no large objects live here
def main():
    train_results = phase_train(MODEL_ID, DATA_PATH)
    eval_results = phase_evaluate(MODEL_ID, ADAPTER_PATH)
    compose_results = phase_compose(ADAPTER_PATHS)
    write_paper(train_results, eval_results, compose_results)
```

**Why:** When a function returns, all its local references die. The garbage collector
can then reclaim those buffers. A monolithic `main()` keeps everything alive until the
script exits.

---

## 2. Cleanup Between Phases (MANDATORY)

After every function that allocates model weights, optimizer state, or large tensors,
you MUST release them. The pattern differs by framework but the principle is identical:
delete references first, then garbage-collect, then release the framework cache.

### MLX (Apple Silicon)

Reference: https://ml-explore.github.io/mlx/build/html/python/memory_management.html

```python
import gc
import mlx.core as mx

def cleanup(*objects):
    """Release MLX memory between phases."""
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()  # returns Metal buffers to OS

# Monitor memory between phases:
print(f"Active: {mx.get_active_memory() / 1e9:.2f} GB")
print(f"Cache:  {mx.get_cache_memory() / 1e9:.2f} GB")
print(f"Peak:   {mx.get_peak_memory() / 1e9:.2f} GB")
mx.reset_peak_memory()
```

**How MLX memory works:** `del array` moves memory from "active" to "cache" (still
held by the Metal allocator). `mx.clear_cache()` returns cached buffers to the OS.
Without `clear_cache()`, freed memory stays in the MLX buffer pool indefinitely.

**Memory limits** (set at script startup for safety):
```python
device = mx.device_info()
total = device["memory_size"]
# Leave 8 GB for system + other processes
mx.set_memory_limit(total - 8 * 1024**3)
# Limit cache to 2 GB (prevents unbounded cache growth)
mx.set_cache_limit(2 * 1024**3)
```

**Lazy evaluation:** MLX uses lazy evaluation. Without `mx.eval()` the computation
graph grows unboundedly and consumes memory. Always call `mx.eval()` at the end of
each training step:
```python
for step in range(n_iters):
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)  # MANDATORY — forces execution
```

Reference: https://github.com/ml-explore/mlx-examples/blob/main/lora/lora.py

### PyTorch (CUDA)

Reference: https://docs.pytorch.org/docs/stable/notes/cuda.html

```python
import gc
import torch

def cleanup(*objects):
    """Release PyTorch CUDA memory between phases."""
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Monitor memory between phases:
print(torch.cuda.memory_summary())
```

**Environment** (set before importing torch):
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

**How PyTorch memory works:** PyTorch's caching allocator holds freed CUDA memory for
reuse. `nvidia-smi` shows reserved memory, not actual usage. `empty_cache()` returns
unoccupied cached memory to CUDA.

**Common OOM trap — accumulating history in training loops:**
```python
# WRONG — accumulates autograd graph across iterations
total_loss += loss

# RIGHT — detach the scalar
total_loss += loss.item()
```

Reference: https://docs.pytorch.org/docs/stable/notes/faq.html

---

## 3. Adapter and Tensor Storage

Never accumulate adapter parameters in memory across cycles. Save to disk, load on
demand.

```python
# WRONG — holds all adapters in memory for the entire run
all_adapter_params = {}
for cycle in range(N_CYCLES):
    for domain in DOMAINS:
        params = train_and_get_params(model, domain, cycle)
        all_adapter_params[domain][cycle] = params  # memory grows every iteration

# RIGHT — save to disk, load only when needed
for cycle in range(N_CYCLES):
    for domain in DOMAINS:
        params = train_and_get_params(model, domain, cycle)
        save_adapter(params, f"adapters/{domain}/cycle_{cycle}.npz")
        del params

# Later, for comparison:
def compare_adapters(path_a, path_b):
    a = load_adapter(path_a)
    b = load_adapter(path_b)
    cos = cosine_similarity(a, b)
    del a, b
    return cos
```

**Why:** The 2026-03-25 crash stored 3 cycles x 2 domains of full adapter snapshots
(each ~50 MB at rank-16 on a 2B model) in a dict that persisted for 40+ minutes.

---

## 4. Training

### MLX
```python
# Disable Python GC during tight training loops (re-enable after)
gc.disable()
for step in range(n_iters):
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
gc.enable()
gc.collect()
```

### PyTorch
```python
# Use inference_mode for eval (stricter than no_grad, less memory)
with torch.inference_mode():
    outputs = model(inputs)

# Gradient checkpointing (trades ~20% compute for major memory savings)
model.gradient_checkpointing_enable()

# bf16 training (Ampere+ GPUs) — prefer over fp16, no loss scaling needed
from transformers import TrainingArguments
args = TrainingArguments(
    bf16=True,
    gradient_checkpointing=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
)
```

Reference: https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one

---

## 5. Evaluation

Clean up after EVERY forward pass during evaluation to prevent memory accumulation:

### MLX
```python
for item in eval_data:
    logits = model(mx.array(tokens)[None, :])
    mx.eval(logits)
    score = compute_metric(logits)
    del logits  # free before next iteration
```

### PyTorch
```python
with torch.inference_mode():
    for item in eval_data:
        outputs = model(inputs)
        score = compute_metric(outputs)
        del outputs, inputs
    # Periodic cleanup every N items
    if i % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()
```

---

## 6. Script Structure Template

Every experiment script should follow this structure:

```python
#!/usr/bin/env python3
"""Experiment: <name>. Kill criteria: K1: ..., K2: ..."""

import gc
import json
import time
from pathlib import Path

# Framework-specific imports and setup
# (MLX)
import mlx.core as mx
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

# (PyTorch)
# import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# import torch

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

def log_memory(label=""):
    """Print current memory usage."""
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")

def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

# --- Phase functions (each self-contained) ---

def phase_load_data():
    # ... load and return data ...
    return train_data, val_data

def phase_train(model_id, train_data, adapter_save_path):
    model, tokenizer = load(model_id)
    # ... setup and train ...
    save_adapter(model, adapter_save_path)
    results = {"loss": final_loss, "time": elapsed}
    log_memory("post-train")
    cleanup(model, tokenizer)
    return results

def phase_evaluate(model_id, adapter_path, val_data):
    model, tokenizer = load(model_id)
    load_adapter_weights(model, adapter_path)
    # ... evaluate ...
    log_memory("post-eval")
    cleanup(model, tokenizer)
    return metrics

def phase_composition_safety(adapter_paths):
    # Load adapters from disk one pair at a time
    results = []
    for i, path_a in enumerate(adapter_paths):
        for path_b in adapter_paths[i+1:]:
            cos = compare_adapters(path_a, path_b)
            results.append(cos)
    return results

def main():
    t0 = time.time()
    log_memory("start")

    train_data, val_data = phase_load_data()

    train_results = phase_train(MODEL_ID, train_data, ADAPTER_PATH)
    log_memory("after-train-phase")

    eval_results = phase_evaluate(MODEL_ID, ADAPTER_PATH, val_data)
    log_memory("after-eval-phase")

    safety = phase_composition_safety(ADAPTER_PATHS)
    log_memory("after-compose-phase")

    results = {**train_results, **eval_results, "safety": safety,
               "total_time_s": round(time.time() - t0, 1)}
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
```

---

## 7. Running Experiments via Pueue (MANDATORY for local)

All local experiments MUST be submitted through `experiment run`, never via bare
`uv run python`. Pueue manages the process lifecycle: serial execution, process
group isolation, and guaranteed resource recycling on exit/crash/timeout.

```bash
# Run by experiment ID (looks up experiment_dir from DB)
experiment run <id>

# Run by script path
experiment run micro/models/<name>/run_experiment.py

# Submit and return immediately (ralph/async use)
experiment run --no-wait <id>

# Queue management
experiment run --status                  # check queue
experiment run --clean                   # remove finished tasks
experiment run --kill <id>               # kill a running experiment
```

**Why:** On macOS + Metal, memory is only fully recycled when the process exits.
Pueue kills the entire process group (parent + all children) on completion, crash,
or cancellation. This guarantees Metal buffers are returned to the OS between
experiments. Without this, orphan python processes accumulate and consume 10-50 GB
of compressed memory + swap over days (see: 2026-04-04 incident, 50 GB memory
accumulation from orphan experiment processes).

**Direct pueue commands** (for debugging):
```bash
pueue status --group experiments     # queue state
pueue log <task-id>                  # full output of a task
pueue follow <task-id>              # stream live output
pueue kill <task-id>                # kill a running experiment
pueue clean --group experiments     # remove finished tasks from list
```

**Configuration:** Pueue daemon starts on login via launchd. The `experiments`
group is set to `parallel 1` (serial execution). Experiments get full system
resources — no memory caps.

---

## 8. RunPod Compatibility

- Always support `SMOKE_TEST=1` environment variable for quick validation (<60s)
- Handle `MAX_RUNTIME` timeout via `signal.SIGALRM`
- Use `--break-system-packages` for pip on newer RunPod images

---

## 9. What NOT To Do

- **Don't chain phases in one function scope** — the #1 cause of OOM
- **Don't accumulate tensors/params in dicts** across training cycles — save to disk
- **Don't skip `mx.eval()`** in MLX training loops — the computation graph grows unboundedly
- **Don't skip cleanup between phases** — memory doesn't free itself
- **Don't use `loss` directly for logging** — use `loss.item()` to avoid graph accumulation
- **Don't hardcode model paths** — use env vars with sensible defaults
- **Don't install packages that conflict with the base image** (vLLM version wars)
- **Don't re-evaluate base model benchmarks** — use published numbers
- **Don't run experiments with bare `uv run python`** — use `bin/run-experiment` for process lifecycle management
