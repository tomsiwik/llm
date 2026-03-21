# GPU Coding Guidelines

Rules for writing GPU scripts in this project. Every experiment script MUST follow these.

## Memory Management

### Function Scoping (MANDATORY)
Each GPU-intensive phase MUST run inside its own function. Never load multiple models
in the same function scope — hidden references (trainer state, optimizer, closures)
survive `del` and cause OOM.

```python
# CORRECT
def train():
    model = load_model()
    trainer = SFTTrainer(model, ...)
    trainer.train()
    model.save_pretrained(path)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return results

def evaluate():
    model = load_model()  # clean GPU — previous function's refs are dead
    ...

train()
evaluate()
```

### Cleanup Pattern
After each function that uses GPU:
1. `del model, trainer` (and any other large objects)
2. `gc.collect()`
3. `torch.cuda.empty_cache()`
4. Function return ensures all local refs die

### Environment
Always set before importing torch:
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

## Training

- Disable GC during training loops: `gc.disable()` before, `gc.enable()` after (saves ~500ms/step)
- Use explicit dtype detection, not autocast:
  ```python
  use_bf16 = torch.cuda.is_bf16_supported()
  ```
- `dataloader_pin_memory=True` + `dataloader_num_workers=2` for faster data loading
- Save intermediate checkpoints (`save_steps=100, save_total_limit=2`)

## Evaluation

- `del outputs, inputs; torch.cuda.empty_cache()` after EVERY generation call
- `gc.collect()` every 50 examples
- Use published benchmark numbers for base models — don't re-evaluate

## Inference

- Use **vLLM offline batch inference** for evaluation, not sequential `model.generate()`:
  ```python
  from vllm import LLM, SamplingParams
  from vllm.lora.request import LoRARequest

  llm = LLM(model="Qwen/Qwen2.5-7B", enable_lora=True, max_lora_rank=16)
  outputs = llm.generate(all_prompts, SamplingParams(max_tokens=2048, temperature=0),
                         lora_request=LoRARequest("reasoning", 1, adapter_path))
  ```
- vLLM handles memory via PagedAttention — no manual cleanup needed
- Batched inference gets ~100% GPU utilization vs ~20% for sequential HF generate()
- For training, HF/TRL is still appropriate

## RunPod Compatibility

- Always support `SMOKE_TEST=1` for quick validation (<60s)
- Handle `MAX_RUNTIME` timeout via `signal.SIGALRM`
- Use `--break-system-packages` for pip on newer RunPod images

## Adapter Discovery

When loading pilot50 adapters, always fall through to filesystem scan if
benchmark JSON is empty:
```python
if ranked:  # Only use benchmark if it has entries
    return [name for name, _ in ranked]
# Fall through to directory scan
return sorted(d.name for d in ADAPTER_DIR.iterdir()
              if d.is_dir() and (d / "adapter_config.json").exists())
```

## What NOT To Do

- Don't install packages that conflict with the base image (vLLM version wars)
- Don't chain GPU phases in one function scope
- Don't re-evaluate base model benchmarks (use published numbers)
- Don't use subprocess isolation when function scoping suffices
- Don't hardcode model paths — use env vars with sensible defaults
