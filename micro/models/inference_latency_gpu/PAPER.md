# GPU Latency Validation: Research Digest

## Hypothesis

Inference latency for composed LoRA expert models is independent of the total
expert count N on production GPU hardware (RTX 4090). Pre-merge composition has
zero overhead at any N. Dynamic top-k overhead depends only on k, not N.

**Falsifiable:** If pre-merge overhead exceeds 5% at any N in {5, 10, 20, 50},
or if dynamic top-k overhead scales with N (slope > 0.1%/expert), the hypothesis
is killed.

## What This Experiment Is

This is the GPU-scale validation of `exp_inference_latency_vs_N`, which proved
N-independence on CPU with micro-scale synthetic LoRA weights. That experiment
showed:
- Pre-merge: 0% overhead (within noise) at all N
- Dynamic: ~260% overhead but **constant** across N (implementation-bound)
- Hash ring routing: <1us at all N

This experiment validates the same property on:
- **Real hardware**: NVIDIA RTX 4090 (24GB, CUDA)
- **Real model**: Qwen2.5-7B (FP16)
- **Real adapters**: 50 trained LoRA experts from the distillation pilot
- **Production stack**: PEFT library for adapter management

## Lineage in the Arena

```
micro/models/inference_latency_vs_N (CPU, synthetic, d=128)
    |
    +-- THIS: micro/models/inference_latency_gpu (GPU, real, Qwen2.5-7B)
        |
        +-- [FUTURE] vLLM fused kernel validation (S-LoRA style)
```

## Key References

- exp_inference_latency_vs_N — proved N-independence on CPU (this project)
- macro/batched_lora_latency — showed -4% overhead at k=1 with direct copy
- S-LoRA (Sheng et al., 2024) — fused CUDA kernels for multi-LoRA serving
- vLLM multi-LoRA — production serving with fused MoE-LoRA kernels

## Empirical Results

**STATUS: AWAITING EXECUTION**

The GPU latency benchmark is queued to run after the pilot50 benchmark completes.
It will measure pre-merge and dynamic LoRA latency at N=5,10,20,50.

Check progress: `ssh runpod 'cat /workspace/gpu_latency_bench.log'`
Results file: `/workspace/llm/results/gpu_latency_benchmark.json`

### Pre-Merge Overhead vs Base (seq_len=256)

| N | Latency (ms) | Overhead | Merge Time (ms) |
|---|-------------|----------|-----------------|
| Base | pending | -- | -- |
| 5 | pending | pending | pending |
| 10 | pending | pending | pending |
| 20 | pending | pending | pending |
| 50 | pending | pending | pending |

### Dynamic Top-k Overhead (seq_len=256)

| N | k=1 Overhead | k=2 Overhead |
|---|-------------|-------------|
| 5 | pending | pending |
| 10 | pending | pending |
| 20 | pending | pending |
| 50 | pending | pending |

### Kill Criteria Assessment

| Criterion | Threshold | Actual | Verdict |
|-----------|-----------|--------|---------|
| K1: Pre-merge overhead | <= 5% at all N | pending | pending |
| K2: Dynamic scales with N | slope <= 0.1%/expert | pending | pending |
| K3: Fused kernel overhead | <= 10% vs monolithic at k=2 | not tested (no vLLM) | deferred |

## Micro-Scale Limitations

1. **PEFT, not fused kernels** -- This experiment uses HuggingFace PEFT for
   dynamic LoRA application, not vLLM's fused CUDA kernels. The absolute dynamic
   overhead will be higher than production. The N-independence property should
   still hold because it is architectural, not implementation-dependent.

2. **Single GPU** -- No multi-GPU or tensor-parallel testing. The latency
   characteristics may differ with model parallelism.

3. **Batch size 1** -- Production serving typically batches queries. LoRA overhead
   fraction decreases with batch size (LoRA FLOPs are per-token, but batching
   amortizes kernel launch overhead).

4. **Full precision adapters** -- Real adapters are FP16 LoRA on FP16 base.
   Quantized base (4-bit) with FP16 adapters would show different characteristics.

## What Would Kill This

**At this scale:**
- Pre-merge overhead > 5% at any N (unexpected -- would indicate merged weights
  cause different kernel behavior, e.g., denormalized floats)
- Dynamic overhead grows linearly with N at fixed k (would indicate memory
  pressure or cache pollution from storing many adapter tensors)

**At production scale:**
- vLLM fused kernel overhead exceeds 10% (would kill the "near-zero overhead" claim)
- Batched serving with multiple LoRA adapters causes kernel contention
- Multi-tenant serving (different adapters per request) breaks N-independence

## Files

| File | Purpose |
|------|---------|
| `scripts/gpu_latency_bench.py` | GPU latency benchmark (RunPod) |
| `results/gpu_latency_benchmark.json` | Results (pending) |
| `micro/models/inference_latency_gpu/MATH.md` | Mathematical analysis |
| `micro/models/inference_latency_gpu/PAPER.md` | This file |
