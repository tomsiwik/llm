# Current Direction: exp_bitnet_adapter_inference_speed

## Status: RUNNING (micro experiment cycle)

## Experiment
Measure single vs composed adapter inference latency on Apple Silicon using MLX.
Pure MLX implementation (no mlx-lm dependency) with synthetic LoRA weights on a
micro transformer, mirroring the methodology of exp_inference_latency_vs_N but
targeting Metal GPU via MLX.

## Kill Criteria
- K1: Single-adapter overhead > 15% vs base -> KILL
- K2: N-adapter overhead grows faster than O(N) (superlinear) -> KILL

## Platform
- Apple M5 Pro, 48GB unified memory
- MLX 0.31.1 with Metal GPU
- Micro model: d=128, 4 layers, 4 heads, rank=8

## Context
- Prior CPU (llama.cpp): 9.5% + 7.5%*N overhead
- Prior CPU (PyTorch): Pre-merge 0% overhead, dynamic ~260% (impl-bound)
- Prior GPU (RTX 4090): Pre-merge max +3.3% at N=50
- This fills the Apple Silicon Metal GPU gap in the serving matrix
