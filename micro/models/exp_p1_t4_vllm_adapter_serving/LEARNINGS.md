# LEARNINGS.md — T4.3: MLX-Native Adapter Serving with Runtime Hot-Swap

## Core Finding
MLX-native adapter hot-swap (`load_weights()` + `mx.eval()`) is a viable serving primitive:
p99 swap = 4.77ms, throughput = 90.8% of base, routing = <1μs. Full pipeline verified:
TF-IDF(0.3ms) → swap(~5ms) → generate(37+ tok/s) = <10ms overhead to first token.

## Why It Works
Adapters are ~4MB each (rank=6 q_proj only); swap cost is I/O-bound not compute-bound.
Exclusive routing (T3.4) guarantees zero interference — only one adapter is active at inference.
`load_weights(strict=False)` patches only LoRA parameters, leaving 4B base frozen.

## Key Lesson: Use Bandwidth Model, Not FLOPs Model
FLOPs analysis predicted 99.5% throughput; measured 90.8% (gap: LoRALinear reads x twice —
once for base path, once for LoRA path, doubling memory traffic over 35 q_proj layers).
On memory-bound Apple Silicon, throughput predictions must use bytes-moved-per-token, not FLOP counts.

## Implications for Next Experiment
T4.6 (end-to-end latency under load): use bandwidth model for predictions; warm-cache adapters
before measuring production SLAs (Phase 4 showed 3.9–10.4ms on cold load vs 4.77ms p99 steady-state).
N=25 serving (T4.2 LSH routing) can extend this pipeline with sub-linear domain lookup.
