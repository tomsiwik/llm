# Current Direction: Wave 3 Micro COMPLETE

## Status: ALL MICRO NODES DRAINED

All 9 Wave 3 micro experiments are resolved. No open or active micro-scale nodes remain in HYPOTHESES.yml.

## Wave 3 Results Summary (9 experiments, 5 supported, 4 killed)

### Track 1 — Foundation Fixes
- exp_bitnet_effective_delta_cosine: **KILLED** — raw param cosine better than effective-delta at d=2560
- exp_bitnet_kr_test_evaluation: **SUPPORTED** — KR-Test correlates with task accuracy (rho=1.0)
- exp_bitnet_lori_sparse_b: **KILLED** — ternary base already provides near-zero interference

### Track 2 — Base-Free Scaffold
- exp_bitnet_scaffold_fresh_adapters: **KILLED** — scaffold adapter PPL 36-642x worse
- exp_bitnet_galore_scaffold: **SUPPORTED** — GaLore scaffold viable but quantization gap is bottleneck
- exp_bitnet_meta_scaffold: **KILLED** — FOMAML cannot beat GaLore, destabilizes scaffold

### Track 3 — Production Serving
- exp_bitnet_llamacpp_serving: **SUPPORTED** — llama.cpp serves BitNet + 5 LoRA adapters
- exp_bitnet_per_token_routing: **SUPPORTED** — top-2 routing beats uniform 1/N by 13.9%

### Track 4 — Evolve Redesign
- exp_bitnet_retrain_evolve: **SUPPORTED** — retrain-from-scratch works (4.4x PPL improvement)

## What's Next
All remaining open/active experiments are scale: macro (GPU required). Micro backlog is exhausted.
Key proven micro findings ready for macro validation:
- GaLore scaffold (quantization gap at scale?)
- KR-Test metric (need n~540 for statistical power)
- llama.cpp serving path (production latency at scale)
- Per-token routing (scale beyond 15 domains)
- Retrain-evolve (convergence over multiple cycles)
