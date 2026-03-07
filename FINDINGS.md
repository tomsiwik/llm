# Research Findings

## Conclusive Results (Macro Scale, Qwen2.5-0.5B)

| Finding | Result | Evidence |
|---------|--------|----------|
| LoRA orthogonality is structural | cos=0.0002 at d=896 (50x better than theory) | macro/ortho_scaling/ |
| MoE beats joint training | -0.70% vs joint (4 domains, 3 seeds) | macro/lora_moe_benchmark/ |
| Gap predicts calibration (d=256) | r²=0.865 at N=4 (self-defeating at d=896) | macro/gap_as_signal_bridge/ |
| Hash routing plug-and-play | 5.3% displacement at N=20 | macro/hash_routing_scale/ |
| Prune-then-compose order invariant | +0.012% gap (170x margin) | macro/prune_compose_macro/ |
| L2 norm composition stable | 0/25 catastrophic failures | macro/l2_norm_macro/ |
| Batched LoRA k=1 overhead | -4% (faster than monolithic) | macro/batched_lora_latency/ |
| Compose CLI works E2E | 5 adapters, full workflow tested | macro/compose_e2e/ |

## Killed at Macro

| Finding | Result | Evidence |
|---------|--------|----------|
| SwiGLU gate pruning | +196% quality loss | macro/swiglu_pruning_macro/ |
| Gap-as-signal at d=896 | r²=0.22 (no variance) | macro/gap_signal_lora/ |

## Current Direction: Living Composable Model

The research phase is complete. The proven findings above are the foundation for:

1. **Distill** — Create experts at scale via teacher distillation ($0.25/expert)
2. **Compose** — Serve with hash ring routing on vLLM (zero recalibration)
3. **Evolve** — Clone-and-compete mechanism for continuous improvement without retraining

See `VISION.md` for full architecture. See `HYPOTHESES.yml` for active roadmap (10 focused items).

## Per-Experiment Details

Each macro/ and micro/ experiment directory contains its own PAPER.md with full methodology, results, and kill criteria assessment.

- Macro experiments: `macro/*/PAPER.md`
- Micro experiments: `micro/models/*/PAPER.md`
- Archived hypotheses: `ARCHIVE.yml` (84 completed/deferred experiments)
- Active roadmap: `HYPOTHESES.yml` (10 items, priority-ordered)
