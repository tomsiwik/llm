# LGME Ablation Findings

## 32-Combination Ablation Study

All 2^5 combinations of phase flags were tested:

| Phase | Flag | Effect on Loss | Verdict |
|-------|------|----------------|---------|
| 1. ART-modulated LR | `PHASE1_ART_LR` | +0.028 (hurts) | Starves optimizer on KNOWN inputs |
| 2. Bloom filter gate | `PHASE2_BLOOM_GATE` | ~0.000 (none) | Zero effect on loss |
| 3. Splay cache | `PHASE3_SPLAY_CACHE` | ~0.000 (none) | Zero effect on loss |
| 4. MoE routing | `PHASE4_MLP_ROUTING` | -0.003 (helps) | Only phase that improves loss |
| 5. ART spawn | `PHASE5_ART_SPAWN` | +0.005 (hurts) | Adam buffer disruption on consolidation |

## Key Results

- **ALL ON was the worst configuration** — phases compound each other's harm
- **MoE routing is the only beneficial phase** — slight but consistent improvement
- **Bloom filter, Splay tree, HNSW**: zero measurable effect on training loss
- **ART-LR modulation**: actively harmful — scaling down LR for "known" inputs prevents the optimizer from refining already-learned patterns
- **Spawning/consolidation**: hurts because merging experts disrupts Adam momentum/variance buffers

## Pivot Decision

The "cognitive stack as routing optimizer" narrative is dead. The valuable research direction is **continual learning without full retraining**, where MoE expert isolation is the core mechanism.

### Supporting Literature (ICLR 2025, Feb 2025)

- MoE expert isolation at the FFN layer works for preventing forgetting
- **Shared attention is the forgetting bottleneck** — not the experts
- EWC on shared attention params is the known fix
- Kohonen routing for MoE-CL appears to be novel (no published precedent found)

## Archived Artifacts

- `archive/ablation.py` — the 32-combination ablation runner
- `archive/ablation_results.csv` — raw results
- `archive/ablation_chart.png` — visualization
