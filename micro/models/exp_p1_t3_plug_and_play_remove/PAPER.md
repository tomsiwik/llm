# T3.7: Plug-and-Play Hot-Remove — Prediction vs Measurement

## Summary

Hot-remove of an adapter from a live registry is structurally free under exclusive routing.
The symmetric pair T3.6 (hot-add) + T3.7 (hot-remove) completes the plug-and-play proof.

## Prediction vs Measurement Table

| Kill | Theorem | Prediction | Measured | Result |
|------|---------|-----------|----------|--------|
| K1070: bit-exact remaining outputs | Theorem 1 | 0 token differences across N×4 domains | 0/40 diffs (math×10, medical×10, legal×10, finance×10) | **PASS** |
| K1071: freed slot immediately reusable | Theorem 2 | new adapter > 4% on new domain | history = 100% vs base = 4% | **PASS** |
| K1072: p99 remove latency < 10ms | Theorem 3 | ~0.005ms | p99 = 0.000922ms (10,800× margin) | **PASS** |

## Experiment Setup

- Registry: 6 domains (math, code, medical, legal, finance, geography)
- Phase 1: Collect pre-removal outputs for 4 domains × 10 queries each
- Phase 2: Hot-remove geography; verify remaining outputs unchanged
- Phase 3: Hot-add history adapter; verify >4% accuracy
- Phase 4: N=100 latency microbenchmark for remove operation

## Results Detail

**K1070 — Remove Invariance:**  
All four remaining domains (math, medical, legal, finance) produced 0/10 token changes
after geography removal. Total differences = 0/40.

Pre/post accuracy identical:
- math: 0% → 0% (same queries, base adapter)
- medical: 50% → 50%
- legal: 40% → 40%
- finance: 80% → 80%

**K1071 — Freed Slot Reusable:**  
Geography label freed. New "history" adapter (high_school_european_history) added.  
history accuracy = 100% vs base = 4% (+96pp). Immediate functionality confirmed.

**K1072 — Latency:**  
N=100 microbenchmark: mean=0.000205ms, p99=0.000922ms, max=0.0015ms.  
All < 1ms threshold. 10,800× below the 10ms kill criterion.

**Runtime:** 116s total (49.5s pre-eval + 45s post+history eval + 13.4s latency bench)

## Interpretation

Exclusive routing makes hot-remove structurally free by the same argument as hot-add (T3.6):
adapter j depends only on R[j], which is untouched by removal of k ≠ j. The proof is
algebraically exact — no approximation or statistical component.

The plug-and-play interface (add/remove in < 1ms, zero interference to peers) is now
fully verified for the N=5 real-adapter registry at both add and remove directions.

## Connection to T3 Findings

| Experiment | Status | Key Result |
|-----------|--------|-----------|
| T3.1 (pairwise_interference) | KILLED | Simultaneous N=5 activation → math 82→8%, routing required |
| T3.2 (mmlu_preservation) | KILLED | Scale ≥ 12 degrades MMLU, scale=6 is safe |
| T3.3 (activation_space_bounds) | SUPPORTED | Power law alpha=0.15, routing load-bearing |
| T3.4 (n25_composition) | SUPPORTED | N=25 Grassmannian, max cos=2.2e-8 |
| T3.6 (plug_and_play_add) | SUPPORTED | Hot-add: bit-exact, 0.004ms, immediate |
| **T3.7 (plug_and_play_remove)** | **SUPPORTED** | **Hot-remove: bit-exact, 0.001ms, slot reusable** |
