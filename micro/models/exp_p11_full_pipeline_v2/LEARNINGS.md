# LEARNINGS.md — P11.M0: Full Pipeline v2

## Core Finding
Design reviewed and approved (PROCEED). Experiment not yet run — depends on adapter outputs from
exp_p11_rsd_aligned_traces (task 19), exp_p11_grpo_improve (task 20), and exp_p11_injection_decoding (task 13).

## Why
Three theorems motivate the pipeline: (1) weight-space adapter + input-space PS prompt are
approximately independent, enabling additive decomposition; (2) plan-and-solve priming (arXiv:2209.01510)
acts as beam search initialization in thinking space; (3) injection decoding is pre-registered as ~0pp
because D_avg=2614 >> D_threshold=1500 (Finding #P11.Z1).

## Key Pre-Registrations
- K1546c (injection >= 1pp): **EXPECTED FAIL** by design — injection threshold never triggers at Gemma 4 token scale
- K1544 (MMLU-Pro >= 70%): UNCERTAIN — expected range 67-69%, high variance at 35q/condition
- K1546_all omnibus: always FALSE by design (K1546c pre-registered FAIL); use K1546a + K1546b individually

## Implications for Next Experiment
When full run completes, the operative finding will be the additive delta table:
adapter Δ + PS Δ ≈ total gain. If adapter Δ ≈ 0 (adapter fallback triggered), K1546b fails and
the pipeline collapses to a PS-only result — that would redirect effort toward better domain adapters.
