# LEARNINGS: exp_bench_livecodebench_v6

**Status**: Design approved (PROCEED). Results pending (pueue task 11).

---

## Core Finding (Design)
LiveCodeBench v6 benchmark for Gemma 4 E4B-4bit requires `--n 1` + date filter
(2025-01-01/2025-04-30) to fit within M5 Pro 8h budget (~50-100 problems, ~1-3h).
`--n 10` would require 5000+ generations (~100h) — not viable.

## Why
LCB `--n` = samples *per problem* (not total problems). Date filter limits scope to recent
competitive programming problems. Code adapter is CodeAlpaca-trained (trivial instruction
following), not competitive programming — domain gap cos≈0.2 → K1421 (adapter +5pp) expected
to FAIL.

## Key Predictions
- K1420: base 4-bit ≥ 42% — UNCERTAIN (Google 52% in float; 4-bit gap ~5-10pp expected)
- K1421: code adapter ≥ base +5pp — EXPECTED FAIL (CodeAlpaca vs competitive programming gap)
- K1422: eval < 8h — EXPECTED PASS (~1-3h at n=1, ~100 problems)

## Implications for Next Experiment
If K1420 PASSES: 4-bit W4A16 viable for code generation benchmarks (supports P0 pipeline).
If K1420 FAILS: either need higher bit-width or code-generation-specific post-quant fine-tuning.
Adapter delta signal (K1421) will characterize domain-mismatch penalty for LCB-class tasks.
