# PAPER.md: exp_bench_livecodebench_v6

## Type: Guided Exploration
**Status:** Queued (pueue task 11) — results pending

---

## Prediction vs Measurement Table

| # | Prediction | Value | Actual | Pass? |
|---|---|---|---|---|
| Theorem 1 | Base E4B-4bit LCB v6 pass@1 ≥ 42% | 39–47% | TBD | TBD |
| Theorem 2 | Code adapter LCB delta < 5pp | ~1–3pp uplift | TBD | TBD |
| K1420 | Base 4-bit ≥ 42% (within 10pp of 52.0%) | UNCERTAIN | TBD | TBD |
| K1421 | Code adapter ≥ base + 5pp | EXPECTED FAIL | TBD | TBD |
| K1422 | Eval < 8h on M5 Pro (--n 1, ~50-100 problems) | ~1–3h | TBD | TBD |

---

## Design Notes

- **Sampling strategy**: `--n 1` (pass@1 from 1 sample) + `--start_date 2025-01-01 --end_date 2025-04-30`
  limits to ~50–100 recent problems. `--n 10` in LCB means 10 samples *per problem*, not 10 problems —
  full v6 (500+ problems × 10 samples = 5000+ gens) would take ~100h. Fixed: n=1, date-filtered.
- **Reference**: Google reports 52.0% pass@1 for Gemma 4 E4B (float precision, thinking on).
- **Quantization prior**: MMLU-Pro 8-bit→4-bit gap = < 5pp. Code generation may show larger gap
  (open-ended generation harder than MCQ under weight noise). Conservative bound: ≤ 10pp.
- **Code adapter**: code-codealpaca-knowledge-v0 at `micro/models/exp_p1_t2_single_domain_training/adapters/code/`
  (corrected from wrong `adapters/code` root path). Trained on CodeAlpaca-20k (simple tasks).
  HumanEval: 63%. LCB v6 requires competitive programming — domain gap expected.

---

## Findings (to be filled when results.json arrives)

### Base Model Performance
- Pass@1 LCB v6 (n=1, ~50-100 problems): **TBD**
- vs Google 52.0% target: **TBD**
- K1420 (≥ 42%): **TBD**

### Code Adapter Performance
- Pass@1 LCB v6 with adapter (n=1): **TBD**
- Delta vs base: **TBD**
- K1421 (adapter ≥ base + 5pp): **TBD**

### Timing
- Total eval time: **TBD**
- K1422 (< 8h): **TBD**

---

## Key Notes

1. **Domain mismatch**: CodeAlpaca adapter (trivial instruction following) vs LCB v6
   (competitive programming from LeetCode/AtCoder/CodeForces). Theorem 2 gradient alignment
   cos(D_ca, D_lcb) ≈ 0.2 → expected adapter_delta ≈ 2.2pp (FAIL K1421).

2. **Variance**: ~50-100 problems with --n 1 gives ~3-5pp CI at 95%. Directional signal valid.
   Not suitable for 1pp-level comparisons, but sufficient for K1420 (±10pp) and K1421 (±5pp).

3. **If K1420 FAILS**: code generation needs higher bit-width or post-quantization fine-tuning.
   If PASSES: 4-bit W4A16 viable for code generation benchmarks.

---

## Evidence (to be added after run)

```json
{}
```
