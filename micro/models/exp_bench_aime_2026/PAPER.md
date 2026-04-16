# PAPER.md: exp_bench_aime_2026

## Type: Guided Exploration
**Status:** Queued (pueue task 9) — results pending

---

## Prediction vs Measurement Table

| # | Prediction | Value | Actual | Pass? |
|---|---|---|---|---|
| Theorem 1 | Base E4B-4bit pass@2 AIME 2026 | 30–40% | TBD | TBD |
| Theorem 2 | Math adapter AIME uplift | < 5pp (≈ 0) | TBD | TBD |
| K1417 | Base within 10pp of 42.5% (range: 32.5–52.5%) | EXPECTED PASS (~37%) | TBD | TBD |
| K1418 | Math adapter ≥ base + 10pp | EXPECTED FAIL | TBD | TBD |
| K1419 | Eval completes in < 2h (n=2 seeds) | EXPECTED PASS | TBD | TBD |

---

## Design Notes

- **n=2 seeds** (not n=4): Theorem 3 shows n=4 × 30 problems × 90s ≈ 3h, exceeding budget.
  n=2 × 30 × 60s ≈ 60min. Results are higher-variance but within budget.
- **Reference point:** Google reports 42.5% pass@4 for E4B (likely full-precision, thinking on).
- **Quantization prior:** MMLU-Pro degradation = 7.3pp absolute (10.5% relative).
  On harder multi-step tasks (AIME), expected 10–18% relative degradation → ~35–38% base.

---

## Findings (to be filled when results.json arrives)

### Base Model Performance
- Pass@2 AIME 2026: **TBD**
- vs Google 42.5% target: **TBD**
- K1417 (within 10pp): **TBD**

### Math Adapter Performance
- Pass@2 AIME 2026: **TBD**
- Delta vs base: **TBD**
- K1418 (adapter ≥ base + 10pp): **TBD**

### Timing
- Total eval time: **TBD**
- K1419 (< 2h): **TBD**

---

## Key Notes

1. **Domain mismatch:** GSM8K math adapter (grade-school arithmetic) vs AIME (olympiad).
   Finding #179 already showed adapter has ~0pp delta on MMLU-Pro (MCQ regression).
   AIME uplift expected to be negligible or negative.

2. **High variance:** N=30 problems, so 1 problem = 3.3pp. Results have wide CI.
   Pass@2 vs pass@4 also increases variance (fewer independent draws per problem).

3. **Reviewer flags (non-blocking):**
   - Theorem 3 per-problem timing inconsistency (n=2: 60s/prob vs n=4: 90s/prob) — accepted
   - MathArena model config `model` field may need to match mlx_lm.server advertised ID

---

## Evidence (to be added after run)

```json
{}
```
