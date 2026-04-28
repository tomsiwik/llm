# PAPER.md — N=3 Composition Test

## Summary

Tested whether N=3 LoRA adapters (math, code, medical) composed via correct Σ(A_i @ B_i) preserve per-domain accuracy within 5pp of single-adapter baselines. **Result: KILLED.** Composition degrades math by 10pp and code by 12pp, exceeding the 5pp tolerance. The composition is non-tautological (PPL differs) and cross-domain interference is low, but the accuracy penalty from naive uniform averaging is too large.

## Prediction vs Measurement

| Prediction | Predicted | Measured | Match |
|---|---|---|---|
| Per-domain drop ≤5pp | ≤5pp all domains | math -10pp, code -12pp, med -2pp | NO |
| PPL differs (not tautological) | diff > 0.01 | math 31.94, code 22.84, med 720.72 | YES |
| Math adapter on MedQA ≤55% | ≤55% | 14.0% | YES |

## Single-Adapter Baselines (N=50)

| Domain | Benchmark | Accuracy | PPL |
|---|---|---|---|
| Math | GSM8K | 72.0% | 82.72 |
| Code | HumanEval | 70.0% | 62.47 |
| Medical | MedQA | 68.0% | 3320.12 |

## Composed Model (N=3, scale=2.0 per adapter)

| Domain | Benchmark | Accuracy | PPL | Drop (pp) |
|---|---|---|---|---|
| Math | GSM8K | 62.0% | 114.66 | -10.0 |
| Code | HumanEval | 58.0% | 85.31 | -12.0 |
| Medical | MedQA | 66.0% | 2599.40 | -2.0 |

## Kill Criteria

| KC | Metric | Threshold | Result | Verdict |
|---|---|---|---|---|
| K2062 | Accuracy drop ≤5pp | GSM8K ≥67%, HumanEval ≥65%, MedQA ≥63% | 62%, 58%, 66% | **FAIL** |
| K2063 | PPL differs from single | diff > 0.01 | 31.94, 22.84, 720.72 | PASS |
| K2064 | Math on MedQA ≤55% | ≤55% | 14.0% | PASS |

## Analysis

1. **Uniform averaging hurts.** Dividing lora_scale equally (6/3=2.0 per adapter) dilutes each adapter's contribution. Math and code adapters lose ~10-12pp; medical loses only 2pp (possibly because medical PPL is already very high, suggesting the adapter provides a different kind of signal).

2. **Non-tautological composition confirmed.** PPL values differ substantially between single-adapter and composed models across all domains, confirming the composition is doing real work — just not enough of the right work.

3. **Cross-domain interference is low.** Math adapter on MedQA scores 14%, well below the medical baseline of 68%, confirming domain adapters don't leak across domains.

4. **Implication for routing.** Since uniform composition degrades accuracy, per-sample routing (selecting the right adapter per input) is necessary. This directly motivates `exp_p1_n3_routing_accuracy`.

## Verdict

**KILLED** — K2062 fails. Naive uniform composition with Σ(A_i @ B_i) / N degrades math and code accuracy beyond tolerance. The mechanism (composition) works (non-tautological, no cross-contamination), but uniform weighting is insufficient. Routing or learned composition weights are required.

## Runtime

- Platform: Apple M5 Pro 48GB, MLX
- Model: gemma-4-e4b-it-4bit
- N_eval: 50 per benchmark
- Total time: 638s (~10.6 min)
