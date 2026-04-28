# Cross-Domain Interference Matrix — Results

## Summary
3x3 evaluation matrix: each of 3 adapters (math, python, medical) evaluated on all 3 benchmarks (GSM8K, HumanEval, MedQA) plus base model baselines. N=50 per evaluation.

## Prediction vs Measurement

| Prediction | Expected | Measured | Match |
|---|---|---|---|
| On-domain lift ≥10pp | ≥10pp each | +22, +48, +62pp | YES |
| Off-domain degradation ≤3pp | ≤3pp each | -14pp (python→GSM8K), -12pp (medical→HumanEval) | NO |
| Asymmetry (gains >> losses) | gains >> losses | gains 22-62pp, losses 12-14pp | PARTIAL |

## Full Matrix

### Base model
| Benchmark | Accuracy |
|---|---|
| GSM8K | 50.0% |
| HumanEval | 22.0% |
| MedQA | 6.0% |

### Adapter × Benchmark (delta from base)
| Adapter | GSM8K | HumanEval | MedQA |
|---|---|---|---|
| math | **72.0%** (+22) | 24.0% (+2) | 14.0% (+8) |
| python | 36.0% (-14) | **70.0%** (+48) | 56.0% (+50) |
| medical | 66.0% (+16) | 10.0% (-12) | **68.0%** (+62) |

Bold = on-domain.

## Kill Criteria

| KC | Threshold | Result | Verdict |
|---|---|---|---|
| K2067: off-domain ≤3pp degradation | all off-domain ≥ -3pp | worst = -14pp (python→GSM8K) | **FAIL** |
| K2068: on-domain ≥10pp improvement | all on-domain ≥ +10pp | min = +22pp (math→GSM8K) | **PASS** |

## Observations

1. **Interference is real and asymmetric**: python adapter badly hurts math (-14pp), medical hurts code (-12pp). Math adapter is relatively benign off-domain (+2, +8).
2. **Surprising positive transfers**: python helps MedQA (+50pp), medical helps GSM8K (+16pp). These adapters learned general reasoning, not just domain-specific skills.
3. **Interference pattern**: adapters that learned strong general features (python, medical) interfere more aggressively with other domains. The math adapter, which learned narrower numeric reasoning, is more contained.
4. **Implication for composition**: naive uniform composition will suffer from the python→math and medical→code interference. Gating/routing is necessary, but must be soft (per Finding #826, hard routing loses to uniform composition).

## Verdict
**KILLED** — K2067 FAIL (off-domain degradation exceeds 3pp threshold). On-domain value is strong (K2068 PASS), confirming adapters work. But cross-domain interference is real and substantial enough to require routing.
