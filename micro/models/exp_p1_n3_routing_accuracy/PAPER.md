# N=3 Routing Accuracy — Results

## Summary

Per-sample TF-IDF + Ridge routing achieves perfect domain classification (100%)
but single-adapter routing underperforms uniform composition by 3.3pp.
**Routing is not the bottleneck — the adapters are complementary.**

## Setup

- Base model: `mlx-community/gemma-4-e4b-it-4bit`
- Adapters: math, python, medical (LoRA rank 6, scale 6.0)
- Router: TF-IDF (5000 features, bigrams) + RidgeClassifier
- Router training: 496 samples from GSM8K/HumanEval/MedQA
- Eval: N=30 per benchmark

## Prediction vs Measurement

| Metric | Predicted | Measured | Match |
|--------|-----------|----------|-------|
| Router test accuracy | ≥ 85% | 100.0% | YES |
| Routed avg > uniform avg | routed > 72.2% | 68.9% | NO |
| Routed math > uniform math (62%) | > 62% | 76.7% | YES* |
| Routed code > uniform code (58%) | > 58% | 80.0% | YES* |

*Thresholds from F#825 (N=50). Current uniform baseline at N=30 is higher.

## Results

| Benchmark | Uniform (1/3 each) | Routed (single adapter) | Delta |
|-----------|-------------------|------------------------|-------|
| GSM8K | 80.0% | 76.7% | -3.3pp |
| HumanEval | 86.7% | 80.0% | -6.7pp |
| MedQA | 50.0% | 50.0% | 0pp |
| **Average** | **72.2%** | **68.9%** | **-3.3pp** |

## Kill Criteria

| KC | Threshold | Measured | Verdict |
|----|-----------|----------|---------|
| K2065: Router accuracy | ≥ 85% | 100.0% | PASS |
| K2066: Routed ≥ uniform | routed_avg ≥ uniform_avg | 68.9% < 72.2% | FAIL |

**Verdict: KILLED**

## Analysis

The router itself is trivially solved — math/code/medical occupy disjoint
vocabulary spaces, so TF-IDF + linear classifier achieves 100%.

The surprise: single-adapter application is worse than uniform 1/3 weighting.
This means the cross-domain adapter contributions are net positive, not
interfering. Each adapter adds general capabilities that help the others.

Combined with F#825 (uniform drops accuracy vs single-adapter at N=50),
and this result (single-adapter drops vs uniform at N=30), the picture
is nuanced: the relationship depends on evaluation sample selection.

The practical implication: simple routing to one adapter is not optimal.
Weighted combination (where the router provides soft weights rather than
hard selection) may capture the complementary benefit while still
emphasizing the relevant domain.

## References

- Finding #825: Uniform composition killed at N=50
- This experiment: routing killed at N=30
