# N=3 Routing Accuracy

## Theorem (Router Separability)

Given N=3 domain-specific datasets D_i with distinct vocabulary distributions,
a TF-IDF + linear classifier achieves top-1 routing accuracy ≥ 85% on held-out samples.

**Proof sketch.** Math (GSM8K), code (HumanEval), and medical (MedQA) occupy
near-disjoint TF-IDF subspaces: medical has clinical terminology, code has
Python syntax tokens, math has numeric/algebraic patterns. A ridge classifier
on 5000 TF-IDF features with bigrams is a linear separator in this high-dimensional
space. Since the domains are semantically distant, the margin is large and
generalization error is low (standard VC/Rademacher argument for linear classifiers
on well-separated clusters). ∎

## Theorem (Routing ≥ Uniform Composition)

For adapters {A_i} trained on distinct domains, per-sample routing to the correct
single adapter yields task accuracy ≥ uniform composition Σ(A_i)/N.

**Proof sketch.** From Finding #825 (exp_p1_n3_composition_test), uniform 1/3
weighting degrades accuracy by 10-12pp on math/code. This is expected:
each adapter's ΔW is tuned for its domain; averaging dilutes the signal by 1/N
and adds cross-domain noise. Routing to the correct single adapter applies
the full ΔW without dilution. If the router is accurate (≥85% from above),
the routed system applies the correct adapter ≥85% of the time, recovering
most of the single-adapter accuracy. ∎

## Predictions

| Metric | Prediction | Basis |
|--------|-----------|-------|
| Router test accuracy | ≥ 85% | Large TF-IDF margin between domains |
| Routed avg accuracy | > uniform avg | Full adapter signal vs 1/3 diluted |
| Routed math accuracy | > 62% (uniform) | Single math adapter ≈ 72% from F#825 |
| Routed code accuracy | > 58% (uniform) | Single code adapter ≈ 70% from F#825 |

## References

- Finding #825: Uniform composition killed (math 72→62%, code 70→58%)
- SIGReg method: routing eliminates interference by construction
