# E2E N=25 Quality Validation — Learnings

## Outcome: All kill criteria PASS. P0 "25 domains" gate CLOSED.

### What We Proved
Theorem 1 (Q_routed = α·Q_oracle + (1−α)·Q_base) holds at N=25 with ≤1.6pp
prediction error when measured routing accuracy is used. The E2E pipeline
scales from N=3 → N=10 → N=25 without quality degradation.

### Key Numbers
- GSM8K: 76% routed vs 77% oracle → 1pp loss (predicted 3.1pp)
- HumanEval: 57% routed vs 57% oracle → 0pp loss (predicted 4.7pp)
- MedMCQA: 56% routed vs 58% oracle → 2pp loss (predicted 6.0pp)
- Max loss: 2.0pp (K1489 threshold: 10pp) — 5× better than threshold
- N=10 → N=25 routing accuracy: IMPROVED (math 98%, code 100%, medical 88%)

### Unexpected Finding: Routing Improves With Scale
A-priori predictions assumed routing accuracy would degrade when going from
N=10 to N=25 (more distractors). The opposite happened: routing improved
by 0-3pp for all adapter domains. Hypothesis: the logistic classifier with
sentence embeddings benefits from MORE contrastive examples — the 15 additional
MMLU subjects help the decision boundary learn what math/code/medical is NOT.
This is consistent with the linear separability of sentence embedding spaces.

### What This Closes
- P0 gate "25 domains" is CLOSED by this experiment
- Combined logistic routing + sentence embeddings is validated as the
  production routing architecture for N=25
- Theorem 1 is validated across N=3, N=10, N=25 (3 independent measurements)

### Structural Note on Misrouting
The 12% MedMCQA misroutes (chemistry, statistics, physics) all go to
non-adapter MMLU domains → base model fallback. This is BENIGN — no
wrong-adapter risk. The base model quality (28%) averages into the 56%
result (vs 58% oracle). If medical queries went to a wrong adapter (e.g.,
code adapter), quality could drop to 0-5%. The current 25-domain structure
with only 3 adapters provides a natural safety zone.

### Next Steps (from PAPER.md)
1. Train adapters for more domains (legal, finance, etc.) to increase coverage
2. Test with 25 trained adapters (not just 3) to measure wrong-adapter risk
3. The N=100 scaling path is clear if α > 85% per adapter domain
