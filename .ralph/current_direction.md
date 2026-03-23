# Current Direction: exp_bitnet_retrain_evolve

## Hypothesis
Retrain-from-scratch + quality gate produces monotonic adapter improvement.

## Focus
Legal domain adapter: the worst adapter (degenerate, trained on fallback data).
Three rounds of retrain with progressively better data from nguha/legalbench.
Quality gate = KR-Test delta > 0.03 AND |cos| < 0.01 with existing experts.

## Key Insight
The original legal adapter failed because HuggingFace download failed and it
trained on 80 copies of the same fallback sentence. Retraining from scratch on
REAL legalbench data (multiple subtasks, diverse examples) should produce a
fundamentally better adapter. This is the Evolve mechanism: retrain, evaluate,
gate, accept.

## Kill Criteria
- K1: retrained adapter not better than original on KR-Test
- K2: quality gate fails to distinguish good from bad adapters

## Plan
1. Download REAL legalbench data (multiple subtasks, not just contract_nli)
2. Round 1: Train legal adapter from scratch on 800 diverse legalbench samples
3. Evaluate via KR-Test + cosine + PPL
4. Round 2: Add more data (1200 samples from different subtasks)
5. Round 3: Add even more data (1600 samples, widest coverage)
6. Verify monotonic improvement across rounds
7. Quality gate: each round must beat previous on KR-Test delta
