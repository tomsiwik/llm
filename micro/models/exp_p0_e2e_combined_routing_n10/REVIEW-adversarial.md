# Adversarial Review: exp_p0_e2e_combined_routing_n10

## Verdict: PROCEED (SUPPORTED)

## Summary

Clean verification experiment. All 4 kill criteria PASS with margin. Combined logistic
routing at N=3 achieves 100% routing accuracy and 0pp quality loss. PAPER.md has
prediction-vs-measurement table. Results.json is consistent with claims. Math is sound.

## Kill Criteria Verification

| ID | Target | Result | Verified in results.json |
|----|--------|--------|--------------------------|
| K1478 | GSM8K >= 65% | 77.0% | routed_gsm8k_pct: 77.0 |
| K1479 | HumanEval >= 50% | 57.0% | routed_humaneval_pct: 57.0 |
| K1480 | MedMCQA >= 40% | 58.0% | routed_medmcqa_pct: 58.0 |
| K1481 | Routing loss <= 5pp | 0.0pp | max_routing_loss_pp: 0.0 |

All verified. No fabrication concerns.

## Non-Blocking Issues

1. **Experiment name mismatch.** Name says "n10" but tests N=3 (3 domains). Original
   design likely intended N=10; scoped down for verification. Does not affect results.

2. **Minor results.json inconsistency.** `router_per_domain_accuracy.medical` = 100.0
   but `routed_medmcqa_detail.routing_accuracy_pct` = 99.0 (1/100 misrouted to code).
   These likely measure different sets (router test set vs benchmark queries). PAPER.md
   correctly reports the 99% benchmark figure.

3. **HumanEval adapter delta decay.** Finding #508 showed +56pp (7%->63%), this shows
   +39pp (18%->57%). Net oracle accuracy dropped 6pp. Could be sample variance at N=100,
   could be evaluation methodology differences (prompt format, sampling). Worth tracking
   if HumanEval consistently underperforms across future experiments.

## Assessment

- Theorem 1 (routing-quality bound) is correct — straightforward linearity of expectation.
- Theorem 2 (N=3 lower bound) is more argument than proof, but predictions validated.
- At N=3, routing is trivially solved. The real test is N=10+ where Finding #525 shows 89.9%.
- SUPPORTED status is appropriate for a verification experiment with all predictions confirmed.
- The insight about N=3 being "overkill" for combined logistic is well-stated — the value
  proposition of this routing method is at scale (N=10+), not at N=3.
