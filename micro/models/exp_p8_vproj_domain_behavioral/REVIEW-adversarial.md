# REVIEW — exp_p8_vproj_domain_behavioral

**Verdict: PROCEED**

## Summary

Core finding is sound: v_proj+o_proj adapters improve behavioral text quality vs q_proj
across all 5 domains. Data is consistent, per-query results verify aggregate rates,
and MATH.md reasoning about output-path vs query-path modification is mechanistically correct.
Status SUPPORTED is appropriate (2/4 kill criteria pass, directional finding validated).

## Issues

### 1. K1315 composition test is trivially satisfied (non-blocking)

K1315 reports 100% retention under "composition" — but this is sequential serving where
each adapter runs independently. By construction, solo and composition rates are identical
because no parameter merging occurs. Theorem 3 (MATH.md) predicts retention under actual
Grassmannian parameter composition (merged ΔW). The test validates serving infrastructure
(already confirmed by Finding #503) but does not test the theorem's prediction.

**Impact on finding:** Does not invalidate the v_proj+o_proj mechanism finding. But K1315
should be reframed in PAPER.md as "sequential serving confirmed" rather than "composition
retention validated." Actual composition testing is deferred to an experiment that merges
adapter weights.

### 2. Legal at 35% undercuts "all domains improve" narrative (non-blocking)

Domain improvement rates: medical 70%, math 55%, code 50%, finance 50%, legal 35%.
Only 1/5 domains exceeds 60%. Legal's 35% is notably weak — worse than finance and code
despite legal having rich domain-specific vocabulary. PAPER.md discusses math/code ceiling
effects but doesn't address legal's underperformance.

**Note:** Legal still improved vs q_proj (20% → 35%), so the directional claim holds.
But the magnitude is concerning and worth acknowledging.

### 3. "Ceiling effect" explanation is post-hoc (non-blocking)

MATH.md predictions for math (70-80%) and code (65-75%) substantially overestimate
measured values (55%, 50%). The post-hoc explanation (base model already competent,
limited training data) is plausible but unfalsifiable. A stronger experiment would
predict which domains face ceiling effects based on base model competence scores.

**Note:** The 80-example training set (8-10 unique, cycled) is genuinely small. This
is a reasonable limitation to acknowledge, not a finding flaw.

## Data Integrity Check

- Per-query counts match aggregate rates: math 11/20=55% ✓, code 10/20=50% ✓,
  medical 14/20=70% ✓, legal 7/20=35% ✓, finance 10/20=50% ✓
- results.json kill_criteria flags match computed rates ✓
- Training times consistent (2.1-2.8 min per domain, 12.2 min total) ✓
- q_proj baseline values match behavioral E2E killed experiment claims ✓

## Verdict Rationale

PROCEED. The finding that v_proj+o_proj is the correct projection target for behavioral
quality is well-supported by data across 5 domains and correctly explained by the
output-path mechanism. Kill criteria results are honestly reported. The non-blocking
issues are caveats for the analyst to capture in LEARNINGS.md, not blocking revisions.
