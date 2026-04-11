# REVIEW-adversarial.md — P3.C4: Rank-16 Diverse Adapter

**Verdict: PROCEED (KILLED, Finding #471)**

## Checklist

- [x] PAPER.md has prediction-vs-measurement table
- [x] Kill criteria results match evidence (K1205 FAIL: 73.3% < 80%)
- [x] Finding status appropriate (KILLED = kill criteria failed)
- [x] experiment complete + finding-add already run (previous iteration)
- [x] MATH.md present

## Adversarial Concerns

### Non-blocking: Cache Bug Confound
The training data cache check validated file existence but not line count, resulting in
10 examples being used instead of 167. This confounds data vs rank attribution.

**However**: the direction of the confound actually strengthens the rank hypothesis.
Rank-16 + 10 examples (73.3%) beats rank-4 + 167 examples (60%). If rank were not
the bottleneck, we'd expect the opposite. The cache bug makes Theorem 1 a conservative
test — P3.C5 with correct data should only improve.

### Non-blocking: Theorem 1 Coverage Argument
Theorem 1 predicts rank(16) > n_categories(10) → coverage → 80%+. The measured 73.3%
misses threshold. Three candidate explanations given:
1. Data shortage (10 vs 167) — resolved by P3.C5
2. Question-type floor (within-category variation beyond rank)
3. Category count undercounts required style directions

The PAPER.md correctly identifies these are not distinguishable from this experiment
alone. P3.C5 resolves explanation 1; if C5 still fails at ~73-75%, then 2 or 3 is primary.

### Non-blocking: "Hard floor" Characterization
The 4 failures are all in COVERED categories (physics, CS, earth science), not
underrepresented ones. Philosophy and economics questions PASSED. This is correctly
flagged as surprising — if rank/data were the only constraint, covered categories
should be easiest. The token probability floor for certain question formulations
is real and P3.C5 should test whether 167 diverse examples pushes through it.

## Summary

PAPER.md is rigorous, analysis is honest about the confound, prediction table is
complete, impossibility structure is appropriate. KILLED is correct — K1205 FAIL is
unambiguous. P3.C5 (rank-16 + fix cache + 167 examples) is the natural next step.

**Verdict: PROCEED → Analyst writes LEARNINGS.md**
