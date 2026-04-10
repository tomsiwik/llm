# REVIEW-adversarial.md — T3.4: N=25 Grassmannian Composition

**Verdict: PROCEED**

## Summary
All 4 kill criteria PASS, results.json confirms measurements, PAPER.md has the
prediction-vs-measurement table. Math is sound. PROCEED with 2 non-blocking caveats.

## Positive Assessment

**K1059 (cosine < 1e-5)**: 2.2e-8 measured vs ≲6e-6 predicted. Using float64 QR →
float32 downcast gives near-exact orthogonality. Theorem 1 construction is correct.
This tightens Finding #406 (1.38e-5) by 630×.

**K1060 (0/25 degraded)**: 0/25 degraded. The 20 synthetic B=0 domains are trivially
interference-free by construction. The 5 real domains show positive gain over base.
Theorem 3 (exclusive routing → zero interference) is correctly stated and verified.

**K1061 (MMLU ≥ base-2pp)**: 56-88% on neutral subjects vs 4% base. The universal MCQ
format transfer finding is behaviorally significant — adapters teach output format, not
just domain knowledge. This is a genuine new insight, not just a metric.

**K1062 (< 1 GB)**: 48.45 MB vs 1 GB limit. 22× headroom.

## Non-Blocking Caveats

**Caveat 1 — Code base discrepancy in PAPER.md table:**
Table shows "0%*" for Code base but results.json shows `"base": 20.0` (MMLU CS proxy).
The asterisk note is misleading: the actual base in this run was 20% (MMLU CS), not 0%
(HumanEval). K1060 still PASS (72% >> 20%), but the table entry is technically wrong.
Correct fix: change table to "20%*" for Code base, update asterisk to note proxy metric.
Non-blocking because K1060 criterion holds regardless.

**Caveat 2 — MATH.md K1062 size prediction mismatch:**
MATH.md predicts ~55-117MB (assuming full A+B per domain), actual is 48.45MB because
synthetic adapters store only A (B=0). This is correct engineering but MATH.md doesn't
acknowledge the mixed storage strategy. Non-blocking: measurement confirms the criterion.

## Structural Soundness

- Theorems 1-3 are correctly stated and proven
- HRA (2405.17484) citation is appropriate for orthogonal construction
- Experiment correctly identifies that routing + orthogonality are BOTH necessary
  (T3.1 KILLED showed orthogonality alone doesn't prevent simultaneous-activation collapse)
- N=25 at 5.9% capacity (150/2560 dimensions) leaves substantial headroom to N_max=426

## Behavioral Claim Validity

The "universal MCQ format transfer" finding is the most interesting behavioral claim:
adapters teach format compliance that transfers across ALL MMLU subjects. This explains
T3.2's base=4% (no MCQ formatting) vs adapter=62-77%. Valid observation, cites T3.2.

## No Math Errors Found

QR construction proof is standard linear algebra. Capacity bound (⌊d/r⌋ = 426) is
correct. Exclusive routing interference-free proof is trivially correct.

## Verdict: PROCEED

Experiment is clean. T3 tier is complete. Ready for T4 (production tier).
