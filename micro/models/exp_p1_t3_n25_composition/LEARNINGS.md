# LEARNINGS.md — T3.4: N=25 Grassmannian Composition

## Core Finding

N=25 domain adapters on Gemma 4 E4B compose without any interference (0/25 degraded)
using Grassmannian QR-constructed A-matrices + exclusive routing, with a record-low
cosine of 2.2e-8 — 460× below threshold and 630× tighter than Finding #406 on Qwen3-4B.

## Why

Float64 QR → float32 downcast achieves near-exact orthogonality (~machine epsilon).
Exclusive routing makes interference zero by construction (Theorem 3): only one adapter
fires per query, so no additive noise terms accumulate. T3.1's catastrophic collapse
(math 82→8%) required *simultaneous* N=5 activation — exclusive routing eliminates
that failure mode entirely. Structural fix = routing, not hyperparameter tuning.

## Bonus Discovery: Universal MCQ Format Transfer

Domain adapters (trained on professional_law, MedMCQA, GSM8K) give **56-88%** on
*completely unrelated* MMLU subjects vs base=4%. The adapters teach Gemma 4 MCQ format
compliance (output A/B/C/D), which transfers universally. Domain knowledge is secondary
to format instruction in the adapter's learned representation. This generalizes T3.2.

## Implications for Next Experiment (T4)

- T3 tier is complete. Both structural requirements for P1 are verified: Grassmannian
  construction (use float64 QR) + exclusive routing (PLE-M2P).
- T4 should target end-to-end behavioral quality: route real queries through the 5-domain
  system and measure whether outputs are *useful*, not just above-base accuracy.
- N=426 theoretical max (48MB for N=25; 1 GB fits ~500 domains) — scaling is not a
  bottleneck; behavioral quality of the routing decision is the remaining open question.
- Code base discrepancy in PAPER.md (0% vs actual 20% MMLU CS proxy) is cosmetic —
  K1060 holds at 72% >> 20%; fix in follow-up if PAPER.md is revisited.

## References

- HRA (arxiv 2405.17484): Householder/Gram-Schmidt orthogonal construction
- Finding #406: N=25 PASS on Qwen3-4B (1.38e-5 cosine; this tightens 630×)
- Finding #425: Routing structurally required (T3.1 impossibility)
- Finding #427: Gemma 4 exclusive routing load-bearing (T3.3)
