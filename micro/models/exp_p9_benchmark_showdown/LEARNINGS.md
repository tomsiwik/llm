# LEARNINGS: P9.G1 — Benchmark Showdown

**Status**: Design complete, PROCEED. Experiment not yet run (no results.json).

---

## Core Finding (Design Phase)

Both tautological kill criteria (K1391, K1392) were replaced with freshly-measured
gains: K1391 now measures math adapter GSM8K gain over base (phases 1+2), K1392
measures medical adapter MedMCQA gain over base (phases 3+4). Neither can be
pre-determined from hardcoded constants.

## Why

Tautological criteria (63-42=21≥20 always PASS, 4.3B/27.2B=14.8%<50% always PASS)
are anti-patterns — they consume experiment slots without generating falsifiable
knowledge. The REVISE round correctly identified and removed both.

## Predictions Before Run

- K1390 (math ≥ 27B 90%): LIKELY FAIL — math adapter 82% < 90% published benchmark
- K1391 (math gain ≥ base + 20pp): EXPECTED PASS — if base ~55%, gain ~27pp
- K1392 (medical gain ≥ base + 3pp): UNCERTAIN — medical adapter 50% MedMCQA; base unknown

Finding #517 (math adapter hurts MCQ: -26pp on MMLU-Pro) predicts oracle routing
will underperform base on general knowledge categories.

## Implications for Next Experiment

If K1390 FAILs (math adapter 82% < 27B 90%), the 8pp gap quantifies "scale debt"
— the improvement needed from P1 (GRPO + thinking + s1K) to close parity with 27B.
This sets a concrete target for P11 experiments.

If K1392 FAILs (medical adapter adds <3pp), it confirms domain adapters trained on
NTP with q_proj only provide minimal MCQ lift — consistent with Finding #517.
The fix is GRPO-style reinforcement training rather than NTP.
