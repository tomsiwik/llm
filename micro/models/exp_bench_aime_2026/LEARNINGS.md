# LEARNINGS.md: exp_bench_aime_2026

**Status:** Queued — results pending (pueue task 9)

---

## Core Finding

Design complete. Using MathArena harness to benchmark E4B-4bit on AIME 2026 (30 olympiad problems).
Predicted base pass@2 ≈ 35–38% (10–18% relative quantization loss from Google's 42.5% pass@4).
Math adapter expected to show negligible uplift (<5pp) due to GSM8K ↔ AIME domain gap.

## Why

Theorem 1 grounds the 4-bit quantization penalty in our MMLU-Pro calibration (10.5% relative, Finding #530).
Theorem 2 formalizes the domain transfer bound: GSM8K trains on arithmetic patterns absent from olympiad proofs.
Finding #179 already showed math adapter has ~0pp delta on MMLU-Pro MCQ — AIME expected similar.

## Implications for Next Experiment

1. If K1417 PASSES (~37% within 10pp of 42.5%): 4-bit quantization is not the bottleneck for olympiad reasoning.
   The gap to frontier is architectural/data, not precision. Informs P1 priority.
2. If K1417 FAILS (<27.5%): quantization degrades multi-step symbolic reasoning disproportionately.
   W4A16 verification (task 4) would become higher priority — need 8-bit for olympiad-class tasks.
3. K1418 (adapter uplift) expected FAIL: confirms domain-specific adapters need domain-matched training data.
   To improve AIME, need AIME-class traces (e.g., AoPS problems, not GSM8K).

## Reviewer Issues (resolved/noted)

- **Fix 1 (applied):** Crash guard for `pass_at_n=None` — `pct = "N/A" if None` already in run_experiment.py:167
- **Fix 2 (applied):** K1417 relabeled EXPECTED PASS in MATH.md (~37% is within 10pp of 42.5%)
- **Fix 3 (applied):** PAPER.md skeleton written (this iteration)
- **Non-blocking A:** Theorem 3 per-problem timing inconsistency — accepted, conclusion unaffected
- **Non-blocking B:** MathArena model config model field — may need verification at runtime
