# PAPER.md — P11.L0: RSD Aligned Traces

## Experiment Type
Guided Exploration — Proven framework (distribution alignment via Rejection Sampling,
von Neumann 1951 + Liu et al. arXiv:2309.06657), unknown: whether NLL-filtered s1K traces
outperform unfiltered traces on a 4-bit student model (Gemma 4B 4-bit).

---

## Practical Approximation Caveat

**True RSD** requires teacher log-probs P_T(x_t | x_{<t}) to compute the acceptance ratio
P_S(x_t)/P_T(x_t). Since DeepSeek-R1-Distill (the s1K trace generator) is not locally
available, we use a proxy:

- **Approximation**: token t "accepted" if student NLL ≤ 6.9 (i.e., P_S(x_t) ≥ exp(-6.9) ≈ 0.001)
- **Threshold interpretation**: 250× chance level for vocab_size=256k
- **Risk**: This is an absolute threshold, not the ratio P_S/P_T. At vocab_size=256k, the
  threshold may accept 70-90% of tokens trivially, inflating K1543 acceptance rates beyond
  what true RSD would predict.
- **Directional claim still holds**: d_TV(D_rsd, π_S) ≤ d_TV(D_raw, π_S) by construction
  (we keep only traces the student finds plausible). The magnitude of improvement is weaker
  than true RSD.

This experiment should be labeled "NLL-filtered s1K" in downstream references, not "true RSD".

---

## Prediction vs. Measurement Table

| # | Prediction | Basis | Measured | Status |
|---|------------|-------|----------|--------|
| P1 | Acceptance rate: 50–70% of s1K traces pass RSD filter | Theorem 1 Step 2: competition math has complex notation → student assigns low prob | TBD | TBD |
| P2 | NLL scoring: 1000 traces < 24h on M5 Pro (~30–60 min) | K1542 feasibility gate (2s/trace forward pass) | TBD | TBD |
| P3 | RSD adapter MMLU-Pro: 61–65% (+2 to +4pp over P11.F0 raw) | Theorem 1: reduced distribution shift → less gradient variance → less forgetting | TBD | TBD |
| P4 | SERT adapter MMLU-Pro: 61–63% (no forgetting, no breadth) | Theorem 2: d_TV=0 for self-generated traces; GSM8K limited to arithmetic | TBD | TBD |
| P5 | SERT adapter GSM8K: 82–88% (strong on-distribution signal) | Theorem 2: self-generated correct traces → strong task-specific signal | TBD | TBD |

---

## Kill Criteria Summary

| ID | Criterion | Prediction | Measured | Pass? |
|----|-----------|------------|----------|-------|
| K1541 | RSD MMLU-Pro ≥ P11.F0 + 3pp | ~63–65% (if P11.F0 ≈ 60%) | TBD | TBD |
| K1542 | NLL scoring time < 24h | ~30–60 min | TBD | TBD |
| K1543 | ≥60% of s1K traces pass filter | 50–70% expected | TBD | TBD |

---

## Smoke Test Note

No smoke test completed before REVISE round 1 — the `--train-splits`/`--val-splits` bug
(blocking fix 2) would have caused training to fail silently (argparse exit 2). After
applying the subdirectory fix (data/rsd/ and data/sert/ with standard train.jsonl/valid.jsonl),
a smoke test should be run to verify training succeeds before the full queue run.

---

## P11.F0 Baseline Note

K1541 depends on P11.F0 MMLU-Pro accuracy. The conservative default is 60.0% (used if
P11.F0 results.json is unavailable). Note: P11.F0 had a known thinking regex bug in its
base eval phase (Phase 4a showed 12.5% with wrong regex), but the adapter eval (Phase 4b)
may be valid. If P11.F0 results are unavailable or suspect, K1541 falls back to 60.0%.

---

## Connection to Vision

This experiment addresses data quality for domain adapter training — a prerequisite for
P11.M0 (Full Pipeline v2). If teachers generate traces misaligned with our 4-bit student,
every domain adapter suffers distribution-induced forgetting. RSD/NLL filtering provides
a cheap (one forward pass per trace) quality gate.
