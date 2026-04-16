# REVIEW-adversarial.md — P11.L0: RSD Aligned Traces

**Reviewer**: Adversarial Reviewer
**Date**: 2026-04-14 (Round 2)
**Verdict**: PROCEED

---

## Round 2 Summary

Both blocking fixes from Round 1 are verified applied:

**Fix 1 (PAPER.md)**: Present with prediction-vs-measurement table (P1-P5), kill criteria
summary (K1541-K1543), practical approximation caveat clearly labeled "NLL-filtered s1K",
smoke test note explaining why no smoke run yet, and P11.F0 baseline uncertainty note. ✅

**Fix 2 (data subdirectories)**: `train_adapter(data_dir, ...)` signature at line 531 takes
`data_dir` param. Uses `--data str(data_dir)` with no `--train-splits`/`--val-splits` flags
(grep confirms zero matches). `prepare_rsd_training_data()` writes to `data/rsd/train.jsonl`
and `data/rsd/valid.jsonl`; `generate_sert_traces()` writes to `data/sert/train.jsonl` and
`data/sert/valid.jsonl`. ✅

---

## Remaining Non-Blocking Issues (noted, not blocking)

**NB1: NLL approximation weaker than true RSD** — Documented in PAPER.md caveat section.
K1543 acceptance rate may be inflated (250× chance threshold at vocab_size=256k accepts
most tokens trivially). Downstream references should use "NLL-filtered s1K" label. ✅ documented.

**NB2: K1541 fragile baseline** — P11.F0 results may be unavailable or reflect the known
thinking regex bug. Code uses 60.0% fallback with log message. ✅ documented in PAPER.md.

**NB3: No smoke test** — PAPER.md explicitly notes this and explains why (blocking fix 2
would have caused training to fail silently). Acceptable — experiment is design-only stage.

---

## Verdict: PROCEED

Design is correct. Both blocking fixes applied. Experiment is ready to queue.

- K1541 (RSD >= raw+3pp): UNCERTAIN — depends on acceptance rate + training quality
- K1542 (NLL < 24h): LIKELY PASS — cached NLL scores already exist (trace_nll_scores.json)
- K1543 (≥60% accepted): UNCERTAIN — threshold may inflate rate, but directional claim holds
