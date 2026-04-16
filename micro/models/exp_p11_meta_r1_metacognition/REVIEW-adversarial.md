# REVIEW: exp_p11_meta_r1_metacognition (P11.D0)

**Reviewer**: Ralph (adversarial hat)
**Date**: 2026-04-14
**Verdict**: PROCEED (after REVISE fix applied)

---

## Round 2: REVISE Fix Verified

PAPER.md was the only blocking fix. It now contains:
- Prediction-vs-measurement table (Theorems 1-3 vs smoke) ✓
- Smoke findings (2/14 yield statistical noise, 1864.5 chars thinking, metacognitive prompting broken) ✓
- TBD rows for full run ✓
- K1502 risk analysis (format injection adds ~150 chars overhead, near-boundary at ~2000 chars) ✓
- K1504 overcount caveat (pattern too broad, recommend PLAN:/CHECK: tighter metrics) ✓

Design is correct. Full run queued as pueue task 16.

---

## Original Review (Round 1)

---

## Summary

MATH.md is mathematically coherent. Three theorems cover token reduction, forgetting prevention,
and accuracy under compression. Proof structure cites real papers (arXiv:2508.17291, arXiv:1612.00796).
Kill criteria are appropriate and quantitatively grounded.

---

## Blocking Issues

### 1. PAPER.md missing (BLOCKING)

PAPER.md is required by protocol. It must contain:
- Prediction-vs-measurement table (Theorems 1-3 predictions vs smoke test observations)
- Smoke test findings (including the critical yield issue below)
- TBD rows for full run
- K1502/K1503/K1504 status

**Fix**: Write PAPER.md with smoke test findings.

---

## Non-Blocking Issues

### 2. Smoke yield 14.3% (2/14) — statistically suspicious

Smoke test used N_SAMPLE=10 → 1 question per category (14 total). Expected yield ~57%
(from GRPO smoke test on same model). Getting 2/14 has P ≈ 3% under Binomial(14, 0.57).

Root cause: stratified 1/cat = 14 diverse hard questions. `avg_thinking_chars=1864` is normal
(thinking IS being extracted). Full run uses N_SAMPLE=200 → 14/cat → expected ~100+ correct.
Smoke abort at Phase 2 (`insufficient_training_data`) will NOT affect full run.

**Action**: Document in PAPER.md. No code change needed.

### 3. Format injection may *increase* thinking tokens (K1502 at risk)

Training data wraps correct traces with PLAN prefix + CHECK suffix:
`PLAN: I need to analyze... + {original ~1864 chars} + CHECK: Based on my analysis...`

Each training trace is ~150 chars LONGER than original. The model learns to generate
PLAN/CHECK wrappers, but not necessarily to stop early — it has never seen SHORT traces.
K1502 (30% reduction) may fail. This is already documented in MATH.md Failure Mode 1.

**Action**: Note expected K1502 risk in PAPER.md prediction table.

### 4. `has_metacognitive_structure` pattern too broad

Pattern `r'First,?\s+I'` matches natural language like "First, I need to consider..."
which Gemma 4 generates anyway. K1504 (≥50% structured) will trivially pass and
doesn't actually confirm PLAN/CHECK injection was learned.

**Action**: Note in PAPER.md that K1504 is a necessary but not sufficient condition.

---

## Code Quality

- REPO_ROOT = .parent.parent.parent ✓ (3 levels correct)
- Gemma 4 thinking regex `<\|channel>thought.*?<channel|>` ✓
- Memory cleanup with del model + gc.collect() + mx.clear_cache() ✓
- Stratified sampling consistent across phases (same SEED=42) ✓
- Phase 2 minimum training guard (n < 5 → abort) is appropriate

---

## Kill Criteria Prognosis

| Criterion | Prediction | Confidence |
|-----------|------------|------------|
| K1502: ≤2160 thinking chars | LIKELY FAIL — format injection adds overhead | Low |
| K1503: accuracy ≥ 62.1% | UNCERTAIN — depends on actual training yield | Medium |
| K1504: ≥50% structured | LIKELY PASS — but `has_metacognitive_structure` is too broad | High |

---

## Required Fix

**Apply exactly 1 fix**:

1. Write PAPER.md with: prediction table (Theorems 1-3), smoke test yield finding (2/14),
   K1502 risk note, K1504 overcount caveat. Include TBD rows for full run results.
