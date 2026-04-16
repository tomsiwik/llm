# REVIEW-adversarial.md: exp_bench_aime_2026

**Verdict: PROCEED** — all 3 blocking fixes from Round 1 applied

---

## Round 2 Verification

### Blocking Fix 1: Score parsing crash ✅ FIXED
`run_experiment.py:167` now has None guard:
```python
pct = f"{score['pass_at_n']:.1%}" if score['pass_at_n'] is not None else "N/A"
```
The fallback glob enhancement (suggested in Round 1) was not added separately, but
the None guard fully covers the crash scenario — `/dev/null` → `parse_score_from_status`
returns None → "N/A" displayed safely. No crash possible.

### Blocking Fix 2: K1417 expected outcome ✅ FIXED
`MATH.md:44` now correctly reads:
> "K1417 (within 10pp of 42.5%): EXPECTED PASS. ~37% is 5.5pp below 42.5%..."

`PAPER.md` prediction table also shows "EXPECTED PASS (~37%)". Consistent.

### Blocking Fix 3: PAPER.md missing ✅ FIXED
PAPER.md exists with prediction-vs-measurement table and design notes.
TBD rows will be filled when results.json arrives after pueue task 9 completes.

---

## Non-blocking Issues (carried forward, unresolved)

**Note A**: Theorem 3 timing inconsistency (n=2: 60s/prob vs n=4: 90s/prob). Accepted.

**Note B**: MathArena model config `model` field may not match mlx_lm.server advertised ID.
If experiment fails with 404/model-not-found, fix: set `model: mlx-community/gemma-4-e4b-it-4bit`.
This is acceptable as a known risk to watch for in results.json stderr.

---

## Finding Status

Experiment is queued (pueue task 9, will run after P9.G0 full stack integration). 
Results still TBD. Finding will be recorded as `provisional` (guided exploration,
results pending) once PAPER.md is filled with actuals.

Design is sound. Proceed to analyst (LEARNINGS.md context notes).
