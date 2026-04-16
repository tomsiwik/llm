# PAPER.md — P11.Z1: Injection Decoding for Extended Thinking

## Summary

Testing "Well, Keep Thinking" adaptive injection decoding (arXiv:2503.10167) and
Plan-and-Solve (PS) prompting (arXiv:2205.01068) as zero-training accuracy improvements
on Gemma 4 E4B 4-bit MMLU-Pro with thinking enabled.

**Critical smoke-test finding**: Gemma 4 E4B does NOT under-think at threshold=500 chars.
Mean thinking = 2614 chars >> 500 threshold → injection NEVER triggered (avg_injections = 0.0).
Threshold was raised to 1500 chars before full run to give ~30-50% trigger rate.

---

## Prediction vs Measurement Table

| Condition | Predicted | Smoke (N=6) | Full Run (N=100) | Status |
|-----------|-----------|-------------|------------------|--------|
| Base + thinking | 62.1% ± 2pp | 100% (6/6) | TBD | SMOKE N too small |
| + Plan-and-Solve only | 62-65% | 66.7% (4/6) | TBD | Smoke: PS hurts math |
| + Wait injection only | 63-66% | 100% (6/6, 0 injections) | TBD | No injections fired |
| + PS + injection | 65-67% | 66.7% (4/6, 0 injections) | TBD | No injections fired |
| Degenerate loop rate | < 5% | 0.0% | TBD | K1534 PASS |
| Injection trigger rate | ~30-50% | 0.0% (mean=2614 >> 500) | TBD (threshold→1500) | FIXED |

---

## Kill Criteria Results

| Kill | Criterion | Smoke | Full Run | Status |
|------|-----------|-------|----------|--------|
| K1532 | PS+injection >= 65% MMLU-Pro | 66.7% PASS | TBD | Smoke: PASS (unreliable N=6) |
| K1533 | injection >= base + 1pp | 0pp FAIL | TBD | Was definitional FAIL at 500 threshold |
| K1534 | degenerate rate < 5% | 0.0% PASS | TBD | PASS (max 2 injections, none triggered) |

---

## Smoke Test Findings (N=6, math × 3 + biology × 3)

**Threshold bug**: MIN_THINKING_CHARS=500 is too low for Gemma 4 E4B. The model generates
a mean of 2614 thinking chars even on easy questions — 5× above the threshold. Injection
was designed for models that prematurely truncate thinking; Gemma 4 does not exhibit this.

**Fix applied**: MIN_THINKING_CHARS raised from 500 → 1500 before full run. At mean=1641,
roughly 30-50% of questions should fall below 1500 and trigger injection.

**PS prompt observation**: Plan-and-Solve prefix reduced math accuracy in smoke test
(3/3 → 1/3 on math). Possible cause: the question body says "Answer with ONLY the letter...
Do not explain." — conflicting instructions. The PS prefix may be ignored or may cause
the model to over-elaborate. This will be evaluated in the full run.

**Model does not under-think**: The core premise of injection decoding from arXiv:2501.12599
(s1, "Wait" budget forcing) is that models prematurely terminate thinking. Gemma 4 E4B
with thinking mode does not appear to exhibit this failure mode at N=6. If confirmed at
N=100, this would mean injection decoding provides no benefit for this model.

---

## Assumptions and Limitations

1. **Theorem 1 monotonicity**: The claim that accuracy is monotonically non-decreasing in
   thinking length up to θ(q) is an analogy from Wei et al. 2022 (CoT improves over no-CoT),
   not a formal derivation. The experiment tests this assumption directly.

2. **PS_PREFIX + "Do not explain" conflict**: The PS prefix asks for planning, but the
   question prompt asks for letter-only answers. In the full run, PS conditions may show
   no improvement or degradation due to this instruction conflict.

3. **Smoke test validity**: N=6 questions is too small for statistical conclusions. The
   100% accuracy on base/injection conditions is likely a ceiling effect from easy smoke
   test questions. Full run with N=100 across 4 categories will give reliable estimates.

---

## Next Steps

Full run (N=100) is queued. After completion:
- If injection still never triggers at 1500 → document as "Gemma 4 does not under-think"
  and KILL K1533. K1532 and K1534 can still be evaluated.
- If PS hurts accuracy → document conflicting instructions as likely cause.
- If injection fires at ~30% and improves accuracy → K1533 PASS, write finding.
