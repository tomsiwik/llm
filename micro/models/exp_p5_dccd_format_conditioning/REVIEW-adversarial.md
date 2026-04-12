# REVIEW — exp_p5_dccd_format_conditioning

## Verdict: PROCEED (KILLED experiment, findings valid)

## Summary

Correctly KILLED (2/3 fail). The core theorem (temporal separation prevents #483 cross-projection catastrophe) is conclusively verified. The failure analysis is honest: re-prompting is a weaker proxy for true DCCD grammar masking, and PAPER.md correctly identifies this implementation gap.

## Issues

### 1. Theorem 2 gap: zero interference != information preservation (non-blocking)

MATH.md Theorem 2 proves Interference(Phase 1, Phase 2) = 0 by temporal separation, then predicts K1268 "~0pp degradation." But zero interference is necessary, not sufficient, for information preservation. The draft contains full domain content (11.6 avg keywords), but re-prompting through the base model loses 38% of it (7.2 avg keywords). The theorem guarantees no CROSS-PROJECTION damage but says nothing about the re-prompting channel's fidelity.

PAPER.md acknowledges this correctly ("re-prompting artifact, not DCCD architectural failure") but the MATH.md prediction of ~0pp was wrong for the architecture actually tested. Future work should distinguish: (a) interference = 0 (proven), (b) channel fidelity depends on Phase 2 implementation (not proven).

### 2. Theorem 1 untested (non-blocking)

MATH.md's primary theoretical contribution (projection tax amortization via draft conditioning) was not tested because the implementation uses re-prompting instead of token-level grammar masking. The 80%+ SOAP prediction was derived from "grammar enforces structure" but no grammar was implemented. This is acknowledged in PAPER.md but should be explicit in findings: the DCCD *paper's* mechanism was not implemented, only a simplified version.

### 3. N=10 eval with high variance (non-blocking)

SOAP-only adapter achieves 60% (6/10), meaning a single sample flip changes the rate by 10pp. The 70% threshold is at the edge of statistical reliability at this sample size. The directional findings (DCCD >> weight-composed on all metrics) are robust, but exact percentages should be treated as approximate.

## What's solid

- Weight-composed baseline reproduces #483 catastrophically: pad tokens, Korean/Arabic garbage, 0.2 avg keywords. Strong negative control.
- DCCD dominates weight-composition on every metric. The comparison is unambiguous.
- 100% coherence vs 80% for weight-composed is the key result — temporal separation works.
- Root cause analysis and fix paths (grammar masking, SOAP adapter Phase 2) are well-reasoned.
- Finding #479 connection (RLHF suppresses SOAP format in base model) explains why re-prompting underperforms.

## Finding recommendation

Status: **supported** (not killed — the principle is verified, the implementation variant failed)

Core result: Temporal separation eliminates cross-projection catastrophe (#483). Re-prompting is insufficient for format compliance; token-level grammar masking or Phase 2 format adapter needed.
