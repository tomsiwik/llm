# LEARNINGS — exp_p5_dccd_format_conditioning

**Status:** KILLED (2/3 fail) | Finding: supported

## Core Finding
Temporal separation (DCCD principle, arXiv:2603.03305) eliminates #483 cross-projection catastrophe: DCCD achieves 100% coherence vs 80% for weight-composition, and preserves partial domain (70%) and format (40%) capability that weight-composition destroys entirely (0%/0%).

## Why It Worked / Why It Failed
Temporal separation works because Phase 1 and Phase 2 adapters never interact in weight space — interference = 0 by construction. The failure was in the Phase 2 implementation: re-prompting relies on Gemma 4's base instruction-following, which is RLHF-suppressed for SOAP format (Finding #479). Re-prompting achieves only 40% SOAP vs the 70% target. The DCCD *paper's* actual mechanism (token-level grammar masking) was not implemented.

## Impossibility Structure (from KILL)
Re-prompting DCCD cannot exceed the base model's RLHF-suppressed SOAP capability ceiling. The same behavioral prior that makes q_proj insufficient for SOAP (Finding #479) limits re-prompting to ~40%. Fix: structural enforcement, not instruction-following.

## Implications for Next Experiment
Two proven fix paths:
1. **SOAP adapter in Phase 2** — Phase 1: domain adapter → draft. Phase 2: SOAP adapter (v_proj+o_proj, Finding #480) reformats. Still temporal separation (one adapter active at a time). Expected: 60%+ SOAP, <10pp domain degradation.
2. **Token-level grammar masking** — State-machine logit masking in MLX loop forces 100% SOAP by construction. Higher implementation cost, guaranteed result.

The SOAP adapter in Phase 2 is the fastest unblocked path: both adapters are already trained. Test whether the Phase 2 SOAP adapter can reformat Phase 1 draft without information loss.

## References
- arXiv:2603.03305 — Draft-Conditioned Constrained Decoding
- Finding #483 — Cross-projection catastrophe (reproduced as baseline)
- Finding #480 — v_proj+o_proj SOAP adapter (+70pp compliance)
- Finding #479 — RLHF suppresses SOAP in base model (explains re-prompting failure)
