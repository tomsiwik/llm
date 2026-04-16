# PAPER.md — P11.J0: Adapter Composition via Exclusive Routing

## Theorem Predictions vs Measurements

| Criterion | Theorem Prediction | Full Run Result | Status |
|-----------|-------------------|-----------------|--------|
| K1526: routed ≥ domain_only + 3pp | LIKELY (Theorem 1, perfect router) | TBD | TBD |
| K1527: routed_knowledge ≥ thinking_only_knowledge + 2pp | LIKELY (domain adapter helps knowledge) | TBD | TBD |
| K1528: router accuracy ≥ 85% | LIKELY (vocab separation, JL-lemma intuition) | TBD | TBD |

## Smoke Test Results

*(Smoke test pending — depends on exp_p11_thinking_adapter_universal completing first.
The thinking-openthoughts-universal-v0 adapter at adapters/ is a 10-step smoke artifact.
Full run requires the 1000-step trained adapter from task 17.)*

**Prerequisite check**: ADAPTER_THINKING at `adapters/thinking-openthoughts-universal-v0/`
must be populated by exp_p11_thinking_adapter_universal (pueue task 17, queued).

## Experimental Design

- **4 target categories**: math (reasoning), physics (reasoning), biology (knowledge), law (knowledge)
- **3 conditions**: thinking_only, domain_only (GT routing), embedding_routed (cosine sim)
- **Router calibration**: 10 seed examples × 14 categories → 2 centroids
- **Router evaluation**: 20 held-out examples per category × 14 = 280 examples (K1528)
- **Condition evaluation**: 20 questions per target category × 4 = 80 questions each condition

## Key Caveat: Domain Adapter Quality

**Finding #517 warning**: existing domain adapters (math-gsm8k, medical-medmcqa, legal-mmlu)
were trained with thinking=False on q_proj only (NTP method). Finding #517 showed these
DEGRADE MCQ performance by ~26pp for the math adapter.

If domain adapters actively hurt MCQ accuracy, then K1527 (routed_knowledge ≥ thinking + 2pp)
will FAIL because routing to domain adapter for knowledge questions is worse than staying on
the thinking adapter.

**Revised expectation**: The most likely "good" outcome is:
- K1528 PASSES (embedding router works)
- K1527 FAILS (domain adapters hurt, not help)
- K1526 ambiguous (routed condition avoids worst domain adapters, partially recovers)

This would motivate P11.L0 (RSD-aligned traces) to create better domain adapters trained
with thinking=True on MCQ-format data.

## Architecture

```
Query → tokenize → embed_tokens → mean pool → cosine sim → routing decision
                                    ↓
                            thinking centroid
                            knowledge centroid

Routing:
  reasoning → thinking-openthoughts-universal-v0 adapter
  knowledge → domain adapter (math/medical/legal) or thinking as fallback
```

## Budget Estimate

- Phase 1-2 (router build + accuracy): ~10 min (embedding ops only, no generation)
- Phase 3 (thinking_only, 80q): ~27 min  
- Phase 4 (domain_only, 80q, multiple adapters): ~27 min
- Phase 5 (embedding_routed, 80q): ~27 min
- **Total: ~91 min** (within 2h budget)
