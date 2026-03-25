# Scratchpad

## 2026-03-25: Wave 3 Micro Complete

research.start triggered for Wave 3 micro. Ran orphan check and node selection.

**Orphan check:** exp_bitnet_basefree_exploration has both REVIEW-adversarial.md and LEARNINGS.md. No orphans.

**Node selection:** All 9 Wave 3 micro experiments are resolved:
- Track 1: effective_delta_cosine (killed), kr_test_evaluation (supported), lori_sparse_b (killed)
- Track 2: scaffold_fresh_adapters (killed), galore_scaffold (supported), meta_scaffold (killed)
- Track 3: llamacpp_serving (supported), per_token_routing (supported)
- Track 4: retrain_evolve (supported)

Plus bonus experiments also complete: spectral_surgery (killed), eigenlorax_subspace (killed), semantic_compositionality (killed)

All remaining open/active nodes are scale: macro. Micro constraint prevents picking any.
Zero open micro nodes → soft-freeze allows generating 1 new hypothesis, but the rules require "you just completed an experiment" which isn't true this iteration.

**Decision:** Emit that Wave 3 micro is fully drained. All micro backlog exhausted. Next work requires macro scale (GPU).

## 2026-03-25: LOOP_COMPLETE — Wave 3 Micro Objective Satisfied

Received `research.complete` event confirming all 9/9 Wave 3 micro experiments resolved.
No open tasks. No orphan experiments. Scale constraint (micro only) prevents further work.

**Supported findings to carry forward to macro:**
1. KR-Test metric (replaces PPL for knowledge retention)
2. GaLore scaffold (base-free training approach)
3. llama.cpp serving (multi-adapter CPU inference)
4. Top-2 per-token routing (MoLoRA-style)
5. Retrain-evolve (retrain-from-scratch + KR-Test quality gate)

**Killed (not pursuing further):**
1. Effective delta cosine
2. LoRI sparse B
3. Scaffold fresh adapters
4. Meta scaffold (MAML)

Loop complete. Next phase requires GPU (macro scale).
