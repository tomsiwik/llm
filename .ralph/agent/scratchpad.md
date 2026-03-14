## 2026-03-15 Iteration: Generate new macro hypotheses

### Situation
- No open macro nodes with satisfied deps remain
- 25 tasks pending in GPU queue, 1 active (held-out eval running ~18h)
- 3 open macro nodes blocked: exp_evolution_convergence, exp_cross_domain_semantic_transfer, exp_expert_pruning_from_composition
- Recent failures: composition_quality (IndexError), several scripts missing from remote (rsync stale)
- Per objective: "If none remain, generate 2-3 new macro hypotheses"

### New Hypotheses Plan
All have deps satisfied by exp_distillation_pilot_50 (supported) only.

1. **exp_sole_vs_full_finetune_union** - THE fundamental value prop test. Compare composed 50-expert SOLE model vs a single LoRA trained on the union of all 50 training datasets. If SOLE wins, modularity has genuine quality advantages. If it loses, the value is purely in updatability/cost, not quality. Uses existing pilot50 adapters + cached base.

2. **exp_expert_continual_addition** - Train 10 NEW experts on domains not in pilot50 (e.g., Haskell, geology, philosophy). Add them to existing 50-expert composed model. Measure: (a) existing expert quality preservation, (b) new expert quality, (c) composition cost. Tests the "add without retraining" promise.

3. **exp_sole_inference_throughput** - End-to-end throughput measurement of composed SOLE model on RunPod A5000. Pre-merged mode (fastest), batch sizes 1/4/16/32. Compare tok/s and latency vs base Qwen2.5-7B. If pre-merge matches base (micro finding: -4% overhead), SOLE is deployment-ready.
