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

## 2026-03-15 Iteration 2: Status update and housekeeping

### Situation
- All 3 new hypotheses (sole_vs_full_finetune, expert_continual_addition, sole_inference_throughput) already had scripts written and submitted to GPU queue in previous iteration
- They were still `open` in HYPOTHESES.yml — updated all 3 to `active` with evidence
- GPU queue: 1 active (run_all_eval, running ~20h, may be stuck), 28 pending
- Failed tasks: 3 scripts not found on remote (stale rsync) — resubmissions in queue should auto-sync
- run_composition_quality failed after 75s with rc=1 (actual bug, not file-not-found) — resubmission pending
- No new open macro nodes with satisfied deps remain after activating the 3 new ones
- All blocked nodes depend on active experiments that are in the GPU queue

### Action taken
- Updated HYPOTHESES.yml: 3 nodes open→active with evidence
- No new scripts to write — all eligible work is queued
- Next productive step: wait for GPU results, then integrate findings

## 2026-03-15 Iteration 3: Queue Fix & Partial MMLU Results

### Key Finding: Preliminary MMLU Results (Partial — 3/50 adapters only)
- run_all_eval completed MMLU phase but HumanEval got stuck (infinite loop in generated code, 17hrs)
- Only 5 adapters on RunPod (bash, math, medical, python, sql) — rest destroyed by rsync --delete
- MMLU results: 0/3 adapters beat base, avg delta: -3.67pp
- Kill criteria met: win_rate_below_80=True, avg_improvement_below_2=True
- **CAVEAT**: NOT a valid kill — only 3/50 adapters tested, and those may be corrupted.
  retrain_all_adapters must run first, then re-evaluate with proper adapters.

### Queue Fix Applied
- Killed stuck eval_humaneval process (PID 57617, 17hrs, 3% GPU)
- Removed 5 stale/duplicate entries from pending queue
- Reordered: retrain_all_adapters first, then training, then experiments, evals last
- Re-added missing train_reasoning_expert task (lost from queue)
- Synced all scripts to RunPod (rsync without --delete)
- Worker running run_lte_no_reset; after that → retrain_all_adapters → train_reasoning_expert
- Current queue: 20 tasks pending, properly ordered

### Next Iteration Should:
- Check lte_no_reset results → integrate into HYPOTHESES.yml
- Check if retrain_all_adapters completed
- Re-submit held-out eval with all 50 adapters after retrain completes
- Monitor queue progress, integrate completed results
