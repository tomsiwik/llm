Integrated exp_lora_flow_comparison (supported) and exp_layerwise_order_sensitivity (killed). Both adversarial reviews PROCEED. Generated 2 new micro hypotheses: composition_weight_sensitivity and order_sensitivity_cosine_threshold. consecutive_kills reset to 0.

Closed task-1773533017-a1d9: copied teacher comparison scripts from worktree to main.

Starting exp_composition_weight_sensitivity (P4 micro): dilution test for pre-merged SOLE as N grows. Sweep N=2,5,10,20,50,100. Key question: does per-expert signal drop below noise at N<50? If so, selective top-k or weighted merge needed.

## 2026-03-15 Iteration: Fix SFTConfig API breakage

### Situation
- GPU worker: 1 active (lte_no_reset), 21 pending tasks
- Recent 5 completions ALL failed:
  - train_reasoning_expert: `SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`
  - train_8b_adapters, train_70b_missing: scripts not found on worker (sync issue)
  - run_all_eval: HumanEval eval failed (rc=241, ~4.7hrs)
  - run_composition_quality: script not found

### Action taken
- Fixed `max_seq_length` → `max_length` in SFTConfig across 7 files
- Committed (11a1c06) and resubmitted reasoning expert training (train_reasoning_expert_1773533500)
- rsync completed — all pending scripts now have correct code on worker

### Queue state
- Active: run_lte_no_reset (macro LTE experiment)
- 22 pending (21 original + 1 new reasoning expert resubmit)
- All active macro hypotheses have scripts queued — no new work to submit

### Next steps
- Monitor queue results as tasks complete
- Integrate results from completed experiments
- No new open macro nodes with satisfied deps remain unqueued

## 2026-03-15 Iteration: Generate new macro hypotheses

### Situation
- Worker RUNNING: `retrain_all_adapters` active (20GB VRAM), 21 pending
- No open macro nodes with satisfied deps → generating 3 new hypotheses
- Key recent findings informing new hypotheses:
  1. FFN-only: 25% fewer params, better orthogonality (micro proven, macro untested)
  2. Answer-only PPL: r=0.811 accuracy prediction (micro proven, macro testing queued)
  3. Cross-domain dilution: equal-weight averaging wastes signal (micro proven)

### New hypotheses added
1. exp_ffn_only_macro_composition (P3): FFN-only vs all-modules LoRA at Qwen2.5-7B
2. exp_ppl_guided_expert_refinement (P4): Answer-only PPL as automated quality signal for iterative improvement
3. exp_dynamic_weight_composition_macro (P3): Per-query expert weighting by cosine similarity to expert centroids

### Action
- Added to HYPOTHESES.yml
- Writing experiment script for exp_ffn_only_macro_composition (highest impact, simplest test)
- Submit to GPU queue
