Two parallel P4 micro experiments complete: exp_lora_flow_comparison (PROVEN — SOLE positioning confirmed, LoRA-Flow infeasible at N>100) and exp_layerwise_order_sensitivity (KILLED K2 — attn/FFN identical sensitivity scaling, ratio=1.01x).

## 2026-03-15 Iteration: Fix held-out eval HumanEval failure

### Situation
- GPU queue: 1 active (lte_no_reset) + 20 pending tasks
- All eligible macro experiments already submitted
- `run_all_eval_1773497259` (held-out eval, priority 2): MMLU passed (6624s), HumanEval FAILED (10161s)
- `train_8b/70b` scripts: FAILED (0s) — scripts missing from main (only in worktree)
- `train_reasoning_expert_1773512937`: FAILED (203s) — already resubmitted

### Action taken
1. Fixed `eval_humaneval.py`: generation timeout (120s SIGALRM), incremental checkpoints, resume
2. Fixed `run_all_eval.py`: HumanEval failure no longer aborts pipeline
3. Created `run_humaneval_only.py` standalone resubmission
4. Submitted `run_humaneval_only_1773533006` to GPU queue
5. Created task-1773533017-a1d9: copy teacher comparison scripts from worktree to main

### Next
- Fix teacher comparison scripts (copy from worktree)
- Wait for GPU queue results to drive next decisions
