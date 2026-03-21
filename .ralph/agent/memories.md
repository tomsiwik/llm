# Memories

## Patterns

## Decisions

### mem-1773535380-18dc
> MMLU held-out eval (run_all_eval_1773497259): kill criteria met, avg delta -3.67pp vs base. Adapters were pre-retrain (old config). Retrain with rank-16 all-modules in progress. Do not declare killed until retrained adapters are re-evaluated.
<!-- tags: evaluation, mmlu, distillation | created: 2026-03-15 -->

## Fixes

### mem-1773566058-0375
> run_full_base_free.py used GradScaler with bfloat16 causing AssertionError. Fix: remove GradScaler entirely, bf16 doesnt need loss scaling. All 4 training loops (conventional base, relora warmup, relora cycles, expert training) were affected.
<!-- tags: gpu, training, bf16 | created: 2026-03-15 -->

### mem-1773565188-7699
> PEFT add_weighted_adapter with combination_type='linear' applies weights as literal multipliers, NOT as mixture weights. For SOLE averaging (W + sum(delta_i)/N), use weights=[1/N]*N, not [1.0]*N. Bug was in 8 scripts causing PPL in the trillions at N=50.
<!-- tags: composition, peft, adapter-weighting | created: 2026-03-15 -->

### mem-1773534947-6b74
> LTE no-reset prepare_domain_data assumes OUTPUT_DIR exists but prepare_data may skip mkdir if cached. Always mkdir before writing.
<!-- tags: lte, gpu-queue, macro | created: 2026-03-15 -->

### mem-1773534690-5a33
> run_lte_no_reset failed with FileNotFoundError for results/lte_no_reset/domain_code_tokens.npy. Root cause: OUTPUT_DIR.mkdir not called before prepare_domain_data in old code. Fixed in current code at run_experiment() line 894. Always ensure output dirs exist before data prep functions.
<!-- tags: lte, gpu-queue, macro | created: 2026-03-15 -->

### mem-1773533520-c7b7
> SFTConfig on RunPod TRL uses max_length not max_seq_length. Changed in 7 files: composer/distill.py, composer/evolve.py, composer/rank_sweep.py, macro/automated_correction_pipeline/run_correction_pipeline.py, macro/clone_compete_evolution/run_clone_compete.py, micro/models/ffn_only_matched_rank/train_ffn_only.py, micro/models/reasoning_expert_distillation/train_reasoning_expert.py. Note: FastLanguageModel.from_pretrained still uses max_seq_length (Unsloth API, different from TRL SFTConfig).
<!-- tags: trl, training, runpod | created: 2026-03-15 -->

### mem-1773532570-7fb7
> HumanEval eval_humaneval.py can get stuck in infinite loops from generated code execution. Add timeout per problem (e.g., 30s) to prevent blocking the GPU queue for hours.
<!-- tags: evaluation, humaneval, gpu-queue | created: 2026-03-14 -->

### mem-1773515266-0447
> run_all_eval HumanEval hung: process shows 0% CPU, no log output for hours. Cause: likely HumanEval code execution sandbox timeout. Fix: kill process, resubmit. Check with: compare log mtime vs current time.
<!-- tags: gpu-queue, humaneval, stuck-process | created: 2026-03-14 -->

### mem-1773515264-9951
> GPU queue task ordering matters: retrain_all_adapters must run BEFORE adapter-dependent experiments. When reordering queue, group by dependency: (1) independent training, (2) evals needing trained models, (3) retrain/rebuild, (4) tasks needing retrained adapters. Queue file is at /workspace/gpu_queue/pending.jsonl on RunPod.
<!-- tags: gpu-queue, runpod, ordering | created: 2026-03-14 -->

### mem-1773514595-790c
> rsync --delete in gpu_queue.py destroys RunPod-only files (adapters, results). Fixed by excluding adapter_model.safetensors. Any large files trained on RunPod should be excluded from rsync --delete or they'll be deleted on next submit.
<!-- tags: gpu-queue, rsync, adapters | created: 2026-03-14 -->

## Context
