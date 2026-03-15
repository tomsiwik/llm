exp_composition_weight_sensitivity: Adversarial review v2 PROCEED. Integrated as supported (K2 only). Two non-blocking fixes noted (noise norm double-division, MATH.md/PAPER.md noise scaling contradiction).

## 2026-03-15 Iteration: Queue Audit & Wait State

### Queue Status
- **Active**: train_reasoning_expert_1773532315 (micro, running)
- **Pending**: 28 tasks
- **Worker**: RUNNING, GPU 32%, 11.9/24.6 GiB VRAM
- **Recent failures**: All 5 failures already diagnosed and fixes applied (timeout, max_seq_length, missing data dir)

### Script Audit Results
Audited all 22 queued scripts for known bug patterns:
- max_seq_length issue: **NOT present** in any queued script (all use max_length)
- mkdir issue: **NOT present** — all scripts use .mkdir(parents=True, exist_ok=True)
- Timeout issue: **NOT present** — no hardcoded timeouts in experiment scripts
- /workspace/ paths: Correct for RunPod environment

### State Assessment
- No new completed results to integrate (only failures already addressed)
- 5 open macro nodes — ALL blocked by active experiments (composition_quality, clone_compete, answer_conditioned_shadow, cat_weight_convergence, cross_domain_composition)
- 3 new hypotheses already generated last iteration (small_n, redundancy, interpolation) — all queued
- retrain_all_adapters resubmitted with 10800s timeout (in queue as retrain_all_adapters_1773536208)
- Queue is healthy and processing — waiting for results

### Key Pending Results to Watch
1. retrain_all_adapters_1773536208 → re-run MMLU eval after retrain
2. run_humaneval_only_1773533006 → HumanEval for pilot50
3. run_small_n_eval_1773535943 → diagnose if -3.67pp is N-dependent
4. run_individual_eval_1773536662 → individual expert held-out
5. run_selective_composition_1773536672 → topic-selective composition

### Decision
No actionable work remains. All scripts are queued and audited clean. Wait for GPU queue to process.

## 2026-03-15 Iteration: New Diagnostic Hypotheses for -3.67pp Regression

### Context
- Queue still processing: train_reasoning_expert active, 28 pending
- No new results completed since last iteration
- All open macro nodes still blocked on active experiments
- Previous iteration generated 3 hypotheses (small_n, redundancy, interpolation)
- Reviewed dependency graph: exp_cross_domain_semantic_transfer appeared eligible but
  actually depends on BOTH exp_cross_domain_composition (proven) AND exp_pilot50_composition_quality (active) → BLOCKED

### Analysis: Regression Diagnostic Gap
Existing queue addresses N-scaling, individual experts, and subject-matching.
Two diagnostic angles were missing:
1. **Weight interference vs distillation quality** — is the problem in HOW we compose (weight addition)
   or in WHAT we compose (adapter quality)?
2. **Embedding-based routing** — more realistic routing than manual subject-domain mapping

### New Hypotheses Generated (2)

1. **exp_logit_ensemble_vs_merge** (priority 2, macro)
   - Runs each adapter independently, averages logits (vs weight-space merge)
   - If ensemble >> merge → weight interference causes regression → fix composition
   - If ensemble ≈ merge → distillation quality causes regression → fix training
   - Submitted: run_logit_ensemble_1773538646

2. **exp_top_k_embedding_routing** (priority 3, macro)
   - Uses base model hidden-state embeddings for question-to-domain routing
   - Tests K=1,3,5,10 from N=50
   - Complementary to selective_composition (which uses manual subject mapping)
   - Submitted: run_top_k_routing_1773538657

### Queue Status After Submissions
- Active: train_reasoning_expert_1773532315
- Pending: 30 tasks (28 prior + 2 new)
- Key: logit ensemble result will determine whether to focus on improving
  distillation quality or composition strategy
