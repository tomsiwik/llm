# Current Direction

## Active Experiment
`exp_fix_grassmannian_loading_retest_routing` - Fix Grassmannian A-matrix loading bug and re-test routing at N=24.

## The Bug
All prior routing experiments at N=24 used `mlx_lm.LoRALinear` which initializes `lora_a` randomly. But the trained adapters were created with `TernaryLoRALinear` using per-domain Grassmannian A matrices from `grassmannian_skeleton_n24.npz`. The adapter.npz files only contain `lora_b` weights. LoRA output = scale * (x @ A) @ B. With random A, trained B produces noise.

## Evidence
- N=25 training experiment (correct A): 35.2% avg PPL improvement
- Centralized routing (random A): individual adapter PPL 10.12 vs base 10.06 = -0.6% (WORSE)
- Same adapters, same data, same base model

## Plan
1. Load model with correct TernaryLoRALinear + Grassmannian A matrices
2. Measure oracle PPL per domain (correct A + correct B)
3. Train centralized softmax router on hidden states
4. Measure routing accuracy and routed PPL

## Previous: exp_hierarchical_routing_n24 -- KILLED
Killed because adapters showed 0.04% PPL benefit. But this was caused by the A-matrix loading bug.
Finding #198 is likely invalid.
