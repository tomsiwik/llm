# Pierre Research: Proof-of-Concept Experiment Queue

## Overview

Complete the remaining experiments to close all adversarial review gaps.
Each task is one experiment. Check `experiment list -s open,active` for the live queue.

## Task 1: Multi-Cycle Promotion (V1-4, CRITICAL)
Run `exp_m2p_multi_cycle_promotion`. 3-5 promotion cycles on toy model.
Kill: All domain qualities >= 80% after 3 cycles. No domain degrades >20%.
Resolves: Last open adversarial critique (V1-4).

## Task 2: Qwen3-4B M2P
Run `exp_m2p_qwen4b_gsm8k`. Full M2P pipeline on production model.
Kill: M2P quality >= 60% of SFT on GSM8K. Generation < 100ms.

## Task 3: Epsilon Map Measurements
Run these 3 quick experiments:
- `exp_condition_number_per_layer` — kappa(W) per layer
- `exp_q_wrong_real_domains` — Q_wrong for real domain pairs
- `exp_intrinsic_dim_real_tasks` — d_int of SFT adapter

## Task 4: Per-User Adapter PoC
Run `exp_m2p_per_user_poc`. 3 simulated personas.
Kill: Cohen's d > 0.3 behavioral differentiation.

## Task 5: Real-Text Routing + Composition
Run these in sequence:
- `exp_tfidf_routing_real_text` — TF-IDF on math/code/text
- `exp_m2p_2domain_compose_qwen06b` — 2-domain end-to-end composition

## Task 6: Product Metrics
Run these:
- `exp_m2p_generation_speed` — M2P latency measurement
- `exp_adapter_hotswap_latency` — hot-swap timing
- `exp_premerge_vs_runtime_qwen06b` — serving strategy comparison

## Task 7: Improvement Experiments
Run these game-dev inspired fixes:
- `exp_slerp_b_composition` — SLERP B-matrix composition
- `exp_pbd_interference_correction` — PBD runtime correction
- `exp_symplectic_promotion` — Symplectic promotion cycles

## Task 8: N=4200 Statistical Power Test
Properly powered M2P vs SFT comparison on GSM8K.
Need n=4200 per group for 80% power at 2.8pp effect size.

## Completion Criteria

All `experiment list -s open` returns empty. All adversarial review V2 blocking items resolved.
