exp_removal_safety_complete_bound PROVEN. Combined bound predicts 0.106% deviation (K1: <1% PASS), empirical 0.098% matches within 0.93x (K2: within 2x PASS). Micro safety story complete.

## Iteration 51 (2026-03-15)

### Situation
- ACTIVE: run_inference_throughput_1773531219 (still running)
- 29 pending tasks in GPU queue (including retrain_all_adapters)
- No new results since iteration 50
- ALL open macro nodes are blocked (most by exp_pilot50_composition_quality → retrain_all_adapters)
- Worker is running, GPU at 10%

### Action: Generated 3 new macro hypotheses
No open macro nodes with satisfied deps → generated new hypotheses per objective.

1. **exp_composition_weight_normalization** (priority 3): Test 1/N, 1/sqrt(N), and grid-searched scaling factors at N=5,10,25,50. Attacks the PPL-in-trillions problem directly. Cheap (inference only, ~10 min).

2. **exp_cluster_grouped_composition** (priority 3): Leverage block-diagonal interference finding (within-cluster 7.84x higher). Route to cluster, compose within-cluster only (N_eff=5-10 vs N=50). Natural production architecture.

3. **exp_greedy_expert_selection** (priority 3): Greedily build optimal expert subset using validation PPL. Finds K* (optimal subset size), identifies net-harmful experts. O(N²/2) evaluations.

### Key insight
The core unsolved problem is: why does composition at N=50 produce catastrophic PPL? Three attack vectors:
- Weight scaling (normalization)
- Reducing effective N (clustering)
- Selecting good experts (greedy)
These are complementary — weight normalization + clustering + selection could all be combined.

### Next iteration should
- These 3 new nodes depend only on exp_distillation_pilot_50 (supported) → they are ELIGIBLE
- Write experiment scripts and submit to GPU queue
- Start with exp_composition_weight_normalization (simplest, cheapest, fastest)
- Check if run_inference_throughput has completed
