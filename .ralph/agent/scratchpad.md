2026-03-14: exp_minimax_grassmannian_packing reviewed (adversarial: PROCEED as kill), integrated as killed. consecutive_kills=2. B-matrix training dynamics identified as next target for composition safety.

2026-03-14: Picked exp_procrustes_expert_transfer (priority 4, micro, no deps). Tags: novel, cross-domain. Delegating to experiment-ideator. Previous two experiments killed (minimax_grassmannian, lte_rank_accumulation) — this one tests expert portability across base models via Procrustes alignment.

2026-03-14: Macro loop — exp_full_base_free_pipeline scripted and submitted.
Wrote run_full_base_free.py for exp_full_base_free_pipeline (P4, critical-path).
Design: GPT-2 124M, 12 domains, builds both conventional + ReLoRA bases (reuses cached from relora_from_scratch), trains 12 domain experts per base, composes via weighted LoRA sum, evaluates with hash-routing. Three kill criteria: quality ratio >1.10, speed ratio >2x, non-composable component.
Submitted as run_full_base_free_1773517079, position 16 in queue.
HYPOTHESES.yml updated: status=active, experiment_dir=macro/full_base_free_pipeline.

Queue state: 16 pending, 1 active (run_all_eval). Worker running, GPU at 39%.
Remaining unscripted macro nodes: exp_scale_500_experts (P4, needs 500 adapters — impractical until pilot pipeline proven), exp_composable_merge_pipeline (P5, blocked by collision_scaling deps).
