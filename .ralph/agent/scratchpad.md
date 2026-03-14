## 2026-03-14: Delegating exp_compressed_expert_sweep

Picked exp_compressed_expert_sweep (P3, micro, deps satisfied). Storage scaling is critical for SOLE at 5K+ experts.
LoRA=300GB vs LoRA-XS=215MB is 140x difference. Delegating to experiment-ideator.

## 2026-03-14: Submitted exp_clone_compete_evolution (P2 macro)

Highest-priority unqueued macro node with satisfied deps (pilot_50=supported, hash_ring=proven).
Wrote macro/clone_compete_evolution/run_clone_compete.py and submitted to GPU queue.

Script tests 5 domains (python, bash, math, medical, sql):
1. Generate corrections by finding high-loss examples in eval data
2. Clone each expert, fine-tune with 50 correction steps
3. Shadow score: answer-conditioned PPL on general + correction queries
4. Check kill criteria: K1 win rate >70%, K2 convergence <50K queries, K3 regression <2%

Already queued (8 tasks): held_out_eval (active), diagnose_adapters2, converged_adapter_ortho,
pilot50_composition_quality, held_out_eval (rerun), cat_weight_convergence, 3x teacher_comparison.
Clone-compete is task #9 in queue.

Next eligible macro nodes not yet queued: scale_500_experts (P4), automated_correction_pipeline (P4),
full_base_free_pipeline (P4), composable_merge_pipeline (P5).
