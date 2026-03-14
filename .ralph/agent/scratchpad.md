2026-03-14: Parallel micro experiments complete. exp_inference_routing_strategies KILLED (K3: 41.5% quality capture < 90%), exp_lte_parallel_base_construction PROVEN (parallel=sequential substrates). 4th consecutive kill (routing) but 1 proven (LTE). Next: review both.

2026-03-14 (macro iteration): Integrated two completed GPU results:
- exp_cat_weight_convergence: PROVEN. CAT weights converge to ~1.0 with orthogonal experts. K1 SURVIVES (mean|w-1| within 0.1), K2 SURVIVES (PPL imp ≤5%). Validates SOLE unit-weight assumption.
- exp_attention_layer_orthogonality: PROVEN. K1 PASS (0.0% dissimilar pairs above bound), K2 PASS (max cos=0.0). Attention layers maintain structural orthogonality at macro d=3584.
GPU queue: 1 active (pilot50_held_out_eval), 12 pending. Two prior failures: run_composition_quality (IndexError: empty domains_with_data) and measure_orthogonality (instant fail) — both requeued.
Both proven results strengthen SOLE core theory: orthogonality holds across all module types, and CAT optimization is unnecessary.
