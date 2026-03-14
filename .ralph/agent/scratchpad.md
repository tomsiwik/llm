## 2026-03-15 — Scale 500 Experiment

### Analysis
- Reviewed HYPOTHESES.yml: only open macro node with satisfied deps is `exp_scale_500_experts` (P4, depends on `exp_distillation_pilot_50` which is supported)
- Other open macro nodes blocked: `exp_cross_domain_semantic_transfer` (needs active `exp_pilot50_composition_quality`), `exp_expert_pruning_from_composition` (same), `exp_evolution_convergence` (needs active `exp_clone_compete_evolution`)
- GPU queue has 1 active task (held-out eval) + 22 pending. Worker is running.

### Approach
- 3-phase pipeline: (1) data prep from SlimOrca dataset, (2) train 450 new adapters at 100 steps each, (3) evaluate composition at N=50,100,250,500
- Data: partitioned Open-Orca/SlimOrca into 450 domain buckets of 300 examples each (domains named domain_050 through domain_499)
- Training: 100 steps per adapter (vs 300 for pilot50) — sufficient for scale test, ~5 min/adapter
- Eval: hash ring displacement (mathematical), composition PPL degradation, inference latency
- Estimated cost: ~$13 for training + ~$1 for eval

### Submitted
- `prepare_scale_data_1773530168` — data prep from SlimOrca
- `train_scale_adapters_1773530181` — train 450 adapters
- `eval_scale_500_1773530195` — composition eval at N=50,100,250,500

### Status
Set `exp_scale_500_experts` to `active` in HYPOTHESES.yml.
