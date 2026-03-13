# SOLE Orchestration Plan — Path to Macro Production

**Created:** 2026-03-11
**Updated:** 2026-03-11
**Current readiness:** ~40% toward production SOLE
**Goal:** Finish each phase with macro evidence, close all adversarial gaps

---

## Cycle 1 — COMPLETE (CPU) / PARTIAL (GPU)

### Results

| Experiment | Instance | Verdict | Key Finding |
|---|---|---|---|
| exp_content_aware_routing | CPU | KILLED | 26.5% accuracy < 60% threshold |
| exp_premerge_vs_dynamic_quality | CPU | KILLED | Vacuous at micro (0% specialization) |
| exp_answer_conditioned_scoring | CPU | PROVEN | r=0.811 answer-only PPL |
| exp_delta_rank_scaling | CPU | REVISE | K1 killed, v2 convergence fix applied |
| exp_oae_vs_lora_soups | CPU | PROVEN | 5 structural advantages, complementary |
| exp_collision_scaling | CPU | SUPPORTED | beta=-0.575, 1.23% at N=20 |
| exp_ffn_only_matched_rank | GPU | KILLED | PPL +66.7%, ortho 424% worse |

### Decisions Locked

| Decision | Answer | Implication |
|---|---|---|
| FFN-only vs all-modules? | **All-modules** (FFN-only killed) | 25% more params but essential for quality + orthogonality |
| Pre-merge vs dynamic routing? | **Pre-merge** at small N | Zero routing overhead, needs macro validation at large N |
| PPL metric for shadow scoring? | **Answer-conditioned PPL** (r=0.811) | Full-sequence PPL killed (r=0.08) |
| Content-aware routing needed? | **No** at micro scale | Hash ring sufficient; cluster-level trivial (96%) |

### Still Open from Cycle 1

| Decision | Status |
|---|---|
| Delta rank scaling at d=128,256? | Revised — K1 killed, r_95 metric promising but needs more data |
| Distillation pilot 50? | NOT STARTED — GPU loop ran out of time on FFN-only |
| GPU latency validation? | NOT STARTED |
| ReLoRA composition macro? | NOT STARTED |

---

## Cycle 1.5 — RUNNING NOW

| Instance | Experiments | Expected Output |
|---|---|---|
| CPU (micro) | composition_vs_monolithic, cross_domain_composition, attention_as_domain_similarity, shadow_scoring, data_scaling, adaptive_rank, hierarchical_composition | Composition baseline, shadow scoring impl, data/rank curves |
| GPU (macro) | distillation_pilot_50, gpu_latency_validation, relora_composition_macro, distillation_quality_vs_teacher | 50-expert pipeline built, GPU latency proven, teacher comparison |

### Gate: Cycle 1.5 → Cycle 3

| Decision | If YES | If NO |
|---|---|---|
| Distillation pilot 50 succeeds? | Scale to 500 → Cycle 3 GPU | Debug pipeline, rerun |
| Composition beats monolithic? | SOLE validated as architecture | Fundamental problem — investigate |
| Shadow scoring overhead <5%? | Evolution path clear | Optimize or find alternative |
| GPU latency N-independent? | Serving architecture locked | Investigate fused kernel issues |

---

## Cycle 2: Close Phase 1 (Distill + Compose)

**Trigger:** Cycle 1 GPU instance completes distillation_pilot_50
**Goal:** A working 50-expert SOLE model benchmarked against base

### CPU Instance 2 Prompt

```
Work through HYPOTHESES.yml. Focus on closing Phase 1 gaps from the adversarial review.
Priority order:

1. exp_composition_vs_monolithic (P3) — CRITICAL BASELINE. Compare 5 composed
   rank-8 experts vs 1 monolithic rank-40 LoRA on same data. Same parameter budget.
   This is the "why not just fine-tune?" question.

2. exp_cross_domain_composition (P3) — Test "Python to Bash" style queries on
   merged model. Does composition help or hurt cross-domain queries?

3. exp_data_scaling_per_expert (P4) — How many examples per expert? Train python
   expert at 50, 100, 200, 500, 1000, 2000, 5000 examples. Find the knee.

4. exp_synthetic_vs_real_data (P4) — Synthetic (Groq) vs real (HuggingFace) vs
   mixed training data quality comparison.

5. exp_adaptive_rank_selection (P4) — Does domain complexity predict optimal rank?
   Rank sweep across 5+ domains.

6. exp_attention_as_domain_similarity (P3) — Can attention cosine predict domain
   overlap for routing decisions?

All CPU/numpy. No MLX. Run full cycle per node.
```

### GPU Instance 2 Prompt

```
Continue GPU experiments on RunPod. SSH configured: 'ssh runpod' works.
Priority order:

1. exp_scale_500_experts (P4) — IF distillation_pilot_50 succeeded, scale to 500.
   Measure hash ring displacement, vLLM serving, per-domain retention.

2. exp_distillation_quality_vs_teacher (P3) — Compare 8B vs 70B teacher quality.
   Critical for cost optimization (10x difference per expert).

3. exp_composable_merge_pipeline (P5) — Build composer/merge.py: the production
   merge pipeline with quality gates. Orthogonalize → merge → benchmark → accept/reject.

4. exp_runtime_expert_loading (P6) — Hot-swap experts via vLLM runtime LoRA.
   Measure swap latency. Target <100ms.

Track $/experiment. Budget: $50 total for this cycle.
```

---

## Cycle 3: Close Phase 2 (Routing + Serving)

**Trigger:** 50-expert model serving on vLLM, routing strategy decided from Cycle 1
**Goal:** Production-ready serving with correct routing

### CPU Instance 3

```
Routing and serving experiments:

1. exp_routing_at_scale (P3) — Test routing at N=100, 500, 1000 with FAISS index.
2. exp_inference_routing_strategies (P3) — Pareto frontier: routing accuracy vs latency.
3. exp_quality_degradation_detection (P4) — How to detect when expert N degrades expert M.
4. exp_expert_removal_graceful (P4) — Removing expert doesn't break others.
5. exp_quantized_composition (P5) — Does AWQ/GPTQ survive composition?
6. exp_hierarchical_composition (P5) — Two-level expert hierarchy for related domains.
```

### GPU Instance 3

```
Production validation on RunPod:

1. exp_gpu_latency_validation (P4) — IF not done in Cycle 1, do now with fused kernels.
2. Production E2E benchmark: serve 50-expert model on 4090, measure throughput,
   latency p50/p95/p99, memory usage.
3. exp_composition_vs_moe (P5) — Compare SOLE vs trained MoE on same data/compute.
```

---

## Cycle 4: Close Phase 3 (Evolution)

**Trigger:** Serving works, routing works, answer-conditioned PPL validated
**Goal:** Working clone-and-compete loop with measured convergence

### CPU Instance 4

```
Evolution mechanism experiments:

1. exp_shadow_scoring (P3) — Shadow overhead <5% latency, correlates with quality.
2. exp_correction_signal_quality (P4) — Compare human vs teacher vs execution feedback.
3. exp_model_collapse_detection (P5) — Self-learning doesn't cause output collapse.
4. exp_end_to_end_cost_accounting (P6) — True $/expert including all overhead.
```

### GPU Instance 4

```
Evolution on RunPod with real experts:

1. exp_clone_compete_evolution (P2) — THE KEY INNOVATION. Take 5 experts from pilot,
   inject errors, clone, fine-tune, tournament. Requires hash_ring_remove_expert.
2. exp_hash_ring_remove_expert (P5) — Expert removal for pruning losers.
3. exp_evolution_convergence (P5) — 10 evolution cycles, monotonic improvement.
4. exp_execution_based_self_learning (P3) — Code experts self-improve via test execution.
```

---

## Cycle 5: Base-Freedom (Optional/Parallel)

**Trigger:** delta_rank_scaling shows favorable scaling
**Goal:** Prove the entire model can be adapter-only

```
1. exp_relora_merge_cycle_scaling (P4) — K=50-200 merge cycles.
2. exp_full_base_free_pipeline (P4) — End-to-end: random init → ReLoRA base → 50 experts → serve.
3. exp_lte_parallel_base_construction (P3) — LTE as alternative to ReLoRA.
4. exp_fewshot_adaptation_recovery (P3) — Close zero-shot gap with 10-50 steps.
5. exp_amplification_factor_scaling (P4) — Does error amplification decrease with d?
6. exp_procrustes_expert_transfer (P4) — Transfer experts across base versions.
```

---

## Adversarial Review Concerns → Resolution Map

| Attack | Resolution Experiment(s) | Cycle |
|--------|------------------------|-------|
| "Everything is micro scale" | ffn_only_matched_rank, distillation_pilot_50, relora_composition_macro, scale_500 | 1, 2 |
| "Pre-merge dilution at large N" | premerge_vs_dynamic_quality, scale_500 | 1, 2 |
| "No router — hash ring is content-agnostic" | content_aware_routing, routing_at_scale, inference_routing_strategies | 1, 3 |
| "LoRA Soups already did this" | oae_vs_lora_soups, composition_vs_moe | 1, 3 |
| "Evolution has zero evidence" | answer_conditioned_scoring → shadow_scoring → clone_compete → evolution_convergence | 1, 4 |
| "Base-freedom rank scaling unknown" | delta_rank_scaling → relora_from_scratch → full_base_free_pipeline | 1, 5 |
| "PPL metric is broken" | answer_conditioned_scoring | 1 |
| "No comparison to alternatives" | composition_vs_monolithic, composition_vs_moe | 2, 3 |
| "Training data quality unknown" | synthetic_vs_real_data, data_scaling_per_expert | 2 |
| "Economics are projected, not measured" | end_to_end_cost_accounting, distillation_pilot_50 | 2, 4 |

---

## Phase Completion Criteria

### Phase 1: DISTILL (Target: end of Cycle 2)
- [ ] 50 experts trained and composed on Qwen2.5-7B
- [ ] Composed model beats base on >80% of expert domains
- [ ] FFN-only confirmed as default (or reverted to all-modules with evidence)
- [ ] Pipeline cost verified at <$0.50/expert
- [ ] Teacher size comparison done (8B vs 70B)
- [ ] Data scaling curve mapped
- [ ] Composition vs monolithic comparison done

### Phase 2: COMPOSE (Target: end of Cycle 3)
- [ ] Content-aware routing working at >80% accuracy OR pre-merge validated at N=50
- [ ] vLLM serving with <5% overhead on GPU
- [ ] Expert add/remove without quality regression
- [ ] Routing latency <10ms at N=500
- [ ] Quality degradation detection working
- [ ] Quantization compatibility tested

### Phase 3: EVOLVE (Target: end of Cycle 4)
- [ ] Answer-conditioned PPL validated as quality signal
- [ ] Shadow scoring working with <5% overhead
- [ ] Clone-and-compete resolves correctly >90% of time
- [ ] 10 evolution cycles show monotonic improvement
- [ ] No model collapse after 5 self-learning cycles
- [ ] True $/expert accounting done

### Phase 4: BASE-FREEDOM (Target: Cycle 5, stretch goal)
- [ ] Delta rank ratio measured at d=128, d=256, d=896
- [ ] ReLoRA base built from scratch at macro scale
- [ ] Full base-free pipeline demonstrated
- [ ] Zero-shot transfer gap <1% at rank-32

---

## Readiness Projection

| After Cycle | Readiness | Key Unlock |
|-------------|-----------|------------|
| Cycle 1 (now) | ~30% → **50%** | Architecture decisions locked, pilot built |
| Cycle 2 | **50% → 70%** | 50+ expert model serving, baselines compared |
| Cycle 3 | **70% → 85%** | Production routing and serving validated |
| Cycle 4 | **85% → 95%** | Evolution loop working |
| Cycle 5 | **95% → 100%** | Base-freedom proven (or cleanly killed) |

---

## Budget Tracking

| Cycle | GPU Budget | Expected Spend |
|-------|-----------|---------------|
| 1 | $50 | ~$15 (ffn $0.20, pilot $13, relora ~$1, latency ~$0.50) |
| 2 | $50 | ~$35 (scale 500 ~$25, teacher comparison ~$5, merge pipeline ~$5) |
| 3 | $50 | ~$15 (GPU validation ~$5, E2E benchmark ~$5, MoE comparison ~$5) |
| 4 | $50 | ~$20 (clone-compete ~$10, evolution cycles ~$10) |
| 5 | $50 | ~$10 (ReLoRA from scratch ~$5, transfers ~$5) |
| **Total** | **$250** | **~$95** |
