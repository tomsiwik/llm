# Next Cycle Commands — Ready to Copy-Paste

## Design Principles

- **8hr max per loop** — every loop must finish meaningful work within 8 hours
- **No single experiment >2hrs** — break large tasks into phases
- **Quick wins first** — finish 6 small experiments rather than 40% of 1 big one
- **Long GPU jobs run as nohup background** — agent checks progress, doesn't babysit
- **Interleave on GPU** — start long job → do short experiments → check long job

---

## Pre-flight (run before each cycle)

```bash
ralph loops prune
ralph clean
```

---

## Cycle 1 — COMPLETE

### CPU (6 experiments, 4h)
| Experiment | Verdict |
|---|---|
| exp_content_aware_routing | KILLED (26.5% < 60%) |
| exp_premerge_vs_dynamic_quality | KILLED (vacuous at micro, methodology validated) |
| exp_answer_conditioned_scoring | PROVEN (r=0.811) |
| exp_delta_rank_scaling | REVISE (K1 killed, v2 with convergence fix done) |
| exp_oae_vs_lora_soups | PROVEN (5 structural advantages) |
| exp_collision_scaling | SUPPORTED (beta=-0.575, 1.23% at N=20) |

### GPU (1 experiment, 4.5h)
| Experiment | Verdict |
|---|---|
| exp_ffn_only_matched_rank | KILLED (PPL +66.7%, ortho 424% worse) |

**Key decisions locked:**
- All-modules adapters (not FFN-only) — attention regularizes FFN diversity
- Hash ring + pre-merge sufficient at small N (content-aware routing killed)
- Answer-conditioned PPL is the shadow scoring metric (full-seq PPL killed)

---

## Cycle 1.5 — IN PROGRESS

### State
- GPU: Distillation pilot 50 data gen DONE (50/50 domains, $6.49). Training running on RunPod via nohup (12/50 as of last check, ~16hr total). Background job — continues even if ralph dies.
- CPU: Cycle 2 micro experiments running.

---

## Cycle 2 — NEXT (when current loops die)

### CPU Instance (~8hrs, target 6 experiments)

```bash
ralph run -a -p "Work through HYPOTHESES.yml micro-scale CPU experiments. 8-HOUR BUDGET — prioritize completing experiments, not starting big ones. Each experiment should take 1-2hrs max.
CONTEXT: FFN-only KILLED (use all-modules). Answer-conditioned PPL PROVEN (r=0.811). Content-aware routing KILLED. Pre-merge validated. SOLE vs LoRA Soups resolved. Collision scaling supported (beta=-0.575).
RULE: If any experiment is taking >2hrs, wrap up what you have, write PAPER.md with partial results, and move to the next one.
Priority order (quick wins first):
1. exp_composition_vs_monolithic (P3, ~1.5hr) — 5 composed rank-8 experts vs 1 monolithic rank-40 LoRA. Same param budget. This answers 'why not just fine-tune?'
2. exp_attention_as_domain_similarity (P3, ~1hr) — Attention cosine predicts domain overlap. Quick: extract attention LoRA cosines from existing adapters, correlate with domain similarity.
3. exp_shadow_scoring (P3, ~1.5hr) — Implement shadow scoring using answer-conditioned PPL. Measure overhead (<5% target).
4. exp_cross_domain_composition (P3, ~1.5hr) — Test cross-domain queries. Does composition help or hurt?
5. exp_data_scaling_per_expert (P4, ~1.5hr) — Train at 50-5000 examples. Find the data scaling knee.
6. exp_adaptive_rank_selection (P4, ~1hr) — Rank sweep (8,16,32,64,128) across 5+ domains.
All numpy/CPU only. No MLX. Full cycle per node: research, implement, run, PAPER.md, review, integrate."
```

### GPU Instance (~8hrs, target 4 experiments + check background job)

```bash
ralph run -a -p "Interleaved macro experiments on RunPod. SSH: 'ssh runpod'. 8-HOUR BUDGET.
RULE: NEVER babysit a long-running job. Start it as nohup background, move to next experiment, check back later.
RULE: Each experiment should take <2hrs of active agent time. If longer, break into phases.
Step 1 — CHECK DISTILLATION STATUS (5 min):
  ssh runpod 'ls /workspace/llm/adapters/*/adapter_config.json 2>/dev/null | wc -l'
  ssh runpod 'grep -cE \"done in|FAILED\" /workspace/pilot50_train2.log'
  If 50/50 trained → run benchmark (scripts/pilot50_bench.py), write PAPER.md, mark proven/killed. (~2hrs)
  If still training → note progress, move on. It's running via nohup, don't touch it.
Step 2 — QUICK EXPERIMENTS while training continues (~2hrs each):
  a. exp_gpu_latency_validation (P4) — Measure fused kernel latency with whatever adapters are done. N=5,10,20 if partial. Quick script, quick result.
  b. exp_relora_composition_macro (P3) — Train 5 QLoRA adapters on Qwen2.5-7B (5 domains, 100 steps each = ~25min total). Measure pairwise cosine at d=3584. Compare to micro prediction.
Step 3 — CHECK DISTILLATION AGAIN:
  If done now → benchmark. If not → pull partial results (benchmark whatever's trained so far).
Step 4 — IF TIME REMAINS:
  c. exp_distillation_quality_vs_teacher (P3) — Pick 3 domains. Generate data with 8B and 70B teacher. Train 6 adapters (3 per teacher). Compare quality. ~2hrs.
Budget: \$50 total. Track \$/experiment."
```

---

## Gate Check: Cycle 2 → Cycle 3

```bash
# Verify distillation pilot completed
grep 'exp_distillation_pilot_50' HYPOTHESES.yml | head -1
# Verify composition vs monolithic baseline
grep 'exp_composition_vs_monolithic' HYPOTHESES.yml | head -1
# Check shadow scoring
grep 'exp_shadow_scoring' HYPOTHESES.yml | head -1
# Count progress
grep -cE 'status: (proven|killed|supported|revise)' HYPOTHESES.yml
```

| Decision | If YES | If NO |
|---|---|---|
| Distillation pilot 50 succeeds? | Scale to 500 (Cycle 3 GPU) | Debug pipeline, rerun |
| Composition beats monolithic? | SOLE validated as architecture | Fundamental problem — investigate |
| Shadow scoring works? | Evolution path clear (Cycle 4) | Need alternative quality metric |
| GPU latency validated? | Serving architecture locked | Investigate fused kernel issues |

---

## Cycle 3 — Compose + Serve (~8hrs each)

### CPU Instance

```bash
ralph run -a -p "Routing and serving micro experiments. 8-HOUR BUDGET, target 5+ experiments.
RULE: Each experiment <2hrs. Wrap up and move on if stuck.
Priority order (quick wins first):
1. exp_expert_removal_graceful (P4, ~1hr) — Remove expert from merged model, verify no regression.
2. exp_quality_degradation_detection (P4, ~1.5hr) — Detect when adding expert N degrades expert M.
3. exp_routing_at_scale (P3, ~1.5hr) — Test routing strategies at N=100, 500, 1000.
4. exp_inference_routing_strategies (P3, ~1.5hr) — Pareto frontier: hash vs classifier vs embedding.
5. exp_quantized_composition (P5, ~1hr) — Does AWQ/GPTQ preserve composition?
6. exp_subspace_capacity_empirical (P6, ~1hr) — Progressively merge experts (5,10,20,50,100), find capacity cliff.
Full cycle per node. numpy/CPU."
```

### GPU Instance

```bash
ralph run -a -p "Production validation on RunPod. SSH: 'ssh runpod'. 8-HOUR BUDGET.
RULE: No single task >2hrs. Interleave. Background long jobs.
PREREQUISITE: Check if distillation_pilot_50 is proven. If not, benchmark first.
1. exp_scale_500_experts (P4, ~3hrs total but interleaved) — ONLY IF pilot proven. Start training as nohup background (similar to pilot script). While training:
   a. Serve existing 50-expert model via vLLM. Measure throughput, latency p50/p95/p99, memory (~1.5hr).
   b. Check 500-expert training progress periodically.
2. exp_composition_vs_moe (P5, ~2hrs) — SOLE vs standard MoE. Same data, same params.
3. Return to 500-expert training: benchmark partial results or full results.
Budget: \$50."
```

---

## Cycle 4 — Evolve (~8hrs each)

### CPU Instance

```bash
ralph run -a -p "Evolution mechanism experiments. 8-HOUR BUDGET, target 4+ experiments.
RULE: Each experiment <2hrs.
1. exp_shadow_scoring (P3, ~1.5hr) — IF NOT DONE IN CYCLE 2. Implement using answer-conditioned PPL.
2. exp_correction_signal_quality (P4, ~1.5hr) — Compare 3 correction sources. Quick simulation.
3. exp_model_collapse_detection (P5, ~1.5hr) — Track diversity across 5 self-learning cycles.
4. exp_end_to_end_cost_accounting (P6, ~1hr) — Time every pipeline step for 5 experts.
5. exp_cat_weight_convergence (P3, ~1hr) — Do CAT-optimized weights converge to ~1.0?
Full cycle per node. numpy/CPU."
```

### GPU Instance

```bash
ralph run -a -p "Evolution on RunPod. SSH: 'ssh runpod'. 8-HOUR BUDGET.
RULE: No blocking. Background long jobs. Interleave.
PREREQUISITE: shadow_scoring must be proven. If not, run it first (<1hr on GPU).
1. exp_hash_ring_remove_expert (P5, ~1hr) — Remove one expert from N=8, measure redistribution.
2. exp_clone_compete_evolution (P2, ~3hrs interleaved) — Take 5 pilot experts, inject errors in 2, clone, fine-tune clones (50 steps, ~30sec each), shadow score 1K queries. Start scoring as background → do next experiment → check results.
3. exp_execution_based_self_learning (P3, ~2hrs) — Python expert generates solutions, execute against tests, train on passing. 5 cycles.
4. exp_evolution_convergence (P5, ~1.5hr) — 10 evolution cycles on 3 experts. Plot quality curve.
Budget: \$50."
```

---

## Cycle 5 — Base-Freedom (stretch, ~8hrs)

```bash
ralph run -a -p "Base-freedom experiments. 8-HOUR BUDGET, target 4+ experiments.
RULE: Each experiment <2hrs.
1. exp_relora_merge_cycle_scaling (P4, ~1.5hr) — K=5,25,50,100,200 merge cycles at micro. Quick sweep.
2. exp_fewshot_adaptation_recovery (P3, ~1hr) — 10-50 adaptation steps to close ZS transfer gap.
3. exp_amplification_factor_scaling (P4, ~1.5hr) — Error amplification at d=64, d=128, d=256.
4. exp_full_base_free_pipeline (P4, ~2hrs) — End-to-end: random skeleton → ReLoRA base → 10 experts → compose.
5. exp_lte_parallel_base_construction (P3, ~1.5hr) — LTE parallel LoRA as alternative to ReLoRA.
6. exp_procrustes_expert_transfer (P4, ~1hr) — Transfer experts across base model versions.
Mix of micro (numpy) and macro (RunPod). Budget: \$50."
```
