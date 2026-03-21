# SOLE Project Adversarial Review

**Date:** 2026-03-19
**Reviewer scope:** Full project audit -- architecture, findings, gaps, production readiness
**Documents reviewed:** VISION.md, FINDINGS.md, HYPOTHESES.yml (all nodes), GPU_CODING_GUIDELINES.md, 15+ experiment PAPER.md files, ARCHIVE.yml

---

## 1. What Is Actually Proven vs Claimed

### The strongest claims (genuinely validated)

**Orthogonality is structural at high d.** The finding that LoRA adapter cosine similarity scales as 1/sqrt(D_flat) is mathematically sound and empirically validated across d=64 to d=1024 at micro and d=896 at macro. The random baseline comparison (LoRA/random ratio 0.93-1.13) is honest and important: orthogonality comes from dimensionality, not from any clever property of LoRA training. A skeptical reviewer would accept this, but would note it means the "structural" claim is really "high-dimensional random vectors are near-orthogonal," which is a well-known property, not a discovery.

**Distillation pipeline produces functional adapters.** HumanEval +9.1pp (19.5% vs 10.4%) for the python adapter is a real, held-out, execution-based result. This is the single strongest piece of evidence that distillation transfers real skill. However, 3/3 tested adapters DEGRADED MMLU performance (-3.71pp average), which means the adapters are format-specialized, not knowledge-additive. The 98% win rate on contaminated training data is essentially meaningless as a quality signal.

**Routing latency is not a bottleneck.** All 6 strategies under 21us at N=1000. This is clean and would survive review.

**Hash ring expert add/remove works.** Mean -2.23% degradation, max -4.53%. Clean micro result.

### Claims that are weaker than presented

**"cos=0.0002 at d=896" -- the headline number is misleading.** This number comes from 5 adapters at macro scale (minimum_viable_base + structural_orthogonality_proof). When converged_adapter_orthogonality tested TRAINED adapters at d=896, the result was KILLED: mean cos=0.142 (35.6x the micro prediction). The math-medical pair hit cos=0.703. The VISION.md headline still says "cos=0.0002" without qualifying that trained, converged adapters on semantically related domains violate this by 3500x. This is the most misleading claim in the project.

**"MoE beats joint training -0.70%"** This is from a macro experiment at d=256 on 4 domains with 3 seeds using the 0.5B model. The -0.70% is on training-distribution evaluation. There is no held-out evaluation of the composition versus joint training. At d=256 with 4 domains, orthogonality is nearly guaranteed by dimensionality alone. This result does not transfer to 50 domains on a 7B model where math-medical-style overlaps exist.

**"Composition is plug-and-play"** -- directly contradicted by macro evidence. The composition_dropout_robustness experiment found CV=112.2% and PPL in the TRILLIONS at N=5 equal-weight composition. The composition_quality experiment found +127% mean degradation. A single SQL adapter reduced PPL from 31.6T to 17,683 when removed (99.99% improvement). This is not "plug-and-play." It is catastrophically fragile.

**"Evolution through competition"** -- zero macro evidence. Clone-and-compete has never been tested. Answer-conditioned PPL (the Evolve scoring metric) was KILLED at macro (r=-0.63, AUC=0.0, 100% disagreement with teacher ranking). The micro result (r=0.811) did not transfer. The Evolve phase currently has NO validated scoring mechanism.

**"Orthogonality is free"** -- only for dissimilar domains. VISION.md presents this as universal. The converged adapter experiment showed it holds for programming-vs-STEM but breaks for math-vs-medical (cos=0.703). The N_max formula (d^2/r^2 = 609K at 7B) is an upper bound for maximally diverse domains. The effective capacity depends on domain diversity, and the project has no validated method to predict which domain pairs will overlap.

### The "proven" micro results that a skeptical reviewer would challenge

Most micro results (40+ experiments at d=64, r=8, character-level data, 4-layer GPT) demonstrate mechanisms in principle. That is their intended scope. However, several micro "proven" results failed at macro:

| Micro claim | Micro result | Macro reality |
|---|---|---|
| Answer-conditioned PPL works | r=0.811 | KILLED at macro (r=-0.63) |
| Content-aware routing | killed at micro (as expected) | N/A -- never needed |
| Composition weight sensitivity | zero degradation N=2..100 | +127% degradation at N=5 macro |
| Converged adapter orthogonality | cos << prediction | KILLED (35.6x worse) |
| Composition robust to dropout | N/A (never micro-tested this way) | KILLED (CV=112%) |

The micro-to-macro transfer rate for composition-critical claims is poor. The micro experiments that work cleanly are the geometric/structural ones (orthogonality scaling, Grassmannian packing, routing latency, expert removal). The micro experiments that fail at macro are the ones involving actual quality measurement under composition.

---

## 2. Critical Gaps

### Gap 1: No validated composition strategy at production scale

Equal-weight composition produces PPL in the trillions. The PPL-probe weighting (r=0.990 oracle correlation) was validated only at micro (d=64, r=8, zero expert specialization). It has never been tested at macro with real adapters. The project claims this is the known fix but has not executed it. Until PPL-probe weighting is validated at macro, there is no working composition strategy for N>1.

### Gap 2: No validated Evolve scoring metric

Answer-conditioned PPL was KILLED at macro. The project now has no cheap quality signal for clone-and-compete tournaments. The alternatives are:
- Teacher model judging ($0.001/query) -- works but defeats the "autonomous evolution" narrative
- Execution-based feedback -- works for code only
- Human feedback -- expensive, not scalable
This is a fundamental gap. Without a scoring metric, the Evolve phase cannot function autonomously.

### Gap 3: Orthogonality for trained adapters is not what was predicted

The structural orthogonality proof (micro) used models that never converged (loss at random baseline). When converged adapters were measured at d=896, the result was 35.6x worse than predicted, with one pair at cos=0.703. The project acknowledges this but has not updated the N_max calculations or VISION.md scaling tables accordingly. At effective cos=0.14 (the macro mean including the outlier), the interference is not negligible.

### Gap 4: MMLU regression is unresolved

3/3 tested adapters degraded MMLU performance. The project attributes this to "format mismatch" (instruction-following vs MCQ), which is plausible but unproven. Only 3 of 50 adapters were evaluated. The retrained adapters have never been re-evaluated on MMLU. This is the most operationally important gap: if adapters degrade general capability, the composed model is worse than base on non-specialized tasks.

### Gap 5: N>5 composition never tested at macro

All macro composition experiments use N=5 (or fewer, since only 5 adapters were available on GPU instances). The project extrapolates from N=5 to N=50, N=500, N=100,000. There is zero empirical evidence for composition quality at N>5 with real trained adapters. The micro evidence (N=100 at d=64 with zero specialization) is vacuous for this purpose.

### Gap 6: Grassmannian skeleton never used in a real training run

The AP skeleton is a beautiful theoretical construction. It has been validated geometrically (coherence bounds, capacity, mixed-rank packing). But no adapter has ever been trained with frozen-A initialized from the skeleton at macro scale. The drift=0.02% result is from micro (d=64). Whether frozen-A training produces competitive adapters at d=4096 is untested.

### Gap 7: No cross-domain composition test at macro

The micro cross_domain_composition experiment tested sequential task evaluation (do task A, then task B). Real cross-domain queries ("explain this algorithm using medical terminology") require simultaneous domain understanding. This has never been tested, and is specifically flagged as an open question.

---

## 3. The Composition Problem

This is the project's existential crisis. The evidence is stark:

**What works:** Individual adapters work well. Python adapter gets +9.1pp on HumanEval. Individual adapters are roughly neutral on MMLU (-0.95pp average). The base model is not destroyed by a single adapter.

**What breaks catastrophically:** Equal-weight composition of N=5 adapters produces PPL in the trillions (pre-retrain) or +127% degradation (post-retrain). One SQL adapter poisons everything. Dropout of any adapter causes 112% CV in PPL.

### Why does this happen despite cos=0.0002 orthogonality?

The answer is that cosine similarity between flattened LoRA delta vectors is the WRONG metric for composition quality. Here is why:

1. **Cosine measures direction, not magnitude.** Two LoRA adapters can point in perfectly orthogonal directions but have wildly different magnitudes. When combined at equal weight, the larger-magnitude adapter dominates, and the effective output is pulled far from any individual adapter's optimal point.

2. **Orthogonality is in weight space, not function space.** cos(delta_W_i, delta_W_j) = 0 does NOT mean the adapters produce independent outputs. Through the base model's nonlinearities, orthogonal weight perturbations can produce correlated or adversarial output perturbations. The L2 norm stability experiment (0/25 catastrophic failures) used much weaker adapters -- the ones from the pilot are 10-100x stronger deltas.

3. **The SQL adapter problem.** Dropping SQL reduced PPL from 31.6T to 17,683. This is a 1.8-billion-fold improvement. The SQL adapter was the worst-performing in the pilot (-21.4% PPL regression on its own domain). A single poorly-trained adapter, when added at 1/5 weight, destroyed the entire composition. Orthogonality does not protect against this because the SQL adapter's damage operates through logit-scale effects, not subspace interference.

4. **Logit-scale mismatch.** This was identified in the original ADVERSARIAL_REVIEW.md (now missing from the repo). Different adapters push logits to different scales. When combined, the adapter with the largest logit perturbation dominates, regardless of orthogonality. At d=896 with rank-16, the adapters are modifying a tiny fraction of the weight space, but through the transformer's residual stream and layer normalization, these perturbations get amplified nonlinearly.

### Is selective routing sufficient to fix this?

PPL-probe weighting (r=0.990 with oracle at micro) is the proposed fix. It requires K+1 forward passes on 10 probe examples per query to determine adapter weights. This has three problems:

1. **Never validated at macro.** The r=0.990 correlation is at d=64 with adapters that have zero specialization. At macro, where adapters have real but conflicting specializations, the probe signal may be too noisy.

2. **Latency cost.** K+1 forward passes per query is K+1 times the inference cost. At K=5, this is 6x inference cost. The project claims "near-free composition" but PPL-probe routing makes it 6x more expensive.

3. **The oracle is unknowable at runtime.** The r=0.990 correlation is with the oracle (which knows the best weight per adapter). At runtime, you do not have the oracle. The 10 probe examples serve as an approximation, and the quality of that approximation at macro scale is unknown.

### Is the architecture fundamentally flawed?

Not necessarily, but the current formulation (W + sum(delta_i)/N) is demonstrably broken at macro. The project needs to acknowledge that equal-weight pre-merge is NOT viable and stop claiming "plug-and-play composition." The viable path is per-query routing (top-k selection, not weight averaging), which is a fundamentally different architecture from what VISION.md describes.

---

## 4. Competitive Analysis

### Can SOLE compete with frontier models?

VISION.md contains a table claiming a 7B base + 127K experts would match Llama 4 Maverick (400B), 214K experts would match DeepSeek-V3 (671B), and 578K experts would match GPT-4 scale (1.8T). These claims are not supported by any evidence and are, frankly, not credible.

**Why the parameter-counting argument fails:**
- A LoRA adapter at rank-16 adds 3.1M parameters per expert. 127K experts = 395B parameters of stored LoRA deltas. But these parameters are NOT equivalent to 395B dense parameters. A rank-16 perturbation to a 7B model captures a tiny fraction of what a 400B dense model can represent.
- The 7B base model is the bottleneck. No amount of LoRA adapters can give a 7B model the reasoning depth of a 400B model. The base model's attention patterns, vocabulary handling, and representational capacity are fundamentally limited by d=4096 and 32 layers.
- DeepSeek-V3's MoE architecture activates 37B of 671B parameters per token, with jointly-trained expert routing. SOLE activates 7.006B per token (7B base + k=2 adapters at 3.1M each). The quality gap is enormous.

**What SOLE can realistically compete with:**
- Other 7B models on domain-specific tasks (Qwen-7B, Llama-3-8B, Mistral-7B). Here, the adapter library provides breadth.
- Fine-tuned 7B models where the fine-tuning data is limited or domain-specific. SOLE's modular update story is genuine.
- Use cases where switching cost matters (retraining one adapter at $0.25 vs retraining the whole model at $1000+).

**What SOLE cannot compete with:**
- GPT-4/Claude/DeepSeek-V3 on general reasoning, instruction following, or complex multi-step tasks. The 7B base is the ceiling.
- Specialized models trained on curated data at scale (DeepSeek-Coder-33B, Qwen-Coder-72B). Domain depth beats domain breadth.
- Any task requiring >7B-level attention capacity (long-context reasoning, multi-document synthesis).

The honest competitive positioning is: "SOLE provides modular, updatable expertise on a 7B base at very low cost. It trades frontier quality for operational flexibility and economic efficiency." The scaling tables in VISION.md should be removed or heavily qualified.

---

## 5. Base-Freedom Assessment

The "no sacred weights" vision has partial validation:

**What works:**
- ReLoRA from-scratch produces a base that supports composition (cos_ratio=0.875x, loss_ratio=1.122 at GPT-2-124M scale). The base quality gap (12.1%) is expected to close at larger scale per Lialin et al.
- Base model is decomposable into adapter format (rank-16 SVD: loss ratio 1.014 at d=64).
- Zero-shot base transfer works at rank-32 (0.3% loss).
- LTE parallel matches sequential (engineering equivalence for multi-GPU).

**What does not work:**
- ReLoRA from-scratch has never been tested at 7B scale. The GPT-2-124M result has a 12.1% base quality gap that is borderline (1% below the 1.20 kill threshold).
- No adapter has been trained on a ReLoRA-constructed base at production scale.
- The "base as adapter" concept (skeleton + base_adapter + N experts) has only been demonstrated at d=64 with toy data.
- Full base-free pipeline (exp_full_base_free_pipeline) has been submitted to GPU queue but no results yet.

**Assessment:** Base-freedom is a plausible stretch goal with directional micro evidence, but it is 2-3 research cycles away from validation. It should not be in the main narrative until ReLoRA works at 7B scale with real adapters composed on top.

---

## 6. Production Readiness Gap Analysis

### Distill Phase: ~60% validated

| Component | Status | Evidence | Gap |
|---|---|---|---|
| Teacher data generation | Working | $0.19/expert via Groq batch | Teacher quality vs size untested (exp_distillation_quality_vs_teacher active) |
| QLoRA training | Working | 500 steps, 15min, $0.25/expert | Only tested with 70B teacher |
| Quality validation | BROKEN | Contaminated eval (98% win rate is meaningless); MMLU shows -3.71pp; HumanEval shows +9.1pp | No general-purpose held-out benchmark; format mismatch with MCQ |
| Cost accounting | Working | $0.44-0.48/expert total | Overhead ratio 4.61x (teacher-dominated) |

### Compose Phase: ~20% validated

| Component | Status | Evidence | Gap |
|---|---|---|---|
| Equal-weight pre-merge | BROKEN | PPL in trillions, +127% degradation at N=5 | One bad adapter poisons everything |
| PPL-probe weighting | Micro only | r=0.990 oracle correlation at d=64 | Never tested at macro |
| Hash ring routing | Working (add/remove) | -2.23% mean, -4.53% max degradation | Content-agnostic; no quality-aware routing |
| vLLM serving | Partially working | 97% throughput at bs=1 | KILLED at bs=32 (12.9% degradation) |
| Expert hot-swap | Micro validated | 80.9ms worst-case at N=50 | Never tested at macro with vLLM |
| Quality monitoring | BROKEN | Cosine gating killed (r=-0.46), KL killed (rho=-0.7) | Only canary queries work (FNR=2%, but requires per-expert curation) |

### Evolve Phase: ~10% validated

| Component | Status | Evidence | Gap |
|---|---|---|---|
| Clone-and-compete | Never tested | Zero evidence | Core mechanism untested |
| Shadow scoring | KILLED at macro | r=-0.63, AUC=0.0, 100% disagreement | No validated scoring metric |
| Correction generation | Micro simulation only | Teacher error 19.6% (K1 barely passes) | Never tested with real teacher on real expert outputs |
| Model collapse detection | Micro simulation | LoRA + norm bounding prevents collapse | Never validated empirically |
| Execution-based self-learning | Micro simulation | DPO +23pp pass@1 in simulation | Never run with real code execution |

### Serving Infrastructure: ~30% validated

| Component | Status | Evidence | Gap |
|---|---|---|---|
| compose.py CLI | Conceptual | No production code exists | Not implemented |
| Expert registry | Conceptual | Hash ring add/remove tested | No persistent registry |
| Adapter storage/loading | Working (manual) | safetensors, 6MB each | No automated loading pipeline |
| Monitoring/observability | Not started | -- | No health checks, no metrics |
| Multi-GPU/scaling | Not tested | -- | Single-GPU only |

**Overall production readiness: ~25-30%**, not the 70% claimed in VISION.md.

The 70% figure appears to be calculated by averaging micro and macro readiness per pillar, giving full credit to micro results that have not transferred to macro. If we weight by "actually works at production scale with real data," the number drops dramatically. The two critical phases (Compose and Evolve) are at 20% and 10% respectively.

---

## 7. Honest Next Steps

### Budget reality: ~$28 remaining

At $0.16/hr (A5000) to $0.34/hr (4090), this buys approximately 80-175 GPU-hours. That is enough for 3-5 well-designed macro experiments.

### What to stop working on immediately

1. **Base-freedom (ReLoRA/LTE).** Interesting but not on the critical path. The base model works fine as-is. Spending budget on ReLoRA from-scratch at 7B scale when the compose phase is broken is misallocated.

2. **New micro experiments.** The micro experiment suite is extensive (40+ proven, 10+ killed). Additional micro experiments have diminishing returns. The remaining budget should go entirely to macro validation.

3. **Grassmannian skeleton refinements.** AP is proven optimal. TAAP killed. Mixed-rank packing supported. The skeleton is done. Stop.

4. **Domain taxonomy generation.** Validated as "infrastructure with weak validation." Not blocking anything. Park it.

### What to do with the remaining budget (priority order)

**Experiment 1: PPL-probe weighted composition at macro ($4-5, ~25 GPU-hours)**
This is THE experiment. Take the 5 macro adapters (bash, math, medical, python, sql). Implement PPL-probe weighting (from micro/models/cross_domain_dilution_vs_k/) at macro scale. Measure composed quality with PPL-probe weights vs equal-weight vs top-1 routing. If PPL-probe composition reduces the +127% degradation to under +20%, the compose phase is saved. If not, the architecture needs fundamental rethinking.

Kill criteria:
- PPL-probe composed model > 2x single-expert PPL on >50% of domains (routing insufficient)
- PPL-probe adds >100ms per query latency on 7B model (too slow)

**Experiment 2: Poisoned adapter detection and removal ($2-3)**
The SQL adapter poisons everything. Build a simple leave-one-out PPL screen: for each adapter in the composition, measure PPL with and without it. Remove adapters that increase PPL when added. This is the leave-one-out experiment already in HYPOTHESES.yml (exp_leave_one_out_expert_ranking) -- finish it.

**Experiment 3: SOLE vs monolithic LoRA ($3-4)**
exp_sole_vs_full_finetune was submitted but crashed. Fix and re-run. This is the fundamental value proposition test: does modular composition beat a single LoRA trained on the same data? If it loses on quality, the value prop is purely operational (updatability). That is still valuable but changes the narrative.

**Experiment 4: Reasoning adapter composition ($3-4)**
The reasoning adapter shows +10.6pp on MATH-500 (K1 PASS). K2 (composition interference) and K3 (composition beats single) are untested. This is the "killer demo" -- finish it. If reasoning composes with domain adapters without interference, that is genuinely novel and publishable.

### What to save budget for

Reserve $5-8 for unexpected follow-ups on the above experiments. Do not start any new experiment streams.

### The honest assessment

The SOLE project has produced real, valuable research:
- Structural orthogonality scaling law (legitimate, publishable)
- Grassmannian expert packing theory (novel, complete)
- Distillation pipeline with cost accounting ($0.44/expert at 7B scale)
- Systematic experiment methodology (HYPOTHESES.yml, kill criteria discipline, adversarial reviews)

The project is NOT close to production. The compose phase is broken. The evolve phase has no validated scoring metric. The competitive positioning vs frontier models is not credible. The parameter-counting scaling tables are misleading.

The minimum viable demo is: "5-10 domain experts, PPL-probe weighted composition, showing genuine domain improvement on held-out benchmarks without degrading general capability." This is achievable with the remaining budget if focused exclusively on Experiments 1-4 above.

**Should you continue, pivot, or stop?**

**Continue, but with ruthless focus.** The underlying research (orthogonality, distillation, Grassmannian geometry) is sound. The gap is in composition quality at macro scale. If PPL-probe weighting works at macro (Experiment 1), the project has a viable path. If it does not, the architecture needs to pivot from pre-merge composition to per-query routing (top-k expert selection), which is a more conventional MoE-like architecture but one that could actually work.

Do not spend another dollar on micro experiments, base-freedom, or scaling projections until the N=5 composition problem is solved at macro scale. Everything else is irrelevant if you cannot compose 5 adapters without destroying quality.

---

## Summary of Verdict by Phase

| Phase | VISION.md Claim | Reality | Verdict |
|---|---|---|---|
| Distill | "Create experts at $0.25/each" | Works for training; quality validation incomplete (HumanEval +9.1pp, MMLU -3.71pp) | **REVISE** -- quality gates needed |
| Compose | "Plug-and-play, zero recalibration" | Equal-weight composition broken (PPL trillions); PPL-probe untested at macro | **CRITICAL BLOCKER** |
| Evolve | "Clone, compete, prune autonomously" | Shadow scoring KILLED at macro; mechanism never tested | **NOT STARTED** (effectively) |
| Base-freedom | "No sacred weights" | Directional micro evidence; never tested at 7B | **STRETCH GOAL** -- park it |
| Competitive scaling | "7B + experts = GPT-4 scale" | Not credible; 7B base is the ceiling | **REMOVE from VISION.md** |
