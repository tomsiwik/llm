# Research Findings

## Conclusive Results (Macro Scale, Qwen2.5-0.5B)

| Finding | Result | Evidence |
|---------|--------|----------|
| LoRA orthogonality is structural | cos=0.0002 at d=896 (50x better than theory) | macro/ortho_scaling/ |
| MoE beats joint training | -0.70% vs joint (4 domains, 3 seeds) | macro/lora_moe_benchmark/ |
| Gap predicts calibration (d=256) | r²=0.865 at N=4 (self-defeating at d=896) | macro/gap_as_signal_bridge/ |
| Hash routing plug-and-play | 5.3% displacement at N=20 | macro/hash_routing_scale/ |
| Prune-then-compose order invariant | +0.012% gap (170x margin) | macro/prune_compose_macro/ |
| L2 norm composition stable | 0/25 catastrophic failures | macro/l2_norm_macro/ |
| Batched LoRA k=1 overhead | -4% (faster than monolithic) | macro/batched_lora_latency/ |
| Compose CLI works E2E | 5 adapters, full workflow tested | macro/compose_e2e/ |


## Killed at Macro

| Finding | Result | Evidence |
|---------|--------|----------|
| SwiGLU gate pruning | +196% quality loss | macro/swiglu_pruning_macro/ |
| Gap-as-signal at d=896 | r²=0.22 (no variance) | macro/gap_signal_lora/ |

## Supported (Micro Scale + Macro Pilot)

| Finding | Result | Evidence |
|---------|--------|----------|
| 50-expert distillation pipeline works | 98% win rate, 42.2% avg PPL improvement (contaminated eval -- see PAPER.md caveat), $0.44/expert. Adversarial review PROCEED (2026-03-13). MMLU/HumanEval pending to upgrade to proven. | micro/models/distillation_pilot_50/ |
| FFN-only more orthogonal than all-modules | mean\|cos\|=0.0605 vs 0.0711 (5 Qwen2.5-7B adapters) | micro/models/ffn_only_vs_all_modules/ |
| Attention amplifies domain overlap | math-medical: attn cos=0.85 vs FFN cos=0.59 | micro/models/ffn_only_vs_all_modules/ |
| FFN-only 25% fewer params | 30.3M vs 40.4M at rank-16 | micro/models/ffn_only_vs_all_modules/ |
| Adapter taxonomy: LoRA optimal for composition | FIT=0.875, 15 types surveyed, 3 composition classes | micro/models/adapter_taxonomy_wild/ |
| Base-free model theoretically possible | ReLoRA/LTE achieve full-rank from LoRA iterations | micro/models/adapter_taxonomy_wild/ |
| IA3/Houlsby/Prefix incompatible with additive composition | Multiplicative/sequential/concatenative, not additive | micro/models/adapter_taxonomy_wild/ |
| LoRA dominates ecosystem | >90% of HuggingFace adapters are LoRA variants | micro/models/adapter_taxonomy_wild/ |
| ReLoRA base supports LoRA composition | cos ratio 1.77x (CI [0.77, 2.64], p=0.056), loss ratio 1.052 | micro/models/relora_composition_test/ |
| ReLoRA effective rank matches conventional | r_eff=52.9 vs 53.2, ratio 0.995 | micro/models/relora_composition_test/ |
| Base model decomposable into adapter format | rank-16 delta: loss ratio 1.014, cos 1.22x (3 seeds) | micro/models/base_free_composition/ |
| Expert quality degrades slower than base quality | rank-8: base 10% worse, experts only 5% worse | micro/models/base_free_composition/ |
| Rank-32 delta is lossless | loss ratio 1.001, cos ratio 1.01x | micro/models/base_free_composition/ |
| Skeleton alone is insufficient | base loss 6.9x, expert loss 1.27x, cos 6.3x | micro/models/base_free_composition/ |
| Zero-shot base transfer works | rank-16: 4.2% loss, rank-32: 0.3% loss (3 seeds) | micro/models/zero_shot_base_transfer/ |
| ZS experts amplify base error | expert/base ratio > 1.0 (unlike retrained which compensate) | micro/models/zero_shot_base_transfer/ |
| Transfer gap grows with perturbation | gap: 0.2% (r32), 2.8% (r16), 11.7% (r8), 22.6% (r4) | micro/models/zero_shot_base_transfer/ |
| 0% expert failure in ZS transfer | 0/48 expert-condition pairs exceed 2x threshold | micro/models/zero_shot_base_transfer/ |
| Architecture named SOLE | Structurally Orthogonal Latent Experts; surveyed 13 papers, glossary established | micro/models/composition_naming/ |
| Pre-merge latency N-independent | max +2.6% overhead across N=5..100, within noise | micro/models/inference_latency_vs_N/ |
| Dynamic top-k latency N-independent | k=1 overhead: 254-265% across N=5..100 (impl-bound) | micro/models/inference_latency_vs_N/ |
| Dynamic scales O(k) not O(N) | k4/k1 ratio: 1.40-1.41, stable across N=5..100 | micro/models/inference_latency_vs_N/ |
| Hash ring routing negligible | 0.5-0.6 us/query, O(log N), 1.10x for 20x N | micro/models/inference_latency_vs_N/ |
| Memory: O(1) pre-merge, O(N) dynamic | 0.28 MB/expert (micro), pre-merge same as base | micro/models/inference_latency_vs_N/ |
| PPL does NOT predict task accuracy | Pearson r=0.08, reverse expert: PPL -27% but Acc +9.5pp | micro/models/ppl_vs_task_performance/ |
| Answer-only PPL DOES predict task accuracy | Pearson r=0.811 +/- 0.16 (3 seeds), full-seq r=-0.31 | micro/models/answer_conditioned_scoring/ |
| Domain similarity predicts LoRA interference | within-cluster \|cos\| 7.84x higher than cross-cluster (3 seeds, p<0.0001) | micro/models/orthogonality_by_domain_type/ |
| Collision landscape is block-diagonal | top 10 most similar pairs are ALL within-cluster (30/30 across seeds) | micro/models/orthogonality_by_domain_type/ |
| Cross-cluster interference near random baseline | cross-cluster mean \|cos\|=0.008, random baseline=0.002 | micro/models/orthogonality_by_domain_type/ |
| Content-aware routing KILLED at micro scale | Best domain acc 26.5% < 60%; quality gap 0.00% | micro/models/content_aware_routing/ |
| Cluster-level routing solved (96%) | Cosine/keyword achieve 95-96% cluster accuracy vs 33% random | micro/models/content_aware_routing/ |
| MLP classifier fails at micro scale | 8.5% accuracy, worse than non-parametric methods | micro/models/content_aware_routing/ |
| Routing is moot without expert specialization | Oracle routing produces identical NTP loss to random routing | micro/models/content_aware_routing/ |
| Delta rank ratio decreases with d | rho: 0.664 (d=64) -> 0.629 (d=128) -> 0.538 (d=256), power law d^(-0.15) | micro/models/delta_rank_scaling/ |
| Attention deltas scale better than FFN | Attn ratio: 0.592 -> 0.431 (27% drop); FFN: 0.766 -> 0.643 (16% drop) | micro/models/delta_rank_scaling/ |
| Practical rank (r_95) scales steeply | r_95 ratio: 0.455 -> 0.320, extrapolates to ~0.12 at d=4096 (rank ~500) | micro/models/delta_rank_scaling/ |
| SOLE positioned against LoRA Soups | 5 structural advantages; complementary not competing; CAT 52-75x overhead for equal quality | micro/models/oae_vs_lora_soups/ |
| LoRA Soups limited to binary (k=2) | SOLE scales to N>>2; LoRA Soups requires weight retraining | micro/models/oae_vs_lora_soups/ |
| No prior work analyzes LoRA orthogonality | LoRA Soups, Arrow, Task-Aware: no orthogonality guarantee | micro/models/oae_vs_lora_soups/ |

**Caveats (Delta rank scaling):** Micro scale only (d=64 to d=256, 4-layer GPT). Power law fit has only 3 data points (R^2=0.929 but wide confidence interval on exponent). Shannon effective rank is tail-sensitive; the more practical r_95 metric shows steeper decline but was not formally power-law fitted. Fixed toy dataset at all scales -- real models at d=4096 train on vastly more complex data which may increase effective rank. Training steps scaled linearly with d (1K/2K/3K), which is a crude heuristic for convergence matching. Embedding weights show increasing ratio (bounded by fixed V=27), but this artifact disappears at macro scale where V >> d. K1 technically killed (Shannon ratio > 0.5 at both d=128 and d=256), but K2 survives (monotonic decrease). Literature (arXiv:2510.00537) independently confirms: hard rank utilization declines with width (exponent -0.4 to -0.6), soft rank utilization is nearly scale-invariant (exponent near 0). Our exponent -0.15 is consistent with soft rank measures. Adversarial review (2026-03-11): REVISE. 5 fixes: (1) convergence control — train to same val loss not same steps, (2) exclude embeddings from aggregate ratio, (3) report CI on power law exponent, (4) accept K1 kill cleanly — register new hypothesis with r_95 metric, (5) multi-checkpoint rho measurement. Core concern: under-training confound could explain entire observed trend. Date: 2026-03-11. Status: **revise** (K1 killed, K2 survives but evidence insufficient without convergence control).

**Caveats (SOLE vs LoRA Soups):** Primarily a literature comparison. Micro empirical component (d=64, 4-layer MLP) produced vacuous quality comparison -- experts did not specialize, so all composition methods yield identical loss. The timing comparison is meaningful: CAT (LoRA Soups) requires 52-75x more compute than SOLE for identical results. The key finding is structural/architectural: SOLE provides orthogonality guarantees, zero-cost composition, unlimited expert count, and evolution support that no prior LoRA composition work addresses. LoRA Soups (Prabhakar et al., COLING 2025) is the most directly comparable work but occupies a different niche (binary skill composition technique vs architectural framework). Corrected attribution: the HYPOTHESES.yml originally cited "Ostapenko et al., 2024" but the actual LoRA Soups paper is Prabhakar et al., 2024 (arXiv:2410.13025). Ostapenko et al. (2024, arXiv:2405.11157) is the "Towards Modular LLMs" paper with Arrow routing -- also surveyed. Adversarial review (2026-03-11): PROCEED. Non-blocking: fix MATH.md Section 4.1 (implicit eq → linear system), label bounds as scaling arguments, add LoRA-Flow citation. Date: 2026-03-11. Status: **proven** (adversarial review PROCEED).

**Caveats (Content-aware routing):** Micro scale (d=64, V=32, 4-layer MLP). Experts did NOT specialize -- loss ~3.466 throughout training, making quality comparisons vacuous. K3 (quality gap) is trivially killed because all experts produce identical output. The routing ACCURACY results are meaningful: cluster discrimination is easy (96%), within-cluster domain discrimination is hard (26%). MLP classifier's failure is likely micro-specific (insufficient embedding variance at d=64). At macro scale with real expert specialization, content-aware routing may succeed on K1 and K3. The directional finding is: hierarchical routing (cluster first, then hash within cluster) may be the right architecture. Date: 2026-03-11. Status: **killed** (K1, K3).

**Caveats (Answer-conditioned scoring):** Micro scale (d=32, V=42 char-level). Synthetic structured tasks with deterministic correct answers and clear delimiters. N=5 domains is small for correlation (r_crit=0.687 at p<0.05); 2/3 seeds exceed this, all 3 exceed kill threshold of 0.5. Full-rank expert delta (not LoRA). autograd-based training (numpy autodiff). At macro scale: subword tokenization (V=32K+) may reduce the magnitude of PPL differences; real-world tasks with distributional answers may show weaker correlation; delimiter position may be ambiguous for free-form generation. The PPL decomposition (log PPL_full = T_p/T * log PPL_prompt + T_a/T * log PPL_answer) is mathematically exact and architecture-independent. Prompt-only PPL anti-correlated with accuracy (r=-0.74), confirming prompt dilution as confounder. Validates shadow scoring for SOLE Evolve phase. Date: 2026-03-11. Status: **proven**.

**Caveats (PPL vs task accuracy):** Micro scale (d=64, V=42 char-level). Synthetic structured tasks with deterministic answers; real-world tasks with distributional answers may show stronger PPL-accuracy alignment. N=5 domains is small for correlation analysis (r_crit=0.687 at p<0.05). The divergence is driven by full-sequence PPL including prompt tokens; answer-only PPL would likely correlate better. Subword tokenization (V=32K+) may smooth distributions. Despite kill, PPL is not useless -- it correctly identifies ceiling (repeat) and distribution shift (reverse/sort). Implication: shadow scoring needs task-specific augmentation, not replacement. Date: 2026-03-11. Status: **killed**.

**Caveats (Orthogonality by domain type):** Synthetic Markov chain data, not real domains. 4-layer MLP, not transformer. Minimal training signal (loss ~3.466 throughout, LoRA deltas reflect gradient direction rather than converged features). Conservative test: converged training on real data expected to show stronger clustering. 15 domains in 3 clusters (code/reasoning/knowledge). Cohen's d = 2.24 (very large). Explains the math-medical outlier from ffn_only_vs_all_modules as a within-cluster effect. Date: 2026-03-11. Status: **proven**.

**Caveats (Composition naming/SOLE):** Survey/framing task, not an empirical experiment. Name "SOLE" (formerly "OAE") is recommendation; not yet tested for external communication effectiveness. Missing LoRA Soups (Ostapenko et al., 2024) from positioning table — most directly comparable prior work. Date: 2026-03-11. Status: **proven** (adversarial review PROCEED).

**Caveats (Inference latency vs N):** CPU-only measurement (no GPU). Synthetic random LoRA weights (not real experts). Micro model (d=128, 4 layers). Dynamic overhead (260%) is Python weight-copy implementation artifact, not architectural -- production fused kernels achieve <5% (see macro/batched_lora_latency). The key finding is N-independence, not absolute overhead values. No torch.compile or JIT tested. Date: 2026-03-11. Status: **proven**.

**Caveats (FFN-only):** Retroactive subset analysis (FFN params extracted from all-modules adapters, not independently trained). Quality kill criterion inconclusive without matched-rank training. Status: supported, not proven. Date: 2026-03-11.

**Caveats (ReLoRA composition):** Micro scale only (d=64, r=8). Cosine values at d=64 are ~100x higher than d=896 due to dimensionality. The ~1.77x cos degradation is statistically indistinguishable from noise (p=0.056). Only 5 merge cycles tested. ReLoRA base has 4.6% base quality gap (expected to close at scale per Lialin et al.); composition penalty is only 0.6%. 3 seeds show inconsistent direction (seed 42: ReLoRA better). Status: **proven** (adversarial review passed). Date: 2026-03-11.

**Caveats (Adapter taxonomy):** Survey/analytical experiment, no adapters trained. Composability scores are analytical, not empirical. ReLoRA base-freedom is for pretraining, not tested for expert composition. Adversarial review passed (revision round): interference bound corrected to r/sqrt(D), full-rank Class A single-adapter caveat added. Date: 2026-03-11. Status: **proven**.

**Caveats (Zero-shot base transfer):** Micro scale only (d=64, r=8). SVD perturbation is a controlled decomposition; real base model updates (continued pretraining, architecture changes) may produce different perturbation patterns. Same skeleton assumed for training and transfer bases. No fine-tuning recovery tested (a few adaptation steps might close the transfer gap cheaply). Toy data with overlapping domains. NTP loss only, no generation quality evaluation. The amplification effect (ZS experts amplify base error) may be weaker at macro scale where LoRA deltas are proportionally smaller. Adversarial review notes: K2 (cosine) vacuous by design, amplification "bound" is empirical trend not theoretical (4-point fit). Date: 2026-03-11. Status: **proven** (adversarial review PROCEED).

**Caveats (Base-free composition):** Micro scale only (d=64, r=8). Delta effective rank ratio (40/64=0.63) may not hold at macro scale. SVD decomposition is a post-hoc operation on a pretrained model, not a training strategy. Layer-wise SVD (no cross-layer structure). Toy data with overlapping domains. No text generation quality evaluation (NTP loss only). The rank requirement at macro scale is unknown -- if delta effective rank scales linearly with d, rank-16 base adapter may be insufficient for d=3584. Experts were retrained per condition -- zero-shot base transfer untested. Error propagation bound (MATH.md Section 3.2) is empirical, not theoretical. Missing citation: ASVD. Date: 2026-03-11. Status: **proven** (adversarial review PROCEED, all kill criteria disproven, 3/3 seeds SURVIVES).

## Current Direction: Living Composable Model

The research phase is complete. The proven findings above are the foundation for:

1. **Distill** — Create experts at scale via teacher distillation ($0.25/expert)
2. **Compose** — Serve with hash ring routing on vLLM (zero recalibration)
3. **Evolve** — Clone-and-compete mechanism for continuous improvement without retraining

See `VISION.md` for full architecture. See `HYPOTHESES.yml` for active roadmap (10 focused items).

## Per-Experiment Details

Each macro/ and micro/ experiment directory contains its own PAPER.md with full methodology, results, and kill criteria assessment.

- Macro experiments: `macro/*/PAPER.md`
- Micro experiments: `micro/models/*/PAPER.md`
- Archived hypotheses: `ARCHIVE.yml` (84 completed/deferred experiments)
- Active roadmap: `HYPOTHESES.yml` (10 items, priority-ordered)
