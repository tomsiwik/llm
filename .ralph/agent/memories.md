# Agent Memories

## Patterns

### mem-1772584117-78b6
> Domain fine-tuning from shared base produces ~54% shared knowledge (consistent across 3 seeds: 53.6-54.1%). Task arithmetic merging (+44%) dilutes catastrophically. Shared-only model (+30%) proves unique knowledge essential. Decomposition is informative for analysis but not useful for composition of nonlinear modules.
<!-- tags: composition, decomposition | created: 2026-03-04 -->

### task-routing-beats-identity-routing (2026-03-04)
At micro scale with similar domains, routing by reconstruction loss (which groups minimize prediction error) outperforms routing by domain identity (contrastive/InfoNCE loss). The softmax router achieves +0.2% vs joint; contrastive keys achieve +141% worse. When domains share representation structure, task-aligned routing dominates identity-aligned routing.

### domain-discriminability-prerequisite (2026-03-04)
Contrastive routing keys (InfoNCE-trained K_i) require domain-discriminative hidden states to work. At d=64 with character-level tokenization (a-m vs n-z), even a linear probe only gets ~60% accuracy — domains are indistinguishable. This is a hard prerequisite, not a hyperparameter issue (tau sweep 0.05-1.0 all give ~53%).

### shared-attention-bottleneck (2026-03-04)
Shared attention is the composition bottleneck: independent composition (separate attention) fails at +13.5% vs joint. Shared-base composition works (-0.3% vs joint) but requires calibration (~100 steps). Any composition protocol must share attention layers.

## Decisions

### mem-1772584110-ca23
> Weight-space decomposition of capsule groups into shared + unique components is exact (reconstruction error <6e-08), but FAILS in function space due to ReLU nonlinearity. ReLU(shared_group(x)) + ReLU(unique_group(x)) ≠ ReLU((shared+unique)(x)). Result: +5.7% vs joint, worse than concatenation (-0.2%). Concatenation remains the validated composition method.
<!-- tags: composition, architecture, nonlinearity | created: 2026-03-04 -->

### softmax-router-validated-baseline (2026-03-04)
The softmax router calibration protocol is the validated composition routing baseline. It routes by task quality (reconstruction loss), not domain identity. +0.2% vs joint in 100 calibration steps. Contrastive keys were attempted as a replacement and killed at micro scale.

### contrastive-keys-deferred-to-macro (2026-03-04)
Contrastive routing keys killed at micro scale (53.3% accuracy vs 85% target). NOT necessarily dead at macro scale — stronger domain signal expected with larger models (d=256+), real domains (Python vs JavaScript), and BPE tokenization. Re-evaluate when macro validation begins.

## Context

### experiment-progression (2026-03-04)
Completed: gpt (dense baseline, 202K) → moe (standard MoE, 596K) → moe_freeze (lifecycle) → capsule_moe (rank-1 capsules, composition validated) → contrastive_router (KILLED). Next: Exp 2 (sparse routing — top-1 matching top-2 quality, uses existing softmax router).

### micro-scale-limitations (2026-03-04)
The micro arena (d=64, 4 layers, character-level names) has inherent limitations for domain routing research: (1) domains are nearly indistinguishable in hidden space, (2) G=4 groups too small for learned routing advantages over uniform, (3) character vocabulary shared across domains. These don't invalidate findings but constrain which routing mechanisms can be tested. Task-based routing works; identity-based routing doesn't at this scale.

### hard-vs-soft-selection-phase-transition (2026-03-04)
Hard selection (k=1) and soft selection (k>=2) are fundamentally different regimes, separated by a phase transition — not a gradual tradeoff. At k=1, w_{g*}=1.0 removes gradient information about router confidence. At k=2+, relative weights between selected groups preserve confidence. k=2/4/8 are within 1.6% of each other; k=1 degrades by +200%. The "knee" in the quality-compute curve is between k=1 and k=2.

### minimum-routing-bandwidth (2026-03-04)
There exists a minimum "routing bandwidth" (number of active groups) below which quality collapses. At micro scale with 8K params/group, that minimum is k=2. Below it, a single group lacks capacity to represent the language model. This is capacity-bound, not mechanism-bound — Switch Transformer uses k=1 successfully at scale with much larger experts. k=1 viability scales with expert capacity.

### soft-selection-as-portfolio-diversification (2026-03-04)
k=2 succeeds via a "portfolio effect": soft mixing of 2 groups provides diversification that smooths over routing uncertainty. The router's value is in PREVENTING bad routing (vs uniform: catastrophic), not in achieving great routing (+1.3% vs joint). Learned routing is essential but operates as a safety net, not a precision instrument, at micro scale.

### router-entropy-inversion (2026-03-04)
Router entropy is HIGHER at k=1 (0.861 H_max) than k=2 (0.756 H_max). At k=1, gradients only reach the selected group — no comparative signal about alternatives. At k=2+, gradients flow to multiple groups, providing relative feedback that sharpens the router. This means k=1 training actively degrades routing quality.

## Decisions

### k2-optimal-sparsity (2026-03-04)
k=2 is the optimal composition sparsity at micro scale. No quality improvement from k=4/8 (within 1.6%), catastrophic degradation at k=1 (+200%). At k=2, composition achieves +1.3% vs joint — nearly parity. No need to increase k for quality; no opportunity to decrease k for compute at this scale.

### sparse-routing-deferred-to-macro (2026-03-04)
Top-1 routing killed at micro scale (two kill thresholds massively exceeded: +200% vs 10%, +204% vs 15%). This is capacity-bound — 8K params/group insufficient for single-group representation. Switch Transformer's k=1 success at scale implies this mechanism works with sufficient expert capacity. Re-evaluate at macro scale with larger groups.

### n5-scaling-validated (2026-03-04)
Composition protocol scales to N=5 domains with +1.6% degradation vs joint (within 5% threshold). Orthogonality degrades gracefully: mean cosine sim 0.000 (N=2) → 0.112 (N=5), max 0.167, all well below 0.5 concern threshold. Linear extrapolation suggests orthogonality concern around N≈9-10 at d=64. Calibration scales linearly (200 steps for N=5 vs 100 for N=2).

### micro-arena-exhausted (2026-03-04)
The micro arena (d=64, 4 layers, character-level names) is fully explored after 5 experiments. Validated mechanisms: softmax routing by task quality, k=2 minimum sparsity, concatenation composition, shared attention. Killed mechanisms: A-matrix self-routing, contrastive keys, top-1 sparse routing, Procrustes decomposition (all at micro scale). Remaining questions are scale-bound — transition to macro (0.5B + LoRA experts vs 1.5B monolithic).

### data-quantity-affects-composition (2026-03-04)
Smaller domains degrade more under composition: u_z (2.4K names) shows +3.0% vs joint, while a_e (10.5K) shows -0.1%. Less training data → less-specialized capsule groups → more degradation when composed. At macro scale with real domains, ensure sufficient per-domain training data.
