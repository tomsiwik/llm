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

### routing-irrelevant-at-micro-scale (2026-03-05)
At G=8 with homogeneous character-level data, all routing strategies (softmax, softmax-no-balance-loss, uniform, LSH T=1/2/4/8) produce statistically indistinguishable quality (max spread 1.34%, no p<0.05). Routing quality only matters at larger G with diverse data. Implication: micro-scale routing experiments are ceiling-limited.

### inter-layer-coupling-revival (2026-03-05)
Freezing upstream MLP layers reduces downstream neuron revival by 79-94%. Self-revival is minimal (2-8%, overlaps profiling noise floor). Revival is strictly feed-forward (0% upstream). Practical rule: prune AFTER training completes, not during, because upstream weight changes revive downstream dead neurons.

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

### dead-capsule-pruning-validated (2026-03-04)
57% of composed ReLU Router capsules are dead (never fire). Pruning at t=0 produces EXACT zero quality change — a definitional property of ReLU. Prune-then-calibrate achieves -1.1% vs joint (comparable, within noise) with 37% fewer params. Layer 0 is special (0.4% dead, processes generic embeddings), layers 1-3 are 71-82% dead. Most death is training-induced (~92%), not domain-induced. Pruning and calibration are order-independent. Updated protocol: compose → prune dead → calibrate.

### pruning-composition-protocol (2026-03-04)
The validated composition protocol is now: pretrain shared base → fine-tune capsule pools/domain → compose by concatenation → profile activations (20 batches) → prune dead capsules (tau=0) → calibrate surviving capsules (100 steps). This strictly dominates weight averaging (better quality, fewer params when calibration budget available).

### relu-pruning-transfer-caveat (2026-03-04)
The exact zero-change pruning theorem depends fundamentally on ReLU's hard gating (zero output for negative inputs). Does NOT transfer to GELU/SiLU (no hard zero). Does NOT transfer to standard LoRA (no ReLU between A and B). Transfer clean ONLY if macro architecture uses ReLU capsule pools (consistent with ReLU Router lineage). For non-ReLU, need approximate pruning with magnitude thresholds and Section 3.2 bounded error analysis.

### dead-capsule-pruning-is-general-relu (2026-03-04)
Exp 10 (Pruning Controls) proved dead capsule pruning is a GENERAL ReLU technique, not composition-specific. 54.3% of capsules are dead in single-domain models (before composition), accounting for 87% of the 62.1% composed death rate. Composition-induced death is only 7.7%. This means pruning is applicable to ANY ReLU-based adapter, not just composed ones. Revises Exp 9's Assumption 6.

### random-vs-targeted-pruning (2026-03-04)
Random pruning at the same rate as targeted pruning is competitive. Without calibration, random is 2.9% BETTER than targeted (regularization effect from removing alive capsules in overparameterized model). With calibration, targeted wins by 0.7-0.8pp but not statistically significant at n=3 seeds (effect size ~0.45 SD). Profiling matters directionally but the advantage is small. At macro scale with more evaluation data, the effect should become detectable if real.

### death-trajectory-non-monotonic (2026-03-04)
ReLU capsule death follows a "spike and slow decay" trajectory, NOT monotonic accumulation. 18.8% at init → 55.1% peak at ~50 steps → slow recovery to 47.3% by 3200 steps. Dead neurons CAN revive through inter-layer coupling: weight updates in layers 0..l-1 shift input distributions to layer l, pushing dead neurons' pre-activations above zero without requiring gradient flow through the dead neuron. The irreversibility argument (dead neuron ⇒ zero gradient ⇒ permanently dead) is correct for single-layer but fails for multi-layer transformers.

### pruning-robust-to-duration (2026-03-04)
Death rate stays in the 40-55% range for any training duration >50 steps. The Exp 10 measurement (54.3% at S=200) is representative, not an early-training artifact. Pruning opportunity is duration-robust. At macro scale with 100K+ steps, expect 40-45% dead (logarithmic extrapolation, ~2.5pp decrease per 10x training, treat with caution).

### capsule-revival-is-real (2026-03-04)
Per-capsule tracking (Exp 18) proved true revival dominates aggregate death decrease: 28.1% of capsules dead at S=100 revive by S=3200. Revival accelerates with training (5.8% to 15.9% per interval). Jaccard(dead_100, dead_3200) = 0.669 — death is sticky but not permanent. Layer 3 has highest revival (inter-layer coupling confirmed). Practical implication: prune AFTER training completes, not during.

### revival-narrative-caveat (2026-03-04)
Exp 18's decomposition narrative has a labeling issue: the -3.8 pp "new deaths avoided" actually means new deaths OCCURRED and offset revival. The math is correct (15.8 - 3.8 = 12.0 pp net decrease), but the narrative overstates "gradient shrinkage prevents new deaths." Fix: relabel as "new death offset."

### death-loss-decoupled (2026-03-04)
Val loss and death rate are decoupled (Pearson r=0.027 over training duration sweep). Quality improvement comes from alive-neuron specialization, not from reducing the number of dead neurons. Consistent with Exp 9's exact pruning theorem (dead neurons contribute exactly zero).

### exp17-reviewer-caveats (2026-03-04)
Adversarial review of Exp 17 identified: (1) kill criterion trigger is marginal — 5.7pp vs 5pp threshold, t≈1.5, p≈0.13, not conventionally significant at n=3; (2) per-capsule identity tracking absent — aggregate rate decrease could be population turnover, not same-capsule revival; (3) constant-LR limitation — warmup/cosine decay at macro will qualitatively change the trajectory; (4) curve fit tau=25 at grid boundary.

### exp10-reviewer-caveats (2026-03-04)
Adversarial review of Exp 10 identified: (1) "87% already dead" is aggregate-rate comparison, not identity-level capsule tracking — same rate doesn't prove same capsules die; (2) decomposition table arithmetic issue — delta_domain overlaps with delta_shift, fractions sum to 104.6%; (3) targeted vs random+cal advantage not statistically significant. These are presentation/precision issues, not fundamental flaws.

### profiling-noise-negligible (2026-03-04)
Exp 12 (Profiling Noise Quantification) proved profiling noise is negligible: same-checkpoint disagreement 2.6-3.8%, well below 20% threshold. Noise fraction -6.2% (negative — consensus shows MORE revival). All Exp 18 metrics reproduced within noise. The profiling protocol (20 batches x 32 samples) is confirmed sufficient for binary f=0 classification. Practical implication: profiling-based pruning decisions are reliable.

### profiling-neff-caveat (2026-03-04)
The binomial noise model in Exp 12 MATH.md uses N_prof=640 (number of names), but activation frequency is computed over (name, token) positions — effective N is ~10K+ (640 names x ~16 tokens each). This makes the profiling 16x more reliable than claimed, but also means the observed 2.6-3.8% disagreement EXCEEDS what the corrected binomial predicts — suggesting non-binomial noise sources (token-level correlations, name-dependent patterns). The simple binomial model is insufficient; the empirical measurement is the authoritative result.

### lr-schedule-changes-death-trajectory (2026-03-04)
Warmup eliminates 74% of the death spike (13.2% vs 51.6% at S=50). Cosine decay doubles revival (+11.8pp vs +5.1pp). Warmup+cosine equilibrium death is 19.6% — less than half constant LR's 47.3%. Quality and neuron survival are synergistic (best val loss at lowest death). Revised macro prediction: ~20% dead under standard warmup+cosine training, not 47%. Pruning yield ~13% param reduction (not 37%).

### warmup-cosine-revival-caveat (2026-03-04)
Warmup+cosine shows only 2.0pp absolute revival (from 21.6% to 19.6%) — LESS than constant LR's 5.1pp. The kill criterion for cosine revival passes only because cosine-only (11.8pp) carries it via max() aggregation. Lower starting death rate means fewer capsules available to revive. The "revival" metric at aggregate level conflates true per-capsule revival with reduced new deaths.

### consensus-comparison-asymmetry (2026-03-04)
Consensus dead/alive tracking has a subtle asymmetry: consensus "alive" at S2 means alive in EITHER profiling run (more permissive), while single-run "alive" requires alive in that specific run. This biases consensus revival upward independently of noise correction. The -6.2% noise fraction is contaminated by this denominator shift. The clean metric is same-checkpoint disagreement (2.6-3.8%), not the transition comparison.

### warmup-fraction-sensitivity (2026-03-04)
Warmup fraction is a first-order determinant of ReLU death. 1% warmup (R=0.64) captures only 31% of spike suppression benefit; 5% (R=3.2) captures 90%. The cumulative-LR-integral model predicts death@50 within 0.6pp MAE across all 5 fractions — one-parameter model calibrated from constant-LR baseline. Quality saturates before death: wc_05 and wc_10 have identical val loss despite 6pp death difference. Equilibrium death is warmup-dependent: 17.5% (20%) to 38.0% (1%) at S=3200.

### macro-warmup-steps-not-fraction (2026-03-04)
CRITICAL for macro extrapolation: absolute warmup steps (S_w) vs T_spike matters, NOT warmup fraction (f_w). At macro scale (S_total=300K+), even tiny fractions give large S_w. Chinchilla's 0.33% warmup at 1.5M steps gives S_w=5000, R=100 (strong suppression), NOT the R=0.22 computed using micro's S_total=3200. The paper's fraction-based macro predictions are misleading without this correction. Always compute R = S_w / T_spike using absolute steps.

### silu-pruning-negative-result (2026-03-04)
SiLU magnitude-threshold pruning passes kill criterion (<5% degradation) but provides NO free compression. Activation floor ~0.046 prevents dead-neuron formation. 0% prunable at safe thresholds (tau<=0.01). At aggressive tau=0.1, 32% pruned but high seed variance (3.7-85.7%). ReLU pruning is strictly superior (17.6% single-domain / 57% composed at exact 0% loss). For macro SiLU models, investigate: (1) SwiGLU-gated output profiling, (2) gradient-based importance, (3) low-rank factorization.

### kill-criterion-design-lesson (2026-03-04)
Kill criteria must test the failure mode being investigated, not just pass/fail quality. Exp 15's criterion "degradation >5%" is vacuous because tau=0 (prune nothing) always passes. Better criterion: "achieves <10% compression at <5% degradation" — tests whether the method WORKS, not just whether it's SAFE.

### training-time-compat-killed (2026-03-04)
Exp 11 KILL: Auxiliary losses during fine-tuning (weight orthogonality, output-norm matching) WORSEN zero-shot concatenation composition. Best is no_aux at +6.4% vs joint; all aux conditions worse. Root cause: `ReLU(A_1 x) + ReLU(A_2 x) != ReLU((A_1+A_2) x)` is scale-invariant — weight-space regularization cannot close the function-space gap for concatenation. However, combined aux improves weight averaging to -0.2% vs joint (from +1.3% no-aux). This eliminates the entire class of "training-time weight-space regularization for concatenation" approaches.

### cumulative-lr-integral-model (2026-03-04)
Death rate at S=T_spike is proportional to the integral of LR over [0, T_spike]. For linear warmup: F = 25.5/S_w (for S_w >= T_spike) or F = 1 - (S_w-1)/100 (for S_w < T_spike). This is a scaling law (curve fit), not a mechanistic theory — validated for linear warmup + cosine only. Step-function or exponential warmup could break it due to gradient-magnitude dynamics. The "arbitrary schedule" prediction claim is speculative beyond linear warmup.

### pre-composition-profiling-validated (2026-03-04)
Exp 16 proved the same capsule indices die in single-domain and composed models. Jaccard=0.895 (threshold 0.50), overlap=0.986. Composition kills ~29 extra capsules/domain-half (~6%) but revives only ~4. Pre-composition profiling validated: prune each domain independently before composing. Updated protocol: profile → prune → compose → calibrate. N=5 scaling untested — monitor Jaccard as N increases.

### cross-setting-vs-cross-time-stability (2026-03-04)
Death identity is more stable across composition (Jaccard=0.895, Exp 16) than across training time (Jaccard=0.669, Exp 18). Composition perturbs death identity less than continued training. Implication: pruning decisions are robust to composition but not to continued fine-tuning.

### behavioral-dedup-threshold-sensitive (2026-03-04)
Behavioral (co-activation Jaccard) dedup is highly threshold-sensitive: tau_rho=0.3 → 19.3% behavioral-only capsules (PASS), tau_rho=0.5 → 10.8% (PASS), tau_rho=0.7 → 1.4% (KILL). Co-firing is abundant but strict output correlation is rare. Weight-cosine found ~0 pairs — the comparison is trivially won. All redundancy concentrated in Layer 0 (J=0.527); deeper layers specialize (J<0.05). Not a compression strategy (merging +0.3% vs concat). Value is diagnostic: Layer 0 capsules share generic character detectors across domains. Practical implication: share Layer 0 pools across domains.

### adversarial-review-cycle-pattern (2026-03-04)
Behavioral dedup went through a full REVISE→re-review cycle. The 5 fixes (bug fix, threshold sweep, per-seed table, narrative reframe, baseline acknowledgment) collectively transformed a weak paper into a PROCEED. Key lesson: threshold sweeps and per-seed reporting are the most impactful fixes — they reveal whether a finding is robust or an artifact of a single parameter choice.

### hybrid-attention-composition-proven (2026-03-05)
Simplified gated linear recurrence (omitting delta rule, L2 norm, conv1d, per-dim gating) is composition-compatible. Median gap +1.27% (5 seeds), interference ratio 0.59x (linear < full, excl Layer 0). Depth confound NOT confirmed. Critical caveat: ~20% catastrophic failure rate from unnormalized QK products — real GatedDeltaNet's L2 norm would address this. Delta rule interference reversal is the #1 macro risk (retrieval-and-correction mechanism fundamentally changes how composed domains interfere through shared state). Simplified variant only; 3 follow-up hypotheses generated.

### l2-norm-eliminates-composition-instability (2026-03-05)
L2 normalization on Q and K in linear attention eliminates catastrophic composition failures: 0/25 (0.0%) vs 4/25 (16.0%) unnormalized (Fisher's exact p=0.038). Median gap improves from +2.54% to -0.33%. Variance drops 22x (1.02% vs 22.26%). The same seeds that fail without L2 norm succeed with it. Hybrid attention conditional pass is now unconditional. Caveat: value norm boundedness is assumed but not guaranteed by L2 QK norm — this is a training dynamics property, not a hard guarantee.

### delta-rule-composition-safe (2026-03-05)
GatedDeltaNet delta rule (retrieval-and-correction: v_t - S^T k_t) does NOT reverse the favorable interference ordering (linear < full). Ratio = 0.74x (well under 1.0x threshold), median gap +0.39%, 0/7 catastrophic failures. Combined with L2 norm (0/25 failures) and simplified variant (0.59x): three hybrid attention experiments individually validate each GatedDeltaNet component for composition. Full-stack test (all components combined) is the remaining gap before declaring GatedDeltaNet composition-safe.

### subtree-grafting-viable-not-superior (2026-03-05)
Subtree grafting composition works (+0.67% vs weight averaging, within 3% threshold) but does not beat weight averaging. Root-only calibration is insufficient (+2.42%); all-gates recalibration is needed (+0.67%), which partially undermines the "preserved routing decisions" argument. Practical value: 2x cheaper fine-tuning (each domain trains half the tree, 66K vs 133K params). The function-space gap moves from "blended weights" to "reconnected gates" — reconciliation cost is similar for both methods.

### huffman-routing-needs-skew (2026-03-05)
Huffman-shaped expert tree is mathematically optimal (E[d] >= H(f), Shannon bound) but requires non-uniform routing to provide benefit. Micro-scale homogeneous data produces near-uniform routing (H=2.999/3.0), so Huffman degenerates to balanced tree (0% reduction). Mechanism validated with synthetic frequencies: 12% reduction at heavy skew, 26% at extreme. Quality insensitive to tree shape (max +0.30%). Scaling law: Zipf(1.0) at L=64 predicts 18.5% reduction; Zipf(1.5) predicts 62%. DeepSeek-V3 confirms non-uniform expert load at production scale.

### combined-parallel-linear-validated (2026-03-05)
Combined parallel blocks + pure-linear attention degrades only +1.48% vs sequential+hybrid (5% threshold, 95% CI ~3.3%). 2x2 factorial (5 seeds) shows additive effects (interaction +0.31%, within noise). Result dominated by pure-linear effect; parallel contribution is neutral. Validates simplest composition-safe block: x = x + GDN(Norm(x)) + CapsulePool(Norm(x)). Key reviewer insight: throughput is the real macro value, not quality. N=5 scaling untested.

### lsh-routing-needs-fair-baseline (2026-03-05)
LSH capsule routing review: REVISE. The "LSH beats softmax" claim is confounded by balance loss handicapping softmax (forcing near-uniform routing, entropy 0.73-0.84). Two blocking fixes: (1) softmax with balance_loss_coeff=0.0 control, (2) uniform routing baseline. Also: FLOP scaling table overstates O(T*d) when actual implementation is O(T*G*d); isotropy assumption after RMSNorm is incorrect. Direction is consistent across 3 seeds but p~0.044 is marginal.

### death-recovery-needs-hygiene-fixes (2026-03-05)
Death recovery mechanism review: REVISE. Core finding real (79-85% revival reduction from upstream freeze, 0% upstream revival from downstream training). Six analytical hygiene fixes needed: (1) verify embedding freeze status — if trainable, "upstream frozen = no input change" claim is incorrect; (2) explain 97.3% L1 revival from train-only-L0 vs 12.6% baseline — likely L1's own training creates offsetting new deaths; (3) report S=100 dead counts per condition (denominator differences); (4) fix code to separate upstream/downstream revival; (5) soften "confirmed" to "strongly supported"; (6) exclude L0 self-revival (0/0 denominator).

### parallel-blocks-composition-compatible (2026-03-05)
Parallel transformer blocks (attention+MLP from same normalized input, Tiny Aya style) are composition-compatible. Mean gap -0.39pp vs sequential (3 seeds), kill criterion (>5%) not triggered. ~30% faster fine-tuning (MLX-specific). The improvement claim is not statistically significant (driven by single outlier seed 777). Honest characterization: parallel is equivalent to sequential for composition, with a throughput bonus. Mechanistic claim (shorter interference chain) is untested — per-layer interference measurement absent.

### pure-linear-no-scaffolding-needed (2026-03-05)
Pure-linear attention (all 4 layers linear) does NOT need full attention scaffolding for composition. Degradation only +1.02% vs hybrid 3:1 (7 seeds, zero catastrophic failures). Interference is actually LOWER in pure-linear (0.54) than hybrid (0.83) or full (0.93) at deepest layer. Scaffolding hypothesis falsified. Key macro risks from review: (1) state capacity saturation at d_h=256 (linear attention finite recurrent state), (2) cumulative global context loss across 24+ layers, (3) 1.8x variance amplification. The +4% param advantage for pure-linear is a minor confound.

### split-freeze-protocol-validated (2026-03-05)
Split-and-freeze contribution protocol validated with calibration scope requirement. KC1: warm-start matches cold-start at -0.03% (3 seeds) — neutral at micro scale. KC2: freeze is structurally sound (zero weight drift) but calibration scope is critical: root-only catastrophic (+31.28%), gates-only marginal (+2.5%), right-tree (gates+leaves) clean (+0.09%). Novel finding: in freeze scenario, grafted leaves (not just gates) must be trainable during calibration — extends subtree_grafting's root-only-insufficient finding. split_leaf() mechanism remains untested (warm-start proxy only). Protocol: freeze subtree → graft new subtree → calibrate ALL unfrozen params (gates+leaves, ~200 steps).

### shared-layer0-pool-proven (2026-03-05)
Sharing a single Layer 0 capsule pool across domains IMPROVES zero-shot composition quality by 1.7-3.0% vs full concatenation (3 seeds). Mechanism: redundant Layer 0 pools cause per-layer residual stream magnitude distortion (distinct from falsified global loudness — a global scalar cannot correct per-layer imbalance). Average strategy recommended for D>2 (statistically indistinguishable from base/first but principled). 8.1% parameter savings. Cross-pool Jaccard = 0.544 (confirms behavioral_dedup). Scoped to zero-shot only — calibrated comparison untested. Capacity-reduction is an alternative explanation (Limitation 7).

### hierarchical-tree-routing-validated (2026-03-05)
Binary tree routing (depth-3, beam-2, 8 leaf groups) matches or beats flat softmax routing. Val loss 0.5177 vs 0.5223 (-0.87%, tree better all 3 seeds). Composition gap +0.17% vs +0.26%. Routing entropy 0.745 (moderately sharp vs flat's near-uniform). 11% fewer routing params (455 vs 512). Caveats from adversarial review: (1) tree has auxiliary gate entropy loss that flat lacks — confounds the -0.87% improvement; (2) composition used weight averaging which doesn't leverage tree topology — subtree grafting is the real structural test; (3) 3 seeds with overlapping ranges, "tree beats flat" is more accurately "tree is not worse than flat."

### bloom-filter-routing-killed (2026-03-05)
Bloom filters are fundamentally unsuited for expert routing pre-filtering. They provide exact membership testing but neural routing needs approximate similarity. Quantized hidden states produce different hash keys for similar tokens (bin boundary problem). At practical m (256-4096 bits): 0% elimination, 100% FPR. At large m (100K): 74-97% elimination but 76-99% false negatives (catastrophic). Scale-independent — structural mismatch. Use similarity-preserving structures (LSH, KD-trees) instead.

### revival-suppressed-under-composition (2026-03-05)
Composition suppresses capsule revival by 8.6pp (17.1% single → 8.5% composed). 88% of suppression is structural (dimensionality dilution from 2x capsules), only 12% from cross-domain gradient cancellation. At practical calibration length (100 steps), composed revival is only 2.9%. Strengthens pre-composition pruning — dead capsules stay dead. The reviewer noted the paper's "~1.9 sigma" is calculated with wrong denominator; actual significance is higher (t~3.7).

### par-pure-linear-scales-to-n5 (2026-03-05)
Parallel+pure-linear composition at N=5: gap +3.32% (threshold 8%, margin 2.4x). Cross-architecture degradation +1.19% (SMALLER than N=2's +1.48%). Architectural penalty does NOT amplify with N. Both architectures degrade ~2.5-3pp from N=2 to N=5. Zero catastrophic failures. The simplified block x = x + GDN(Norm(x)) + CapsulePool(Norm(x)) is validated at N=5. Reviewer caution: "amplification falsified" overstates — -0.29pp difference is within noise on 3 seeds.

### split-leaf-mechanism (2026-03-05)
Leaf splitting preserves function exactly at sigma=0 (mathematical identity from ReLU additivity). At sigma=0.001: 0.69% error (7x margin). Split matches independent at +0.16% (31x margin on 5%). CRITICAL: default sigma=0.01 FAILS at 6.53% — must use sigma=0.001. Routing weights halve contribution after split (50/50 gate normalization); fine-tuning corrects this. Convergence advantage 2/3 seeds (directional only).

### sequential-tree-graft-killed (2026-03-05)
Sequential tree grafting fails at N>2: degradation ratio 3.65x (threshold 2.0x). Progressive halving creates irrecoverable capacity imbalance — routing drift compounds superlinearly (alpha~1.18). Extended calibration (3x budget) does not help (3.63x). Selective calibration (root+graft gates) catastrophic at N>2 (24-35%). N=2 viable (+3.72%). For N>2 use flat MoE composition (concatenation + pruning + calibration, +1.6% at N=5).

### swiglu-gate-pruning-proven (2026-03-06)
SwiGLU gate-product profiling enables 66.5% capsule pruning at +1.22% quality loss (3-seed mean, tau=0.05). Gate product floor ~0.014, 3.3x lower than SiLU-only ~0.046. Random pruning baseline: gate-product 2.3x better. 95% CI [-2.27%, +4.72%] includes 3% threshold (n=3 limitation). CAVEAT: aux sparsity loss (L1 target 50%) inflates absolute pruning rate — macro transfer uncertain without aux loss. Core mechanism: SwiGLU up-projection = learned suppression mask creating bimodal distribution. Directly unblocks macro-scale pruning for SwiGLU architectures (Qwen/Llama/DeepSeek). Review required 5 fixes (floor bound, aux disclosure, random baseline, CI, citation); all applied correctly.

### consistent-hash-routing-proven (2026-03-06)
Consistent hash ring routing enables zero-recalibration expert add. N=8->9: +0.20% degradation (25x margin on 5%), 9.1% displacement (3.3x margin on 30%). 150 virtual nodes per expert, FNV1a hashing. New expert immediately receives ~1/N traffic. Softmax baseline: 0% displacement but new expert dead without recal. Extends LSH finding (routing quality irrelevant at G=8) to dynamic expert sets. Follow-ups: expert removal, N=5 scaling, hash vs softmax at equal recal budget.

### lora-merging-bakeoff-proven (2026-03-06)
Simple average dominates for merging orthogonal low-rank (r=8) LoRA deltas. TIES hurts (+7% N=2) because trimming compressed deltas destroys signal. DARE degrades monotonically with drop rate (p=0.3 ≈ avg, p=0.9 = +7%). Concat+cal wins N=2 (+1.14%) but loses N=5 (+5.07%) — suspected router underfit at 100 calibration steps. Method quality tracks information preservation of original deltas. Default: simple average for N>=3, concat+cal for N=2 with calibration budget. TIES zero-mask dilution bug fixed in v2 (negligible impact).

### lora-rank-composition-killed (2026-03-06)
LoRA rank has no effect on composition quality at micro scale. 32x sweep (r=2..64): 0.70pp quality range (<1pp threshold). Orthogonality-rank r²=0.156 (<0.2). Root cause: effective rank saturates at ~8 regardless of nominal rank — task dimensionality ~8 makes all ranks equivalent. Shared fraction ~50%, dead neuron rate ~30%, both rank-independent. Composition gap dominated by mechanism, not adapter capacity. CAVEAT: alpha/r confound (fixed alpha=1.0) — use alpha=r for future rank sweeps. Rank sensitivity expected at macro scale with diverse domains.

### reed-solomon-expert-encoding-proven (2026-03-06)
Lagrange interpolation reconstructs expert weights to float64 precision (max_err <5e-14). Any N-of-(N+k) experts reconstruct all N. Chebyshev nodes 13x better than uniform. Encoding 7ms, reconstruction 0.5ms — zero runtime cost. KC2 passes at N>=10 (20% overhead). Cross-layer parity useless (100,000+% degradation). Same primitive as Shamir (low novelty). Open question: cross-domain parity at same layer depth as interpolation experts.

### lz-dictionary-moe-revising (2026-03-06)
LZ Dictionary MoE under revision: 3 blocking fixes. (1) Kill criterion says "same total params" but no config matches Standard MoE's 596K. (2) Near-uniform alpha (entropy 0.999) means architecture degenerates to shared_mean_mlp + delta — needs shared-base ablation control. (3) Alpha gradient flow unverified in test suite. The -0.9% "advantage" may be regularization from fewer params, not dictionary composition. Also: statistical significance untested (n=3), effective rank derivation incorrect through ReLU nonlinearity.

### macro-pruning-anti-signal (2026-03-06)
Gate-product mean magnitude is an ANTI-signal for safe prunability at macro scale (Qwen2.5-0.5B). Profiled pruning 8.9x worse than random because low mean activation correlates with specialist function. This inverts the micro finding (2.3x better than random with aux loss). The aux sparsity loss provides robustness training (redistribution), not just distribution shaping. Bimodality is architectural (transfers) but prunability is training-regime-dependent (does not transfer). Zero-shot structured pruning of SwiGLU neurons is not viable at macro scale.

### skip-list-routing-proven (2026-03-06)
Skip-list multi-resolution routing matches or beats flat softmax (-0.93% vs flat, 3 seeds). Stick-breaking confidence gates learn token difficulty: 67.2% of weight at coarsest level (1 express expert), only 4.6% reaches Level 0 (all 8 experts). Ensemble confound experimentally refuted (4x ensemble +0.59% worse than single flat, skip adaptive beats ensemble by -1.51%). Main scalability concern: 16x training FLOP cost from recursive coarse expert evaluation (all levels computed during soft routing). Hard inference routing (threshold-based early stopping) is the key untested mechanism. 780-param trainable confound between adaptive and fixed-depth controls acknowledged.

### batched-lora-latency-proven (2026-03-07)
Sequential LoRA overhead (256% at k=2) is implementation-bound, not architectural. Persistent hooks achieve 71% overhead (2.2x speedup) with near-exact numerical equivalence (3.3e-05 max diff). Hook-based 61% is numerically incorrect (3.45 max diff — different computational graph). 168 Python hooks per forward add ~31% overhead even at k=1. Path to <5%: fused CUDA kernels (S-LoRA BGMV/Punica). Memory: 18.4 MB per expert at fp32. Seq_len=31 exaggerates framework overhead; production sequences would show lower %. fp16/bf16 untested.

### channel-capacity-bound-killed (2026-03-06)
Shannon channel capacity bound for max composable experts is predictively useless. Held-out validation R²=-53.2 on N=3,4,6,7 (catastrophic). Gap is non-monotonic in N (N=5 gap < N=3 gap), falsifying the constant-interference assumption. All models (Shannon, linear, power-law) fail equally. Root cause: domain-split quality dominates; gap is not a function of N alone. To isolate N effect: compose subsets of a FIXED partition (e.g., always 8 domains, compose 2..8 of them). The MAC channel analogy is conceptually sound but empirically useless when split quality is uncontrolled.

### discriminability-drives-gradients (2026-03-07)
Expert discriminability (||f_A(x) - f_B(x)||) drives router gradients, NOT the CE gap between composed and joint models. Gap and gradient are negatively correlated. Corrected causal chain: cos -> discriminability -> gradient (r²=0.63) -> quality (r²=0.75). Phase transition at cos~0.5: gradients flat for cos<=0.5, collapse for cos>=0.7. At real scale (cos~0.0002), discriminability is always maximal -- the distinction is moot. The "gap-as-signal" framing should be "discriminability-as-signal" but practically it doesn't matter. Reviewer noted: gradient derivation incomplete (omits softmax saturation), r² on 7 points is descriptive not inferential, and N=2 mixing dynamics may not generalize to N>2 selection.

### discriminability-generalizes-to-n-gt-2 (2026-03-07)
Discriminability mechanism generalizes from N=2 to N=8 with top-k=2 selection, but attenuates: r²=0.46 (vs 0.95 at N=2), gradients 5-7x smaller. Phase transition softens (6.1x vs 19x ratio). At real scale (cos~0.0002), all experts maximally discriminable so attenuation is irrelevant. Calibration at larger N needs more steps or higher LR to compensate for k/N gradient dilution. Dense backpropagation could restore N=2 dynamics. No further micro experiments needed on discriminability branch.

### adam-cancels-gradient-attenuation (2026-03-07)
Adam's adaptive second-moment normalization (m/sqrt(v)) makes effective update magnitude invariant to gradient scaling. The 5-7x gradient attenuation at N=8 vs N=2 does NOT require LR or step compensation. Calibration protocol is trivially simple: LR=3e-3, steps=100, independent of N. Higher LR is actively harmful (especially at low N where gradient attenuation doesn't regularize). This closes the calibration branch entirely. Caveat: specific to adaptive optimizers — SGD would need LR scaling. At extreme N (1000+), Adam's eps term could become non-negligible.

### gap-as-signal-binary-only (2026-03-07)
Gap-as-signal provides ZERO discrimination within the practical regime (cos < 0.3). Within-regime r²=0.013, SNR=0.33, Cohen's d=0.24 (small), F-ratio=0.01. The 0.64pp quality range is 3x smaller than the 1.92pp noise floor. Gap-as-signal works only as binary classifier (cos>0.5 = pathological). No gap measurement needed in contribution protocol -- orthogonality guaranteed by dimensionality. The tested regime [0.0, 0.3] is 1000x wider than actual operating regime [0.0000, 0.0010], so result is conservative.
