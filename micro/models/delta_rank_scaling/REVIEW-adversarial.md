# Peer Review: Delta Rank Scaling

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that manual analysis suffices. The core claim is a 3-point power law fit on SVD effective rank ratios.

## Mathematical Soundness

### Effective rank computation: CORRECT

The implementation of Roy & Vetterli (2007) effective rank at `effective_rank()` (line 186-197) is correct:
- Singular values computed via `torch.linalg.svdvals`
- Near-zero values filtered at 1e-12
- Shannon entropy of normalized singular values
- Exponentiated to get effective rank

The `rank_at_threshold()` function (line 200-211) correctly uses cumulative Frobenius energy (sum of squared singular values), not raw singular values.

### Power law fit: TECHNICALLY CORRECT BUT MISLEADING

The log-log linear regression at `fit_power_law()` (line 455-496) is mathematically correct. However:

**Three data points cannot constrain a power law.** Any monotonic 3-point sequence will produce a reasonable R^2 in log-log space. R^2 = 0.929 with 3 points and 2 parameters (a, b) leaves 1 degree of freedom. This is not a meaningful goodness-of-fit test. A straight line (rho = c0 + c1*d), a logarithmic fit (rho = c/log(d)), or an exponential fit would all produce comparable R^2 on these 3 points. The paper acknowledges this in Limitations but the extrapolation table in Section 4.4 of MATH.md presents predictions to d=8192 with false precision (4 significant figures).

**The confidence interval on b is not reported.** With 3 points, the standard error on the slope in log-log space is enormous. A simple bootstrap or leave-one-out analysis would reveal this. The exponent b = -0.152 could plausibly be anywhere from -0.05 to -0.30.

### Convergence matching: HIDDEN ASSUMPTION

Training steps scale linearly: 1000/2000/3000 for d=64/128/256. This is a crude heuristic. Model capacity scales quadratically with d (d^2 parameters in attention, d * 4d in FFN), so linear step scaling likely under-trains larger models relative to smaller ones.

**This confound biases the result in the favorable direction.** Under-trained models have lower-rank deltas because gradient descent has explored fewer directions. The observed decline in rho(d) could be partially or entirely an artifact of under-training at larger d.

The paper lists this as Limitation #3 but does not attempt to control for it (e.g., training to a fixed validation loss target, or measuring rho at multiple training checkpoints to assess convergence).

### Averaging across weight types: METHODOLOGICAL CONCERN

The aggregate "mean ratio" averages FFN, attention, and embedding ratios with equal weight. But these weight types have very different dimensions and behaviors:
- Embedding weights (V=27 x d): min_dim = V = 27 for all d, so ratio is not a function of d at all. Including them contaminates the d-scaling signal.
- Attention weights (d x d): square matrices, ratio is a function of d.
- FFN weights (4d x d): rectangular, ratio is a function of d.

The PAPER.md correctly separates these in the "By Weight Type" table, but the kill criteria are evaluated on the mean across all types, including the noise from embeddings. At d=256, embeddings contribute an upward bias of 0.829 against the FFN (0.643) and attention (0.431) that actually carry the scaling signal. Excluding embeddings, the mean ratio at d=256 would be approximately (0.643 + 0.431) / 2 = 0.537, and at d=128 approximately (0.727 + 0.546) / 2 = 0.637. The K1 kill at d=256 would be borderline rather than clear.

This does not change the direction of the finding but it inflates the absolute ratio values.

## Novelty Assessment

### Prior art: This is well-trodden ground

The experiment cites Aghajanyan et al. (2021), "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning," which already established that intrinsic dimensionality of fine-tuning tasks grows sublinearly with model size. That paper measured the intrinsic dimension needed for 90% of fine-tuning performance and found it scales as roughly d^0.5 to d^0.7 (sublinear).

The experiment also cites arXiv:2510.00537 on spectral scaling laws, which already provides the hard/soft rank distinction and power-law exponents.

**The delta here over prior work is small.** This experiment measures r_eff of the *pretraining* delta (W_trained - W_random_init) rather than the *fine-tuning* delta. That is a meaningful distinction for the base-as-adapter architecture, but the paper does not clearly articulate why pretraining deltas should behave differently from fine-tuning deltas, and whether the cited literature already covers this case.

### No reinvention detected

The code is original and does not duplicate anything in `references/`. The `references/delta-compression-models/` folder covers BitDelta and DeltaZip (compression methods), not rank scaling measurements.

## Experimental Design

### Does the experiment test the hypothesis? PARTIALLY

The hypothesis is: "rho(d) decreases as d increases, making base-as-adapter more practical at macro scale." The experiment measures rho at 3 values of d and observes a decrease. This is directionally supportive.

However, the experiment conflates two variables:
1. **Model dimension d** (the variable of interest)
2. **Model capacity relative to task complexity** (a confound)

At d=256, the model has 3.2M parameters learning a character-level name distribution with V=27. This model is massively over-parameterized for the task. At d=64 (203K params), it is already over-parameterized but less so. The declining rank ratio may reflect increasing over-parameterization rather than a fundamental property of how weight deltas scale with dimension.

At macro scale (d=4096, V=151K, internet-scale training data), the model is NOT over-parameterized. The scaling trend could reverse.

### Controls: INSUFFICIENT

There are no ablation controls for the convergence confound. Specifically:
- No training-to-convergence condition (train all models to the same validation loss)
- No multi-checkpoint analysis (measure rho at 25%, 50%, 75%, 100% of training)
- No task-complexity scaling (increase dataset complexity with d)

The 3-seed replication is good for measuring variance at each d but does not address systematic bias.

### Kill criteria design: K1 THRESHOLD IS ARBITRARY

The K1 threshold of 0.5 for Shannon effective rank has no theoretical grounding. The paper retroactively argues that r_95 (practical rank) is more relevant and has a lower ratio, which is true but was not the pre-registered criterion. Changing the metric post-hoc to avoid a kill is a red flag, even though the reasoning is sound.

The paper is transparent about this ("K1 was defined before we understood the distinction"), which is honest, but the correct response is to acknowledge the kill and register a new experiment with the revised metric, not to interpret the kill as "technical rather than substantive."

## Hypothesis Graph Consistency

The `exp_delta_rank_scaling` node in HYPOTHESES.yml has:
- K1: "r_needed/d ratio stays above 0.5 at d=128 and d=256" -- KILLED (both > 0.5)
- K2: "larger models show higher effective rank ratio than smaller ones" -- SURVIVES

Status is correctly recorded as `weak_kill`. The experiment directory matches. No conflicts with other nodes. This experiment does not block anything in the current roadmap, which is appropriate given its informational role.

## Macro-Scale Risks (advisory)

1. **Over-parameterization confound reverses.** At macro scale, models are capacity-limited, not over-parameterized. The rank ratio could increase rather than decrease. The literature (Aghajanyan 2021) measured fine-tuning deltas on real tasks and found sublinear scaling, which is encouraging, but pretraining deltas on internet-scale data are a different object.

2. **The r_95 extrapolation to rank-500 at d=4096 is the load-bearing claim.** This extrapolation is based on 3 data points of r_95 ratio (0.455, 0.415, 0.320) without a formal power-law fit. The paper approximates the exponent as "d^(-0.25) approximately" without reporting the fit quality. This is the number that feeds into the feasibility assessment of VISION.md (5% of model size for base adapter).

3. **The rank-500 cost estimate in PAPER.md Section "Revised Feasibility Assessment" uses d=4096 as Qwen 7B.** Qwen 2.5-7B actually has d=3584, not d=4096. At d=3584, the predicted rank would be slightly different. This is a minor error but shows the extrapolation chain is fragile.

4. **Cheapest validation:** Measure r_eff on the actual delta between a fine-tuned Qwen 0.5B (d=896) and its base weights. This costs nothing (download two checkpoints, compute SVD). If rho(896) > 0.5, the power law is wrong. The paper correctly identifies this as the next step.

## Verdict

**REVISE**

The experiment provides useful directional evidence (the ratio does decrease monotonically, with very high effect size). The mechanism is sound in principle. However, the claims overreach the evidence, and one specific confound (under-training at larger d) could explain the entire effect. Specific fixes:

1. **Add a convergence control.** Train all three model sizes to the same validation loss (not the same number of steps). Re-measure rho. If the trend holds, the convergence confound is eliminated. This is cheap -- just extend training at d=256 until loss matches d=64, then re-run the SVD analysis.

2. **Exclude embeddings from the aggregate ratio used for kill criteria.** Embeddings are bounded by V=27 and do not scale with d. The scaling analysis should report "FFN+Attention mean ratio" as the primary metric. This changes the narrative from "K1 killed" to "K1 killed by a wider margin once embedding noise is removed" (since embedding ratios pull the aggregate down at small d and up at large d -- wait, at d=256 embeddings are 0.829 which pulls UP, making K1 worse). Actually, excluding embeddings makes the d=256 ratio LOWER (0.537 vs 0.538), so this is a minor point. Keep the current reporting but add the FFN+Attn-only aggregate as the primary scaling metric.

3. **Report confidence intervals on the power law exponent.** With 3 points this will be wide, which is honest. Remove the 4-significant-figure extrapolation table or add error bars. A simple parametric bootstrap (resample from the per-seed ratios) would suffice.

4. **Do NOT retroactively reinterpret K1.** The Shannon ratio > 0.5 criterion was pre-registered. It was killed. Accept the kill. If r_95 is the better metric, register a new hypothesis node with r_95-based kill criteria and test it. The current WEAK_KILL status is appropriate but the paper's framing of "technical rather than substantive" kill is special pleading.

5. **Add multi-checkpoint rho measurement (optional but valuable).** Measure rho at 25%, 50%, 75%, 100% of training for each d. If rho is still declining with training steps at the final checkpoint, the model is under-trained and the inter-d comparison is contaminated. If rho plateaus before the final checkpoint, convergence is confirmed.
