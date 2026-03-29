# Peer Review: Flat-LoRA Training

## NotebookLM Findings

Skipped -- the experiment is a clean negative result with straightforward math. Deep review not warranted for a KILL verdict that the authors themselves reach.

## Mathematical Soundness

### What holds

1. **SAM formulation is correct.** The minimax formulation (Section 1.2), the first-order epsilon approximation, and the two-pass gradient procedure all match Foret et al. (2010.01412). The implementation in `train_sam()` (lines 308-432 of run_experiment.py) faithfully implements this: compute gradient, normalize, perturb, compute second gradient, restore, update.

2. **The merge perturbation bound is correct in form.** The Taylor expansion argument (Section 2.2) that |L(theta + delta) - L(theta)| <= lambda_max * ||delta||^2 / 2 is standard second-order analysis (though technically an approximation, not a bound, unless the loss is quadratic or the Hessian is bounded uniformly).

3. **The post-hoc analysis (Section 9) is the strongest part.** The argument that Grassmannian orthogonality makes ||delta_i_projected|| ~ 0, rendering the distinction between sharp and flat minima irrelevant, is logically sound given the measured cos=0.001.

### What does not hold

1. **Misattribution of orthogonality to "Grassmannian skeleton."** This is the most significant error in the paper. The code uses `mx.random.uniform` initialization for lora_a (line 186) and trains BOTH lora_a and lora_b (line 550: `module.unfreeze(keys=["lora_a", "lora_b"])`). There is no Grassmannian construction anywhere in this experiment. The A-matrices are not frozen, not sampled from a Grassmannian manifold, and not shared across a skeleton.

   The observed cos=0.001 is a high-dimensional geometry effect. With ~17.2M parameters per adapter, two independent random vectors in R^{17.2M} have expected |cos| ~ 1/sqrt(17.2M) ~ 0.00024. The observed 0.001 is 4x higher (training correlation), but still near-zero by construction. The PAPER.md claims "the Grassmannian skeleton already solves the problem" -- but this experiment has no Grassmannian skeleton. It has random-init trained adapters that happen to be near-orthogonal because they live in a 17M-dimensional space.

   This distinction matters: the Grassmannian skeleton in VISION.md is a deliberate architectural choice (frozen A-matrices sampled from Gr(r,d)). This experiment tests something different -- independently trained full-LoRA adapters -- and the orthogonality mechanism is different (random vector concentration vs. deliberate construction).

2. **The cosine metric measures the wrong thing.** The orthogonality is computed on concatenated (lora_a, lora_b) vectors across all layers (lines 835-836). This is NOT the same as the cross-term A_i^T @ A_j that matters for composition. The theoretically relevant quantity for Task Arithmetic merge interference is:

       interference_ij = || (B_i @ A_i)^T (B_j @ A_j) ||_F

   or at minimum, per-layer A_i^T @ A_j. Concatenating all parameters into a single vector and computing cosine is a much weaker statement. Two adapters could have cos(flat_params)=0.001 but have high per-layer interference in the layers that matter most.

3. **The Hessian is never measured.** The core claim of SAM is that it reduces lambda_max (maximum Hessian eigenvalue). The experiment measures "sharpness" via random perturbation PPL change (lines 663-715), which is a proxy. But with perturbation scale 0.01 * RMS(param) and only 5 trials, this does not reliably estimate curvature -- it estimates a noisy average over random directions, dominated by the median eigenvalue, not lambda_max. SAM's benefit is specifically on the maximum eigenvalue direction. The "sharpness" measurement is too coarse to detect SAM's effect.

4. **Worked example (Section 6) uses invented numbers.** Lambda_1 = 50 for standard vs. 5 for SAM (10x reduction) is not justified by any measurement or citation. The actual sharpness measurements show NO difference between methods (both < 0.1%). The worked example overstates the expected effect by orders of magnitude.

## Novelty Assessment

**Low novelty, but that is acceptable for a mechanism-testing experiment.**

- SAM for LoRA merging has been explored by Sun et al. (2409.14396), which this experiment cites. The experiment is testing whether that result transfers to the ternary/Grassmannian setting, not claiming novelty.
- The negative result (orthogonality makes flatness irrelevant) is genuinely informative for the project's architectural story.
- No reinvention of existing code -- the experiment correctly builds on the bitnet_2b_real_composition pipeline.

## Experimental Design

### What works

1. **Controlled comparison.** Same model, same data, same hyperparameters (except SAM perturbation). Same random seed progression. This is a clean A/B test.
2. **Multiple merge methods tested.** Task Arithmetic, TIES, DARE, and Direct Sum cover the relevant merge strategy space.
3. **Training time overhead measured.** 1.95x matches the theoretical 2x, confirming the implementation is not doing wasted work.

### What does not work

1. **The verdict logic is wrong.** Line 882: `elif results["s1_pass"]: verdict = "SUPPORTED" else: verdict = "SUPPORTED"` -- the code always returns SUPPORTED if K1 and K2 pass. K2 is defined as "best_delta > 0" (any positive number). So 0.01pp improvement counts as SUPPORTED. The PAPER.md correctly calls this "practically FAIL" but the code's verdict is "SUPPORTED," creating a contradiction between the results.json (`"verdict": "SUPPORTED"`) and the paper (`"Status: KILLED"`).

2. **Two of five standard-trained adapters did not converge.** Python: last_50_loss (1.12) > first_50_loss (1.03). Creative: last_50_loss (1.58) > first_50_loss (1.17). These adapters got WORSE during training. SAM shows the same pattern for the same domains. This means both methods fail to train python and creative adapters in 200 steps -- which should be reported as a confound. The merge comparison is between two sets of partially-failed adapters.

3. **No Grassmannian baseline.** The paper claims Grassmannian orthogonality renders SAM unnecessary, but the experiment uses random init with trained A-matrices, not frozen Grassmannian A-matrices. To make the stated claim, you would need to compare: (a) Grassmannian-frozen-A + standard training vs. (b) Grassmannian-frozen-A + SAM training. The current experiment tests random-init-trained-A, which is a different regime.

4. **Val set is only 25 samples per domain.** With 5 domains x 25 samples, the merge PPL comparison is based on ~125 total evaluation points. The standard error on PPL is large enough that 0.07pp is undetectable as signal.

5. **Sharpness perturbation scale is arbitrary.** Using 0.01 * RMS(param) for perturbation (line 689) is not motivated. The relevant perturbation magnitude for composition is ||sum_j B_j @ A_j|| projected onto each adapter's subspace. No analysis connects the 1% random perturbation to the actual merge perturbation magnitude.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment concludes this is a dead end. No macro follow-up recommended.

If anyone revisited this:
- At N=50+ adapters, the merge perturbation per adapter grows. Even with near-orthogonality, the accumulated interference could be large enough for SAM to matter.
- Full weight-space SAM (Flat-LoRA proper) was not tested and could give different results than LoRA-space SAM.
- With frozen Grassmannian A-matrices, the theoretical argument is cleaner but the practical effect may differ.

## Verdict

**PROCEED** (as a negative result)

The conclusion is directionally correct: SAM provides no merge benefit when adapters are already near-orthogonal. The experiment runs cleanly, the code is correct, and the A/B comparison is fair.

However, the paper requires corrections before being trusted as evidence:

### Required fixes (for FINDINGS.md / PAPER.md accuracy)

1. **Remove all "Grassmannian skeleton" attribution.** This experiment does not use Grassmannian init or frozen A-matrices. The orthogonality is from high-dimensional random vector concentration. Replace "Grassmannian orthogonality already ensures flat merge landscape" with "high-dimensional adapter parameter space produces near-orthogonal adapters under independent random initialization and training." This is a weaker but honest claim.

2. **Fix results.json verdict.** The code outputs `"verdict": "SUPPORTED"` but the paper says KILLED. The code's verdict logic (lines 875-883) is broken -- it returns SUPPORTED whenever K2 passes, but K2 passing at 0.07pp is meaningless. Change the code or manually correct results.json to match the paper's KILLED conclusion.

3. **Acknowledge non-convergence.** Two of five domains (python, creative) did not converge under either method. This should be noted as a limitation -- the comparison is between partially-trained adapters.

4. **Weaken the architectural implication.** The paper claims "No training-time technique can improve on this for orthogonal adapters." This is too strong. The experiment tested one technique (LoRA-space SAM) at one scale (200 steps, 5 domains). It does not rule out other training-time techniques or full weight-space SAM.

### What this tells us about the mechanism story

The key finding is sound despite the attribution error: independently trained adapters at this parameter count are naturally near-orthogonal, and this near-orthogonality makes merge quality insensitive to loss landscape curvature. This is informative for the SOLE architecture -- it suggests that the Grassmannian skeleton's value may lie more in guaranteeing orthogonality than in improving it beyond what random chance provides. This is worth investigating separately.
