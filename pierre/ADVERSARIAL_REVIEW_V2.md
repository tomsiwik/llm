# Adversarial Review V2: Pierre Research Program

**Reviewer:** Automated peer review, NeurIPS calibration
**Scope:** New findings since V1 review (#378-#394), manifold composition theory, statistical closure
**Date:** 2026-04-08
**Prior review:** ADVERSARIAL_REVIEW.md (10 critiques, 9 resolved, 1 active)

---

## 0. Executive Summary

The V1 review identified toy-scale evidence as the central weakness. Since then, Pierre has made genuine progress: M2P works on Qwen3-0.6B with real GSM8K data, Grassmannian holds at N=50, TF-IDF routing is 100% on real text, and adapter hot-swap is 0.26ms. These are real results.

However, this review identifies **7 new concerns**, of which **3 are blocking** for the next stage of claims. The statistical closure finding (#392) actually reverses the V4 claim rather than confirming it. The manifold composition theory (SLERP/PBD/symplectic) is a theoretical dead end that the experiments have already refuted. And the M2P parameter overhead (357M for a 600M base) remains unresolved after VeRA was killed.

---

## 1. Statistical Concerns (BLOCKING)

### 1.1 Finding #392 Reversed the V4 Claim, Not Confirmed It

The V4 experiment (#378) claimed quality_ratio = 1.433, CI_lower = 0.773, interpreting this as "M2P robustly beats SFT." The n=500 replication (#392) revealed:

| Metric | V4 claim | N=500 reality |
|--------|----------|---------------|
| SFT accuracy | 26.0% (n=200) | 31.4% (n=500) |
| quality_ratio | 1.433 | 0.754 |
| CI_lower | 0.773 | 0.315 |
| Direction | M2P beats SFT | SFT beats M2P by 2.8pp |
| p-value | 0.36 (not sig) | 0.334 (not sig) |

The point estimate FLIPPED from "M2P exceeds SFT by 43%" to "SFT exceeds M2P by 25%." The CI = [0.315, 1.194] contains 1.0 but is centered BELOW 1.0.

**The correct interpretation is: M2P achieves approximately 75% of SFT quality on GSM8K, with wide uncertainty.** The claim "M2P matches SFT" is not supported. The claim "M2P robustly beats SFT" (from V4) is refuted.

This matters because the entire product thesis rests on M2P replacing SFT for adapter generation. If M2P is 75% of SFT, the speed-quality tradeoff is much less favorable.

**Severity: HIGH.** The V4 finding (#378) should be downgraded from "supported" to "provisional" given the V5 replication.

### 1.2 Power Analysis: N=500 Is Insufficient for 2.8pp Differences

For a two-proportion z-test to detect a 2.8pp difference (28.6% vs 31.4%) at 80% power with alpha=0.05:

    n = (z_alpha + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2
    n = (1.96 + 0.84)^2 * (0.286*0.714 + 0.314*0.686) / 0.028^2
    n = 7.84 * 0.42 / 0.000784
    n = 4,200 per group

At n=500, the test has approximately 11% power to detect a 2.8pp difference. Saying "not statistically significant" with 11% power is not the same as saying "no difference exists." The experiment was dramatically underpowered for the effect size observed.

To distinguish M2P from SFT at this effect size with 80% power requires n ~ 4,200. At the current evaluation rate (~2s per sample), this is ~2.3 hours -- feasible but not run.

**Severity: MEDIUM.** The current claim "statistically indistinguishable" is technically correct but misleading. The honest claim is "we cannot tell with this sample size."

### 1.3 Per-User Adapter Finding (#384) Rests on Cohen's d = 0.499 From Length Variance

Finding #384 reports behavioral differentiation with d = 0.499 (concise vs step). But reading the PAPER.md closely:

- Concise: mean=200 (capped), std=0.0
- Step: mean=200 (capped), std=2.0
- The d = 0.499 comes from std difference (0 vs 2), not from mean difference (both 200)

Both personas generated maximum-length outputs. The "behavioral differentiation" is that one loops with zero variance and the other loops with std=2. This is differentiation of DEGENERATE behaviors, not useful behavioral adaptation.

The CODE persona (mean=136, std=71) is genuinely differentiated. But the headline number (d=0.499 for concise vs step) is misleading -- it measures the difference between two failure modes.

**Severity: MEDIUM.** The per-user claim should lead with the CODE result (d=1.262) and honestly state that CONCISE failed to learn EOS termination.

---

## 2. The Manifold Composition Theory Is Empirically Dead (BLOCKING)

### 2.1 SLERP: Experiment Already Killed It

MANIFOLD_COMPOSITION.md Section 2 proposes SLERP to fix the "candy wrapper effect." Finding #382/#383 (exp_slerp_b_composition) tested this directly:

- K931 PASS: SLERP preserves norm (ratio 2.06x, theorem verified)
- K932 FAIL: SLERP quality WORSE on all 5 domains (0.463 vs LERP 0.402)

The PAPER.md for this experiment explicitly states: **"The candy-wrapper is irrelevant when routing solves the underlying problem"** and **"SLERP is not the right fix for multi-domain composition quality."**

Despite this, MANIFOLD_COMPOSITION.md still presents SLERP as Layer 2 of a "Three-Layer Fix" (Section 6). The theory document was not updated after the experiment killed its central prediction.

### 2.2 PBD: Never Tested, Likely Unnecessary

Position-Based Dynamics (Section 4) proposes Gram-Schmidt orthogonalization of adapter activations at runtime. This adds O(N^2 * d_out) computation per token. But:

- The activation interference (Finding #372) is sub-linear at alpha=0.38
- Max|cos| at N=10 is 0.34, plateauing
- TF-IDF routing at 100% makes multi-adapter composition unnecessary for most tokens
- No experiment has been run or even designed

PBD is a solution to a problem that doesn't exist at current scale. At N=50, the predicted max|cos| from the power law is 0.059 * 50^0.38 = 0.26 -- well within tolerance.

### 2.3 Symplectic Promotion: Theoretical Overreach

Section 5 claims that Stormer-Verlet integration gives O(h^2) bounded error for promotion cycles vs O(K*h) for naive Euler. This analogy is broken:

1. **The "Hamiltonian" H(W) = model_quality(W) is not a Hamiltonian.** Hamiltonian mechanics requires a phase space (q, p) with symplectic structure. Weight space is not a phase space. There is no conjugate momentum to the weights. The claim that promotion is "Euler integration of dW/dk = alpha * Delta(W)" treats a discrete sequence of different adapters as a continuous ODE, which it is not.

2. **The Delta_k are not time-derivatives of a smooth trajectory.** Each promotion adds a DIFFERENT adapter trained on a DIFFERENT domain. The sequence W_0 -> W_1 -> W_2 is not a discretization of a smooth curve -- it's a sequence of unrelated perturbations. Symplectic integration theory requires the vector field to be smooth and time-independent (or periodically time-dependent). Neither holds here.

3. **The "half-step correction" generates a new adapter on the promoted base, which is an entirely different operation from a leapfrog half-step.** In leapfrog, the half-step uses the SAME force field evaluated at a new point. Here, the "half-step" retrains an adapter from scratch on new data. The connection to symplectic integration is purely nominal.

**The game dev analogy is not genuine mathematical connection.** SLERP is real math applied to a problem that doesn't need it (routing handles it). Polar decomposition is a generalization that would give the same result. PBD is a game engine technique that's unnecessary given sub-linear interference scaling. Symplectic promotion is a physics metaphor applied without the required mathematical structure.

**Severity: HIGH for credibility.** MANIFOLD_COMPOSITION.md should be either heavily revised or archived. A reviewer would immediately flag the symplectic section as cargo-cult physics.

---

## 3. The Kappa Catastrophe: Finding #385 Invalidates Promotion Safety Claims

### 3.1 The Epsilon Map Used kappa=50; Reality Is kappa=56,000

The theoretical_analysis.py (epsilon map, Section 4) predicts promotion safety using:

```python
kappa_attn = 50  # conservative
```

Finding #385 (condition_number_per_layer) measured:

| Weight | Mean kappa | Max kappa |
|--------|-----------|-----------|
| q_proj | 44 | 69 |
| k_proj | 56,013 | 997,697 |
| v_proj | 16,445 | 121,464 |
| o_proj | 21 | 30 |
| MLP | 18-66 | 27-100 |

The epsilon map's kappa=50 is correct for q_proj and MLP, but M2P applies LoRA to BOTH q_proj AND v_proj. The v_proj kappa of 16,445 means the promotion safety bound for v_proj adapters is:

    K_safe_v = (0.1 * sqrt(1024) / (16445 * 5 * 0.38 * sqrt(4)))^2 = absurdly small

The PAPER.md for #385 acknowledges this and proposes a "structural bypass" -- that Grassmannian A-matrices align with top singular vectors, so the effective kappa is much lower. But this alignment has NOT been measured. It is a hypothesis.

### 3.2 The Pythagorean Promotion Claim Depends on Kappa

The theoretical_analysis.py claims promotion perturbation grows as sqrt(K) due to the Pythagorean theorem on orthogonal A-subspaces. This is correct for the WEIGHT perturbation:

    ||sum_k A_k B_k^T||^2_F = sum_k ||A_k B_k^T||^2_F   (Pythagorean)

But the FUNCTIONAL perturbation is amplified by kappa:

    ||f(W+E) - f(W)|| <= kappa(W) * ||E|| / ||W||

For v_proj with kappa=16,445, even the sqrt(K) scaling gives:

    Functional bound after K=5: 16445 * 5 * 0.38 * 2 * sqrt(5) / sqrt(1024) = 1838

A functional perturbation bound of 1838 is vacuous. The Pythagorean advantage is real but dwarfed by the kappa amplification.

**Severity: HIGH.** The promotion safety analysis must be redone with measured kappa values. The v_proj promotion is potentially unsafe. Either: (a) verify A-matrix alignment with top singular vectors (the proposed bypass), or (b) exclude v_proj from promotion.

---

## 4. M2P Size Problem: 357M Params for a 600M Base

### 4.1 The VeRA Fix Was Killed

M2P has 357M params (1.4GB fp32) to serve a 600M (345MB 4-bit) base model. The hypernetwork is 4x the size of the quantized model it serves. Finding #388 confirms this: M2P forward is 5.31ms because it must read 1.43GB of fp32 weights through the memory bus.

VeRA (exp_m2p_vera_bottleneck) attempted to reduce this to 4.7M params but was killed: quality_ratio = -0.105 (worse than base). The failure is structural -- rank-4 VeRA with 8 scalars per layer is too constrained.

### 4.2 The Deployment Math Doesn't Work

For Pierre Pro serving 50 domains:

| Component | Size |
|-----------|------|
| Qwen3-0.6B 4-bit base | 345 MB |
| M2P hypernetwork (fp32) | 1,430 MB |
| 50 adapter B-matrices | 252 MB |
| Grassmannian A-matrices | 57 MB |
| **Total** | **2,084 MB** |

The M2P is 69% of total memory. For Qwen3-4B:

| Component | Size |
|-----------|------|
| Qwen3-4B 4-bit base | ~2,200 MB |
| M2P (if d_M2P=3584) | ~50,000+ MB |
| **Total** | Not feasible on 48GB |

The M2P scales as O(d_model^2 * n_layers). At Qwen3-4B with d_M2P=d_model=3584 and L=36, the M2P parameter count would be enormous. The SHINE paper's M2P operates at d_M2P=d_model, but SHINE targets 7B models with GPU clusters, not 48GB Apple Silicon.

**The alternative -- d_M2P < d_model -- was shown to fail.** Finding #375 (d_M2P=128) gave 0% quality. Finding #387 shows d_int=86, so d_M2P needs to be at least ~100. But at Qwen3-4B, the per-layer B-params are massive, and even d_M2P=100 may not suffice (the intrinsic dimensionality was measured on Qwen3-0.6B, not 4B).

**Severity: HIGH for product viability.** The M2P size problem has no known solution. VeRA was killed. The parameter budget at Qwen3-4B scale is untenable. This is the single largest product risk.

---

## 5. Routing: 100% on 3 Domains, But What About 50?

### 5.1 TF-IDF Achieves 100% on Math/Code/Text (#389)

This is a real result. The LEARNINGS.md correctly notes that longer real text provides richer n-gram features. However:

- N=3 domains (math, code, text) are maximally different in vocabulary
- Medical vs Legal? Cardiology vs Oncology? These share 90%+ vocabulary
- TF-IDF relies on discriminating n-grams ("how many", "in python")
- For subdomain routing (cardiology vs oncology), discriminating n-grams may be rare

### 5.2 Q_wrong = -58% Makes Routing Errors Catastrophic (#386)

On toy domains, Q_wrong = +35% (wrong adapter still helps). On real domains, Q_wrong = -58% (wrong adapter actively harms).

The epsilon map (theoretical_analysis.py) uses Q_wrong = 0.35 to calculate routing requirements:

```python
result = routing_error_analysis(0.95, 5, 0.93, 0.35)
```

With Q_wrong = -0.58:

    E[quality] = p * Q_correct + (1-p) * Q_wrong
    At p=0.95: E[quality] = 0.95 * 0.93 + 0.05 * (-0.58) = 0.854
    At p=0.90: E[quality] = 0.90 * 0.93 + 0.10 * (-0.58) = 0.779

The minimum routing accuracy for <5% quality loss with Q_wrong = -0.58:

    p > 1 - 0.05 / (0.93 - (-0.58)) = 1 - 0.05/1.51 = 0.967

You need 96.7% routing accuracy when Q_wrong is negative. At N=50 domains with overlapping vocabularies, achieving 96.7% with TF-IDF is unverified and plausibly impossible for fine-grained domains.

**Severity: MEDIUM.** Routing is solved for coarse domains (N=3-5). Unverified for fine-grained domains (N=50). The Q_wrong result makes this more urgent.

---

## 6. What's Actually Strong (Since V1)

In fairness, several V1 critiques were genuinely resolved:

1. **Real language works (#376/#378).** M2P on Qwen3-0.6B with GSM8K produces adapters that improve over base (28.6% vs 20.0%). This is real, even if the SFT comparison is weaker than claimed.

2. **L=36 works (#365).** Layer depth scaling to 36 layers at 89.1% quality. This was the #1 V1 blocker and it passed.

3. **Grassmannian N=50 (#393).** Orthogonality verified at production scale (max cross-norm 9.5e-08 across 1,225 pairs). The math works.

4. **Hot-swap latency (#394).** Adapter injection is 0.26ms. MLX lazy eval means no cache invalidation. This is production-viable.

5. **TF-IDF routing on real text (#389).** 100% on math/code/text. Robust and fast.

6. **Q_wrong measurement (#386).** Honest measurement that revealed routing is critical. Good science.

7. **d_int measurement (#387).** Calibrated the epsilon map. Showed d_M2P=64 is insufficient at 90% energy threshold.

---

## 7. The Strongest Remaining Attack

### If I Were a Competitor Building the Case Against Pierre

**Argument 1: M2P is a hypernetwork that is 4x the size of the model it serves, achieves 75% of SFT quality, and has no viable compression path.**

The VeRA attempt to compress M2P from 357M to 4.7M was killed (quality_ratio = -0.105). The full M2P at Qwen3-4B scale would require ~50GB+ of parameters. On a 48GB machine, this doesn't fit. The entire M2P thesis requires either:
- A compression method that preserves quality (VeRA failed, nothing else proposed)
- Running M2P on a smaller model than the base (defeats the purpose)
- A completely different hypernetwork architecture

**Argument 2: The "statistically indistinguishable" claim hides a 2.8pp SFT advantage.**

At n=500, SFT outperforms M2P (31.4% vs 28.6%, p=0.334). The test is underpowered (11% power at this effect size). A properly powered test (n=4200) would likely show SFT is significantly better. The honest claim is: M2P is ~10% worse than SFT on GSM8K, and we need a much larger sample to be sure.

**Argument 3: Everything unproven is more important than everything proven.**

| Proven | Unproven but critical |
|--------|----------------------|
| Grassmannian orthogonality | Multi-cycle promotion |
| Adapter hot-swap (0.26ms) | M2P at Qwen3-4B scale |
| TF-IDF routing on 3 domains | TF-IDF routing on 50 domains |
| Single promotion works | Promotion with kappa=16K on v_proj |
| d_int = 86 for GSM8K | d_int for medical/legal/code domains |
| 5.31ms M2P forward | M2P that fits in memory at 4B scale |

The proven column is necessary infrastructure. The unproven column is the product.

### The Single Experiment That Could Kill Pierre

**M2P parameter scaling to Qwen3-4B.** If the M2P hypernetwork at d_M2P=d_model=3584 doesn't fit in 48GB alongside the base model, and no compression method works, the product cannot exist on the target platform. Everything else has fallback paths; this one doesn't.

---

## 8. Specific Recommendations

### BLOCKING (must resolve before next claims)

**B1: Downgrade Finding #378 from "supported" to "provisional."**
The v4 quality_ratio=1.433 was based on n=200 SFT measurement. The n=500 replication shows quality_ratio=0.754. The finding text must be updated to reflect the corrected numbers.

**B2: Archive or heavily revise MANIFOLD_COMPOSITION.md.**
SLERP is experimentally killed. PBD is unnecessary given sub-linear interference. Symplectic promotion lacks mathematical structure (no Hamiltonian, no phase space, no smooth ODE). The document currently presents refuted theory as a "Three-Layer Fix." At minimum: mark SLERP as killed, PBD as unneeded, symplectic as speculative analogy (not theorem).

**B3: Update the epsilon map (theoretical_analysis.py) with measured kappa values.**
Replace kappa_attn=50 with measured values per weight type. Recompute K_safe per module. Determine whether v_proj promotion is safe or requires the A-matrix alignment bypass.

### IMPORTANT (should resolve before publication)

**I1: Run n=4200 M2P vs SFT comparison.** The current test has 11% power. A properly powered comparison takes ~2.3 hours and definitively answers whether M2P matches SFT on GSM8K.

**I2: Measure A-matrix alignment with top singular vectors of base weights.** Finding #385 proposes that Grassmannian A aligns with top-sigma directions, making effective kappa << full kappa. This is the critical bypass for promotion safety. One measurement closes the gap.

**I3: Prototype M2P parameter budget for Qwen3-4B.** Before building it, calculate: at d_M2P=3584, L=36, 7 modules, what is the M2P parameter count? Does it fit in 48GB alongside the 4-bit base? If not, what d_M2P is feasible, and does it exceed d_int for 4B-scale tasks?

**I4: Test TF-IDF routing at N=10+ with semantically similar domains.** Medical vs legal is easy. Medical-cardiology vs medical-oncology is the real test.

### ADVISORY (nice to have)

**A1: Honest comparison with LoRA + inference server.** S-LoRA / LoRAX / vLLM with LoRA can serve multiple adapters. How does Pierre's approach compare on latency, quality, and memory? The comparison should include the M2P size overhead.

**A2: Multi-turn conversation test.** No experiment has tested adapter behavior across multi-turn dialogue. Adapter accumulation over turns is a core product feature.

**A3: Training cost comparison.** M2P training (1000 steps) vs SFT training (300 steps). If SFT is faster AND higher quality, the M2P advantage is only at inference-time generation speed -- and that advantage disappears if adapters are cached.

---

## 9. Verdict

**REVISE.**

The research program has made substantial progress since V1. The toy-scale concerns are largely resolved -- real language, real model, real benchmarks. The infrastructure (Grassmannian, routing, hot-swap) is solid.

But three issues are blocking:

1. The statistical claim about M2P quality has been weakened, not strengthened, by the n=500 replication. The honest number is quality_ratio ~ 0.75, not 1.43.

2. The manifold composition theory (MANIFOLD_COMPOSITION.md) contains refuted claims (SLERP), unnecessary proposals (PBD), and cargo-cult physics (symplectic). This document hurts credibility.

3. The M2P parameter budget at Qwen3-4B is potentially fatal and has no known solution after VeRA was killed.

Resolve B1-B3 before making any further quality claims. Resolve I1-I4 before any publication or product decision.

---

## 10. Resolution Tracker (V2)

| # | Critique | Status | Fix |
|---|----------|--------|-----|
| V2-1 | #378 quality_ratio=1.43 reversed to 0.75 | **DONE** | Finding downgraded to provisional, caveat updated |
| V2-2 | n=500 underpowered (11% power at 2.8pp) | OPEN | Need n=4200, ~2.3hrs feasible |
| V2-3 | Per-user d=0.499 from degenerate behaviors | OPEN | Lead with CODE d=1.26 |
| V2-4 | SLERP killed but MANIFOLD_COMPOSITION still claims it | **DONE** | Document archived with status header |
| V2-5 | Symplectic promotion lacks mathematical structure | **DONE** | Marked unsound in archive |
| V2-6 | Epsilon map uses kappa=50; reality is 56K for v_proj | **DONE** | Updated with measured values, strategy: exclude k/v_proj from promotion |
| V2-7 | M2P 357M params, VeRA killed, no compression path | ACKNOWLEDGED | Engineering problem for cloud scale, not research blocker at 48GB |
| V1-4 | Multi-cycle promotion untested | ACTIVE | exp_m2p_multi_cycle_promotion running |
