# Learnings: Adapter Promotion via NRE Composition (KILLED)

## Core Finding

**At scale=20, LoRA deltas become large enough that the linear regime assumption fails.** Individual adapters degrade their own domains (medical: -72.8% PPL). NRE composition cannot preserve the benefit of adapters that provide no benefit. The experiment repeats Finding #330's catastrophic failure at scale=20, with the additional insight that NRE composition of harmful deltas produces marginally less harm.

---

## Why This Happened

### 1. Linearization Fails in the Nonlinear Regime

The mathematical framework (MATH.md Section III) assumes:
```
ΔL_i ≈ -⟨∇_W L_i, ΔW⟩    [first-order Taylor approximation]
```

This approximation is valid when ‖ΔW‖ is small relative to the curvature of the loss landscape. At scale=20:

- **Medical adapter alone:** Degrades medical domain PPL from 6.107 → 10.553 (-72.8% improvement)
- **This means the adapter is harmful in isolation**, not beneficial
- The first-order approximation predicts benefit (η ≥ 0.45), but the actual loss landscape is dominated by **second- and higher-order terms**

**Literature support:**
- Flat-LoRA (ICML 2025) shows solutions that appear optimal in LoRA weight space can be in sharp regions of the full parameter space, breaking linear approximations
- SubLoRA (arXiv) demonstrates second-order (Hessian-based) analysis is necessary when first-order linearization fails

### 2. Precondition Violation: Adapters Must Be Beneficial

The entire NRE retention lemma (MATH.md Section II) is predicated on:
- Each adapter ΔW_i has a positive gradient alignment: ⟨∇L, ΔW_i⟩ < 0 (reduces loss)
- Composition preserves a fraction η of this benefit

**What actually happened:**
- Medical adapter has ⟨∇L, ΔW_medical⟩ > 0 (increases loss, not decreases)
- When a delta is harmful, the retention formula is **undefined** — there is no benefit to retain

The code correctly handles this with a guard: `if (base_ppl - promoted_ppl) > 0 else 0`, returning 0 retention.

### 3. Scale=20 Was Already Known to Fail

Finding #330 established:
- Scale=13: N=5 composition is near-lossless (-4pp MMLU degradation)
- Scale=20: Domain-specific PPL improves **but out-of-distribution severely degrades** (-42pp MMLU)

Finding #328 showed scale=20 causes catastrophic OOD failures.

**The experiment repeated this known failure** by using scale=20 without theoretical justification for why scale=20 would be safe for adapter promotion, despite prior findings establishing it is unsafe.

---

## Confirming Evidence

### 1. Scale=5 vs Scale=20 Contrast (From `expert_promotion` experiment)

| Scale | Medical PPL (solo) | Status | Regime |
|-------|-------------------|--------|--------|
| **5** | 6.058 → 5.249 (+13.4% benefit) | SUPPORTED | Linear |
| **20** | 6.107 → 10.553 (-72.8% harm) | KILLED | Nonlinear |

At scale=5, the medical adapter is beneficial and composition succeeds. At scale=20, the same adapter is harmful and composition fails. This demonstrates the critical role of staying in the linear regime.

**Literature support:**
- Davis-Kahan sin theorem (matrix perturbation theory): When perturbation magnitude ‖E‖_op / Δ ≥ 1 (where Δ is the spectral gap), the bound becomes vacuous (sin θ ≤ 1.0). The `expert_promotion` MATH.md explicitly calculates this failure for scale=20.

### 2. Cross-Domain Degradation Pattern

All 5 domains degrade by 2.08–2.45× when composed:

| Domain | Base PPL | Composed PPL | Ratio |
|--------|----------|--------------|-------|
| Code | 5.495 | 13.43 | 2.45× |
| Math | 4.657 | 11.116 | 2.39× |
| Legal | 24.472 | 55.743 | 2.28× |
| Finance | 20.395 | 42.461 | 2.08× |

This is far worse than NRE's predicted bound (1.5×), indicating either:
- Adapters are not orthogonal (contradicts Finding #126)
- The composition formula itself breaks down in the nonlinear regime

The uniform degradation pattern (all ~2.2×) suggests a systematic failure, not outlier behavior.

### 3. Consistency with LLM Adapter Composition Literature

Research on multiple adapter composition (LLM-Adapters 2023, MTLoRA) identifies interference as a primary failure mode when:
- Adapters are trained on dissimilar domains (medical vs code vs finance)
- Composition mechanism lacks domain-specific routing or weighting

Our approach (uniform NRE averaging) provides **no mechanism to prevent interference**. Each domain's gradient is treated equally, even though domains are orthogonal in their learned representations.

---

## Contradicting Evidence

### 1. Finding #275: NRE Norm Preservation

Finding #275 proved NRE matches Fisher-Rao Karcher mean and prevents 1/√N norm shrinkage. However:
- Finding #275 was verified on **scale=5 to scale=13**, not scale=20
- Norm preservation is a **necessary condition** for composition, not sufficient
- A composition with preserved norm can still have harmful direction (pointing toward high loss)

### 2. Finding #126: Structural Orthogonality

Finding #126 proved adapters have near-zero cosine similarity (well below Welch bound). However:
- Orthogonality in weight space ≠ orthogonality in loss landscape space
- Two weight-orthogonal deltas can still interfere in loss space if they explore sharply curved regions

---

## Alternative Approaches

### 1. ✓ Promotion at Scale=5 (Proven, `expert_promotion` experiment)

**Approach:** Merge medical adapter into base, then train 4 others on promoted base
- **Evidence:** `expert_promotion` achieves SUPPORTED status with Davis-Kahan proofs
- **Why it works:** Scale=5 is in linear regime; first-order approximations hold; medical benefit is preserved

**Recommended** — this is the path forward for true adapter promotion.

### 2. Input-Conditioned Routing (Theory from LeJEPA)

**Approach:** Replace uniform NRE averaging with learned routing: r(x) → select or weight adapters per input
- **Motivation:** Finding #335 (room_gradient_analysis) proved that static adapter geometry (B-matrix spatial structure) cannot discriminate domains — routing must come from **learned features of the input**, not adapter weight geometry
- **Evidence:** LeJEPA framework (arXiv 2511.08544) shows routing function is the necessary and sufficient condition for composition
- **Why it could work:** Input-conditioned routing moves the problem from weight space (where NRE applies) to feature space (where the current input can select the right adapter)

**Requires:** New experiment to test learned routing with careful proof of why input conditioning enables composition.

### 3. Subspace Projection (Grassmannian Geometry)

**Approach:** Instead of averaging deltas in parameter space, average in subspace spanned by each adapter
- **Motivation:** Avoid direct interference by projecting each delta onto its own principal subspace
- **Evidence:** Adapter composition research (MTLoRA) shows parameter sharing mitigates interference
- **Why it could work:** Subspaces can be nearly orthogonal even when weight matrices are not

**Barrier:** Requires deriving the projection formula and proving it preserves benefit.

### 4. ✗ Uniform NRE Averaging (Killed)

**Approach:** Compose all 5 adapters with equal weight, normalize with NRE
- **Why it fails:** Assumes each adapter is beneficial (precondition), assumes linear regime holds (false at scale=20)
- **Does not scale:** As N grows, individual adapter contribution ∝ 1/√N, eventually becomes negligible

---

## Implications for Next Experiments

### 1. Stay in the Linear Regime

Any experiment composing multiple adapters **must operate at scale=5–13**, where:
- First-order Taylor approximation ‖ΔL‖ ~ ⟨∇L, ΔW⟩ holds
- Individual adapters are beneficial on their own domains
- Davis-Kahan bound is not vacuous (sin θ ≤ meaningful bound)

**Verification:** For any proposed scale, first run solo adapter test. If solo PPL improves > 10%, the scale is likely safe. If solo degrades, precondition is violated.

### 2. Routing Is the Bottleneck

The real problem is **not norm preservation or orthogonality** — it's **selecting the right adapter for the current input**. Findings #334 and #335 show:
- Pre-summing without routing = unrouted mixture = guaranteed interference
- Routing function r(x) is the necessary and sufficient condition

Future work should focus on **learned routing mechanisms**, not better averaging formulas.

### 3. Behavioral Validation Before Composition

Experiments on adapter composition should test:
1. **Behavioral quality:** Does the composed model generate coherent text across domains?
2. **Not just PPL:** PPL can be misleading (recall the project proved r=0.08 with behavioral quality)
3. **Per-domain generation samples:** Do code domain outputs improve? Medical outputs degrade?

---

## Recommended Follow-Up

### Primary Path: `expert_promotion` → Scale-5 Composition Research

1. **Confirm `expert_promotion` (scale=5)** reaches SUPPORTED status with full LEARNINGS.md
2. **Extend to N=10 domains** at scale=5 with `expert_promotion`
3. **Research: Why scale=5 works** — derive Davis-Kahan bound showing first-order approximation error is bounded

### Secondary Path: Learned Routing for Composition

**Motivation:** Finding #335 proved routing signal must come from input, not adapter geometry.

1. **Experiment:** Train a routing network r(x) that predicts which adapter(s) to use for input x
2. **Framework:** Gumbel-Sigmoid routing (already proven in project) applied to multi-adapter composition
3. **Proof requirement:** Show that r(x) makes interference impossible by construction

**Literature support:** LLM-Adapters (EMNLP 2023) and AdapterFusion show learned weighting mechanisms successfully reduce interference.

---

## Summary

| Aspect | Verdict | Evidence |
|--------|---------|----------|
| **Scale choice** | WRONG | Finding #330/#328 already proved scale=20 fails |
| **Precondition violated** | YES | Medical adapter degrades its own domain by 72.8% |
| **Proof applicability** | NO | Linear regime assumption breaks down at scale=20 |
| **Path forward** | YES | Scale=5 composition (`expert_promotion`) works; learned routing shows promise |

The experiment was well-designed for the scale it used, but **operating outside the linear regime is a structural impossibility, not a tuning problem**. The correct approach is scale=5 promotion followed by learned routing for multi-domain composition.

---

## References

- **Finding #275** (conclusive): NRE Norm Preservation  
- **Finding #126** (conclusive): Structural Orthogonality  
- **Finding #330** (supported): Scale behavior (scale=13 safe, scale=20 catastrophic)  
- **Finding #328** (killed): Scale=20 OOD degradation  
- **Finding #334** (conclusive): Pre-sum without routing = unrouted mixture  
- **Finding #335** (killed): B-matrix Sobel gradients cannot discriminate domains  
- **expert_promotion** (SUPPORTED, scale=5): Proven adapter promotion approach  

**External literature:**
- Flat-LoRA (ICML 2025): Loss landscape sharpness in LoRA weight space  
- Davis-Kahan sin theorem: Perturbation bounds for singular vectors  
- LLM-Adapters (EMNLP 2023): Adapter composition and interference mitigation  
- MTLoRA: Multi-task adapter disentanglement through parameter sharing  
- LeJEPA (2511.08544): Input-conditioned routing necessity for composition
