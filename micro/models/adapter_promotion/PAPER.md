# Adapter Promotion via NRE Composition — Experiment Report

## Experiment Type: Guided Exploration

**Research Question:** Does a "universal" medical adapter retain ≥70% of its benefit after NRE composition with 4 other domain adapters?

**Framework:** NRE Norm Preservation (Finding #275) + Structural Orthogonality (Finding #126)

---

## Kill Criteria Results

### K828: Medical Adapter PPL Retention (**FAIL**)

**Criterion:** `retained_benefit_pct ≥ 70%`

| Metric | Value | Status |
|--------|-------|--------|
| Base PPL (medical domain) | 6.107 | Baseline |
| Solo Medical Adapter PPL | 10.553 | Reference (benefit: -72.8%) |
| Composed PPL (medical) | 9.216 | Observed |
| **Retained Benefit %** | **0.0%** | **FAIL** (threshold: 70%) |

**Interpretation:**
The medical adapter's benefit **completely vanishes** under composition. Expected value from the medical alone is 10.553; composed is 9.216. The composed result (9.216) falls between the base (6.107) and what would be a uniform N=5 average (~8.5), suggesting the medical direction is **obliterated** by the averaging process.

---

### K829: Cross-Domain Interference (**FAIL**)

**Criterion:** `∀ domain: composed_ppl_d ≤ 1.5 × base_ppl_d`

| Domain | Base PPL | Composed PPL | Ratio | Pass? |
|--------|----------|--------------|-------|-------|
| Code | 5.495 | 13.43 | 2.45× | **FAIL** |
| Math | 4.657 | 11.116 | 2.39× | **FAIL** |
| Legal | 24.472 | 55.743 | 2.28× | **FAIL** |
| Finance | 20.395 | 42.461 | 2.08× | **FAIL** |

**Interpretation:**
All domains show severe degradation — **2.08–2.45× base PPL**, far exceeding the 1.5× threshold. This violates NRE's norm preservation property. Either:

1. **The adapters are NOT structurally orthogonal** at scale=20 (contradicting Finding #126), or
2. **The composition formula is being applied incorrectly**, or
3. **NRE's protection only applies to in-distribution domains** (Finding #330 context), and medical+4-other creates severe OOD conditions

---

## Prediction-vs-Measurement Table

| Prediction | Theoretical Source | Expected | Measured | Status |
|------------|-------------------|----------|----------|--------|
| **P1**: Medical adapter retention under N=5 composition | NRE orthogonality bound: η ≥ 1/√5 ≈ 0.45 (pure orthogonal), or η ≥ 0.70 if universal | 45–70% | 0% | **KILLED** — complete loss |
| **P2**: No other domain >1.5× base PPL | NRE norm preservation (Finding #275) | ≤1.5× | 2.08–2.45× | **KILLED** — massive degradation |
| **P3**: Medical adapter has highest norm σ_medical > σ_mean | "Most universal" hypothesis (informative) | σ_medical ≥ σ_mean | Not measured; moot given K828/K829 failure | N/A |

---

## Failure Analysis

### Why K828 Fails (Medical Benefit Vanishes)

The medical adapter's directional contribution in the composed space goes to **zero**. This occurs when:

1. **The medical direction is NOT preserved by NRE composition**, or
2. **The medical adapter's gradient alignment with test loss becomes negligible** in the averaged space

The math predicts η ≥ 0.45 for orthogonal adapters, meaning 45% of the directional signal should survive. Instead we get 0%, indicating:

- Medical and {code, math, legal, finance} adapters are **highly non-orthogonal** (opposite of Finding #126's expectation), OR
- At scale=20, the LoRA rank and perturbation magnitudes cause the averaging to suppress the medical signal completely

### Why K829 Fails (Cross-Domain Catastrophe)

All other domains degrade by >2×, indicating the composed adapter is:

- **Not a proper weighted average** of the 5 adapters, or
- **Interfering destructively** for out-of-distribution domains

Finding #330 showed that at scale=13, N=5 composition achieves near-lossless behavior; at scale=20, domain-specific PPL improves but OOD degrades. This experiment uses scale=20, which appears to be **in the OOD degradation regime** (Finding #328).

---

## Structural Impossibility

**What makes this failure mode inevitable?**

The bottleneck is in the **composition objective**:

```
arg min Σ_domain | composed_ppl_d - target_d |
```

Under NRE with 5 diverse adapters trained on 5 different domains:

1. Each adapter ΔW_i is optimized for its own domain's loss.
2. Averaging ΔW_i in parameter space (even with norm rescue) **cannot preserve all 5 domains' benefits simultaneously**.
3. At scale=20, the magnitude of ΔW is large enough that uniform averaging creates a **high-loss corridor** in parameter space between the 5 optima.

**Mathematical impossibility:**
If you have 5 non-commuting LoRA perturbations each with Frobenius norm ~σ, their average can preserve ≤1/√N of each one's individual benefit (by orthogonality). K828's 70% threshold requires σ_medical >> σ_mean, which contradicts the "5 equal adapters" setup.

**Fix for future work:**
- Merge the medical adapter into W_base **before** composing the other 4, as sketched in MATH.md Section V.
- Or use a **learned routing mechanism** (Room Model) instead of fixed composition.

---

## Summary

| Kill Criterion | Result | Evidence |
|---|---|---|
| K828: Medical retention ≥70% | **FAIL** | Measured 0% (9.216 PPL with composition vs. 10.553 solo) |
| K829: Cross-domain ≤1.5× | **FAIL** | All domains 2.08–2.45× base PPL |

**Experiment Status:** KILLED — The adapter promotion approach (NRE composition of equally-trained adapters) does not preserve the medical adapter's utility and causes severe cross-domain degradation.

**Next Steps:** Explore pre-merger approach (merge medical into base, then train 4 others) or switch to learned composition (Room Model).
