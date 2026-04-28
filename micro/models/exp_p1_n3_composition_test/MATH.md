# MATH.md — N=3 Composition Test

## Theorem (LoRA Composition Linearity)

**Statement.** Given N LoRA adapters with weight updates ΔW_i = B_i A_i where A_i ∈ ℝ^{r×d}, B_i ∈ ℝ^{d×r}, the composed update ΔW = Σ_{i=1}^{N} B_i A_i preserves each adapter's contribution as a rank-r subspace addition. The composed model W + αΔW applies all N domain specializations simultaneously.

**Proof sketch.** Each ΔW_i = B_i A_i is rank ≤ r. The sum Σ B_i A_i has rank ≤ Nr. For N=3, r=6: rank(ΔW) ≤ 18, acting on d=1024 (Gemma 4 q_proj). The subspace occupied is 18/1024 ≈ 1.8% of the weight space, leaving ample room for non-interference.

**Critical distinction.** Correct: Σ(B_i @ A_i). Incorrect: (ΣB_i) @ (ΣA_i) — the cross-product introduces N(N-1) spurious interference terms.

**Why scale matters.** With α = lora_scale / N = 6/3 = 2.0 per adapter, the perturbation magnitude is bounded: ‖αΔW‖/‖W‖ ≪ 1 for the q_proj layer at Gemma 4 dimensions.

## Predictions

1. Per-domain accuracy under composition drops ≤5pp vs single-adapter (each adapter's rank-r subspace is preserved).
2. Composed PPL ≠ single-adapter PPL on at least one domain (composition is not tautological).
3. Math adapter alone on MedQA scores ≤55% (domain specialization, not general improvement).

## Kill Criteria

| KC | Metric | Threshold | Type |
|----|--------|-----------|------|
| K2062 | Per-domain accuracy drop | ≤5pp (GSM8K ≥67%, HumanEval ≥65%, MedQA ≥63%) | target |
| K2063 | Composed PPL ≠ single PPL | diff > 0.01 on any domain | proxy |
| K2064 | Cross-domain interference | math on MedQA ≤55% | target |

## References

- Ilharco et al. 2022, Task Arithmetic (arxiv:2212.04089)
- Hu et al. 2021, LoRA (arxiv:2106.09685)
- Finding #666: Target-gated kill rule
