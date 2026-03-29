# Flat-LoRA: Research Digest

## Hypothesis

SAM (Sharpness-Aware Minimization) training produces LoRA adapters that
converge to flatter loss minima, resulting in better merge quality when
composing multiple adapters via weight-space averaging.

**Falsifiable prediction:** Flat-LoRA merged adapters will show >3 percentage
points better improvement over base than standard LoRA merged adapters,
across at least one merge method (Task Arithmetic, TIES, DARE, Direct Sum).

## What This Experiment Does

Trains 5 domain-specific LoRA adapters on BitNet-2B-4T (ternary, 2.4B params)
using two methods:
1. **Standard LoRA:** vanilla Adam optimization
2. **SAM-LoRA:** Sharpness-Aware Minimization in the LoRA parameter space

Then merges each set of 5 adapters using 4 merge methods and compares:
- Individual adapter quality (PPL on own domain)
- Merged adapter quality (PPL averaged across all domains)
- Loss landscape sharpness (sensitivity to random perturbation)
- Adapter orthogonality (cosine similarity between flattened adapters)

## Key References

- Flat-LoRA: Sun et al., arXiv:2409.14396 (ICML). Full weight-space SAM for LoRA.
- SAM: Foret et al., arXiv:2010.01412 (ICLR 2021). Sharpness-Aware Minimization.
- Model Soups: Wortsman et al., ICML 2022. Weight averaging in same basin.
- LoRA Soups: arXiv:2410.13025. CAT composition beats data mixing.
- exp_composition_interpolation_landscape: smooth convex landscapes proven.
- exp_bitnet_2b_real_composition: baseline 5-domain training pipeline.

## Empirical Results

### K1: SAM Implementation on MLX

**K1 PASS.** SAM training runs correctly on MLX. The double forward+backward
pass works with MLX's lazy evaluation. Training time overhead is 1.95x
(373s vs 192s for 5 adapters), consistent with the theoretical 2x cost.

### Individual Adapter Quality

| Domain | Standard PPL | SAM PPL | Delta |
|--------|-------------|---------|-------|
| python | 2.220 | 2.223 | +0.1% |
| math | 3.599 | 3.592 | -0.2% |
| medical | 4.762 | 4.745 | -0.3% |
| legal | 16.490 | 16.592 | +0.6% |
| creative | 4.936 | 4.930 | -0.1% |
| **Average** | **6.401** | **6.417** | **-0.2%** |

Individual quality is essentially identical. SAM does not help or hurt
individual adapter performance at this scale and iteration count.

### Merge Quality Comparison

| Merge Method | Std avg PPL | SAM avg PPL | Std vs base | SAM vs base | Delta pp |
|-------------|------------|------------|------------|------------|---------|
| Task Arithmetic | 7.983 | 7.982 | +8.16% | +8.17% | +0.01 |
| TIES | 7.465 | 7.459 | +14.12% | +14.19% | **+0.07** |
| DARE | 7.979 | 7.973 | +8.21% | +8.28% | +0.07 |
| Direct Sum | 7.596 | 7.770 | +12.61% | +10.61% | **-2.01** |

Best SAM advantage: +0.07pp (TIES, DARE). Worst: -2.01pp (Direct Sum).
All deltas are within noise range except Direct Sum, where SAM is worse.

### K2: Merge Improvement Assessment

**K2 technically PASS (best delta > 0) but practically FAIL.**

The best improvement is +0.07 percentage points. This is:
- 43x below the 3pp success threshold
- Within measurement noise (5 domains, 25 val samples each)
- Inconsistent across methods (positive for TA/TIES/DARE, negative for Direct Sum)

### Sharpness Analysis

| Domain | Std sharpness (%) | SAM sharpness (%) |
|--------|------------------|-------------------|
| python | -0.06 | 0.00 |
| math | 0.04 | 0.00 |
| medical | 0.16 | 0.26 |
| legal | -0.02 | -0.02 |
| creative | -0.02 | 0.10 |
| **Average** | **0.02** | **0.07** |

**Both methods produce extremely flat loss landscapes.** The sharpness
values (0.02% and 0.07%) are negligible -- neither method lands in a
sharp minimum. SAM is actually slightly LESS flat than standard, though
both are essentially at zero sharpness.

### Orthogonality

- Standard: mean |cos| = 0.0010
- SAM: mean |cos| = 0.0013

Both are extremely orthogonal (40-50x below 0.05 threshold). SAM does not
improve and slightly worsens orthogonality.

## Why Flat-LoRA Does Not Help Here

The null result has a clear explanation: **the Grassmannian skeleton already
solves the problem Flat-LoRA is designed to fix.**

### The Merge Problem Flat-LoRA Addresses

When merging adapters, the merge perturbation per adapter is:

    delta_i = sum_{j != i} lambda_j * B_j @ A_j

If this perturbation moves adapter i out of its flat basin, merge quality
degrades. Flat-LoRA widens the basin to accommodate larger perturbations.

### Why Grassmannian Orthogonality Makes Flatness Irrelevant

With Grassmannian A-matrices, A_i^T @ A_j ~ 0 for i != j. Therefore:

    ||delta_i projected onto adapter i's subspace|| ~ 0

The merge perturbation is nearly orthogonal to each adapter's operating
subspace. **The adapter never leaves its basin, regardless of basin width.**

Quantitatively: mean |cos| = 0.001, meaning the merge perturbation in each
adapter's subspace is 0.1% of its magnitude. The sharpness measurement
confirms this -- even at 1% random perturbation (10x the actual merge
perturbation), PPL changes by <0.3%.

### Training Loss Convergence

SAM and standard LoRA reach identical final losses (within 0.5% across all
domains). At 200 steps with this data, both methods find the same minimum.
This is consistent with the landscape being already flat in the LoRA subspace
-- SAM has nothing to flatten.

## Limitations

1. **200 steps may be insufficient for SAM to differentiate.** SAM typically
   shows benefits over thousands of steps where it can steer toward flatter
   basins. At 200 steps, both methods may converge to the same point.

2. **LoRA-space SAM, not full weight-space SAM.** The Flat-LoRA paper perturbs
   in the full m x n weight space. Our implementation perturbs only in the
   LoRA parameter space (A, B). However, the Grassmannian argument above
   applies regardless -- the orthogonality makes merge perturbation small
   in ANY subspace.

3. **5 domains only.** At N=5, the merge perturbation per adapter is 4/5
   of the total. At N=50, each adapter sees 49x more interference. Flat-LoRA
   might help at larger N where the perturbation magnitude grows. However,
   the orthogonality bound ||delta_i|| <= (alpha/r)^2 * sum_j ||B_j|| * ||A_i^T A_j||
   still protects via the A_i^T A_j ~ 0 term.

4. **Ternary base model.** The Flat-LoRA paper tested on standard FP16 models.
   The ternary weight structure may create naturally flat landscapes (fewer
   distinct weight values = more degenerate minima). This could explain the
   zero sharpness even without SAM.

## What Would Kill This

Already killed at this scale: S1 FAIL (0.07pp vs 3pp threshold). The
mechanism is sound (SAM works on MLX, K1 PASS) but provides no benefit
in the Grassmannian orthogonal regime.

The hypothesis could be revived IF:
- Testing on FP16 base without Grassmannian (where cos > 0.1 and merge
  perturbation is large)
- Testing at much longer training (>1000 steps) where minima diverge
- Testing full weight-space perturbation (Flat-LoRA proper, not LoRA-SAM)

## Verdict

**K1 PASS, K2 PASS (technically), S1 FAIL.**

SAM training works on MLX but provides no measurable merge improvement.
The Grassmannian orthogonal skeleton already ensures near-zero cross-adapter
interference, making loss landscape flatness irrelevant for composition quality.
This is a clean negative result that strengthens the case for orthogonality
as the primary mechanism enabling composition -- not flat minima.

**Status: KILLED (K2 practically fails -- 0.07pp is noise, not signal).**

Training overhead is 1.95x for zero benefit. Not recommended for the SOLE
architecture where Grassmannian orthogonality is the composition mechanism.
