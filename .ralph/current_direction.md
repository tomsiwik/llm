# Current Direction: Procrustes Decomposition (Exp 3)

## Research Question

**Can Procrustes/SVD alignment of domain-specific capsule groups decompose them
into a shared component (applied always, no routing) and unique residuals
(routed via softmax), achieving equivalent composition quality with fewer
domain-specific parameters?**

This tests VISION.md Protocol step 2: "Decompose: Procrustes-align, extract
shared + unique components." The current composition concatenates full domain
groups. This experiment asks whether we can factor out the common knowledge,
apply it unconditionally, and route only the domain-specific residuals.

## Target Model

- **Name**: `procrustes_decomp`
- **Directory**: `micro/models/procrustes_decomp/`
- **Parent**: `capsule_moe`
- **New params**: ~0 (decomposition is post-hoc linear algebra on trained weights)

## Success Criteria

| Criterion | Target | Kill Threshold |
|-----------|--------|----------------|
| Shared+unique vs full-group composition | <3% degradation | >10% degradation |
| Shared subspace variance explained | >30% of total | <10% (no shared structure) |
| Unique residual effective rank | <70% of full group rank | >90% (no compression) |
| Decomposition stability across seeds | <3% variance in shared fraction | >5% variance |

## Kill Criteria (pre-registered)

1. Shared+unique composition degrades >10% vs full-group composition
2. Shared subspace captures <10% of variance (no meaningful shared structure)
3. Applying shared component hurts vs not applying it (shared is noise)
4. Decomposition is unstable across seeds (>5% variance in shared fraction)

## Method

### Phase 1: Decomposition analysis (diagnostic, no new model)
1. Train via standard shared-base composition protocol
2. Extract capsule weight matrices A_g, B_g for all 8 composed groups
3. Stack → SVD → analyze variance spectrum
4. Measure shared fraction at various rank cutoffs
5. Compare shared subspaces across 3 seeds for stability

### Phase 2: Shared+unique forward pass
1. Decompose into shared component (top-r singular vectors) + unique residuals
2. Forward pass: y = shared(x) + sum_selected w_g * unique_g(x)
3. shared(x) uses the shared A/B projection, always applied (no routing)
4. unique_g(x) uses per-group residual A/B, routed by softmax router
5. Re-calibrate router on unique residuals only (~50-100 steps)
6. Compare val loss vs full-group composition baseline

### Phase 3: Parameter efficiency analysis
1. Measure effective rank of unique residuals vs full groups
2. Test truncated unique residuals (drop lowest singular components)
3. Find the quality-parameter tradeoff: how much of unique can be dropped?

## What Prior Work Informs This

1. **capsule_moe**: shared-base composition validated at -0.3% vs joint (our baseline)
2. **contrastive_router**: domains indistinguishable at d=64 → expect LARGE shared
   component (most capsule knowledge is domain-agnostic at micro scale)
3. **sparse_router**: k=2 optimal, router routes by task quality → unique residuals
   should also be routed by reconstruction loss
4. **VISION.md**: "Shared knowledge stays at full strength. Unique knowledge activates
   only when relevant, also at full strength." — this is exactly what we're testing
5. **Memory — task-routing > identity-routing**: unique residuals should be task-routed,
   which aligns with the softmax router's validated mechanism

## Key Risks

1. **Domains too similar**: at micro scale, a-m vs n-z groups may be nearly identical
   → shared component captures everything, unique residuals are just noise. This
   would be informative (validates shared structure) but doesn't test residual routing.
2. **SVD linearity assumption**: capsule groups may have non-linear relationships
   that SVD misses. Mitigation: check reconstruction error of the decomposition.
3. **Shared component interference**: different groups using the "same" subspace for
   different purposes. Forcing a shared component destroys both. Mitigation:
   measure per-group reconstruction fidelity.

## Why Highest-Impact

1. Item #4 in VISION.md "What Remains" — the immediate next step
2. Tests VISION.md's core decomposition principle (shared+unique separation)
3. Orthogonal to routing failures: doesn't need domain discrimination or sparse routing
4. If shared component is large (expected at micro scale), validates the scalability
   argument: adding expert N+1 = add only the unique residual, not full groups
5. Builds directly on capsule_moe composition (validated protocol)
6. Pure linear algebra — low implementation risk, highly informative diagnostics

## Connection to Vision

If Procrustes decomposition works, the expert library becomes:
- Shared basis: O(1) component applied to all tokens (grows slowly with N)
- Unique residuals: O(N) small per-expert additions (routed, much smaller than full groups)
- Adding expert N+1: train new groups, project onto shared subspace, store only residual
- This is the scalability path: huge-model knowledge at small-model active parameters
