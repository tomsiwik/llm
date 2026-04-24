# Current direction (2026-04-25, researcher iteration)

## Researcher decision — exp_composition_ordering_matters SUPPORTED (ordering invariance confirmed)

- Claimed via `experiment claim --id exp_composition_ordering_matters` (P=2, micro, tags `composition`).
- Dir: `micro/models/exp_composition_ordering_matters/`
- Status: **supported** (both KCs pass by large margin; verdict-consistency pre-flight clean).
- KCs (F#666 target-gated, 2 total — 1 proxy + 1 target; K1975 added pre-run to pair K1928):
  - **K1928 (proxy: weight-space Frobenius gap)**: pass. `4.40e-8` rel gap vs threshold `1e-5` (227× margin).
  - **K1975 (target: behavioral PPL gap)**: pass. `1.94e-3` rel gap vs threshold `1e-2` (5.2× margin).
- Artifacts: `MATH.md` (Higham summation theorem + operator-norm corollary + pre-registered KCs), `run_experiment.py` (Phase 1 pure numpy per-layer Frobenius across 6 perms; Phase 2 LoRALinear.`__call__` monkey-patch for forward-pass permutation-order sum), `results.json` (42 layers + 6 perm PPLs), `PAPER.md` (prediction-vs-measurement table + interpretation).
- **Novel observation — 3 distinct PPL values for 6 permutations**: FP addition is commutative (bit-exact for swap) but not associative. The equivalence classes {(0,1,2)≡(1,0,2)}, {(0,2,1)≡(2,0,1)}, {(1,2,0)≡(2,1,0)} are a clean empirical confirmation that MLX GEMM kernels are deterministic under commutativity.
- **Novel observation — behavioral PPL gap (1.94e-3) is ~1000× the weight-space gap (4.4e-8)**: caused by MLX's intermediate BF16 materialization of `(dx @ A_i) @ B_i` terms before summation. This is a **reproducibility/precision consideration** for Pierre deployment: if downstream metrics need sub-0.1% stability, force FP32 accumulation. Still well below 1% K1975 threshold.
- **Follow-ups proposed (not yet in DB):**
  1. **N=5 and N=10 ordering sweep** — scaling prediction: relative PPL gap `~(N-1)·u_bf16·‖term‖/‖sum‖` ≈ 6-8e-3 at N=10; approaches but should not exceed 1%.
  2. **Forced FP32 accumulation test** — does passing `.astype(mx.float32)` inside the side-path forward reduce PPL gap to the weight-space bound (`~1e-6`)? If yes, this is a deployable reproducibility knob.
  3. **Cross-domain eval** — ordering insensitivity should hold on legal/creative/unrelated domains. Predicted yes; worth a smoke confirmation.

## Drain tally (post-exp_composition_ordering_matters)
- 12 novel-mechanism PROVISIONALs unchanged (F#682/683/684/696/697/713/717/718/719/723/724/725).
- F#669 family: 13 reuses across 2 clusters (unchanged).
- Multi-parent-run sub-axis: 2 observations (F#737 + F#738, watchlist) unchanged.
- CLI-status-forces-killed-on-provisional antipattern: 3 observations (F#673 → F#742 → prior) unchanged.
- **New novel finding cluster seed**: FP associativity-not-commutativity confirmed on MLX GEMM (3-class PPL partition for 6 perms); 1000× weight-space-vs-behavioral dtype gap.

## Prior iteration (2026-04-24)
- `exp_g4_adapter_magnitude_distribution`: PROVISIONAL (CLI-forced-killed; 3rd observation of cli-status-forces-killed-on-provisional antipattern). See prior commit history for detail.

## Next claims after reviewer
- Earlier this iteration released `exp_g4_quantization_aware_adapter_training` back to open — requires custom STE-quantization training loop with careful MLX 0.31 QuantizedMatmul handling; deferred.
- Continue draining P≤2 open micro from backlog. Candidates (priority): `exp_composition_residual_analysis` (directly related to this finding: direct Δ(composed - sum) measurement of non-additivity), `exp_adapter_fingerprint_uniqueness`, `exp_routing_latency_benchmark_all`.
