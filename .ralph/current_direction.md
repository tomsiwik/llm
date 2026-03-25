# Current Direction: Revising exp_bitnet_float_merge_fp32

## Status: COMPLETE - all 4 fixes applied, experiment re-run, write-ups updated

## Fixes Applied
1. **Fix 1 (CRITICAL)**: Fixed 1/N^2 scaling bug in `compose_adapters_runtime`. Now sums A/B params without scaling; 1/N scaling applied via `lora_scale = LORA_SCALE/N` when constructing LoRALinear. Result: the "7% PPL gap" disappeared entirely -- all three methods now within 0.6%.
2. **Fix 2**: Retracted wrong explanation. The "bf16 truncation in per-token computation" claim was incorrect -- the gap was from the 1/N^2 bug. Updated PAPER.md with correct analysis.
3. **Fix 3**: K2 marked INCONCLUSIVE with note about MLX lazy evaluation.
4. **Fix 4**: Increased BENCH_RUNS from 3 to 5, added stddev for all latency measurements.

## Key Revised Results
- Runtime LoRA N=5: PPL 7.17 (was 7.72 -- massive improvement from bug fix)
- fp32 merge N=5: PPL 7.19 (unchanged)
- bf16 merge N=5: PPL 7.22 (unchanged)
- All three methods PPL-equivalent within 0.6%
- Verdict still KILLED on K3 (fp32 latency)
- bf16 merge remains the recommended serving path (39% faster than runtime, 0.6% PPL cost)

## Files Updated
- micro/models/bitnet_float_merge_fp32/run_experiment.py (code fixes)
- micro/models/bitnet_float_merge_fp32/results.json (new results)
- micro/models/bitnet_float_merge_fp32/PAPER.md (full rewrite)
- micro/models/bitnet_float_merge_fp32/MATH.md (corrected sections)
- HYPOTHESES.yml (updated evidence)
- FINDINGS.md (corrected entry)
