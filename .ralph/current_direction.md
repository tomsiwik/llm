# Current Direction: Pierre Pro Grassmannian Skeleton Ready

**Experiment:** exp_pro_grassmannian_init
**Type:** Verification (Type 1)
**Status:** SUPPORTED (both kill criteria PASS)
**Completed:** 2026-04-06
**Finding:** #318

## Results Summary

Grassmannian A-matrix initialization verified on Qwen3-4B-4bit (GQA architecture).
Exact orthogonality confirmed at N=5 and N=24 across all modules and layers.

| Prediction | Measured | Match |
|-----------|----------|-------|
| cos = 0.0 at N=5 | 0.000000 (2,520 pairs) | YES |
| cos = 0.0 at N=24 | 0.000000 (69,552 pairs) | YES |
| N_max = 160 (d=2560, r=16) | 160 | YES |
| GQA same capacity as MHA | All modules verified | YES |

## Architecture Dimensions (Qwen3-4B-4bit)

| Module | in_features | N_max (r=16) |
|--------|-------------|-------------|
| q/k/v_proj | 2560 | 160 |
| o_proj | 4096 | 256 |
| gate/up_proj | 2560 | 160 |
| down_proj | 9728 | 608 |

## Bug Fixed

4-bit quantized models store packed weight shapes (in_features / 8 for 4-bit).
Reading `weight.shape[-1]` gives wrong dimensions (320 instead of 2560).
Fix: read from model config, cross-validate with `shape * (32/bits)`.

## Skeleton Files Ready

- `grassmannian_skeleton_n5.npz` -- 1,260 keys, 5 domains
- `grassmannian_skeleton_n24.npz` -- 6,048 keys, 24 domains

## What's Next

Skeleton is ready. Next experiment:
1. **exp_pro_sft_5_adapters** -- Train 5 SFT domain adapters using the N=5 skeleton
   - Load frozen A-matrices from grassmannian_skeleton_n5.npz
   - Train B-matrices only (LoRA with frozen A)
   - Validate composition quality
