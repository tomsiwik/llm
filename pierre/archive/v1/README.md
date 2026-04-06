# Pierre v1 — Original 4-file implementation

Archived 2026-04-04 after consolidation into `pierre.py` (v2).

## Why this was replaced

552 lines across 4 files → 200 lines in 1 file. Same functionality.

### Redundancies found:
- `RidgeRouter` class wrapped a single matmul + argmax
- `RouterStatistics` class wrapped two accumulations
- `NullSpaceProjector` class wrapped a 3-line SVD
- `Pipeline` class (221 lines) duplicated forward pass logic — and missed the causal mask, causing routing to drop from 99.6% to 16.8%

### Bugs caught during v1→v2:
1. Missing causal mask + final norm in hidden state extraction
2. Wrong Grassmannian A-matrix (used domain_0 for all adapters)
3. `mx.linalg.solve` needs `stream=mx.cpu`

## Files
- `router.py` — RidgeRouter, RouterStatistics, solve_ridge
- `compose.py` — compute_delta, nre_merge_deltas, premerge_deltas_into_model
- `nullspace.py` — NullSpaceProjector
- `pipeline.py` — Pipeline orchestration class
