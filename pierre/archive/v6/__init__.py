"""Pierre v6 — Precomputed concatenated deltas. 60 dispatches target."""

from pierre.v6.pierre import (
    precompute_deltas, precompute_memory_mb,
    inject_precomputed,
    PrecomputedDeltaLinear, ConcatQKVDeltaLinear,
    extract_hidden, calibrate_router, route,
    load_adapter, load_skeleton,
)
