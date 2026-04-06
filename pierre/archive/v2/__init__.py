"""Pierre v2 — single-file implementation (~200 lines)."""

from pierre.v2.pierre import (
    # Hidden state extraction
    extract_hidden,
    # Router
    calibrate_router,
    route,
    # Compose
    lora_delta,
    nre_merge,
    build_deltas,
    merge_deltas,
    # Isolate
    null_space_projector,
    # Pre-merge
    premerge,
    # I/O
    load_adapter,
    load_skeleton,
    # Constants
    TARGET_MODULES,
)
