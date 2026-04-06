"""Pierre v4 — fully ternary composable experts. Zero inference overhead.

Key change from v3: ternary premerge instead of bf16 side-path.
Merge LoRA into BitLinear, re-quantize, repack → native Metal kernel speed.
"""

from pierre.v4.pierre import (
    # Ternary operations
    unpack_ternary,
    pack_ternary,
    quantize_to_ternary,
    # Ternary premerge
    ternary_premerge,
    ternary_premerge_composed,
    # Hidden state
    extract_hidden,
    # Router
    calibrate_router,
    route,
    # Compose
    nre_merge,
    # Isolate
    null_space_projector,
    # I/O
    load_adapter,
    load_skeleton,
    # Constants
    TARGET_MODULES,
)
