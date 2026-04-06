"""Pierre v3 — model-native LoRA with BitLinear side-path (~220 lines).

Key change from v2: no unpacking. BitLinear stays packed.
LoRA runs as side computation: y = BitLinear(x) + scale * (x @ A) @ B
"""

from pierre.v3.pierre import (
    # Model wrapping
    LoRASideLayer,
    inject_lora,
    inject_composed_lora,
    strip_lora,
    # Hidden state
    extract_hidden,
    # Router
    calibrate_router,
    route,
    route_topk,
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
