"""Pierre v5.3 — Lazy bf16 LoRA side-path. Zero eval overhead.

BitLinear base (native ternary kernel) + bf16 adapter (lazy, no per-module eval).
17x less dispatch overhead than v5's BitLinear wrappers.
"""

from pierre.v53.pierre import (
    LazyLoRASideLayer, inject_lazy_lora, strip_lora,
    extract_hidden, calibrate_router, route,
    nre_merge, null_space_projector,
    load_adapter, load_skeleton, TARGET_MODULES,
)
