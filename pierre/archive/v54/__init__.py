"""Pierre v5.4 — quantized_matmul side-path. 2-bit lazy, no wrapper overhead."""

from pierre.v54.pierre import (
    QuantizedLoRASideLayer, inject_quantized_lora, strip_lora,
    extract_hidden, calibrate_router, route,
    nre_merge, null_space_projector,
    load_adapter, load_skeleton, TARGET_MODULES,
)
