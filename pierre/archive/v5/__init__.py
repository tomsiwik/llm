"""Pierre v5 — Fully ternary composable experts. Orthogonal by construction.

All matmuls use native ternary Metal kernels:
  y = BitLinear_base(x) + scale * BitLinear_B(BitLinear_A(x))

Orthogonality: Grassmannian A_i ⊥ A_j → interference impossible.
"""

from pierre.v5.pierre import (
    # Ternary ops
    pack_ternary,
    unpack_ternary,
    quantize_matrix_to_bitlinear,
    # Model wrapping
    TernaryLoRASideLayer,
    inject_ternary_lora,
    strip_lora,
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
