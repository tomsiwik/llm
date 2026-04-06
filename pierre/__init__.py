"""Pierre — Runtime LoRA on Ternary Language Models.

  pierre.py         The implementation (~220 lines)
  archive/          Version history (v1–v7, research artifacts)
  SPEED_RESEARCH.md 10 approaches to 100+ tok/s
  KEYFRAME_RESEARCH.md  Deterministic verification research
"""

from pierre.pierre import (
    # Adapter management
    RuntimeLoRA,
    attach_adapter,
    detach_adapters,
    compose_adapters,
    # Router
    encode,
    fit_router,
    route,
    # Isolation
    null_space_projector,
    # I/O
    load_adapter,
    load_frozen_A,
    # Constants
    ADAPTER_TARGETS,
)
