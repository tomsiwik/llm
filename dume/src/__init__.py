"""DUME: Training-Free Dynamic Upcycling of Expert Language Models (MLX)."""

from dume.src.merge import moerge
from dume.src.router import RidgeRouter, extract_router_weights
from dume.src.model import DUMEMoEBlock, DUMEModel

__all__ = [
    "moerge",
    "RidgeRouter",
    "extract_router_weights",
    "DUMEMoEBlock",
    "DUMEModel",
]
