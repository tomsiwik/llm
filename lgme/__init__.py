"""LGME — Living Graph of Micro-Experts (module package)."""

from .model import linear, softmax, rmsnorm, embed, attn_forward, mlp_forward, output_forward
from .graph import Node, Graph
from .router import (route_mlp_experts, clone_mlp_expert, kohonen_update, spawn_mlp_expert,
                     consolidate_experts, cosine_sim, init_router_key,
                     kohonen_update_som, route_with_lateral)
from .optimizer import adam_step
from .si import si_init, si_accumulate, si_consolidate, si_penalty
from .som import som_init, som_sigma, som_neighborhood, som_grow, som_accumulate_error, som_get_lateral_weights
