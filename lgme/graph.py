"""Node and Graph — the structural backbone of LGME."""

from .model import embed, attn_forward, mlp_forward, output_forward


class Node:
    def __init__(self, node_type, param_keys, expert_id=0, layer_idx=0):
        self.node_type = node_type      # 'attn', 'mlp', or 'output'
        self.param_keys = param_keys    # which state_dict keys this node owns
        self.layer_idx = layer_idx      # which transformer layer this node belongs to
        self.kv_keys = []               # KV-cache keys (attn nodes only)
        self.kv_values = []             # KV-cache values (attn nodes only)
        self.expert_id = expert_id      # MLP expert index
        self.router_key = None          # vector for routing (set by router)
        self.activation_count = 0       # how often this expert has been selected
        self.frozen = False             # if True, skip gradient updates


class Graph:
    def __init__(self, sd, nodes, n_head, head_dim):
        self.sd = sd
        self.nodes = nodes
        self.n_head = n_head
        self.head_dim = head_dim

    def get_expert_keys(self, expert_id):
        """Return state_dict keys for a given expert's weights."""
        if expert_id == 0:
            return {'layer0.mlp_fc1', 'layer0.mlp_fc2'}
        return {f'expert{expert_id}.mlp_fc1', f'expert{expert_id}.mlp_fc2'}

    def get_attn_keys(self):
        """Return state_dict keys for shared attention params."""
        return {k for k in self.sd if '.attn_w' in k}

    def reset_kv(self):
        for n in self.nodes:
            if n.node_type == 'attn':
                n.kv_keys = []
                n.kv_values = []

    def forward(self, token_id, pos_id, sd=None, mlp_experts=None, route_fn=None,
                route_produces_logits=False):
        """Forward pass. sd is accepted for mx.value_and_grad tracing.

        Args:
            route_produces_logits: if True, route_fn returns (logits, selected)
                instead of (hidden_state, selected). The output node is skipped.
        """
        if sd is None:
            sd = self.sd
        x = embed(token_id, pos_id, sd)
        for node in self.nodes:
            if node.node_type == 'attn':
                x = attn_forward(x, node.layer_idx, sd,
                                 node.kv_keys, node.kv_values,
                                 self.n_head, self.head_dim)
            elif node.node_type == 'mlp':
                if mlp_experts is not None and route_fn is not None:
                    result = route_fn(x, mlp_experts, sd)
                    if route_produces_logits:
                        return result  # result is already logits
                    x = result
                else:
                    fc1_key = node.param_keys[0]
                    fc2_key = node.param_keys[1]
                    x = mlp_forward(x, sd, fc1_key, fc2_key)
            elif node.node_type == 'output':
                return output_forward(x, sd)
        return x
