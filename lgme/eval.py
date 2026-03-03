"""Evaluation — forward-only loss on a held-out set."""

import mlx.core as mx
from .router import route_mlp_experts


def eval_loss(graph, docs, uchars, BOS, block_size, mlp_experts=None, route_fn=None,
              route_produces_logits=False):
    """Compute mean loss across docs without backward pass or grad accumulation.

    Args:
        graph: Graph instance
        docs: list of name strings
        uchars: sorted unique characters
        BOS: BOS token id
        block_size: max sequence length
        mlp_experts: list of expert dicts (None for dense model)
        route_fn: routing function (None for dense model)
        route_produces_logits: if True, route_fn returns logits directly

    Returns:
        float — mean cross-entropy loss
    """
    total_loss = 0.0
    for doc in docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        graph.reset_kv()
        doc_loss = 0.0
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            if mlp_experts is not None and route_fn is not None:
                logits = graph.forward(token_id, pos_id, mlp_experts=mlp_experts,
                                       route_fn=route_fn,
                                       route_produces_logits=route_produces_logits)
            else:
                logits = graph.forward(token_id, pos_id)
            probs = mx.softmax(logits)
            doc_loss += (-mx.log(probs[target_id] + 1e-8)).item()
        total_loss += doc_loss / n
    return total_loss / len(docs)
