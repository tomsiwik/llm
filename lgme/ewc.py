"""EWC (Elastic Weight Consolidation) — prevent forgetting in shared attention params."""

import mlx.core as mx


def compute_fisher(graph, docs, uchars, BOS, block_size,
                   attn_keys, sample_size=50, rng=None):
    """Compute diagonal Fisher Information Matrix over attention params.

    Uses mx.grad for per-document gradient computation.

    Args:
        graph: Graph instance
        docs: training docs to estimate Fisher over
        uchars: sorted unique characters
        BOS: BOS token id
        block_size: max sequence length
        attn_keys: set of state_dict keys for shared attention weights
        sample_size: number of docs to sample
        rng: optional random.Random instance

    Returns:
        dict mapping key → mx.array fisher values (only for attn_keys)
    """
    import random as _random
    if rng is None:
        rng = _random.Random(42)

    sample = docs[:sample_size] if len(docs) <= sample_size else rng.sample(docs, sample_size)
    fisher = {k: mx.zeros_like(graph.sd[k]) for k in attn_keys}

    for doc in sample:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        def loss_fn(sd):
            graph.reset_kv()
            losses = []
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = graph.forward(token_id, pos_id, sd=sd)
                probs = mx.softmax(logits)
                losses.append(-mx.log(probs[target_id] + 1e-8))
            return mx.mean(mx.stack(losses))

        grads = mx.grad(loss_fn)(graph.sd)
        for k in attn_keys:
            if k in grads:
                fisher[k] = fisher[k] + grads[k] ** 2

    for k in fisher:
        fisher[k] = fisher[k] / len(sample)

    mx.eval(fisher)
    return fisher


def ewc_penalty(sd, fisher, theta_star, lambda_ewc):
    """Compute EWC penalty: lambda/2 * sum_k F_k * (theta_k - theta_star_k)^2.

    Args:
        sd: current state_dict (traced by value_and_grad)
        fisher: dict mapping key → fisher mx.array
        theta_star: dict mapping key → optimal param mx.array
        lambda_ewc: regularization strength

    Returns:
        mx.array scalar — penalty term to add to loss
    """
    penalty = mx.array(0.0)
    for k, f_val in fisher.items():
        diff = sd[k] - theta_star[k]
        penalty = penalty + mx.sum(f_val * diff * diff)
    return (lambda_ewc / 2) * penalty
