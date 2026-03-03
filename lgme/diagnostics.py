"""Diagnostics — physics-inspired measurements for continual learning analysis.

Implements:
- Spin glass overlap parameter (Edwards-Anderson order parameter)
- Approximate maximum Lyapunov exponent via power iteration + JVP
- Rate-distortion bound estimation
"""

import math
import mlx.core as mx


def spin_glass_overlap(sd_task1, sd_task2, keys=None):
    """Compute Edwards-Anderson overlap parameter between two weight snapshots.

    q ≈ 1: weights barely changed (stable)
    q ≈ 0: weights completely reoriented (catastrophic forgetting)

    Args:
        sd_task1: state_dict snapshot after task 1
        sd_task2: state_dict snapshot after task 2
        keys: optional set of keys to measure (defaults to all shared keys)

    Returns:
        float — overlap parameter q in [-1, 1]
    """
    if keys is None:
        keys = set(sd_task1.keys()) & set(sd_task2.keys())

    if not keys:
        return 0.0

    numerator = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for k in keys:
        w1 = sd_task1[k]
        w2 = sd_task2[k]
        numerator += mx.sum(w1 * w2).item()
        norm1_sq += mx.sum(w1 * w1).item()
        norm2_sq += mx.sum(w2 * w2).item()

    norm1 = math.sqrt(norm1_sq) + 1e-8
    norm2 = math.sqrt(norm2_sq) + 1e-8

    return numerator / (norm1 * norm2)


def approx_lyapunov(graph, sd, token_ids, pos_ids, mlp_experts=None,
                    route_fn=None, n_power_iter=3, seed=42):
    """Approximate maximum Lyapunov exponent of hidden state dynamics.

    Uses power iteration on the Jacobian-vector product (via mx.jvp)
    to estimate the spectral radius of the hidden-state-to-hidden-state map.

    λ_max ≈ 0: edge of chaos (optimal information processing)
    λ_max << 0: too stable (can't learn)
    λ_max >> 0: chaotic (interference)

    Args:
        graph: Graph instance
        sd: state_dict
        token_ids: list of token ids for a sequence
        pos_ids: list of position ids
        mlp_experts: optional expert list
        route_fn: optional routing function
        n_power_iter: number of power iteration steps
        seed: random seed for initial vector

    Returns:
        float — approximate log(spectral_radius) = λ_max
    """
    if len(token_ids) < 2:
        return 0.0

    from .model import embed, rmsnorm
    n_embd = sd['wte'].shape[1]

    # Get hidden state after first token
    graph.reset_kv()
    x = embed(token_ids[0], pos_ids[0], sd)
    for node in graph.nodes:
        if node.node_type == 'attn':
            from .model import attn_forward
            x = attn_forward(x, node.layer_idx, sd,
                             node.kv_keys, node.kv_values,
                             graph.n_head, graph.head_dim)
            break  # just the first attn layer

    # Define one-step map: hidden state -> hidden state through MLP
    def one_step(h):
        h_normed = rmsnorm(h)
        # Use first expert's MLP as representative
        out = mx.maximum(sd['layer0.mlp_fc1'] @ h_normed, 0)
        out = sd['layer0.mlp_fc2'] @ out
        return h + out  # residual connection

    # Power iteration to estimate spectral radius
    v = mx.random.normal((n_embd,), key=mx.random.key(seed))
    v = v / (mx.sqrt(mx.sum(v * v)) + 1e-8)

    for _ in range(n_power_iter):
        # Jacobian-vector product: J @ v where J = d(one_step)/dx
        # mx.jvp returns (primals_out: list, tangents_out: list)
        _, tangents_out = mx.jvp(one_step, primals=[x], tangents=[v])
        jv = tangents_out[0]
        jv_norm = mx.sqrt(mx.sum(jv * jv) + 1e-12)
        v = jv / jv_norm
        mx.eval(v, jv_norm)

    # Final estimate: λ_max ≈ log(‖J·v‖)
    _, tangents_final = mx.jvp(one_step, primals=[x], tangents=[v])
    jv_final = tangents_final[0]
    jv_norm_final = mx.sqrt(mx.sum(jv_final * jv_final) + 1e-12)
    mx.eval(jv_norm_final)

    lambda_max = math.log(max(jv_norm_final.item(), 1e-12))
    return lambda_max


def lyapunov_regularizer(graph, sd, token_id, pos_id):
    """Compute λ_max² as a differentiable regularizer.

    Returns mx.array scalar that can be added to the loss for gradient computation.

    Args:
        graph: Graph instance
        sd: state_dict (traced by value_and_grad)
        token_id: single token id
        pos_id: single position id

    Returns:
        mx.array scalar — λ_max² (drives λ_max toward 0)
    """
    from .model import embed, rmsnorm

    x = embed(token_id, pos_id, sd)

    def one_step(h):
        h_normed = rmsnorm(h)
        out = mx.maximum(sd['layer0.mlp_fc1'] @ h_normed, 0)
        out = sd['layer0.mlp_fc2'] @ out
        return h + out

    # Single power iteration (cheap but rough)
    n_embd = x.shape[0]
    # Use deterministic "random" vector based on input
    v = x / (mx.sqrt(mx.sum(x * x)) + 1e-8)

    _, tangents_out = mx.jvp(one_step, primals=[x], tangents=[v])
    jv = tangents_out[0]
    jv_norm = mx.sqrt(mx.sum(jv * jv) + 1e-12)

    # λ_max ≈ log(‖J·v‖), regularize λ_max² = log(‖J·v‖)²
    log_norm = mx.log(jv_norm + 1e-12)
    return log_norm * log_norm


def rate_distortion_bound(n_unfrozen_params, task_similarity, bits_per_param=32):
    """Estimate theoretical lower bound on forgetting given parameter budget.

    Based on Shannon's rate-distortion theory: minimum distortion (forgetting)
    achievable given a capacity constraint (number of unfrozen parameters).

    Args:
        n_unfrozen_params: number of trainable parameters
        task_similarity: estimated similarity between tasks (0 = orthogonal, 1 = identical)
        bits_per_param: effective bits per parameter (default 32 for float32)

    Returns:
        float — theoretical minimum forgetting (D_min)
    """
    # Rate: total information capacity
    R = n_unfrozen_params * bits_per_param

    # For dissimilar tasks, more capacity needed to remember old while learning new
    # D(R) ≈ (1 - task_similarity) * exp(-R / R_0) where R_0 is a scale constant
    # R_0 chosen so that at R = n_params * 32 bits, we get meaningful predictions
    R_0 = max(n_unfrozen_params * 8, 1)  # scale factor

    D_min = (1 - task_similarity) * math.exp(-R / R_0)
    return D_min
