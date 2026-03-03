"""Forward pass building blocks — linear, softmax, rmsnorm, embed, attn, mlp, output.

All ops use mlx.core arrays for Metal GPU acceleration.
"""

import mlx.core as mx


def linear(x, w):
    return w @ x


def softmax(logits):
    return mx.softmax(logits)


def rmsnorm(x):
    return x * mx.rsqrt(mx.mean(x * x) + 1e-5)


def embed(token_id, pos_id, sd):
    return rmsnorm(sd['wte'][token_id] + sd['wpe'][pos_id])


def attn_forward(x, li, sd, kv_keys, kv_values, n_head, head_dim):
    """Multi-head attention with KV caching.

    Args:
        x: (n_embd,) input hidden state
        li: layer index (for sd key lookup only)
        sd: state_dict of mx.arrays
        kv_keys: list to append keys to (flat, not indexed by layer)
        kv_values: list to append values to (flat, not indexed by layer)
        n_head: number of attention heads
        head_dim: dimension per head
    """
    x_residual = x
    x = rmsnorm(x)
    q = sd[f'layer{li}.attn_wq'] @ x
    k = sd[f'layer{li}.attn_wk'] @ x
    v = sd[f'layer{li}.attn_wv'] @ x
    kv_keys.append(k)
    kv_values.append(v)

    K = mx.stack(kv_keys)    # (seq_len, n_embd)
    V = mx.stack(kv_values)  # (seq_len, n_embd)

    heads = []
    for h in range(n_head):
        hs = h * head_dim
        he = hs + head_dim
        q_h = q[hs:he]          # (head_dim,)
        K_h = K[:, hs:he]       # (seq_len, head_dim)
        V_h = V[:, hs:he]       # (seq_len, head_dim)

        scores = K_h @ q_h / (head_dim ** 0.5)  # (seq_len,)
        weights = mx.softmax(scores)              # (seq_len,)
        head_out = V_h.T @ weights                # (head_dim,)
        heads.append(head_out)

    x_attn = mx.concatenate(heads)               # (n_embd,)
    x = sd[f'layer{li}.attn_wo'] @ x_attn
    return x + x_residual


def mlp_forward(x, sd, fc1_key, fc2_key):
    x_residual = x
    x = rmsnorm(x)
    x = mx.maximum(sd[fc1_key] @ x, 0)
    x = sd[fc2_key] @ x
    return x + x_residual


def output_forward(x, sd):
    return sd['lm_head'] @ x
