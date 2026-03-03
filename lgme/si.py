"""Synaptic Intelligence — online importance-weighted regularization.

Tracks parameter importance *during* training via path integrals,
unlike EWC which estimates importance post-hoc via Fisher.

Reference: Zenke, Poole & Ganguli (2017) — "Continual Learning through
Synaptic Intelligence"
"""

import mlx.core as mx


def si_init(sd, param_keys):
    """Initialize SI tracking state for a set of parameter keys.

    Args:
        sd: state_dict {key: mx.array}
        param_keys: set of keys to track (e.g. attention param keys)

    Returns:
        dict with keys:
            'omega': accumulated importance per key (mx.array)
            'prev_params': param values at last consolidation (mx.array)
            'running_sum': online gradient * delta accumulator (mx.array)
            'keys': the tracked param keys
    """
    return {
        'omega': {k: mx.zeros_like(sd[k]) for k in param_keys},
        'prev_params': {k: sd[k] for k in param_keys},
        'running_sum': {k: mx.zeros_like(sd[k]) for k in param_keys},
        'keys': param_keys,
    }


def si_accumulate(sd, grads, si_state):
    """Accumulate importance signal after an optimizer step.

    Must be called AFTER adam_step (which updates sd).
    grads should be from BEFORE adam_step (from value_and_grad).

    Args:
        sd: state_dict (after optimizer step)
        grads: gradient dict from value_and_grad
        si_state: state dict from si_init
    """
    for k in si_state['keys']:
        if k in grads:
            delta = sd[k] - si_state['prev_params'][k]
            si_state['running_sum'][k] = si_state['running_sum'][k] + grads[k] * delta
            si_state['prev_params'][k] = sd[k]


def si_consolidate(sd, si_state, damping=0.1):
    """Consolidate importance at task boundary.

    Transfers running_sum into omega (permanent importance) and resets
    for the next task.

    Args:
        sd: state_dict
        si_state: state dict from si_init
        damping: small constant for numerical stability (xi in paper)
    """
    for k in si_state['keys']:
        delta_sq = (sd[k] - si_state['prev_params'][k]) ** 2 + damping
        si_state['omega'][k] = si_state['omega'][k] + si_state['running_sum'][k] / delta_sq
        si_state['running_sum'][k] = mx.zeros_like(sd[k])
    si_state['prev_params'] = {k: sd[k] for k in si_state['keys']}
    mx.eval(si_state['omega'], si_state['prev_params'], si_state['running_sum'])


def si_penalty(sd, si_state, lambda_si):
    """Compute SI penalty: lambda * sum_k omega_k * (theta_k - theta_star_k)^2.

    Args:
        sd: current state_dict (traced by value_and_grad)
        si_state: state dict (after consolidation)
        lambda_si: regularization strength

    Returns:
        mx.array scalar — penalty term to add to loss (differentiable)
    """
    penalty = mx.array(0.0)
    for k in si_state['keys']:
        omega = si_state['omega'][k]
        diff = sd[k] - si_state['prev_params'][k]
        penalty = penalty + mx.sum(omega * diff * diff)
    return lambda_si * penalty
