"""Adaptive Freeze Maps with Expert Inheritance (AFMEI).

Phase 1: Continuous freeze maps with task-boundary decay.
Phase 2: Offspring-based freeze map updates — child experts evaluate parents
         and update their freeze maps based on comparative performance.
"""

import mlx.core as mx
import random as _random


def init_freeze_map(sd, initial_tau=1.0):
    """Create freeze map with all parameters at initial temperature.

    Args:
        sd: state_dict
        initial_tau: starting temperature (1.0 = fully trainable)

    Returns:
        dict mapping key -> tau float
    """
    return {k: initial_tau for k in sd}


def decay_freeze_map(freeze_map, decay_rate=0.7, exclude_keys=None):
    """Decay all temperatures toward frozen at task boundary.

    tau_new = tau_old * decay_rate

    Args:
        freeze_map: dict mapping key -> tau (modified in place)
        decay_rate: multiplicative decay (0.7 = 30% consolidation per boundary)
        exclude_keys: optional set of keys to skip (keep fully trainable)
    """
    if exclude_keys is None:
        exclude_keys = set()
    for k in freeze_map:
        if k not in exclude_keys:
            freeze_map[k] *= decay_rate


def set_expert_tau(freeze_map, expert, tau):
    """Set freeze temperature for all of an expert's parameters.

    Args:
        freeze_map: dict mapping key -> tau (modified in place)
        expert: expert dict with 'fc1', 'fc2', optionally 'lm_head'
        tau: new temperature value
    """
    for key_name in ['fc1', 'fc2', 'lm_head']:
        if key_name in expert:
            freeze_map[expert[key_name]] = tau


def create_offspring(sd, parent_a, parent_b, child_id, n_embd,
                     strategy='blend', alpha=0.5, noise_std=0.01, rng=None):
    """Create a child expert from two parents via parameter crossover.

    Args:
        sd: state_dict (modified in place — new keys added)
        parent_a, parent_b: expert dicts with 'fc1', 'fc2', optionally 'lm_head'
        child_id: integer id for the new expert
        n_embd: embedding dimension (for router key)
        strategy: 'blend' (weighted average) or 'select' (pick better per-key)
        alpha: blend weight for parent_a (1-alpha for parent_b)
        noise_std: noise added for diversity
        rng: random.Random instance

    Returns:
        child expert dict
    """
    if rng is None:
        rng = _random.Random(0)

    child_fc1 = f'expert{child_id}.mlp_fc1'
    child_fc2 = f'expert{child_id}.mlp_fc2'

    key_pairs = [(parent_a['fc1'], parent_b['fc1'], child_fc1),
                 (parent_a['fc2'], parent_b['fc2'], child_fc2)]

    for key_a, key_b, key_child in key_pairs:
        if strategy == 'blend':
            sd[key_child] = alpha * sd[key_a] + (1 - alpha) * sd[key_b]
        elif strategy == 'select':
            # Per-row selection: randomly pick rows from either parent
            rows = sd[key_a].shape[0]
            mask = mx.array([1.0 if rng.random() < alpha else 0.0
                             for _ in range(rows)])
            sd[key_child] = (mask[:, None] * sd[key_a]
                             + (1 - mask[:, None]) * sd[key_b])
        # Add small noise for diversity
        shape = sd[key_child].shape
        noise = mx.array([[rng.gauss(0, noise_std) for _ in range(shape[1])]
                          for _ in range(shape[0])])
        sd[key_child] = sd[key_child] + noise

    # Router key: average of parents + noise (router keys are Python lists, not sd keys)
    rk_a = parent_a['router_key']
    rk_b = parent_b['router_key']
    child_rk = [(a + b) / 2 + rng.gauss(0, noise_std)
                for a, b in zip(rk_a, rk_b)]

    child = {
        'id': child_id,
        'fc1': child_fc1,
        'fc2': child_fc2,
        'router_key': child_rk,
        'activation_count': 0,
        'parent_ids': {parent_a['id'], parent_b['id']},
    }

    # Handle lm_head if parents have them
    if 'lm_head' in parent_a and 'lm_head' in parent_b:
        child_head = f'expert{child_id}.lm_head'
        sd[child_head] = alpha * sd[parent_a['lm_head']] + (1 - alpha) * sd[parent_b['lm_head']]
        shape = sd[child_head].shape
        noise = mx.array([[rng.gauss(0, noise_std) for _ in range(shape[1])]
                          for _ in range(shape[0])])
        sd[child_head] = sd[child_head] + noise
        child['lm_head'] = child_head

    return child


def offspring_update_freeze_map(freeze_map, parent, child_loss, parent_loss,
                                thaw_amount=0.3, freeze_amount=0.2):
    """Update a parent's freeze map based on offspring performance comparison.

    Core AFMEI mechanism: the offspring evaluates the parent.
    - If offspring outperforms parent on old data → thaw parent (it can improve)
    - If parent outperforms offspring on old data → freeze parent more (knowledge is valuable)

    Args:
        freeze_map: dict mapping key -> tau (modified in place)
        parent: expert dict
        child_loss: offspring's loss on old task data
        parent_loss: parent's loss on old task data
        thaw_amount: how much to increase tau when offspring wins
        freeze_amount: how much to decrease tau when parent wins
    """
    for key_name in ['fc1', 'fc2', 'lm_head']:
        if key_name not in parent:
            continue
        k = parent[key_name]
        if k not in freeze_map:
            continue

        if child_loss < parent_loss:
            # Offspring better → parent's knowledge is suboptimal → thaw
            freeze_map[k] = min(freeze_map[k] + thaw_amount, 1.0)
        else:
            # Parent better → parent's knowledge is valuable → freeze more
            freeze_map[k] = max(freeze_map[k] - freeze_amount, 0.0)


def offspring_update_freeze_map_per_key(freeze_map, parent, child_grads,
                                         parent_grads, alpha_thaw=0.2):
    """Update parent freeze map using per-key gradient agreement with offspring.

    More granular than loss-based: uses gradient direction agreement per weight key.
    - Where child and parent gradients agree → safe to thaw (shared improvement direction)
    - Where they disagree → keep frozen (conflicting objectives)

    Args:
        freeze_map: dict mapping key -> tau (modified in place)
        parent: expert dict
        child_grads: gradient dict from offspring's loss
        parent_grads: gradient dict from parent's loss on old data
        alpha_thaw: maximum thaw amount per update
    """
    for key_name in ['fc1', 'fc2', 'lm_head']:
        if key_name not in parent:
            continue
        k = parent[key_name]
        if k not in freeze_map or k not in child_grads or k not in parent_grads:
            continue

        gc = child_grads[k].reshape(-1)
        gp = parent_grads[k].reshape(-1)
        norm_c = mx.sqrt(mx.sum(gc * gc))
        norm_p = mx.sqrt(mx.sum(gp * gp))
        cos = mx.sum(gc * gp) / (norm_c * norm_p + 1e-8)
        agreement = max(cos.item(), 0.0)  # only thaw on positive agreement

        # Scale thaw by how strongly both gradients push (normalized)
        grad_strength = min(norm_c.item(), norm_p.item()) / (
            max(norm_c.item(), norm_p.item()) + 1e-8)

        delta = alpha_thaw * agreement * grad_strength
        freeze_map[k] = min(freeze_map[k] + delta, 1.0)
