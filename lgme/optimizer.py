"""Adam optimizer step — dict-based for MLX arrays.

Includes standard Adam and evolutionary Adam (evo_adam_step) which uses
gradient-guided candidates + evolutionary selection to prevent forgetting.
"""

import mlx.core as mx
import random as _random


def adam_step(sd, grads, adam_m, adam_v, lr_t, step, beta1=0.85, beta2=0.99, eps=1e-8,
              frozen_keys=None, freeze_map=None):
    """One Adam optimizer step. Updates sd in place.

    Args:
        sd: state_dict {key: mx.array}
        grads: gradient dict {key: mx.array} (same keys as sd)
        adam_m: first moment dict {key: mx.array}
        adam_v: second moment dict {key: mx.array}
        frozen_keys: optional set of keys to skip (no gradient update).
        freeze_map: optional dict {key: float} with temperatures in [0, 1].
            When provided, gradients are scaled by tau. Keys with tau < 0.01
            are effectively frozen. Takes precedence over frozen_keys.
    """
    for key in sd:
        if key not in grads:
            continue

        # Freeze map takes precedence: continuous temperature scaling
        if freeze_map is not None:
            tau = freeze_map.get(key, 1.0)
            if tau < 0.01:
                continue  # effectively frozen
        elif frozen_keys and key in frozen_keys:
            continue
        else:
            tau = 1.0

        g = grads[key]
        adam_m[key] = beta1 * adam_m[key] + (1 - beta1) * g
        adam_v[key] = beta2 * adam_v[key] + (1 - beta2) * g * g
        m_hat = adam_m[key] / (1 - beta1 ** (step + 1))
        v_hat = adam_v[key] / (1 - beta2 ** (step + 1))
        update = lr_t * m_hat / (mx.sqrt(v_hat) + eps)
        sd[key] = sd[key] - tau * update


def evo_adam_step(sd, grads, adam_m, adam_v, lr_t, step,
                  eval_fn, n_candidates=5,
                  beta1=0.85, beta2=0.99, eps=1e-8,
                  frozen_keys=None, mutation_std=0.02, rng=None):
    """Evolutionary Adam: gradient-guided candidates + fitness selection.

    Instead of blindly applying the Adam update, creates N candidate weight
    sets and picks the one with best multi-task fitness.

    Candidates:
      0: standard Adam update (full step)
      1: half-step (conservative)
      2-N: Adam update + random mutations on expert keys

    Args:
        sd: state_dict (modified in place to best candidate)
        grads: gradient dict
        adam_m, adam_v: Adam moment buffers (updated in place)
        lr_t: current learning rate
        step: current step number
        eval_fn: callable(sd) -> float loss (evaluates on combined task data)
        n_candidates: number of candidate updates to try
        frozen_keys: keys to never modify
        mutation_std: std of Gaussian noise for mutation candidates
        rng: random.Random instance
    """
    if rng is None:
        rng = _random.Random(step)

    if frozen_keys is None:
        frozen_keys = set()

    # Step 1: Compute the standard Adam update for all trainable keys
    updates = {}
    for key in sd:
        if key not in grads or key in frozen_keys:
            continue
        g = grads[key]
        adam_m[key] = beta1 * adam_m[key] + (1 - beta1) * g
        adam_v[key] = beta2 * adam_v[key] + (1 - beta2) * g * g
        m_hat = adam_m[key] / (1 - beta1 ** (step + 1))
        v_hat = adam_v[key] / (1 - beta2 ** (step + 1))
        updates[key] = lr_t * m_hat / (mx.sqrt(v_hat) + eps)

    if not updates:
        return

    # Save original weights
    originals = {k: mx.array(sd[k]) for k in updates}

    # Step 2: Generate candidates
    best_loss = float('inf')
    best_candidate = None

    for c in range(n_candidates):
        # Apply candidate update
        for key in updates:
            if c == 0:
                # Full Adam step
                sd[key] = originals[key] - updates[key]
            elif c == 1:
                # Half step (conservative)
                sd[key] = originals[key] - 0.5 * updates[key]
            else:
                # Adam step + mutation
                noise_scale = mutation_std * (1.0 + rng.random())
                noise = mx.random.normal(updates[key].shape) * noise_scale
                sd[key] = originals[key] - updates[key] + noise

        mx.eval(sd)

        # Evaluate this candidate
        loss = eval_fn(sd)

        if loss < best_loss:
            best_loss = loss
            best_candidate = {k: mx.array(sd[k]) for k in updates}

    # Step 3: Apply best candidate
    if best_candidate is not None:
        for key in best_candidate:
            sd[key] = best_candidate[key]
    else:
        # Fallback: standard Adam (candidate 0)
        for key in updates:
            sd[key] = originals[key] - updates[key]


def pareto_adam_step(sd, grads, adam_m, adam_v, lr_t, step,
                     eval_new_fn, eval_old_fn,
                     n_candidates=4,
                     beta1=0.85, beta2=0.99, eps=1e-8,
                     frozen_keys=None, mutation_std=0.01,
                     old_loss_slack=0.05, rng=None):
    """Pareto Adam: maximize new-task learning under old-task constraint.

    Two-objective selection:
      1. Filter candidates where old_loss <= baseline_old + slack
      2. Among passing candidates, pick lowest new_loss
      3. If none pass, pick lowest old_loss (preserve knowledge)

    Candidates:
      0: full Adam step
      1: 1.5x Adam step (aggressive learning)
      2: half step (conservative)
      3+: Adam step + mutation

    Args:
        eval_new_fn: callable(sd) -> float (new task loss)
        eval_old_fn: callable(sd) -> float (old task loss)
        old_loss_slack: max allowed increase in old-task loss over baseline
    """
    if rng is None:
        rng = _random.Random(step)

    if frozen_keys is None:
        frozen_keys = set()

    # Compute Adam updates
    updates = {}
    for key in sd:
        if key not in grads or key in frozen_keys:
            continue
        g = grads[key]
        adam_m[key] = beta1 * adam_m[key] + (1 - beta1) * g
        adam_v[key] = beta2 * adam_v[key] + (1 - beta2) * g * g
        m_hat = adam_m[key] / (1 - beta1 ** (step + 1))
        v_hat = adam_v[key] / (1 - beta2 ** (step + 1))
        updates[key] = lr_t * m_hat / (mx.sqrt(v_hat) + eps)

    if not updates:
        return

    originals = {k: mx.array(sd[k]) for k in updates}

    # Measure baseline old-task loss (before any update)
    baseline_old = eval_old_fn(sd)
    threshold = baseline_old + old_loss_slack

    # Evaluate candidates
    candidates = []  # list of (new_loss, old_loss, weights_dict)

    for c in range(n_candidates):
        for key in updates:
            if c == 0:
                # Full Adam step
                sd[key] = originals[key] - updates[key]
            elif c == 1:
                # Aggressive: 1.5x step (learn faster)
                sd[key] = originals[key] - 1.5 * updates[key]
            elif c == 2:
                # Conservative: half step
                sd[key] = originals[key] - 0.5 * updates[key]
            else:
                # Adam step + mutation
                noise_scale = mutation_std * (1.0 + rng.random())
                noise = mx.random.normal(updates[key].shape) * noise_scale
                sd[key] = originals[key] - updates[key] + noise

        mx.eval(sd)

        new_loss = eval_new_fn(sd)
        old_loss = eval_old_fn(sd)
        candidates.append((new_loss, old_loss, {k: mx.array(sd[k]) for k in updates}))

    # Pareto selection: filter by old-task constraint, pick best new-task
    passing = [(nl, ol, w) for nl, ol, w in candidates if ol <= threshold]

    if passing:
        # Among candidates that don't hurt old task, pick best new-task learner
        best = min(passing, key=lambda x: x[0])
    else:
        # No candidate passes — pick the one that hurts old task least
        best = min(candidates, key=lambda x: x[1])

    for key in best[2]:
        sd[key] = best[2][key]
