"""SOM (Self-Organizing Map) topology for expert routing.

1D grid topology with proper neighborhood functions, sigma decay,
error-driven growth, and lateral weight computation.
"""

import math


def som_init(num_positions, sigma_start=2.0, sigma_end=0.3, total_steps=1000):
    """Initialize SOM state for a 1D grid of expert positions.

    Args:
        num_positions: initial number of grid positions
        sigma_start: initial neighborhood radius
        sigma_end: final neighborhood radius
        total_steps: total training steps for sigma schedule

    Returns:
        dict with SOM state
    """
    return {
        'grid_size': num_positions,
        'sigma_start': sigma_start,
        'sigma_end': sigma_end,
        'total_steps': total_steps,
        'error_accumulator': {i: 0.0 for i in range(num_positions)},
        'error_counts': {i: 0 for i in range(num_positions)},
    }


def som_sigma(som_state, step):
    """Compute current neighborhood radius via exponential decay.

    sigma(t) = sigma_start * (sigma_end / sigma_start) ^ (t / T)
    """
    ratio = som_state['sigma_end'] / som_state['sigma_start']
    t_frac = min(step / max(som_state['total_steps'], 1), 1.0)
    return som_state['sigma_start'] * (ratio ** t_frac)


def som_neighborhood(pos_i, pos_j, sigma):
    """Gaussian neighborhood function on 1D grid.

    h(i,j) = exp(-d^2 / (2 * sigma^2))
    """
    d = abs(pos_i - pos_j)
    if sigma < 1e-8:
        return 1.0 if d == 0 else 0.0
    return math.exp(-(d * d) / (2 * sigma * sigma))


def som_grow(som_state, experts):
    """Insert a new grid position at the highest-error location.

    Shifts existing positions above the insertion point to make room.

    Args:
        som_state: SOM state dict
        experts: list of expert dicts (each with 'som_pos')

    Returns:
        int — the new grid position for the spawned expert
    """
    # Find position with highest mean error
    errors = som_state['error_accumulator']
    counts = som_state['error_counts']
    mean_errors = {}
    for pos in errors:
        if counts.get(pos, 0) > 0:
            mean_errors[pos] = errors[pos] / counts[pos]
        else:
            mean_errors[pos] = 0.0

    if not mean_errors:
        new_pos = som_state['grid_size']
    else:
        best_pos = max(mean_errors, key=mean_errors.get)
        # Insert new position right after the highest-error position
        new_pos = best_pos + 1

    # Shift all positions >= new_pos up by 1
    for exp in experts:
        if exp.get('som_pos', 0) >= new_pos:
            exp['som_pos'] += 1

    # Update error accumulator keys
    old_errors = dict(som_state['error_accumulator'])
    old_counts = dict(som_state['error_counts'])
    som_state['error_accumulator'] = {}
    som_state['error_counts'] = {}
    for pos in sorted(old_errors.keys()):
        if pos >= new_pos:
            som_state['error_accumulator'][pos + 1] = old_errors[pos]
            som_state['error_counts'][pos + 1] = old_counts.get(pos, 0)
        else:
            som_state['error_accumulator'][pos] = old_errors[pos]
            som_state['error_counts'][pos] = old_counts.get(pos, 0)

    # Initialize new position with zero error
    som_state['error_accumulator'][new_pos] = 0.0
    som_state['error_counts'][new_pos] = 0
    som_state['grid_size'] += 1

    return new_pos


def som_accumulate_error(som_state, bmu_pos, x_data, router_key):
    """Track quantization error at the BMU position.

    Error = ||x - router_key||^2 (squared Euclidean distance).

    Args:
        som_state: SOM state dict
        bmu_pos: grid position of best-matching unit
        x_data: input vector (list of floats)
        router_key: BMU's router key (list of floats)
    """
    error = sum((a - b) ** 2 for a, b in zip(x_data, router_key))
    if bmu_pos not in som_state['error_accumulator']:
        som_state['error_accumulator'][bmu_pos] = 0.0
        som_state['error_counts'][bmu_pos] = 0
    som_state['error_accumulator'][bmu_pos] += error
    som_state['error_counts'][bmu_pos] += 1


def som_get_lateral_weights(experts, bmu_pos, sigma):
    """Compute SOM neighborhood weights from BMU to all experts.

    Args:
        experts: list of expert dicts (each with 'som_pos')
        bmu_pos: grid position of best-matching unit
        sigma: current neighborhood radius

    Returns:
        dict mapping expert index → neighborhood weight
    """
    weights = {}
    for idx, exp in enumerate(experts):
        h = som_neighborhood(bmu_pos, exp.get('som_pos', 0), sigma)
        if h > 0.01:
            weights[idx] = h
    return weights
