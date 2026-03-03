"""Router — multi-expert MLP routing, Kohonen updates, spawn/consolidation.

Also implements cross-disciplinary mechanisms:
- Niche exclusion dynamics (ecology: Gause's competitive exclusion)
- Idiotypic self-regulation (immunology: Jerne's idiotypic network)
- Replicator dynamics routing (evolutionary game theory)
- Dual-process expert composition (System 1 parallel / System 2 sequential)
"""

import math
import random
import mlx.core as mx
from .model import rmsnorm
from .som import som_sigma, som_neighborhood, som_accumulate_error


# ── MLX-Vectorized Routing Primitives ──

def cosine_sim(a, b):
    """Cosine similarity between two float vectors (Python lists)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 + 1e-8
    nb = sum(x * x for x in b) ** 0.5 + 1e-8
    return dot / (na * nb)


def build_router_key_matrix(experts):
    """Stack router keys into an mx.array for vectorized scoring.

    Returns:
        mx.array (K, n_embd) — one row per expert
    """
    return mx.array([exp['router_key'] for exp in experts])


def compute_scores_mx(x, router_keys_mx):
    """Vectorized cosine similarity: x against all router keys at once.

    Args:
        x: mx.array (n_embd,)
        router_keys_mx: mx.array (K, n_embd)

    Returns:
        mx.array (K,) — cosine similarity scores
    """
    dots = router_keys_mx @ x                                      # (K,)
    key_norms = mx.sqrt(mx.sum(router_keys_mx * router_keys_mx, axis=1))  # (K,)
    x_norm = mx.sqrt(mx.sum(x * x))                               # scalar
    return dots / (key_norms * x_norm + 1e-8)


def batch_expert_mlp(sd, experts, x_normed, indices):
    """Batched expert MLP forward pass — single matmul instead of loop.

    Args:
        sd: state_dict
        experts: list of expert dicts
        x_normed: mx.array (n_embd,) — rmsnorm'd hidden state
        indices: list of int — which expert indices to compute

    Returns:
        mx.array (K, n_embd) — expert MLP outputs (no residual)
    """
    selected = [experts[i] for i in indices]
    K = len(selected)
    if K == 0:
        return mx.zeros((0, x_normed.shape[0]))

    # Stack weight matrices: (K, 4*n_embd, n_embd) and (K, n_embd, 4*n_embd)
    fc1_stack = mx.stack([sd[exp['fc1']] for exp in selected])
    fc2_stack = mx.stack([sd[exp['fc2']] for exp in selected])

    # Batched fc1: (K, 4*n_embd, n_embd) @ (n_embd,) → (K, 4*n_embd)
    h = mx.maximum(fc1_stack @ x_normed, 0)

    # Batched fc2: (K, n_embd, 4*n_embd) @ (K, 4*n_embd, 1) → (K, n_embd, 1) → (K, n_embd)
    out = mx.squeeze(fc2_stack @ mx.expand_dims(h, -1), axis=-1)
    return out


def clone_mlp_expert(sd, src_fc1, src_fc2, new_fc1, new_fc2, noise_std=0.02, rng=None):
    """Deep-copy MLP weights from src keys into new keys, adding small noise."""
    if rng is None:
        rng = random.Random(0)
    rows1, cols1 = sd[src_fc1].shape
    rows2, cols2 = sd[src_fc2].shape
    noise1 = mx.array([[rng.gauss(0, noise_std) for _ in range(cols1)]
                        for _ in range(rows1)])
    noise2 = mx.array([[rng.gauss(0, noise_std) for _ in range(cols2)]
                        for _ in range(rows2)])
    sd[new_fc1] = sd[src_fc1] + noise1
    sd[new_fc2] = sd[src_fc2] + noise2


def init_router_key(sd, fc1_key, n_embd):
    """Compute initial router key from weight statistics of an expert's fc1."""
    return mx.mean(sd[fc1_key], axis=0).tolist()


def route_mlp_experts(x, experts, sd, top_k=2, router_keys_mx=None):
    """Route input x through top-K MLP experts by cosine similarity scoring.

    Uses vectorized mx operations when router_keys_mx is provided (faster).
    Falls back to Python cosine_sim when router_keys_mx is None.

    Args:
        x: mx.array (n_embd,) — current hidden state
        experts: list of dicts with keys 'fc1', 'fc2', 'router_key', 'id'
        sd: state_dict
        top_k: number of experts to activate
        router_keys_mx: optional precomputed mx.array (K, n_embd) from
                        build_router_key_matrix()

    Returns:
        x_out: mx.array (n_embd,) — weighted sum of expert outputs (with residual)
        selected: list of expert dicts that were activated
    """
    # Vectorized scoring when router_keys_mx available
    if router_keys_mx is not None:
        scores_mx = compute_scores_mx(x, router_keys_mx)
        # Top-K selection: extract indices and scores to Python for selection
        scores_list = scores_mx.tolist()
    else:
        x_data = x.tolist()
        scores_list = [cosine_sim(x_data, exp['router_key']) for exp in experts]

    indexed = sorted(enumerate(scores_list), key=lambda t: -t[1])[:top_k]
    selected_indices = [i for i, _ in indexed]
    selected_scores = [s for _, s in indexed]

    # Softmax weights over selected experts
    max_s = max(selected_scores)
    exp_s = [math.exp(s - max_s) for s in selected_scores]
    total = sum(exp_s)
    weights = [e / total for e in exp_s]

    x_normed = rmsnorm(x)

    # Batched expert forward
    out_batch = batch_expert_mlp(sd, experts, x_normed, selected_indices)

    selected = []
    for idx in selected_indices:
        exp = experts[idx]
        exp['activation_count'] = exp.get('activation_count', 0) + 1
        selected.append(exp)

    # Weighted sum with residual
    weights_mx = mx.array(weights)  # (K,)
    x_out = x + mx.sum(weights_mx[:, None] * out_batch, axis=0)

    return x_out, selected


def kohonen_update(expert, x_data, alpha=0.01):
    """Pull expert's router key toward the input pattern (gradient-free)."""
    rk = expert['router_key']
    for j in range(len(rk)):
        rk[j] += alpha * (x_data[j] - rk[j])


def spawn_mlp_expert(sd, src_expert, new_id, n_embd, noise_std=0.02, rng=None):
    """Clone an existing expert and initialize router key near the source."""
    new_fc1 = f'expert{new_id}.mlp_fc1'
    new_fc2 = f'expert{new_id}.mlp_fc2'
    clone_mlp_expert(sd, src_expert['fc1'], src_expert['fc2'], new_fc1, new_fc2, noise_std, rng=rng)
    new_key = init_router_key(sd, new_fc1, n_embd)
    return {
        'id': new_id,
        'fc1': new_fc1,
        'fc2': new_fc2,
        'router_key': new_key,
        'activation_count': 0,
    }


def consolidate_experts(experts, sd, threshold=0.9):
    """Merge experts whose router keys are too similar (cosine > threshold).
    Returns the surviving experts list and list of removed expert ids."""
    if len(experts) <= 1:
        return experts, []

    removed_ids = []
    i = 0
    while i < len(experts):
        j = i + 1
        while j < len(experts):
            cos = cosine_sim(experts[i]['router_key'], experts[j]['router_key'])
            if cos > threshold:
                keep, drop = (i, j) if experts[i].get('activation_count', 0) >= experts[j].get('activation_count', 0) else (j, i)
                removed_ids.append(experts[drop]['id'])
                sd.pop(experts[drop]['fc1'], None)
                sd.pop(experts[drop]['fc2'], None)
                experts.pop(drop)
                if drop < i:
                    i -= 1
            else:
                j += 1
        i += 1
    return experts, removed_ids


def kohonen_update_som(experts, bmu_idx, x_data, som_state, alpha, step):
    """SOM-aware Kohonen update: update BMU AND neighbors weighted by topology."""
    sigma = som_sigma(som_state, step)
    bmu_pos = experts[bmu_idx].get('som_pos', 0)
    for exp in experts:
        h = som_neighborhood(bmu_pos, exp.get('som_pos', 0), sigma)
        if h > 0.01:
            rk = exp['router_key']
            for j in range(len(rk)):
                rk[j] += alpha * h * (x_data[j] - rk[j])


def route_with_lateral(x, experts, sd, som_state, step,
                       frozen_expert_ids=None, top_k=2):
    """Route through experts using SOM lateral connections.

    Frozen experts contribute via SOM neighborhood weights, but their
    gradients are detached (mx.stop_gradient).
    """
    if frozen_expert_ids is None:
        frozen_expert_ids = set()

    x_data = x.tolist()
    sigma = som_sigma(som_state, step)

    scores = [cosine_sim(x_data, exp['router_key']) for exp in experts]
    bmu_idx = max(range(len(scores)), key=lambda i: scores[i])
    bmu_pos = experts[bmu_idx].get('som_pos', 0)

    som_accumulate_error(som_state, bmu_pos, x_data,
                         experts[bmu_idx]['router_key'])

    x_normed = rmsnorm(x)

    active_indices = []
    active_weights = []
    for idx, exp in enumerate(experts):
        h = som_neighborhood(bmu_pos, exp.get('som_pos', 0), sigma)
        if h > 0.01:
            active_indices.append(idx)
            active_weights.append(h)

    if not active_indices:
        active_indices = [bmu_idx]
        active_weights = [1.0]

    total_w = sum(active_weights)
    active_weights = [w / total_w for w in active_weights]

    x_out = x  # residual
    selected = []
    for idx, weight in zip(active_indices, active_weights):
        exp = experts[idx]
        exp['activation_count'] = exp.get('activation_count', 0) + 1
        selected.append(exp)

        h = mx.maximum(sd[exp['fc1']] @ x_normed, 0)
        out = sd[exp['fc2']] @ h

        if exp['id'] in frozen_expert_ids:
            out = mx.stop_gradient(out)

        x_out = x_out + weight * out

    return x_out, selected


# ── Mechanism 1: Niche Exclusion Dynamics (Ecology → F2) ──

def niche_overlap(experts, grads, sd):
    """Compute pairwise gradient-based niche overlap between experts.

    O_ij = |<∇L_i, ∇L_j>| / (‖∇L_i‖ · ‖∇L_j‖ + ε)

    Args:
        experts: list of expert dicts
        grads: gradient dict from mx.value_and_grad
        sd: state_dict

    Returns:
        dict mapping (i, j) → overlap float, and mean overlap float
    """
    # Collect per-expert gradient vectors
    expert_grads = {}
    for exp in experts:
        fc1_key, fc2_key = exp['fc1'], exp['fc2']
        if fc1_key in grads and fc2_key in grads:
            g1 = grads[fc1_key].reshape(-1)
            g2 = grads[fc2_key].reshape(-1)
            expert_grads[exp['id']] = mx.concatenate([g1, g2])

    overlaps = {}
    ids = list(expert_grads.keys())
    for i_idx in range(len(ids)):
        for j_idx in range(i_idx + 1, len(ids)):
            ei, ej = ids[i_idx], ids[j_idx]
            gi, gj = expert_grads[ei], expert_grads[ej]
            dot = mx.sum(gi * gj)
            ni = mx.sqrt(mx.sum(gi * gi)) + 1e-8
            nj = mx.sqrt(mx.sum(gj * gj)) + 1e-8
            o = mx.abs(dot) / (ni * nj)
            overlaps[(ei, ej)] = o.item()

    mean_overlap = sum(overlaps.values()) / max(len(overlaps), 1)
    return overlaps, mean_overlap


def niche_exclusion_penalty(experts, grads, sd, tau=0.5, gamma=1.0):
    """Compute niche exclusion loss: penalize expert pairs with overlap > tau.

    L_niche = γ · Σ_{i<j} max(0, O_ij - τ)²

    Args:
        experts: list of expert dicts
        grads: gradient dict (from previous step, used as running estimate)
        sd: state_dict (traced)
        tau: overlap threshold
        gamma: penalty strength

    Returns:
        mx.array scalar — penalty to add to loss
    """
    expert_grads = {}
    for exp in experts:
        fc1_key, fc2_key = exp['fc1'], exp['fc2']
        if fc1_key in grads and fc2_key in grads:
            g1 = grads[fc1_key].reshape(-1)
            g2 = grads[fc2_key].reshape(-1)
            expert_grads[exp['id']] = mx.concatenate([g1, g2])

    if len(expert_grads) < 2:
        return mx.array(0.0)

    penalty = mx.array(0.0)
    ids = list(expert_grads.keys())
    for i_idx in range(len(ids)):
        for j_idx in range(i_idx + 1, len(ids)):
            gi = expert_grads[ids[i_idx]]
            gj = expert_grads[ids[j_idx]]
            dot = mx.sum(gi * gj)
            ni = mx.sqrt(mx.sum(gi * gi)) + 1e-8
            nj = mx.sqrt(mx.sum(gj * gj)) + 1e-8
            o_ij = mx.abs(dot) / (ni * nj)
            excess = mx.maximum(o_ij - tau, 0.0)
            penalty = penalty + excess * excess

    return gamma * penalty


def niche_repulsion(experts, tau=0.5, alpha_repel=0.01):
    """Push router keys apart for expert pairs with high similarity.

    Gradient-free repulsive force on router keys.

    Args:
        experts: list of expert dicts (modified in place)
        tau: overlap threshold (only repel if router key similarity > tau)
        alpha_repel: repulsion step size
    """
    for i in range(len(experts)):
        for j in range(i + 1, len(experts)):
            rk_i = experts[i]['router_key']
            rk_j = experts[j]['router_key']
            sim = cosine_sim(rk_i, rk_j)
            if sim > tau:
                diff = [rk_i[d] - rk_j[d] for d in range(len(rk_i))]
                norm = (sum(d * d for d in diff) ** 0.5) + 1e-8
                for d in range(len(rk_i)):
                    rk_i[d] += alpha_repel * diff[d] / norm
                    rk_j[d] -= alpha_repel * diff[d] / norm


# ── Mechanism 2: Idiotypic Self-Regulation (Immunology → F1) ──

def route_with_idiotypic(x, experts, sd, frozen_expert_ids=None,
                         theta_suppress=0.0, top_k=2):
    """Route through experts with idiotypic inter-expert regulation.

    Inspired by Jerne's Idiotypic Network Theory: experts stimulate or
    suppress each other based on output agreement/disagreement, creating
    emergent homeostasis.

    Args:
        x: mx.array (n_embd,) — current hidden state
        experts: list of expert dicts
        sd: state_dict (traced)
        frozen_expert_ids: set of frozen expert ids
        theta_suppress: suppression threshold
        top_k: base number of experts to select

    Returns:
        x_out: mx.array (n_embd,) — modulated expert combination
        selected: list of expert dicts activated
    """
    if frozen_expert_ids is None:
        frozen_expert_ids = set()

    x_data = x.tolist()
    x_normed = rmsnorm(x)

    scores = [cosine_sim(x_data, exp['router_key']) for exp in experts]
    indexed = sorted(enumerate(scores), key=lambda t: -t[1])[:top_k]
    selected_indices = [i for i, _ in indexed]
    selected_scores = [s for _, s in indexed]

    max_s = max(selected_scores)
    exp_s = [math.exp(s - max_s) for s in selected_scores]
    total = sum(exp_s)
    base_weights = [e / total for e in exp_s]

    expert_outputs = []
    selected = []
    for idx in selected_indices:
        exp = experts[idx]
        exp['activation_count'] = exp.get('activation_count', 0) + 1
        selected.append(exp)
        h = mx.maximum(sd[exp['fc1']] @ x_normed, 0)
        out = sd[exp['fc2']] @ h
        if exp['id'] in frozen_expert_ids:
            out = mx.stop_gradient(out)
        expert_outputs.append(out)

    K = len(expert_outputs)
    if K <= 1:
        x_out = x
        for k in range(K):
            x_out = x_out + base_weights[k] * expert_outputs[k]
        return x_out, selected

    # Inter-expert interaction matrix I_ij = cosine_sim(out_i, out_j)
    interaction = []
    for i in range(K):
        row = []
        for j in range(K):
            if i == j:
                row.append(mx.array(1.0))
            else:
                dot = mx.sum(expert_outputs[i] * expert_outputs[j])
                ni = mx.sqrt(mx.sum(expert_outputs[i] ** 2)) + 1e-8
                nj = mx.sqrt(mx.sum(expert_outputs[j] ** 2)) + 1e-8
                row.append(dot / (ni * nj))
        interaction.append(row)

    # Idiotypic modulation: a_i' = a_i · σ(Σ_j I_ij · a_j - θ_suppress)
    modulated_weights = []
    for i in range(K):
        stimulation = mx.array(0.0)
        for j in range(K):
            stimulation = stimulation + interaction[i][j] * base_weights[j]
        modulation = mx.sigmoid(stimulation - theta_suppress)
        modulated_weights.append(base_weights[i] * modulation)

    # Re-normalize
    total_mod = sum(w.item() if isinstance(w, mx.array) else w
                    for w in modulated_weights)
    if total_mod < 1e-8:
        total_mod = 1.0
    modulated_weights = [w / total_mod for w in modulated_weights]

    x_out = x  # residual
    for k in range(K):
        x_out = x_out + modulated_weights[k] * expert_outputs[k]

    return x_out, selected


# ── Mechanism 4: Replicator Dynamics Routing (Evolutionary Game Theory → F3) ──

def replicator_init(experts):
    """Initialize replicator dynamics state with uniform allocation.

    Returns:
        dict with allocation weights and parameters
    """
    n = len(experts)
    return {
        'allocation': {exp['id']: 1.0 / n for exp in experts},
        'beta': 1.0,
    }


def replicator_update(replicator_state, expert_fitness, beta=None):
    """Update allocation using discrete-time replicator equation.

    x_i(t+1) = x_i(t) · exp(β · (π_i - π̄)) / Z

    Args:
        replicator_state: dict with 'allocation' weights
        expert_fitness: dict mapping expert_id → fitness (negative loss)
        beta: selection pressure (overrides state if provided)
    """
    alloc = replicator_state['allocation']
    if beta is None:
        beta = replicator_state['beta']

    if not expert_fitness:
        return

    pi_bar = sum(alloc.get(eid, 0) * fit
                 for eid, fit in expert_fitness.items())

    new_alloc = {}
    for eid, x_i in alloc.items():
        pi_i = expert_fitness.get(eid, pi_bar)
        new_alloc[eid] = x_i * math.exp(beta * (pi_i - pi_bar))

    total = sum(new_alloc.values())
    if total < 1e-12:
        n = len(new_alloc)
        for eid in new_alloc:
            new_alloc[eid] = 1.0 / n
    else:
        for eid in new_alloc:
            new_alloc[eid] /= total

    replicator_state['allocation'] = new_alloc


def route_replicator(x, experts, sd, replicator_state,
                     frozen_expert_ids=None, top_k=2):
    """Route using replicator dynamics allocation as expert weights.

    Experts selected by allocation weight, outputs weighted by allocation.

    Returns:
        x_out: mx.array (n_embd,)
        selected: list of expert dicts activated
    """
    if frozen_expert_ids is None:
        frozen_expert_ids = set()

    alloc = replicator_state['allocation']

    sorted_experts = sorted(experts, key=lambda e: -alloc.get(e['id'], 0))
    top_experts = sorted_experts[:top_k]

    weights_raw = [alloc.get(e['id'], 1e-6) for e in top_experts]
    total_w = sum(weights_raw)
    weights = [w / total_w for w in weights_raw]

    x_normed = rmsnorm(x)

    expert_outputs = []
    selected = []
    for exp in top_experts:
        exp['activation_count'] = exp.get('activation_count', 0) + 1
        selected.append(exp)
        h = mx.maximum(sd[exp['fc1']] @ x_normed, 0)
        out = sd[exp['fc2']] @ h
        if exp['id'] in frozen_expert_ids:
            out = mx.stop_gradient(out)
        expert_outputs.append(out)

    x_out = x  # residual
    for k, w in enumerate(weights):
        x_out = x_out + w * expert_outputs[k]

    return x_out, selected


def replicator_compute_fitness(experts, sd, x_normed):
    """Compute per-expert fitness as negative output magnitude (proxy).

    Args:
        experts: list of expert dicts
        sd: state_dict
        x_normed: rmsnorm'd hidden state

    Returns:
        dict mapping expert_id → fitness
    """
    fitness = {}
    for exp in experts:
        h = mx.maximum(sd[exp['fc1']] @ x_normed, 0)
        out = sd[exp['fc2']] @ h
        fitness[exp['id']] = -mx.sum(out * out).item()
    return fitness


# ── Expert-Specific Output Heads (Allosteric Output Projection) ──

def init_expert_heads(sd, experts, noise_std=0.02, rng=None):
    """Create per-expert output heads (lm_head copies) in the state_dict.

    Each expert gets its own lm_head_i cloned from the shared lm_head
    with small noise. This allows freezing expert i's full pipeline
    (MLP_i + lm_head_i) independently.

    Args:
        sd: state_dict (modified in place)
        experts: list of expert dicts (modified in place — adds 'lm_head' key)
        noise_std: noise std for cloning
        rng: random.Random instance
    """
    if rng is None:
        rng = random.Random(0)
    shared_lm = sd['lm_head']
    rows, cols = shared_lm.shape
    for exp in experts:
        head_key = f'expert{exp["id"]}.lm_head'
        noise = mx.array([[rng.gauss(0, noise_std) for _ in range(cols)]
                          for _ in range(rows)])
        sd[head_key] = shared_lm + noise
        exp['lm_head'] = head_key


def route_with_expert_heads(x, experts, sd, frozen_expert_ids=None, top_k=2,
                            router_keys_mx=None, guaranteed_ids=None):
    """Route through top-K experts, each producing logits via its own lm_head.

    Instead of: experts → shared hidden → shared lm_head → logits
    Does:       expert_i → hidden_i → lm_head_i → logits_i → weighted merge

    This isolates each expert's full output pipeline so freezing an expert
    also freezes its output projection (the #1 source of forgetting at 43%).

    Args:
        x: mx.array (n_embd,) — hidden state after attention
        experts: list of expert dicts (must have 'lm_head' key)
        sd: state_dict (traced)
        frozen_expert_ids: set of frozen expert ids
        top_k: number of experts to activate
        router_keys_mx: optional precomputed (K, n_embd) for fast scoring
        guaranteed_ids: set of expert ids that must always be selected

    Returns:
        logits: mx.array (vocab_size,) — weighted expert logits
        selected: list of expert dicts activated
    """
    if frozen_expert_ids is None:
        frozen_expert_ids = set()
    if guaranteed_ids is None:
        guaranteed_ids = set()

    # Vectorized scoring
    if router_keys_mx is not None:
        scores_list = compute_scores_mx(x, router_keys_mx).tolist()
    else:
        x_data = x.tolist()
        scores_list = [cosine_sim(x_data, exp['router_key']) for exp in experts]

    # Guaranteed experts always selected; fill remaining top-K slots normally
    guaranteed_indices = [i for i, exp in enumerate(experts)
                         if exp['id'] in guaranteed_ids]
    remaining_slots = max(top_k - len(guaranteed_indices), 0)
    non_guaranteed = [(i, scores_list[i]) for i in range(len(experts))
                      if i not in guaranteed_indices]
    non_guaranteed.sort(key=lambda t: -t[1])
    top_non_guaranteed = non_guaranteed[:remaining_slots]

    selected_indices = guaranteed_indices + [i for i, _ in top_non_guaranteed]
    selected_scores = [scores_list[i] for i in selected_indices]

    max_s = max(selected_scores)
    exp_s = [math.exp(s - max_s) for s in selected_scores]
    total_s = sum(exp_s)
    weights = [e / total_s for e in exp_s]

    x_normed = rmsnorm(x)

    # Batched expert MLP
    out_batch = batch_expert_mlp(sd, experts, x_normed, selected_indices)

    # Per-expert: residual + expert-specific lm_head → logits
    logits_list = []
    selected = []
    for k, idx in enumerate(selected_indices):
        exp = experts[idx]
        exp['activation_count'] = exp.get('activation_count', 0) + 1
        selected.append(exp)

        x_expert = x + out_batch[k]
        expert_logits = sd[exp['lm_head']] @ x_expert

        if exp['id'] in frozen_expert_ids:
            expert_logits = mx.stop_gradient(expert_logits)
        logits_list.append(expert_logits)

    # Weighted merge
    logits = weights[0] * logits_list[0]
    for k in range(1, len(weights)):
        logits = logits + weights[k] * logits_list[k]

    return logits, selected


# ── Dual-Process Expert Composition (System 1 parallel / System 2 sequential) ──

def route_dual_process(x, experts, sd, frozen_expert_ids=None, top_k=2,
                       router_keys_mx=None, mode='sys1',
                       entropy_threshold=2.0, use_expert_heads=False,
                       guaranteed_ids=None):
    """Dual-process routing: System 1 (parallel) or System 2 (sequential chain).

    System 1 (parallel, fast): Top-K experts fire simultaneously, outputs
    are weighted-averaged. Uses expert-specific heads if available. Good for
    familiar inputs where cached expert knowledge suffices.

    System 2 (sequential, deep): ALL experts chain sequentially, each adding
    a layer of processing. Gives effective depth K×D. Uses shared lm_head
    (or last expert's head). Good for novel/complex inputs requiring
    compositional reasoning.

    Blended mode: runs both, blends by output confidence (soft switching).

    Args:
        x: mx.array (n_embd,) — hidden state after attention
        experts: list of expert dicts
        sd: state_dict (traced)
        frozen_expert_ids: set of frozen expert ids
        top_k: number of experts for System 1
        router_keys_mx: precomputed (K, n_embd) for fast scoring
        mode: 'sys1' (parallel only), 'sys2' (sequential only), 'blend' (both)
        entropy_threshold: output entropy threshold for blend weighting
        use_expert_heads: whether experts have per-expert lm_heads
        guaranteed_ids: set of expert ids that must always be selected in sys1

    Returns:
        logits: mx.array (vocab_size,) — final logits
        selected: list of expert dicts activated
    """
    if frozen_expert_ids is None:
        frozen_expert_ids = set()
    if guaranteed_ids is None:
        guaranteed_ids = set()

    # Vectorized scoring
    if router_keys_mx is not None:
        scores_list = compute_scores_mx(x, router_keys_mx).tolist()
    else:
        x_data = x.tolist()
        scores_list = [cosine_sim(x_data, exp['router_key']) for exp in experts]

    # ── System 1: parallel top-K experts ──
    logits_sys1 = None
    selected = []
    if mode in ('sys1', 'blend'):
        # Guaranteed experts always selected; fill remaining slots normally
        guaranteed_indices = [i for i, exp in enumerate(experts)
                             if exp['id'] in guaranteed_ids]
        remaining_slots = max(top_k - len(guaranteed_indices), 0)
        non_guaranteed = [(i, scores_list[i]) for i in range(len(experts))
                          if i not in guaranteed_indices]
        non_guaranteed.sort(key=lambda t: -t[1])
        top_non_guaranteed = non_guaranteed[:remaining_slots]
        selected_indices = guaranteed_indices + [i for i, _ in top_non_guaranteed]
        selected_scores = [scores_list[i] for i in selected_indices]

        max_s = max(selected_scores)
        exp_s = [math.exp(s - max_s) for s in selected_scores]
        total_s = sum(exp_s)
        weights = [e / total_s for e in exp_s]

        x_normed = rmsnorm(x)
        out_batch = batch_expert_mlp(sd, experts, x_normed, selected_indices)

        for k, idx in enumerate(selected_indices):
            exp = experts[idx]
            exp['activation_count'] = exp.get('activation_count', 0) + 1
            selected.append(exp)

        if use_expert_heads:
            # Per-expert logits via expert-specific lm_heads
            logits_parts = []
            for k, idx in enumerate(selected_indices):
                exp = experts[idx]
                x_expert = x + out_batch[k]
                expert_logits = sd[exp['lm_head']] @ x_expert
                if exp['id'] in frozen_expert_ids:
                    expert_logits = mx.stop_gradient(expert_logits)
                logits_parts.append(expert_logits)
            logits_sys1 = weights[0] * logits_parts[0]
            for k in range(1, len(weights)):
                logits_sys1 = logits_sys1 + weights[k] * logits_parts[k]
        else:
            # Shared lm_head
            weights_mx = mx.array(weights)
            x_out = x + mx.sum(weights_mx[:, None] * out_batch, axis=0)
            logits_sys1 = sd['lm_head'] @ x_out

    # ── System 2: sequential expert chain ──
    logits_sys2 = None
    if mode in ('sys2', 'blend'):
        # Chain ALL experts in order of routing score (most relevant first)
        sorted_indices = sorted(range(len(scores_list)),
                                key=lambda i: -scores_list[i])

        h = x
        for idx in sorted_indices:
            exp = experts[idx]
            h_normed = rmsnorm(h)
            hidden = mx.maximum(sd[exp['fc1']] @ h_normed, 0)
            out = sd[exp['fc2']] @ hidden
            if exp['id'] in frozen_expert_ids:
                out = mx.stop_gradient(out)
            h = h + out  # residual per chain step

        if use_expert_heads:
            # Use routing-score-weighted expert heads on composed representation
            # This preserves the frozen-head benefit for memory while
            # leveraging the deep composition for capacity
            all_scores = [scores_list[i] for i in sorted_indices]
            max_as = max(all_scores)
            exp_as = [math.exp(s - max_as) for s in all_scores]
            total_as = sum(exp_as)
            head_weights = [e / total_as for e in exp_as]

            logits_parts = []
            for k, idx in enumerate(sorted_indices):
                exp = experts[idx]
                expert_logits = sd[exp['lm_head']] @ h
                if exp['id'] in frozen_expert_ids:
                    expert_logits = mx.stop_gradient(expert_logits)
                logits_parts.append(expert_logits)
            logits_sys2 = head_weights[0] * logits_parts[0]
            for k in range(1, len(head_weights)):
                logits_sys2 = logits_sys2 + head_weights[k] * logits_parts[k]
        else:
            # Shared lm_head
            logits_sys2 = sd['lm_head'] @ h

        if mode == 'sys2':
            # Track all experts as selected
            selected = list(experts)
            for exp in selected:
                exp['activation_count'] = exp.get('activation_count', 0) + 1

    # ── Mode dispatch ──
    if mode == 'sys1':
        return logits_sys1, selected

    if mode == 'sys2':
        return logits_sys2, selected

    # ── Blend: soft switch based on System 1 output confidence ──
    # High entropy → uncertain → lean toward System 2 (more processing)
    # Low entropy → confident → lean toward System 1 (fast path)
    probs_sys1 = mx.softmax(logits_sys1)
    output_entropy = -mx.sum(probs_sys1 * mx.log(probs_sys1 + 1e-12))
    # alpha: 0 when confident (entropy << threshold), 1 when uncertain
    alpha = mx.sigmoid(output_entropy - entropy_threshold)
    logits = (1 - alpha) * logits_sys1 + alpha * logits_sys2

    return logits, selected
