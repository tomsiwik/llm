"""Evolutionary expert optimization using DEAP.

Evolves MLP expert weights at task boundaries using multi-task fitness:
combined loss on old task data + new task data. Uses SBX crossover,
polynomial mutation, and tournament selection.
"""

import numpy as np
import mlx.core as mx
from deap import base, creator, tools, algorithms


def _flatten_expert(sd, expert):
    """Flatten an expert's weights into a 1D numpy array."""
    parts = []
    keys = []
    for key_name in ['fc1', 'fc2', 'lm_head']:
        if key_name in expert:
            k = expert[key_name]
            arr = np.array(sd[k]).flatten()
            parts.append(arr)
            keys.append((key_name, k, sd[k].shape))
    return np.concatenate(parts).astype(np.float32), keys


def _unflatten_expert(flat, keys):
    """Unflatten a 1D numpy array back into weight dict entries."""
    result = {}
    offset = 0
    for key_name, sd_key, shape in keys:
        size = int(np.prod(shape))
        result[sd_key] = mx.array(flat[offset:offset + size].reshape(shape))
        offset += size
    return result


def _ensure_creator():
    """Register DEAP creator types (idempotent)."""
    if not hasattr(creator, "EvoFitness"):
        creator.create("EvoFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "EvoIndividual"):
        creator.create("EvoIndividual", list, fitness=creator.EvoFitness)


def evolve_experts(sd, experts, graph, old_docs, new_docs,
                   uchars, BOS, block_size,
                   parent_ids=None,
                   pop_size=40, n_gen=20,
                   cxpb=0.6, mutpb=0.4,
                   eta_cx=15.0, eta_mut=15.0,
                   tournament_size=3,
                   eval_sample=30,
                   use_expert_heads=True,
                   seed=42):
    """Evolve expert weights using DEAP at task boundary.

    Creates a population from crossovers/mutations of parent experts,
    evaluates fitness on combined old+new task data, and returns the
    best evolved weights.

    Args:
        sd: state_dict (NOT modified — returns new weights separately)
        experts: list of expert dicts
        graph: Graph instance for forward passes
        old_docs: list of strings from previous task
        new_docs: list of strings from new task
        uchars, BOS, block_size: tokenization params
        parent_ids: which expert IDs to use as parents (default: top-2 activated)
        pop_size: population size
        n_gen: number of generations
        cxpb: crossover probability
        mutpb: mutation probability
        eta_cx: SBX distribution index (higher = more conservative crossover)
        eta_mut: polynomial mutation distribution index
        tournament_size: tournament selection size
        eval_sample: max docs to evaluate per task for fitness
        use_expert_heads: whether experts have lm_heads
        seed: random seed

    Returns:
        dict with:
            'weights': dict of sd_key -> mx.array (evolved weights for each parent)
            'fitness_improvement': float (best fitness - parent mean fitness)
            'best_fitness': float
            'parent_fitness': dict of expert_id -> fitness
    """
    import random as _random
    rng = _random.Random(seed)
    np.random.seed(seed)

    _ensure_creator()

    # Select parents
    if parent_ids is None:
        sorted_exps = sorted(experts,
                             key=lambda e: e.get('activation_count', 0),
                             reverse=True)
        parents = sorted_exps[:2]
    else:
        parents = [e for e in experts if e['id'] in parent_ids]

    if len(parents) < 2:
        return None  # need at least 2 parents

    # Flatten parent weights
    parent_flats = []
    parent_keys = None
    for p in parents:
        flat, keys = _flatten_expert(sd, p)
        parent_flats.append(flat)
        if parent_keys is None:
            parent_keys = keys  # all experts have same structure

    n_weights = len(parent_flats[0])

    # Compute weight bounds for SBX/polynomial mutation
    all_vals = np.concatenate(parent_flats)
    w_min = float(np.min(all_vals)) - 1.0
    w_max = float(np.max(all_vals)) + 1.0

    # Prepare eval data
    old_sample = old_docs[:eval_sample]
    new_sample = new_docs[:eval_sample]

    # Build route function for evaluation
    from lgme.router import route_with_expert_heads, route_mlp_experts

    def _eval_route(x, e, sd_local):
        if use_expert_heads:
            logits, _ = route_with_expert_heads(x, e, sd_local, top_k=2)
            return logits
        else:
            x_out, _ = route_mlp_experts(x, e, sd_local, top_k=2)
            return x_out

    def evaluate_with_weights(weight_updates):
        """Evaluate model with temporarily updated expert weights."""
        # Save originals
        originals = {}
        for sd_key, new_val in weight_updates.items():
            originals[sd_key] = sd[sd_key]
            sd[sd_key] = new_val
        mx.eval(sd)

        # Evaluate on both tasks
        total_loss = 0.0
        all_docs = old_sample + new_sample
        for doc in all_docs:
            tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
            n = min(block_size, len(tokens) - 1)
            graph.reset_kv()
            doc_loss = 0.0
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = graph.forward(token_id, pos_id, sd=sd,
                                       mlp_experts=experts,
                                       route_fn=_eval_route,
                                       route_produces_logits=use_expert_heads)
                probs = mx.softmax(logits)
                doc_loss += (-mx.log(probs[target_id] + 1e-8)).item()
            total_loss += doc_loss / n

        # Restore originals
        for sd_key, orig_val in originals.items():
            sd[sd_key] = orig_val

        return total_loss / len(all_docs)

    # Evaluate parent baselines
    parent_fitness = {}
    for p in parents:
        flat, keys = _flatten_expert(sd, p)
        weight_updates = _unflatten_expert(flat, keys)
        loss = evaluate_with_weights(weight_updates)
        parent_fitness[p['id']] = -loss  # negative loss = fitness

    # Target expert for weight update (first parent — most activated)
    target_expert = parents[0]
    target_keys = parent_keys

    # DEAP setup
    toolbox = base.Toolbox()

    def make_individual():
        r = rng.random()
        if r < 0.35:
            # Blend with random alpha
            alpha = rng.uniform(0.0, 1.0)
            flat = alpha * parent_flats[0] + (1 - alpha) * parent_flats[1]
        elif r < 0.55:
            # DE-style: parent_a + F * (parent_b - parent_a)
            F = rng.uniform(0.3, 1.5)
            flat = parent_flats[0] + F * (parent_flats[1] - parent_flats[0])
        elif r < 0.75:
            # Parent A + noise
            flat = parent_flats[0].copy()
        else:
            # Parent B + noise
            flat = parent_flats[1].copy()
        noise = np.random.randn(n_weights).astype(np.float32) * 0.05
        return creator.EvoIndividual((flat + noise).tolist())

    def fitness_fn(individual):
        flat = np.array(individual, dtype=np.float32)
        weight_updates = _unflatten_expert(flat, target_keys)
        loss = evaluate_with_weights(weight_updates)
        return (-loss,)  # negative loss = fitness

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     eta=eta_cx, low=w_min, up=w_max)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=eta_mut, low=w_min, up=w_max, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Create population, inject parents
    pop = toolbox.population(n=pop_size)
    pop[0] = creator.EvoIndividual(parent_flats[0].tolist())
    pop[1] = creator.EvoIndividual(parent_flats[1].tolist())

    hof = tools.HallOfFame(1)

    # Evolve
    pop, log = algorithms.eaSimple(pop, toolbox,
                                    cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
                                    halloffame=hof, verbose=False)

    # Extract best
    best_flat = np.array(hof[0], dtype=np.float32)
    best_weights = _unflatten_expert(best_flat, target_keys)
    best_fitness = hof[0].fitness.values[0]
    mean_parent_fitness = np.mean(list(parent_fitness.values()))

    return {
        'weights': best_weights,
        'target_expert': target_expert,
        'fitness_improvement': best_fitness - mean_parent_fitness,
        'best_fitness': best_fitness,
        'parent_fitness': parent_fitness,
    }
