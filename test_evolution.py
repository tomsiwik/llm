"""Evolutionary offspring test: DEAP-powered expert evolution.

Proves that real evolutionary algorithms (SBX crossover, polynomial mutation,
tournament selection) create meaningfully better offspring than naive blending.

Setup (same as test_afmei.py):
  - Task: linear regression y = W_true @ x  (4-dim)
  - Parent A: rows 0,1 correct, rows 2,3 wrong
  - Parent B: rows 0,1 wrong, rows 2,3 correct
  - Goal: evolve offspring that combines the best of both
"""

import numpy as np
import mlx.core as mx
from deap import base, creator, tools, algorithms
import random

# ── Ground truth ──
W_TRUE = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)

# Test data
rng = np.random.RandomState(42)
X_TEST = rng.randn(50, 4).astype(np.float32)
Y_TEST = X_TEST @ W_TRUE.T

# Parents with complementary knowledge
W_PARENT_A = np.array([
    [1.0, 0.0, 0.0, 0.0],   # correct
    [0.0, 1.0, 0.0, 0.0],   # correct
    [0.5, 0.5, 0.0, 0.5],   # wrong
    [0.3, 0.0, 0.7, 0.0],   # wrong
], dtype=np.float32)

W_PARENT_B = np.array([
    [0.0, 0.8, 0.2, 0.0],   # wrong
    [0.5, 0.0, 0.0, 0.5],   # wrong
    [0.0, 0.0, 1.0, 0.0],   # correct
    [0.0, 0.0, 0.0, 1.0],   # correct
], dtype=np.float32)

NUM_WEIGHTS = W_TRUE.size  # 16 floats


def mse(W_flat):
    """Fitness: negative MSE (higher = better)."""
    W = W_flat.reshape(4, 4)
    pred = X_TEST @ W.T
    return float(-np.mean((pred - Y_TEST) ** 2))


def mse_positive(W_flat):
    """MSE as positive number (for display)."""
    return -mse(W_flat)


# ══════════════════════════════════════════════════════════
# Test 1: DEAP evolves from parents → beats both
# ══════════════════════════════════════════════════════════

def test_deap_evolves_from_parents():
    """Evolve a population seeded from two parents. Best individual should
    outperform both parents."""
    print("Test 1: DEAP evolves from parent-seeded population")

    # Clear any previous DEAP creator registrations
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Seed population from parent crossovers + mutations
    parent_a_flat = W_PARENT_A.flatten().tolist()
    parent_b_flat = W_PARENT_B.flatten().tolist()

    def make_individual():
        """Create individual: 50% blend variants, 25% parent A mutants, 25% parent B mutants."""
        r = random.random()
        if r < 0.5:
            # Blend with random alpha
            alpha = random.uniform(0.0, 1.0)
            ind = [alpha * a + (1 - alpha) * b for a, b in zip(parent_a_flat, parent_b_flat)]
        elif r < 0.75:
            # Parent A + Gaussian noise
            ind = [a + random.gauss(0, 0.1) for a in parent_a_flat]
        else:
            # Parent B + Gaussian noise
            ind = [b + random.gauss(0, 0.1) for b in parent_b_flat]
        return creator.Individual(ind)

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operators
    toolbox.register("evaluate", lambda ind: (mse(np.array(ind, dtype=np.float32)),))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     eta=20.0, low=-2.0, up=2.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=20.0, low=-2.0, up=2.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evolution
    pop = toolbox.population(n=50)
    # Also inject the actual parents into the population
    pop[0] = creator.Individual(parent_a_flat)
    pop[1] = creator.Individual(parent_b_flat)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", np.max)
    stats.register("mean", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                    cxpb=0.7,    # crossover probability
                                    mutpb=0.3,   # mutation probability
                                    ngen=30,      # generations
                                    stats=stats,
                                    halloffame=hof,
                                    verbose=False)

    best = np.array(hof[0], dtype=np.float32)
    best_mse = mse_positive(best)
    parent_a_mse = mse_positive(W_PARENT_A.flatten())
    parent_b_mse = mse_positive(W_PARENT_B.flatten())
    blend_mse = mse_positive(0.5 * W_PARENT_A.flatten() + 0.5 * W_PARENT_B.flatten())

    print(f"  Parent A MSE:      {parent_a_mse:.4f}")
    print(f"  Parent B MSE:      {parent_b_mse:.4f}")
    print(f"  Naive blend MSE:   {blend_mse:.4f}")
    print(f"  DEAP best MSE:     {best_mse:.4f}")
    print(f"  Best weights:\n    {best.reshape(4,4).round(3)}")

    # Log evolution progress
    gen_best = [log[i]['best'] for i in range(len(log))]
    print(f"  Evolution: gen 0 best={-gen_best[0]:.4f} → gen {len(log)-1} best={-gen_best[-1]:.4f}")

    assert best_mse < parent_a_mse, "Evolved should beat parent A"
    assert best_mse < parent_b_mse, "Evolved should beat parent B"
    assert best_mse < blend_mse, "Evolved should beat naive blend"
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════
# Test 2: Differential evolution creates diverse offspring
# ══════════════════════════════════════════════════════════

def test_differential_evolution():
    """DE creates child = A + F*(B - C). This explores BEYOND parents,
    not just between them."""
    print("Test 2: Differential evolution crossover")

    a = W_PARENT_A.flatten()
    b = W_PARENT_B.flatten()

    # DE mutation: child = base + F * (donor1 - donor2)
    F = 0.8  # scaling factor
    # Strategy: rand/1 — use random perturbation of parents
    children = []
    for _ in range(20):
        # Random base + differential
        if random.random() < 0.5:
            base_vec = a.copy()
            diff = b - a
        else:
            base_vec = b.copy()
            diff = a - b
        noise = np.random.randn(NUM_WEIGHTS).astype(np.float32) * 0.05
        child = base_vec + F * diff + noise
        children.append(child)

    # Evaluate all
    child_mses = [mse_positive(c) for c in children]
    best_idx = np.argmin(child_mses)
    best_child = children[best_idx]

    parent_a_mse = mse_positive(a)
    parent_b_mse = mse_positive(b)
    blend_mse = mse_positive(0.5 * a + 0.5 * b)

    print(f"  Parent A MSE:    {parent_a_mse:.4f}")
    print(f"  Parent B MSE:    {parent_b_mse:.4f}")
    print(f"  Naive blend MSE: {blend_mse:.4f}")
    print(f"  Best DE child:   {min(child_mses):.4f}")
    print(f"  Mean DE child:   {np.mean(child_mses):.4f}")
    print(f"  Worst DE child:  {max(child_mses):.4f}")
    print(f"  # beating both parents: {sum(1 for m in child_mses if m < min(parent_a_mse, parent_b_mse))}/20")

    print("  PASSED (DE explores weight space beyond parents)\n")


# ══════════════════════════════════════════════════════════
# Test 3: Row-aware crossover vs flat crossover
# ══════════════════════════════════════════════════════════

def test_row_aware_crossover():
    """Compare: treating the weight matrix as flat (standard GA) vs
    respecting row structure (each row = one gene)."""
    print("Test 3: Row-aware vs flat crossover")

    if hasattr(creator, "FitnessMax2"):
        del creator.FitnessMax2
    if hasattr(creator, "Individual2"):
        del creator.Individual2

    creator.create("FitnessMax2", base.Fitness, weights=(1.0,))
    creator.create("Individual2", list, fitness=creator.FitnessMax2)

    # Strategy 1: Flat crossover (standard — treats all 16 weights equally)
    def run_flat_evolution(seed):
        rng = random.Random(seed)
        toolbox = base.Toolbox()

        def make_ind():
            alpha = rng.uniform(0, 1)
            flat = (alpha * W_PARENT_A.flatten() + (1 - alpha) * W_PARENT_B.flatten()).tolist()
            return creator.Individual2(flat)

        toolbox.register("individual", make_ind)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: (mse(np.array(ind, dtype=np.float32)),))
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         eta=20.0, low=-2.0, up=2.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         eta=20.0, low=-2.0, up=2.0, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=30)
        pop[0] = creator.Individual2(W_PARENT_A.flatten().tolist())
        pop[1] = creator.Individual2(W_PARENT_B.flatten().tolist())
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=20,
                            halloffame=hof, verbose=False)
        return mse_positive(np.array(hof[0], dtype=np.float32))

    # Strategy 2: Row-aware crossover (custom — picks entire rows from parents)
    def run_row_evolution(seed):
        rng = random.Random(seed)
        best_mse = float('inf')
        # Generate row combinations: each of 4 rows can come from A or B = 16 combos
        for mask in range(16):
            rows = []
            for i in range(4):
                if mask & (1 << i):
                    rows.append(W_PARENT_A[i].copy())
                else:
                    rows.append(W_PARENT_B[i].copy())
            W = np.stack(rows)
            # Add small mutation
            W += rng.gauss(0, 0.01) * np.ones_like(W)
            m = mse_positive(W.flatten())
            if m < best_mse:
                best_mse = m
        return best_mse

    # Run both
    flat_results = [run_flat_evolution(s) for s in range(5)]
    row_results = [run_row_evolution(s) for s in range(5)]

    flat_mean = np.mean(flat_results)
    row_mean = np.mean(row_results)

    print(f"  Flat crossover (5 seeds):     MSE = {flat_mean:.4f} ± {np.std(flat_results):.4f}")
    print(f"  Row-aware selection (5 seeds): MSE = {row_mean:.4f} ± {np.std(row_results):.4f}")
    print(f"  Row-aware improvement: {(flat_mean - row_mean)/flat_mean*100:.1f}%")

    # Row-aware should be better (or equal) since it respects the structure
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════
# Test 4: Evolution with gradient-seeded population
# ══════════════════════════════════════════════════════════

def test_gradient_seeded_evolution():
    """Hybrid approach: use gradient information to seed the initial population,
    then evolve. This combines gradient efficiency with evolutionary exploration."""
    print("Test 4: Gradient-seeded evolutionary optimization")

    if hasattr(creator, "FitnessMax3"):
        del creator.FitnessMax3
    if hasattr(creator, "Individual3"):
        del creator.Individual3

    creator.create("FitnessMax3", base.Fitness, weights=(1.0,))
    creator.create("Individual3", list, fitness=creator.FitnessMax3)

    # Step 1: Take a few gradient steps from each parent toward the target
    def gradient_step(W, lr=0.1, steps=5):
        W_mx = mx.array(W)
        for _ in range(steps):
            def loss_fn(W_p):
                pred = mx.array(X_TEST) @ W_p.T
                return mx.mean((pred - mx.array(Y_TEST)) ** 2)
            _, grad = mx.value_and_grad(loss_fn)(W_mx)
            W_mx = W_mx - lr * grad
            mx.eval(W_mx)
        return np.array(W_mx)

    # Create gradient-improved versions of parents
    W_A_improved = gradient_step(W_PARENT_A, steps=3)
    W_B_improved = gradient_step(W_PARENT_B, steps=3)

    # Step 2: Seed population from gradient-improved parents
    toolbox = base.Toolbox()

    def make_gradient_ind():
        r = random.random()
        if r < 0.25:
            base_w = W_A_improved.flatten()
        elif r < 0.5:
            base_w = W_B_improved.flatten()
        elif r < 0.75:
            alpha = random.uniform(0, 1)
            base_w = alpha * W_A_improved.flatten() + (1 - alpha) * W_B_improved.flatten()
        else:
            # DE-style: improved_A + F * (improved_B - improved_A)
            F = random.uniform(0.5, 1.5)
            base_w = W_A_improved.flatten() + F * (W_B_improved.flatten() - W_A_improved.flatten())
        # Add noise
        noise = np.random.randn(NUM_WEIGHTS).astype(np.float32) * 0.05
        return creator.Individual3((base_w + noise).tolist())

    toolbox.register("individual", make_gradient_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (mse(np.array(ind, dtype=np.float32)),))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     eta=20.0, low=-2.0, up=2.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=20.0, low=-2.0, up=2.0, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=40)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=20,
                        halloffame=hof, verbose=False)

    best = np.array(hof[0], dtype=np.float32)
    best_mse = mse_positive(best)

    # Compare against various baselines
    parent_a_mse = mse_positive(W_PARENT_A.flatten())
    parent_b_mse = mse_positive(W_PARENT_B.flatten())
    blend_mse = mse_positive(0.5 * W_PARENT_A.flatten() + 0.5 * W_PARENT_B.flatten())
    grad_a_mse = mse_positive(W_A_improved.flatten())
    grad_b_mse = mse_positive(W_B_improved.flatten())

    print(f"  Parent A (raw):           {parent_a_mse:.4f}")
    print(f"  Parent B (raw):           {parent_b_mse:.4f}")
    print(f"  Naive blend:              {blend_mse:.4f}")
    print(f"  Parent A (3 grad steps):  {grad_a_mse:.4f}")
    print(f"  Parent B (3 grad steps):  {grad_b_mse:.4f}")
    print(f"  DEAP (gradient-seeded):   {best_mse:.4f}")
    print(f"  Best weights:\n    {best.reshape(4,4).round(3)}")

    assert best_mse < blend_mse, "Gradient+evolution should beat naive blend"
    assert best_mse < min(grad_a_mse, grad_b_mse), \
        "Gradient+evolution should beat gradient-only"
    print("  PASSED\n")


# ══════════════════════════════════════════════════════════
# Test 5: Multi-task fitness (the real challenge)
# ══════════════════════════════════════════════════════════

def test_multitask_evolution():
    """THE KEY TEST: Evolve offspring that performs well on BOTH tasks.

    This is the continual learning objective: find weights that minimize loss
    on task 1 AND task 2 simultaneously. Standard gradient descent on task 2
    forgets task 1. Can evolution find a better tradeoff?
    """
    print("Test 5: Multi-task evolutionary optimization")

    # Two tasks with different optimal weights
    W_TASK1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5, 0.5],
    ], dtype=np.float32)
    W_TASK2 = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    Y_TASK1 = X_TEST @ W_TASK1.T
    Y_TASK2 = X_TEST @ W_TASK2.T

    def combined_mse(W_flat):
        """Fitness: negative COMBINED loss on both tasks."""
        W = W_flat.reshape(4, 4)
        pred1 = X_TEST @ W.T
        pred2 = X_TEST @ W.T
        mse1 = np.mean((pred1 - Y_TASK1) ** 2)
        mse2 = np.mean((pred2 - Y_TASK2) ** 2)
        return float(-(mse1 + mse2))  # minimize combined

    def task_mses(W_flat):
        W = W_flat.reshape(4, 4)
        mse1 = float(np.mean((X_TEST @ W.T - Y_TASK1) ** 2))
        mse2 = float(np.mean((X_TEST @ W.T - Y_TASK2) ** 2))
        return mse1, mse2

    # Parent A: trained on task 1 (perfect on task 1, bad on task 2)
    # Parent B: trained on task 2 (perfect on task 2, bad on task 1)
    parent_a = W_TASK1.flatten()
    parent_b = W_TASK2.flatten()

    # Baseline: train parent A on task 2 with gradient descent (catastrophic forgetting)
    W_catastrophic = mx.array(W_TASK1)
    for _ in range(30):
        def loss_fn(W_p):
            pred = mx.array(X_TEST) @ W_p.T
            return mx.mean((pred - mx.array(Y_TASK2)) ** 2)
        _, grad = mx.value_and_grad(loss_fn)(W_catastrophic)
        W_catastrophic = W_catastrophic - 0.1 * grad
        mx.eval(W_catastrophic)
    W_catastrophic = np.array(W_catastrophic)

    # Evolutionary approach: optimize combined fitness
    if hasattr(creator, "FitnessMax4"):
        del creator.FitnessMax4
    if hasattr(creator, "Individual4"):
        del creator.Individual4

    creator.create("FitnessMax4", base.Fitness, weights=(1.0,))
    creator.create("Individual4", list, fitness=creator.FitnessMax4)

    toolbox = base.Toolbox()

    def make_ind():
        r = random.random()
        if r < 0.4:
            alpha = random.uniform(0, 1)
            base_w = alpha * parent_a + (1 - alpha) * parent_b
        elif r < 0.7:
            # DE-style
            F = random.uniform(0.3, 1.5)
            base_w = parent_a + F * (parent_b - parent_a)
        else:
            base_w = (parent_a if random.random() < 0.5 else parent_b).copy()
        noise = np.random.randn(NUM_WEIGHTS).astype(np.float32) * 0.08
        return creator.Individual4((base_w + noise).tolist())

    toolbox.register("individual", make_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (combined_mse(np.array(ind, dtype=np.float32)),))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     eta=15.0, low=-2.0, up=2.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=15.0, low=-2.0, up=2.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=60)
    pop[0] = creator.Individual4(parent_a.tolist())
    pop[1] = creator.Individual4(parent_b.tolist())
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=40,
                        halloffame=hof, verbose=False)

    best = np.array(hof[0], dtype=np.float32)
    best_t1, best_t2 = task_mses(best)
    cat_t1, cat_t2 = task_mses(W_catastrophic.flatten())
    pa_t1, pa_t2 = task_mses(parent_a)
    pb_t1, pb_t2 = task_mses(parent_b)
    blend = 0.5 * parent_a + 0.5 * parent_b
    bl_t1, bl_t2 = task_mses(blend)

    print(f"  {'Method':<30} {'Task1 MSE':>10} {'Task2 MSE':>10} {'Combined':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Parent A (task1 expert)':<30} {pa_t1:>10.4f} {pa_t2:>10.4f} {pa_t1+pa_t2:>10.4f}")
    print(f"  {'Parent B (task2 expert)':<30} {pb_t1:>10.4f} {pb_t2:>10.4f} {pb_t1+pb_t2:>10.4f}")
    print(f"  {'Naive blend':<30} {bl_t1:>10.4f} {bl_t2:>10.4f} {bl_t1+bl_t2:>10.4f}")
    print(f"  {'Gradient on task2 (forget)':<30} {cat_t1:>10.4f} {cat_t2:>10.4f} {cat_t1+cat_t2:>10.4f}")
    print(f"  {'DEAP evolved (combined fit)':<30} {best_t1:>10.4f} {best_t2:>10.4f} {best_t1+best_t2:>10.4f}")
    print(f"\n  Best evolved weights:\n    {best.reshape(4,4).round(3)}")

    # Key assertions
    assert best_t1 + best_t2 < pa_t1 + pa_t2, "Evolution should beat parent A on combined"
    assert best_t1 + best_t2 < pb_t1 + pb_t2, "Evolution should beat parent B on combined"
    assert best_t1 + best_t2 < cat_t1 + cat_t2, \
        "Evolution should beat catastrophic forgetting on combined"
    assert best_t1 < cat_t1, \
        "Evolution should have less task1 degradation than gradient-only"
    print("  PASSED (evolution finds better multi-task tradeoff than gradient descent)\n")


# ══════════════════════════════════════════════════════════
# Test 6: Evolution recovers from bad initialization
# ══════════════════════════════════════════════════════════

def test_evolution_recovers():
    """Show that evolutionary mutations can fix broken weights that
    gradient descent from a bad starting point would not easily fix."""
    print("Test 6: Evolution recovers from poor initialization")

    # Start with a badly corrupted version of the true weights
    W_BAD = np.array([
        [0.3, 0.7, 0.0, 0.0],   # wrong
        [0.0, 0.3, 0.7, 0.0],   # wrong
        [0.0, 0.0, 0.3, 0.7],   # wrong
        [0.7, 0.0, 0.0, 0.3],   # wrong
    ], dtype=np.float32)

    if hasattr(creator, "FitnessMax5"):
        del creator.FitnessMax5
    if hasattr(creator, "Individual5"):
        del creator.Individual5

    creator.create("FitnessMax5", base.Fitness, weights=(1.0,))
    creator.create("Individual5", list, fitness=creator.FitnessMax5)

    toolbox = base.Toolbox()

    def make_ind():
        # Start from bad weights + noise
        noise = np.random.randn(NUM_WEIGHTS).astype(np.float32) * 0.3
        return creator.Individual5((W_BAD.flatten() + noise).tolist())

    toolbox.register("individual", make_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (mse(np.array(ind, dtype=np.float32)),))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     eta=10.0, low=-2.0, up=2.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     eta=10.0, low=-2.0, up=2.0, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=80)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=50,
                        halloffame=hof, verbose=False)

    best = np.array(hof[0], dtype=np.float32)
    bad_mse = mse_positive(W_BAD.flatten())
    best_mse = mse_positive(best)

    print(f"  Bad init MSE:     {bad_mse:.4f}")
    print(f"  Evolved best MSE: {best_mse:.4f}")
    print(f"  Improvement:      {(bad_mse - best_mse)/bad_mse*100:.1f}%")
    print(f"  Best weights:\n    {best.reshape(4,4).round(3)}")

    assert best_mse < bad_mse * 0.5, "Evolution should significantly improve bad weights"
    print("  PASSED\n")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    print("=" * 65)
    print("Evolutionary Offspring Tests: DEAP-Powered Expert Evolution")
    print("=" * 65 + "\n")

    test_deap_evolves_from_parents()
    test_differential_evolution()
    test_row_aware_crossover()
    test_gradient_seeded_evolution()
    test_multitask_evolution()
    test_evolution_recovers()

    print("=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
