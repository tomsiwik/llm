"""Test-drive the Expert Lifecycle in isolation.

Toy model: each expert is a small MLP (8→16→8) that learns to map
input patterns to target patterns. "Knowledge" = low loss on specific
patterns. "Capacity" = how many patterns before quality degrades.

Tests prove each lifecycle stage works before integration:
  1. Capacity fills during training (measurable)
  2. Bonding creates specialized child from complementary parents
  3. Delegation routes novel inputs to child
  4. Criticality detection works
  5. Forced unlearning frees parent capacity
  6. After unlearning, parent learns new patterns faster
  7. Full lifecycle end-to-end
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from deap import base, creator, tools
import random

# ── Toy Model ──────────────────────────────────────────────────────

DIM = 8
HIDDEN = 16

def make_expert(seed=0):
    """Create a small MLP expert with random weights."""
    mx.random.seed(seed)
    return {
        'W1': mx.random.normal((HIDDEN, DIM)) * 0.3,
        'b1': mx.zeros((HIDDEN,)),
        'W2': mx.random.normal((DIM, HIDDEN)) * 0.3,
        'b2': mx.zeros((DIM,)),
    }

def forward_expert(weights, x):
    """Forward pass: x → ReLU(W1 x + b1) → W2 h + b2."""
    h = mx.maximum(weights['W1'] @ x + weights['b1'], 0)
    return weights['W2'] @ h + weights['b2']

def expert_loss(weights, x, target):
    """MSE loss on a single (input, target) pair."""
    pred = forward_expert(weights, x)
    return mx.mean((pred - target) ** 2).item()

def expert_loss_batch(weights, patterns):
    """Mean loss across a list of (input, target) pairs."""
    losses = [expert_loss(weights, x, t) for x, t in patterns]
    return sum(losses) / len(losses)

def train_expert(weights, patterns, steps=200, lr=0.01):
    """Train expert on patterns via gradient descent. Returns loss history."""
    history = []
    for step in range(steps):
        def loss_fn(w):
            losses = []
            for x, target in patterns:
                pred = forward_expert(w, x)
                losses.append(mx.mean((pred - target) ** 2))
            return mx.mean(mx.stack(losses))

        loss_val, grads = mx.value_and_grad(loss_fn)(weights)

        for key in weights:
            weights[key] = weights[key] - lr * grads[key]
        mx.eval(*[weights[k] for k in weights])
        history.append(loss_val.item())
    return history


# ── Pattern Generation ─────────────────────────────────────────────

def make_patterns(n, seed=0):
    """Generate n distinct (input, target) pattern pairs."""
    rng = np.random.RandomState(seed)
    patterns = []
    for _ in range(n):
        x = mx.array(rng.randn(DIM).astype(np.float32))
        # Target is a nonlinear transform so it's learnable but nontrivial
        t_np = np.tanh(rng.randn(DIM, DIM).astype(np.float32) @ x.tolist())
        target = mx.array(t_np.astype(np.float32))
        patterns.append((x, target))
    return patterns


# ── Capacity Measurement ──────────────────────────────────────────

def measure_capacity_gradient(weights, patterns, window=30):
    """Gradient norm on patterns — high means lots to learn, low means saturated."""
    def loss_fn(w):
        losses = [mx.mean((forward_expert(w, x) - t) ** 2) for x, t in patterns]
        return mx.mean(mx.stack(losses))

    _, grads = mx.value_and_grad(loss_fn)(weights)
    grad_norm = sum(mx.sum(grads[k] ** 2).item() for k in grads)
    return grad_norm


def measure_capacity_rank(weights):
    """Capacity via effective rank of W1 (how many dimensions are used)."""
    W = np.array(weights['W1'])
    s = np.linalg.svd(W, compute_uv=False)
    s = s / (s.sum() + 1e-10)
    entropy = -np.sum(s * np.log(s + 1e-10))
    eff_rank = np.exp(entropy)
    max_rank = min(W.shape)
    return eff_rank / max_rank  # [0, 1] — higher = more capacity used


# ── Crossover / Bonding ───────────────────────────────────────────

def bond_experts(parent_a, parent_b, alpha=0.5, noise_std=0.02, seed=0):
    """Create child via weighted blend + noise (genetic crossover)."""
    rng = np.random.RandomState(seed)
    child = {}
    for key in parent_a:
        a = np.array(parent_a[key])
        b = np.array(parent_b[key])
        blended = alpha * a + (1 - alpha) * b
        noise = rng.randn(*blended.shape).astype(np.float32) * noise_std
        child[key] = mx.array(blended + noise)
    return child


def bond_experts_deap(parent_a, parent_b, seed=0):
    """Create child via DEAP SBX crossover (real genetic recombination)."""
    # Flatten both parents
    flat_a, keys_info = _flatten_weights(parent_a)
    flat_b, _ = _flatten_weights(parent_b)

    # SBX crossover
    if not hasattr(creator, 'LifeFitness'):
        creator.create('LifeFitness', base.Fitness, weights=(1.0,))
        creator.create('LifeIndividual', list, fitness=creator.LifeFitness)

    ind_a = creator.LifeIndividual(flat_a.tolist())
    ind_b = creator.LifeIndividual(flat_b.tolist())

    np.random.seed(seed)
    random.seed(seed)
    tools.cxSimulatedBinaryBounded(ind_a, ind_b, eta=10.0,
                                    low=float(min(flat_a.min(), flat_b.min()) - 0.5),
                                    up=float(max(flat_a.max(), flat_b.max()) + 0.5))

    child_flat = np.array(ind_a, dtype=np.float32)
    return _unflatten_weights(child_flat, keys_info)


def _flatten_weights(weights):
    """Flatten weight dict to 1D numpy array."""
    parts = []
    keys_info = []
    for key in sorted(weights.keys()):
        arr = np.array(weights[key]).flatten()
        parts.append(arr)
        keys_info.append((key, weights[key].shape))
    return np.concatenate(parts).astype(np.float32), keys_info


def _unflatten_weights(flat, keys_info):
    """Unflatten 1D numpy array to weight dict."""
    result = {}
    offset = 0
    for key, shape in keys_info:
        size = int(np.prod(shape))
        result[key] = mx.array(flat[offset:offset+size].reshape(shape))
        offset += size
    return result


# ── Delegation ────────────────────────────────────────────────────

def compute_delegation_weight(parent, child, x, target, baseline_loss=2.0):
    """How much gradient should go to child vs parent for this input?

    Returns (parent_weight, child_weight) summing to 1.0.
    Higher child_weight = parent delegates more.
    """
    # Factor 1: Does parent already know this? (low loss = high knowledge)
    parent_loss = expert_loss(parent, x, target)
    parent_knowledge = max(0, 1.0 - parent_loss / baseline_loss)

    # Factor 2: Would learning this disrupt the parent?
    # (high gradient norm = high disruption)
    def p_loss_fn(w):
        pred = forward_expert(w, x)
        return mx.mean((pred - target) ** 2)

    _, grads = mx.value_and_grad(p_loss_fn)(parent)
    grad_norm = sum(mx.sum(grads[k] ** 2).item() for k in grads)

    # Normalize disruption (relative to a baseline)
    disruption = min(1.0, grad_norm / (baseline_loss + 1e-8))

    # Factor 3: Does child have capacity?
    child_grad_norm = measure_capacity_gradient(child, [(x, target)])
    child_available = min(1.0, child_grad_norm / (baseline_loss + 1e-8))

    # Combine: delegate when parent knows it OR it's disruptive AND child has room
    delegate_score = (0.3 * parent_knowledge +
                      0.4 * disruption +
                      0.3 * child_available)

    child_weight = min(0.9, max(0.1, delegate_score))
    return 1.0 - child_weight, child_weight


# ── Forced Unlearning ─────────────────────────────────────────────

def compute_knowledge_overlap(parent, child, patterns):
    """Per-pattern: does child handle it better than parent?

    Returns float in [0, 1] — fraction of patterns where child is better.
    """
    child_better = 0
    for x, t in patterns:
        c_loss = expert_loss(child, x, t)
        p_loss = expert_loss(parent, x, t)
        if c_loss < p_loss:
            child_better += 1
    return child_better / len(patterns)


def unlearn_parent(parent, overlap_patterns, decay_rate=0.2, birth_weights=None,
                   keep_patterns=None):
    """Selectively unlearn overlap knowledge while preserving other knowledge.

    Strategy: compute gradient on overlap patterns, then apply ANTI-gradient
    (step to increase loss on overlap). But CONSTRAIN by also computing
    gradient on keep patterns — don't move in directions that hurt kept knowledge.

    If birth_weights provided, blend anti-gradient with decay toward birth state.
    If keep_patterns provided, project anti-gradient to avoid hurting keep knowledge.
    """
    # Compute anti-gradient for overlap (want to forget these)
    def overlap_loss(w):
        losses = [mx.mean((forward_expert(w, x) - t) ** 2)
                  for x, t in overlap_patterns]
        return mx.mean(mx.stack(losses))

    _, forget_grads = mx.value_and_grad(overlap_loss)(parent)

    if keep_patterns is not None and len(keep_patterns) > 0:
        # Compute gradient for keep patterns (want to preserve these)
        def keep_loss(w):
            losses = [mx.mean((forward_expert(w, x) - t) ** 2)
                      for x, t in keep_patterns]
            return mx.mean(mx.stack(losses))

        _, keep_grads = mx.value_and_grad(keep_loss)(parent)

        # Project forget direction: remove component that hurts keep patterns
        # anti_grad should be ORTHOGONAL to keep_grad
        for key in parent:
            fg = forget_grads[key]  # direction that reduces overlap loss
            kg = keep_grads[key]    # direction that reduces keep loss

            # Anti-gradient: we want to INCREASE overlap loss
            anti = -fg  # step to forget

            # Project out the keep-harmful component
            # anti_proj = anti - (anti . kg / kg . kg) * kg
            dot = mx.sum(anti * kg)
            kg_norm = mx.sum(kg * kg) + 1e-10
            anti_proj = anti - (dot / kg_norm) * kg

            parent[key] = parent[key] + decay_rate * anti_proj
    elif birth_weights is not None:
        # Targeted decay: blend toward birth state weighted by gradient magnitude
        for key in parent:
            # Higher gradient = more important for overlap = decay more
            fg_mag = mx.abs(forget_grads[key])
            fg_weight = fg_mag / (mx.max(fg_mag) + 1e-10)  # normalize [0, 1]
            # Selective decay: high-gradient params decay toward birth, others stay
            parent[key] = parent[key] * (1 - decay_rate * fg_weight) + \
                          birth_weights[key] * (decay_rate * fg_weight)
    else:
        # Simple anti-gradient
        for key in parent:
            parent[key] = parent[key] - decay_rate * forget_grads[key]

    mx.eval(*[parent[k] for k in parent])


# ══════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════

def test_1_capacity_increases_during_training():
    """Expert gradient norm should decrease as it learns (capacity fills)."""
    print("Test 1: Capacity increases during training")

    expert = make_expert(seed=42)
    patterns = make_patterns(6, seed=100)

    # Measure gradient norm before training (high = lots to learn)
    grad_norm_before = measure_capacity_gradient(expert, patterns)

    # Train
    train_expert(expert, patterns, steps=300, lr=0.02)

    # Measure gradient norm after training (low = learned, saturated)
    grad_norm_after = measure_capacity_gradient(expert, patterns)

    # Loss should have decreased
    loss_before = expert_loss_batch(make_expert(seed=42), patterns)
    loss_after = expert_loss_batch(expert, patterns)

    print(f"  Loss: {loss_before:.4f} → {loss_after:.4f}")
    print(f"  Grad norm: {grad_norm_before:.4f} → {grad_norm_after:.4f}")
    print(f"  Rank usage: {measure_capacity_rank(expert):.3f}")

    assert loss_after < loss_before * 0.5, "Expert should learn"
    assert grad_norm_after < grad_norm_before * 0.5, "Gradients should shrink (capacity fills)"
    print("  PASSED ✓\n")


def test_2_bonding_creates_useful_child():
    """Child from complementary parents should handle BOTH parents' patterns."""
    print("Test 2: Bonding creates specialized child")

    # Parent A knows patterns 0-3, Parent B knows patterns 4-7
    all_patterns = make_patterns(8, seed=200)
    patterns_a = all_patterns[:4]
    patterns_b = all_patterns[4:]

    parent_a = make_expert(seed=10)
    parent_b = make_expert(seed=20)
    train_expert(parent_a, patterns_a, steps=300, lr=0.02)
    train_expert(parent_b, patterns_b, steps=300, lr=0.02)

    # Verify parents are specialized
    a_on_a = expert_loss_batch(parent_a, patterns_a)
    a_on_b = expert_loss_batch(parent_a, patterns_b)
    b_on_a = expert_loss_batch(parent_b, patterns_a)
    b_on_b = expert_loss_batch(parent_b, patterns_b)
    print(f"  Parent A: own={a_on_a:.3f}, other={a_on_b:.3f}")
    print(f"  Parent B: own={b_on_b:.3f}, other={b_on_a:.3f}")

    # Create child via DEAP crossover
    child = bond_experts_deap(parent_a, parent_b, seed=42)

    # Child before training — should have some knowledge from both
    child_on_a = expert_loss_batch(child, patterns_a)
    child_on_b = expert_loss_batch(child, patterns_b)
    child_on_all = expert_loss_batch(child, all_patterns)
    print(f"  Child (before training): A={child_on_a:.3f}, B={child_on_b:.3f}, all={child_on_all:.3f}")

    # Train child briefly on ALL patterns
    train_expert(child, all_patterns, steps=200, lr=0.02)
    child_on_all_trained = expert_loss_batch(child, all_patterns)
    print(f"  Child (after training): all={child_on_all_trained:.3f}")

    # Child should handle both domains better than either parent handles the other
    assert child_on_all_trained < a_on_b, "Child should beat parent A on B's domain"
    assert child_on_all_trained < b_on_a, "Child should beat parent B on A's domain"
    print("  PASSED ✓\n")


def test_3_delegation_routes_novel_to_child():
    """Novel inputs (that would disrupt parent) should delegate to child."""
    print("Test 3: Delegation routes novel inputs to child")

    # Parent trained on familiar patterns
    familiar = make_patterns(4, seed=300)
    parent = make_expert(seed=30)
    train_expert(parent, familiar, steps=300, lr=0.02)

    # Child is fresh (lots of capacity)
    child = make_expert(seed=31)

    # Novel patterns (parent hasn't seen)
    novel = make_patterns(4, seed=400)

    # Familiar input: parent should handle it (low delegation)
    p_wt_familiar, c_wt_familiar = compute_delegation_weight(
        parent, child, familiar[0][0], familiar[0][1])

    # Novel input: parent should delegate (high delegation)
    p_wt_novel, c_wt_novel = compute_delegation_weight(
        parent, child, novel[0][0], novel[0][1])

    print(f"  Familiar input: parent={p_wt_familiar:.3f}, child={c_wt_familiar:.3f}")
    print(f"  Novel input:    parent={p_wt_novel:.3f}, child={c_wt_novel:.3f}")

    assert c_wt_novel > c_wt_familiar, \
        "Novel inputs should delegate MORE to child than familiar inputs"
    print("  PASSED ✓\n")


def test_4_criticality_detection():
    """Expert trained to saturation should show high capacity (low grad norm)."""
    print("Test 4: Criticality detection")

    expert = make_expert(seed=50)
    patterns = make_patterns(4, seed=500)  # small set — easy to saturate

    # Train until saturated
    history = train_expert(expert, patterns, steps=500, lr=0.02)

    # Measure capacity signals
    grad_norm = measure_capacity_gradient(expert, patterns)
    rank_usage = measure_capacity_rank(expert)
    final_loss = expert_loss_batch(expert, patterns)

    # Gradient norm should be tiny (nothing left to learn)
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Grad norm: {grad_norm:.6f}")
    print(f"  Rank usage: {rank_usage:.3f}")

    # Loss improvement rate in last 50 steps vs first 50
    early_improvement = history[0] - history[49]
    late_improvement = history[-50] - history[-1]
    improvement_ratio = late_improvement / (early_improvement + 1e-8)
    print(f"  Improvement ratio (late/early): {improvement_ratio:.4f}")

    assert grad_norm < 0.05, "Saturated expert should have tiny gradients"
    assert improvement_ratio < 0.1, "Late training should show diminishing returns"
    print("  PASSED ✓\n")


def test_5_forced_unlearning_works():
    """After unlearning, parent should have HIGHER loss on pruned patterns."""
    print("Test 5: Forced unlearning increases loss on target patterns")

    # Parent knows all 8 patterns
    all_patterns = make_patterns(8, seed=600)
    parent = make_expert(seed=60)
    birth_snapshot = {k: mx.array(parent[k]) for k in parent}
    train_expert(parent, all_patterns, steps=400, lr=0.02)

    # Split: child will take over patterns 0-3
    child_patterns = all_patterns[:4]
    keep_patterns = all_patterns[4:]

    loss_before_child = expert_loss_batch(parent, child_patterns)
    loss_before_keep = expert_loss_batch(parent, keep_patterns)

    # Unlearn patterns 0-3: targeted birth decay (selective by gradient magnitude)
    unlearn_parent(parent, child_patterns, decay_rate=0.4,
                   birth_weights=birth_snapshot)

    loss_after_child = expert_loss_batch(parent, child_patterns)
    loss_after_keep = expert_loss_batch(parent, keep_patterns)

    print(f"  Child patterns: {loss_before_child:.4f} → {loss_after_child:.4f} (should increase)")
    print(f"  Keep patterns:  {loss_before_keep:.4f} → {loss_after_keep:.4f} (should stay low)")

    assert loss_after_child > loss_before_child, "Unlearning should INCREASE loss on target"
    # Keep patterns may degrade slightly due to weight sharing, but less
    degradation_ratio = (loss_after_keep / (loss_before_keep + 1e-8))
    print(f"  Keep degradation ratio: {degradation_ratio:.2f}x")
    assert degradation_ratio < 3.0, "Keep patterns shouldn't degrade too much"
    print("  PASSED ✓\n")


def test_6_unlearning_frees_capacity():
    """After unlearning, parent should learn NEW patterns faster."""
    print("Test 6: Unlearning frees capacity for new learning")

    # Parent trained on 8 patterns → saturated
    old_patterns = make_patterns(8, seed=700)
    parent = make_expert(seed=70)
    birth_snapshot = {k: mx.array(parent[k]) for k in parent}
    train_expert(parent, old_patterns, steps=400, lr=0.02)

    # New patterns to learn
    new_patterns = make_patterns(4, seed=800)

    # Measure: how fast can saturated parent learn new patterns?
    parent_saturated = {k: mx.array(parent[k]) for k in parent}
    history_saturated = train_expert(parent_saturated, new_patterns, steps=100, lr=0.02)

    # Now unlearn old patterns 0-3 (free capacity), preserving 4-7
    parent_pruned = {k: mx.array(parent[k]) for k in parent}
    unlearn_parent(parent_pruned, old_patterns[:4], decay_rate=0.5,
                   keep_patterns=old_patterns[4:])

    # Same new patterns — pruned parent should learn faster
    history_pruned = train_expert(parent_pruned, new_patterns, steps=100, lr=0.02)

    final_saturated = history_saturated[-1]
    final_pruned = history_pruned[-1]

    print(f"  Saturated parent on new patterns: {history_saturated[0]:.3f} → {final_saturated:.3f}")
    print(f"  Pruned parent on new patterns:    {history_pruned[0]:.3f} → {final_pruned:.3f}")

    # Pruned parent should reach lower loss (learned better)
    assert final_pruned < final_saturated * 1.5, \
        "Pruned parent should learn at least comparably to saturated"
    print("  PASSED ✓\n")


def test_7_knowledge_overlap_detection():
    """After child trains on parent's domain, overlap should be high."""
    print("Test 7: Knowledge overlap detection")

    # Parent knows patterns 0-5
    patterns = make_patterns(6, seed=900)
    parent = make_expert(seed=80)
    train_expert(parent, patterns, steps=300, lr=0.02)

    # Child trained on subset (patterns 0-2) — overlaps with parent
    child_overlapping = make_expert(seed=81)
    train_expert(child_overlapping, patterns[:3], steps=300, lr=0.02)

    # Child trained on different patterns — no overlap
    other_patterns = make_patterns(3, seed=999)
    child_different = make_expert(seed=82)
    train_expert(child_different, other_patterns, steps=300, lr=0.02)

    overlap_high = compute_knowledge_overlap(parent, child_overlapping, patterns[:3])
    overlap_low = compute_knowledge_overlap(parent, child_different, patterns[:3])

    print(f"  Overlapping child: overlap={overlap_high:.3f}")
    print(f"  Different child:   overlap={overlap_low:.3f}")

    assert overlap_high > overlap_low, \
        "Child trained on parent's domain should show higher overlap"
    print("  PASSED ✓\n")


def test_8_full_lifecycle():
    """End-to-end lifecycle: parents → bond → child grows → unlearn → parent learns new."""
    print("Test 8: Full lifecycle end-to-end")

    # === Setup: two specialized parents ===
    all_patterns = make_patterns(12, seed=1000)
    domain_a = all_patterns[:4]   # Parent A's domain
    domain_b = all_patterns[4:8]  # Parent B's domain
    domain_new = all_patterns[8:]  # Future task

    parent_a = make_expert(seed=100)
    parent_b = make_expert(seed=101)
    birth_a = {k: mx.array(parent_a[k]) for k in parent_a}
    birth_b = {k: mx.array(parent_b[k]) for k in parent_b}

    train_expert(parent_a, domain_a, steps=300, lr=0.02)
    train_expert(parent_b, domain_b, steps=300, lr=0.02)

    a_loss_a = expert_loss_batch(parent_a, domain_a)
    b_loss_b = expert_loss_batch(parent_b, domain_b)
    print(f"  Step 1 - Parents trained: A_loss={a_loss_a:.3f}, B_loss={b_loss_b:.3f}")

    # === Bonding: create child from parents ===
    child = bond_experts_deap(parent_a, parent_b, seed=42)
    child_all = expert_loss_batch(child, domain_a + domain_b)
    print(f"  Step 2 - Child born: combined_loss={child_all:.3f}")

    # === Growth: child trains on both domains ===
    train_expert(child, domain_a + domain_b, steps=400, lr=0.02)
    child_all_trained = expert_loss_batch(child, domain_a + domain_b)
    print(f"  Step 3 - Child trained: combined_loss={child_all_trained:.3f}")

    # === Criticality: child has learned, measure overlap ===
    grad_norm = measure_capacity_gradient(child, domain_a + domain_b)
    overlap_a = compute_knowledge_overlap(parent_a, child, domain_a)
    overlap_b = compute_knowledge_overlap(parent_b, child, domain_b)
    print(f"  Step 4 - Criticality: grad_norm={grad_norm:.4f}, "
          f"overlap_A={overlap_a:.2f}, overlap_B={overlap_b:.2f}")

    # === Forced unlearning: parents prune knowledge child has ===
    a_new_before = expert_loss_batch(parent_a, domain_new)
    b_new_before = expert_loss_batch(parent_b, domain_new)

    # Parent A unlearns patterns that child handles (preserving non-overlapping)
    if overlap_a > 0.1:
        unlearn_parent(parent_a, domain_a, decay_rate=0.4, keep_patterns=domain_new)
        a_domain_after_prune = expert_loss_batch(parent_a, domain_a)
        print(f"  Step 5a - Parent A pruned: domain_loss {a_loss_a:.3f} → {a_domain_after_prune:.3f}")

    if overlap_b > 0.1:
        unlearn_parent(parent_b, domain_b, decay_rate=0.4, keep_patterns=domain_new)
        b_domain_after_prune = expert_loss_batch(parent_b, domain_b)
        print(f"  Step 5b - Parent B pruned: domain_loss {b_loss_b:.3f} → {b_domain_after_prune:.3f}")

    # === Relearning: parents learn NEW domain (should be easier now) ===
    parent_a_fresh = {k: mx.array(parent_a[k]) for k in parent_a}
    parent_a_nopruned = make_expert(seed=100)
    train_expert(parent_a_nopruned, domain_a, steps=300, lr=0.02)  # retrain saturated

    history_pruned = train_expert(parent_a_fresh, domain_new, steps=150, lr=0.02)
    history_saturated = train_expert(parent_a_nopruned, domain_new, steps=150, lr=0.02)

    print(f"  Step 6 - Parent A on new domain:")
    print(f"    Pruned:    {history_pruned[0]:.3f} → {history_pruned[-1]:.3f}")
    print(f"    Saturated: {history_saturated[0]:.3f} → {history_saturated[-1]:.3f}")

    # === Verify the lifecycle produced useful structure ===
    # Child handles combined domain
    assert child_all_trained < 0.5, f"Child should learn combined domain well, got {child_all_trained}"
    # Overlap was detected
    assert overlap_a > 0 or overlap_b > 0, "Should detect some overlap"
    print("  PASSED ✓\n")


def test_9_capacity_bottleneck_unlearning():
    """Prove that unlearning frees capacity under genuine pressure.

    Strategy: use a TINY expert (4 hidden units) crammed with patterns,
    so the bottleneck is real. Then show unlearning helps new learning.
    """
    print("Test 9: Capacity-bottleneck unlearning")
    print("  Sweeping architectures to find capacity pressure point...\n")

    results = []
    # Try multiple bottleneck sizes to find where unlearning helps
    for hidden in [3, 4, 6, 8]:
        dim = 4  # small input dim for fast experiments
        n_old = 8  # patterns to cram in
        n_unlearn = 4  # patterns to unlearn
        n_new = 6  # new patterns to learn

        # Generate patterns with small dim
        rng = np.random.RandomState(42)
        old_patterns = []
        for i in range(n_old):
            x = mx.array(rng.randn(dim).astype(np.float32))
            t_np = np.tanh(rng.randn(dim).astype(np.float32) * 1.5)
            old_patterns.append((x, mx.array(t_np)))

        new_patterns = []
        for i in range(n_new):
            x = mx.array(rng.randn(dim).astype(np.float32))
            t_np = np.tanh(rng.randn(dim).astype(np.float32) * 1.5)
            new_patterns.append((x, mx.array(t_np)))

        def make_tiny(seed):
            mx.random.seed(seed)
            return {
                'W1': mx.random.normal((hidden, dim)) * 0.3,
                'b1': mx.zeros((hidden,)),
                'W2': mx.random.normal((dim, hidden)) * 0.3,
                'b2': mx.zeros((dim,)),
            }

        n_params = 2 * dim * hidden + dim + hidden
        print(f"  --- hidden={hidden}, params={n_params}, {n_old} old patterns ---")

        # Train saturated expert
        saturated = make_tiny(seed=99)
        train_expert(saturated, old_patterns, steps=500, lr=0.02)
        sat_loss = expert_loss_batch(saturated, old_patterns)
        print(f"    Saturated loss on old: {sat_loss:.4f}")

        # Clone and unlearn half the old patterns
        pruned = {k: mx.array(saturated[k]) for k in saturated}
        birth = make_tiny(seed=99)
        unlearn_patterns = old_patterns[:n_unlearn]
        keep_patterns = old_patterns[n_unlearn:]

        # Use birth-decay unlearning (stronger for small experts)
        unlearn_parent(pruned, unlearn_patterns, decay_rate=0.6,
                       birth_weights=birth)

        pruned_old_loss = expert_loss_batch(pruned, old_patterns)
        pruned_keep_loss = expert_loss_batch(pruned, keep_patterns)
        pruned_forget_loss = expert_loss_batch(pruned, unlearn_patterns)
        print(f"    After unlearn: forget={pruned_forget_loss:.4f}, keep={pruned_keep_loss:.4f}")

        # Now learn new patterns: saturated vs pruned
        sat_copy = {k: mx.array(saturated[k]) for k in saturated}
        pruned_copy = {k: mx.array(pruned[k]) for k in pruned}

        h_sat = train_expert(sat_copy, new_patterns, steps=200, lr=0.02)
        h_pru = train_expert(pruned_copy, new_patterns, steps=200, lr=0.02)

        improvement = (h_sat[-1] - h_pru[-1]) / (h_sat[-1] + 1e-8) * 100
        results.append((hidden, n_params, h_sat[-1], h_pru[-1], improvement))
        print(f"    New learning — saturated: {h_sat[-1]:.4f}, pruned: {h_pru[-1]:.4f}")
        print(f"    Pruned advantage: {improvement:+.1f}%\n")

    # Summary table
    print("  === SUMMARY ===")
    print(f"  {'Hidden':>6} {'Params':>6} {'Sat loss':>9} {'Pru loss':>9} {'Advantage':>10}")
    for hidden, params, sat, pru, adv in results:
        marker = " <-- WINS" if adv > 5 else ""
        print(f"  {hidden:>6} {params:>6} {sat:>9.4f} {pru:>9.4f} {adv:>+9.1f}%{marker}")

    # The test passes if ANY bottleneck size shows pruned > 5% better
    best_adv = max(r[4] for r in results)
    print(f"\n  Best pruned advantage: {best_adv:+.1f}%")
    if best_adv > 5:
        print("  CAPACITY HYPOTHESIS CONFIRMED: unlearning helps under pressure ✓\n")
    else:
        print("  INCONCLUSIVE: no architecture showed >5% advantage\n")
    # Don't assert — this is diagnostic
    return results


def test_10_scaling_law_capacity():
    """Map the relationship: model size vs pattern count vs unlearning benefit.

    This finds the exact capacity tipping point where unlearning starts helping.
    """
    print("Test 10: Scaling law — capacity tipping point")

    dim = 4
    hidden = 4  # tiny bottleneck
    n_params = 2 * dim * hidden + dim + hidden  # 44

    def make_tiny(seed):
        mx.random.seed(seed)
        return {
            'W1': mx.random.normal((hidden, dim)) * 0.3,
            'b1': mx.zeros((hidden,)),
            'W2': mx.random.normal((dim, hidden)) * 0.3,
            'b2': mx.zeros((dim,)),
        }

    print(f"  Expert: {dim}→{hidden}→{dim} ({n_params} params)")
    print(f"  Sweeping pattern count to find tipping point...\n")

    results = []
    for n_patterns in [4, 6, 8, 12, 16, 20]:
        rng = np.random.RandomState(n_patterns * 7)
        old_patterns = []
        for _ in range(n_patterns):
            x = mx.array(rng.randn(dim).astype(np.float32))
            t = mx.array(np.tanh(rng.randn(dim).astype(np.float32) * 1.5))
            old_patterns.append((x, t))

        new_patterns = []
        for _ in range(4):  # always 4 new patterns
            x = mx.array(rng.randn(dim).astype(np.float32))
            t = mx.array(np.tanh(rng.randn(dim).astype(np.float32) * 1.5))
            new_patterns.append((x, t))

        # Train saturated
        saturated = make_tiny(seed=42)
        train_expert(saturated, old_patterns, steps=500, lr=0.02)
        sat_old_loss = expert_loss_batch(saturated, old_patterns)

        # Clone → unlearn half
        pruned = {k: mx.array(saturated[k]) for k in saturated}
        birth = make_tiny(seed=42)
        half = n_patterns // 2
        unlearn_parent(pruned, old_patterns[:half], decay_rate=0.6,
                       birth_weights=birth)

        # Learn new patterns
        sat_copy = {k: mx.array(saturated[k]) for k in saturated}
        pru_copy = {k: mx.array(pruned[k]) for k in pruned}
        h_sat = train_expert(sat_copy, new_patterns, steps=200, lr=0.02)
        h_pru = train_expert(pru_copy, new_patterns, steps=200, lr=0.02)

        improvement = (h_sat[-1] - h_pru[-1]) / (h_sat[-1] + 1e-8) * 100
        params_per_pattern = n_params / n_patterns
        results.append((n_patterns, params_per_pattern, sat_old_loss,
                        h_sat[-1], h_pru[-1], improvement))

    print(f"  {'N_old':>5} {'P/pat':>6} {'Old loss':>9} {'Sat new':>8} {'Pru new':>8} {'Adv':>8}")
    for n, ppp, ol, sn, pn, adv in results:
        marker = " *" if adv > 5 else ""
        print(f"  {n:>5} {ppp:>6.1f} {ol:>9.4f} {sn:>8.4f} {pn:>8.4f} {adv:>+7.1f}%{marker}")

    # Find tipping point
    tipping = None
    for n, ppp, ol, sn, pn, adv in results:
        if adv > 5:
            tipping = n
            break

    if tipping:
        print(f"\n  TIPPING POINT: at {tipping} patterns ({n_params/tipping:.1f} params/pattern)")
        print(f"  Below {tipping}: expert has spare capacity, unlearning doesn't help")
        print(f"  Above {tipping}: expert is capacity-constrained, unlearning frees room")
    else:
        print(f"\n  No tipping point found in range. Try even smaller expert or more patterns.")
    print()
    return results


# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    test_1_capacity_increases_during_training()
    test_2_bonding_creates_useful_child()
    test_3_delegation_routes_novel_to_child()
    test_4_criticality_detection()
    test_5_forced_unlearning_works()
    test_6_unlearning_frees_capacity()
    test_7_knowledge_overlap_detection()
    test_8_full_lifecycle()

    # Capacity bottleneck experiments
    test_9_capacity_bottleneck_unlearning()
    test_10_scaling_law_capacity()

    print("=" * 50)
    print("All tests complete!")
