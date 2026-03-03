# PLAN: Expert Lifecycle Implementation

## Overview

Implement the expert lifecycle system: bonding → growth → delegation →
criticality → forced unlearning → role reversal. Build incrementally,
test each stage in isolation before combining.

## Architecture

### Expert State (extended)

```python
expert = {
    'id': int,
    'fc1': str,           # sd key for first layer
    'fc2': str,           # sd key for second layer
    'lm_head': str,       # sd key for expert-specific head
    'router_key': list,   # routing vector

    # Lifecycle state (NEW)
    'parent_ids': [],     # [] for root experts, [a_id, b_id] for children
    'child_ids': [],      # IDs of children spawned from this expert
    'generation': 0,      # 0 = root, 1 = child, 2 = grandchild
    'capacity': 0.0,      # [0, 1] — how "full" is this expert
    'lr_multiplier': 1.0, # decays as capacity fills
    'frozen_at_step': None,  # step when expert froze (None = still learning)
    'birth_step': 0,      # step when expert was created
    'birth_weights': {},  # snapshot of parent weights at birth (for unlearning)
    'activation_history': [],  # rolling window of recent activations
}
```

### Knowledge Tree

```python
class KnowledgeTree:
    """Tracks parent-child relationships and lifecycle state."""

    def __init__(self):
        self.experts = {}       # id → expert dict
        self.edges = {}         # child_id → (parent_a_id, parent_b_id)

    def bond(self, parent_a, parent_b, sd, n_embd, rng):
        """Create child from two parents via genetic crossover."""
        child = create_offspring(sd, parent_a, parent_b, ...)
        child['parent_ids'] = [parent_a['id'], parent_b['id']]
        child['generation'] = max(parent_a['generation'],
                                   parent_b['generation']) + 1
        parent_a['child_ids'].append(child['id'])
        parent_b['child_ids'].append(child['id'])
        return child

    def should_bond(self, expert_a, expert_b, routing_log):
        """Determine if two experts should produce offspring."""
        # Bond when: router keys are close enough to suggest
        # complementary knowledge, AND both are near capacity
        # (i.e., they need help), AND they co-activate frequently
        ...

    def update_capacity(self, expert, sd):
        """Measure how full an expert is."""
        ...

    def check_criticality(self, expert):
        """Has this expert reached capacity? Trigger cascade."""
        ...

    def compute_unlearn_pressure(self, parent, child, sd, eval_docs):
        """How much should parent forget in child's domain?"""
        ...
```

## Phase 1: Capacity Measurement

### What is capacity?

An expert is "full" when it can no longer improve on its training data
without degrading on old data. This is the stability-plasticity tradeoff
at the individual expert level.

### Implementation: Gradient-based capacity

```python
def measure_capacity(expert, sd, recent_grads, window=50):
    """Capacity = 1 - (recent_improvement / early_improvement).

    When the expert was young, gradients caused big loss improvements.
    As it fills up, the same gradient magnitude causes smaller improvements.
    When improvements plateau, capacity ≈ 1.0.
    """
    if len(recent_grads) < window:
        return 0.0  # too early to measure

    # Effective gradient magnitude (Fisher-weighted)
    expert_keys = [expert['fc1'], expert['fc2']]
    if 'lm_head' in expert:
        expert_keys.append(expert['lm_head'])

    recent_norms = []
    for grad_snapshot in recent_grads[-window:]:
        norm = sum(mx.sum(grad_snapshot[k] ** 2).item()
                   for k in expert_keys if k in grad_snapshot)
        recent_norms.append(norm)

    early_norms = []
    for grad_snapshot in recent_grads[:window]:
        norm = sum(mx.sum(grad_snapshot[k] ** 2).item()
                   for k in expert_keys if k in grad_snapshot)
        early_norms.append(norm)

    # If gradient norms have dropped significantly, expert is saturating
    early_mean = sum(early_norms) / len(early_norms) if early_norms else 1.0
    recent_mean = sum(recent_norms) / len(recent_norms) if recent_norms else 0.0

    if early_mean < 1e-10:
        return 1.0  # no gradients at all — fully frozen
    return min(1.0, 1.0 - (recent_mean / early_mean))
```

Alternative: **effective rank** of the weight matrix.
```python
def effective_rank(W):
    """How many dimensions of W are actually used?"""
    s = mx.linalg.svd(W, compute_uv=False)
    s_norm = s / mx.sum(s)
    entropy = -mx.sum(s_norm * mx.log(s_norm + 1e-10))
    return mx.exp(entropy).item()

def capacity_by_rank(expert, sd):
    W = sd[expert['fc1']]
    current_rank = effective_rank(W)
    max_rank = min(W.shape)
    return current_rank / max_rank
```

### Test: test_lifecycle.py — Test 1

```python
def test_capacity_increases_during_training():
    """Expert capacity should increase as it trains on data."""
    # Create expert, measure capacity (should be low)
    # Train for N steps
    # Measure capacity again (should be higher)
    # Train more
    # Capacity should approach 1.0
```

### File: `lgme/lifecycle.py` — `measure_capacity()`

## Phase 2: Knowledge Delegation

### When does a parent delegate?

```python
def compute_delegation_weight(parent, child, input_x, sd):
    """How much of this input's gradient should go to the child vs parent?

    Returns (parent_weight, child_weight) summing to 1.0.
    """
    # Factor 1: Does parent already know this?
    parent_loss = forward_expert(parent, input_x, sd)
    parent_knowledge = max(0, 1.0 - parent_loss / baseline_loss)

    # Factor 2: Would this disrupt the parent?
    parent_grad_norm = expert_grad_norm(parent, input_x, sd)
    disruption = parent_grad_norm / (parent.avg_grad_norm + 1e-8)

    # Factor 3: Does the child have capacity?
    child_capacity = child['capacity']
    child_available = 1.0 - child_capacity

    # Delegate more when: parent knows it, it's disruptive, child has room
    delegate_score = (0.3 * parent_knowledge +
                      0.4 * min(1.0, disruption) +
                      0.3 * child_available)

    child_weight = min(0.9, delegate_score)
    parent_weight = 1.0 - child_weight
    return parent_weight, child_weight
```

### Integration with training

In the `train()` loop, when an input activates a parent that has children:

```python
# During loss_fn or gradient computation:
if expert has children:
    for child in children:
        p_wt, c_wt = compute_delegation_weight(parent, child, x, sd)
        # Scale gradients: parent gets p_wt fraction, child gets c_wt
        for key in parent_expert_keys:
            grads[key] *= p_wt
        for key in child_expert_keys:
            grads[key] *= c_wt  # child learns more from this input
```

This requires the child to be in the model's forward pass. Both parent
and child process the input, but the gradient scaling determines who
actually learns from it.

### Test: test_lifecycle.py — Test 2

```python
def test_delegation_routes_novel_to_child():
    """Novel inputs (high parent grad norm) should delegate to child."""
    # Create parent with established knowledge
    # Create child (fresh)
    # Present novel input
    # Check: child_weight > parent_weight
```

### File: `lgme/lifecycle.py` — `compute_delegation_weight()`

## Phase 3: Criticality and Forced Unlearning

### When does criticality trigger?

```python
CRITICALITY_THRESHOLD = 0.85  # capacity at which cascade begins

def check_criticality(expert):
    if expert['capacity'] >= CRITICALITY_THRESHOLD:
        return True
    return False
```

### What happens at criticality?

```python
def criticality_cascade(child, parents, sd, eval_docs):
    """Child reached capacity. Trigger:
    1. Freeze child (reduce lr_multiplier to near zero)
    2. Compute overlap between child and each parent
    3. Force parents to unlearn overlapping knowledge
    4. Update routing priorities
    """
    # 1. Freeze child
    child['lr_multiplier'] = 0.05  # nearly frozen, slight plasticity remains
    child['frozen_at_step'] = current_step

    # 2. For each parent, compute knowledge overlap
    for parent in parents:
        overlap = compute_knowledge_overlap(parent, child, sd, eval_docs)
        # overlap: dict {sd_key: float in [0,1]} — per-parameter overlap

        # 3. Force unlearning: decay parent weights toward pre-birth state
        #    in regions where child is better
        unlearn_parent(parent, child, overlap, sd)

    # 4. Routing: child gets priority for its specialized domain
    # This happens naturally through lower loss → higher routing score
    # But we can also explicitly boost child's router key activation
```

### Knowledge overlap measurement

```python
def compute_knowledge_overlap(parent, child, sd, eval_docs):
    """Which parameters does the child handle better than the parent?

    For each eval doc, compute:
    - child_loss: how well child handles it
    - parent_loss: how well parent handles it
    - If child_loss < parent_loss consistently, those activations
      represent the child's domain → parent should unlearn them.

    Returns per-parameter overlap score.
    """
    child_better_count = {}  # sd_key → count of docs where child is better
    total_count = {}

    for doc in eval_docs:
        # Forward through both
        child_loss = eval_expert_on_doc(child, doc, sd)
        parent_loss = eval_expert_on_doc(parent, doc, sd)

        # Which parameter regions were active?
        child_activations = get_activation_pattern(child, doc, sd)

        for key in child.expert_keys:
            total_count[key] = total_count.get(key, 0) + 1
            if child_loss < parent_loss:
                child_better_count[key] = child_better_count.get(key, 0) + 1

    overlap = {}
    for key in total_count:
        overlap[key] = child_better_count.get(key, 0) / total_count[key]
    return overlap
```

### Forced unlearning implementation

```python
def unlearn_parent(parent, child, overlap, sd,
                   decay_rate=0.1, min_overlap=0.3):
    """Decay parent weights toward a more general state in regions
    where the child has taken over.

    This frees capacity for the parent to learn new things.
    """
    for key in overlap:
        if overlap[key] < min_overlap:
            continue  # child doesn't dominate here — don't touch parent

        # Decay strength proportional to how much better child is
        alpha = decay_rate * overlap[key]

        # Option A: Decay toward zero (aggressive pruning)
        # sd[key] = sd[key] * (1 - alpha)

        # Option B: Decay toward birth snapshot (reset to generalist state)
        if key in parent.get('birth_weights', {}):
            sd[key] = (1 - alpha) * sd[key] + alpha * parent['birth_weights'][key]

        # Option C: Add noise proportional to decay (partial randomization)
        # noise = mx.random.normal(sd[key].shape) * alpha * 0.01
        # sd[key] = sd[key] * (1 - alpha) + noise

    # Update parent capacity (should decrease — capacity freed)
    parent['capacity'] = max(0, parent['capacity'] - decay_rate * 0.5)
```

### Test: test_lifecycle.py — Tests 3-5

```python
def test_criticality_freezes_child():
    """Child reaching capacity threshold should freeze."""

def test_overlap_detection():
    """Child trained on parent's domain should show high overlap."""

def test_unlearning_frees_parent_capacity():
    """After unlearning, parent capacity should decrease,
    and parent should be able to learn new things faster."""
```

### File: `lgme/lifecycle.py` — `criticality_cascade()`, `compute_knowledge_overlap()`, `unlearn_parent()`

## Phase 4: Bonding Trigger

### When should two experts create offspring?

```python
def should_bond(expert_a, expert_b, routing_log, step):
    """Two experts should bond when:
    1. They co-activate frequently (complementary coverage)
    2. Both are near capacity (need help)
    3. Their router keys are close but not identical (overlap with distinction)
    4. Neither has a child that already covers the overlap
    """
    # Co-activation frequency
    coactivation = count_coactivations(expert_a, expert_b, routing_log)
    if coactivation < MIN_COACTIVATION:
        return False

    # Both near capacity
    if expert_a['capacity'] < 0.6 or expert_b['capacity'] < 0.6:
        return False  # still have room to learn, don't need offspring

    # Router key similarity (complementary but not redundant)
    sim = cosine_sim(expert_a['router_key'], expert_b['router_key'])
    if sim < 0.3 or sim > 0.9:
        return False  # too different (unrelated) or too similar (redundant)

    # No existing child covers this overlap
    for child_id in expert_a['child_ids']:
        if child_id in expert_b['child_ids']:
            return False  # already have a shared child

    return True
```

### Bonding at task boundary vs during training

Option A: Only bond at task boundaries (simpler, current approach)
Option B: Bond during training when conditions are met (more dynamic)

Start with Option A. At each task boundary:
1. Compute capacity for all experts
2. Check all expert pairs for bonding criteria
3. Create offspring for qualifying pairs
4. Add offspring to expert list with fresh lr_multiplier=1.0

### File: `lgme/lifecycle.py` — `should_bond()`, `trigger_bonding()`

## Phase 5: Full Lifecycle Integration

### In `run_two_task()`:

```python
# Phase 1: train on set A (standard)
tree = KnowledgeTree(experts)

# Task boundary: check for bonding
for pair in all_expert_pairs(experts):
    if tree.should_bond(*pair, routing_log, step):
        child = tree.bond(*pair, sd, n_embd, rng)
        experts.append(child)
        extend_adam_buffers(child, adam_m, adam_v, sd)

# Phase 2: train on set B with lifecycle
# Inside train() loop:
#   - Track gradients per expert for capacity measurement
#   - Compute delegation weights for parent-child pairs
#   - Scale gradients accordingly
#   - Every K steps: update capacity, check criticality
#   - If criticality: trigger cascade (freeze child, unlearn parent)
```

### In `train()`:

```python
# New parameter: knowledge_tree=None
if knowledge_tree is not None:
    # Every 50 steps: update capacity for all experts
    if step % 50 == 0:
        for exp in mlp_experts:
            knowledge_tree.update_capacity(exp, sd)

        # Check criticality for all non-frozen experts
        for exp in mlp_experts:
            if exp['frozen_at_step'] is None and check_criticality(exp):
                parents = knowledge_tree.get_parents(exp)
                criticality_cascade(exp, parents, sd, old_docs)

    # Gradient delegation for parent-child pairs
    # (integrated into loss_fn or post-gradient scaling)
```

## Phase 6: Experiments

### Experiment 1: Lifecycle vs Flat MoE (ablation)

```
(A) Flat MoE + freeze (baseline, config d)
(B) Flat MoE + freeze + meta-optimizer (config ax)
(C) Lifecycle MoE (bonding + delegation + no pruning)
(D) Lifecycle MoE (bonding + delegation + criticality + pruning)
(E) Lifecycle MoE + meta-optimizer (full system)
```

5 configs × 5 seeds × 500 steps. Measure BWT, L_B, knowledge_efficiency.

### Experiment 2: Lifecycle depth (5-task protocol)

With 5 sequential tasks, the tree can grow deeper:
- After task 1-2: root experts fill up
- After task 2-3: first children spawned
- After task 3-4: children fill up, grandchildren possible
- After task 4-5: three-level tree

Measure: how does tree depth correlate with forgetting?

### Experiment 3: Unlearning effect

Compare parent capacity before and after forced unlearning.
Does the parent actually learn subsequent tasks better?

## Files

| File | Action | Lines |
|------|--------|-------|
| lgme/lifecycle.py | NEW | ~300 |
| lgme/lifecycle.py | measure_capacity, delegation, criticality, bonding | |
| test_lifecycle.py | NEW | ~200 |
| test_lifecycle.py | 8-10 isolated tests per phase | |
| continual.py | Add lifecycle configs, wire KnowledgeTree into train loop | ~60 |

## Implementation Order

```
1. lgme/lifecycle.py: measure_capacity()         — test in isolation
2. lgme/lifecycle.py: compute_delegation_weight() — test in isolation
3. lgme/lifecycle.py: knowledge_overlap + unlearn — test in isolation
4. lgme/lifecycle.py: should_bond + bond          — test in isolation
5. Wire into train() with lifecycle_step_interval=50
6. Wire into run_two_task() and run_five_task()
7. Add configs, run experiments
```

Each step has its own test before integration.

## Success Criteria

1. Full lifecycle (config E) beats flat MoE + meta-optimizer (config B)
   on BOTH BWT and L_B — the lifecycle is not just preventing forgetting,
   it's actually helping learning through better knowledge organization.
2. Parent capacity measurably decreases after forced unlearning.
3. The knowledge tree shows meaningful structure (not random branching).
4. The system works on the 5-task protocol without manual tuning.

## Risk Assessment

- **High risk**: Capacity measurement noise. If capacity is noisy, the
  bonding and criticality triggers fire at wrong times, creating chaos.
  Mitigation: use EMA smoothing and conservative thresholds.

- **Medium risk**: Model too small for lifecycle. With n_embd=16 and
  4 experts, there may not be enough parameters for meaningful children.
  Mitigation: test at n_embd=32 or 64 if needed.

- **Low risk**: Forced unlearning destroys useful knowledge. The decay
  toward birth snapshot is gentle and reversible (parent can relearn).
  Mitigation: small decay_rate (0.05-0.1).
