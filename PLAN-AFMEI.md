# Implementation Plan: Adaptive Freeze Maps with Expert Inheritance (AFMEI)

## Overview

Implement AFMEI in 5 incremental phases, each independently testable.
Each phase adds one mechanism and produces measurable results.

---

## Phase 1: Continuous Freeze Maps (Temperature-Scaled Gradients)

### What
Replace binary `frozen_keys: set` with per-matrix `freeze_map: dict[str, float]`.
Gradient scaling: `effective_grad[k] = tau[k] * grad[k]`

### Implementation

**File: `lgme/freeze.py`** (NEW)
```python
def init_freeze_map(sd, initial_tau=1.0):
    """Create freeze map with all parameters fully trainable."""
    return {k: initial_tau for k in sd}

def decay_freeze_map(freeze_map, decay_rate=0.7, exclude_keys=None):
    """Decay all temperatures toward frozen at task boundary."""
    for k in freeze_map:
        if exclude_keys and k in exclude_keys:
            continue
        freeze_map[k] *= decay_rate

def apply_freeze_map(grads, freeze_map):
    """Scale gradients by freeze temperature."""
    scaled = {}
    for k in grads:
        tau = freeze_map.get(k, 1.0)
        scaled[k] = tau * grads[k]
    return scaled
```

**File: `lgme/optimizer.py`** (MODIFY)
- Add `freeze_map` parameter to `adam_step`
- Replace `if k in frozen_keys: continue` with `tau = freeze_map.get(k, 1.0); grad = tau * grad`

**File: `continual.py`** (MODIFY)
- New config: `(ae) MoE + freeze-map + exp heads`
- Replace `frozen_keys` with `freeze_map` for AFMEI configs
- At task boundary: `decay_freeze_map(freeze_map, decay_rate=0.7)`

### Measurable Predictions
- With decay_rate=0.7: tau after 1 task boundary = 0.7 (partial plasticity)
- BWT should be between full-freeze (-0.013) and no-freeze (-0.581)
- L_B should be between full-freeze (2.904) and no-freeze (2.234)
- **If decay_rate sweepable**: find the Pareto-optimal BWT/L_B tradeoff

### Verification
```bash
uv run --with mlx python3 -c "
# Sweep decay_rate from 0.3 to 0.95
# Plot BWT vs L_B Pareto frontier
"
```

---

## Phase 2: Temperature Decay with Expert-Specific Maps

### What
Each expert gets its OWN freeze map. When an expert is frozen, only ITS parameters
get temperature decay. New experts start with tau=1.0 on their parameters.

### Implementation

**File: `lgme/freeze.py`** (EXTEND)
```python
def init_expert_freeze_map(sd, expert, initial_tau=1.0):
    """Create freeze map for a single expert's parameters."""
    keys = [expert['fc1'], expert['fc2']]
    if 'lm_head' in expert:
        keys.append(expert['lm_head'])
    return {k: initial_tau for k in keys}

def decay_expert_map(expert_map, decay_rate):
    """Decay one expert's freeze temperatures."""
    for k in expert_map:
        expert_map[k] *= decay_rate

def merge_freeze_maps(expert_maps, shared_map):
    """Combine per-expert maps with shared parameter map."""
    combined = dict(shared_map)
    for emap in expert_maps.values():
        combined.update(emap)
    return combined
```

### Measurable Predictions
- Old expert tau after 2 tasks: ~0.49 (0.7^2)
- New expert tau: 1.0 (fully plastic)
- Effective trainable capacity: ~60% (vs 18% with binary freeze)
- L_B should improve significantly over Phase 1

---

## Phase 3: Expert-Driven Thawing

### What
New experts can raise old experts' temperatures where gradient agreement exists.

### Implementation

**File: `lgme/freeze.py`** (EXTEND)
```python
def compute_gradient_agreement(grads_new, grads_old, keys):
    """Compute cosine similarity of gradients per weight matrix."""
    agreements = {}
    for k in keys:
        if k in grads_new and k in grads_old:
            g_new = grads_new[k].reshape(-1)
            g_old = grads_old[k].reshape(-1)
            cos = mx.sum(g_new * g_old) / (
                mx.sqrt(mx.sum(g_new * g_new)) *
                mx.sqrt(mx.sum(g_old * g_old)) + 1e-8)
            agreements[k] = max(cos.item(), 0.0)  # only positive agreement
    return agreements

def thaw_expert(old_expert_map, agreements, alpha_thaw=0.1):
    """Raise old expert temperatures where gradients agree."""
    for k in old_expert_map:
        if k in agreements:
            old_expert_map[k] = min(
                old_expert_map[k] + alpha_thaw * agreements[k],
                1.0)
```

**File: `continual.py`** (MODIFY)
- After first training step on new task: compute gradient agreement between
  new expert and each frozen expert
- Apply thawing to frozen expert maps
- Repeat periodically (every 50 steps) to refine thawing signal

### Measurable Predictions
- Old experts with thawed parameters should have LOWER old-task loss
  (thawing reinforces shared structure, doesn't destroy it)
- BWT should improve over Phase 2 (less forgetting due to selective adaptation)
- If thawing INCREASES forgetting, alpha_thaw is too high -- reduce it

---

## Phase 4: Child Experts (Genetic Recombination)

### What
Create offspring from two parent experts. Child inherits crossover of parameters
and the maximum (least frozen) temperature map.

### Implementation

**File: `lgme/freeze.py`** (EXTEND)
```python
def create_child_expert(sd, parent1, parent2, child_id, n_embd,
                        parent1_map, parent2_map,
                        crossover='blend', alpha=0.5, rng=None):
    """Create child expert via parameter crossover.

    Returns (child_expert_dict, child_freeze_map)
    """
    if rng is None:
        rng = random.Random(0)

    child_fc1 = f'expert{child_id}.mlp_fc1'
    child_fc2 = f'expert{child_id}.mlp_fc2'

    if crossover == 'blend':
        sd[child_fc1] = alpha * sd[parent1['fc1']] + (1-alpha) * sd[parent2['fc1']]
        sd[child_fc2] = alpha * sd[parent1['fc2']] + (1-alpha) * sd[parent2['fc2']]
    elif crossover == 'uniform':
        # Per-row crossover
        rows1 = sd[parent1['fc1']].shape[0]
        mask = mx.array([1.0 if rng.random() < 0.5 else 0.0 for _ in range(rows1)])
        sd[child_fc1] = mask[:, None] * sd[parent1['fc1']] + (1-mask[:, None]) * sd[parent2['fc1']]
        # ... same for fc2

    # Add small noise for diversity
    noise_std = 0.01
    sd[child_fc1] = sd[child_fc1] + mx.array(
        rng.gauss(0, noise_std) for _ in range(sd[child_fc1].size)).reshape(sd[child_fc1].shape)

    # Child freeze map: max(parent1, parent2) * plasticity_boost
    child_map = {}
    for k_suffix in ['mlp_fc1', 'mlp_fc2']:
        p1_key = parent1[k_suffix.split('_', 1)[0] if '.' in k_suffix else k_suffix]
        p2_key = parent2[k_suffix.split('_', 1)[0] if '.' in k_suffix else k_suffix]
        child_key = f'expert{child_id}.{k_suffix}'
        tau1 = parent1_map.get(p1_key, 0.5)
        tau2 = parent2_map.get(p2_key, 0.5)
        child_map[child_key] = min(max(tau1, tau2) * 1.5, 1.0)  # plasticity boost

    child_expert = {
        'id': child_id,
        'fc1': child_fc1, 'fc2': child_fc2,
        'router_key': init_router_key(sd, child_fc1, n_embd),
        'activation_count': 0,
        'parent_ids': {parent1['id'], parent2['id']},
        'age': 0,
    }
    return child_expert, child_map
```

**File: `continual.py`** (MODIFY)
- At task boundary: identify parent pair candidates (high route overlap, low param redundancy)
- Create child expert
- Train child alongside other experts
- After N steps: evaluate child fitness, decide survival

### Measurable Predictions
- Child should achieve lower loss than either parent on combined task data
- Population diversity (mean pairwise cosine) should increase after child creation
- If child consistently underperforms: crossover strategy is wrong, try 'blend' instead of 'uniform'

---

## Phase 5: Evolutionary Selection

### What
Fitness evaluation and population control. Weak experts culled, strong children promoted.

### Implementation

**File: `lgme/freeze.py`** (EXTEND)
```python
def evaluate_fitness(expert, sd, g, task_sets, uchars, BOS, block_size):
    """Evaluate expert on all known tasks. Returns fitness score."""
    total_loss = 0
    for task_docs in task_sets:
        # Route only through this expert
        loss = eval_single_expert_loss(g, task_docs[:20], expert, sd, uchars, BOS, block_size)
        total_loss += loss
    return -total_loss  # negative loss = fitness

def select_survivors(experts, fitness_scores, max_population):
    """Cull lowest-fitness experts if population exceeds max."""
    if len(experts) <= max_population:
        return experts, []
    ranked = sorted(experts, key=lambda e: fitness_scores.get(e['id'], -999), reverse=True)
    survivors = ranked[:max_population]
    culled = ranked[max_population:]
    return survivors, culled
```

### Measurable Predictions
- Population should stabilize at max_population after 2-3 task boundaries
- Mean fitness should monotonically increase over generations
- Culled experts should be those with lowest activation counts OR highest redundancy

---

## New Configs

| Config | Phase | Key Parameters |
|--------|-------|---------------|
| (ae) freeze-map + exp heads | 1 | decay_rate=0.7 |
| (af) freeze-map decay sweep | 1 | decay_rate in [0.3, 0.5, 0.7, 0.9] |
| (ag) per-expert maps | 2 | per-expert tau, decay=0.7 |
| (ah) expert-driven thawing | 3 | alpha_thaw=0.1 |
| (ai) child experts (blend) | 4 | crossover='blend', alpha=0.5 |
| (aj) child + thawing + EWC | 3+4 | combined |
| (ak) full AFMEI | 5 | all mechanisms, max_pop=6 |

---

## Implementation Order

```
1. Phase 1: freeze.py + modify optimizer.py + config (ae)
   Verify: BWT and L_B between binary extremes
   Time estimate: straightforward, builds on existing code

2. Phase 2: per-expert maps
   Verify: new expert tau=1.0, old expert tau<1.0

3. Phase 3: gradient agreement thawing
   Verify: thawed params don't increase forgetting

4. Phase 4: child experts
   Verify: child outperforms parents on combined data

5. Phase 5: evolutionary selection
   Verify: population stable, mean fitness increases

6. Full experiment: all AFMEI configs x 5 seeds x 2 protocols
```

---

## Files Modified/Created

- `lgme/freeze.py` — NEW: freeze maps, decay, thawing, child creation, selection
- `lgme/optimizer.py` — MODIFY: accept freeze_map instead of frozen_keys
- `continual.py` — MODIFY: new configs, freeze map initialization/decay, thawing loop

---

## Success Criteria (Adversarial Reviewer Standard)

### The Problem We Haven't Solved

After 30 configs, the Pareto frontier is:
```
Config                  BWT        L_B      Tradeoff
(a)  Dense baseline    -0.581     2.234     No protection at all
(d)  MoE+freeze+EWC   -0.469     2.240     Best known CL technique
(ab) Selective freeze  -0.224     2.482     Middle ground (HIGH variance ±0.250)
(r)  Expert heads      -0.013     2.904     No forgetting, can't learn
```

NO config Pareto-dominates (d). After all our novel mechanisms, the textbook
approach (EWC) is still the honest winner on the combined metric.

### Concrete Targets (What the Reviewer Demands)

| Tier | BWT | L_B | std(both) | Verdict |
|------|-----|-----|-----------|---------|
| **Noise** | ≥ -0.45 | ≤ 2.25 | any | "Within error bars of (d), reject" |
| **Incremental** | ≥ -0.30 | ≤ 2.30 | ≤ 0.10 | "Useful but not novel enough" |
| **Minimum viable** | ≥ -0.20 | ≤ 2.35 | ≤ 0.05 | "Accept with revisions" |
| **Strong result** | ≥ -0.10 | ≤ 2.30 | ≤ 0.05 | "Clear accept" |
| **Groundbreaking** | ≥ -0.05 | ≤ 2.25 | ≤ 0.03 | "Best paper candidate" |

### Minimum Viable Result for AFMEI

```
BWT ≥ -0.20  AND  L_B ≤ 2.35  AND  std ≤ 0.05 on both
```

This means:
- 57% less forgetting than (d) MoE+freeze+EWC
- Only 5% worse new-task learning than baseline
- Consistent across seeds (low variance)
- Pareto-dominates EVERY existing config

### Required Evidence Beyond Numbers

1. **Consistency**: Result must hold across 2-task AND 5-task protocols
2. **Scale**: Result must hold at both small (16-dim) and large (32-dim) scales
3. **Ablation**: Each AFMEI component must show measurable independent contribution
4. **Not hyperparameter sensitivity**: Result must be robust to ±20% change in decay_rate, alpha_thaw
5. **Diagnostic explanation**: Show WHY it works via freeze map evolution plots (tau over time)

### Kill Criteria (When to Abandon)

- If Phase 1 (continuous freeze maps) alone can't beat BWT=-0.40 with L_B≤2.40:
  the continuous temperature idea is insufficient and we should pivot
- If Phase 3 (thawing) increases forgetting: gradient agreement signal is too noisy
- If Phase 4 (children) don't outperform parents: crossover is destroying weight geometry
- If total overhead exceeds 3x training time: mechanism is impractical
