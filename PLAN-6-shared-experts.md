# Phase 6: Shared Expert Discovery (P7)

## Goal
Emergently discover which experts should be "always-on" shared experts (DeepSeek-style) vs specialized routed experts. No hardcoding — the lifecycle decides.

## Background

DeepSeek-V3 designates shared experts architecturally: certain experts are always activated to capture common knowledge. But this requires knowing a priori which experts should be shared.

Our insight: **the lifecycle can discover shared experts emergently**. An expert that:
1. Has high fitness across ALL tasks (generalist)
2. Has low unique knowledge (everything it knows, others also know)
3. BUT is consistently routed to (high utilization)

...is a natural candidate for promotion to "shared" status.

## Changes

### 6a. New Lifecycle State: SHARED

**`tribe/core.py`** — extend `State` enum:
```python
class State(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    DORMANT = "dormant"
    RECYCLED = "recycled"
    SHARED = "shared"    # always-on, trains slowly, provides base signal
```

**`TribeMember`** — update properties:
```python
@property
def is_routable(self):
    return self.state in (State.ACTIVE, State.FROZEN, State.SHARED)

@property
def is_trainable(self):
    return self.state in (State.ACTIVE, State.SHARED)

@property
def is_shared(self):
    return self.state == State.SHARED
```

### 6b. Promotion Criteria

**`tribe/core.py`** — add `identify_shared_candidates()`:
```python
def identify_shared_candidates(self, min_coverage=0.7, min_fitness_rank=0.5):
    """Find experts that should be promoted to SHARED status.

    Criteria:
    1. Coverage: handles patterns from >= min_coverage fraction of all tasks
    2. Fitness: above median fitness across all its domain patterns
    3. Low uniqueness: < 20% of its knowledge is unique
    4. Not already SHARED or FROZEN
    """
    candidates = []
    all_task_ids = set()
    for m in self.routable_members():
        task_ids = set()  # which tasks this expert's domain covers
        for x, t in m.domain:
            # Infer task from target/label structure
            task_ids.add(self._task_id(x, t))
        all_task_ids.update(task_ids)

    for m in self.active_members():
        if not m.domain:
            continue
        # Coverage: fraction of tasks represented in domain
        m_tasks = set(self._task_id(x, t) for x, t in m.domain)
        coverage = len(m_tasks) / max(len(all_task_ids), 1)

        # Uniqueness: fraction of domain that's unique to this expert
        unique = self.unique_knowledge(m)
        uniqueness = len(unique) / max(len(m.domain), 1)

        # Fitness: compare to median
        fitnesses = [m2.fitness() for m2 in self.routable_members()
                     if m2.domain]
        median_fit = sorted(fitnesses)[len(fitnesses)//2]
        above_median = m.fitness() >= median_fit

        if coverage >= min_coverage and uniqueness < 0.2 and above_median:
            candidates.append((m.id, coverage, uniqueness))

    return candidates
```

### 6c. Shared Expert Behavior During Training

Shared experts participate in EVERY training step (not just routed ones):

```python
def train_step_with_shared(shared_experts, routed_expert, X, labels, fwd):
    """Training step: shared experts contribute base signal, routed expert specializes."""

    def loss_fn(all_params):
        # Shared expert outputs (always-on)
        shared_output = mx.zeros((X.shape[0], num_classes))
        for se in shared_experts:
            out = fwd(se.weights, X)
            shared_output = shared_output + out / len(shared_experts)

        # Routed expert output
        routed_output = fwd(routed_expert.weights, X)

        # Combined output: shared base + routed specialization
        combined = shared_output + routed_output  # additive composition
        return cross_entropy(combined, labels)

    loss, grads = mx.value_and_grad(loss_fn)(all_params)
    return loss, grads
```

**Key design**: Shared experts use a **lower learning rate** (0.1x) to provide stable base signal. Routed experts learn faster on top.

### 6d. Shared Expert Demotion

If a shared expert's coverage drops (it stops being useful for new tasks), demote back to ACTIVE:

```python
def demote_shared(self, mid):
    """Demote shared expert back to ACTIVE if it's no longer general enough."""
    m = self.members[mid]
    assert m.state == State.SHARED
    m.state = State.ACTIVE
    self._log(f"DEMOTE SHARED→ACTIVE member {mid}")
```

Trigger: coverage < 0.3 after new tasks arrive that the shared expert can't handle.

### 6e. Integration With Router

When routing with a learned router, shared experts are excluded from the routing decision (they're always on):

```python
def route_with_shared(self, X, top_k=2):
    """Route: shared experts always contribute, router selects among non-shared."""
    shared = [m for m in self.routable_members() if m.is_shared]
    candidates = [m for m in self.routable_members() if not m.is_shared]

    # Route among non-shared experts
    if self.router:
        assignments, aux, stats = self.router.route(X_flat, exclude=shared_ids)
    else:
        assignments = self.route_by_loss(X, ...)

    return shared, assignments
```

### 6f. Health Check Integration

**`tribe/core.py`** — extend `health_check()`:
```python
# In health_check():
# Check for shared promotion candidates
shared_candidates = self.identify_shared_candidates()
for mid, coverage, uniqueness in shared_candidates:
    recommendations.append(('promote_shared', mid,
                           f'coverage={coverage:.2f}, uniqueness={uniqueness:.2f}'))
```

## Verification
```bash
# Test shared expert detection
uv run --with mlx python -c "
from tribe import Tribe
# Create tribe, train on 5 tasks, check if any expert is a shared candidate
"

# Benchmark: compare with/without shared expert promotion
uv run --with mlx python bench_cifar100.py
# Expected: shared expert provides stable base, reduces forgetting
```

## Files Modified
| File | Change | Lines |
|------|--------|-------|
| tribe/core.py | SHARED state, `identify_shared_candidates()`, promotion/demotion | ~60 |
| bench_cifar100.py | `train_step_with_shared()`, shared routing integration | ~40 |
| tribe/router.py | `exclude` parameter for routing around shared experts | ~10 |

## Success Criteria
- At least 1 expert naturally promoted to SHARED during 10-task CIFAR benchmark
- Shared expert covers 70%+ of task domains
- System with shared expert has lower forgetting than without
- Shared expert's output is complementary (additive) with routed experts
