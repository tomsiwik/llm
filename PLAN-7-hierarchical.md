# Phase 7: Hierarchical Lifecycle + Knowledge Tree (P8 + P9)

## Goal
Implement the knowledge tree from IDEA-EXPERT-LIFECYCLE.md. Experts form a hierarchy of increasing specialization. This is the truly novel contribution — no published work does this.

## Background

From IDEA-EXPERT-LIFECYCLE.md:
```
Root: General backbone (frozen)
├── Expert A: Natural scenes
│   └── Child AB: Animals in natural scenes
│       └── Grandchild ABC: Birds in flight
├── Expert B: Urban scenes
│   └── Child BD: Street signs
└── SHARED Expert S: Common visual features (always-on)
```

Each level is more specialized. Routing traverses the tree top-down. New knowledge flows to leaves.

## Changes

### 7a. Knowledge Tree Data Structure

**`tribe/core.py`** — add tree structure to `Tribe`:
```python
class Tribe:
    def __init__(self, ...):
        ...
        self.parent_of = {}    # mid → parent_mid (tree edge)
        self.children_of = {}  # mid → [child_mids]
        self.depth = {}        # mid → depth in tree (0 = root)

    def add_child(self, parent_mid, child_weights, **kwargs):
        """Add a child expert under parent in the knowledge tree."""
        child = self.add_member(child_weights, parent_ids=[parent_mid], **kwargs)
        self.parent_of[child.id] = parent_mid
        self.children_of.setdefault(parent_mid, []).append(child.id)
        self.depth[child.id] = self.depth.get(parent_mid, 0) + 1
        self.connect(child.id, parent_mid)
        return child

    def tree_roots(self):
        """Experts with no parent (top of tree)."""
        all_children = set(self.parent_of.keys())
        return [m for m in self.routable_members() if m.id not in all_children]

    def tree_path(self, mid):
        """Path from root to this expert."""
        path = [mid]
        while mid in self.parent_of:
            mid = self.parent_of[mid]
            path.append(mid)
        return list(reversed(path))

    def subtree(self, mid):
        """All descendants of this expert."""
        desc = []
        queue = list(self.children_of.get(mid, []))
        while queue:
            child = queue.pop(0)
            desc.append(child)
            queue.extend(self.children_of.get(child, []))
        return desc
```

### 7b. Hierarchical Routing

Route top-down through the tree: start at roots, pick best root, descend to best child, repeat until leaf.

```python
def route_hierarchical(self, x, target=None):
    """Top-down tree routing. O(depth * branching) instead of O(total_experts)."""
    # Start at root level
    candidates = self.tree_roots()
    path = []

    while candidates:
        # Score candidates at this level
        if target is not None:
            scored = [(m, loss_on(m.weights, [(x, target)], fwd=self.fwd))
                      for m in candidates if m.is_routable]
        else:
            # Max-confidence: pick expert with highest softmax peak
            X = mx.expand_dims(x, 0)
            scored = []
            for m in candidates:
                if not m.is_routable:
                    continue
                logits = self.fwd(m.weights, X)
                conf = mx.max(mx.softmax(logits, axis=-1)).item()
                scored.append((m, -conf))  # negate so lower = better

        if not scored:
            break
        scored.sort(key=lambda s: s[1])
        best = scored[0][0]
        path.append(best)

        # Descend to children
        children_ids = self.children_of.get(best.id, [])
        candidates = [self.members[cid] for cid in children_ids
                      if cid in self.members and self.members[cid].is_routable]

    return path  # [root, ..., leaf] — each more specialized
```

### 7c. Hierarchical Training

Each level in the path contributes to the output. Deeper = more specialized = higher weight:

```python
def hierarchical_forward(self, path, X):
    """Aggregate outputs along tree path with depth-weighted mixing."""
    outputs = []
    weights_list = []
    for i, member in enumerate(path):
        out = self.fwd(member.weights, X)
        depth_weight = 2 ** i  # deeper experts contribute more
        scale = member.warmup_scale if hasattr(member, 'warmup_scale') else 1.0
        outputs.append(out * scale)
        weights_list.append(depth_weight)

    # Weighted sum, normalized
    total_w = sum(weights_list)
    mixed = sum(w/total_w * o for w, o in zip(weights_list, outputs))
    return mixed
```

### 7d. Tree Growth via Lifecycle

Bond triggers tree growth when two overlapping experts at the same depth should produce a more specialized child:

```python
def bond_hierarchical(self, parent_mid, data_subset):
    """Create specialized child from parent + new data.

    Unlike flat bond (merge two parents), hierarchical bond creates
    a DEEPER specialist initialized from one parent.
    """
    parent = self.members[parent_mid]

    # Initialize child from parent weights + noise
    child_weights = clone(parent.weights)
    rng = np.random.RandomState(self.generation * 13 + parent_mid)
    for k in child_weights:
        noise = rng.randn(*np.array(child_weights[k]).shape).astype(np.float32) * 0.01
        child_weights[k] = child_weights[k] + mx.array(noise)

    child = self.add_child(parent_mid, child_weights)
    child.domain = list(data_subset)  # specialized domain

    self._log(f"TREE GROW: {parent_mid} → child {child.id} "
              f"(depth={self.depth[child.id]}, domain={len(data_subset)})")
    return child
```

### 7e. Tree Pruning via Lifecycle

Recycle removes leaf nodes; if a parent loses all children, it becomes a leaf again:

```python
def prune_tree(self, mid):
    """Remove leaf node from tree. Parent becomes active leaf."""
    if self.children_of.get(mid):
        # Not a leaf — can't prune directly
        # First prune children recursively
        for child_id in list(self.children_of[mid]):
            self.prune_tree(child_id)

    parent_id = self.parent_of.get(mid)
    if parent_id is not None:
        self.children_of[parent_id].remove(mid)
        if not self.children_of[parent_id]:
            # Parent is now a leaf — reactivate if frozen
            parent = self.members[parent_id]
            if parent.state == State.FROZEN:
                parent.reactivate()
                self._log(f"TREE PRUNE: {mid} removed, parent {parent_id} reactivated")

    del self.parent_of[mid]
    del self.depth[mid]
    self.recycle(mid)
```

### 7f. Depth-Aware Health Check

```python
def tree_health_check(self):
    """Health check that respects tree structure."""
    recommendations = []

    for m in self.active_members():
        depth = self.depth.get(m.id, 0)

        # Shallow experts (depth 0-1): generous thresholds (they're generalists)
        # Deep experts (depth 2+): strict thresholds (they must be highly specialized)
        if depth >= 2:
            unique = self.unique_knowledge(m, margin=0.8)  # stricter
            if len(unique) < 3:
                recommendations.append(('prune', m.id,
                    f'deep expert with only {len(unique)} unique patterns'))

        # Check if a parent should spawn a child
        if len(m.domain) > 50 and depth < 3:  # max depth = 3
            # Check if domain is heterogeneous enough to warrant splitting
            # Use loss variance as a proxy: high variance = diverse domain
            losses = [loss_on(m.weights, [(x,t)]) for x,t in m.domain[:20]]
            if np.std(losses) > 0.5 * np.mean(losses):
                recommendations.append(('split', m.id,
                    f'heterogeneous domain (loss_cv={np.std(losses)/np.mean(losses):.2f})'))

    return recommendations
```

### 7g. Multiple Seeds + Error Bars (P9)

**`bench_cifar100.py`** — add multi-seed runner:
```python
def run_multi_seed(method_fn, n_seeds=5):
    """Run method with multiple seeds, report mean ± std."""
    results = []
    for seed in range(n_seeds):
        fa, bwt, fgt = method_fn(seed=seed)
        results.append({'fa': fa, 'bwt': bwt, 'fgt': fgt})

    fa_vals = [r['fa'] for r in results]
    fgt_vals = [r['fgt'] for r in results]
    print(f"  FA:  {np.mean(fa_vals):.1f}% ± {np.std(fa_vals):.1f}%")
    print(f"  Fgt: {np.mean(fgt_vals):.1f}% ± {np.std(fgt_vals):.1f}%")
    return results
```

## Verification
```bash
# Test tree structure
uv run --with mlx python -c "
from tribe import Tribe, make_expert
t = Tribe()
root = t.add_member(make_expert(seed=0))
child = t.add_child(root.id, make_expert(seed=1))
print(f'root depth={t.depth.get(root.id,0)}, child depth={t.depth[child.id]}')
print(f'path to child: {t.tree_path(child.id)}')
print(f'subtree of root: {t.subtree(root.id)}')
"

# Test hierarchical routing
uv run --with mlx python -c "
# Create 2-level tree, route input, verify path selection
"

# Full benchmark with tree + multi-seed
uv run --with mlx python bench_cifar100.py --hierarchical --seeds=5
```

## Files Modified
| File | Change | Lines |
|------|--------|-------|
| tribe/core.py | Tree structure, `route_hierarchical()`, `bond_hierarchical()`, `prune_tree()`, `tree_health_check()` | ~150 |
| bench_cifar100.py | Add hierarchical tribe method, multi-seed runner | ~80 |

## Success Criteria
- Tree grows to depth 2-3 during 10-task benchmark
- Hierarchical routing is faster than flat routing (O(log n) vs O(n))
- Deeper experts have higher specialization (lower unique knowledge overlap)
- Tree structure reduces forgetting vs flat lifecycle
- Multi-seed results show consistent advantage (p < 0.05)
- This is the submission-ready novel contribution
