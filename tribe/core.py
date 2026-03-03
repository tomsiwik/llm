"""Core tribe data structures: State, TribeMember, Tribe.

The Tribe is a graph of TribeMembers that self-organizes through
lifecycle operations: bond, freeze, shed, recycle, distill.
"""

import mlx.core as mx
import numpy as np
from enum import Enum

from tribe.expert import forward_batch, loss_on, clone, blend_weights, make_expert, reset_optimizer_state
from tribe.distill import distill
from tribe.patterns import pattern_id


class State(Enum):
    ACTIVE = "active"      # trains, routes, costs compute+memory
    FROZEN = "frozen"      # routes, no training, costs memory only
    DORMANT = "dormant"    # no routing, no training, preserves weights
    RECYCLED = "recycled"  # parameters reused for new member
    SHARED = "shared"      # always-on, trains slowly, provides base signal


class TribeMember:
    """A single expert in the tribe graph."""

    def __init__(self, member_id, weights, generation=0, parent_ids=None):
        self.id = member_id
        self.weights = weights
        self.state = State.ACTIVE
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.birth_weights = clone(weights)
        self.domain = []   # (input, target) pairs this member handles
        self.age = 0       # training steps seen
        self.warmup_remaining = 0  # steps of output warmup left
        self.warmup_total = 0      # original warmup duration

    @property
    def warmup_scale(self):
        """Output scale factor: ramps from 0→1 during warmup."""
        if self.warmup_total <= 0:
            return 1.0
        return 1.0 - self.warmup_remaining / self.warmup_total

    def fitness(self, patterns=None, fwd=None):
        """Lower loss = higher fitness. Returns negative loss."""
        pats = patterns or self.domain
        if not pats:
            return -float('inf')
        return -loss_on(self.weights, pats, fwd=fwd)

    def freeze(self):
        self.state = State.FROZEN

    def dormant(self):
        self.state = State.DORMANT

    def reactivate(self):
        self.state = State.ACTIVE

    @property
    def is_trainable(self):
        return self.state in (State.ACTIVE, State.SHARED)

    @property
    def is_routable(self):
        return self.state in (State.ACTIVE, State.FROZEN, State.SHARED)

    @property
    def is_shared(self):
        return self.state == State.SHARED


class Tribe:
    """A self-organizing graph of expert members."""

    def __init__(self, fwd=None, make_expert_fn=None):
        self.members = {}   # id → TribeMember
        self.edges = {}     # id → set of neighbor ids
        self.next_id = 0
        self.generation = 0
        self.history = []   # (generation, event_string)
        self.fwd = fwd or forward_batch
        self.make_expert_fn = make_expert_fn or make_expert
        self.router = None  # optional SwitchRouter
        # Knowledge tree structure
        self.parent_of = {}    # mid → parent_mid (tree edge)
        self.children_of = {}  # mid → [child_mids]
        self.depth = {}        # mid → depth in tree (0 = root)

    def set_router(self, router):
        """Attach a learned router for Switch-style gating."""
        self.router = router

    def route_learned(self, X, T=None):
        """Route using the attached SwitchRouter. Returns (assignments, aux_loss, stats)."""
        assert self.router is not None, "No router attached — call set_router() first"
        return self.router.route(X, T)

    # ── Graph Management ─────────────────────────────────────

    def add_member(self, weights, generation=None, parent_ids=None):
        mid = self.next_id
        self.next_id += 1
        gen = generation if generation is not None else self.generation
        m = TribeMember(mid, weights, generation=gen, parent_ids=parent_ids)
        self.members[mid] = m
        self.edges[mid] = set()
        self.depth[mid] = 0  # root-level by default
        return m

    def connect(self, id_a, id_b):
        self.edges[id_a].add(id_b)
        self.edges[id_b].add(id_a)

    def neighbors(self, mid):
        return [self.members[nid] for nid in self.edges.get(mid, set())
                if nid in self.members]

    def active_members(self):
        return [m for m in self.members.values() if m.state == State.ACTIVE]

    def routable_members(self):
        return [m for m in self.members.values() if m.is_routable]

    def frozen_members(self):
        return [m for m in self.members.values() if m.state == State.FROZEN]

    def shared_members(self):
        return [m for m in self.members.values() if m.state == State.SHARED]

    # ── Routing ──────────────────────────────────────────────

    def route(self, x, top_k=2):
        """Route input to best members by prediction confidence."""
        routable = self.routable_members()
        if not routable:
            return []
        scores = []
        X = mx.expand_dims(x, axis=0)
        for m in routable:
            pred = self.fwd(m.weights, X)
            scores.append((m, mx.sum(pred ** 2)))
        mx.eval(*[s for _, s in scores])
        scores = [(m, s.item()) for m, s in scores]
        scores.sort(key=lambda s: s[1])
        return [m for m, _ in scores[:top_k]]

    def route_by_loss(self, x, target, top_k=2):
        """Route to members with lowest loss on (input, target)."""
        routable = self.routable_members()
        if not routable:
            return []
        X = mx.expand_dims(x, axis=0)
        T = mx.expand_dims(target, axis=0)
        losses = []
        for m in routable:
            pred = self.fwd(m.weights, X)
            losses.append((m, mx.mean((pred - T) ** 2)))
        mx.eval(*[l for _, l in losses])
        scored = [(m, l.item()) for m, l in losses]
        scored.sort(key=lambda s: s[1])
        return scored[:top_k]

    # ── Knowledge Measurement ────────────────────────────────

    def measure_overlap(self, m1, m2, competence_threshold=None):
        """Fraction of m1's domain where m2 is equally good or better.

        Args:
            competence_threshold: if set, only count patterns where at least
                one expert has loss below this value (filters out mutual ignorance).
        """
        if not m1.domain:
            return 0.0
        X = mx.stack([x for x, _ in m1.domain])
        T = mx.stack([t for _, t in m1.domain])
        preds1 = self.fwd(m1.weights, X)
        preds2 = self.fwd(m2.weights, X)
        losses1 = mx.mean((preds1 - T) ** 2, axis=1)
        losses2 = mx.mean((preds2 - T) ** 2, axis=1)

        if competence_threshold is not None:
            quality_mask = mx.minimum(losses1, losses2) < competence_threshold
            overlap_count = mx.sum((losses2 <= losses1 * 1.1) & quality_mask)
            denom = mx.sum(quality_mask)
            mx.eval(overlap_count, denom)
            return (overlap_count / denom).item() if denom.item() > 0 else 0.0
        else:
            m2_better = mx.sum(losses2 <= losses1 * 1.1)
            mx.eval(m2_better)
            return m2_better.item() / len(m1.domain)

    def unique_knowledge(self, member, margin=0.9):
        """Patterns where this member is the BEST in the tribe.

        Args:
            margin: member must beat best other by this factor (0.9 = 10% better,
                    0.95 = 5% better). Lower margin = stricter uniqueness.
        """
        if not member.domain:
            return []
        others = [m for m in self.routable_members() if m.id != member.id]
        if not others:
            return list(member.domain)
        X = mx.stack([x for x, _ in member.domain])
        T = mx.stack([t for _, t in member.domain])
        preds_m = self.fwd(member.weights, X)
        m_losses = mx.mean((preds_m - T) ** 2, axis=1)
        other_losses = []
        for o in others:
            preds_o = self.fwd(o.weights, X)
            other_losses.append(mx.mean((preds_o - T) ** 2, axis=1))
        best_other = mx.min(mx.stack(other_losses), axis=0)
        mask = m_losses < best_other * margin
        mx.eval(mask)
        return [member.domain[i] for i in range(len(member.domain)) if mask[i].item()]

    def identify_shared_candidates(self, min_domain_ratio=0.5, max_uniqueness=0.2):
        """Find experts that should be promoted to SHARED status.

        Criteria:
        1. Has above-median domain size (serves many patterns)
        2. Low uniqueness: < max_uniqueness of its knowledge is unique
        3. Above median fitness
        4. Currently ACTIVE (not already frozen/shared)
        """
        active = self.active_members()
        active_with_domain = [m for m in active if m.domain]
        if not active_with_domain:
            return []

        # Compute median domain size
        domain_sizes = sorted(len(m.domain) for m in active_with_domain)
        median_domain = domain_sizes[len(domain_sizes) // 2]

        # Compute median fitness
        fitnesses = sorted(m.fitness(fwd=self.fwd) for m in active_with_domain)
        median_fitness = fitnesses[len(fitnesses) // 2]

        candidates = []
        for m in active_with_domain:
            # 1. Above-median domain size
            if len(m.domain) < median_domain:
                continue

            # 2. Low uniqueness
            unique = self.unique_knowledge(m)
            uniqueness = len(unique) / max(len(m.domain), 1)
            if uniqueness >= max_uniqueness:
                continue

            # 3. Above median fitness
            if m.fitness(fwd=self.fwd) < median_fitness:
                continue

            candidates.append((m.id, len(m.domain), uniqueness))

        return candidates

    # ── Lifecycle Operations ─────────────────────────────────

    def promote_to_shared(self, mid):
        """Promote an active member to shared (always-on) status."""
        m = self.members[mid]
        m.state = State.SHARED
        self._log(f"PROMOTE SHARED member {mid} (gen={m.generation})")

    def demote_shared(self, mid):
        """Demote shared member back to active."""
        m = self.members[mid]
        m.state = State.ACTIVE
        self._log(f"DEMOTE SHARED→ACTIVE member {mid}")

    def freeze_member(self, mid):
        m = self.members[mid]
        m.freeze()
        self._log(f"FREEZE member {mid} (gen={m.generation}, "
                  f"domain={len(m.domain)} patterns)")

    def make_dormant(self, mid):
        self.members[mid].dormant()
        self._log(f"DORMANT member {mid}")

    def reactivate_member(self, mid):
        self.members[mid].reactivate()
        self._log(f"REACTIVATE member {mid}")

    def distill_to_neighbors(self, mid, probe_inputs, steps=100):
        """Distill member's unique knowledge into neighbors."""
        member = self.members[mid]
        nbrs = self.neighbors(mid)
        if not nbrs:
            return

        unique = self.unique_knowledge(member)
        if not unique:
            self._log(f"DISTILL member {mid}: no unique knowledge to transfer")
            return

        for x, t in unique:
            best_nbr = min(nbrs, key=lambda n: loss_on(n.weights, [(x, t)], fwd=self.fwd))
            if best_nbr.is_trainable:
                distill(member.weights, best_nbr.weights, [x], steps=steps, fwd=self.fwd)
                nbr_ids = {pattern_id(a, b) for a, b in best_nbr.domain}
                if pattern_id(x, t) not in nbr_ids:
                    best_nbr.domain.append((x, t))

        self._log(f"DISTILL member {mid}: {len(unique)} patterns → neighbors")

    def recycle(self, mid, probe_inputs=None, distill_steps=100,
                optimizer=None, warmup_steps=50):
        """Recycle: distill out, then reincarnate from neighbor blend.

        Args:
            optimizer: if provided, reset Adam moments for the new weights.
            warmup_steps: number of training steps to ramp output from 0→1.
        """
        member = self.members[mid]

        if probe_inputs:
            self.distill_to_neighbors(mid, probe_inputs, steps=distill_steps)

        nbrs = self.neighbors(mid)
        if nbrs:
            fitnesses = []
            for n in nbrs:
                f = -loss_on(n.weights, n.domain, fwd=self.fwd) if n.domain else 0.0
                fitnesses.append(max(f, 0.001))
            total = sum(fitnesses)
            contributions = [f / total for f in fitnesses]
            new_weights = blend_weights([n.weights for n in nbrs], contributions)
            parent_ids = [n.id for n in nbrs]
        else:
            new_weights = self.make_expert_fn(seed=mid * 7 + self.generation * 31)
            parent_ids = []

        member.state = State.RECYCLED
        old_gen = member.generation

        new_member = TribeMember(
            member_id=mid,
            weights=new_weights,
            generation=self.generation,
            parent_ids=parent_ids
        )
        new_member.warmup_remaining = warmup_steps
        new_member.warmup_total = warmup_steps
        self.members[mid] = new_member

        if optimizer is not None:
            reset_optimizer_state(optimizer, weight_keys=list(new_weights.keys()))

        self._log(f"RECYCLE member {mid}: gen {old_gen}→{self.generation}, "
                  f"seeded from {len(nbrs)} neighbors")
        return new_member

    def bond(self, id_a, id_b, seed=0):
        """Create child from two parents via weight blend."""
        a = self.members[id_a]
        b = self.members[id_b]

        child_weights = blend_weights([a.weights, b.weights], [0.5, 0.5])

        rng = np.random.RandomState(seed)
        for k in child_weights:
            noise = rng.randn(*np.array(child_weights[k]).shape).astype(np.float32) * 0.01
            child_weights[k] = child_weights[k] + mx.array(noise)

        child = self.add_member(child_weights,
                                generation=self.generation,
                                parent_ids=[id_a, id_b])
        self.connect(child.id, id_a)
        self.connect(child.id, id_b)

        self._log(f"BOND {id_a}+{id_b} → child {child.id} (gen={self.generation})")
        return child

    # ── Health Check ─────────────────────────────────────────

    def health_check(self, overlap_threshold=0.5, min_active=2,
                     freeze_grad_threshold=None, competence_threshold=None):
        """Evaluate tribe health. Returns list of (type, target, reason).

        Args:
            overlap_threshold: pairwise overlap above this triggers bond recommendation
            min_active: don't freeze if it would leave fewer than this many active
            freeze_grad_threshold: mean squared gradient per param below this → freeze.
                If None (default), uses relative threshold: freeze if grad < 1% of
                mean grad across all active members. Pass a numeric value for absolute.
            competence_threshold: passed to measure_overlap to filter mutual ignorance
        """
        recommendations = []
        n_active = len(self.active_members())
        fwd = self.fwd

        # First pass: compute grad norms for all active members with sufficient domain
        all_grad_norms = []
        for m in self.active_members():
            if m.domain and len(m.domain) >= 3:
                X = mx.stack([x for x, _ in m.domain])
                T = mx.stack([t for _, t in m.domain])
                def loss_fn(w, X=X, T=T):
                    preds = fwd(w, X)
                    return mx.mean((preds - T) ** 2)
                _, grads = mx.value_and_grad(loss_fn)(m.weights)
                total_params = sum(grads[k].size for k in grads)
                grad_norm = sum(mx.sum(grads[k] ** 2).item() for k in grads) / total_params
                all_grad_norms.append((m, grad_norm))

        # Determine effective freeze threshold
        effective_threshold = freeze_grad_threshold
        if effective_threshold is None and all_grad_norms:
            # Relative threshold: freeze if grad < 1% of mean
            mean_grad = sum(g for _, g in all_grad_norms) / len(all_grad_norms)
            effective_threshold = mean_grad * 0.01
        elif effective_threshold is None:
            effective_threshold = 1e-5  # fallback if no grad norms computed

        # Second pass: apply freeze decisions
        for m, grad_norm in all_grad_norms:
            if grad_norm < effective_threshold and n_active > min_active:
                recommendations.append(('freeze', m.id,
                                        f'saturated (grad={grad_norm:.6f})'))
                n_active -= 1

        for m in self.active_members():
            for other in self.routable_members():
                if other.id <= m.id:
                    continue
                if not m.domain or not other.domain:
                    continue
                overlap = self.measure_overlap(m, other,
                                               competence_threshold=competence_threshold)
                if overlap > overlap_threshold:
                    recommendations.append(('overlap', (m.id, other.id),
                                            f'overlap={overlap:.2f}'))

        # Check for shared promotion candidates
        shared_candidates = self.identify_shared_candidates()
        for mid, domain_size, uniqueness in shared_candidates:
            recommendations.append(('promote_shared', mid,
                                   f'generalist (domain={domain_size}, unique={uniqueness:.2f})'))

        return recommendations

    # ── Knowledge Tree ────────────────────────────────────────

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
        """Path from root to this expert (list of member IDs)."""
        path = [mid]
        while mid in self.parent_of:
            mid = self.parent_of[mid]
            path.append(mid)
        return list(reversed(path))

    def subtree(self, mid):
        """All descendant member IDs of this expert."""
        desc = []
        queue = list(self.children_of.get(mid, []))
        while queue:
            child = queue.pop(0)
            desc.append(child)
            queue.extend(self.children_of.get(child, []))
        return desc

    def route_hierarchical(self, x, target=None):
        """Top-down tree routing. O(depth * branching) instead of O(total_experts).

        Start at root level, pick best expert, descend to children, repeat.
        Returns path [root, ..., leaf] — each more specialized.
        """
        # If no tree structure, fall back to all routable members
        candidates = self.tree_roots()
        if not candidates:
            candidates = self.routable_members()
        path = []

        while candidates:
            if target is not None:
                scored = [(m, loss_on(m.weights, [(x, target)], fwd=self.fwd))
                          for m in candidates if m.is_routable]
            else:
                X = mx.expand_dims(x, 0)
                scored = []
                for m in candidates:
                    if not m.is_routable:
                        continue
                    logits = self.fwd(m.weights, X)
                    conf = mx.max(mx.softmax(logits, axis=-1)).item()
                    scored.append((m, -conf))

            if not scored:
                break
            scored.sort(key=lambda s: s[1])
            best = scored[0][0]
            path.append(best)

            children_ids = self.children_of.get(best.id, [])
            candidates = [self.members[cid] for cid in children_ids
                          if cid in self.members and self.members[cid].is_routable]

        return path

    def hierarchical_forward(self, path, X):
        """Aggregate outputs along tree path with depth-weighted mixing.
        Deeper = more specialized = higher weight (2^depth).
        """
        if not path:
            return None
        outputs = []
        weights_list = []
        for i, member in enumerate(path):
            out = self.fwd(member.weights, X)
            depth_weight = 2 ** i
            scale = member.warmup_scale
            outputs.append(out * scale)
            weights_list.append(depth_weight)
        total_w = sum(weights_list)
        mixed = sum(w/total_w * o for w, o in zip(weights_list, outputs))
        return mixed

    def bond_hierarchical(self, parent_mid, seed=None):
        """Create specialized child initialized from parent + noise.
        Unlike flat bond (merge two parents), this creates a DEEPER specialist.
        """
        parent = self.members[parent_mid]
        child_weights = clone(parent.weights)
        rng_seed = seed if seed is not None else self.generation * 13 + parent_mid
        rng = np.random.RandomState(rng_seed)
        for k in child_weights:
            noise = rng.randn(*np.array(child_weights[k]).shape).astype(np.float32) * 0.01
            child_weights[k] = child_weights[k] + mx.array(noise)
        child = self.add_child(parent_mid, child_weights)
        self._log(f"TREE GROW: {parent_mid} → child {child.id} "
                  f"(depth={self.depth[child.id]})")
        return child

    def prune_tree(self, mid):
        """Remove leaf node from tree. Parent becomes active leaf if all children pruned."""
        if self.children_of.get(mid):
            for child_id in list(self.children_of[mid]):
                self.prune_tree(child_id)

        parent_id = self.parent_of.get(mid)
        if parent_id is not None:
            self.children_of[parent_id].remove(mid)
            if not self.children_of[parent_id]:
                del self.children_of[parent_id]
                parent = self.members[parent_id]
                if parent.state == State.FROZEN:
                    parent.reactivate()
                    self._log(f"TREE PRUNE: {mid} removed, parent {parent_id} reactivated")

        if mid in self.parent_of:
            del self.parent_of[mid]
        if mid in self.depth:
            del self.depth[mid]
        if mid in self.children_of:
            del self.children_of[mid]
        # Mark as recycled
        if mid in self.members:
            self.members[mid].state = State.RECYCLED
            self._log(f"TREE PRUNE: removed {mid}")

    def tree_health_check(self):
        """Health check that respects tree structure.

        Returns list of (action, target_id, reason) recommendations.
        """
        recommendations = []
        for m in self.active_members():
            depth = self.depth.get(m.id, 0)
            # Deep experts must be highly specialized
            if depth >= 2 and m.domain:
                unique = self.unique_knowledge(m, margin=0.8)
                if len(unique) < 3:
                    recommendations.append(('prune', m.id,
                        f'deep expert with only {len(unique)} unique patterns'))
            # Check if parent should spawn a child (max depth 3)
            if m.domain and len(m.domain) > 50 and depth < 3:
                recommendations.append(('split', m.id,
                    f'large domain ({len(m.domain)}) at depth {depth}'))
        return recommendations

    # ── Logging ──────────────────────────────────────────────

    def _log(self, msg):
        self.history.append((self.generation, msg))

    def print_status(self):
        print(f"  Tribe (gen={self.generation}): "
              f"{len(self.active_members())} active, "
              f"{len(self.frozen_members())} frozen, "
              f"{len(self.shared_members())} shared, "
              f"{sum(1 for m in self.members.values() if m.state == State.DORMANT)} dormant")
        for m in sorted(self.members.values(), key=lambda x: x.id):
            if m.state == State.RECYCLED:
                continue
            domain_loss = loss_on(m.weights, m.domain, fwd=self.fwd) if m.domain else float('nan')
            depth_info = f" depth={self.depth.get(m.id, 0)}"
            print(f"    [{m.id}] {m.state.value:8s} gen={m.generation} "
                  f"domain={len(m.domain)} loss={domain_loss:.4f} "
                  f"parents={m.parent_ids}{depth_info}")
