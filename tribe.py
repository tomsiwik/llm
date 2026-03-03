"""Tribe: A graph of experts with lifecycle state management.

Each expert (TribeMember) transitions through states:
  ACTIVE  → trains, routes inputs, costs compute+memory
  FROZEN  → routes inputs, no training, costs memory only
  DORMANT → no routing, no training, preserves weights
  RECYCLED → parameters reused for new member

Key mechanism: "reincarnation with memory"
When a member is recycled, its neighbors absorb its unique knowledge
via distillation. The recycled slot gets a new member seeded from
neighbor weights — starting from locally relevant representations.

Over generations, knowledge naturally migrates toward its correct
location in the semantic graph. Each generation, information settles
into more precisely located experts.
"""

import mlx.core as mx
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ── Constants ────────────────────────────────────────────────────

DIM = 4
HIDDEN = 6
N_PARAMS = 2 * DIM * HIDDEN + DIM + HIDDEN  # 58


class State(Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    DORMANT = "dormant"
    RECYCLED = "recycled"


# ── Expert Operations ────────────────────────────────────────────

def make_expert(seed=0):
    mx.random.seed(seed)
    return {
        'W1': mx.random.normal((HIDDEN, DIM)) * 0.3,
        'b1': mx.zeros((HIDDEN,)),
        'W2': mx.random.normal((DIM, HIDDEN)) * 0.3,
        'b2': mx.zeros((DIM,)),
    }


def forward(weights, x):
    h = mx.maximum(weights['W1'] @ x + weights['b1'], 0)
    return weights['W2'] @ h + weights['b2']


def loss_on(weights, patterns):
    """Mean MSE across patterns."""
    if not patterns:
        return float('inf')
    losses = []
    for x, t in patterns:
        pred = forward(weights, x)
        losses.append(mx.mean((pred - t) ** 2).item())
    return sum(losses) / len(losses)


def train(weights, patterns, steps=300, lr=0.02):
    """Train expert on patterns. Returns final loss."""
    for _ in range(steps):
        def loss_fn(w):
            ls = [mx.mean((forward(w, x) - t) ** 2) for x, t in patterns]
            return mx.mean(mx.stack(ls))
        _, grads = mx.value_and_grad(loss_fn)(weights)
        for k in weights:
            weights[k] = weights[k] - lr * grads[k]
        mx.eval(*[weights[k] for k in weights])
    return loss_on(weights, patterns)


def clone(weights):
    return {k: mx.array(weights[k]) for k in weights}


def blend_weights(weight_list, contributions):
    """Weighted blend of multiple experts' weights.

    contributions: list of floats summing to ~1.0
    """
    result = {}
    for k in weight_list[0]:
        blended = sum(c * np.array(w[k]) for w, c in zip(weight_list, contributions))
        result[k] = mx.array(blended.astype(np.float32))
    return result


# ── Knowledge Distillation ───────────────────────────────────────

def distill(teacher, student, probe_inputs, steps=100, lr=0.015):
    """Transfer knowledge from teacher to student on probe inputs.

    Student learns to match teacher's outputs (not ground truth).
    This is how neighbors absorb a dying member's knowledge.
    """
    # Generate teacher's outputs (the knowledge to transfer)
    teacher_outputs = [(x, forward(teacher, x)) for x in probe_inputs]

    for _ in range(steps):
        def loss_fn(w):
            ls = [mx.mean((forward(w, x) - t) ** 2) for x, t in teacher_outputs]
            return mx.mean(mx.stack(ls))
        _, grads = mx.value_and_grad(loss_fn)(student)
        for k in student:
            student[k] = student[k] - lr * grads[k]
        mx.eval(*[student[k] for k in student])


# ── Tribe Member ─────────────────────────────────────────────────

class TribeMember:
    def __init__(self, member_id, weights, generation=0, parent_ids=None):
        self.id = member_id
        self.weights = weights
        self.state = State.ACTIVE
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.birth_weights = clone(weights)
        self.domain = []  # patterns this member is responsible for
        self.age = 0  # training steps seen

    def fitness(self, patterns=None):
        """Lower loss = higher fitness. Returns negative loss."""
        pats = patterns or self.domain
        if not pats:
            return -float('inf')
        return -loss_on(self.weights, pats)

    def freeze(self):
        self.state = State.FROZEN

    def dormant(self):
        self.state = State.DORMANT

    def reactivate(self):
        self.state = State.ACTIVE

    @property
    def is_trainable(self):
        return self.state == State.ACTIVE

    @property
    def is_routable(self):
        return self.state in (State.ACTIVE, State.FROZEN)


# ── Tribe ────────────────────────────────────────────────────────

class Tribe:
    def __init__(self):
        self.members = {}  # id → TribeMember
        self.edges = {}  # id → set of neighbor ids
        self.next_id = 0
        self.generation = 0
        self.history = []  # log of events

    def add_member(self, weights, generation=None, parent_ids=None):
        mid = self.next_id
        self.next_id += 1
        gen = generation if generation is not None else self.generation
        m = TribeMember(mid, weights, generation=gen, parent_ids=parent_ids)
        self.members[mid] = m
        self.edges[mid] = set()
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

    # ── Routing ──────────────────────────────────────────────

    def route(self, x, top_k=2):
        """Route input to best members (lowest loss = best fit)."""
        routable = self.routable_members()
        if not routable:
            return []
        scores = []
        for m in routable:
            pred = forward(m.weights, x)
            # Use negative prediction magnitude as a simple affinity
            # (in real system, use router keys)
            loss_proxy = mx.sum((pred) ** 2).item()  # how "confident"
            scores.append((m, loss_proxy))
        scores.sort(key=lambda x: x[1])
        return [m for m, _ in scores[:top_k]]

    def route_by_loss(self, x, target, top_k=2):
        """Route to members with lowest loss on this (input, target)."""
        routable = self.routable_members()
        scored = [(m, loss_on(m.weights, [(x, target)])) for m in routable]
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    # ── Knowledge Operations ─────────────────────────────────

    def measure_overlap(self, m1, m2):
        """Fraction of m1's domain where m2 is equally good or better."""
        if not m1.domain:
            return 0.0
        m2_better = 0
        for x, t in m1.domain:
            l1 = loss_on(m1.weights, [(x, t)])
            l2 = loss_on(m2.weights, [(x, t)])
            if l2 <= l1 * 1.1:  # m2 is within 10% of m1
                m2_better += 1
        return m2_better / len(m1.domain)

    def unique_knowledge(self, member):
        """Patterns where this member is the BEST in the tribe."""
        unique = []
        for x, t in member.domain:
            m_loss = loss_on(member.weights, [(x, t)])
            others = [m for m in self.routable_members() if m.id != member.id]
            if not others:
                unique.append((x, t))
                continue
            best_other = min(loss_on(o.weights, [(x, t)]) for o in others)
            if m_loss < best_other * 0.9:  # member is >10% better
                unique.append((x, t))
        return unique

    # ── Lifecycle Operations ─────────────────────────────────

    def freeze_member(self, mid):
        m = self.members[mid]
        m.freeze()
        self._log(f"FREEZE member {mid} (gen={m.generation}, domain={len(m.domain)} patterns)")

    def make_dormant(self, mid):
        m = self.members[mid]
        m.dormant()
        self._log(f"DORMANT member {mid}")

    def reactivate_member(self, mid):
        m = self.members[mid]
        m.reactivate()
        self._log(f"REACTIVATE member {mid}")

    def distill_to_neighbors(self, mid, probe_inputs, steps=100):
        """Distill member's unique knowledge into its neighbors.

        Each neighbor absorbs the knowledge closest to its own domain.
        This is how knowledge naturally flows to the right location.
        """
        member = self.members[mid]
        nbrs = self.neighbors(mid)
        if not nbrs:
            return

        unique = self.unique_knowledge(member)
        if not unique:
            self._log(f"DISTILL member {mid}: no unique knowledge to transfer")
            return

        # For each unique pattern, find the neighbor most suited to absorb it
        # (lowest loss = most relevant existing knowledge)
        for x, t in unique:
            best_nbr = min(nbrs, key=lambda n: loss_on(n.weights, [(x, t)]))
            if best_nbr.is_trainable:
                # Distill this pattern into the best neighbor
                distill(member.weights, best_nbr.weights, [x], steps=steps)
                nbr_ids = {pattern_id(a, b) for a, b in best_nbr.domain}
                if pattern_id(x, t) not in nbr_ids:
                    best_nbr.domain.append((x, t))

        self._log(f"DISTILL member {mid}: {len(unique)} patterns → neighbors")

    def recycle(self, mid, probe_inputs=None, distill_steps=100):
        """Recycle a member: distill knowledge out, then reincarnate.

        The recycled slot gets a new member whose weights are seeded
        from neighbor weights — starting with locally relevant knowledge.
        """
        member = self.members[mid]

        # Step 1: Distill unique knowledge to neighbors
        if probe_inputs:
            self.distill_to_neighbors(mid, probe_inputs, steps=distill_steps)

        # Step 2: Create new member from neighbor blend
        nbrs = self.neighbors(mid)
        if nbrs:
            # Weight neighbors by fitness (better neighbors contribute more)
            fitnesses = []
            for n in nbrs:
                f = -loss_on(n.weights, n.domain) if n.domain else 0.0
                fitnesses.append(max(f, 0.001))
            total = sum(fitnesses)
            contributions = [f / total for f in fitnesses]

            new_weights = blend_weights([n.weights for n in nbrs], contributions)
            parent_ids = [n.id for n in nbrs]
        else:
            # No neighbors — random init
            new_weights = make_expert(seed=mid * 7 + self.generation * 31)
            parent_ids = []

        # Step 3: Mark old member as recycled
        member.state = State.RECYCLED
        old_gen = member.generation

        # Step 4: Create new member in the same slot
        new_member = TribeMember(
            member_id=mid,
            weights=new_weights,
            generation=self.generation,
            parent_ids=parent_ids
        )
        self.members[mid] = new_member
        # Edges preserved — same position in the graph

        self._log(f"RECYCLE member {mid}: gen {old_gen}→{self.generation}, "
                  f"seeded from {len(nbrs)} neighbors")
        return new_member

    def bond(self, id_a, id_b, seed=0):
        """Create child from two parents (bonding).

        Child inherits blended weights and connects to both parents.
        """
        a = self.members[id_a]
        b = self.members[id_b]

        # Blend: equal contribution from both parents
        child_weights = blend_weights([a.weights, b.weights], [0.5, 0.5])

        # Add small noise for diversity
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

    def health_check(self, overlap_threshold=0.5, min_active=2):
        """Evaluate tribe health. Returns recommendations.

        Never recommends freezing if it would drop active count below min_active.
        """
        recommendations = []
        n_active = len(self.active_members())

        for m in self.active_members():
            # Check capacity (gradient norm on domain)
            if m.domain and len(m.domain) >= 3:
                def loss_fn(w):
                    ls = [mx.mean((forward(w, x) - t) ** 2) for x, t in m.domain]
                    return mx.mean(mx.stack(ls))
                _, grads = mx.value_and_grad(loss_fn)(m.weights)
                grad_norm = sum(mx.sum(grads[k] ** 2).item() for k in grads)

                if grad_norm < 0.005 and n_active > min_active:
                    recommendations.append(('freeze', m.id,
                                            f'saturated (grad={grad_norm:.4f})'))
                    n_active -= 1  # account for pending freeze

            # Check overlap with all routable members (not just neighbors)
            for other in self.routable_members():
                if other.id <= m.id:
                    continue
                if not m.domain or not other.domain:
                    continue
                overlap = self.measure_overlap(m, other)
                if overlap > overlap_threshold:
                    recommendations.append(('overlap', (m.id, other.id),
                                            f'overlap={overlap:.2f}'))

        return recommendations

    # ── Logging ──────────────────────────────────────────────

    def _log(self, msg):
        self.history.append((self.generation, msg))

    def print_status(self):
        print(f"  Tribe (gen={self.generation}): "
              f"{len(self.active_members())} active, "
              f"{len(self.frozen_members())} frozen, "
              f"{sum(1 for m in self.members.values() if m.state == State.DORMANT)} dormant")
        for m in sorted(self.members.values(), key=lambda x: x.id):
            if m.state == State.RECYCLED:
                continue
            domain_loss = loss_on(m.weights, m.domain) if m.domain else float('nan')
            print(f"    [{m.id}] {m.state.value:8s} gen={m.generation} "
                  f"domain={len(m.domain)} loss={domain_loss:.4f} "
                  f"parents={m.parent_ids}")


# ── Pattern Generation ───────────────────────────────────────────

def pattern_id(x, t):
    """Hashable identity for a pattern pair (for set membership tests)."""
    return (tuple(np.array(x).flat), tuple(np.array(t).flat))


def patterns_match(domain, pattern_ids_to_check):
    """Filter domain to patterns NOT in the given id set."""
    return [(x, t) for x, t in domain
            if pattern_id(x, t) not in pattern_ids_to_check]


def make_clustered_patterns(n_clusters, patterns_per_cluster, seed=0):
    """Generate patterns organized in semantic clusters.

    Each cluster has a center and a fixed transform, and patterns are
    variations around it. This simulates natural knowledge domains.
    Returns (clusters, transforms) where transforms[i] is the matrix
    for cluster i — used to generate related patterns later.
    """
    rng = np.random.RandomState(seed)
    clusters = []
    transforms = []
    for c in range(n_clusters):
        center = rng.randn(DIM).astype(np.float32)
        target_transform = rng.randn(DIM, DIM).astype(np.float32) * 0.5
        transforms.append((center, target_transform))
        cluster = []
        for p in range(patterns_per_cluster):
            x = center + rng.randn(DIM).astype(np.float32) * 0.3
            t = np.tanh(target_transform @ x)
            cluster.append((mx.array(x), mx.array(t.astype(np.float32))))
        clusters.append(cluster)
    return clusters, transforms


# ══════════════════════════════════════════════════════════════════
# MULTI-GENERATION TEST
# ══════════════════════════════════════════════════════════════════

def measure_knowledge_precision(tribe, clusters):
    """How precisely is knowledge located?

    For each cluster, find the BEST expert. Precision = how much better
    the best expert is compared to the second-best.
    Higher = more specialized = knowledge is in the right place.
    """
    precisions = []
    for cluster in clusters:
        losses = []
        for m in tribe.routable_members():
            l = loss_on(m.weights, cluster)
            losses.append((m.id, l))
        if len(losses) < 2:
            continue
        losses.sort(key=lambda x: x[1])
        best_loss = losses[0][1]
        second_loss = losses[1][1]
        # Precision: how much better is best vs second
        if best_loss > 0:
            precision = second_loss / (best_loss + 1e-8)
        else:
            precision = 1.0
        precisions.append(precision)
    return np.mean(precisions) if precisions else 1.0


def measure_system_loss(tribe, all_patterns):
    """Total system loss: for each pattern, use the best routable expert."""
    routable = tribe.routable_members()
    if not routable:
        return float('inf')
    total = 0
    for x, t in all_patterns:
        best = min(loss_on(m.weights, [(x, t)]) for m in routable)
        total += best
    return total / len(all_patterns)


def measure_redundancy(tribe):
    """Average pairwise overlap across all routable member pairs."""
    routable = tribe.routable_members()
    if len(routable) < 2:
        return 0.0
    overlaps = []
    for i, m1 in enumerate(routable):
        for m2 in routable[i+1:]:
            if m1.domain and m2.domain:
                o = tribe.measure_overlap(m1, m2)
                overlaps.append(o)
    return np.mean(overlaps) if overlaps else 0.0


def run_generation(tribe, new_patterns, train_steps=300, label=""):
    """Run one generation: assign patterns, train, health check, adapt."""
    print(f"\n{'='*60}")
    print(f"  GENERATION {tribe.generation}{f' — {label}' if label else ''}")
    print(f"{'='*60}")

    # Assign new patterns to active members (round-robin for simplicity,
    # but route_by_loss would be better in production)
    active = tribe.active_members()
    if not active:
        print("  No active members!")
        return

    # Route each pattern to best active member
    for x, t in new_patterns:
        scored = [(m, loss_on(m.weights, [(x, t)])) for m in active]
        scored.sort(key=lambda s: s[1])
        best = scored[0][0]
        existing_ids = {pattern_id(a, b) for a, b in best.domain}
        if pattern_id(x, t) not in existing_ids:
            best.domain.append((x, t))

    # Train active members on their domains
    for m in active:
        if m.domain:
            final_loss = train(m.weights, m.domain, steps=train_steps)
            m.age += train_steps
            print(f"    Member {m.id}: trained on {len(m.domain)} patterns, "
                  f"loss={final_loss:.4f}")

    # Health check
    recs = tribe.health_check()
    for rec_type, target, reason in recs:
        print(f"    HEALTH: {rec_type} → {target} ({reason})")

    tribe.print_status()


def test_knowledge_consolidation():
    """Multi-generation test: knowledge migrates to correct locations."""
    print("=" * 60)
    print("  KNOWLEDGE CONSOLIDATION OVER GENERATIONS")
    print("  Proving: knowledge naturally settles into precise locations")
    print("=" * 60)

    # Create 4 semantic clusters with 4 patterns each = 16 patterns
    clusters, transforms = make_clustered_patterns(4, 4, seed=42)
    all_patterns = [p for c in clusters for p in c]

    # Tracking metrics across generations
    gen_metrics = []

    # ── Generation 0: Initial tribe with 4 random experts ────
    tribe = Tribe()
    for i in range(4):
        m = tribe.add_member(make_expert(seed=i * 100))
        # Connect in a ring (each expert neighbors with next)
        if i > 0:
            tribe.connect(m.id, m.id - 1)
    tribe.connect(0, 3)  # close the ring

    # Assign ALL patterns to experts (deliberately scattered)
    rng = np.random.RandomState(0)
    shuffled_idx = rng.permutation(len(all_patterns))
    active = tribe.active_members()
    for i, idx in enumerate(shuffled_idx):
        active[i % len(active)].domain.append(all_patterns[idx])

    # Train
    print(f"\n{'='*60}")
    print(f"  GENERATION 0 — Initial random assignment")
    print(f"{'='*60}")
    for m in active:
        final_loss = train(m.weights, m.domain, steps=400)
        print(f"    Member {m.id}: {len(m.domain)} patterns, loss={final_loss:.4f}")

    precision_0 = measure_knowledge_precision(tribe, clusters)
    sys_loss_0 = measure_system_loss(tribe, all_patterns)
    redundancy_0 = measure_redundancy(tribe)
    gen_metrics.append((0, precision_0, sys_loss_0, redundancy_0))
    print(f"\n  Metrics: precision={precision_0:.2f}, sys_loss={sys_loss_0:.4f}, "
          f"redundancy={redundancy_0:.2f}")
    tribe.print_status()

    # ── Generation 1: Detect overlap, bond, specialize ───────
    tribe.generation = 1
    print(f"\n{'='*60}")
    print(f"  GENERATION 1 — Detect overlap, bond where needed")
    print(f"{'='*60}")

    # Find most overlapping pair
    best_overlap = 0
    best_pair = None
    active = tribe.active_members()
    for i, m1 in enumerate(active):
        for m2 in active[i+1:]:
            o = tribe.measure_overlap(m1, m2)
            if o > best_overlap:
                best_overlap = o
                best_pair = (m1.id, m2.id)

    if best_pair and best_overlap > 0.2:
        print(f"    Highest overlap: members {best_pair} at {best_overlap:.2f}")

        # Bond the overlapping pair → child
        child = tribe.bond(best_pair[0], best_pair[1], seed=42)

        # Child's domain = intersection of parent domains
        p1 = tribe.members[best_pair[0]]
        p2 = tribe.members[best_pair[1]]
        shared = []
        for x, t in p1.domain:
            l1 = loss_on(p1.weights, [(x, t)])
            l2 = loss_on(p2.weights, [(x, t)])
            if l1 < 0.5 and l2 < 0.5:  # both handle it
                shared.append((x, t))
        child.domain = shared if shared else p1.domain[:2]

        # Train child on shared knowledge
        train(child.weights, child.domain, steps=300)
        print(f"    Child {child.id}: trained on {len(child.domain)} shared patterns, "
              f"loss={loss_on(child.weights, child.domain):.4f}")

        # Parents shed shared knowledge (retrain on unique only)
        child_ids = {pattern_id(x, t) for x, t in child.domain}
        for pid in best_pair:
            parent = tribe.members[pid]
            unique = patterns_match(parent.domain, child_ids)
            if unique:
                train(parent.weights, unique, steps=150, lr=0.015)
                parent.domain = unique  # shed shared from domain
                print(f"    Parent {pid}: shed → {len(unique)} unique patterns, "
                      f"loss={loss_on(parent.weights, unique):.4f}")

    # Freeze any saturated members
    for m in tribe.active_members():
        if m.domain:
            def loss_fn(w):
                ls = [mx.mean((forward(w, x) - t) ** 2) for x, t in m.domain]
                return mx.mean(mx.stack(ls))
            _, grads = mx.value_and_grad(loss_fn)(m.weights)
            grad_norm = sum(mx.sum(grads[k] ** 2).item() for k in grads)
            if grad_norm < 0.02 and m.age > 200:
                tribe.freeze_member(m.id)

    precision_1 = measure_knowledge_precision(tribe, clusters)
    sys_loss_1 = measure_system_loss(tribe, all_patterns)
    redundancy_1 = measure_redundancy(tribe)
    gen_metrics.append((1, precision_1, sys_loss_1, redundancy_1))
    print(f"\n  Metrics: precision={precision_1:.2f}, sys_loss={sys_loss_1:.4f}, "
          f"redundancy={redundancy_1:.2f}")
    tribe.print_status()

    # ── Generation 2: New patterns arrive, recycle underperformers ─
    tribe.generation = 2
    print(f"\n{'='*60}")
    print(f"  GENERATION 2 — New patterns, recycle + reincarnate")
    print(f"{'='*60}")

    # New patterns from 2 additional clusters
    new_clusters, _ = make_clustered_patterns(2, 3, seed=99)
    new_patterns = [p for c in new_clusters for p in c]
    clusters_all = clusters + new_clusters
    all_patterns_extended = all_patterns + new_patterns

    # Find the least useful active member (if any)
    active = tribe.active_members()
    if active:
        # Score by: how many patterns does this member uniquely handle?
        worst = None
        worst_unique = float('inf')
        for m in active:
            unique = tribe.unique_knowledge(m)
            if len(unique) < worst_unique:
                worst_unique = len(unique)
                worst = m

        if worst and worst_unique <= 1:
            print(f"    Recycling member {worst.id} (only {worst_unique} unique patterns)")

            # Extract probe inputs for distillation
            probe_inputs = [x for x, t in worst.domain] if worst.domain else []

            # Recycle: distill out → blend neighbors → new member
            new_m = tribe.recycle(worst.id, probe_inputs=probe_inputs, distill_steps=80)

            # New member gets the new patterns
            new_m.domain = new_patterns
            train(new_m.weights, new_m.domain, steps=300)
            print(f"    Reincarnated member {new_m.id}: {len(new_m.domain)} new patterns, "
                  f"loss={loss_on(new_m.weights, new_m.domain):.4f}")
        else:
            # No one to recycle — assign new patterns to active members
            for m in active:
                for x, t in new_patterns:
                    if (x, t) not in m.domain:
                        m.domain.append((x, t))
                train(m.weights, m.domain, steps=200)
    else:
        print("    No active members to absorb new patterns")

    precision_2 = measure_knowledge_precision(tribe, clusters_all)
    sys_loss_2 = measure_system_loss(tribe, all_patterns_extended)
    redundancy_2 = measure_redundancy(tribe)
    gen_metrics.append((2, precision_2, sys_loss_2, redundancy_2))
    print(f"\n  Metrics: precision={precision_2:.2f}, sys_loss={sys_loss_2:.4f}, "
          f"redundancy={redundancy_2:.2f}")
    tribe.print_status()

    # ── Generation 3: Another wave, more consolidation ───────
    tribe.generation = 3
    print(f"\n{'='*60}")
    print(f"  GENERATION 3 — Further consolidation")
    print(f"{'='*60}")

    # More patterns from existing cluster domains (reinforcement)
    reinforce_clusters, _ = make_clustered_patterns(4, 2, seed=200)
    reinforce_patterns = [p for c in reinforce_clusters for p in c]

    # Route reinforcement patterns to best members
    for x, t in reinforce_patterns:
        scored = tribe.route_by_loss(x, t, top_k=1)
        if scored:
            best = scored[0][0]
            if best.is_trainable:
                existing_ids = {pattern_id(a, b) for a, b in best.domain}
                if pattern_id(x, t) not in existing_ids:
                    best.domain.append((x, t))

    # Train active members
    for m in tribe.active_members():
        if m.domain:
            train(m.weights, m.domain, steps=200)

    # Check if any frozen member should be recycled
    # (its knowledge is now fully covered by active members)
    for m in tribe.frozen_members():
        unique = tribe.unique_knowledge(m)
        if len(unique) == 0:
            print(f"    Frozen member {m.id} fully covered by others → recycle")
            probe_inputs = [x for x, t in m.domain] if m.domain else []
            tribe.recycle(m.id, probe_inputs=probe_inputs)

    all_patterns_final = all_patterns_extended + reinforce_patterns
    precision_3 = measure_knowledge_precision(tribe, clusters_all)
    sys_loss_3 = measure_system_loss(tribe, all_patterns_final)
    redundancy_3 = measure_redundancy(tribe)
    gen_metrics.append((3, precision_3, sys_loss_3, redundancy_3))
    print(f"\n  Metrics: precision={precision_3:.2f}, sys_loss={sys_loss_3:.4f}, "
          f"redundancy={redundancy_3:.2f}")
    tribe.print_status()

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  EVOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Gen':>3} {'Precision':>10} {'Sys Loss':>10} {'Redundancy':>11}")
    for gen, prec, sl, red in gen_metrics:
        print(f"  {gen:>3} {prec:>10.2f} {sl:>10.4f} {red:>11.2f}")

    # Check: precision should increase (knowledge localizing)
    if precision_3 > precision_0:
        print(f"\n  KNOWLEDGE CONSOLIDATED: precision {precision_0:.2f} → {precision_3:.2f}")
    else:
        print(f"\n  Precision did not improve: {precision_0:.2f} → {precision_3:.2f}")

    # Check: system loss should decrease (getting better overall)
    if sys_loss_3 < sys_loss_0:
        print(f"  SYSTEM IMPROVED: loss {sys_loss_0:.4f} → {sys_loss_3:.4f}")

    # Check: redundancy should decrease (less duplication)
    if redundancy_3 < redundancy_0:
        print(f"  REDUNDANCY REDUCED: {redundancy_0:.2f} → {redundancy_3:.2f}")

    print()
    print("  Lifecycle events:")
    for gen, event in tribe.history:
        print(f"    Gen {gen}: {event}")

    return gen_metrics


def test_reincarnation_with_memory():
    """Prove that recycled members learn faster than random init.

    The key claim: neighbors' knowledge seeds the reincarnated member,
    giving it a head start over random initialization.
    """
    print("\n" + "=" * 60)
    print("  REINCARNATION WITH MEMORY vs RANDOM INIT")
    print("=" * 60)

    advantages = []

    for seed in range(20):
        rng = np.random.RandomState(seed * 17 + 3)

        # Create 3 experts in a triangle
        tribe = Tribe()
        for i in range(3):
            tribe.add_member(make_expert(seed=seed * 100 + i))
        tribe.connect(0, 1)
        tribe.connect(1, 2)
        tribe.connect(0, 2)

        # Each expert learns a cluster — targets use cluster transforms
        clusters, transforms = make_clustered_patterns(3, 3, seed=seed * 50)
        for i, m in enumerate(tribe.active_members()):
            m.domain = clusters[i]
            train(m.weights, m.domain, steps=400)

        # New task: patterns BETWEEN clusters 0 and 1 using BLENDED transforms
        # This is the key: targets are RELATED to what neighbors know
        center_0, transform_0 = transforms[0]
        center_1, transform_1 = transforms[1]
        between_pats = []
        for j in range(4):
            alpha = rng.uniform(0.3, 0.7)
            # Input: interpolated position between clusters
            x_np = (alpha * center_0 + (1 - alpha) * center_1
                    + rng.randn(DIM).astype(np.float32) * 0.2)
            # Target: blended transform (RELATED to both clusters)
            t_np = np.tanh(alpha * transform_0 @ x_np +
                           (1 - alpha) * transform_1 @ x_np)
            between_pats.append((mx.array(x_np), mx.array(t_np.astype(np.float32))))

        # PATH A: Random init expert learns between_pats
        random_expert = make_expert(seed=seed * 999)
        h_random = train(random_expert, between_pats, steps=300)

        # PATH B: Recycled member (seeded from neighbors 0+1) learns between_pats
        m0 = tribe.members[0]
        m1 = tribe.members[1]
        reincarnated_weights = blend_weights(
            [m0.weights, m1.weights],
            [0.5, 0.5]
        )
        h_reincarnated = train(reincarnated_weights, between_pats, steps=300)

        adv = (h_random - h_reincarnated) / (h_random + 1e-8) * 100
        advantages.append(adv)

    adv_arr = np.array(advantages)
    print(f"\n  20-seed results:")
    print(f"    Mean advantage of reincarnation: {adv_arr.mean():+.1f}%")
    print(f"    Std:                             {adv_arr.std():.1f}%")
    print(f"    Reincarnation wins:              {(adv_arr > 0).sum()}/20")
    print(f"    Strong wins (>10%):              {(adv_arr > 10).sum()}/20")
    print(f"    Median:                          {np.median(adv_arr):+.1f}%")

    if adv_arr.mean() > 0:
        print(f"\n  CONFIRMED: Reincarnation with memory beats random init")
    else:
        print(f"\n  NOT CONFIRMED at this scale")

    return adv_arr


def test_knowledge_settling():
    """Track knowledge placement over 6 generations.

    Each generation: new patterns arrive, tribe adapts via lifecycle.
    Show that knowledge migrates to the right experts over time.
    """
    print("\n" + "=" * 60)
    print("  KNOWLEDGE SETTLING OVER GENERATIONS")
    print("  Each gen: new patterns → route → train → health → adapt")
    print("=" * 60)

    # Fixed semantic space: 6 clusters (domains of knowledge)
    all_clusters, transforms = make_clustered_patterns(6, 3, seed=777)
    # We'll reveal clusters one at a time to simulate continual learning

    # Start with 4 experts in a ring
    tribe = Tribe()
    for i in range(4):
        m = tribe.add_member(make_expert(seed=i * 77))
    for i in range(4):
        tribe.connect(i, (i + 1) % 4)
    # Also add cross-connections for richer graph
    tribe.connect(0, 2)
    tribe.connect(1, 3)

    gen_data = []

    for gen in range(6):
        tribe.generation = gen

        # Reveal 1 cluster per generation (first 4 from existing, then 2 new)
        if gen < len(all_clusters):
            new_pats = all_clusters[gen]
        else:
            # Later generations: reinforcement patterns from earlier clusters
            rng_g = np.random.RandomState(gen * 51)
            src = gen % len(all_clusters)
            center, tfm = transforms[src]
            new_pats = []
            for _ in range(3):
                x = center + rng_g.randn(DIM).astype(np.float32) * 0.3
                t = np.tanh(tfm @ x).astype(np.float32)
                new_pats.append((mx.array(x), mx.array(t)))

        # Ensure we have active members (reactivate if needed)
        active = tribe.active_members()
        if not active:
            # Recycle the least useful frozen member
            frozen = tribe.frozen_members()
            if frozen:
                worst_frozen = min(frozen, key=lambda m: len(m.domain))
                probes = [x for x, t in worst_frozen.domain] if worst_frozen.domain else []
                tribe.recycle(worst_frozen.id, probe_inputs=probes, distill_steps=60)
                active = tribe.active_members()
            if not active:
                break

        # Route new patterns to TOP-2 active members (creates natural overlap)
        for x, t in new_pats:
            scored = [(m, loss_on(m.weights, [(x, t)])) for m in active]
            scored.sort(key=lambda s: s[1])
            top = scored[:min(2, len(scored))]
            for m, _ in top:
                existing = {pattern_id(a, b) for a, b in m.domain}
                if pattern_id(x, t) not in existing:
                    m.domain.append((x, t))

        # Train active members
        for m in active:
            if m.domain:
                train(m.weights, m.domain, steps=250)

        # Health check: detect overlap and capacity issues
        recs = tribe.health_check(overlap_threshold=0.3)

        for rec_type, target, reason in recs:
            if rec_type == 'freeze':
                tribe.freeze_member(target)
            elif rec_type == 'overlap':
                id_a, id_b = target
                ma, mb = tribe.members[id_a], tribe.members[id_b]
                if ma.is_trainable and mb.is_trainable:
                    # Bond the overlapping pair
                    child = tribe.bond(id_a, id_b, seed=gen * 100)
                    shared = []
                    for x, t in ma.domain:
                        la = loss_on(ma.weights, [(x, t)])
                        lb = loss_on(mb.weights, [(x, t)])
                        if la < 0.5 and lb < 0.5:
                            shared.append((x, t))
                    child.domain = shared if shared else ma.domain[:1]
                    train(child.weights, child.domain, steps=200)

                    # Parents shed shared knowledge
                    child_ids = {pattern_id(x, t) for x, t in child.domain}
                    for pid in [id_a, id_b]:
                        p = tribe.members[pid]
                        if p.is_trainable:
                            unique = patterns_match(p.domain, child_ids)
                            if unique:
                                train(p.weights, unique, steps=100, lr=0.015)
                                p.domain = unique

        # Recycle: find members with no unique value
        for m in list(tribe.active_members()):
            unique = tribe.unique_knowledge(m)
            if len(unique) == 0 and len(m.domain) > 0:
                probes = [x for x, t in m.domain]
                tribe.recycle(m.id, probe_inputs=probes, distill_steps=60)

        # Metrics
        revealed = all_clusters[:min(gen + 1, len(all_clusters))]
        all_pats = [p for c in revealed for p in c]
        prec = measure_knowledge_precision(tribe, revealed)
        sys_l = measure_system_loss(tribe, all_pats) if all_pats else 0
        red = measure_redundancy(tribe)
        n_active = len(tribe.active_members())
        n_frozen = len(tribe.frozen_members())
        n_total = len([m for m in tribe.members.values() if m.state != State.RECYCLED])

        gen_data.append((gen, prec, sys_l, red, n_active, n_frozen, n_total))

        print(f"  Gen {gen}: prec={prec:6.2f}  sys_loss={sys_l:.4f}  "
              f"redundancy={red:.2f}  members={n_active}A+{n_frozen}F={n_total}")

    # Summary
    print(f"\n  {'Gen':>3} {'Precision':>10} {'Sys Loss':>10} {'Red.':>6} {'Active':>7} {'Total':>6}")
    for gen, prec, sl, red, na, nf, nt in gen_data:
        print(f"  {gen:>3} {prec:>10.2f} {sl:>10.4f} {red:>6.2f} {na:>7} {nt:>6}")

    first_prec = gen_data[0][1]
    last_prec = gen_data[-1][1]
    first_loss = gen_data[0][2]
    last_loss = gen_data[-1][2]

    print(f"\n  Precision: {first_prec:.2f} → {last_prec:.2f} "
          f"({'+' if last_prec > first_prec else ''}{(last_prec/first_prec - 1)*100:.0f}%)")
    print(f"  Sys loss:  {first_loss:.4f} → {last_loss:.4f} "
          f"({(last_loss/first_loss - 1)*100:+.0f}%)")
    print()
    print("  Events:")
    for gen, event in tribe.history:
        print(f"    Gen {gen}: {event}")

    return gen_data


# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    test_reincarnation_with_memory()
    test_knowledge_consolidation()
    test_knowledge_settling()
