"""Tests for the tribe/ package.

Ported from the monolith tribe.py — validates that the batch-refactored
package produces the same directional results:
  1. Reincarnation with memory beats random init
  2. Knowledge consolidates over generations
  3. Knowledge settles (precision improves) over 6 generations
"""

import mlx.core as mx
import numpy as np

from tribe import (
    State, Tribe, TribeMember,
    make_expert, forward, forward_batch, loss_on, train, clone, blend_weights,
    distill,
    pattern_id, patterns_match, make_clustered_patterns,
    measure_knowledge_precision, measure_system_loss, measure_redundancy,
)
from tribe.expert import DIM


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
        center_0, transform_0 = transforms[0]
        center_1, transform_1 = transforms[1]
        between_pats = []
        for j in range(4):
            alpha = rng.uniform(0.3, 0.7)
            x_np = (alpha * center_0 + (1 - alpha) * center_1
                    + rng.randn(DIM).astype(np.float32) * 0.2)
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

    assert adv_arr.mean() > 0, (
        f"Reincarnation should beat random init on average, got {adv_arr.mean():+.1f}%"
    )
    print(f"\n  CONFIRMED: Reincarnation with memory beats random init")
    return adv_arr


def test_knowledge_consolidation():
    """Multi-generation test: knowledge migrates to correct locations."""
    print("\n" + "=" * 60)
    print("  KNOWLEDGE CONSOLIDATION OVER GENERATIONS")
    print("=" * 60)

    clusters, transforms = make_clustered_patterns(4, 4, seed=42)
    all_patterns = [p for c in clusters for p in c]

    gen_metrics = []

    # ── Generation 0: Initial tribe with 4 random experts ────
    tribe = Tribe()
    for i in range(4):
        m = tribe.add_member(make_expert(seed=i * 100))
        if i > 0:
            tribe.connect(m.id, m.id - 1)
    tribe.connect(0, 3)

    rng = np.random.RandomState(0)
    shuffled_idx = rng.permutation(len(all_patterns))
    active = tribe.active_members()
    for i, idx in enumerate(shuffled_idx):
        active[i % len(active)].domain.append(all_patterns[idx])

    print(f"\n  GENERATION 0 — Initial random assignment")
    for m in active:
        final_loss = train(m.weights, m.domain, steps=400)
        print(f"    Member {m.id}: {len(m.domain)} patterns, loss={final_loss:.4f}")

    precision_0 = measure_knowledge_precision(tribe, clusters)
    sys_loss_0 = measure_system_loss(tribe, all_patterns)
    redundancy_0 = measure_redundancy(tribe)
    gen_metrics.append((0, precision_0, sys_loss_0, redundancy_0))
    print(f"  Metrics: precision={precision_0:.2f}, sys_loss={sys_loss_0:.4f}, "
          f"redundancy={redundancy_0:.2f}")

    # ── Generation 1: Detect overlap, bond, specialize ───────
    tribe.generation = 1
    print(f"\n  GENERATION 1 — Detect overlap, bond where needed")

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
        child = tribe.bond(best_pair[0], best_pair[1], seed=42)
        p1 = tribe.members[best_pair[0]]
        p2 = tribe.members[best_pair[1]]
        shared = []
        for x, t in p1.domain:
            l1 = loss_on(p1.weights, [(x, t)])
            l2 = loss_on(p2.weights, [(x, t)])
            if l1 < 0.5 and l2 < 0.5:
                shared.append((x, t))
        child.domain = shared if shared else p1.domain[:2]
        train(child.weights, child.domain, steps=300)

        child_ids = {pattern_id(x, t) for x, t in child.domain}
        for pid in best_pair:
            parent = tribe.members[pid]
            unique = patterns_match(parent.domain, child_ids)
            if unique:
                train(parent.weights, unique, steps=150, lr=0.015)
                parent.domain = unique

    for m in tribe.active_members():
        if m.domain:
            def loss_fn(w):
                X = mx.stack([x for x, _ in m.domain])
                T = mx.stack([t for _, t in m.domain])
                preds = forward_batch(w, X)
                return mx.mean((preds - T) ** 2)
            _, grads = mx.value_and_grad(loss_fn)(m.weights)
            grad_norm = sum(mx.sum(grads[k] ** 2).item() for k in grads)
            if grad_norm < 0.02 and m.age > 200:
                tribe.freeze_member(m.id)

    precision_1 = measure_knowledge_precision(tribe, clusters)
    sys_loss_1 = measure_system_loss(tribe, all_patterns)
    redundancy_1 = measure_redundancy(tribe)
    gen_metrics.append((1, precision_1, sys_loss_1, redundancy_1))

    # ── Generation 2: New patterns arrive, recycle underperformers ─
    tribe.generation = 2
    print(f"\n  GENERATION 2 — New patterns, recycle + reincarnate")

    new_clusters, _ = make_clustered_patterns(2, 3, seed=99)
    new_patterns = [p for c in new_clusters for p in c]
    clusters_all = clusters + new_clusters
    all_patterns_extended = all_patterns + new_patterns

    active = tribe.active_members()
    if active:
        worst = None
        worst_unique = float('inf')
        for m in active:
            unique = tribe.unique_knowledge(m)
            if len(unique) < worst_unique:
                worst_unique = len(unique)
                worst = m

        if worst and worst_unique <= 1:
            print(f"    Recycling member {worst.id} (only {worst_unique} unique patterns)")
            probe_inputs = [x for x, t in worst.domain] if worst.domain else []
            new_m = tribe.recycle(worst.id, probe_inputs=probe_inputs, distill_steps=80)
            new_m.domain = new_patterns
            train(new_m.weights, new_m.domain, steps=300)
        else:
            for m in active:
                for x, t in new_patterns:
                    if (x, t) not in m.domain:
                        m.domain.append((x, t))
                train(m.weights, m.domain, steps=200)

    precision_2 = measure_knowledge_precision(tribe, clusters_all)
    sys_loss_2 = measure_system_loss(tribe, all_patterns_extended)
    redundancy_2 = measure_redundancy(tribe)
    gen_metrics.append((2, precision_2, sys_loss_2, redundancy_2))

    # ── Generation 3: Another wave, more consolidation ───────
    tribe.generation = 3
    print(f"\n  GENERATION 3 — Further consolidation")

    reinforce_clusters, _ = make_clustered_patterns(4, 2, seed=200)
    reinforce_patterns = [p for c in reinforce_clusters for p in c]

    for x, t in reinforce_patterns:
        scored = tribe.route_by_loss(x, t, top_k=1)
        if scored:
            best = scored[0][0]
            if best.is_trainable:
                existing_ids = {pattern_id(a, b) for a, b in best.domain}
                if pattern_id(x, t) not in existing_ids:
                    best.domain.append((x, t))

    for m in tribe.active_members():
        if m.domain:
            train(m.weights, m.domain, steps=200)

    for m in tribe.frozen_members():
        unique = tribe.unique_knowledge(m)
        if len(unique) == 0:
            probe_inputs = [x for x, t in m.domain] if m.domain else []
            tribe.recycle(m.id, probe_inputs=probe_inputs)

    all_patterns_final = all_patterns_extended + reinforce_patterns
    precision_3 = measure_knowledge_precision(tribe, clusters_all)
    sys_loss_3 = measure_system_loss(tribe, all_patterns_final)
    redundancy_3 = measure_redundancy(tribe)
    gen_metrics.append((3, precision_3, sys_loss_3, redundancy_3))

    # ── Summary ──
    print(f"\n  {'Gen':>3} {'Precision':>10} {'Sys Loss':>10} {'Redundancy':>11}")
    for gen, prec, sl, red in gen_metrics:
        print(f"  {gen:>3} {prec:>10.2f} {sl:>10.4f} {red:>11.2f}")

    assert precision_3 > precision_0, (
        f"Precision should improve: {precision_0:.2f} → {precision_3:.2f}"
    )
    print(f"\n  KNOWLEDGE CONSOLIDATED: precision {precision_0:.2f} → {precision_3:.2f}")

    if sys_loss_3 < sys_loss_0:
        print(f"  SYSTEM IMPROVED: loss {sys_loss_0:.4f} → {sys_loss_3:.4f}")
    if redundancy_3 < redundancy_0:
        print(f"  REDUNDANCY REDUCED: {redundancy_0:.2f} → {redundancy_3:.2f}")

    return gen_metrics


def test_knowledge_settling():
    """Track knowledge placement over 6 generations.

    Each generation: new patterns arrive, tribe adapts via lifecycle.
    Show that knowledge migrates to the right experts over time.
    """
    print("\n" + "=" * 60)
    print("  KNOWLEDGE SETTLING OVER GENERATIONS")
    print("=" * 60)

    all_clusters, transforms = make_clustered_patterns(6, 3, seed=777)

    tribe = Tribe()
    for i in range(4):
        m = tribe.add_member(make_expert(seed=i * 77))
    for i in range(4):
        tribe.connect(i, (i + 1) % 4)
    tribe.connect(0, 2)
    tribe.connect(1, 3)

    gen_data = []

    for gen in range(6):
        tribe.generation = gen

        if gen < len(all_clusters):
            new_pats = all_clusters[gen]
        else:
            rng_g = np.random.RandomState(gen * 51)
            src = gen % len(all_clusters)
            center, tfm = transforms[src]
            new_pats = []
            for _ in range(3):
                x = center + rng_g.randn(DIM).astype(np.float32) * 0.3
                t = np.tanh(tfm @ x).astype(np.float32)
                new_pats.append((mx.array(x), mx.array(t)))

        active = tribe.active_members()
        if not active:
            frozen = tribe.frozen_members()
            if frozen:
                worst_frozen = min(frozen, key=lambda m: len(m.domain))
                probes = [x for x, t in worst_frozen.domain] if worst_frozen.domain else []
                tribe.recycle(worst_frozen.id, probe_inputs=probes, distill_steps=60)
                active = tribe.active_members()
            if not active:
                break

        for x, t in new_pats:
            scored = [(m, loss_on(m.weights, [(x, t)])) for m in active]
            scored.sort(key=lambda s: s[1])
            top = scored[:min(2, len(scored))]
            for m, _ in top:
                existing = {pattern_id(a, b) for a, b in m.domain}
                if pattern_id(x, t) not in existing:
                    m.domain.append((x, t))

        for m in active:
            if m.domain:
                train(m.weights, m.domain, steps=250)

        recs = tribe.health_check(overlap_threshold=0.3, freeze_grad_threshold=1e-5)

        for rec_type, target, reason in recs:
            if rec_type == 'freeze':
                tribe.freeze_member(target)
            elif rec_type == 'overlap':
                id_a, id_b = target
                ma, mb = tribe.members[id_a], tribe.members[id_b]
                if ma.is_trainable and mb.is_trainable:
                    child = tribe.bond(id_a, id_b, seed=gen * 100)
                    shared = []
                    for x, t in ma.domain:
                        la = loss_on(ma.weights, [(x, t)])
                        lb = loss_on(mb.weights, [(x, t)])
                        if la < 0.5 and lb < 0.5:
                            shared.append((x, t))
                    child.domain = shared if shared else ma.domain[:1]
                    train(child.weights, child.domain, steps=200)

                    child_ids = {pattern_id(x, t) for x, t in child.domain}
                    for pid in [id_a, id_b]:
                        p = tribe.members[pid]
                        if p.is_trainable:
                            unique = patterns_match(p.domain, child_ids)
                            if unique:
                                train(p.weights, unique, steps=100, lr=0.015)
                                p.domain = unique

        for m in list(tribe.active_members()):
            unique = tribe.unique_knowledge(m)
            if len(unique) == 0 and len(m.domain) > 0:
                probes = [x for x, t in m.domain]
                tribe.recycle(m.id, probe_inputs=probes, distill_steps=60)

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

    print(f"\n  {'Gen':>3} {'Precision':>10} {'Sys Loss':>10} {'Red.':>6} {'Active':>7} {'Total':>6}")
    for gen, prec, sl, red, na, nf, nt in gen_data:
        print(f"  {gen:>3} {prec:>10.2f} {sl:>10.4f} {red:>6.2f} {na:>7} {nt:>6}")

    first_prec = gen_data[0][1]
    last_prec = gen_data[-1][1]

    assert last_prec > first_prec, (
        f"Precision should improve: {first_prec:.2f} → {last_prec:.2f}"
    )
    print(f"\n  Precision: {first_prec:.2f} → {last_prec:.2f} "
          f"({(last_prec/first_prec - 1)*100:+.0f}%)")

    return gen_data


if __name__ == '__main__':
    test_reincarnation_with_memory()
    test_knowledge_consolidation()
    test_knowledge_settling()
