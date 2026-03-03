"""Baseline comparison: prove the tribe lifecycle adds value.

Same data schedule for all:
  Gen 0: digits 0-2, 100/class  (300 patterns)
  Gen 1: digits 3-5, 100/class  (300 patterns)
  Gen 2: digits 6-9, 100/class  (400 patterns)
  Gen 3: all digits, 20/class   (200 patterns, reinforcement)

Baselines:
  A — Single CNN (catastrophic forgetting control)
  B — Static tribe (ensemble, loss-based routing top_k=1, no lifecycle)
  C — Round-robin tribe (ensemble, round-robin routing, no lifecycle)
  D — Lifecycle tribe (loss-based top_k=2 routing → creates overlap → lifecycle)
  E — Round-robin + lifecycle (messy start → lifecycle cleans up)
  F — Learned router + lifecycle (Switch-style gating, research-comparable)

Key comparisons:
  C vs E — same round-robin assignment, only difference is lifecycle ops
  B vs D — loss-based routing, D uses top_k=2 to create overlap for lifecycle
  E vs F — lifecycle with random vs learned routing
"""

import mlx.core as mx
import numpy as np

from tribe import (
    Tribe, State, SwitchRouter,
    make_cnn_expert, cnn_forward_batch,
    load_mnist, make_mnist_patterns,
    loss_on, train, clone,
    pattern_id, patterns_match,
    measure_knowledge_precision, measure_system_loss, measure_redundancy,
)
from test_tribe_mnist import batch_route_by_loss, measure_accuracy


# ── Data schedule ─────────────────────────────────────────────
def make_data_schedule(train_imgs, train_labels):
    """Create the 4-generation data schedule."""
    return [
        ([0, 1, 2], make_mnist_patterns(train_imgs, train_labels, [0, 1, 2],
                                         n_per_class=100, seed=0)),
        ([3, 4, 5], make_mnist_patterns(train_imgs, train_labels, [3, 4, 5],
                                         n_per_class=100, seed=10)),
        ([6, 7, 8, 9], make_mnist_patterns(train_imgs, train_labels, [6, 7, 8, 9],
                                            n_per_class=100, seed=20)),
        (list(range(10)), make_mnist_patterns(train_imgs, train_labels, list(range(10)),
                                              n_per_class=20, seed=30)),
    ]


def make_tribe_4(seed_base=42):
    """Create a 4-member ring tribe with CNN experts."""
    tribe = Tribe(fwd=cnn_forward_batch, make_expert_fn=make_cnn_expert)
    for i in range(4):
        tribe.add_member(make_cnn_expert(seed=i * seed_base))
    for i in range(4):
        tribe.connect(i, (i + 1) % 4)
    return tribe


def eval_metrics(tribe_or_weights, test_imgs, test_labels, all_patterns,
                 cluster_list, is_single=False):
    """Compute accuracy, sys_loss, precision, redundancy for any baseline."""
    if is_single:
        t = Tribe(fwd=cnn_forward_batch)
        m = t.add_member(clone(tribe_or_weights))
        m.domain = all_patterns
        per_class, overall = measure_accuracy(t, test_imgs, test_labels)
        sys_l = measure_system_loss(t, all_patterns)
        prec = 1.0
        red = 0.0
    else:
        tribe = tribe_or_weights
        per_class, overall = measure_accuracy(tribe, test_imgs, test_labels)
        sys_l = measure_system_loss(tribe, all_patterns)
        prec = measure_knowledge_precision(tribe, cluster_list)
        red = measure_redundancy(tribe)
    return per_class, overall, sys_l, prec, red


def assign_patterns(tribe, patterns, existing_domain):
    """Add patterns to a member's domain, deduplicating."""
    existing = {pattern_id(a, b) for a, b in existing_domain}
    for p in patterns:
        pid = pattern_id(*p)
        if pid not in existing:
            existing_domain.append(p)
            existing.add(pid)


def find_shared_patterns(tribe, ma, mb):
    """Find patterns shared between two experts. Top-N by combined loss as fallback."""
    shared = []
    combined = []
    for xi, ti in ma.domain:
        la = loss_on(ma.weights, [(xi, ti)], fwd=cnn_forward_batch)
        lb = loss_on(mb.weights, [(xi, ti)], fwd=cnn_forward_batch)
        combined.append((la + lb, (xi, ti)))
        if la < 0.1 and lb < 0.1:
            shared.append((xi, ti))
    if not shared:
        combined.sort(key=lambda pair: pair[0])
        shared = [p for _, p in combined[:max(10, len(combined) // 4)]]
    return shared


def apply_lifecycle(tribe, gen, patterns, events):
    """Run health check, bond overlapping pairs, recycle underperformers.

    Mutates tribe in-place, appends event strings to events list.
    """
    # Health check
    recs = tribe.health_check(overlap_threshold=0.3,
                              freeze_grad_threshold=1e-5,
                              competence_threshold=0.05,
                              min_active=3)
    for rec_type, target, reason in recs:
        if rec_type == 'overlap':
            id_a, id_b = target
            ma, mb = tribe.members[id_a], tribe.members[id_b]
            if ma.is_trainable and mb.is_trainable:
                child = tribe.bond(id_a, id_b, seed=gen * 100)
                child.domain = find_shared_patterns(tribe, ma, mb)
                train(child.weights, child.domain, steps=150, lr=0.01,
                      fwd=cnn_forward_batch)

                # Parents shed shared knowledge
                child_ids = {pattern_id(xi, ti) for xi, ti in child.domain}
                for pid in [id_a, id_b]:
                    parent = tribe.members[pid]
                    if parent.is_trainable:
                        unique = patterns_match(parent.domain, child_ids)
                        if unique:
                            train(parent.weights, unique, steps=100, lr=0.01,
                                  fwd=cnn_forward_batch)
                            parent.domain = unique

                events.append(f"BOND {id_a}+{id_b}→{child.id}")
        elif rec_type == 'freeze':
            tribe.freeze_member(target)
            events.append(f"FREEZE {target}")

    # Recycle underperformers (proportional threshold)
    active = tribe.active_members()
    if len(active) > 2:
        worst = None
        worst_unique_count = float('inf')
        for m in active:
            unique = tribe.unique_knowledge(m, margin=0.95)
            if len(unique) < worst_unique_count:
                worst_unique_count = len(unique)
                worst = m
        avg_domain = sum(len(m.domain) for m in active) / len(active)
        recycle_threshold = max(5, int(0.05 * avg_domain))
        if worst and worst_unique_count <= recycle_threshold:
            probes = [xi for xi, ti in worst.domain] if worst.domain else []
            new_m = tribe.recycle(worst.id, probe_inputs=probes, distill_steps=50)
            new_m.domain = patterns[:min(100, len(patterns))]
            train(new_m.weights, new_m.domain, steps=200, lr=0.01,
                  fwd=cnn_forward_batch)
            events.append(f"RECYCLE {worst.id}")


def format_tribe_state(tribe):
    """Format tribe state for printing."""
    active_str = [f"{m.id}:{len(m.domain)}" for m in tribe.active_members()]
    frozen_str = [f"{m.id}F" for m in tribe.frozen_members()]
    parts = [f"active=[{', '.join(active_str)}]"]
    if frozen_str:
        parts.append(f"frozen=[{', '.join(frozen_str)}]")
    return ', '.join(parts)


# ── Baseline A: Single CNN ────────────────────────────────────
def run_baseline_a(train_imgs, train_labels, test_imgs, test_labels, schedule):
    """Single CNN trained sequentially — catastrophic forgetting control."""
    print("\n  BASELINE A — Single CNN (catastrophic forgetting control)")
    print("  " + "-" * 55)

    weights = make_cnn_expert(seed=0)
    all_patterns = []
    seen_digits = set()
    gen_metrics = []

    for gen, (digits, patterns) in enumerate(schedule):
        seen_digits.update(digits)
        all_patterns = all_patterns + patterns

        final_loss = train(weights, patterns, steps=200, lr=0.01,
                           fwd=cnn_forward_batch)
        print(f"    Gen {gen}: digits {digits}, train loss={final_loss:.4f}")

        cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                        for d in sorted(seen_digits)]
        per_class, overall, sys_l, prec, red = eval_metrics(
            weights, test_imgs, test_labels, all_patterns,
            cluster_list, is_single=True)

        gen_metrics.append({
            'gen': gen, 'accuracy': overall, 'sys_loss': sys_l,
            'precision': prec, 'redundancy': red, 'per_class': per_class,
            'events': [],
        })

    return gen_metrics


# ── Baseline B: Static tribe ──────────────────────────────────
def run_baseline_b(train_imgs, train_labels, test_imgs, test_labels, schedule):
    """Static tribe: 4 CNNs, loss-based routing top_k=1, NO lifecycle."""
    print("\n  BASELINE B — Static tribe (loss-routing, no lifecycle)")
    print("  " + "-" * 55)

    tribe = make_tribe_4()
    all_patterns = []
    seen_digits = set()
    gen_metrics = []

    for gen, (digits, patterns) in enumerate(schedule):
        tribe.generation = gen
        seen_digits.update(digits)
        all_patterns = all_patterns + patterns

        assignments = batch_route_by_loss(tribe, patterns, top_k=1)
        for mid, pats in assignments.items():
            assign_patterns(tribe, pats, tribe.members[mid].domain)

        for m in tribe.active_members():
            if m.domain:
                train(m.weights, m.domain, steps=200, lr=0.01,
                      fwd=cnn_forward_batch)

        print(f"    Gen {gen}: digits {digits}, "
              f"domains={[len(m.domain) for m in tribe.active_members()]}")

        cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                        for d in sorted(seen_digits)]
        per_class, overall, sys_l, prec, red = eval_metrics(
            tribe, test_imgs, test_labels, all_patterns, cluster_list)

        gen_metrics.append({
            'gen': gen, 'accuracy': overall, 'sys_loss': sys_l,
            'precision': prec, 'redundancy': red, 'per_class': per_class,
            'events': [],
        })

    return gen_metrics


# ── Baseline C: Round-robin tribe ─────────────────────────────
def run_baseline_c(train_imgs, train_labels, test_imgs, test_labels, schedule):
    """Round-robin tribe: 4 CNNs, round-robin routing, no lifecycle."""
    print("\n  BASELINE C — Round-robin tribe (no lifecycle)")
    print("  " + "-" * 55)

    tribe = make_tribe_4()
    all_patterns = []
    seen_digits = set()
    gen_metrics = []

    for gen, (digits, patterns) in enumerate(schedule):
        tribe.generation = gen
        seen_digits.update(digits)
        all_patterns = all_patterns + patterns

        active = tribe.active_members()
        for i, p in enumerate(patterns):
            m = active[i % len(active)]
            assign_patterns(tribe, [p], m.domain)

        for m in active:
            if m.domain:
                train(m.weights, m.domain, steps=200, lr=0.01,
                      fwd=cnn_forward_batch)

        print(f"    Gen {gen}: digits {digits}, "
              f"domains={[len(m.domain) for m in active]}")

        cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                        for d in sorted(seen_digits)]
        per_class, overall, sys_l, prec, red = eval_metrics(
            tribe, test_imgs, test_labels, all_patterns, cluster_list)

        gen_metrics.append({
            'gen': gen, 'accuracy': overall, 'sys_loss': sys_l,
            'precision': prec, 'redundancy': red, 'per_class': per_class,
            'events': [],
        })

    return gen_metrics


# ── Experiment D: Lifecycle tribe (top_k=2) ───────────────────
def run_experiment_d(train_imgs, train_labels, test_imgs, test_labels, schedule):
    """Lifecycle tribe: loss-based top_k=2 routing creates overlap → lifecycle cleans up."""
    print("\n  EXPERIMENT D — Lifecycle tribe (loss-routing top_k=2)")
    print("  " + "-" * 55)

    tribe = make_tribe_4()
    all_patterns = []
    seen_digits = set()
    gen_metrics = []

    for gen, (digits, patterns) in enumerate(schedule):
        tribe.generation = gen
        seen_digits.update(digits)
        all_patterns = all_patterns + patterns

        # top_k=2: each pattern goes to 2 experts → creates overlap
        assignments = batch_route_by_loss(tribe, patterns, top_k=2)
        for mid, pats in assignments.items():
            assign_patterns(tribe, pats, tribe.members[mid].domain)

        for m in tribe.active_members():
            if m.domain:
                train(m.weights, m.domain, steps=200, lr=0.01,
                      fwd=cnn_forward_batch)

        events = []
        if gen > 0:
            apply_lifecycle(tribe, gen, patterns, events)

        print(f"    Gen {gen}: digits {digits}, "
              f"{format_tribe_state(tribe)}, events={events}")

        cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                        for d in sorted(seen_digits)]
        per_class, overall, sys_l, prec, red = eval_metrics(
            tribe, test_imgs, test_labels, all_patterns, cluster_list)

        gen_metrics.append({
            'gen': gen, 'accuracy': overall, 'sys_loss': sys_l,
            'precision': prec, 'redundancy': red, 'per_class': per_class,
            'events': events,
        })

    return gen_metrics


# ── Experiment E: Round-robin + lifecycle ─────────────────────
def run_experiment_e(train_imgs, train_labels, test_imgs, test_labels, schedule):
    """Round-robin assignment (creates mess) → lifecycle cleans up.

    Same assignment as C, but with lifecycle ops after each gen.
    C vs E isolates the lifecycle's effect.
    """
    print("\n  EXPERIMENT E — Round-robin + lifecycle (messy start → cleanup)")
    print("  " + "-" * 55)

    tribe = make_tribe_4()
    all_patterns = []
    seen_digits = set()
    gen_metrics = []

    for gen, (digits, patterns) in enumerate(schedule):
        tribe.generation = gen
        seen_digits.update(digits)
        all_patterns = all_patterns + patterns

        # Round-robin assignment (identical to C)
        active = tribe.active_members()
        for i, p in enumerate(patterns):
            m = active[i % len(active)]
            assign_patterns(tribe, [p], m.domain)

        for m in tribe.active_members():
            if m.domain:
                train(m.weights, m.domain, steps=200, lr=0.01,
                      fwd=cnn_forward_batch)

        events = []
        if gen > 0:
            apply_lifecycle(tribe, gen, patterns, events)

        print(f"    Gen {gen}: digits {digits}, "
              f"{format_tribe_state(tribe)}, events={events}")

        cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                        for d in sorted(seen_digits)]
        per_class, overall, sys_l, prec, red = eval_metrics(
            tribe, test_imgs, test_labels, all_patterns, cluster_list)

        gen_metrics.append({
            'gen': gen, 'accuracy': overall, 'sys_loss': sys_l,
            'precision': prec, 'redundancy': red, 'per_class': per_class,
            'events': events,
        })

    return gen_metrics


# ── Baseline F: Learned Router + Lifecycle ─────────────────────
def run_baseline_f(train_imgs, train_labels, test_imgs, test_labels, schedule):
    """Learned Switch-style router + lifecycle: comparable to published MoE work."""
    print("\n  BASELINE F — Learned router + lifecycle (Switch-style)")
    print("  " + "-" * 55)

    tribe = make_tribe_4()
    router = SwitchRouter(input_dim=784, num_experts=4, top_k=1)
    tribe.set_router(router)

    all_patterns = []
    seen_digits = set()
    gen_metrics = []

    for gen, (digits, patterns) in enumerate(schedule):
        tribe.generation = gen
        seen_digits.update(digits)
        all_patterns = all_patterns + patterns

        # Flatten images for the router: (28,28,1) → (784,)
        X_flat = mx.stack([mx.reshape(x, (-1,)) for x, _ in patterns])

        # Route via learned router
        assignments_idx, aux_loss, stats = router.route(X_flat)

        # Convert index-based assignments to pattern-based
        active = tribe.active_members()
        active_ids = sorted([m.id for m in active])
        for eid, indices in assignments_idx.items():
            if eid >= len(active_ids):
                continue
            mid = active_ids[eid]
            pats = [patterns[i] for i in indices]
            assign_patterns(tribe, pats, tribe.members[mid].domain)

        # Train each expert on its assigned domain
        for m in tribe.active_members():
            if m.domain:
                train(m.weights, m.domain, steps=200, lr=0.01,
                      fwd=cnn_forward_batch)

        # Update router weights (minimize load imbalance)
        router.train_step(X_flat, None, None, None, None, lr=0.01, alpha=0.01)

        # Lifecycle ops after gen 0
        events = []
        if gen > 0:
            apply_lifecycle(tribe, gen, patterns, events)

        print(f"    Gen {gen}: digits {digits}, "
              f"{format_tribe_state(tribe)}, "
              f"router_counts={stats['counts']}, aux={aux_loss.item():.4f}, "
              f"events={events}")

        cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                        for d in sorted(seen_digits)]
        per_class, overall, sys_l, prec, red = eval_metrics(
            tribe, test_imgs, test_labels, all_patterns, cluster_list)

        gen_metrics.append({
            'gen': gen, 'accuracy': overall, 'sys_loss': sys_l,
            'precision': prec, 'redundancy': red, 'per_class': per_class,
            'events': events,
        })

    return gen_metrics


# ── Comparison ────────────────────────────────────────────────
def print_comparison(results, names):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 90)
    print("  BASELINE COMPARISON")
    print("=" * 90)

    # Per-generation table
    print(f"\n  {'Gen':>3}", end="")
    for name in names:
        print(f"  {name:>17s}", end="")
    print()
    print(f"  {'':>3}", end="")
    for _ in names:
        print(f"  {'Acc':>8s} {'Loss':>7s}", end="")
    print()
    print("  " + "-" * (3 + 17 * len(names)))

    for gen in range(4):
        print(f"  {gen:>3}", end="")
        for metrics in results:
            m = metrics[gen]
            print(f"  {m['accuracy']:>8.1%} {m['sys_loss']:>7.4f}", end="")
        print()

    # Backward transfer
    print(f"\n  Backward Transfer (gen-0 digit accuracy at gen 3):")
    for name, metrics in zip(names, results):
        final = metrics[3]['per_class']
        early_accs = [final.get(d, 0) for d in [0, 1, 2]]
        avg_early = np.mean(early_accs) if early_accs else 0
        print(f"    {name:22s}: digits 0-2 avg = {avg_early:.1%}  "
              f"({', '.join(f'{d}:{final.get(d,0):.0%}' for d in [0,1,2])})")

    # Precision and redundancy
    print(f"\n  Precision & Redundancy at gen 3:")
    for name, metrics in zip(names, results):
        m = metrics[3]
        print(f"    {name:22s}: precision={m['precision']:>6.2f}  "
              f"redundancy={m['redundancy']:.2f}")

    # Precision trends for lifecycle experiments
    lifecycle_names = [n for n in names if n.startswith(('D:', 'E:', 'F:'))]
    lifecycle_results = [r for n, r in zip(names, results) if n.startswith(('D:', 'E:', 'F:'))]
    if lifecycle_results:
        print(f"\n  Lifecycle Precision Trends:")
        for name, metrics in zip(lifecycle_names, lifecycle_results):
            print(f"    {name}:")
            for m in metrics:
                events_str = ', '.join(m['events']) if m['events'] else 'none'
                print(f"      Gen {m['gen']}: prec={m['precision']:>6.2f}  "
                      f"red={m['redundancy']:.2f}  events: {events_str}")

    # Key comparisons
    print(f"\n  KEY COMPARISONS:")

    # C vs E (same assignment, lifecycle vs not)
    c_idx = next((i for i, n in enumerate(names) if n.startswith('C:')), None)
    e_idx = next((i for i, n in enumerate(names) if n.startswith('E:')), None)
    if c_idx is not None and e_idx is not None:
        c3, e3 = results[c_idx][3], results[e_idx][3]
        print(f"    C vs E (round-robin ± lifecycle):")
        print(f"      Accuracy:   C={c3['accuracy']:.1%}  E={e3['accuracy']:.1%}  "
              f"({'E wins' if e3['accuracy'] > c3['accuracy'] else 'C wins'})")
        print(f"      Precision:  C={c3['precision']:.2f}  E={e3['precision']:.2f}  "
              f"({'E wins' if e3['precision'] > c3['precision'] else 'C wins'})")
        print(f"      Redundancy: C={c3['redundancy']:.2f}  E={e3['redundancy']:.2f}  "
              f"({'E wins' if e3['redundancy'] < c3['redundancy'] else 'C wins'})")


def run_baselines():
    print("\n" + "=" * 90)
    print("  TRIBE LIFECYCLE BASELINE COMPARISON")
    print("=" * 90)

    train_imgs, train_labels, test_imgs, test_labels = load_mnist()
    print(f"  MNIST loaded: {len(train_imgs)} train, {len(test_imgs)} test")

    schedule = make_data_schedule(train_imgs, train_labels)
    for gen, (digits, pats) in enumerate(schedule):
        print(f"  Gen {gen}: digits {digits} ({len(pats)} patterns)")

    results_a = run_baseline_a(train_imgs, train_labels, test_imgs, test_labels, schedule)
    results_b = run_baseline_b(train_imgs, train_labels, test_imgs, test_labels, schedule)
    results_c = run_baseline_c(train_imgs, train_labels, test_imgs, test_labels, schedule)
    results_d = run_experiment_d(train_imgs, train_labels, test_imgs, test_labels, schedule)
    results_e = run_experiment_e(train_imgs, train_labels, test_imgs, test_labels, schedule)
    results_f = run_baseline_f(train_imgs, train_labels, test_imgs, test_labels, schedule)

    names = ['A: Single CNN', 'B: Static tribe', 'C: Round-robin',
             'D: Lifecycle k=2', 'E: RR + lifecycle', 'F: Learned + LC']
    print_comparison(
        [results_a, results_b, results_c, results_d, results_e, results_f],
        names)


if __name__ == '__main__':
    run_baselines()
