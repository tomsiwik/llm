"""MNIST continual learning with CNN tribe experts.

4-generation experiment:
  Gen 0: digits 0-2 (300 patterns) — initial assignment, train
  Gen 1: digits 3-5 (300 patterns) — route, train, detect overlap, bond
  Gen 2: digits 6-9 (400 patterns) — route, train, recycle underperformers
  Gen 3: all digits (200 patterns) — reinforcement, consolidation

4 CNN experts (~64K params each) in a ring graph.
MSE on one-hot targets, consistent with existing tribe system.
"""

import mlx.core as mx
import numpy as np

from tribe import (
    Tribe, State,
    make_cnn_expert, cnn_forward_batch,
    load_mnist, make_mnist_patterns,
    loss_on, train, clone,
    pattern_id, patterns_match,
    measure_knowledge_precision, measure_system_loss, measure_redundancy,
)


def batch_route_by_loss(tribe, patterns, top_k=1):
    """Route all patterns to experts by loss — batched per expert.

    Returns dict: member_id → list of (input, target) assigned patterns.
    """
    if not patterns:
        return {}
    X = mx.stack([x for x, _ in patterns])
    T = mx.stack([t for _, t in patterns])

    routable = tribe.routable_members()
    if not routable:
        return {}

    # One batched forward per expert
    all_losses = []
    for m in routable:
        preds = tribe.fwd(m.weights, X)
        losses = mx.mean((preds - T) ** 2, axis=1)  # (N,)
        all_losses.append(losses)

    loss_matrix = mx.stack(all_losses)  # (M, N)
    mx.eval(loss_matrix)

    # For each pattern, find the top_k experts with lowest loss
    assignments = {m.id: [] for m in routable}
    loss_np = np.array(loss_matrix)  # (M, N)

    for i in range(len(patterns)):
        expert_losses = [(j, loss_np[j, i]) for j in range(len(routable))]
        expert_losses.sort(key=lambda el: el[1])
        assigned = 0
        for j, _ in expert_losses:
            if assigned >= top_k:
                break
            m = routable[j]
            if m.is_trainable:
                assignments[m.id].append(patterns[i])
                assigned += 1

    return assignments


def measure_accuracy(tribe, images, labels):
    """Per-class and overall accuracy using argmax of expert predictions.

    For each image, uses the expert with lowest loss (oracle routing).
    """
    if len(images) == 0:
        return {}, 0.0

    X = mx.stack([mx.array(img.reshape(28, 28, 1)) if len(img.shape) == 2
                  else mx.array(img) for img in images])
    one_hots = []
    for l in labels:
        oh = np.zeros(10, dtype=np.float32)
        oh[l] = 1.0
        one_hots.append(mx.array(oh))
    T = mx.stack(one_hots)

    routable = tribe.routable_members()
    if not routable:
        return {}, 0.0

    # Find best expert per sample
    all_preds = []
    all_losses = []
    for m in routable:
        preds = tribe.fwd(m.weights, X)
        losses = mx.mean((preds - T) ** 2, axis=1)
        all_preds.append(preds)
        all_losses.append(losses)

    loss_matrix = mx.stack(all_losses)  # (M, N)
    best_expert_idx = mx.argmin(loss_matrix, axis=0)  # (N,)
    mx.eval(best_expert_idx, *all_preds)

    best_expert_np = np.array(best_expert_idx)
    preds_list = [np.array(p) for p in all_preds]
    labels_np = np.array(labels)

    # Gather predictions from best expert for each sample
    N = len(images)
    predicted = np.zeros(N, dtype=np.int32)
    for i in range(N):
        pred = preds_list[best_expert_np[i]][i]
        predicted[i] = np.argmax(pred)

    # Per-class accuracy
    per_class = {}
    for digit in range(10):
        mask = labels_np == digit
        if mask.sum() > 0:
            per_class[digit] = (predicted[mask] == digit).mean()

    overall = (predicted == labels_np).mean()
    return per_class, overall


def run_mnist_experiment():
    print("\n" + "=" * 70)
    print("  MNIST CONTINUAL LEARNING WITH CNN TRIBE EXPERTS")
    print("=" * 70)

    # Load MNIST
    train_imgs, train_labels, test_imgs, test_labels = load_mnist()
    print(f"  MNIST loaded: {len(train_imgs)} train, {len(test_imgs)} test")

    # Create tribe with CNN experts
    tribe = Tribe(fwd=cnn_forward_batch, make_expert_fn=make_cnn_expert)
    for i in range(4):
        tribe.add_member(make_cnn_expert(seed=i * 42))
    # Ring graph
    for i in range(4):
        tribe.connect(i, (i + 1) % 4)

    gen_metrics = []
    all_seen_digits = set()

    # ── Generation 0: digits 0-2 ──────────────────────────────
    tribe.generation = 0
    digits_0 = [0, 1, 2]
    all_seen_digits.update(digits_0)
    patterns_0 = make_mnist_patterns(train_imgs, train_labels, digits_0,
                                     n_per_class=100, seed=0)
    print(f"\n  GENERATION 0 — Digits {digits_0} ({len(patterns_0)} patterns)")

    # Route and assign
    assignments = batch_route_by_loss(tribe, patterns_0, top_k=1)
    for mid, pats in assignments.items():
        m = tribe.members[mid]
        for p in pats:
            existing = {pattern_id(a, b) for a, b in m.domain}
            if pattern_id(*p) not in existing:
                m.domain.append(p)

    # Train
    for m in tribe.active_members():
        if m.domain:
            final_loss = train(m.weights, m.domain, steps=200, lr=0.01,
                               fwd=cnn_forward_batch)
            print(f"    Member {m.id}: {len(m.domain)} patterns, loss={final_loss:.4f}")

    # Metrics
    per_class, overall = measure_accuracy(tribe, test_imgs, test_labels)
    prec = measure_knowledge_precision(tribe,
                                        [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                                         for d in digits_0])
    sys_l = measure_system_loss(tribe, patterns_0)
    red = measure_redundancy(tribe)
    gen_metrics.append((0, overall, sys_l, prec, red))

    print(f"  Accuracy: overall={overall:.1%}")
    for d in sorted(per_class):
        if d in digits_0:
            print(f"    digit {d}: {per_class[d]:.1%}")
    print(f"  Metrics: precision={prec:.2f}, sys_loss={sys_l:.4f}, redundancy={red:.2f}")

    # ── Generation 1: digits 3-5 ──────────────────────────────
    tribe.generation = 1
    digits_1 = [3, 4, 5]
    all_seen_digits.update(digits_1)
    patterns_1 = make_mnist_patterns(train_imgs, train_labels, digits_1,
                                     n_per_class=100, seed=10)
    print(f"\n  GENERATION 1 — Digits {digits_1} ({len(patterns_1)} patterns)")

    # Route new patterns
    assignments = batch_route_by_loss(tribe, patterns_1, top_k=1)
    for mid, pats in assignments.items():
        m = tribe.members[mid]
        for p in pats:
            existing = {pattern_id(a, b) for a, b in m.domain}
            if pattern_id(*p) not in existing:
                m.domain.append(p)

    # Train
    for m in tribe.active_members():
        if m.domain:
            final_loss = train(m.weights, m.domain, steps=200, lr=0.01,
                               fwd=cnn_forward_batch)
            print(f"    Member {m.id}: {len(m.domain)} patterns, loss={final_loss:.4f}")

    # Health check — detect overlap, bond
    recs = tribe.health_check(overlap_threshold=0.3, freeze_grad_threshold=1e-5,
                              competence_threshold=0.05, min_active=3)
    for rec_type, target, reason in recs:
        if rec_type == 'overlap':
            id_a, id_b = target
            print(f"    Overlap detected: {id_a} ↔ {id_b} ({reason})")
            ma, mb = tribe.members[id_a], tribe.members[id_b]
            if ma.is_trainable and mb.is_trainable:
                child = tribe.bond(id_a, id_b, seed=100)
                # Child gets shared patterns: top-N by combined loss as fallback
                shared = []
                combined = []
                for x, t in ma.domain:
                    la = loss_on(ma.weights, [(x, t)], fwd=cnn_forward_batch)
                    lb = loss_on(mb.weights, [(x, t)], fwd=cnn_forward_batch)
                    combined.append((la + lb, (x, t)))
                    if la < 0.1 and lb < 0.1:
                        shared.append((x, t))
                if not shared:
                    # Fallback: top-N patterns by combined loss
                    combined.sort(key=lambda pair: pair[0])
                    shared = [p for _, p in combined[:max(10, len(combined) // 4)]]
                child.domain = shared
                train(child.weights, child.domain, steps=150, lr=0.01,
                      fwd=cnn_forward_batch)
                print(f"    Bonded {id_a}+{id_b} → child {child.id} ({len(child.domain)} patterns)")
        elif rec_type == 'freeze':
            tribe.freeze_member(target)
            print(f"    Frozen member {target} ({reason})")

    per_class, overall = measure_accuracy(tribe, test_imgs, test_labels)
    all_patterns_so_far = patterns_0 + patterns_1
    cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                    for d in sorted(all_seen_digits)]
    prec = measure_knowledge_precision(tribe, cluster_list)
    sys_l = measure_system_loss(tribe, all_patterns_so_far)
    red = measure_redundancy(tribe)
    gen_metrics.append((1, overall, sys_l, prec, red))

    print(f"  Accuracy: overall={overall:.1%}")
    for d in sorted(per_class):
        if d in set(digits_0) | set(digits_1):
            print(f"    digit {d}: {per_class[d]:.1%}")
    print(f"  Metrics: precision={prec:.2f}, sys_loss={sys_l:.4f}, redundancy={red:.2f}")

    # ── Generation 2: digits 6-9 ──────────────────────────────
    tribe.generation = 2
    digits_2 = [6, 7, 8, 9]
    all_seen_digits.update(digits_2)
    patterns_2 = make_mnist_patterns(train_imgs, train_labels, digits_2,
                                     n_per_class=100, seed=20)
    print(f"\n  GENERATION 2 — Digits {digits_2} ({len(patterns_2)} patterns)")

    # Route new patterns
    assignments = batch_route_by_loss(tribe, patterns_2, top_k=1)
    for mid, pats in assignments.items():
        m = tribe.members[mid]
        for p in pats:
            existing = {pattern_id(a, b) for a, b in m.domain}
            if pattern_id(*p) not in existing:
                m.domain.append(p)

    # Train
    for m in tribe.active_members():
        if m.domain:
            final_loss = train(m.weights, m.domain, steps=200, lr=0.01,
                               fwd=cnn_forward_batch)
            print(f"    Member {m.id}: {len(m.domain)} patterns, loss={final_loss:.4f}")

    # Recycle underperformers — proportional threshold
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
            print(f"    Recycling member {worst.id} ({worst_unique_count} unique "
                  f"patterns, threshold={recycle_threshold})")
            probes = [x for x, t in worst.domain] if worst.domain else []
            new_m = tribe.recycle(worst.id, probe_inputs=probes, distill_steps=50)
            # Assign new patterns to recycled member
            new_m.domain = patterns_2[:100]
            train(new_m.weights, new_m.domain, steps=200, lr=0.01,
                  fwd=cnn_forward_batch)

    per_class, overall = measure_accuracy(tribe, test_imgs, test_labels)
    all_patterns_so_far = patterns_0 + patterns_1 + patterns_2
    cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                    for d in sorted(all_seen_digits)]
    prec = measure_knowledge_precision(tribe, cluster_list)
    sys_l = measure_system_loss(tribe, all_patterns_so_far)
    red = measure_redundancy(tribe)
    gen_metrics.append((2, overall, sys_l, prec, red))

    print(f"  Accuracy: overall={overall:.1%}")
    for d in sorted(per_class):
        print(f"    digit {d}: {per_class[d]:.1%}")
    print(f"  Metrics: precision={prec:.2f}, sys_loss={sys_l:.4f}, redundancy={red:.2f}")

    # ── Generation 3: reinforcement with all digits ───────────
    tribe.generation = 3
    digits_3 = list(range(10))
    patterns_3 = make_mnist_patterns(train_imgs, train_labels, digits_3,
                                     n_per_class=20, seed=30)
    print(f"\n  GENERATION 3 — All digits reinforcement ({len(patterns_3)} patterns)")

    # Route reinforcement patterns
    assignments = batch_route_by_loss(tribe, patterns_3, top_k=1)
    for mid, pats in assignments.items():
        m = tribe.members[mid]
        for p in pats:
            existing = {pattern_id(a, b) for a, b in m.domain}
            if pattern_id(*p) not in existing:
                m.domain.append(p)

    # Train
    for m in tribe.active_members():
        if m.domain:
            final_loss = train(m.weights, m.domain, steps=200, lr=0.005,
                               fwd=cnn_forward_batch)
            print(f"    Member {m.id}: {len(m.domain)} patterns, loss={final_loss:.4f}")

    # Health check
    recs = tribe.health_check(overlap_threshold=0.3, freeze_grad_threshold=1e-5,
                              competence_threshold=0.05, min_active=3)
    for rec_type, target, reason in recs:
        if rec_type == 'freeze':
            tribe.freeze_member(target)
            print(f"    Frozen member {target} ({reason})")

    per_class, overall = measure_accuracy(tribe, test_imgs, test_labels)
    all_patterns_final = patterns_0 + patterns_1 + patterns_2 + patterns_3
    cluster_list = [make_mnist_patterns(train_imgs, train_labels, [d], 30, seed=99)
                    for d in range(10)]
    prec = measure_knowledge_precision(tribe, cluster_list)
    sys_l = measure_system_loss(tribe, all_patterns_final)
    red = measure_redundancy(tribe)
    gen_metrics.append((3, overall, sys_l, prec, red))

    print(f"  Accuracy: overall={overall:.1%}")
    for d in sorted(per_class):
        print(f"    digit {d}: {per_class[d]:.1%}")
    print(f"  Metrics: precision={prec:.2f}, sys_loss={sys_l:.4f}, redundancy={red:.2f}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n  {'Gen':>3} {'Accuracy':>9} {'Sys Loss':>10} {'Precision':>10} {'Red.':>6}")
    for gen, acc, sl, prec, red in gen_metrics:
        print(f"  {gen:>3} {acc:>9.1%} {sl:>10.4f} {prec:>10.2f} {red:>6.2f}")

    tribe.print_status()

    # ── Backward Transfer Check ───────────────────────────────
    print(f"\n  BACKWARD TRANSFER CHECK")
    per_class_final, _ = measure_accuracy(tribe, test_imgs, test_labels)
    for d in [0, 1, 2]:
        acc = per_class_final.get(d, 0)
        print(f"    digit {d} (learned gen 0): final accuracy {acc:.1%}")

    print(f"\n  History:")
    for gen, msg in tribe.history:
        print(f"    [gen {gen}] {msg}")

    return gen_metrics


if __name__ == '__main__':
    run_mnist_experiment()
