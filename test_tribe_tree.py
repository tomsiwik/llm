"""Tests for hierarchical knowledge tree in tribe/core.py.

Verifies:
1. Tree structure basics (depth, path, subtree, roots)
2. Hierarchical routing
3. Hierarchical forward (depth-weighted mixing)
4. Bond hierarchical (child creation)
5. Prune tree (leaf removal, parent reactivation)
6. Tree health check
"""

import mlx.core as mx
import numpy as np

from tribe import State, Tribe, make_expert, forward_batch, loss_on, train


def make_tribe_with_tree():
    """Helper: build a 2-level tree with 1 root and 2 children."""
    tribe = Tribe()
    root_w = make_expert(seed=0)
    root = tribe.add_member(root_w)

    child1_w = make_expert(seed=1)
    child1 = tribe.add_child(root.id, child1_w)

    child2_w = make_expert(seed=2)
    child2 = tribe.add_child(root.id, child2_w)

    return tribe, root, child1, child2


def test_tree_structure_basics():
    """Create a Tribe, add root + children, verify depth, tree_path, subtree, tree_roots."""
    print("\n" + "=" * 60)
    print("  TEST: Tree Structure Basics")
    print("=" * 60)

    tribe, root, child1, child2 = make_tribe_with_tree()

    # Verify depth
    assert tribe.depth[root.id] == 0, f"Root depth should be 0, got {tribe.depth[root.id]}"
    assert tribe.depth[child1.id] == 1, f"Child1 depth should be 1, got {tribe.depth[child1.id]}"
    assert tribe.depth[child2.id] == 1, f"Child2 depth should be 1, got {tribe.depth[child2.id]}"

    # Verify parent_of
    assert child1.id in tribe.parent_of, "Child1 should have a parent"
    assert tribe.parent_of[child1.id] == root.id, "Child1's parent should be root"
    assert tribe.parent_of[child2.id] == root.id, "Child2's parent should be root"
    assert root.id not in tribe.parent_of, "Root should have no parent"

    # Verify children_of
    assert root.id in tribe.children_of, "Root should have children"
    assert set(tribe.children_of[root.id]) == {child1.id, child2.id}

    # Verify tree_roots
    roots = tribe.tree_roots()
    assert len(roots) == 1, f"Should have 1 root, got {len(roots)}"
    assert roots[0].id == root.id

    # Verify tree_path
    path_child1 = tribe.tree_path(child1.id)
    assert path_child1 == [root.id, child1.id], f"Path to child1 wrong: {path_child1}"

    path_root = tribe.tree_path(root.id)
    assert path_root == [root.id], f"Path to root wrong: {path_root}"

    # Verify subtree
    desc = tribe.subtree(root.id)
    assert set(desc) == {child1.id, child2.id}, f"Subtree of root wrong: {desc}"

    desc_child = tribe.subtree(child1.id)
    assert desc_child == [], f"Leaf should have no descendants: {desc_child}"

    # Add grandchild to test deeper tree
    grandchild_w = make_expert(seed=3)
    grandchild = tribe.add_child(child1.id, grandchild_w)
    assert tribe.depth[grandchild.id] == 2, f"Grandchild depth should be 2, got {tribe.depth[grandchild.id]}"

    path_gc = tribe.tree_path(grandchild.id)
    assert path_gc == [root.id, child1.id, grandchild.id], f"Path to grandchild wrong: {path_gc}"

    desc_root = tribe.subtree(root.id)
    assert set(desc_root) == {child1.id, child2.id, grandchild.id}

    # Verify edges (graph connectivity)
    assert root.id in tribe.edges[child1.id]
    assert child1.id in tribe.edges[root.id]

    print("  PASSED: tree depth, parent_of, children_of, tree_roots, tree_path, subtree")
    return True


def test_add_member_sets_depth_zero():
    """Verify that add_member sets depth=0 for root-level members."""
    print("\n" + "=" * 60)
    print("  TEST: add_member sets depth=0")
    print("=" * 60)

    tribe = Tribe()
    m0 = tribe.add_member(make_expert(seed=0))
    m1 = tribe.add_member(make_expert(seed=1))

    assert tribe.depth[m0.id] == 0, f"Member 0 depth should be 0, got {tribe.depth[m0.id]}"
    assert tribe.depth[m1.id] == 0, f"Member 1 depth should be 0, got {tribe.depth[m1.id]}"

    print("  PASSED: add_member sets depth=0 for root members")
    return True


def test_hierarchical_routing():
    """Create 2-level tree with different experts, verify route_hierarchical returns correct path."""
    print("\n" + "=" * 60)
    print("  TEST: Hierarchical Routing")
    print("=" * 60)

    tribe = Tribe()

    # Create two roots
    root1 = tribe.add_member(make_expert(seed=10))
    root2 = tribe.add_member(make_expert(seed=20))

    # Add children to root1
    child1a = tribe.add_child(root1.id, make_expert(seed=11))
    child1b = tribe.add_child(root1.id, make_expert(seed=12))

    # Train root1 and children on specific patterns
    rng = np.random.RandomState(42)
    x = mx.array(rng.randn(4).astype(np.float32))
    t = mx.array(rng.randn(4).astype(np.float32))

    # Route with target (loss-based)
    path = tribe.route_hierarchical(x, target=t)
    assert len(path) >= 1, "Path should have at least 1 member"
    # First in path should be a root
    root_ids = {root1.id, root2.id}
    assert path[0].id in root_ids, f"First in path should be a root, got {path[0].id}"

    # If root1 was selected, path should descend to one of its children
    if path[0].id == root1.id and len(path) > 1:
        assert path[1].id in {child1a.id, child1b.id}, \
            f"Second in path should be child of root1, got {path[1].id}"

    # Route without target (confidence-based)
    path_conf = tribe.route_hierarchical(x, target=None)
    assert len(path_conf) >= 1, "Confidence path should have at least 1 member"

    print(f"  Loss-based path: {[m.id for m in path]}")
    print(f"  Confidence path: {[m.id for m in path_conf]}")
    print("  PASSED: hierarchical routing produces valid tree paths")
    return True


def test_hierarchical_forward():
    """Verify depth-weighted mixing produces expected output shape."""
    print("\n" + "=" * 60)
    print("  TEST: Hierarchical Forward")
    print("=" * 60)

    tribe, root, child1, child2 = make_tribe_with_tree()

    rng = np.random.RandomState(42)
    X = mx.array(rng.randn(5, 4).astype(np.float32))  # batch of 5

    # Forward through a path of [root, child1]
    path = [root, child1]
    output = tribe.hierarchical_forward(path, X)
    assert output is not None, "Output should not be None"
    assert output.shape == (5, 4), f"Output shape should be (5, 4), got {output.shape}"

    # Empty path returns None
    none_output = tribe.hierarchical_forward([], X)
    assert none_output is None, "Empty path should return None"

    # Single member path should return that member's output (weight=1.0)
    single_out = tribe.hierarchical_forward([root], X)
    expected = forward_batch(root.weights, X)
    mx.eval(single_out, expected)
    diff = mx.max(mx.abs(single_out - expected)).item()
    assert diff < 1e-5, f"Single member forward should match direct forward, diff={diff}"

    # Verify depth weighting: child should have higher weight
    # For path [root, child1]: weights are [2^0=1, 2^1=2], so child1 gets 2/3
    root_out = forward_batch(root.weights, X)
    child_out = forward_batch(child1.weights, X)
    mx.eval(root_out, child_out)
    expected_mixed = (1.0/3.0) * root_out + (2.0/3.0) * child_out
    mx.eval(output, expected_mixed)
    diff = mx.max(mx.abs(output - expected_mixed)).item()
    assert diff < 1e-5, f"Depth-weighted mix incorrect, diff={diff}"

    print("  PASSED: hierarchical forward produces correct shape and depth weighting")
    return True


def test_bond_hierarchical():
    """Verify creating a child from parent produces correct tree structure."""
    print("\n" + "=" * 60)
    print("  TEST: Bond Hierarchical")
    print("=" * 60)

    tribe = Tribe()
    root = tribe.add_member(make_expert(seed=0))

    # Bond hierarchical creates a child
    child = tribe.bond_hierarchical(root.id, seed=42)

    assert child.id != root.id, "Child should have different ID"
    assert tribe.parent_of[child.id] == root.id, "Child's parent should be root"
    assert child.id in tribe.children_of[root.id], "Root should list child"
    assert tribe.depth[child.id] == 1, f"Child depth should be 1, got {tribe.depth[child.id]}"

    # Child weights should be close to parent (parent + small noise)
    for k in root.weights:
        diff = mx.max(mx.abs(child.weights[k] - root.weights[k])).item()
        assert diff < 0.1, f"Child weights should be close to parent, diff={diff} for key {k}"

    # Verify graph edge exists
    assert root.id in tribe.edges[child.id]
    assert child.id in tribe.edges[root.id]

    # Create grandchild
    grandchild = tribe.bond_hierarchical(child.id, seed=99)
    assert tribe.depth[grandchild.id] == 2
    assert tribe.parent_of[grandchild.id] == child.id

    # Verify history logged
    tree_events = [msg for _, msg in tribe.history if 'TREE GROW' in msg]
    assert len(tree_events) == 2, f"Should have 2 TREE GROW events, got {len(tree_events)}"

    print(f"  Created: root={root.id}, child={child.id} (depth=1), grandchild={grandchild.id} (depth=2)")
    print("  PASSED: bond_hierarchical creates correct tree structure")
    return True


def test_prune_tree():
    """Verify pruning a leaf removes it, parent reactivates if it was frozen."""
    print("\n" + "=" * 60)
    print("  TEST: Prune Tree")
    print("=" * 60)

    tribe, root, child1, child2 = make_tribe_with_tree()

    # Prune child1 (leaf)
    tribe.prune_tree(child1.id)
    assert tribe.members[child1.id].state == State.RECYCLED, "Pruned member should be RECYCLED"
    assert child1.id not in tribe.parent_of, "Pruned member should not be in parent_of"
    assert child1.id not in tribe.depth, "Pruned member should not be in depth"

    # Root should still have child2
    assert child2.id in tribe.children_of[root.id]
    assert child1.id not in tribe.children_of[root.id]

    # Test parent reactivation: freeze root, prune last child
    root.freeze()
    assert root.state == State.FROZEN

    tribe.prune_tree(child2.id)
    assert tribe.members[child2.id].state == State.RECYCLED
    assert root.state == State.ACTIVE, "Root should be reactivated after all children pruned"
    assert root.id not in tribe.children_of, "Root should have no children_of entry"

    print("  PASSED: prune_tree removes leaf, reactivates frozen parent")
    return True


def test_prune_subtree():
    """Verify pruning a non-leaf recursively prunes all descendants."""
    print("\n" + "=" * 60)
    print("  TEST: Prune Subtree")
    print("=" * 60)

    tribe = Tribe()
    root = tribe.add_member(make_expert(seed=0))
    child = tribe.add_child(root.id, make_expert(seed=1))
    grandchild = tribe.add_child(child.id, make_expert(seed=2))

    # Prune child (non-leaf) should also prune grandchild
    tribe.prune_tree(child.id)

    assert tribe.members[child.id].state == State.RECYCLED
    assert tribe.members[grandchild.id].state == State.RECYCLED
    assert grandchild.id not in tribe.parent_of
    assert child.id not in tribe.parent_of
    assert grandchild.id not in tribe.depth
    assert child.id not in tribe.depth

    print("  PASSED: prune_tree recursively prunes subtree")
    return True


def test_tree_health_check():
    """Create a tree with known properties, verify recommendations."""
    print("\n" + "=" * 60)
    print("  TEST: Tree Health Check")
    print("=" * 60)

    tribe = Tribe()
    root = tribe.add_member(make_expert(seed=0))

    # Create a depth-2 expert with a small domain (should trigger prune recommendation)
    child = tribe.add_child(root.id, make_expert(seed=1))
    grandchild = tribe.add_child(child.id, make_expert(seed=2))

    # Give grandchild a tiny domain (< 3 unique patterns)
    rng = np.random.RandomState(42)
    grandchild.domain = [
        (mx.array(rng.randn(4).astype(np.float32)),
         mx.array(rng.randn(4).astype(np.float32)))
        for _ in range(2)
    ]

    recs = tribe.tree_health_check()
    prune_recs = [r for r in recs if r[0] == 'prune']
    assert len(prune_recs) >= 1, f"Should recommend pruning deep expert with few unique patterns, got {recs}"
    assert any(r[1] == grandchild.id for r in prune_recs), \
        f"Should recommend pruning grandchild {grandchild.id}"

    # Create a member with large domain at depth 0 (should trigger split)
    big_member = tribe.add_member(make_expert(seed=10))
    big_member.domain = [
        (mx.array(rng.randn(4).astype(np.float32)),
         mx.array(rng.randn(4).astype(np.float32)))
        for _ in range(60)
    ]

    recs2 = tribe.tree_health_check()
    split_recs = [r for r in recs2 if r[0] == 'split']
    assert len(split_recs) >= 1, f"Should recommend splitting large domain member, got {recs2}"
    assert any(r[1] == big_member.id for r in split_recs), \
        f"Should recommend splitting member {big_member.id}"

    print(f"  Recommendations: {[(r[0], r[1]) for r in recs2]}")
    print("  PASSED: tree_health_check produces correct recommendations")
    return True


def test_print_status_with_depth():
    """Verify print_status includes depth info without errors."""
    print("\n" + "=" * 60)
    print("  TEST: print_status with depth")
    print("=" * 60)

    tribe, root, child1, child2 = make_tribe_with_tree()

    # Should not raise
    tribe.print_status()

    print("  PASSED: print_status works with tree structure")
    return True


if __name__ == '__main__':
    test_tree_structure_basics()
    test_add_member_sets_depth_zero()
    test_hierarchical_routing()
    test_hierarchical_forward()
    test_bond_hierarchical()
    test_prune_tree()
    test_prune_subtree()
    test_tree_health_check()
    test_print_status_with_depth()
    print("\n" + "=" * 60)
    print("  ALL TREE TESTS PASSED")
    print("=" * 60)
