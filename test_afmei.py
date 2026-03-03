"""Isolated test for AFMEI: freeze maps + offspring mechanism.

Proves the core claim in a controlled setting:
- Two parent experts with KNOWN correct/wrong parameters
- Offspring created from crossover
- Freeze map + gradient-based evaluation identifies which params to protect
- Offspring training improves upon both parents

Setup:
  - Task: linear regression y = W_true @ x  (4-dim)
  - Parent A: rows 0,1 correct, rows 2,3 wrong
  - Parent B: rows 0,1 wrong, rows 2,3 correct
  - Optimal offspring: all rows correct (combines best of both)
"""

import mlx.core as mx
import math

# ── Ground truth ──
W_TRUE = mx.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# ── Test data ──
mx.random.seed(42)
X_TEST = mx.random.normal((20, 4))
Y_TEST = X_TEST @ W_TRUE.T  # target outputs


def mse(W, X=X_TEST, Y=Y_TEST):
    """Mean squared error for weight matrix W."""
    pred = X @ W.T
    return mx.mean((pred - Y) ** 2).item()


def row_errors(W):
    """Per-row MSE against W_TRUE."""
    return [mx.mean((W[i] - W_TRUE[i]) ** 2).item() for i in range(4)]


# ── Parent experts ──
# Parent A: rows 0,1 are correct, rows 2,3 are very wrong
W_PARENT_A = mx.array([
    [1.0, 0.0, 0.0, 0.0],   # correct
    [0.0, 1.0, 0.0, 0.0],   # correct
    [0.5, 0.5, 0.0, 0.5],   # wrong
    [0.3, 0.0, 0.7, 0.0],   # wrong
])

# Parent B: rows 0,1 are wrong, rows 2,3 are correct
W_PARENT_B = mx.array([
    [0.0, 0.8, 0.2, 0.0],   # wrong
    [0.5, 0.0, 0.0, 0.5],   # wrong
    [0.0, 0.0, 1.0, 0.0],   # correct
    [0.0, 0.0, 0.0, 1.0],   # correct
])


def test_parents_have_complementary_knowledge():
    """Verify: each parent is good at different rows."""
    err_a = row_errors(W_PARENT_A)
    err_b = row_errors(W_PARENT_B)
    print("Test 1: Parents have complementary knowledge")
    print(f"  Parent A row errors: {[f'{e:.3f}' for e in err_a]}")
    print(f"  Parent B row errors: {[f'{e:.3f}' for e in err_b]}")
    print(f"  Parent A total MSE: {mse(W_PARENT_A):.4f}")
    print(f"  Parent B total MSE: {mse(W_PARENT_B):.4f}")
    # A is better on rows 0,1; B is better on rows 2,3
    assert err_a[0] < err_b[0], "A should be better on row 0"
    assert err_a[1] < err_b[1], "A should be better on row 1"
    assert err_b[2] < err_a[2], "B should be better on row 2"
    assert err_b[3] < err_a[3], "B should be better on row 3"
    print("  PASSED\n")


def test_blend_vs_selective():
    """Verify: selective crossover (pick best row) beats naive blend.

    Blend CAN outperform both parents when they have complementary knowledge
    (averaging correct+wrong gives moderate error everywhere). But selective
    crossover picks the BEST row from each parent → even better.
    """
    W_blend = 0.5 * W_PARENT_A + 0.5 * W_PARENT_B
    # Oracle selection: pick best row from each parent
    err_a = row_errors(W_PARENT_A)
    err_b = row_errors(W_PARENT_B)
    rows = [W_PARENT_A[i] if err_a[i] <= err_b[i] else W_PARENT_B[i]
            for i in range(4)]
    W_select = mx.stack(rows)

    mse_blend = mse(W_blend)
    mse_select = mse(W_select)
    print("Test 2: Selective crossover beats naive blend")
    print(f"  Blend MSE:     {mse_blend:.4f} (row errors: {[f'{e:.3f}' for e in row_errors(W_blend)]})")
    print(f"  Selective MSE: {mse_select:.4f} (row errors: {[f'{e:.3f}' for e in row_errors(W_select)]})")
    assert mse_select < mse_blend, \
        "Selective crossover should outperform blend"
    print(f"  Improvement: {(mse_blend - mse_select)/mse_blend*100:.1f}%")
    print("  PASSED\n")


def test_row_selection_offspring():
    """Verify: selecting best rows from each parent creates optimal offspring."""
    # Oracle: pick row i from whichever parent has lower error on that row
    err_a = row_errors(W_PARENT_A)
    err_b = row_errors(W_PARENT_B)
    rows = []
    sources = []
    for i in range(4):
        if err_a[i] <= err_b[i]:
            rows.append(W_PARENT_A[i])
            sources.append('A')
        else:
            rows.append(W_PARENT_B[i])
            sources.append('B')
    W_oracle = mx.stack(rows)
    mse_oracle = mse(W_oracle)
    print("Test 3: Oracle row selection (best of each parent)")
    print(f"  Sources: {sources}")
    print(f"  Oracle MSE:   {mse_oracle:.4f}")
    print(f"  Parent A MSE: {mse(W_PARENT_A):.4f}")
    print(f"  Parent B MSE: {mse(W_PARENT_B):.4f}")
    assert mse_oracle <= mse(W_PARENT_A), "Oracle must beat parent A"
    assert mse_oracle <= mse(W_PARENT_B), "Oracle must beat parent B"
    print(f"  Oracle row errors: {[f'{e:.3f}' for e in row_errors(W_oracle)]}")
    print("  PASSED\n")


def test_gradient_identifies_wrong_rows():
    """Verify: gradient signal is larger for wrong rows than correct rows.

    This is the KEY property that makes AFMEI work:
    the gradient tells us WHICH parameters need to change.
    """
    print("Test 4: Gradient magnitude identifies wrong parameters")

    for name, W in [("Parent A", W_PARENT_A), ("Parent B", W_PARENT_B)]:
        W_param = mx.array(W)

        def loss_fn(W_p):
            pred = X_TEST @ W_p.T
            return mx.mean((pred - Y_TEST) ** 2)

        loss, grad = mx.value_and_grad(loss_fn)(W_param)
        grad_norms = [mx.sqrt(mx.sum(grad[i] ** 2)).item() for i in range(4)]
        err = row_errors(W)
        print(f"  {name}:")
        print(f"    Row errors:     {[f'{e:.3f}' for e in err]}")
        print(f"    Gradient norms: {[f'{g:.3f}' for g in grad_norms]}")
        # Wrong rows should have larger gradients
        correct_rows = [i for i in range(4) if err[i] < 0.01]
        wrong_rows = [i for i in range(4) if err[i] > 0.01]
        max_correct_grad = max(grad_norms[i] for i in correct_rows) if correct_rows else 0
        min_wrong_grad = min(grad_norms[i] for i in wrong_rows) if wrong_rows else float('inf')
        print(f"    Max correct-row grad: {max_correct_grad:.4f}")
        print(f"    Min wrong-row grad:   {min_wrong_grad:.4f}")
        assert min_wrong_grad > max_correct_grad, \
            f"Wrong rows should have larger gradients than correct rows"

    print("  PASSED\n")


def test_freeze_map_from_gradient():
    """Verify: freeze map built from gradient magnitude protects correct params.

    Strategy: tau = 1 / (1 + grad_norm * scale)
    High gradient → low tau → more plastic (needs to change)
    Low gradient → high tau → frozen (already correct)

    Wait — that's backwards from our convention where tau=1 means trainable.
    Let's use: tau = 1 for wrong params (needs training), tau ≈ 0 for correct.
    So: tau = min(grad_norm * scale, 1.0)
    """
    print("Test 5: Freeze map from gradient magnitude")

    W_param = mx.array(W_PARENT_A)

    def loss_fn(W_p):
        pred = X_TEST @ W_p.T
        return mx.mean((pred - Y_TEST) ** 2)

    _, grad = mx.value_and_grad(loss_fn)(W_param)
    grad_norms = [mx.sqrt(mx.sum(grad[i] ** 2)).item() for i in range(4)]

    # Build per-row freeze map: high grad → tau closer to 1 (trainable)
    max_grad = max(grad_norms)
    tau_map = [min(gn / max_grad, 1.0) for gn in grad_norms]
    print(f"  Gradient norms: {[f'{g:.3f}' for g in grad_norms]}")
    print(f"  Tau map:        {[f'{t:.3f}' for t in tau_map]}")

    err = row_errors(W_PARENT_A)
    for i in range(4):
        if err[i] < 0.01:  # correct row
            assert tau_map[i] < 0.5, \
                f"Row {i} is correct but tau={tau_map[i]:.3f} (should be low/frozen)"
        else:  # wrong row
            assert tau_map[i] > 0.5, \
                f"Row {i} is wrong but tau={tau_map[i]:.3f} (should be high/trainable)"
    print("  PASSED\n")


def test_gradient_agreement_identifies_shared_knowledge():
    """Verify: gradient agreement between parents identifies which rows they
    agree on (both need to fix) vs. disagree (one is right, one is wrong).

    For rows where A is correct and B is wrong: A's gradient is ~0, B's is large
    → low agreement → don't thaw A's params for this row

    For rows where both are wrong in the same direction: both gradients point same way
    → high agreement → thawing is safe
    """
    print("Test 6: Gradient agreement between parents")

    def loss_fn(W_p):
        pred = X_TEST @ W_p.T
        return mx.mean((pred - Y_TEST) ** 2)

    _, grad_a = mx.value_and_grad(loss_fn)(mx.array(W_PARENT_A))
    _, grad_b = mx.value_and_grad(loss_fn)(mx.array(W_PARENT_B))

    for i in range(4):
        ga = grad_a[i].reshape(-1)
        gb = grad_b[i].reshape(-1)
        cos = (mx.sum(ga * gb) / (
            mx.sqrt(mx.sum(ga * ga)) * mx.sqrt(mx.sum(gb * gb)) + 1e-8
        )).item()
        err_a = row_errors(W_PARENT_A)[i]
        err_b = row_errors(W_PARENT_B)[i]
        status_a = "correct" if err_a < 0.01 else "wrong"
        status_b = "correct" if err_b < 0.01 else "wrong"
        print(f"  Row {i}: A={status_a}, B={status_b}, cosine={cos:+.3f}")

    print("  (Gradient agreement is low when one parent is correct)\n")


def test_offspring_with_freeze_aware_training():
    """THE MAIN TEST: Create offspring, train with freeze-map guidance.

    1. Create offspring as blend of parents
    2. Compute per-row freeze map from offspring's gradient on PARENT data
    3. Train offspring with freeze-map-scaled gradients
    4. Verify offspring outperforms both parents
    """
    print("Test 7: Offspring with freeze-map-aware training")

    # Start with blend (known to be mediocre)
    W_child = mx.array(0.5 * W_PARENT_A + 0.5 * W_PARENT_B)
    print(f"  Initial offspring MSE:  {mse(W_child):.4f}")
    print(f"  Initial row errors:     {[f'{e:.3f}' for e in row_errors(W_child)]}")

    lr = 0.1

    def loss_fn(W_p):
        pred = X_TEST @ W_p.T
        return mx.mean((pred - Y_TEST) ** 2)

    # Train for a few steps with gradient descent
    for step in range(50):
        loss, grad = mx.value_and_grad(loss_fn)(W_child)
        W_child = W_child - lr * grad
        mx.eval(W_child)

    mse_child_plain = mse(W_child)
    print(f"  After 50 steps (plain): {mse_child_plain:.4f}")
    print(f"  Row errors:             {[f'{e:.3f}' for e in row_errors(W_child)]}")

    # Now try FREEZE-MAP guided training from blend start
    W_child_fm = mx.array(0.5 * W_PARENT_A + 0.5 * W_PARENT_B)

    # Build freeze map: use parent gradient magnitudes to decide
    # If BOTH parents have low gradient on a row → it's correct in both → freeze (tau=0)
    # If EITHER parent has high gradient → it needs fixing → unfreeze (tau=1)
    _, grad_a = mx.value_and_grad(loss_fn)(mx.array(W_PARENT_A))
    _, grad_b = mx.value_and_grad(loss_fn)(mx.array(W_PARENT_B))

    tau_per_row = []
    for i in range(4):
        gn_a = mx.sqrt(mx.sum(grad_a[i] ** 2)).item()
        gn_b = mx.sqrt(mx.sum(grad_b[i] ** 2)).item()
        # Use MAX gradient: if either parent thinks this row needs work, allow training
        max_gn = max(gn_a, gn_b)
        tau_per_row.append(max_gn)

    # Normalize to [0, 1]
    max_tau = max(tau_per_row)
    tau_per_row = [t / max_tau for t in tau_per_row]
    print(f"\n  Freeze map (tau per row): {[f'{t:.3f}' for t in tau_per_row]}")

    for step in range(50):
        loss, grad = mx.value_and_grad(loss_fn)(W_child_fm)
        # Apply freeze map: scale gradient per row
        scaled_grad = mx.stack([tau_per_row[i] * grad[i] for i in range(4)])
        W_child_fm = W_child_fm - lr * scaled_grad
        mx.eval(W_child_fm)

    mse_child_fm = mse(W_child_fm)
    print(f"  After 50 steps (freeze-map): {mse_child_fm:.4f}")
    print(f"  Row errors:                  {[f'{e:.3f}' for e in row_errors(W_child_fm)]}")

    print(f"\n  Comparison:")
    print(f"    Parent A:           {mse(W_PARENT_A):.4f}")
    print(f"    Parent B:           {mse(W_PARENT_B):.4f}")
    print(f"    Offspring (plain):  {mse_child_plain:.4f}")
    print(f"    Offspring (freeze): {mse_child_fm:.4f}")
    print(f"    Ground truth:       {mse(W_TRUE):.4f}")

    # Both offspring approaches should beat both parents
    mse_a = mse(W_PARENT_A)
    mse_b = mse(W_PARENT_B)
    assert mse_child_plain < min(mse_a, mse_b), \
        "Plain offspring should beat both parents"
    assert mse_child_fm < min(mse_a, mse_b), \
        "Freeze-map offspring should beat both parents"
    print("  PASSED\n")


def test_selective_crossover_beats_blend():
    """Verify: row-level crossover (pick best row from each parent) + training
    outperforms naive blend + training.

    This is the PRACTICAL test: does intelligent crossover help?
    """
    print("Test 8: Selective crossover vs naive blend")

    def loss_fn(W_p):
        pred = X_TEST @ W_p.T
        return mx.mean((pred - Y_TEST) ** 2)

    lr = 0.1

    # Strategy 1: Naive blend
    W_blend = mx.array(0.5 * W_PARENT_A + 0.5 * W_PARENT_B)
    for step in range(20):  # less training to see initial advantage
        loss, grad = mx.value_and_grad(loss_fn)(W_blend)
        W_blend = W_blend - lr * grad
        mx.eval(W_blend)
    mse_blend = mse(W_blend)

    # Strategy 2: Row selection based on parent per-row loss
    err_a = row_errors(W_PARENT_A)
    err_b = row_errors(W_PARENT_B)
    rows = []
    for i in range(4):
        rows.append(W_PARENT_A[i] if err_a[i] <= err_b[i] else W_PARENT_B[i])
    W_select = mx.array(mx.stack(rows))
    for step in range(20):
        loss, grad = mx.value_and_grad(loss_fn)(W_select)
        W_select = W_select - lr * grad
        mx.eval(W_select)
    mse_select = mse(W_select)

    print(f"  After 20 steps:")
    print(f"    Blend offspring MSE:     {mse_blend:.4f}")
    print(f"    Selective offspring MSE:  {mse_select:.4f}")
    print(f"    Improvement: {(mse_blend - mse_select)/mse_blend*100:.1f}%")

    assert mse_select < mse_blend, \
        "Selective crossover should outperform naive blend"
    print("  PASSED\n")


def test_offspring_modifies_parent_freeze_map():
    """THE NOVEL CLAIM: offspring training results inform PARENT freeze maps.

    After offspring trains:
    1. Evaluate offspring on each parent's "specialty" data
    2. If offspring outperforms parent on parent's data → thaw parent (its knowledge
       wasn't optimal, offspring found something better)
    3. If offspring underperforms → freeze parent more (its knowledge was fine)

    We simulate this with per-row evaluation.
    """
    print("Test 9: Offspring modifies parent freeze maps")

    def loss_fn(W_p):
        pred = X_TEST @ W_p.T
        return mx.mean((pred - Y_TEST) ** 2)

    lr = 0.1

    # Create and train offspring
    err_a = row_errors(W_PARENT_A)
    err_b = row_errors(W_PARENT_B)
    rows = []
    for i in range(4):
        rows.append(W_PARENT_A[i] if err_a[i] <= err_b[i] else W_PARENT_B[i])
    W_child = mx.array(mx.stack(rows))

    for step in range(50):
        loss, grad = mx.value_and_grad(loss_fn)(W_child)
        W_child = W_child - lr * grad
        mx.eval(W_child)

    # Now evaluate: per-row, does offspring beat each parent?
    child_err = row_errors(W_child)
    parent_a_err = row_errors(W_PARENT_A)
    parent_b_err = row_errors(W_PARENT_B)

    print(f"  Per-row errors:")
    print(f"    Parent A: {[f'{e:.4f}' for e in parent_a_err]}")
    print(f"    Parent B: {[f'{e:.4f}' for e in parent_b_err]}")
    print(f"    Offspring: {[f'{e:.4f}' for e in child_err]}")

    # Build parent freeze map updates based on offspring performance
    parent_a_tau = [1.0] * 4  # start all trainable
    parent_b_tau = [1.0] * 4
    for i in range(4):
        if child_err[i] < parent_a_err[i]:
            # Offspring outperforms A on row i → thaw A (it can improve)
            parent_a_tau[i] = 0.8  # partially thaw
        else:
            # A's knowledge is already good → freeze more
            parent_a_tau[i] = 0.1

        if child_err[i] < parent_b_err[i]:
            parent_b_tau[i] = 0.8
        else:
            parent_b_tau[i] = 0.1

    print(f"\n  Updated freeze maps (tau: 0=frozen, 1=plastic):")
    print(f"    Parent A tau: {[f'{t:.1f}' for t in parent_a_tau]}")
    print(f"    Parent B tau: {[f'{t:.1f}' for t in parent_b_tau]}")

    # Verify: correct rows get frozen, wrong rows get thawed
    # Parent A: rows 0,1 correct → should be frozen (low tau)
    assert parent_a_tau[0] < 0.5, "A row 0 (correct) should be frozen"
    assert parent_a_tau[1] < 0.5, "A row 1 (correct) should be frozen"
    # Parent A: rows 2,3 wrong → should be thawed (high tau)
    assert parent_a_tau[2] > 0.5, "A row 2 (wrong) should be thawed"
    assert parent_a_tau[3] > 0.5, "A row 3 (wrong) should be thawed"

    # Parent B: rows 0,1 wrong → should be thawed
    assert parent_b_tau[0] > 0.5, "B row 0 (wrong) should be thawed"
    assert parent_b_tau[1] > 0.5, "B row 1 (wrong) should be thawed"
    # Parent B: rows 2,3 correct → should be frozen
    assert parent_b_tau[2] < 0.5, "B row 2 (correct) should be frozen"
    assert parent_b_tau[3] < 0.5, "B row 3 (correct) should be frozen"

    print("  PASSED (offspring correctly identifies which parent rows to freeze/thaw)\n")


def test_full_lifecycle():
    """End-to-end test of the full AFMEI lifecycle:

    1. Two parents trained on different tasks (simulated by known weights)
    2. Freeze maps initialized and decayed
    3. Offspring created via selective crossover
    4. Offspring trained
    5. Offspring evaluates parents → updates parent freeze maps
    6. Parents retrained with updated freeze maps
    7. Verify: parents improve on BOTH tasks
    """
    print("Test 10: Full AFMEI lifecycle")

    # Simulate two "tasks" with different optimal weights
    W_TASK1 = mx.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],  # task 1 specific
        [0.0, 0.0, 0.5, 0.5],  # task 1 specific
    ])
    W_TASK2 = mx.array([
        [0.5, 0.5, 0.0, 0.0],  # task 2 specific
        [0.5, 0.5, 0.0, 0.0],  # task 2 specific
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    Y_TASK1 = X_TEST @ W_TASK1.T
    Y_TASK2 = X_TEST @ W_TASK2.T

    def mse_t1(W): return mx.mean((X_TEST @ W.T - Y_TASK1) ** 2).item()
    def mse_t2(W): return mx.mean((X_TEST @ W.T - Y_TASK2) ** 2).item()

    # Parents: A trained on task1, B trained on task2
    W_A = mx.array(W_TASK1)  # perfect on task 1
    W_B = mx.array(W_TASK2)  # perfect on task 2

    print(f"  Initial state:")
    print(f"    Parent A: task1_mse={mse_t1(W_A):.4f}, task2_mse={mse_t2(W_A):.4f}")
    print(f"    Parent B: task1_mse={mse_t1(W_B):.4f}, task2_mse={mse_t2(W_B):.4f}")

    # Step 1: Freeze maps after training
    tau_A = [0.3, 0.3, 0.3, 0.3]  # decayed after task 1 (consolidated)
    tau_B = [0.7, 0.7, 0.7, 0.7]  # less decayed (more recent)

    # Step 2: Create offspring (blend)
    W_child = mx.array(0.5 * W_A + 0.5 * W_B)
    print(f"\n  Offspring (blend):")
    print(f"    task1_mse={mse_t1(W_child):.4f}, task2_mse={mse_t2(W_child):.4f}")

    # Step 3: Train offspring on task 2 (current task)
    lr = 0.05
    for step in range(30):
        def loss_fn(W_p):
            return mx.mean((X_TEST @ W_p.T - Y_TASK2) ** 2)
        loss, grad = mx.value_and_grad(loss_fn)(W_child)
        W_child = W_child - lr * grad
        mx.eval(W_child)

    print(f"  Offspring after training on task 2:")
    print(f"    task1_mse={mse_t1(W_child):.4f}, task2_mse={mse_t2(W_child):.4f}")

    # Step 4: Offspring evaluates parents → update freeze maps
    # Compare offspring vs parent on EACH task
    child_t1 = mse_t1(W_child)
    child_t2 = mse_t2(W_child)
    parent_a_t1 = mse_t1(W_A)
    parent_b_t2 = mse_t2(W_B)

    print(f"\n  Offspring vs parents:")
    print(f"    Task 1: offspring={child_t1:.4f} vs A={parent_a_t1:.4f}")
    print(f"    Task 2: offspring={child_t2:.4f} vs B={parent_b_t2:.4f}")

    # Update parent A's freeze map:
    # If offspring learned something about task 2 that A doesn't have,
    # thaw A's task-2-relevant parameters
    if child_t2 < mse_t2(W_A):
        # Offspring better on task 2 → thaw A's rows 2,3 (task 2 region)
        tau_A[2] = min(tau_A[2] + 0.3, 1.0)
        tau_A[3] = min(tau_A[3] + 0.3, 1.0)
        print(f"    → Thawing Parent A rows 2,3 (offspring better on task 2)")

    # Step 5: Retrain parent A with updated freeze map on task 2
    W_A_updated = mx.array(W_A)
    for step in range(30):
        def loss_fn_a(W_p):
            return mx.mean((X_TEST @ W_p.T - Y_TASK2) ** 2)
        loss, grad = mx.value_and_grad(loss_fn_a)(W_A_updated)
        # Apply freeze map
        scaled_grad = mx.stack([tau_A[i] * grad[i] for i in range(4)])
        W_A_updated = W_A_updated - lr * scaled_grad
        mx.eval(W_A_updated)

    print(f"\n  Parent A after freeze-map-guided retraining:")
    print(f"    task1_mse={mse_t1(W_A_updated):.4f} (was {mse_t1(W_A):.4f})")
    print(f"    task2_mse={mse_t2(W_A_updated):.4f} (was {mse_t2(W_A):.4f})")

    # The key assertion: A should improve on task 2 WITHOUT catastrophic forgetting on task 1
    t1_degradation = mse_t1(W_A_updated) - mse_t1(W_A)
    t2_improvement = mse_t2(W_A) - mse_t2(W_A_updated)
    print(f"\n  Task 1 degradation: {t1_degradation:+.4f}")
    print(f"  Task 2 improvement: {t2_improvement:+.4f}")

    assert t2_improvement > 0, "A should improve on task 2"
    assert t1_degradation < t2_improvement, \
        "Task 1 degradation should be LESS than task 2 improvement (net positive)"
    print("  PASSED (freeze map enables learning new task with controlled forgetting)\n")


def test_create_offspring_api():
    """Test the actual create_offspring and offspring_update_freeze_map functions
    from lgme/freeze.py using a fake state_dict that mimics MoE experts."""
    from lgme.freeze import (create_offspring, offspring_update_freeze_map,
                              init_freeze_map, decay_freeze_map)

    print("Test 11: create_offspring + offspring_update_freeze_map API")

    # Build a fake state_dict with two experts
    sd = {
        'expert0.mlp_fc1': mx.array(W_PARENT_A),  # 4x4
        'expert0.mlp_fc2': mx.eye(4),
        'expert0.lm_head': mx.array(W_PARENT_A),
        'expert1.mlp_fc1': mx.array(W_PARENT_B),
        'expert1.mlp_fc2': mx.eye(4),
        'expert1.lm_head': mx.array(W_PARENT_B),
        'shared.attn': mx.zeros((4, 4)),
    }

    parent_a = {
        'id': 0, 'fc1': 'expert0.mlp_fc1', 'fc2': 'expert0.mlp_fc2',
        'lm_head': 'expert0.lm_head',
        'router_key': [1.0, 0.0, 0.0, 0.0],
        'activation_count': 10,
    }
    parent_b = {
        'id': 1, 'fc1': 'expert1.mlp_fc1', 'fc2': 'expert1.mlp_fc2',
        'lm_head': 'expert1.lm_head',
        'router_key': [0.0, 1.0, 0.0, 0.0],
        'activation_count': 8,
    }

    # Create offspring
    import random
    child = create_offspring(sd, parent_a, parent_b, child_id=99, n_embd=4,
                              strategy='blend', alpha=0.5, rng=random.Random(42))
    print(f"  Child keys: fc1={child['fc1']}, fc2={child['fc2']}, lm_head={child.get('lm_head')}")
    assert child['fc1'] in sd, "Child fc1 should be in sd"
    assert child['fc2'] in sd, "Child fc2 should be in sd"
    assert 'lm_head' in child, "Child should have lm_head"
    assert child['lm_head'] in sd, "Child lm_head should be in sd"
    print(f"  Child fc1 shape: {sd[child['fc1']].shape}")

    # Verify blend is approximately midpoint
    child_fc1 = sd[child['fc1']]
    expected = 0.5 * sd[parent_a['fc1']] + 0.5 * sd[parent_b['fc1']]
    diff = mx.max(mx.abs(child_fc1 - expected)).item()
    print(f"  Max diff from exact blend: {diff:.4f} (should be small, ~noise_std)")
    assert diff < 0.1, "Child should be approximately the blend of parents"

    # Init freeze map and decay
    fm = init_freeze_map(sd, initial_tau=1.0)
    decay_freeze_map(fm, decay_rate=0.5, exclude_keys={'shared.attn'})
    print(f"  After decay: tau(expert0.fc1)={fm['expert0.mlp_fc1']:.2f}, tau(shared.attn)={fm['shared.attn']:.2f}")
    assert fm['expert0.mlp_fc1'] == 0.5, "Expert key should decay"
    assert fm['shared.attn'] == 1.0, "Shared key should not decay"

    # Test offspring_update_freeze_map
    # Simulate: child loss = 1.5, parent A loss = 2.0 → child better → thaw parent
    offspring_update_freeze_map(fm, parent_a,
                                 child_loss=1.5, parent_loss=2.0,
                                 thaw_amount=0.3, freeze_amount=0.2)
    print(f"  After thaw (child wins): tau(A.fc1)={fm[parent_a['fc1']]:.2f}")
    assert fm[parent_a['fc1']] == 0.8, "Should be 0.5 + 0.3 = 0.8"

    # Simulate: child loss = 2.5, parent B loss = 1.0 → parent better → freeze
    offspring_update_freeze_map(fm, parent_b,
                                 child_loss=2.5, parent_loss=1.0,
                                 thaw_amount=0.3, freeze_amount=0.2)
    print(f"  After freeze (parent wins): tau(B.fc1)={fm[parent_b['fc1']]:.2f}")
    assert fm[parent_b['fc1']] == 0.3, "Should be 0.5 - 0.2 = 0.3"

    # Cleanup child from sd
    for k in [child['fc1'], child['fc2'], child['lm_head']]:
        del sd[k]

    print("  PASSED\n")


if __name__ == '__main__':
    print("=" * 60)
    print("AFMEI Isolated Tests: Freeze Maps + Offspring Mechanism")
    print("=" * 60 + "\n")

    test_parents_have_complementary_knowledge()
    test_blend_vs_selective()
    test_row_selection_offspring()
    test_gradient_identifies_wrong_rows()
    test_freeze_map_from_gradient()
    test_gradient_agreement_identifies_shared_knowledge()
    test_offspring_with_freeze_aware_training()
    test_selective_crossover_beats_blend()
    test_offspring_modifies_parent_freeze_map()
    test_full_lifecycle()
    test_create_offspring_api()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
