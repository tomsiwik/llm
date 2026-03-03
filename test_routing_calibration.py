"""Tests for tribe/routing_calibration.py — label ordering bug fix + calibration.

Validates:
  1. Domain ordering uses library._labels (registration order), not sorted()
  2. Calibration trains routing keys that discriminate between domains
  3. Evaluate accuracy uses correct domain→expert mapping
  4. Mismatched features raise ValueError
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from tribe.lora_library import SelfRoutingLoRALibrary, collect_library_layers
from tribe.routing_calibration import (
    calibrate_routing_keys,
    evaluate_routing_accuracy,
)


# ── Helpers ─────────────────────────────────────────────────

def make_library(labels, d_in=32, rank=8, d_out=16, top_k=1):
    """Build a SelfRoutingLoRALibrary with given labels in order."""
    W = mx.random.normal((d_out, d_in)) * 0.01
    lib = SelfRoutingLoRALibrary(base_weight=W, top_k=top_k)
    for label in labels:
        A = mx.random.normal((d_in, rank)) * 0.1
        B = mx.random.normal((rank, d_out)) * 0.1
        lib.register_expert(A, B, label=label)
    mx.eval(lib.parameters())
    return lib


def make_separable_features(labels, d_in=32, n_tokens=200, seed=42):
    """Create features where each domain has a distinct signal direction.

    Domain i gets signal along dimension i (one-hot spike + noise).
    This makes routing trivially separable if keys learn the right
    direction for the right expert index.
    """
    rng = np.random.RandomState(seed)
    n_layers = 2  # simulate 2 transformer layers
    features = {}
    for i, label in enumerate(labels):
        layer_feats = {}
        for layer_idx in range(n_layers):
            # Strong signal on dimension i, noise elsewhere
            x = rng.randn(n_tokens, d_in).astype(np.float32) * 0.1
            x[:, i] += 3.0  # strong distinguishing signal
            layer_feats[layer_idx] = mx.array(x)
            mx.eval(layer_feats[layer_idx])
        features[label] = layer_feats
    return features


class FakeModel(nn.Module):
    """Minimal model wrapper that holds library layers at known paths."""

    def __init__(self, libraries):
        super().__init__()
        self.model = FakeModelInner(libraries)


class FakeModelInner(nn.Module):
    def __init__(self, libraries):
        super().__init__()
        self.layers = [FakeLayer(lib) for lib in libraries]


class FakeLayer(nn.Module):
    def __init__(self, lib):
        super().__init__()
        self.self_attn = FakeAttn(lib)


class FakeAttn(nn.Module):
    def __init__(self, lib):
        super().__init__()
        self.q_proj = lib


def build_fake_model(labels, d_in=32, rank=8, d_out=16):
    """Build a FakeModel with library layers at model.layers.N.self_attn.q_proj.

    Returns (model, [lib_0, lib_1, ...]).
    """
    libs = []
    for _ in range(2):  # 2 transformer layers
        lib = make_library(labels, d_in=d_in, rank=rank, d_out=d_out)
        libs.append(lib)
    model = FakeModel(libs)
    mx.eval(model.parameters())
    return model, libs


# ── Tests ───────────────────────────────────────────────────

def test_labels_order_preserved():
    """Library._labels preserves registration order, not sorted."""
    labels = ["python", "javascript", "go"]
    lib = make_library(labels)
    assert lib._labels == labels
    assert lib._labels != sorted(labels)  # "go" < "javascript" < "python"


def test_calibrate_uses_library_labels_not_sorted():
    """The bug: sorted() would assign label 0 to 'go' but expert_0 is 'python'.

    After the fix, domain_to_idx must match registration order.
    We verify this by:
    1. Creating a library with labels in non-sorted order
    2. Making features where domain i has signal on dimension i
    3. After calibration, routing_key_i should fire strongest on domain i features
    """
    labels = ["python", "javascript", "go"]  # NOT sorted (sorted = go, javascript, python)
    d_in = 32

    model, libs = build_fake_model(labels, d_in=d_in)
    features = make_separable_features(labels, d_in=d_in)

    # Initialize routing keys and calibrate
    for lib in libs:
        lib.initialize_routing_keys(d_key=8, init_from_A=True)
    mx.eval(model.parameters())

    calibrate_routing_keys(model, features, steps=30, lr=1e-2,
                           temperature=0.1, verbose=False)

    # After calibration, routing_key_0 should fire on "python" (dim 0),
    # routing_key_1 on "javascript" (dim 1), routing_key_2 on "go" (dim 2).
    # If the old sorted() bug were present, key_0 would fire on "go" (dim 2).
    lib = libs[0]
    for i, label in enumerate(labels):
        K = getattr(lib, f"routing_key_{i}")
        mx.eval(K)
        # Create a test vector with signal only on dimension i
        x_test = mx.zeros((1, d_in))
        x_test = x_test.at[:, i].add(3.0)
        scores, _ = lib._score_experts(x_test)
        mx.eval(scores)
        selected = int(mx.argmax(scores, axis=-1).item())
        assert selected == i, (
            f"routing_key_{i} (label='{label}') should fire on dim-{i} signal, "
            f"but argmax selected expert {selected}"
        )


def test_evaluate_uses_library_labels_not_sorted():
    """evaluate_routing_accuracy must use _labels order for domain→expert mapping.

    If it used sorted(), 'go' tokens (dim 2) would be expected to match expert_0
    (wrong), and accuracy would be near 0 for correctly calibrated keys.
    """
    labels = ["python", "javascript", "go"]
    d_in = 32

    model, libs = build_fake_model(labels, d_in=d_in)
    features = make_separable_features(labels, d_in=d_in)

    # Calibrate
    for lib in libs:
        lib.initialize_routing_keys(d_key=8, init_from_A=True)
    mx.eval(model.parameters())
    calibrate_routing_keys(model, features, steps=30, lr=1e-2,
                           temperature=0.1, verbose=False)

    # Evaluate on same features (training set, so should be high)
    result = evaluate_routing_accuracy(model, features, verbose=False)
    acc = result["mean_accuracy"]

    # With correct label alignment, accuracy should be high (>80%)
    # With the old sorted() bug it would be ~33% (random)
    assert acc > 0.70, (
        f"Routing accuracy {acc:.1%} is too low — label ordering may be wrong"
    )


def test_calibrate_missing_domain_raises():
    """If features don't cover all library domains, raise ValueError."""
    labels = ["python", "javascript", "go"]
    d_in = 32

    model, libs = build_fake_model(labels, d_in=d_in)
    # Features missing "go"
    features = make_separable_features(["python", "javascript"], d_in=d_in)

    for lib in libs:
        lib.initialize_routing_keys(d_key=8, init_from_A=True)
    mx.eval(model.parameters())

    with pytest.raises(ValueError, match="No features for domain 'go'"):
        calibrate_routing_keys(model, features, steps=5, verbose=False)


def test_evaluate_no_library_raises():
    """evaluate_routing_accuracy raises if model has no library layers."""
    model = nn.Linear(16, 8)
    features = {"a": {0: mx.zeros((10, 16))}}

    with pytest.raises(ValueError, match="No SelfRoutingLoRALibrary"):
        evaluate_routing_accuracy(model, features, verbose=False)


def test_calibrate_no_library_raises():
    """calibrate_routing_keys raises if model has no library layers."""
    model = nn.Linear(16, 8)
    features = {"a": {0: mx.zeros((10, 16))}}

    with pytest.raises(ValueError, match="No SelfRoutingLoRALibrary"):
        calibrate_routing_keys(model, features, steps=5, verbose=False)


def test_five_way_calibration():
    """5-domain calibration reaches >80% accuracy on separable features."""
    labels = ["python", "javascript", "go", "rust", "ruby"]
    d_in = 32

    model, libs = build_fake_model(labels, d_in=d_in)

    # Training features (odd-indexed)
    train_features = make_separable_features(labels, d_in=d_in, n_tokens=200, seed=42)
    # Hold-out features (different seed)
    holdout_features = make_separable_features(labels, d_in=d_in, n_tokens=100, seed=99)

    for lib in libs:
        lib.initialize_routing_keys(d_key=8, init_from_A=True)
    mx.eval(model.parameters())

    # Calibrate on training features
    losses = calibrate_routing_keys(model, train_features, steps=50, lr=1e-2,
                                    temperature=0.1, verbose=False)

    # Loss should decrease
    assert losses[-1] < losses[0], "Calibration loss did not decrease"

    # Evaluate on HOLD-OUT features (not training)
    result = evaluate_routing_accuracy(model, holdout_features, verbose=False)
    acc = result["mean_accuracy"]
    assert acc > 0.80, (
        f"5-way hold-out routing accuracy {acc:.1%} < 80% target"
    )

    # Per-layer results should exist
    assert len(result["per_layer"]) > 0
    for layer_result in result["per_layer"]:
        assert "name" in layer_result
        assert "accuracy" in layer_result
        assert "domain_accs" in layer_result
        # Each domain should have an accuracy entry
        for label in labels:
            assert label in layer_result["domain_accs"]


def test_calibration_loss_decreases():
    """Calibration loss should monotonically decrease (roughly)."""
    labels = ["alpha", "beta"]
    d_in = 32

    model, libs = build_fake_model(labels, d_in=d_in)
    features = make_separable_features(labels, d_in=d_in)

    for lib in libs:
        lib.initialize_routing_keys(d_key=8, init_from_A=True)
    mx.eval(model.parameters())

    losses = calibrate_routing_keys(model, features, steps=20, lr=1e-2,
                                    temperature=0.1, verbose=False)

    # First quarter average should be higher than last quarter
    q1 = np.mean(losses[:5])
    q4 = np.mean(losses[-5:])
    assert q4 < q1, f"Loss did not decrease: first-5 avg={q1:.4f}, last-5 avg={q4:.4f}"


def test_labels_order_regression_two_domains():
    """Regression test for the original 2-domain bug.

    With labels=["python", "javascript"]:
    - Registration order: python=0, javascript=1
    - sorted() order:     javascript=0, python=1  ← WRONG
    The fix ensures expert_0 maps to python, not javascript.
    """
    labels = ["python", "javascript"]  # sorted would be ["javascript", "python"]
    d_in = 32

    model, libs = build_fake_model(labels, d_in=d_in)
    features = make_separable_features(labels, d_in=d_in, seed=123)

    for lib in libs:
        lib.initialize_routing_keys(d_key=8, init_from_A=True)
    mx.eval(model.parameters())

    calibrate_routing_keys(model, features, steps=30, lr=1e-2,
                           temperature=0.1, verbose=False)

    result = evaluate_routing_accuracy(model, features, verbose=False)
    acc = result["mean_accuracy"]

    # With correct ordering: should be high
    # With sorted() bug: python tokens would try to match expert_1 (javascript's slot)
    assert acc > 0.80, (
        f"2-domain accuracy {acc:.1%} too low — possible label ordering regression"
    )


# ── Runner ──────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_labels_order_preserved,
        test_calibrate_uses_library_labels_not_sorted,
        test_evaluate_uses_library_labels_not_sorted,
        test_calibrate_missing_domain_raises,
        test_evaluate_no_library_raises,
        test_calibrate_no_library_raises,
        test_five_way_calibration,
        test_calibration_loss_decreases,
        test_labels_order_regression_two_domains,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n  {passed}/{passed + failed} passed")
    if failed:
        raise SystemExit(1)
