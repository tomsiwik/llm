"""Split CIFAR-100 Benchmark with ViT-B/16 + Expert Heads.

Uses pretrained ViT-B/16 backbone (frozen) as feature extractor.
Features are extracted ONCE and cached — each expert is a lightweight
linear head (768 → 100) trained on cached CLS token features.

This is the standard approach used by L2P, CODA-Prompt, DualPrompt, etc.
The lifecycle operates on the lightweight heads (clone, blend, freeze, recycle).

Runs two configurations by default:
  - N_EXPERTS=10 (ceiling reference, 1:1 mapping)
  - N_EXPERTS=5  (lifecycle-critical, 2:1 task-to-expert ratio)

Usage:
    uv run --with mlx,safetensors,huggingface_hub python bench_cifar100_vit.py
    uv run --with mlx,safetensors,huggingface_hub python bench_cifar100_vit.py --quick
    uv run --with mlx,safetensors,huggingface_hub python bench_cifar100_vit.py --experts=5
"""
import math
import os
import sys
import time
import mlx.core as mx
import numpy as np

from tribe.vit import load_vit_backbone, normalize_imagenet, vit_forward
from tribe.cifar100 import load_cifar100, make_split_cifar100
from tribe.core import Tribe, State
from tribe.expert import clone, blend_weights
from tribe.router import SwitchRouter

# ── Config ────────────────────────────────────────────────────
N_TASKS = 10
BATCH_SIZE = 128     # large batch OK with cached features (768-dim)
MAX_GRAD_NORM = 1.0
QUICK_MODE = '--quick' in sys.argv
TRAIN_STEPS = 100 if QUICK_MODE else 500
TRAIN_LR = 0.001    # Adam LR for linear head
JOINT_ALPHA = 0.1    # Load balancing loss coefficient (higher than ResNet: linear heads converge fast)

# Global backbone (loaded once)
BACKBONE = None
FEATURE_DIM = 768

# Parse CLI
N_EXPERTS_CLI = None
for arg in sys.argv[1:]:
    if arg.startswith('--experts='):
        N_EXPERTS_CLI = int(arg.split('=')[1])


def get_backbone():
    global BACKBONE
    if BACKBONE is None:
        print("  Loading ViT-B/16 backbone...")
        BACKBONE = load_vit_backbone()
        n_params = sum(np.prod(BACKBONE[k].shape) for k in BACKBONE)
        print(f"  Backbone loaded: {len(BACKBONE)} tensors, {n_params:,} params")
    return BACKBONE


# ── Expert Head Creation ──────────────────────────────────────

def make_head(seed=0, num_classes=100):
    """Create a lightweight linear head: (768) → (num_classes).

    76,900 params per expert (768*100 + 100).
    Compatible with tribe lifecycle (clone, blend_weights, etc).
    """
    mx.random.seed(seed)
    return {
        'head_w': mx.random.normal((num_classes, FEATURE_DIM)) * 0.01,
        'head_b': mx.zeros((num_classes,)),
    }


def head_forward(weights, features):
    """Linear head forward: (N, 768) → (N, 100)."""
    return features @ weights['head_w'].T + weights['head_b']


def head_param_count(weights):
    return sum(np.prod(weights[k].shape) for k in weights)


# ── Feature Extraction ────────────────────────────────────────

FEATURE_CACHE_DIR = os.path.expanduser("~/.cache/cifar100-vit-features")


def extract_features(images_32, batch_size=64):
    """Extract CLS features from frozen ViT-B/16.

    Args:
        images_32: (N, 32, 32, 3) float32 [0,1] images.
        batch_size: process this many images at a time.

    Returns:
        (N, 768) numpy float32 CLS features.
    """
    backbone = get_backbone()
    N = images_32.shape[0]
    features = np.empty((N, FEATURE_DIM), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = images_32[start:end]
        # Resize 32→224 (nearest neighbor, fast)
        batch_224 = np.repeat(np.repeat(batch, 7, axis=1), 7, axis=2)
        # ImageNet normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        batch_224 = (batch_224 - mean) / std
        # Forward through frozen backbone
        X = mx.array(batch_224)
        cls = vit_forward(backbone, X, return_cls=True)  # (B, 768)
        mx.eval(cls)
        features[start:end] = np.array(cls)

    return features


def extract_all_features(tasks, batch_size=32):
    """Extract features for all tasks, with disk caching.

    First run: ~15 min (ViT-B/16 forward pass on 60K images).
    Subsequent runs: instant (loads from ~/.cache/cifar100-vit-features/).

    Returns:
        list of (train_features, train_labels, test_features, test_labels) tuples.
    """
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(FEATURE_CACHE_DIR, "features_seed42.npz")

    if os.path.exists(cache_file):
        print("  Loading cached ViT features from disk...")
        data = np.load(cache_file)
        task_features = []
        for t in range(len(tasks)):
            task_features.append((
                data[f'train_f_{t}'], data[f'train_l_{t}'],
                data[f'test_f_{t}'], data[f'test_l_{t}'],
            ))
        total_mem = sum(tf.nbytes + tef.nbytes for tf, _, tef, _ in task_features) / 1e6
        print(f"  Loaded {len(task_features)} tasks, {total_mem:.1f} MB")
        return task_features

    print("  Extracting ViT-B/16 features (one-time cost, will be cached)...")
    task_features = []
    cache_data = {}
    for t, (tr, te) in enumerate(tasks):
        t0 = time.time()
        tr_f = extract_features(tr.images, batch_size=batch_size)
        te_f = extract_features(te.images, batch_size=batch_size)
        task_features.append((tr_f, tr.labels, te_f, te.labels))
        cache_data[f'train_f_{t}'] = tr_f
        cache_data[f'train_l_{t}'] = tr.labels
        cache_data[f'test_f_{t}'] = te_f
        cache_data[f'test_l_{t}'] = te.labels
        print(f"    Task {t}: {tr_f.shape} train, {te_f.shape} test ({time.time()-t0:.1f}s)")

    # Save to disk for future runs
    np.savez_compressed(cache_file, **cache_data)
    total_mem = sum(tf.nbytes + tef.nbytes for tf, _, tef, _ in task_features) / 1e6
    print(f"  Saved features to {cache_file} ({total_mem:.1f} MB)")
    return task_features


# ── Loss & Training ───────────────────────────────────────────

def cross_entropy_loss(logits, labels_int):
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    N = logits.shape[0]
    return -mx.mean(log_probs[mx.arange(N), labels_int])


def cosine_lr(step, total_steps, base_lr=TRAIN_LR, min_lr=1e-5):
    progress = step / max(total_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_head(weights, features, labels, steps=TRAIN_STEPS, lr=TRAIN_LR, member=None):
    """Train a linear head on cached ViT features using Adam.

    Args:
        member: optional TribeMember for warmup scaling. If provided and
                warmup_remaining > 0, scales loss by warmup_scale (0→1 ramp).
    """
    N = features.shape[0]
    F = mx.array(features) if isinstance(features, np.ndarray) else features
    L = mx.array(labels) if isinstance(labels, np.ndarray) else labels
    # Adam state
    m = {k: mx.zeros_like(weights[k]) for k in weights}
    v = {k: mx.zeros_like(weights[k]) for k in weights}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(steps):
        step_lr = cosine_lr(step, steps, base_lr=lr)
        idx = mx.array(np.random.randint(0, N, size=min(BATCH_SIZE, N)))
        Fb, Lb = F[idx], L[idx]

        scale = 1.0
        if member is not None and member.warmup_remaining > 0:
            scale = member.warmup_scale

        def loss_fn(w, _scale=scale):
            logits = head_forward(w, Fb)
            return _scale * cross_entropy_loss(logits, Lb)

        loss, grads = mx.value_and_grad(loss_fn)(weights)
        t_adam = step + 1
        for k in weights:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
            m_hat = m[k] / (1 - beta1 ** t_adam)
            v_hat = v[k] / (1 - beta2 ** t_adam)
            weights[k] = weights[k] - step_lr * m_hat / (mx.sqrt(v_hat) + eps)
        mx.eval(*[weights[k] for k in weights])

        if member is not None:
            if member.warmup_remaining > 0:
                member.warmup_remaining -= 1
            member.age += 1


# ── Joint Router-Expert Training ──────────────────────────────

def train_step_joint_vit(router, expert_weights_list, features_batch, labels,
                         lr=TRAIN_LR, alpha=JOINT_ALPHA, trainable_mask=None):
    """Joint router + expert training step for ViT heads via STE.

    Args:
        trainable_mask: list of bool, True=update weights, False=frozen (still routes).
                        If None, all experts are trainable.
    """
    num_experts = len(expert_weights_list)

    all_params = {'router': router.weights}
    for i, ew in enumerate(expert_weights_list):
        all_params[f'expert_{i}'] = ew

    def joint_loss(params):
        logits_r = features_batch @ params['router']['router_w'].T + params['router']['router_b']
        probs = mx.softmax(logits_r, axis=-1)

        # STE: hard assignment with soft gradient
        hard = mx.stop_gradient(mx.eye(num_experts)[mx.argmax(probs, axis=-1)])
        weights = hard - mx.stop_gradient(probs) + probs  # (N, E)

        # Run all experts
        expert_outs = []
        for i in range(num_experts):
            out = head_forward(params[f'expert_{i}'], features_batch)
            expert_outs.append(out)
        stacked = mx.stack(expert_outs, axis=1)  # (N, E, C)

        mixed = mx.sum(stacked * mx.expand_dims(weights, -1), axis=1)  # (N, C)
        task_loss = cross_entropy_loss(mixed, labels)

        # Load balance loss
        f = mx.mean(hard, axis=0)
        P = mx.mean(probs, axis=0)
        aux = num_experts * mx.sum(f * P)

        return task_loss + alpha * aux

    loss, grads = mx.value_and_grad(joint_loss)(all_params)

    # Update router
    for k in router.weights:
        router.weights[k] = router.weights[k] - lr * grads['router'][k]

    # Update experts (skip frozen)
    for i, ew in enumerate(expert_weights_list):
        if trainable_mask is not None and not trainable_mask[i]:
            continue
        eg = grads[f'expert_{i}']
        for k in ew:
            ew[k] = ew[k] - lr * eg[k]

    mx.eval(*[router.weights[k] for k in router.weights])
    for ew in expert_weights_list:
        mx.eval(*[ew[k] for k in ew])

    return loss.item()


def batch_train_joint_vit(router, expert_weights_list, features, labels,
                          steps=TRAIN_STEPS, lr=TRAIN_LR, alpha=JOINT_ALPHA,
                          trainable_mask=None):
    """Mini-batch joint router-expert training loop for ViT heads."""
    N = features.shape[0]
    F = mx.array(features) if isinstance(features, np.ndarray) else features
    L = mx.array(labels) if isinstance(labels, np.ndarray) else labels

    for step in range(steps):
        step_lr = cosine_lr(step, steps, base_lr=lr)
        idx = mx.array(np.random.randint(0, N, size=min(BATCH_SIZE, N)))
        train_step_joint_vit(router, expert_weights_list, F[idx], L[idx],
                             lr=step_lr, alpha=alpha, trainable_mask=trainable_mask)


# ── Accuracy Measurement ──────────────────────────────────────

def measure_accuracy_head(weights, features, labels):
    """Accuracy of a linear head on cached features."""
    F = mx.array(features) if isinstance(features, np.ndarray) else features
    logits = head_forward(weights, F)
    mx.eval(logits)
    predicted = np.argmax(np.array(logits), axis=1)
    return float((predicted == np.array(labels)).mean())


def measure_accuracy_tribe_oracle(expert_list, features, labels):
    """Oracle-routed: pick best expert per sample by CE loss."""
    if not expert_list:
        return 0.0
    F = mx.array(features) if isinstance(features, np.ndarray) else features
    L_np = np.array(labels)
    N = len(L_np)
    best_pred = np.zeros(N, dtype=np.int32)
    best_loss = np.full(N, np.inf, dtype=np.float32)

    for hw in expert_list:
        logits = head_forward(hw, F)
        mx.eval(logits)
        logits_np = np.array(logits)
        lse = np.log(np.sum(np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True)), axis=1)) + np.max(logits_np, axis=1)
        log_probs = logits_np - lse[:, None]
        sample_losses = -log_probs[np.arange(N), L_np]
        preds = np.argmax(logits_np, axis=1)
        better = sample_losses < best_loss
        best_loss[better] = sample_losses[better]
        best_pred[better] = preds[better]

    return float((best_pred == L_np).mean())


def measure_accuracy_tribe_maxconf(expert_list, features, labels, expert_classes=None):
    """Class-incremental routing: assemble global logits from expert specialists.

    If expert_classes is provided, each expert contributes only its trained
    class logits to a global 100-class logit vector. Prediction = global argmax.
    Falls back to max-confidence routing if expert_classes is None.
    """
    if not expert_list:
        return 0.0
    F = mx.array(features) if isinstance(features, np.ndarray) else features
    L_np = np.array(labels)
    N = len(L_np)

    if expert_classes is not None:
        # Global argmax: each expert fills in its classes' logits
        global_logits = np.full((N, 100), -1e9, dtype=np.float32)
        for ei, hw in enumerate(expert_list):
            logits = head_forward(hw, F)
            mx.eval(logits)
            logits_np = np.array(logits)
            if ei < len(expert_classes) and expert_classes[ei] is not None:
                cls_ids = expert_classes[ei]
                global_logits[:, cls_ids] = logits_np[:, cls_ids]
        preds = np.argmax(global_logits, axis=1)
        return float((preds == L_np).mean())

    # Fallback: max-confidence routing
    best_pred = np.zeros(N, dtype=np.int32)
    best_conf = np.full(N, -1.0, dtype=np.float32)
    for hw in expert_list:
        logits = head_forward(hw, F)
        mx.eval(logits)
        logits_np = np.array(logits)
        exp_l = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        conf = np.max(probs, axis=1)
        preds = np.argmax(logits_np, axis=1)
        better = conf > best_conf
        best_conf[better] = conf[better]
        best_pred[better] = preds[better]
    return float((best_pred == L_np).mean())


def measure_accuracy_learned_vit(router, expert_list, features, labels):
    """Learned router evaluation: route via SwitchRouter. No labels needed."""
    if not expert_list or router is None:
        return 0.0
    F = mx.array(features) if isinstance(features, np.ndarray) else features
    L_np = np.array(labels)
    N = len(L_np)

    # Features are already 768-dim flat — route directly
    assignments, _, _ = router.route(F)

    predicted = np.full(N, -1, dtype=np.int32)
    for eid, indices in assignments.items():
        if eid >= len(expert_list) or not indices:
            continue
        hw = expert_list[eid]
        idx_arr = mx.array(indices)
        F_sub = F[idx_arr]
        logits = head_forward(hw, F_sub)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        preds_np = np.array(preds)
        for j, si in enumerate(indices):
            predicted[si] = preds_np[j]

    # Unrouted samples: max-confidence fallback
    unrouted = np.where(predicted == -1)[0]
    if len(unrouted) > 0:
        F_unr = F[mx.array(unrouted)]
        best_conf = np.full(len(unrouted), -1.0)
        best_pred = np.zeros(len(unrouted), dtype=np.int32)
        for hw in expert_list:
            logits = head_forward(hw, F_unr)
            probs = mx.softmax(logits, axis=-1)
            conf = mx.max(probs, axis=-1)
            pred = mx.argmax(logits, axis=-1)
            mx.eval(conf, pred)
            conf_np = np.array(conf)
            pred_np = np.array(pred)
            better = conf_np > best_conf
            best_conf[better] = conf_np[better]
            best_pred[better] = pred_np[better]
        for j, si in enumerate(unrouted):
            predicted[si] = best_pred[j]

    return float((predicted == L_np).mean())


# ── CL Metrics ──────────────────────────────────────────────

def compute_cl_metrics(acc_matrix):
    T = len(acc_matrix)
    fa = np.mean(acc_matrix[-1])
    bwt = sum(acc_matrix[-1][j] - acc_matrix[j][j] for j in range(T - 1)) / max(T - 1, 1)
    forgetting = 0.0
    for j in range(T - 1):
        max_acc_j = max(acc_matrix[i][j] for i in range(j, T))
        forgetting += max_acc_j - acc_matrix[-1][j]
    forgetting /= max(T - 1, 1)
    return {'FA': fa, 'BWT': bwt, 'Forgetting': forgetting}


# ── Lifecycle Metrics ─────────────────────────────────────────

def compute_expert_similarity(weight_list):
    """Mean pairwise cosine similarity of head weight vectors (redundancy proxy)."""
    if len(weight_list) < 2:
        return 0.0
    vecs = []
    for w in weight_list:
        flat = np.concatenate([np.array(w[k]).flatten() for k in sorted(w.keys())])
        vecs.append(flat)
    vecs = np.array(vecs)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs = vecs / norms
    sim = vecs @ vecs.T
    n = len(vecs)
    total = sum(sim[i, j] for i in range(n) for j in range(i + 1, n))
    count = n * (n - 1) // 2
    return total / max(count, 1)


def print_lifecycle_metrics(tribe, n_experts):
    """Print lifecycle-specific metrics for the tribe."""
    print(f"\n  Lifecycle Metrics ({n_experts} experts):")

    # State counts
    n_active = len(tribe.active_members())
    n_frozen = len(tribe.frozen_members())
    n_recycled = sum(1 for m in tribe.members.values() if m.state == State.RECYCLED)
    print(f"    Final state: {n_active} active, {n_frozen} frozen, {n_recycled} recycled")

    # Generational stats
    gen_counts = {}
    for m in tribe.members.values():
        if m.state != State.RECYCLED:
            gen_counts[m.generation] = gen_counts.get(m.generation, 0) + 1
    if gen_counts:
        gen_str = ", ".join(f"gen{g}={c}" for g, c in sorted(gen_counts.items()))
        print(f"    Generations: {gen_str}")

    # Expert weight similarity
    routable = tribe.routable_members()
    if routable:
        sim = compute_expert_similarity([m.weights for m in routable])
        print(f"    Weight similarity: {sim:.3f} (0=orthogonal, 1=identical)")

    # History summary
    event_types = {}
    for _, event in tribe.history:
        etype = event.split()[0]
        event_types[etype] = event_types.get(etype, 0) + 1
    if event_types:
        ev_str = ", ".join(f"{k}={v}" for k, v in sorted(event_types.items()))
        print(f"    Events: {ev_str}")
    else:
        print("    Events: none")


# ── Methods ───────────────────────────────────────────────────

def run_single_head(task_features):
    """Baseline: single linear head, continual fine-tuning."""
    print(f"\n  [1] SINGLE HEAD (continual fine-tune)")
    head = make_head(seed=0)
    print(f"      Head params: {head_param_count(head):,}")

    acc_matrix = []
    for t, (train_f, train_l, _, _) in enumerate(task_features):
        t0 = time.time()
        train_head(head, train_f, train_l)
        row = [measure_accuracy_head(head, te_f, te_l)
               for _, _, te_f, te_l in task_features]
        acc_matrix.append(row)
        print(f"      Task {t}: acc={row[t]:.1%}, elapsed={time.time()-t0:.1f}s")

    return acc_matrix


def run_static_tribe(task_features, n_experts=10):
    """Static tribe: round-robin assignment, no lifecycle.

    With n_experts < n_tasks, later tasks overwrite earlier experts.
    """
    print(f"\n  [2] STATIC TRIBE ({n_experts} experts, round-robin)")
    tribe = Tribe(fwd=head_forward, make_expert_fn=make_head)
    for i in range(n_experts):
        tribe.add_member(make_head(seed=i * 42))
    for i in range(n_experts):
        tribe.connect(i, (i + 1) % n_experts)

    # Track which classes each member was last trained on
    member_classes = {}

    acc_oracle = []
    acc_maxconf = []

    for t, (train_f, train_l, _, _) in enumerate(task_features):
        t0 = time.time()
        tribe.generation = t
        eid = t % n_experts
        m = tribe.members[eid]
        train_head(m.weights, train_f, train_l)
        member_classes[m.id] = np.unique(train_l)

        # Eval with routable experts
        routable = tribe.routable_members()
        weights_list = [rm.weights for rm in routable]
        cls_list = [member_classes.get(rm.id) for rm in routable]

        row_oracle = [measure_accuracy_tribe_oracle(weights_list, te_f, te_l)
                      for _, _, te_f, te_l in task_features]
        row_maxconf = [measure_accuracy_tribe_maxconf(weights_list, te_f, te_l, cls_list)
                       for _, _, te_f, te_l in task_features]
        acc_oracle.append(row_oracle)
        acc_maxconf.append(row_maxconf)
        print(f"      Task {t}: oracle={row_oracle[t]:.1%}, maxconf={row_maxconf[t]:.1%}, "
              f"elapsed={time.time()-t0:.1f}s")

    return acc_oracle, acc_maxconf, tribe


def run_lifecycle_tribe(task_features, n_experts=10):
    """Lifecycle tribe: round-robin + freeze/recycle management.

    Uses Tribe class with dynamic freeze threshold and warmup integration.
    With n_experts < n_tasks, lifecycle must manage expert reuse.
    """
    print(f"\n  [3] LIFECYCLE TRIBE ({n_experts} experts, dynamic freeze + warmup)")
    tribe = Tribe(fwd=head_forward, make_expert_fn=lambda seed=0: make_head(seed=seed))
    for i in range(n_experts):
        tribe.add_member(make_head(seed=i * 42))
    for i in range(n_experts):
        tribe.connect(i, (i + 1) % n_experts)

    # Track metadata per member
    member_task = {}    # mid → last task_id trained on
    member_classes = {} # mid → class array
    expert_accs = []    # all per-task accuracies (for dynamic threshold)

    acc_oracle = []
    acc_maxconf = []

    for t, (train_f, train_l, _, _) in enumerate(task_features):
        t0 = time.time()
        tribe.generation = t
        events = []

        # ── Just-in-time freeze: protect good expert BEFORE overwrite ──
        # Only matters when t >= n_experts (second pass through experts)
        eid = t % n_experts
        m = tribe.members[eid]

        if t >= n_experts and m.state == State.ACTIVE and m.id in member_task:
            # This expert is about to be overwritten — should we freeze it?
            n_active = len(tribe.active_members())
            min_active = max(n_experts // 2, 2)
            if n_active > min_active and len(expert_accs) >= 3:
                threshold = np.mean(expert_accs) + 0.5 * np.std(expert_accs)
                tid = member_task[m.id]
                _, _, te_f, te_l = task_features[tid]
                acc = measure_accuracy_head(m.weights, te_f, te_l)
                if acc >= threshold:
                    tribe.freeze_member(m.id)
                    events.append(f"FREEZE {m.id} (acc={acc:.1%}, thr={threshold:.1%})")

        # ── Assign expert ──
        m = tribe.members[eid]  # re-read (may be frozen now)

        if m.state == State.FROZEN:
            # Find a fresh expert (never trained)
            fresh = [mid for mid, mem in tribe.members.items()
                     if mem.state == State.ACTIVE and mid not in member_task]
            if fresh:
                eid = fresh[0]
                events.append(f"REDIRECT {t%n_experts}→{eid} (target frozen, using fresh)")
            else:
                # Recycle worst active expert
                active = tribe.active_members()
                if active:
                    worst_acc = 1.0
                    worst_mid = active[0].id
                    for am in active:
                        if am.id in member_task:
                            tid = member_task[am.id]
                            _, _, te_f, te_l = task_features[tid]
                            acc = measure_accuracy_head(am.weights, te_f, te_l)
                            if acc < worst_acc:
                                worst_acc = acc
                                worst_mid = am.id
                        else:
                            worst_mid = am.id
                            worst_acc = 0.0
                            break
                    new_m = tribe.recycle(worst_mid, warmup_steps=50)
                    events.append(f"RECYCLE {worst_mid} (acc={worst_acc:.1%})")
                    eid = worst_mid
            m = tribe.members[eid]

        # ── Train with warmup integration ──
        train_head(m.weights, train_f, train_l, member=m)
        member_task[m.id] = t
        member_classes[m.id] = np.unique(train_l)

        # Measure this expert's accuracy for threshold computation
        _, _, te_f, te_l = task_features[t]
        task_acc = measure_accuracy_head(m.weights, te_f, te_l)
        expert_accs.append(task_acc)

        # Eval with routable experts
        routable = tribe.routable_members()
        weights_list = [rm.weights for rm in routable]
        cls_list = [member_classes.get(rm.id) for rm in routable]

        row_oracle = [measure_accuracy_tribe_oracle(weights_list, te_f, te_l)
                      for _, _, te_f, te_l in task_features]
        row_maxconf = [measure_accuracy_tribe_maxconf(weights_list, te_f, te_l, cls_list)
                       for _, _, te_f, te_l in task_features]
        acc_oracle.append(row_oracle)
        acc_maxconf.append(row_maxconf)

        n_active = len(tribe.active_members())
        n_frozen = len(tribe.frozen_members())
        print(f"      Task {t}: oracle={row_oracle[t]:.1%}, maxconf={row_maxconf[t]:.1%}, "
              f"active={n_active}, frozen={n_frozen}, events={events}, "
              f"elapsed={time.time()-t0:.1f}s")

    return acc_oracle, acc_maxconf, tribe


def _lifecycle_freeze_dynamic(tribe, task_features, member_task, expert_accs,
                              events, n_experts, current_task=0):
    """Freeze experts exceeding dynamic threshold: mean + 0.5 * std.

    Guards:
      - Don't freeze until all experts have been trained once (need stable stats).
      - Keep at least 2/3 of experts active (more conservative than n//2).
    """
    n_active = len(tribe.active_members())
    min_active = max(n_experts - n_experts // 3, 2)
    if n_active <= min_active:
        return

    # Don't freeze until all expert slots have been trained at least once
    if current_task < n_experts:
        return

    if len(expert_accs) < 3:
        return
    threshold = np.mean(expert_accs) + 0.5 * np.std(expert_accs)

    for m in list(tribe.active_members()):
        if m.id not in member_task:
            continue
        tid = member_task[m.id]
        _, _, te_f, te_l = task_features[tid]
        acc = measure_accuracy_head(m.weights, te_f, te_l)
        if acc >= threshold and n_active > min_active:
            tribe.freeze_member(m.id)
            events.append(f"FREEZE {m.id} (acc={acc:.1%}, thr={threshold:.1%})")
            n_active -= 1


def run_learned_lifecycle(task_features, n_experts=10):
    """SwitchRouter-based routing + lifecycle management.

    Joint training: router and experts train together via STE.
    Lifecycle: freeze/recycle based on dynamic threshold.
    """
    print(f"\n  [4] LEARNED ROUTER + LIFECYCLE ({n_experts} experts, joint training)")
    tribe = Tribe(fwd=head_forward, make_expert_fn=lambda seed=0: make_head(seed=seed))
    for i in range(n_experts):
        tribe.add_member(make_head(seed=i * 42))
    for i in range(n_experts):
        tribe.connect(i, (i + 1) % n_experts)

    router = SwitchRouter(input_dim=FEATURE_DIM, num_experts=n_experts, top_k=1)
    tribe.set_router(router)

    member_task = {}
    member_classes = {}
    expert_accs = []

    acc_oracle = []
    acc_maxconf = []
    acc_learned = []

    for t, (train_f, train_l, _, _) in enumerate(task_features):
        t0 = time.time()
        tribe.generation = t
        events = []

        # Joint training: all routable experts (active + frozen)
        # Frozen experts participate in routing but don't update weights
        routable_sorted = sorted(tribe.routable_members(), key=lambda m_: m_.id)
        expert_weights = [m_.weights for m_ in routable_sorted]
        trainable_mask = [m_.is_trainable for m_ in routable_sorted]

        # Router must match number of routable experts
        if len(routable_sorted) != router.num_experts:
            router = SwitchRouter(input_dim=FEATURE_DIM,
                                  num_experts=len(routable_sorted), top_k=1)
            tribe.set_router(router)

        batch_train_joint_vit(router, expert_weights, train_f, train_l,
                              steps=TRAIN_STEPS, lr=TRAIN_LR,
                              trainable_mask=trainable_mask)

        # Track which expert got most data (for task association)
        F = mx.array(train_f) if isinstance(train_f, np.ndarray) else train_f
        assignments, _, stats = router.route(F)
        best_eid = max(assignments.keys(), key=lambda k: len(assignments[k]))
        if best_eid < len(routable_sorted):
            m = routable_sorted[best_eid]
            member_task[m.id] = t
            member_classes[m.id] = np.unique(train_l)

            # Measure accuracy for dynamic threshold
            _, _, te_f, te_l = task_features[t]
            task_acc = measure_accuracy_head(m.weights, te_f, te_l)
            expert_accs.append(task_acc)

        # Lifecycle
        if t > 0:
            _lifecycle_freeze_dynamic(tribe, task_features, member_task,
                                      expert_accs, events, n_experts,
                                      current_task=t)

        # Eval
        routable = tribe.routable_members()
        weights_list = [rm.weights for rm in routable]
        cls_list = [member_classes.get(rm.id) for rm in routable]

        row_oracle = [measure_accuracy_tribe_oracle(weights_list, te_f, te_l)
                      for _, _, te_f, te_l in task_features]
        row_maxconf = [measure_accuracy_tribe_maxconf(weights_list, te_f, te_l, cls_list)
                       for _, _, te_f, te_l in task_features]
        # Learned routing: use sorted routable weights (matches router index order)
        row_learned = [measure_accuracy_learned_vit(router, weights_list, te_f, te_l)
                       for _, _, te_f, te_l in task_features]

        acc_oracle.append(row_oracle)
        acc_maxconf.append(row_maxconf)
        acc_learned.append(row_learned)

        print(f"      Task {t}: oracle={row_oracle[t]:.1%}, maxconf={row_maxconf[t]:.1%}, "
              f"learned={row_learned[t]:.1%}, counts={stats['counts']}, "
              f"events={events}, elapsed={time.time()-t0:.1f}s")

    return acc_oracle, acc_maxconf, acc_learned, tribe


# ── Reporting ─────────────────────────────────────────────────

def print_comparison(results, n_experts):
    print(f"\n  {'='*72}")
    print(f"  RESULTS — {N_TASKS} tasks, {n_experts} experts, "
          f"{'QUICK' if QUICK_MODE else 'FULL'} ({TRAIN_STEPS} steps)")
    print(f"  {'='*72}")

    print(f"\n  Oracle Routing (loss-based, uses ground truth labels):")
    print(f"  {'Method':<35s} {'FA':>7s} {'BWT':>7s} {'Fgt':>7s}")
    print("  " + "-" * 56)
    for name, acc_matrix, _, _, *_ in results:
        m = compute_cl_metrics(acc_matrix)
        print(f"  {name:<35s} {m['FA']:>7.1%} {m['BWT']:>+7.1%} {m['Forgetting']:>7.1%}")

    has_maxconf = any(mc is not None for _, _, mc, _, *_ in results)
    if has_maxconf:
        print(f"\n  Class-Incremental Routing (label-free, global argmax):")
        print(f"  {'Method':<35s} {'FA':>7s} {'BWT':>7s} {'Fgt':>7s}")
        print("  " + "-" * 56)
        for name, _, acc_mc, _, *_ in results:
            if acc_mc is not None:
                m = compute_cl_metrics(acc_mc)
                print(f"  {name:<35s} {m['FA']:>7.1%} {m['BWT']:>+7.1%} {m['Forgetting']:>7.1%}")

    has_learned = any(len(r) > 4 and r[4] is not None for r in results)
    if has_learned:
        print(f"\n  Learned Router (SwitchRouter, label-free):")
        print(f"  {'Method':<35s} {'FA':>7s} {'BWT':>7s} {'Fgt':>7s}")
        print("  " + "-" * 56)
        for r in results:
            if len(r) > 4 and r[4] is not None:
                m = compute_cl_metrics(r[4])
                print(f"  {r[0]:<35s} {m['FA']:>7.1%} {m['BWT']:>+7.1%} {m['Forgetting']:>7.1%}")

    print(f"\n  Per-task final accuracy — Oracle (after all {N_TASKS} tasks):")
    header = f"  {'Method':<35s}" + "".join(f" T{j:>2}" for j in range(N_TASKS))
    print(header)
    print("  " + "-" * (35 + 4 * N_TASKS))
    for name, acc_matrix, _, _, *_ in results:
        final = acc_matrix[-1]
        vals = "".join(f" {v:3.0%}" for v in final)
        print(f"  {name:<35s}{vals}")


# ── Main ──────────────────────────────────────────────────────

def run_benchmark():
    t_start = time.time()
    mode = "QUICK" if QUICK_MODE else "FULL"
    print("\n" + "=" * 80)
    print(f"  SPLIT CIFAR-100 + ViT-B/16 BENCHMARK ({mode})")
    print(f"  {N_TASKS} tasks, {TRAIN_STEPS} steps, lr={TRAIN_LR}, batch={BATCH_SIZE}")
    print("=" * 80)

    # Load backbone
    get_backbone()

    # Load CIFAR-100
    train_imgs, train_labels, test_imgs, test_labels = load_cifar100()
    print(f"  CIFAR-100: {len(train_imgs)} train, {len(test_imgs)} test")

    tasks = make_split_cifar100(train_imgs, train_labels, test_imgs, test_labels,
                                n_tasks=N_TASKS, seed=42)
    for t, (tr, te) in enumerate(tasks):
        print(f"  Task {t}: classes {tr.classes} ({len(tr)} train, {len(te)} test)")

    # Extract features ONCE (cached to disk after first run)
    task_features = extract_all_features(tasks, batch_size=32)

    # Determine expert configurations to run
    if N_EXPERTS_CLI is not None:
        expert_configs = [N_EXPERTS_CLI]
    else:
        expert_configs = [10, 5]

    all_results = {}

    # Single head baseline (independent of expert count, run once)
    acc_single = run_single_head(task_features)

    for n_exp in expert_configs:
        print(f"\n{'='*80}")
        print(f"  === N_EXPERTS = {n_exp} ===")
        print(f"{'='*80}")

        results = []
        results.append(("Single Head (fine-tune)", acc_single, None, None))

        s_or, s_mc, tribe_s = run_static_tribe(task_features, n_experts=n_exp)
        results.append((f"Static Tribe ({n_exp}e)", s_or, s_mc, None))

        lc_or, lc_mc, tribe_lc = run_lifecycle_tribe(task_features, n_experts=n_exp)
        results.append((f"Lifecycle Tribe ({n_exp}e)", lc_or, lc_mc, None))

        lr_or, lr_mc, lr_lr, tribe_lr = run_learned_lifecycle(task_features, n_experts=n_exp)
        results.append((f"Learned+LC ({n_exp}e)", lr_or, lr_mc, None, lr_lr))

        print_comparison(results, n_exp)
        print_lifecycle_metrics(tribe_lc, n_exp)

        all_results[n_exp] = results

    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} minutes")
    return all_results


if __name__ == "__main__":
    run_benchmark()
