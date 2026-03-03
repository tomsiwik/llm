"""Split CIFAR-100 Benchmark: 10 tasks × 10 classes.

Standard continual learning benchmark (class-incremental setting).
Compares: Fine-tune, EWC, DER++, Static Tribe, Lifecycle Tribe, Learned Router + Lifecycle.

Uses softmax cross-entropy loss (standard for classification).
Reports: Final Average Accuracy (FA), Backward Transfer (BWT), Forgetting (F).
"""

import math
import sys
import time
import mlx.core as mx
import numpy as np

from tribe.resnet import make_resnet_expert, resnet_forward_batch, resnet_forward_hidden
from tribe.cifar100 import load_cifar100, make_split_cifar100, augment_batch
from tribe.router import SwitchRouter
from tribe.expert import orthogonality_loss

# ── Config ──────────────────────────────────────────────────
N_TASKS = 10
N_EXPERTS = 10
EXPERT_WIDTH = 16    # ~712K params per expert
QUICK_MODE = '--quick' in sys.argv
N_SEEDS = None  # set by --seeds=N CLI arg
TRAIN_STEPS = 500 if QUICK_MODE else 2000
TRAIN_LR = 0.001
BATCH_SIZE = 64
FWD = resnet_forward_batch
FWD_HIDDEN = resnet_forward_hidden
MAX_GRAD_NORM = 5.0
ALPHA_ORTH = 0.1    # Orthogonality loss weight
ALPHA_VAR = 0.01    # Variance loss weight


def cosine_lr(step, total_steps, base_lr=TRAIN_LR, min_lr=1e-5):
    """Cosine annealing learning rate schedule."""
    progress = step / max(total_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_expert(seed=0):
    return make_resnet_expert(seed=seed, width=EXPERT_WIDTH)


# ── Loss functions ──────────────────────────────────────────

def cross_entropy_loss(logits, labels_int):
    """Softmax cross-entropy. logits: (N, 100), labels_int: (N,) int."""
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    N = logits.shape[0]
    return -mx.mean(log_probs[mx.arange(N), labels_int])


def clip_and_step(weights, grads, lr):
    """Gradient clipping + SGD step."""
    grad_sq = sum(mx.sum(grads[k] ** 2) for k in grads)
    mx.eval(grad_sq)
    grad_norm = grad_sq.item() ** 0.5
    scale = min(1.0, MAX_GRAD_NORM / max(grad_norm, 1e-8))
    for k in weights:
        weights[k] = weights[k] - lr * scale * grads[k]
    mx.eval(*[weights[k] for k in weights])


# ── Joint router-expert training (Phase 3) ─────────────────

JOINT_TRAINING = True   # Use joint router-expert training in method 6
JOINT_ALPHA = 0.01      # Load balancing loss coefficient


def train_step_joint(router, expert_weights_list, X_images, labels, fwd,
                     lr=TRAIN_LR, alpha=JOINT_ALPHA,
                     fwd_hidden=None, alpha_orth=0.0, alpha_var=0.0):
    """Joint router + expert training step via STE.

    Puts router weights and all expert weights in one dict, computes
    joint loss = cross_entropy(mixed_output, labels) + alpha * load_balance
                 + alpha_orth * orthogonality_loss + alpha_var * variance_loss,
    then updates everything with gradient clipping.

    Args:
        router: SwitchRouter instance.
        expert_weights_list: list of expert weight dicts.
        X_images: (N, H, W, C) image batch.
        labels: (N,) integer labels.
        fwd: expert forward function.
        lr: learning rate.
        alpha: load balancing loss coefficient.
        fwd_hidden: forward function returning hidden representations (before FC).
                    Required when alpha_orth > 0.
        alpha_orth: orthogonality loss weight. 0.0 = disabled.
        alpha_var: variance loss weight. 0.0 = disabled.

    Returns:
        loss: scalar joint loss value (float).
    """
    X_flat = mx.reshape(X_images, (X_images.shape[0], -1))
    num_experts = router.num_experts

    # Combine all params into one dict for value_and_grad
    all_params = {'router': router.weights}
    for i, ew in enumerate(expert_weights_list):
        all_params[f'expert_{i}'] = ew

    def joint_loss(params):
        # Router logits -> probs (differentiable)
        logits = X_flat @ params['router']['router_w'].T + params['router']['router_b']
        probs = mx.softmax(logits, axis=-1)  # (N, E)

        # STE: hard assignment with soft gradient
        hard = mx.stop_gradient(mx.eye(num_experts)[mx.argmax(probs, axis=-1)])
        weights = hard - mx.stop_gradient(probs) + probs  # (N, E)

        # Run all experts on all inputs
        expert_outs = []
        for i in range(len(expert_weights_list)):
            out = fwd(params[f'expert_{i}'], X_images)  # (N, C)
            expert_outs.append(out)
        stacked = mx.stack(expert_outs, axis=1)  # (N, E, C)

        # Weighted mix
        mixed = mx.sum(stacked * mx.expand_dims(weights, -1), axis=1)  # (N, C)

        # Task loss
        task_loss = cross_entropy_loss(mixed, labels)

        # Load balance loss
        f = mx.mean(hard, axis=0)  # fraction routed (hard, no grad)
        P = mx.mean(probs, axis=0)  # mean prob (soft, has grad)
        aux = num_experts * mx.sum(f * P)

        total = task_loss + alpha * aux

        # Orthogonality loss: penalize similar expert representations
        if alpha_orth > 0 and fwd_hidden is not None:
            hiddens = [fwd_hidden(params[f'expert_{i}'], X_images)
                       for i in range(len(expert_weights_list))]
            orth_loss = orthogonality_loss(hiddens)
            total = total + alpha_orth * orth_loss

        # Variance loss: encourage decisive routing
        if alpha_var > 0:
            mean_probs = mx.mean(probs, axis=0)
            variance = mx.mean((mean_probs - mx.mean(mean_probs)) ** 2)
            var_loss = -variance  # negative: maximize variance
            total = total + alpha_var * var_loss

        return total

    loss, grads = mx.value_and_grad(joint_loss)(all_params)

    # Clip and update router weights
    router_grads = grads['router']
    r_grad_sq = sum(mx.sum(router_grads[k] ** 2) for k in router_grads)
    mx.eval(r_grad_sq)
    r_scale = min(1.0, MAX_GRAD_NORM / max(r_grad_sq.item() ** 0.5, 1e-8))
    for k in router.weights:
        router.weights[k] = router.weights[k] - lr * r_scale * router_grads[k]

    # Clip and update each expert's weights
    for i, ew in enumerate(expert_weights_list):
        eg = grads[f'expert_{i}']
        eg_sq = sum(mx.sum(eg[k] ** 2) for k in eg)
        mx.eval(eg_sq)
        e_scale = min(1.0, MAX_GRAD_NORM / max(eg_sq.item() ** 0.5, 1e-8))
        for k in ew:
            ew[k] = ew[k] - lr * e_scale * eg[k]

    # Evaluate all updated weights
    mx.eval(*[router.weights[k] for k in router.weights])
    for ew in expert_weights_list:
        mx.eval(*[ew[k] for k in ew])

    return loss.item()


def batch_train_joint(router, expert_weights_list, X, labels, fwd=FWD,
                      steps=TRAIN_STEPS, lr=TRAIN_LR, alpha=JOINT_ALPHA,
                      fwd_hidden=None, alpha_orth=0.0, alpha_var=0.0):
    """Mini-batch joint router-expert training loop.

    Drop-in replacement for separate batch_train() + router.train_step().
    Optionally includes orthogonality and variance specialization losses.
    """
    N = X.shape[0]
    for step in range(steps):
        step_lr = cosine_lr(step, steps, base_lr=lr)
        idx = mx.array(np.random.randint(0, N, size=min(BATCH_SIZE, N)))
        Xb, Lb = X[idx], labels[idx]
        train_step_joint(router, expert_weights_list, Xb, Lb, fwd,
                         lr=step_lr, alpha=alpha,
                         fwd_hidden=fwd_hidden, alpha_orth=alpha_orth,
                         alpha_var=alpha_var)


# ── Batch training ──────────────────────────────────────────

def batch_train(weights, X, labels, steps=TRAIN_STEPS, lr=TRAIN_LR, member=None, use_aug=False):
    """Mini-batch SGD with cross-entropy, optional augmentation and cosine LR.

    Args:
        member: optional TribeMember for warmup scaling. If provided and
                warmup_remaining > 0, scales loss by warmup_scale (0→1 ramp).
        use_aug: if True, apply data augmentation (random crop + flip) per batch.
    """
    N = X.shape[0]
    # Pre-convert to numpy once if augmenting (X may be mx.array)
    if use_aug:
        X_np = np.array(X) if not isinstance(X, np.ndarray) else X
        L_np = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    for step in range(steps):
        step_lr = cosine_lr(step, steps, base_lr=lr)
        if use_aug:
            idx = np.random.randint(0, N, size=min(BATCH_SIZE, N))
            Xb_np = augment_batch(X_np[idx])
            Xb = mx.array(Xb_np)
            Lb = mx.array(L_np[idx])
        else:
            idx = mx.array(np.random.randint(0, N, size=min(BATCH_SIZE, N)))
            Xb, Lb = X[idx], labels[idx]
        scale = 1.0
        if member is not None and member.warmup_remaining > 0:
            scale = member.warmup_scale
        def loss_fn(w, _scale=scale):
            logits = FWD(w, Xb)
            return _scale * cross_entropy_loss(logits, Lb)
        loss, grads = mx.value_and_grad(loss_fn)(weights)
        clip_and_step(weights, grads, step_lr)
        if member is not None:
            if member.warmup_remaining > 0:
                member.warmup_remaining -= 1
            member.age += 1


def batch_train_ewc(weights, X, labels, fisher, star, lam=400.0,
                     steps=TRAIN_STEPS, lr=TRAIN_LR):
    """Mini-batch training with EWC penalty."""
    N = X.shape[0]
    for step in range(steps):
        idx = mx.array(np.random.randint(0, N, size=min(BATCH_SIZE, N)))
        Xb, Lb = X[idx], labels[idx]
        def loss_fn(w):
            logits = FWD(w, Xb)
            task_loss = cross_entropy_loss(logits, Lb)
            ewc_pen = mx.array(0.0)
            for k in w:
                if k in fisher:
                    ewc_pen = ewc_pen + mx.sum(fisher[k] * (w[k] - star[k]) ** 2)
            return task_loss + (lam / 2.0) * ewc_pen
        loss, grads = mx.value_and_grad(loss_fn)(weights)
        clip_and_step(weights, grads, lr)


def compute_fisher(weights, X, labels, n_samples=200):
    """Diagonal Fisher information estimate."""
    idx = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
    Xs, Ls = X[mx.array(idx)], labels[mx.array(idx)]
    def loss_fn(w):
        logits = FWD(w, Xs)
        return cross_entropy_loss(logits, Ls)
    _, grads = mx.value_and_grad(loss_fn)(weights)
    mx.eval(*[grads[k] for k in grads])
    return {k: grads[k] ** 2 for k in grads}


def batch_train_derpp(weights, X, labels, buffer, alpha=0.5, beta=0.5,
                       steps=TRAIN_STEPS, lr=TRAIN_LR):
    """Mini-batch training with DER++ replay."""
    N = X.shape[0]
    has_buf = buffer['X'] is not None and buffer['X'].shape[0] > 0

    for step in range(steps):
        idx = mx.array(np.random.randint(0, N, size=min(BATCH_SIZE, N)))
        Xb, Lb = X[idx], labels[idx]

        if has_buf:
            buf_n = buffer['X'].shape[0]
            buf_idx = mx.array(np.random.randint(0, buf_n, size=min(BATCH_SIZE // 2, buf_n)))
            Xr = buffer['X'][buf_idx]
            Lr_labels = buffer['labels'][buf_idx]
            Lr_logits = buffer['logits'][buf_idx]

            def loss_fn(w):
                logits = FWD(w, Xb)
                task_loss = cross_entropy_loss(logits, Lb)
                replay_logits = FWD(w, Xr)
                # DER: match stored logits (knowledge distillation)
                logit_loss = mx.mean((replay_logits - mx.stop_gradient(Lr_logits)) ** 2)
                # DER++: also match stored class targets
                replay_ce = cross_entropy_loss(replay_logits, Lr_labels)
                return task_loss + alpha * logit_loss + beta * replay_ce
        else:
            def loss_fn(w):
                logits = FWD(w, Xb)
                return cross_entropy_loss(logits, Lb)

        loss, grads = mx.value_and_grad(loss_fn)(weights)
        clip_and_step(weights, grads, lr)


def update_der_buffer(buffer, weights, X, labels, max_size=2000):
    """Reservoir sampling into DER++ buffer."""
    logits = FWD(weights, X)
    mx.eval(logits)
    X_np, L_np, Lo_np = np.array(X), np.array(labels), np.array(logits)

    if buffer['X_np'] is None:
        buffer['X_np'] = X_np[:max_size]
        buffer['L_np'] = L_np[:max_size]
        buffer['Lo_np'] = Lo_np[:max_size]
    else:
        combined_X = np.concatenate([buffer['X_np'], X_np])
        combined_L = np.concatenate([buffer['L_np'], L_np])
        combined_Lo = np.concatenate([buffer['Lo_np'], Lo_np])
        if len(combined_X) > max_size:
            idx = np.random.choice(len(combined_X), max_size, replace=False)
            combined_X = combined_X[idx]
            combined_L = combined_L[idx]
            combined_Lo = combined_Lo[idx]
        buffer['X_np'] = combined_X
        buffer['L_np'] = combined_L
        buffer['Lo_np'] = combined_Lo

    buffer['X'] = mx.array(buffer['X_np'])
    buffer['labels'] = mx.array(buffer['L_np'])
    buffer['logits'] = mx.array(buffer['Lo_np'])


# ── Accuracy measurement ────────────────────────────────────

def measure_accuracy(weights, X_test, labels_test):
    """Classification accuracy."""
    preds = FWD(weights, X_test)
    mx.eval(preds)
    predicted = np.argmax(np.array(preds), axis=1)
    return float((predicted == np.array(labels_test)).mean())


def measure_accuracy_tribe(tribe, X_test, labels_test):
    """Oracle-routed tribe accuracy (best expert per sample by loss)."""
    routable = tribe.routable_members()
    if not routable:
        return 0.0
    all_preds = []
    all_losses = []
    labels_int = mx.array(np.array(labels_test))
    for m in routable:
        preds = FWD(m.weights, X_test)
        # Per-sample CE loss for routing
        log_probs = preds - mx.logsumexp(preds, axis=-1, keepdims=True)
        losses = -log_probs[mx.arange(X_test.shape[0]), labels_int]
        all_preds.append(preds)
        all_losses.append(losses)
    loss_matrix = mx.stack(all_losses)
    best_idx = mx.argmin(loss_matrix, axis=0)
    mx.eval(best_idx, *all_preds)
    best_np = np.array(best_idx)
    preds_list = [np.array(p) for p in all_preds]
    N = X_test.shape[0]
    predicted = np.array([np.argmax(preds_list[best_np[i]][i]) for i in range(N)])
    return float((predicted == np.array(labels_test)).mean())


def measure_accuracy_maxconf(tribe, X_test, labels_test):
    """Max-confidence routing: pick expert with highest softmax peak. No labels needed."""
    routable = tribe.routable_members()
    if not routable:
        return 0.0
    all_confs = []
    all_preds = []
    for m in routable:
        logits = FWD(m.weights, X_test)
        probs = mx.softmax(logits, axis=-1)
        max_conf = mx.max(probs, axis=-1)       # (N,)
        pred_class = mx.argmax(logits, axis=-1)  # (N,)
        all_confs.append(max_conf)
        all_preds.append(pred_class)
    confs = mx.stack(all_confs, axis=0)   # (E, N)
    preds = mx.stack(all_preds, axis=0)   # (E, N)
    best_expert = mx.argmax(confs, axis=0)  # (N,)
    mx.eval(best_expert, preds)
    best_np = np.array(best_expert)
    preds_np = np.array(preds)
    N = X_test.shape[0]
    predicted = np.array([preds_np[best_np[i], i] for i in range(N)])
    return float((predicted == np.array(labels_test)).mean())


def measure_accuracy_learned(router, tribe, X_test, labels_test):
    """Learned router evaluation: route via SwitchRouter at test time. No labels needed."""
    routable = tribe.routable_members()
    if not routable or router is None:
        return 0.0
    X_flat = mx.reshape(X_test, (X_test.shape[0], -1))
    assignments, _, _ = router.route(X_flat)

    active = sorted(routable, key=lambda m: m.id)
    N = X_test.shape[0]
    predicted = np.full(N, -1, dtype=np.int32)
    for eid, indices in assignments.items():
        if eid >= len(active) or not indices:
            continue
        m = active[eid]
        idx_arr = mx.array(indices)
        X_sub = X_test[idx_arr]
        logits = FWD(m.weights, X_sub)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        preds_np = np.array(preds)
        for j, si in enumerate(indices):
            predicted[si] = preds_np[j]

    # Unrouted samples: use max-confidence fallback
    unrouted = np.where(predicted == -1)[0]
    if len(unrouted) > 0:
        X_unr = X_test[mx.array(unrouted)]
        best_conf = np.full(len(unrouted), -1.0)
        best_pred = np.zeros(len(unrouted), dtype=np.int32)
        for m in routable:
            logits = FWD(m.weights, X_unr)
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

    return float((predicted == np.array(labels_test)).mean())


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


# ── Baselines ────────────────────────────────────────────────

def run_finetune(tasks):
    print("\n  [1] FINE-TUNE (single network)")
    weights = make_expert(seed=0)
    acc_matrix = []
    for t, (train_td, _) in enumerate(tasks):
        t0 = time.time()
        X, labels = train_td.as_mx_labels()
        batch_train(weights, X, labels)
        row = [measure_accuracy(weights, *td_te.as_mx_labels()) for _, td_te in tasks]
        acc_matrix.append(row)
        print(f"      Task {t}: acc={row[t]:.1%}, elapsed={time.time()-t0:.1f}s")
    return acc_matrix


def run_ewc(tasks):
    print("\n  [2] EWC (lambda=400)")
    weights = make_expert(seed=0)
    fisher_acc = {}
    star_weights = {}
    acc_matrix = []
    for t, (train_td, _) in enumerate(tasks):
        t0 = time.time()
        X, labels = train_td.as_mx_labels()
        if fisher_acc:
            batch_train_ewc(weights, X, labels, fisher_acc, star_weights, lam=400.0)
        else:
            batch_train(weights, X, labels)
        new_fisher = compute_fisher(weights, X, labels)
        for k in new_fisher:
            fisher_acc[k] = fisher_acc.get(k, mx.zeros_like(new_fisher[k])) + new_fisher[k]
        star_weights = {k: mx.array(weights[k]) for k in weights}
        row = [measure_accuracy(weights, *td_te.as_mx_labels()) for _, td_te in tasks]
        acc_matrix.append(row)
        print(f"      Task {t}: acc={row[t]:.1%}, elapsed={time.time()-t0:.1f}s")
    return acc_matrix


def run_derpp(tasks):
    print("\n  [3] DER++ (buffer=2000)")
    weights = make_expert(seed=0)
    buffer = {'X': None, 'labels': None, 'logits': None,
              'X_np': None, 'L_np': None, 'Lo_np': None}
    acc_matrix = []
    for t, (train_td, _) in enumerate(tasks):
        t0 = time.time()
        X, labels = train_td.as_mx_labels()
        batch_train_derpp(weights, X, labels, buffer)
        update_der_buffer(buffer, weights, X, labels)
        row = [measure_accuracy(weights, *td_te.as_mx_labels()) for _, td_te in tasks]
        acc_matrix.append(row)
        print(f"      Task {t}: acc={row[t]:.1%}, elapsed={time.time()-t0:.1f}s")
    return acc_matrix


def run_static_tribe(tasks):
    from tribe import Tribe
    print(f"\n  [4] STATIC TRIBE ({N_EXPERTS} experts)")
    tribe = Tribe(fwd=FWD, make_expert_fn=make_expert)
    for i in range(N_EXPERTS):
        tribe.add_member(make_expert(seed=i * 42))
    for i in range(N_EXPERTS):
        tribe.connect(i, (i + 1) % N_EXPERTS)
    acc_matrix = []
    acc_maxconf = []
    expert_data = {m.id: {'X': [], 'L': []} for m in tribe.members.values()}

    for t, (train_td, _) in enumerate(tasks):
        t0 = time.time()
        tribe.generation = t
        X, labels = train_td.as_mx_labels()

        # Route by CE loss, top_k=1
        routable = tribe.routable_members()
        labels_int = mx.array(np.array(labels))
        all_losses = []
        for m in routable:
            logits = FWD(m.weights, X)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            losses = -log_probs[mx.arange(X.shape[0]), labels_int]
            all_losses.append(losses)
        loss_matrix = mx.stack(all_losses)
        mx.eval(loss_matrix)
        best = np.argmin(np.array(loss_matrix), axis=0)
        X_np, L_np = np.array(X), np.array(labels)

        for i, m in enumerate(routable):
            mask = best == i
            if mask.any() and m.is_trainable:
                expert_data[m.id]['X'].append(X_np[mask])
                expert_data[m.id]['L'].append(L_np[mask])

        for m in tribe.active_members():
            if expert_data[m.id]['X']:
                eX = mx.array(np.concatenate(expert_data[m.id]['X']))
                eL = mx.array(np.concatenate(expert_data[m.id]['L']))
                batch_train(m.weights, eX, eL, use_aug=True)

        row_oracle = [measure_accuracy_tribe(tribe, *td_te.as_mx_labels()) for _, td_te in tasks]
        row_maxconf = [measure_accuracy_maxconf(tribe, *td_te.as_mx_labels()) for _, td_te in tasks]
        acc_matrix.append(row_oracle)
        acc_maxconf.append(row_maxconf)
        print(f"      Task {t}: oracle={row_oracle[t]:.1%}, maxconf={row_maxconf[t]:.1%}, elapsed={time.time()-t0:.1f}s")

    return acc_matrix, acc_maxconf


def run_lifecycle_tribe(tasks):
    from tribe import Tribe
    print(f"\n  [5] LIFECYCLE TRIBE ({N_EXPERTS} experts, top_k=2)")
    tribe = Tribe(fwd=FWD, make_expert_fn=make_expert)
    for i in range(N_EXPERTS):
        tribe.add_member(make_expert(seed=i * 42))
    for i in range(N_EXPERTS):
        tribe.connect(i, (i + 1) % N_EXPERTS)
    acc_matrix = []
    acc_maxconf = []
    expert_data = {m.id: {'X': [], 'L': []} for m in tribe.members.values()}

    for t, (train_td, _) in enumerate(tasks):
        t0 = time.time()
        tribe.generation = t
        X, labels = train_td.as_mx_labels()

        routable = tribe.routable_members()
        labels_int = mx.array(np.array(labels))
        all_losses = []
        for m in routable:
            logits = FWD(m.weights, X)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            losses = -log_probs[mx.arange(X.shape[0]), labels_int]
            all_losses.append(losses)
        loss_matrix = mx.stack(all_losses)
        mx.eval(loss_matrix)
        loss_np = np.array(loss_matrix)
        X_np, L_np = np.array(X), np.array(labels)

        # Top-2 assignment
        if loss_np.shape[0] >= 2:
            top2 = np.argpartition(loss_np, 2, axis=0)[:2]
        else:
            top2 = np.zeros((2, loss_np.shape[1]), dtype=int)

        for i, m in enumerate(routable):
            mask = (top2[0] == i) | (top2[1] == i)
            if mask.any() and m.is_trainable:
                expert_data[m.id]['X'].append(X_np[mask])
                expert_data[m.id]['L'].append(L_np[mask])

        for m in tribe.active_members():
            if expert_data[m.id]['X']:
                eX = mx.array(np.concatenate(expert_data[m.id]['X']))
                eL = mx.array(np.concatenate(expert_data[m.id]['L']))
                batch_train(m.weights, eX, eL, use_aug=True)

        # Lifecycle
        events = []
        if t > 0:
            _lifecycle(tribe, t, X, labels, expert_data, events)

        row_oracle = [measure_accuracy_tribe(tribe, *td_te.as_mx_labels()) for _, td_te in tasks]
        row_maxconf = [measure_accuracy_maxconf(tribe, *td_te.as_mx_labels()) for _, td_te in tasks]
        acc_matrix.append(row_oracle)
        acc_maxconf.append(row_maxconf)
        na = len(tribe.active_members())
        nf = len(tribe.frozen_members())
        print(f"      Task {t}: oracle={row_oracle[t]:.1%}, maxconf={row_maxconf[t]:.1%}, "
              f"active={na}, frozen={nf}, events={events}, elapsed={time.time()-t0:.1f}s")

    return acc_matrix, acc_maxconf


def run_learned_lifecycle(tasks):
    from tribe import Tribe
    mode = "joint" if JOINT_TRAINING else "separate"
    print(f"\n  [6] LEARNED ROUTER + LIFECYCLE ({N_EXPERTS} experts, {mode})")
    tribe = Tribe(fwd=FWD, make_expert_fn=make_expert)
    for i in range(N_EXPERTS):
        tribe.add_member(make_expert(seed=i * 42))
    for i in range(N_EXPERTS):
        tribe.connect(i, (i + 1) % N_EXPERTS)
    router = SwitchRouter(input_dim=3072, num_experts=N_EXPERTS, top_k=1)
    tribe.set_router(router)
    acc_matrix = []
    acc_maxconf = []
    acc_learned = []
    expert_data = {m.id: {'X': [], 'L': []} for m in tribe.members.values()}

    for t, (train_td, _) in enumerate(tasks):
        t0 = time.time()
        tribe.generation = t
        X, labels = train_td.as_mx_labels()
        X_flat = mx.reshape(X, (X.shape[0], -1))
        assignments_idx, aux_loss, stats = router.route(X_flat)

        active = tribe.active_members()
        active_ids = sorted([m.id for m in active])
        X_np, L_np = np.array(X), np.array(labels)
        for eid, indices in assignments_idx.items():
            if eid >= len(active_ids) or not indices:
                continue
            mid = active_ids[eid]
            expert_data[mid]['X'].append(X_np[indices])
            expert_data[mid]['L'].append(L_np[indices])

        if JOINT_TRAINING:
            # Joint router-expert training: router sees expert loss, experts
            # see routing signal. Uses STE for hard dispatch + soft gradients.
            # Includes specialization losses (orthogonality + variance).
            active = sorted(tribe.active_members(), key=lambda m: m.id)
            expert_weights = [m.weights for m in active]
            batch_train_joint(router, expert_weights, X, labels, fwd=FWD,
                              steps=TRAIN_STEPS, lr=TRAIN_LR,
                              fwd_hidden=FWD_HIDDEN,
                              alpha_orth=ALPHA_ORTH, alpha_var=ALPHA_VAR)
        else:
            # Fallback: separate expert training + router load-balance update
            for m in tribe.active_members():
                if expert_data[m.id]['X']:
                    eX = mx.array(np.concatenate(expert_data[m.id]['X']))
                    eL = mx.array(np.concatenate(expert_data[m.id]['L']))
                    batch_train(m.weights, eX, eL, use_aug=True)
            router.train_step(X_flat, lr=0.01)

        events = []
        if t > 0:
            _lifecycle(tribe, t, X, labels, expert_data, events)

        row_oracle = [measure_accuracy_tribe(tribe, *td_te.as_mx_labels()) for _, td_te in tasks]
        row_maxconf = [measure_accuracy_maxconf(tribe, *td_te.as_mx_labels()) for _, td_te in tasks]
        row_learned = [measure_accuracy_learned(router, tribe, *td_te.as_mx_labels()) for _, td_te in tasks]
        acc_matrix.append(row_oracle)
        acc_maxconf.append(row_maxconf)
        acc_learned.append(row_learned)
        print(f"      Task {t}: oracle={row_oracle[t]:.1%}, maxconf={row_maxconf[t]:.1%}, "
              f"learned={row_learned[t]:.1%}, counts={stats['counts']}, "
              f"events={events}, elapsed={time.time()-t0:.1f}s")

    return acc_matrix, acc_maxconf, acc_learned


# ── Lifecycle helper ─────────────────────────────────────────

def _lifecycle(tribe, gen, X, labels, expert_data, events):
    """Simplified lifecycle: freeze saturated, recycle underperformers."""
    active = tribe.active_members()
    if len(active) <= 3:
        return

    for m in list(active):
        if not expert_data[m.id]['X']:
            continue
        eX = mx.array(np.concatenate(expert_data[m.id]['X'])[:200])
        eL = mx.array(np.concatenate(expert_data[m.id]['L'])[:200])
        def loss_fn(w, eX=eX, eL=eL):
            logits = FWD(w, eX)
            return cross_entropy_loss(logits, eL)
        _, grads = mx.value_and_grad(loss_fn)(m.weights)
        total_params = sum(grads[k].size for k in grads)
        grad_norm = sum(mx.sum(grads[k] ** 2).item() for k in grads) / total_params
        if grad_norm < 1e-6 and len(active) > N_EXPERTS // 2:
            tribe.freeze_member(m.id)
            events.append(f"FREEZE {m.id}")
            active = tribe.active_members()

    active = tribe.active_members()
    if len(active) > N_EXPERTS // 2:
        expert_sizes = []
        for m in active:
            size = sum(x.shape[0] for x in expert_data[m.id]['X']) if expert_data[m.id]['X'] else 0
            expert_sizes.append((m, size))
        expert_sizes.sort(key=lambda x: x[1])
        worst, worst_size = expert_sizes[0]
        avg_size = np.mean([s for _, s in expert_sizes])
        if worst_size < avg_size * 0.1:
            new_m = tribe.recycle(worst.id)
            X_np, L_np = np.array(X), np.array(labels)
            sub_idx = np.random.choice(X_np.shape[0], min(500, X_np.shape[0]), replace=False)
            expert_data[new_m.id] = {'X': [X_np[sub_idx]], 'L': [L_np[sub_idx]]}
            batch_train(new_m.weights, mx.array(X_np[sub_idx]), mx.array(L_np[sub_idx]),
                        steps=200, member=new_m, use_aug=True)
            events.append(f"RECYCLE {worst.id}")


# ── Reporting ────────────────────────────────────────────────

def print_comparison(results):
    print("\n" + "=" * 80)
    print("  SPLIT CIFAR-100 BENCHMARK RESULTS")
    print("=" * 80)

    # Oracle routing (standard, for backward compatibility)
    print(f"\n  Oracle Routing (loss-based, uses ground truth labels):")
    print(f"  {'Method':<30s} {'FA':>7s} {'BWT':>7s} {'Fgt':>7s}")
    print("  " + "-" * 51)
    for name, acc_matrix, _, _ in results:
        m = compute_cl_metrics(acc_matrix)
        print(f"  {name:<30s} {m['FA']:>7.1%} {m['BWT']:>+7.1%} {m['Forgetting']:>7.1%}")

    # Max-confidence routing (label-free, honest class-incremental)
    has_maxconf = any(mc is not None for _, _, mc, _ in results)
    if has_maxconf:
        print(f"\n  Max-Confidence Routing (label-free, class-incremental):")
        print(f"  {'Method':<30s} {'FA':>7s} {'BWT':>7s} {'Fgt':>7s}")
        print("  " + "-" * 51)
        for name, acc_oracle, acc_mc, _ in results:
            if acc_mc is not None:
                m = compute_cl_metrics(acc_mc)
                print(f"  {name:<30s} {m['FA']:>7.1%} {m['BWT']:>+7.1%} {m['Forgetting']:>7.1%}")

    # Learned router (SwitchRouter at test time)
    has_learned = any(lr is not None for _, _, _, lr in results)
    if has_learned:
        print(f"\n  Learned Router (SwitchRouter at test time):")
        print(f"  {'Method':<30s} {'FA':>7s} {'BWT':>7s} {'Fgt':>7s}")
        print("  " + "-" * 51)
        for name, _, _, acc_lr in results:
            if acc_lr is not None:
                m = compute_cl_metrics(acc_lr)
                print(f"  {name:<30s} {m['FA']:>7.1%} {m['BWT']:>+7.1%} {m['Forgetting']:>7.1%}")

    # Per-task breakdown (oracle)
    print(f"\n  Per-task final accuracy — Oracle (after all {N_TASKS} tasks):")
    header = f"  {'Method':<30s}" + "".join(f" T{j:>2}" for j in range(N_TASKS))
    print(header)
    print("  " + "-" * (30 + 4 * N_TASKS))
    for name, acc_matrix, _, _ in results:
        final = acc_matrix[-1]
        vals = "".join(f" {v:3.0%}" for v in final)
        print(f"  {name:<30s}{vals}")

    # Per-task breakdown (max-conf)
    if has_maxconf:
        print(f"\n  Per-task final accuracy — Max-Confidence (after all {N_TASKS} tasks):")
        print(header)
        print("  " + "-" * (30 + 4 * N_TASKS))
        for name, _, acc_mc, _ in results:
            if acc_mc is not None:
                final = acc_mc[-1]
                vals = "".join(f" {v:3.0%}" for v in final)
                print(f"  {name:<30s}{vals}")


# ── Multi-Seed Runner ────────────────────────────────────────

def run_multi_seed(method_fn, method_name, tasks, n_seeds=3):
    """Run a method with multiple seeds, report mean +/- std.

    Args:
        method_fn: function(tasks) that returns results in the same format as run_finetune etc.
        method_name: string name for display.
        tasks: task list from make_split_cifar100.
        n_seeds: number of random seeds.

    Returns:
        list of per-seed results.
    """
    all_results = []
    for seed in range(n_seeds):
        np.random.seed(seed * 1000)
        mx.random.seed(seed * 1000)
        print(f"\n  --- Seed {seed}/{n_seeds} for {method_name} ---")
        result = method_fn(tasks)
        all_results.append(result)

    # Compute stats across seeds
    # result can be: acc_matrix, or (acc_matrix, acc_maxconf), or (acc_matrix, acc_maxconf, acc_learned)
    # Extract oracle matrix from each result
    def get_oracle(r):
        if isinstance(r, tuple):
            return r[0]
        return r

    fa_vals = []
    fgt_vals = []
    for r in all_results:
        m = compute_cl_metrics(get_oracle(r))
        fa_vals.append(m['FA'])
        fgt_vals.append(m['Forgetting'])

    print(f"\n  {method_name} ({n_seeds} seeds):")
    print(f"    FA:  {np.mean(fa_vals)*100:.1f}% +/- {np.std(fa_vals)*100:.1f}%")
    print(f"    Fgt: {np.mean(fgt_vals)*100:.1f}% +/- {np.std(fgt_vals)*100:.1f}%")
    return all_results


# ── Main ─────────────────────────────────────────────────────

def run_benchmark():
    t_start = time.time()
    print("\n" + "=" * 80)
    print("  SPLIT CIFAR-100 CONTINUAL LEARNING BENCHMARK")
    print("  10 tasks × 10 classes, class-incremental setting")
    print("=" * 80)

    train_imgs, train_labels, test_imgs, test_labels = load_cifar100()
    print(f"  CIFAR-100: {len(train_imgs)} train, {len(test_imgs)} test")

    tasks = make_split_cifar100(train_imgs, train_labels, test_imgs, test_labels,
                                 n_tasks=N_TASKS, seed=42)
    for t, (tr, te) in enumerate(tasks):
        print(f"  Task {t}: classes {tr.classes} ({len(tr)} train, {len(te)} test)")

    results = []
    results.append(("Fine-tune", run_finetune(tasks), None, None))
    results.append(("EWC (λ=400)", run_ewc(tasks), None, None))
    results.append(("DER++ (buf=2000)", run_derpp(tasks), None, None))

    static_oracle, static_maxconf = run_static_tribe(tasks)
    results.append(("Static Tribe", static_oracle, static_maxconf, None))

    lc_oracle, lc_maxconf = run_lifecycle_tribe(tasks)
    results.append(("Lifecycle Tribe", lc_oracle, lc_maxconf, None))

    lr_oracle, lr_maxconf, lr_learned = run_learned_lifecycle(tasks)
    results.append(("Learned + Lifecycle", lr_oracle, lr_maxconf, lr_learned))

    print_comparison(results)
    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} minutes")
    return results


if __name__ == "__main__":
    # Parse --seeds=N
    for arg in sys.argv[1:]:
        if arg.startswith('--seeds='):
            N_SEEDS = int(arg.split('=')[1])

    if N_SEEDS is not None and N_SEEDS > 1:
        t_start = time.time()
        print("\n" + "=" * 80)
        print(f"  SPLIT CIFAR-100 MULTI-SEED BENCHMARK ({N_SEEDS} seeds)")
        print("=" * 80)

        train_imgs, train_labels, test_imgs, test_labels = load_cifar100()
        tasks = make_split_cifar100(train_imgs, train_labels, test_imgs, test_labels,
                                     n_tasks=N_TASKS, seed=42)

        methods = [
            (run_finetune, "Fine-tune"),
            (run_ewc, "EWC (lambda=400)"),
            (run_derpp, "DER++ (buf=2000)"),
            (run_static_tribe, "Static Tribe"),
            (run_lifecycle_tribe, "Lifecycle Tribe"),
            (run_learned_lifecycle, "Learned + Lifecycle"),
        ]

        for method_fn, method_name in methods:
            run_multi_seed(method_fn, method_name, tasks, n_seeds=N_SEEDS)

        print(f"\n  Total time: {(time.time()-t_start)/60:.1f} minutes")
    else:
        run_benchmark()
