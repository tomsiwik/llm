"""Standard continual learning baselines: EWC, DER++, Fine-tune.

All operate on flat weight dicts compatible with the tribe system.
"""

import mlx.core as mx
import numpy as np


# ── Fine-tune (catastrophic forgetting control) ──────────────

def finetune_train(weights, patterns, steps=200, lr=0.01, fwd=None, batch_size=64):
    """Standard fine-tuning with mini-batches. No CL mechanism."""
    if not patterns:
        return float('inf')
    X = mx.stack([x for x, _ in patterns])
    T = mx.stack([t for _, t in patterns])
    N = X.shape[0]

    for step in range(steps):
        idx = np.random.randint(0, N, size=min(batch_size, N))
        Xb = X[mx.array(idx)]
        Tb = T[mx.array(idx)]

        def loss_fn(w):
            return mx.mean((fwd(w, Xb) - Tb) ** 2)
        loss, grads = mx.value_and_grad(loss_fn)(weights)
        for k in weights:
            weights[k] = weights[k] - lr * grads[k]
        mx.eval(*[weights[k] for k in weights])

    preds = fwd(weights, X)
    return mx.mean((preds - T) ** 2).item()


# ── EWC: Elastic Weight Consolidation ────────────────────────

class EWC:
    """Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    After each task, compute Fisher Information diagonal as importance weights.
    Penalise changes to important parameters during future tasks.
    """

    def __init__(self, lambda_ewc=400.0):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}       # param_key → importance (same shape as weights)
        self.star_weights = {} # param_key → snapshot after task

    def compute_fisher(self, weights, patterns, fwd, n_samples=200):
        """Estimate diagonal Fisher Information from task data."""
        if not patterns:
            return
        n = min(n_samples, len(patterns))
        idx = np.random.choice(len(patterns), n, replace=False)
        sampled = [patterns[i] for i in idx]
        X = mx.stack([x for x, _ in sampled])
        T = mx.stack([t for _, t in sampled])

        def loss_fn(w):
            return mx.mean((fwd(w, X) - T) ** 2)
        _, grads = mx.value_and_grad(loss_fn)(weights)
        mx.eval(*[grads[k] for k in grads])

        # Fisher ≈ squared gradients (diagonal approximation)
        for k in weights:
            new_fisher = grads[k] ** 2
            if k in self.fisher:
                # Accumulate across tasks
                self.fisher[k] = self.fisher[k] + new_fisher
            else:
                self.fisher[k] = new_fisher
            self.star_weights[k] = mx.array(weights[k])

    def train(self, weights, patterns, steps=200, lr=0.01, fwd=None, batch_size=64):
        """Train with EWC penalty."""
        if not patterns:
            return float('inf')
        X = mx.stack([x for x, _ in patterns])
        T = mx.stack([t for _, t in patterns])
        N = X.shape[0]

        fisher = self.fisher
        star = self.star_weights
        lam = self.lambda_ewc

        for step in range(steps):
            idx = np.random.randint(0, N, size=min(batch_size, N))
            Xb = X[mx.array(idx)]
            Tb = T[mx.array(idx)]

            def loss_fn(w):
                task_loss = mx.mean((fwd(w, Xb) - Tb) ** 2)
                ewc_penalty = mx.array(0.0)
                for k in w:
                    if k in fisher:
                        ewc_penalty = ewc_penalty + mx.sum(
                            fisher[k] * (w[k] - star[k]) ** 2)
                return task_loss + (lam / 2.0) * ewc_penalty
            loss, grads = mx.value_and_grad(loss_fn)(weights)
            for k in weights:
                weights[k] = weights[k] - lr * grads[k]
            mx.eval(*[weights[k] for k in weights])

        preds = fwd(weights, X)
        return mx.mean((preds - T) ** 2).item()


# ── DER++: Dark Experience Replay ─────────────────────────────

class DERPlusPlus:
    """Dark Experience Replay++ (Buzzega et al. NeurIPS 2020).

    Maintains a replay buffer of (input, target, logit) triples.
    Loss = task_loss + alpha * MSE(current_logits, stored_logits) + beta * MSE(current_pred, stored_target)
    """

    def __init__(self, buffer_size=500, alpha=0.5, beta=0.5):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.buffer_x = []
        self.buffer_t = []
        self.buffer_logits = []

    def update_buffer(self, weights, patterns, fwd):
        """Add current task data + logits to replay buffer via reservoir sampling."""
        X = mx.stack([x for x, _ in patterns])
        T = mx.stack([t for _, t in patterns])
        logits = fwd(weights, X)
        mx.eval(logits)

        for i in range(len(patterns)):
            if len(self.buffer_x) < self.buffer_size:
                self.buffer_x.append(X[i])
                self.buffer_t.append(T[i])
                self.buffer_logits.append(logits[i])
            else:
                j = np.random.randint(0, len(self.buffer_x) + i + 1)
                if j < self.buffer_size:
                    self.buffer_x[j] = X[i]
                    self.buffer_t[j] = T[i]
                    self.buffer_logits[j] = logits[i]

    def train(self, weights, patterns, steps=200, lr=0.01, fwd=None, batch_size=64):
        """Train with DER++ replay loss."""
        if not patterns:
            return float('inf')
        X = mx.stack([x for x, _ in patterns])
        T = mx.stack([t for _, t in patterns])
        N = X.shape[0]
        has_buffer = len(self.buffer_x) > 0
        alpha = self.alpha
        beta = self.beta

        if has_buffer:
            buf_X = mx.stack(self.buffer_x)
            buf_T = mx.stack(self.buffer_t)
            buf_L = mx.stack(self.buffer_logits)
            buf_N = buf_X.shape[0]

        for step in range(steps):
            idx = np.random.randint(0, N, size=min(batch_size, N))
            Xb = X[mx.array(idx)]
            Tb = T[mx.array(idx)]

            if has_buffer:
                buf_idx = np.random.randint(0, buf_N, size=min(batch_size // 2, buf_N))
                Xr = buf_X[mx.array(buf_idx)]
                Tr = buf_T[mx.array(buf_idx)]
                Lr = buf_L[mx.array(buf_idx)]

                def loss_fn(w):
                    # Current task loss
                    task_loss = mx.mean((fwd(w, Xb) - Tb) ** 2)
                    # DER: match stored logits
                    replay_logits = fwd(w, Xr)
                    logit_loss = mx.mean((replay_logits - mx.stop_gradient(Lr)) ** 2)
                    # DER++: also match stored targets
                    target_loss = mx.mean((replay_logits - Tr) ** 2)
                    return task_loss + alpha * logit_loss + beta * target_loss
            else:
                def loss_fn(w):
                    return mx.mean((fwd(w, Xb) - Tb) ** 2)

            loss, grads = mx.value_and_grad(loss_fn)(weights)
            for k in weights:
                weights[k] = weights[k] - lr * grads[k]
            mx.eval(*[weights[k] for k in weights])

        preds = fwd(weights, X)
        return mx.mean((preds - T) ** 2).item()
