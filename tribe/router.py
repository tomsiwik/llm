"""Learned top-k gating with load balancing loss (Switch Transformer).

Ref: Fedus et al. 2022 — Switch Transformers
     examples/01_switch_routing.py
"""

import mlx.core as mx
import numpy as np


class SwitchRouter:
    """Learned router: linear projection → softmax → top-k assignment.

    Trains jointly with experts via a combined loss:
        total_loss = expert_loss + alpha * load_balancing_loss
    """

    def __init__(self, input_dim, num_experts, top_k=1):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.weights = {
            'router_w': mx.random.normal((num_experts, input_dim)) * np.sqrt(2.0 / input_dim),
            'router_b': mx.zeros((num_experts,)),
        }

    def _logits(self, X):
        """Compute router logits: (N, input_dim) → (N, num_experts)."""
        return X @ self.weights['router_w'].T + self.weights['router_b']

    def _probs(self, X):
        """Softmax routing probabilities."""
        logits = self._logits(X)
        return mx.softmax(logits, axis=-1)

    def route(self, X, T=None):
        """Route a batch of inputs to experts.

        Args:
            X: (N, input_dim) flattened inputs.
            T: ignored (kept for API compatibility with oracle routing).

        Returns:
            assignments: dict[int, list[int]] — expert_id → list of sample indices.
            aux_loss: scalar load balancing loss.
            stats: dict with 'counts', 'entropy', 'max_imbalance'.
        """
        probs = self._probs(X)  # (N, num_experts)
        mx.eval(probs)

        N = X.shape[0]
        probs_np = np.array(probs)

        # Top-k: for each sample, pick k experts with highest probability
        if self.top_k >= self.num_experts:
            # All experts
            top_indices = np.tile(np.arange(self.num_experts), (N, 1))
        else:
            # argpartition for top-k (no torch.topk in MLX)
            top_indices = np.argpartition(-probs_np, self.top_k, axis=1)[:, :self.top_k]

        # Build assignments: expert_id → list of sample indices
        assignments = {i: [] for i in range(self.num_experts)}
        for sample_idx in range(N):
            for k in range(self.top_k):
                expert_id = int(top_indices[sample_idx, k])
                assignments[expert_id].append(sample_idx)

        # Load balancing loss: L = N_experts * sum(f_i * P_i)
        counts = np.array([len(assignments[i]) for i in range(self.num_experts)])
        total_tokens = max(counts.sum(), 1)
        f = mx.array(counts / total_tokens)       # fraction routed (hard)
        P = mx.mean(probs, axis=0)                 # mean prob per expert (soft)
        aux_loss = self.num_experts * mx.sum(f * P)
        mx.eval(aux_loss)

        # Stats
        entropy = -mx.sum(P * mx.log(P + 1e-10)).item()
        max_imbalance = float(counts.max()) / max(float(counts.min()), 1.0)

        stats = {
            'counts': counts.tolist(),
            'entropy': entropy,
            'max_imbalance': max_imbalance,
        }

        return assignments, aux_loss, stats

    def variance_loss(self, X_flat):
        """Negative variance of mean routing probabilities.

        Low variance = router assigns similar probability to all experts (undecided).
        High variance = router is confident about which expert to use.

        Minimizing this loss maximizes variance, encouraging decisive routing.

        Args:
            X_flat: (N, input_dim) flattened inputs.

        Returns:
            Scalar loss: -Var(mean_routing_probs).
        """
        probs = self._probs(X_flat)  # (N, E)
        mean_probs = mx.mean(probs, axis=0)  # (E,) — average routing prob per expert
        variance = mx.mean((mean_probs - mx.mean(mean_probs)) ** 2)
        return -variance  # negative because we want to MAXIMIZE variance

    def load_balancing_loss(self, X):
        """Compute just the differentiable load balancing loss for use in value_and_grad.

        Args:
            X: (N, input_dim) flattened inputs.

        Returns:
            Scalar aux loss (differentiable w.r.t. router weights).
        """
        probs = self._probs(X)  # (N, num_experts)
        N = X.shape[0]

        # Hard assignment counts (stop gradient — not differentiable)
        hard_probs = mx.stop_gradient(probs)
        assignments = mx.argmax(hard_probs, axis=-1)  # (N,)

        # f_i = fraction of tokens assigned to expert i
        counts = mx.zeros((self.num_experts,))
        for i in range(self.num_experts):
            counts = counts.at[i].add(mx.sum(assignments == i))
        f = counts / N

        # P_i = mean routing probability for expert i
        P = mx.mean(probs, axis=0)

        return self.num_experts * mx.sum(f * P)

    def soft_forward_topk(self, X_flat, X_images, expert_weights_list, fwd, top_k=2):
        """Differentiable MoE forward with straight-through estimation.

        Runs all experts on all inputs, mixes outputs by STE routing weights.
        Gradient flows through both router and expert parameters.

        Args:
            X_flat: (N, input_dim) flattened inputs for router.
            X_images: (N, H, W, C) image inputs for experts.
            expert_weights_list: list of expert weight dicts.
            fwd: expert forward function: fwd(weights, X_images) -> (N, C).
            top_k: number of experts to route to per sample.

        Returns:
            mixed: (N, num_classes) mixed expert outputs.
            aux_loss: scalar load balancing loss.
        """
        probs = self._probs(X_flat)  # (N, E) — differentiable
        N, E = probs.shape

        # Hard assignments via argmax (stop gradient for dispatch)
        if top_k >= E:
            # All experts selected — uniform hard weights
            hard = mx.ones((N, E)) / E
        elif top_k == 1:
            hard = mx.stop_gradient(
                mx.eye(E)[mx.argmax(probs, axis=-1)]
            )
        else:
            # Top-k: zero out non-top-k entries in hard mask
            # Use negative probs for argpartition (want largest)
            probs_np = np.array(mx.stop_gradient(probs))
            top_k_indices = np.argpartition(-probs_np, top_k, axis=1)[:, :top_k]
            hard_np = np.zeros((N, E), dtype=np.float32)
            for i in range(N):
                for k_idx in range(top_k):
                    hard_np[i, top_k_indices[i, k_idx]] = 1.0
            hard = mx.array(hard_np)

        # STE: use hard for dispatch, soft for gradient
        # Trick: hard - stop_gradient(probs) + probs
        # Forward pass uses hard values, backward pass gets probs gradients
        routing_weights = hard - mx.stop_gradient(probs) + probs  # (N, E)

        # Run each expert on ALL inputs and stack
        expert_outputs = []
        for ew in expert_weights_list:
            out = fwd(ew, X_images)  # (N, num_classes)
            expert_outputs.append(out)
        expert_stack = mx.stack(expert_outputs, axis=1)  # (N, E, C)

        # Mix by routing weights
        routing_expanded = mx.expand_dims(routing_weights, axis=-1)  # (N, E, 1)
        mixed = mx.sum(expert_stack * routing_expanded, axis=1)  # (N, C)

        # Load balancing aux loss (differentiable w.r.t. probs)
        f = mx.mean(hard, axis=0)  # fraction routed per expert (hard, no grad)
        P = mx.mean(probs, axis=0)  # mean prob per expert (soft, has grad)
        aux_loss = self.num_experts * mx.sum(f * P)

        return mixed, aux_loss

    def train_step(self, X_flat, T=None, expert_weights=None, expert_ids=None,
                   fwd=None, lr=0.01, alpha=0.01):
        """Update router weights to improve load balance.

        Expert training happens externally. This only updates the router's
        linear projection to push toward uniform assignment.

        Args:
            X_flat: (N, input_dim) flattened inputs for router.
            lr: learning rate for router weight update.
            alpha: not used directly (kept for API consistency).

        Returns:
            aux_loss: scalar load balancing loss (before update).
            stats: routing stats dict.
        """
        assignments, aux_loss, stats = self.route(X_flat)

        # Update router weights via gradient on load balancing loss
        def router_loss_fn(rw):
            logits = X_flat @ rw['router_w'].T + rw['router_b']
            probs = mx.softmax(logits, axis=-1)
            P = mx.mean(probs, axis=0)
            # Minimizing sum(P_i^2) pushes toward uniform P_i = 1/N
            return self.num_experts * mx.sum(P * P)

        loss_val, grads = mx.value_and_grad(router_loss_fn)(self.weights)
        for k in self.weights:
            self.weights[k] = self.weights[k] - lr * grads[k]
        mx.eval(*[self.weights[k] for k in self.weights])

        return aux_loss.item(), stats
