"""CL baselines for LoRA: EWC, Replay, O-LoRA.

Ported from our MLX implementations in bench_llm_peer.py.
All operate on PEFT LoRA models via standard PyTorch APIs.
"""

import torch
import numpy as np


# ── Utilities ───────────────────────────────────────────────


def _iter_lora_params(model):
    """Yield (name, param) for all LoRA A/B parameters in a PEFT model."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield name, param


# ── EWC: Elastic Weight Consolidation ──────────────────────


class EWCState:
    """Fisher diagonal + param snapshots for Elastic Weight Consolidation.

    After each domain, accumulates squared gradients (Fisher approximation)
    and snapshots current parameters. Penalty term during training on next
    domain prevents important parameters from changing.
    """

    def __init__(self, lambda_ewc=100.0):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}       # param_name -> Tensor (accumulated)
        self.star_params = {}  # param_name -> Tensor (snapshot)

    def compute_fisher(self, model, dataloader, n_samples=50, device="cuda"):
        """Estimate Fisher information diagonal after training on a domain.

        Accumulates squared gradients from n_samples mini-batches over
        LoRA A/B parameters only.
        """
        model.eval()
        fisher_acc = {}
        count = 0

        for batch in dataloader:
            if count >= n_samples:
                break
            tokens = batch.to(device)
            model.zero_grad()

            outputs = model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
            outputs.loss.backward()

            for name, param in _iter_lora_params(model):
                if param.grad is not None:
                    sq = param.grad.detach() ** 2
                    if name in fisher_acc:
                        fisher_acc[name] += sq
                    else:
                        fisher_acc[name] = sq.clone()
            count += 1

        # Average and accumulate into total Fisher
        for name in fisher_acc:
            fisher_acc[name] /= count
            if name in self.fisher:
                self.fisher[name] = self.fisher[name] + fisher_acc[name]
            else:
                self.fisher[name] = fisher_acc[name]

        # Snapshot current params
        for name, param in _iter_lora_params(model):
            self.star_params[name] = param.detach().clone()

        model.train()
        print(f"    EWC: computed Fisher over {count} batches, "
              f"{len(self.fisher)} param groups")

    def penalty(self, model):
        """Compute EWC penalty: sum_i F_i * (theta_i - theta*_i)^2."""
        pen = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in _iter_lora_params(model):
            if name in self.fisher:
                diff = param - self.star_params[name]
                pen = pen + (self.fisher[name] * diff * diff).sum()
        return (self.lambda_ewc / 2.0) * pen


# ── Experience Replay ──────────────────────────────────────


class ReplayBuffer:
    """Experience replay buffer using reservoir sampling.

    Stores tokenized sequences from past domains. During training,
    half the batch comes from current domain and half from buffer.
    """

    def __init__(self):
        self.buffer = []  # list of (seq_len,) tensors

    def store_domain(self, dataset, n=100):
        """Store n sequences from a domain via reservoir sampling."""
        rng = np.random.RandomState(len(self.buffer))
        n_available = len(dataset)
        indices = rng.choice(n_available, size=min(n, n_available), replace=False)
        for i in indices:
            self.buffer.append(dataset[int(i)].clone())
        print(f"    REPLAY: stored {len(indices)} seqs, "
              f"buffer size={len(self.buffer)}")

    def sample(self, batch_size, device="cuda"):
        """Sample a batch from the buffer.

        Returns:
            (batch_size, seq_len) tensor, or None if buffer is empty.
        """
        if not self.buffer:
            return None
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = torch.stack([self.buffer[int(i)] for i in indices])
        return batch.to(device)


# ── O-LoRA: Orthogonal LoRA ───────────────────────────────


class OLoRAState:
    """Orthogonal basis accumulator for gradient projection.

    After each domain, extract LoRA-A directions, QR-orthogonalize, and
    accumulate into a basis. During next domain training, project out
    these directions from LoRA-A gradients so new learning is orthogonal
    to previously learned subspaces.
    """

    def __init__(self):
        self.bases = {}  # param_name -> (d_in, accum_rank) orthonormal basis tensor

    @torch.no_grad()
    def accumulate_basis(self, model):
        """Extract lora_A directions, QR-orthogonalize, accumulate."""
        n_new = 0
        for name, param in _iter_lora_params(model):
            if "lora_A" not in name:
                continue

            # lora_A.weight is (rank, in_features) — transpose to get columns as directions
            A = param.detach().cpu().float().T  # (in_features, rank)

            if name in self.bases:
                existing = self.bases[name]
                combined = torch.cat([existing, A], dim=1)
            else:
                combined = A

            # QR decomposition
            Q, _ = torch.linalg.qr(combined)
            self.bases[name] = Q
            n_new += 1

        max_rank = max(v.shape[1] for v in self.bases.values()) if self.bases else 0
        print(f"    O-LoRA: accumulated basis for {n_new} layers, "
              f"max rank={max_rank}")

    def project_gradients(self, model):
        """Project out accumulated basis directions from lora_A gradients.

        grad_A -= Q @ Q.T @ grad_A (remove components in old subspace).
        Must be called after loss.backward() and before optimizer.step().
        """
        with torch.no_grad():
            for name, param in _iter_lora_params(model):
                if "lora_A" not in name:
                    continue
                if name not in self.bases:
                    continue
                if param.grad is None:
                    continue

                Q = self.bases[name].to(param.grad.device)  # (in_features, accum_rank)
                # grad is (rank, in_features), work with transposed view
                grad_T = param.grad.T  # (in_features, rank)
                proj = Q @ (Q.T @ grad_T)
                param.grad -= proj.T  # project out, back to (rank, in_features)
