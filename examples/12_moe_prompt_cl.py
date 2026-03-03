"""12 -- MoE-Prompt CL: Mixture of Experts with Prompt-Based Continual Learning

Paper: Mixture of Experts Meets Prompt-Based Continual Learning
       (Le et al., 2024)
URL:   https://arxiv.org/abs/2405.14124
Repo:  Minhchuyentoancbn/MoE_PromptCL

MoE-PromptCL combines prompt pools (learnable prefix/prompt tokens) with
a MoE-style selection mechanism for continual learning. Instead of routing
to expert sub-networks, they route to expert PROMPTS -- each task gets
dedicated prompt slots, and a key-based matching mechanism selects prompts
to prepend based on input similarity.

Key CL mechanisms:
  1. Prompt pool with learnable keys for input-dependent selection
  2. Task-identity inference (TII) via classifier alignment
  3. Class-conditional feature statistics for replay-free CL
  4. Momentum-based prompt blending to prevent catastrophic forgetting

Extracted from: peft/prompt/hide_prompt.py (EPrompt),
                engines/hide_tii_engine.py (training + classifier alignment)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.distributions.multivariate_normal import MultivariateNormal


# ---------------------------------------------------------------------------
# EPrompt: Expert Prompt Pool with Key-Based Selection
# ---------------------------------------------------------------------------
# TRIBE NOTE: This is the "expert" in their MoE -- but experts are PROMPTS
# (learnable token sequences prepended to the input), not sub-networks.
# Compare with our system:
#   - Their "expert" = a set of learned prompt tokens
#   - Our "expert" = a full sub-network (MLP or CNN)
#   - Their routing = key similarity; our routing = loss-based evaluation
#   - Their capacity = fixed pool_size; ours = dynamic with lifecycle
#
# For CL, each task "claims" top_k slots during training. Old slots are not
# frozen but have momentum regularization to prevent drift -- softer than
# our freeze operation but less explicit about protecting old knowledge.
class EPrompt(nn.Module):
    """Expert Prompt Pool for continual learning."""

    def __init__(self, length=5, embed_dim=768, pool_size=10, top_k=2,
                 batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune=False, num_heads=12):
        super().__init__()
        self.length = length
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.use_prefix_tune = use_prefix_tune

        if use_prefix_tune:
            # TRIBE NOTE: Prefix-tuning: prompts injected as additional K/V
            # Shape: [layers, 2(K+V), pool_size, length, heads, head_dim]
            assert embed_dim % num_heads == 0
            shape = (num_layers, 2, pool_size, length,
                     num_heads, embed_dim // num_heads)
            self.prompt = nn.Parameter(torch.randn(shape))
            nn.init.uniform_(self.prompt, -1, 1)
        else:
            # Prompt-tuning: prompts prepended to token sequence
            shape = (num_layers, pool_size, length, embed_dim)
            self.prompt = nn.Parameter(torch.randn(shape))
            nn.init.uniform_(self.prompt, -1, 1)

        # TRIBE NOTE: Learnable keys for input-dependent prompt selection.
        # Each prompt slot has a key vector; input representation is compared
        # against all keys to select top_k. Analogous to our loss-based
        # routing but uses learned similarity rather than actual performance.
        key_shape = (pool_size, embed_dim)
        self.prompt_key = nn.Parameter(torch.randn(key_shape))
        nn.init.uniform_(self.prompt_key, -1, 1)

    def forward(self, x_embed, prompt_mask=None, prompt_idx=None,
                prompt_weight=None, prompt_momentum=0.0):
        """Select and compose prompts from the pool.

        TRIBE NOTE: Three selection modes:
          1. prompt_mask: explicit task-id-based selection (training)
          2. prompt_idx: pre-computed indices (from key matching)
          3. prompt_weight: soft weighted combination (MoE-style routing)
        """
        out = dict()
        idx = prompt_idx

        # TRIBE NOTE: Batchwise prompt -- consensus routing where the batch
        # "votes" on which experts to use. Reduces variance at the cost of
        # per-sample routing granularity.
        if self.batchwise_prompt and prompt_idx is not None:
            prompt_id, id_counts = torch.unique(
                prompt_idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                prompt_id = torch.cat([prompt_id, torch.full(
                    (self.pool_size - prompt_id.shape[0],),
                    torch.min(prompt_idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full(
                    (self.pool_size - id_counts.shape[0],),
                    0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            idx = prompt_id[major_idx].expand(x_embed.shape[0], -1).contiguous()

        if prompt_mask is not None:
            idx = prompt_mask
        if idx is not None:
            out['prompt_idx'] = idx

        if self.use_prefix_tune:
            if prompt_weight is not None:
                # TRIBE NOTE: Soft routing -- weighted combination of ALL
                # prompts. This is the MoE combination operation:
                # "bp,ndplhe->ndblhe" weights each prompt by its coefficient.
                raw = torch.einsum("bp,ndplhe->ndblhe", prompt_weight, self.prompt)
                raw = raw.unsqueeze(3)
                nl, d, bs, tk, le, nh, hd = raw.shape
                batched_prompt = raw.reshape(nl, bs, d, tk * le, nh, hd)
            elif prompt_momentum > 0 and prompt_mask is not None:
                # TRIBE NOTE: Momentum blending for CL -- current task's
                # prompts blended with MEAN of previous tasks' prompts.
                # Compare with our hard-freeze: they allow gradual drift,
                # we prevent it entirely.
                with torch.no_grad():
                    past = self.prompt[:, :, 0:idx[0][0]].detach().clone()
                    mom = past.mean(2, keepdim=True).unsqueeze(2).repeat(
                        1, 1, idx.shape[0], 1, 1, 1, 1)
                raw = (1 - prompt_momentum) * self.prompt[:, :, idx] + prompt_momentum * mom
                nl, d, bs, tk, le, nh, hd = raw.shape
                batched_prompt = raw.reshape(nl, bs, d, tk * le, nh, hd)
            else:
                raw = self.prompt[:, :, idx]
                nl, d, bs, tk, le, nh, hd = raw.shape
                batched_prompt = raw.reshape(nl, bs, d, tk * le, nh, hd)
        else:
            if prompt_weight is not None:
                raw = torch.einsum("bp,npld->nbpld", prompt_weight, self.prompt)
            else:
                raw = self.prompt[:, idx]
            nl, bs, tk, le, ed = raw.shape
            batched_prompt = raw.reshape(nl, bs, tk * le, ed)

        out['batched_prompt'] = batched_prompt
        return out


# ---------------------------------------------------------------------------
# Classifier Alignment (CA): replay-free anti-forgetting via Gaussian stats
# ---------------------------------------------------------------------------
# TRIBE NOTE: After each task, they compute per-class feature Gaussians
# and fine-tune only the classifier head on synthetic features sampled
# from these distributions. This aligns decision boundaries across all
# seen tasks without replaying real data.
#
# Compare with our system:
#   - We prevent forgetting by FREEZING expert weights (absolute)
#   - They prevent forgetting by ALIGNING the classifier (allows drift)
#   - Their per-class statistics are analogous to our per-expert competence
#     tracking, but used for replay vs. routing

def compute_class_statistics(
    features_per_class: Dict[int, torch.Tensor],
    method: str = 'covariance',
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Compute per-class Gaussian statistics for synthetic replay."""
    cls_mean, cls_cov = {}, {}
    for cls_id, features in features_per_class.items():
        cls_mean[cls_id] = features.mean(dim=0)
        cov = torch.cov(features.T) + torch.eye(
            features.shape[1], device=features.device) * 1e-4
        if method == 'variance':
            cls_cov[cls_id] = torch.diag(cov)
        else:
            cls_cov[cls_id] = cov
    return cls_mean, cls_cov


def classifier_alignment(
    model: nn.Module,
    cls_mean: Dict[int, torch.Tensor],
    cls_cov: Dict[int, torch.Tensor],
    class_mask: List[List[int]],
    task_id: int,
    device: torch.device,
    num_samples: int = 256,
    epochs: int = 5,
    lr: float = 0.01,
    nb_classes: int = 100,
) -> None:
    """Align classifier using synthetic features from stored statistics.

    TRIBE NOTE: Core anti-forgetting mechanism. After each task:
      1. Sample synthetic features from per-class Gaussians
      2. Train ONLY the classifier head (backbone/prompts frozen)
    Requires only O(D^2) storage per class (covariance matrix).
    """
    model.train()
    all_classes = [c for t in range(task_id + 1) for c in class_mask[t]]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        # TRIBE NOTE: Sample synthetic features from per-class Gaussians.
        # Creates balanced "replay" dataset without storing real data.
        data, labels = [], []
        for cls_id in all_classes:
            m = MultivariateNormal(
                cls_mean[cls_id].float().to(device),
                cls_cov[cls_id].float().to(device))
            data.append(m.sample((num_samples,)))
            labels.extend([cls_id] * num_samples)

        data = torch.cat(data, dim=0).to(device)
        labels = torch.tensor(labels).long().to(device)
        perm = torch.randperm(data.size(0))
        data, labels = data[perm], labels[perm]

        # Train classifier on synthetic features (fc_only mode)
        for i in range(len(all_classes)):
            inp = data[i * num_samples:(i + 1) * num_samples]
            tgt = labels[i * num_samples:(i + 1) * num_samples]
            logits = model(inp, fc_only=True)['logits']

            # Mask unseen classes
            seen = [c for t in range(task_id + 1) for c in class_mask[t]]
            not_seen = torch.tensor(
                np.setdiff1d(np.arange(nb_classes), seen),
                dtype=torch.int64).to(device)
            logits = logits.index_fill(1, not_seen, float('-inf'))

            loss = criterion(logits, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


# ---------------------------------------------------------------------------
# CL Training with Output Masking
# ---------------------------------------------------------------------------
# TRIBE NOTE: The class_mask is crucial for CL. During training on task T,
# only logits for T's classes contribute to loss. Our system handles this
# differently: each expert only sees its assigned patterns, so routing
# itself acts as an implicit mask.

def train_one_epoch_cl(model, criterion, data_loader, optimizer, device,
                       task_id, class_mask, nb_classes, max_norm=1.0):
    """Train one epoch on a single CL task with output masking."""
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)['logits']

        # TRIBE NOTE: Output masking -- set non-current-task logits to -inf
        not_mask = torch.tensor(
            np.setdiff1d(np.arange(nb_classes), class_mask[task_id]),
            dtype=torch.int64).to(device)
        logits = logits.index_fill(1, not_mask, float('-inf'))

        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += loss.item() * inputs.shape[0]
        total_correct += logits.max(1)[1].eq(targets).sum().item()
        total_samples += inputs.shape[0]

    return {'loss': total_loss / total_samples, 'acc': total_correct / total_samples}


# ---------------------------------------------------------------------------
# CL Evaluation with Forgetting Metrics
# ---------------------------------------------------------------------------
# TRIBE NOTE: Standard CL protocol. acc_matrix[i,j] = accuracy on task i
# after training task j. Forgetting = max_j(acc[i,j]) - acc[i,current].
# Our "backward transfer retained" (97-98%) measures acc[i,current]/acc[i,i].

def evaluate_cl(model, data_loaders, device, task_id, class_mask, nb_classes,
                acc_matrix=None):
    """Evaluate all tasks seen so far, compute forgetting metrics."""
    model.eval()
    if acc_matrix is None:
        acc_matrix = np.zeros((len(data_loaders), len(data_loaders)))

    for tid in range(task_id + 1):
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in data_loaders[tid]:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)['logits']
                mask_t = torch.tensor(class_mask[tid], dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits) * float('-inf')
                logits = logits + logits_mask.index_fill(1, mask_t, 0)
                correct += logits.max(1)[1].eq(targets).sum().item()
                total += inputs.size(0)
        acc_matrix[tid, task_id] = correct / total

    avg_acc = np.mean(acc_matrix[:task_id + 1, task_id])
    result = {'avg_acc': avg_acc}
    if task_id > 0:
        result['forgetting'] = np.mean(
            (np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
        result['backward_transfer'] = np.mean(
            (acc_matrix[:, task_id] - np.diag(acc_matrix))[:task_id])
    return result
