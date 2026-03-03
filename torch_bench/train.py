"""Training loop and evaluation for CL benchmark.

Provides domain training, perplexity evaluation, and the sequential
CL training pipeline shared across all configs.
"""

import math
import time
import torch
import numpy as np
from torch.utils.data import DataLoader


def train_domain(model, dataset, steps=500, lr=2e-4, batch_size=4,
                 device="cuda", extra_loss_fn=None, post_backward_fn=None,
                 extra_params=None, report_interval=None):
    """Train LoRA adapter on one domain for N steps.

    Args:
        model: PEFT model with LoRA adapters.
        dataset: TokenizedDataset of (seq_len,) token sequences.
        steps: number of gradient steps.
        lr: learning rate.
        batch_size: mini-batch size.
        device: torch device.
        extra_loss_fn: callable(model) -> scalar to add to NTP loss (e.g. EWC penalty).
        post_backward_fn: callable(model) after backward, before step (e.g. O-LoRA projection).
        extra_params: additional parameters to optimize (e.g. lifecycle gate biases).
        report_interval: print loss every N steps (default: steps//5).

    Returns:
        list of (step, loss_value) tuples at report intervals.
    """
    model.train()
    if report_interval is None:
        report_interval = max(steps // 5, 1)

    # Build optimizer with LoRA + any extra params
    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    if extra_params:
        param_groups.append({"params": list(extra_params), "lr": lr})
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0)
    data_iter = iter(dataloader)
    losses = []

    for step in range(steps):
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        tokens = batch.to(device)
        optimizer.zero_grad()

        # Forward: next-token prediction
        outputs = model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
        loss = outputs.loss

        # Optional extra loss (EWC penalty, etc.)
        if extra_loss_fn is not None:
            loss = loss + extra_loss_fn(model)

        loss.backward()

        # Optional post-backward processing (O-LoRA gradient projection, etc.)
        if post_backward_fn is not None:
            post_backward_fn(model)

        optimizer.step()

        if (step + 1) % report_interval == 0 or step == 0:
            loss_val = loss.item()
            ppl = math.exp(loss_val) if loss_val < 20 else float("inf")
            losses.append((step + 1, loss_val))
            print(f"      step {step+1:4d}/{steps}: loss={loss_val:.3f}, ppl={ppl:.1f}")

    return losses


def train_domain_with_replay(model, dataset, replay_buffer, steps=500,
                             lr=2e-4, batch_size=4, device="cuda",
                             report_interval=None):
    """Train with 50/50 current domain + replay buffer mix.

    Half the batch comes from the current domain, half from the replay
    buffer (if available). Otherwise trains normally.
    """
    model.train()
    if report_interval is None:
        report_interval = max(steps // 5, 1)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    half = max(batch_size // 2, 1)
    dataloader = DataLoader(dataset, batch_size=half, shuffle=True,
                            drop_last=True, num_workers=0)
    data_iter = iter(dataloader)
    losses = []

    for step in range(steps):
        try:
            current_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            current_batch = next(data_iter)

        current_tokens = current_batch.to(device)

        # Mix with replay
        replay_batch = replay_buffer.sample(half, device=device)
        if replay_batch is not None:
            tokens = torch.cat([current_tokens, replay_batch], dim=0)
        else:
            tokens = current_tokens

        optimizer.zero_grad()
        outputs = model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
        outputs.loss.backward()
        optimizer.step()

        if (step + 1) % report_interval == 0 or step == 0:
            loss_val = outputs.loss.item()
            ppl = math.exp(loss_val) if loss_val < 20 else float("inf")
            losses.append((step + 1, loss_val))
            print(f"      step {step+1:4d}/{steps}: loss={loss_val:.3f}, ppl={ppl:.1f}")

    return losses


@torch.no_grad()
def evaluate(model, eval_dataset, batch_size=8, device="cuda"):
    """Compute perplexity on an evaluation dataset.

    Returns:
        float: perplexity (exp of mean cross-entropy loss).
    """
    model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        tokens = batch.to(device)
        outputs = model(input_ids=tokens[:, :-1], labels=tokens[:, 1:])
        # Weight by number of tokens in this batch
        n_tokens = tokens[:, 1:].numel()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    model.train()
    return ppl


def evaluate_all_domains(model, eval_datasets, baseline_ppl, device="cuda"):
    """Evaluate model on all domain eval sets, print results.

    Args:
        model: the model to evaluate.
        eval_datasets: dict of domain_name -> TokenizedDataset.
        baseline_ppl: dict of domain_name -> float (base model ppl).
        device: torch device.

    Returns:
        dict of domain_name -> perplexity.
    """
    ppl_row = {}
    for domain_name, eval_ds in eval_datasets.items():
        ppl = evaluate(model, eval_ds, device=device)
        ppl_row[domain_name] = ppl
        delta = ppl - baseline_ppl[domain_name]
        marker = "+" if delta < -0.5 else ("-" if delta > 0.5 else "=")
        print(f"      {domain_name:12s}: ppl={ppl:7.1f} ({marker}{abs(delta):.1f})")
    return ppl_row


def compute_forgetting(ppl_matrix, domains):
    """Compute forgetting metrics from perplexity matrix.

    Args:
        ppl_matrix: dict of task_idx -> {domain: ppl}.
        domains: list of domain names in training order.

    Returns:
        dict of "T{i}>{domain}" -> {absolute, percent, before, after}.
    """
    forgetting = {}
    for task_idx in range(1, len(domains)):
        for prev_idx in range(task_idx):
            prev_domain = domains[prev_idx]
            ppl_before = ppl_matrix[prev_idx][prev_domain]
            ppl_after = ppl_matrix[task_idx][prev_domain]
            fgt = ppl_after - ppl_before
            pct = (fgt / ppl_before) * 100 if ppl_before > 0 else 0.0
            key = f"T{task_idx}>{prev_domain}"
            forgetting[key] = {
                "absolute": fgt,
                "percent": pct,
                "before": ppl_before,
                "after": ppl_after,
            }
    return forgetting
