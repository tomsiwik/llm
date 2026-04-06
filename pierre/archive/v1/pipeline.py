"""Pierre pipeline — wires router + null-space + NRE + pre-merge + generation.

Usage:
    pipe = Pipeline(model_id, adapters_dir, domains)
    pipe.calibrate(cal_data)   # builds ridge router from calibration data
    result = pipe.query(text)  # route → compose → generate
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from pierre.v1.router import RidgeRouter, RouterStatistics, solve_ridge
from pierre.v1.compose import nre_compose, premerge_into_model, TARGET_KEYS
from pierre.v1.nullspace import NullSpaceProjector


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()


class Pipeline:
    """End-to-end composable expert pipeline.

    Wires: ridge router → NRE composition → pre-merge → generation.
    Null-space projection is applied during composition when projectors are provided.

    Args:
        model_id: HuggingFace model ID for base model.
        adapters_dir: path containing domain adapter dirs + grassmannian_skeleton.npz.
        domains: list of domain names (matching adapter subdirectory names).
        lora_rank: LoRA rank (default 16).
        lora_scale: LoRA scaling factor (default 20.0).
        max_seq_length: max sequence length for tokenization (default 256).
    """

    def __init__(
        self,
        model_id: str,
        adapters_dir: str | Path,
        domains: list[str],
        lora_rank: int = 16,
        lora_scale: float = 20.0,
        max_seq_length: int = 256,
    ):
        self.model_id = model_id
        self.adapters_dir = Path(adapters_dir)
        self.domains = domains
        self.n_domains = len(domains)
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.max_seq_length = max_seq_length

        self.router: RidgeRouter | None = None
        self.null_space_projectors: dict[str, mx.array] | None = None
        self._calibrated = False

    def load_skeleton(self) -> dict[str, np.ndarray]:
        """Load Grassmannian A matrices from disk."""
        path = self.adapters_dir / "grassmannian_skeleton.npz"
        return dict(np.load(str(path)))

    def load_adapter(self, domain: str) -> dict[str, mx.array]:
        """Load a single domain adapter's B matrices."""
        path = self.adapters_dir / domain / "adapter.npz"
        return dict(mx.load(str(path)))

    def load_all_adapters(self) -> list[dict[str, mx.array]]:
        """Load all domain adapters. Returns list ordered by self.domains."""
        adapters = []
        for domain in self.domains:
            adapters.append(self.load_adapter(domain))
        return adapters

    def calibrate(
        self,
        model,
        tokenizer,
        cal_data: dict[str, list[str]],
        lam: float = 1.0,
        n_samples: int = 50,
    ) -> dict:
        """Build ridge regression router from calibration data.

        Args:
            model: loaded model (post BitLinear replacement).
            tokenizer: tokenizer for the model.
            cal_data: dict of domain -> list of text samples.
            lam: ridge regression lambda.
            n_samples: max samples per domain.

        Returns:
            Dict with calibration metrics (time, tokens_per_domain).
        """
        t0 = time.time()

        # Use mean-pooled last hidden state for routing (matches Finding #276)
        hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]

        stats = RouterStatistics(hidden_dim, self.n_domains)

        tokens_per_domain = {}
        for di, domain in enumerate(self.domains):
            samples = cal_data[domain][:n_samples]
            n_tokens = 0
            for text in samples:
                toks = tokenizer.encode(text)[: self.max_seq_length]
                if len(toks) < 4:
                    continue
                x = mx.array(toks)[None, :]

                # Get hidden states from last layer
                h = model.model.embed_tokens(x)
                for layer in model.model.layers:
                    h = layer(h, mask=None)
                mx.eval(h)

                # Mean pool over sequence
                pooled = mx.mean(h[0], axis=0, keepdims=True)  # (1, H)
                stats.update(pooled, di)
                n_tokens += len(toks)
                del h, pooled, x

            tokens_per_domain[domain] = n_tokens

        # Solve ridge regression
        W = solve_ridge(stats, lam=lam, column_normalize=True)

        # Create router
        self.router = RidgeRouter(hidden_dim, self.n_domains, top_k=1)
        self.router.weight = W
        self._calibrated = True

        elapsed = time.time() - t0
        return {
            "calibration_time_s": round(elapsed, 2),
            "tokens_per_domain": tokens_per_domain,
            "lambda": lam,
        }

    def route(self, model, tokenizer, text: str) -> tuple[list[int], list[float]]:
        """Route a query to top-k experts.

        Args:
            model: loaded model.
            tokenizer: tokenizer.
            text: input query text.

        Returns:
            (expert_indices, expert_weights) as Python lists.
        """
        if not self._calibrated:
            raise RuntimeError("Pipeline not calibrated. Call calibrate() first.")

        toks = tokenizer.encode(text)[: self.max_seq_length]
        x = mx.array(toks)[None, :]

        # Forward through model to get hidden states
        h = model.model.embed_tokens(x)
        for layer in model.model.layers:
            h = layer(h, mask=None)
        mx.eval(h)

        # Mean pool
        pooled = mx.mean(h[0], axis=0, keepdims=True)  # (1, H)
        indices, weights = self.router(pooled)
        mx.eval(indices, weights)

        idx_list = indices[0].tolist()
        w_list = weights[0].tolist()
        del h, pooled, x
        return idx_list, w_list

    def compose_and_merge(
        self,
        model,
        expert_indices: list[int],
        expert_weights: list[float],
    ) -> int:
        """Compose selected adapters via NRE and pre-merge into model.

        Args:
            model: model with nn.Linear layers (post BitLinear replacement).
            expert_indices: selected expert indices from route().
            expert_weights: corresponding weights.

        Returns:
            Number of layers modified.
        """
        skeleton = self.load_skeleton()

        # Load selected adapters
        selected_b = []
        selected_w = []
        for idx, w in zip(expert_indices, expert_weights):
            domain = self.domains[idx]
            adapter = self.load_adapter(domain)
            selected_b.append(adapter)
            selected_w.append(w)

        # NRE compose
        composed = nre_compose(selected_b, selected_w)

        # Pre-merge into model
        n_layers = len(model.model.layers)
        merge_count = premerge_into_model(
            model, skeleton, composed, self.lora_scale, n_layers
        )

        cleanup(skeleton, selected_b, composed)
        return merge_count
