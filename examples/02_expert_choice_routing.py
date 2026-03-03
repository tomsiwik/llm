"""02 -- Expert Choice Routing (Inverted Top-k)

Paper: Mixture-of-Experts with Expert Choice Routing
       (Zhou, Lei, Liu, Du, Huang, Zhao, Dai, Chen, Le & Laudon, 2022)
URL:   https://arxiv.org/abs/2202.09368
Repo:  Reference implementation adapted from kaiqiancui/GT-MoE
       (src/baselines/expert_choice_new.py) and the paper description.

Standard MoE routing: each *token* picks its top-k experts.
Expert Choice routing: each *expert* picks its top-k tokens.

This inversion guarantees perfect load balance by construction -- every expert
processes exactly `capacity` tokens -- eliminating the need for auxiliary
load-balancing losses or token dropping.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Expert Choice MoE Layer
# ---------------------------------------------------------------------------

class ExpertChoiceMoELayer(nn.Module):
    """Mixture-of-Experts layer with Expert Choice routing.

    Key idea (Section 2.2 of the paper):
      1. Compute router scores  S = softmax(x @ W_g)   shape [n_tokens, n_experts]
      2. Transpose to get       S^T                     shape [n_experts, n_tokens]
      3. Each expert selects its top-k tokens from S^T (along the token dimension)
      4. k = capacity = ceil(n_tokens / n_experts * capacity_factor)

    Because every expert selects exactly k tokens, load is perfectly balanced.
    No auxiliary loss is needed.

    # TRIBE NOTE: This inverts the routing direction compared to both Switch
    # Transformer (file 01) and our `route_by_loss()`.  In TRIBE, tokens pick
    # experts (oracle-style via loss evaluation).  In Expert Choice, experts
    # pick tokens.  The advantage: guaranteed load balance without any
    # auxiliary loss or reactive lifecycle.  The disadvantage: a token may be
    # selected by zero experts (effectively dropped) or by many experts
    # (duplicated computation).  Our `health_check()` detects overload
    # *reactively* via gradient norms; Expert Choice prevents it *proactively*
    # via the capacity constraint.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float,
        intermediate_size: int,
    ):
        """
        Args:
            hidden_size:      model dimension (d_model).
            num_experts:      number of expert FFN copies.
            capacity_factor:  multiplier on ideal per-expert budget.
                capacity = ceil(n_tokens / n_experts * capacity_factor).
                1.0 means each expert sees exactly its fair share.
            intermediate_size: hidden dimension inside each expert FFN.
        """
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Router: projects tokens into expert-selection logits
        # TRIBE NOTE: Same linear router as Switch (file 01), but the top-k
        # is taken along the *token* axis (by each expert) rather than the
        # *expert* axis (by each token).
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert FFNs (SwiGLU-style: w2(silu(w1(x)) * w3(x)))
        self.experts = nn.ModuleList([
            _SwiGLU_FFN(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)  # [n_tokens, hidden_size]
        n_tokens = x_flat.shape[0]

        # --- Step 1: Compute router scores ---
        # S[t, e] = softmax over experts for token t
        router_logits = self.gate(x_flat)          # [n_tokens, n_experts]

        # --- Step 2: Transpose so each expert can select tokens ---
        # S^T[e, t] -- each row is one expert's view of all tokens
        expert_scores = router_logits.t()          # [n_experts, n_tokens]

        # --- Step 3: Each expert selects its top-k tokens ---
        # TRIBE NOTE: `capacity` is the budget per expert.  With
        # capacity_factor=1.0, total slots = n_experts * capacity = n_tokens,
        # so every token is processed on average once.  Increasing
        # capacity_factor gives experts more budget (like our top_k=2 in
        # the CNN experiments).
        capacity = math.ceil((n_tokens / self.num_experts) * self.capacity_factor)
        capacity = min(capacity, n_tokens)  # can't exceed total tokens

        # Top-k along the token dimension (dim=1), per expert
        # TRIBE NOTE: This is the inversion.  In Switch, we do
        #   topk(S, dim=expert_axis)  -- token picks expert
        # Here we do
        #   topk(S^T, dim=token_axis) -- expert picks token
        top_scores, top_indices = torch.topk(
            expert_scores, k=capacity, dim=1
        )  # both [n_experts, capacity]

        # --- Step 4: Run each expert on its selected tokens ---
        output_flat = torch.zeros_like(x_flat)  # [n_tokens, hidden_size]

        for expert_idx in range(self.num_experts):
            # Gather the tokens this expert selected
            token_indices = top_indices[expert_idx]          # [capacity]
            expert_input = x_flat[token_indices]             # [capacity, hidden_size]

            # Forward through this expert
            expert_output = self.experts[expert_idx](expert_input)  # [capacity, hidden_size]

            # Weight by the (normalised) routing score
            # The paper uses softmax-normalised scores as combination weights.
            # TRIBE NOTE: These weights serve the same purpose as the
            # probability scaling in Switch (is_scale_prob=True).  They give
            # the router gradient signal about how much to trust each expert.
            weights = router_logits[token_indices, expert_idx].unsqueeze(-1)
            weights = F.softmax(weights, dim=0)  # [capacity, 1]

            # Scatter-add back to token positions
            # TRIBE NOTE: A token selected by multiple experts gets its
            # outputs *summed*.  This is analogous to our top_k=2 routing
            # where a sample trains 2 experts simultaneously.  The key
            # difference: we *choose* which 2 via loss; here the experts
            # choose which tokens look useful to them.
            output_flat.index_add_(
                0, token_indices, expert_output * weights
            )

        return output_flat.reshape(batch_size, seq_len, hidden_size)


# ---------------------------------------------------------------------------
# Expert FFN (SwiGLU variant used in modern transformers)
# ---------------------------------------------------------------------------

class _SwiGLU_FFN(nn.Module):
    """SwiGLU feed-forward: out = W2( SiLU(W1(x)) * W3(x) )."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Comparison: Token-Choice vs Expert-Choice at a glance
# ---------------------------------------------------------------------------
#
#  Aspect              Token-Choice (Switch)        Expert-Choice (this file)
#  ------              ---------------------        -------------------------
#  Who selects?        Token picks 1 expert         Expert picks k tokens
#  Load balance        Needs aux loss               Guaranteed by construction
#  Token coverage      Every token is processed     Some tokens may be skipped
#  Capacity factor     Caps max load per expert     Sets exact load per expert
#  Router gradient     Through selected expert      Through selected tokens
#
# TRIBE NOTE: Our oracle routing (`route_by_loss`) is a third option:
#   - Who selects?  Token picks expert, but via *loss evaluation* not learned score
#   - Load balance? No guarantee; handled reactively by lifecycle
#   - Coverage?     Every token is processed (we always pick *some* expert)
#   - Cost?         O(n_experts) forward passes per routing decision
