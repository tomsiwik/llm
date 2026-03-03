"""PEER + FoX Lifecycle: neuron-level continual learning for LLMs.

Combines three ideas:
1. PEER (He 2024): Product-key routing over 1024+ single-neuron experts.
   Massive capacity pool, tiny active footprint (32/1024 per token).
2. FoX (Lin et al. 2025): Per-neuron forget gate f=sigmoid(b) that
   interpolates parent vs child knowledge. One neuron serves BOTH old
   and new tasks: effective_v = f*child_v + (1-f)*parent_v.
3. Tribe lifecycle: freeze + clone-with-gate + emancipate + recycle
   at neuron granularity.

The forget gate directly solves the preservation-coverage trade-off:
capacity is no longer zero-sum because a single neuron can hold two
generations of knowledge simultaneously.

Ref: PEER — "Mixture of a Million Experts" (He 2024)
     FoX  — "Forgetting Transformer" (Lin et al. 2025)
"""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _isqrt(n):
    """Integer square root."""
    r = int(math.isqrt(n))
    assert r * r == n, f"n_experts={n} must be a perfect square"
    return r


class ExpertSnapshot(nn.Module):
    """Frozen snapshot of expert weight_up values with trainable FoX gates.

    Stores a generation of preserved knowledge. weight_up is frozen (stop_gradient
    in forward), gate_bias is trainable (FoX interpolation with live weights).
    """

    def __init__(self, weight_up, gate_bias_init, mask, domain=""):
        super().__init__()
        self.weight_up = weight_up          # (N, d_out), will be FROZEN
        self.gate_bias = gate_bias_init     # (N,), TRAINABLE
        self._mask = mask                    # numpy (N,) bool
        self._mask_2d = mx.expand_dims(mx.array(mask.astype(np.float32)), -1)
        self._domain = domain

    @property
    def n_active(self):
        return int(self._mask.sum())


class PEERLifecycleLinear(nn.Module):
    """Drop-in replacement for nn.Linear with PEER routing + FoX forget gates.

    Each expert has:
      - u (d_in,):  down-projection / routing key
      - v (d_out,): up-projection (child value)
      - b (scalar): forget gate bias (active only when has_parent)

    Forward: base_out + scale * delta
      where delta = sum over selected experts of:
        score_i * GELU(u_i · x) * effective_v_i

      effective_v_i = f*child_v + (1-f)*parent_v  if has_parent
                    = child_v                       otherwise
      f = sigmoid(b_i)

    Product key routing selects n_active experts from n_experts via:
      1. Project x → (q1, q2) each of dim d_key
      2. Score against √N sub-keys per half
      3. Top-pk per half → Cartesian product → top n_active final
    """

    def __init__(self, d_in, d_out, n_experts=1024, n_active=32,
                 pk=8, d_key=None, base_weight=None, base_bias=None,
                 scale=1.0):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.n_experts = n_experts
        self._n_active_cfg_val = n_active
        self.pk = pk
        self.scale = scale

        self._sqrt_n = _isqrt(n_experts)
        self._d_key = d_key or (d_in // 2)

        # Frozen base linear
        if base_weight is not None:
            self.weight = base_weight
        if base_bias is not None:
            self.bias = base_bias
        self._has_base = base_weight is not None
        self._has_bias = base_bias is not None

        # Expert parameters
        self.weight_down = mx.random.normal((n_experts, d_in)) * (1.0 / math.sqrt(d_in))
        self.weight_up = mx.zeros((n_experts, d_out))

        # Product key routing
        sk = mx.random.normal((2, self._sqrt_n, self._d_key)) * 0.02
        # L2-normalize sub-keys
        norms = mx.sqrt(mx.sum(sk * sk, axis=-1, keepdims=True) + 1e-8)
        self.sub_keys = sk / norms
        self.query_proj = mx.random.normal((2 * self._d_key, d_in)) * (1.0 / math.sqrt(d_in))

        # Forget gate bias (per expert, inactive until clone sets -4.0)
        self.gate_bias = mx.zeros((n_experts,))

        # Non-parameter state (numpy, underscore prefix → invisible to MLX)
        self._frozen = np.zeros(n_experts, dtype=bool)
        self._has_parent = np.zeros(n_experts, dtype=bool)
        self._parent_idx = np.full(n_experts, -1, dtype=np.int32)

        # Cached mx arrays for hot path (rebuilt on lifecycle ops)
        self._cached_active_mx = None  # (N,) float32 — 1 for active, 0 for frozen
        self._cached_hp_mx = None      # (N, 1) float32 — has_parent mask
        self._cached_pidx_mx = None    # (N,) int32 — parent indices (clamped ≥ 0)
        self._any_parent = False

        # Version tree snapshots (oldest → newest)
        self.snapshots = []

    def _invalidate_cache(self):
        """Invalidate cached mx arrays after lifecycle state changes."""
        self._cached_active_mx = None
        self._cached_hp_mx = None
        self._cached_pidx_mx = None
        self._any_parent = bool(self._has_parent.any())

    def _ensure_cache(self):
        """Lazily build cached mx arrays from numpy state."""
        if self._cached_active_mx is None:
            self._cached_active_mx = mx.array(
                (~self._frozen).astype(np.float32)
            )
        if self._any_parent and self._cached_hp_mx is None:
            self._cached_hp_mx = mx.expand_dims(
                mx.array(self._has_parent.astype(np.float32)), -1
            )
            self._cached_pidx_mx = mx.array(
                np.maximum(self._parent_idx, 0)
            )

    def compute_delta(self, x):
        """Compute PEER expert delta without base linear.

        Returns (*x.shape[:-1], d_out) delta tensor.

        Uses fully dense matmuls instead of sparse gathers for Metal perf:
        1. Dense routing: full outer-sum scores → top-k mask (argpartition)
        2. Dense down-projection: x @ weight_down.T → (B, N)
        3. Dense up-projection: coeff @ effective_up → (B, d_out)
        """
        # Flatten batch dims: (..., d_in) → (B, d_in)
        orig_shape = x.shape
        x_flat = mx.reshape(x, (-1, self.d_in))
        B = x_flat.shape[0]
        N = self.n_experts
        n_active = self._n_active_cfg_val

        # 1. Product key routing — dense full scores
        q = x_flat @ self.query_proj.T                           # (B, 2*d_key)
        q1 = q[:, :self._d_key]                                  # (B, d_key)
        q2 = q[:, self._d_key:]                                  # (B, d_key)
        s1 = q1 @ self.sub_keys[0].T                             # (B, √N)
        s2 = q2 @ self.sub_keys[1].T                             # (B, √N)
        # Outer sum → full routing scores for all N experts
        full_scores = (mx.expand_dims(s1, 2)
                       + mx.expand_dims(s2, 1))                  # (B, √N, √N)
        full_scores = mx.reshape(full_scores, (B, N))            # (B, N)

        # Top-k masking via argpartition (O(N) vs O(N log N) for sort)
        kth_pos = mx.argpartition(-full_scores, kth=n_active - 1,
                                  axis=-1)[:, n_active - 1:n_active]
        kth_vals = mx.take_along_axis(full_scores, kth_pos, axis=-1)
        mask = mx.stop_gradient((full_scores >= kth_vals).astype(mx.float32))
        active_scores = nn.relu(full_scores) * mask              # (B, N)

        # 2. Dense down-projection — all experts at once
        all_proj = nn.gelu(x_flat @ self.weight_down.T)          # (B, N)
        coeff = active_scores * all_proj                         # (B, N) mostly zero

        # 3. Effective weight_up with forget gates (precomputed once, not per-token)
        effective_up = self._effective_weight_up()                # (N, d_out)

        # 4. Dense up-projection
        delta = coeff @ effective_up                              # (B, d_out)

        # Reshape back to original batch dims
        return mx.reshape(delta, (*orig_shape[:-1], self.d_out))

    def __call__(self, x):
        """Forward: base linear + PEER routed expert delta."""
        # Base linear (frozen) — fused addmm when bias present
        if self._has_base:
            if self._has_bias:
                x_flat_base = mx.reshape(x, (-1, self.d_in))
                out = mx.addmm(self.bias, x_flat_base, self.weight)
                out = mx.reshape(out, (*x.shape[:-1], self.d_out))
            else:
                out = x @ self.weight.T
        else:
            out = mx.zeros((*x.shape[:-1], self.d_out))

        return out + self.scale * self.compute_delta(x)

    def _effective_weight_up(self):
        """Compute effective weight_up with snapshot chain + legacy parent gates.

        Resolution order:
        1. Start with live weight_up
        2. Apply snapshot chain (oldest → newest): FoX interpolation at snapshotted positions
        3. Apply legacy parent-child gates (backward compat with clone_with_gate)
        """
        result = self.weight_up

        # Snapshot chain (oldest → newest)
        for snap in self.snapshots:
            f_2d = mx.expand_dims(mx.sigmoid(snap.gate_bias), -1)  # (N, 1)
            snap_wu = mx.stop_gradient(snap.weight_up)              # (N, d_out)
            blended = f_2d * result + (1.0 - f_2d) * snap_wu
            result = snap._mask_2d * blended + (1.0 - snap._mask_2d) * result

        # Legacy parent-child (backward compat)
        if not self._any_parent:
            return result

        self._ensure_cache()
        f_2d = mx.expand_dims(mx.sigmoid(self.gate_bias), -1)   # (N, 1)
        v_parent = mx.stop_gradient(
            result[self._cached_pidx_mx]                          # (N, d_out)
        )
        hp_2d = self._cached_hp_mx                                # (N, 1)
        gated = f_2d * result + (1.0 - f_2d) * v_parent
        return hp_2d * gated + (1.0 - hp_2d) * result

    # ── Lifecycle Operations ────────────────────────────────

    def freeze_experts(self, indices):
        """Freeze experts (still fire in forward, but gradients zeroed)."""
        for i in indices:
            self._frozen[i] = True
        self._invalidate_cache()

    def clone_with_gate(self, parent_idx, child_idx):
        """Create parent→child lineage with forget gate.

        - Copy parent's weight_down and weight_up into child slot
        - Set gate_bias[child] = -4.0 (sigmoid ≈ 0.018, trust parent 98%)
        - Set parent pointer
        - Freeze parent
        """
        wd = np.array(self.weight_down)
        wu = np.array(self.weight_up)
        gb = np.array(self.gate_bias)

        wd[child_idx] = wd[parent_idx]
        wu[child_idx] = wu[parent_idx]
        gb[child_idx] = -4.0

        self.weight_down = mx.array(wd)
        self.weight_up = mx.array(wu)
        self.gate_bias = mx.array(gb)

        self._has_parent[child_idx] = True
        self._parent_idx[child_idx] = parent_idx
        self._frozen[parent_idx] = True
        self._invalidate_cache()

    def emancipate(self, expert_idx, threshold=0.9):
        """Emancipate child when gate > threshold (child has replaced parent).

        Bakes the interpolation into weight_up, clears parent pointer, resets gate.
        Returns True if emancipation happened.
        """
        gb = np.array(self.gate_bias)
        f = 1.0 / (1.0 + np.exp(-gb[expert_idx]))
        if f < threshold:
            return False

        wu = np.array(self.weight_up)
        parent = self._parent_idx[expert_idx]
        if parent >= 0:
            # Bake: v = f*child + (1-f)*parent
            wu[expert_idx] = f * wu[expert_idx] + (1.0 - f) * wu[parent]
        gb[expert_idx] = 0.0

        self.weight_up = mx.array(wu)
        self.gate_bias = mx.array(gb)

        self._has_parent[expert_idx] = False
        self._parent_idx[expert_idx] = -1
        self._invalidate_cache()
        return True

    def snapshot_and_recycle(self, indices, domain="", recycle=True, gate_init=-4.0):
        """Snapshot weight_up at indices in-place, optionally recycle original slots.

        Creates an ExpertSnapshot with frozen weight_up and trainable gate_bias.
        The snapshot preserves knowledge; the original slot is zeroed and unfrozen
        for new learning. FoX gates in _effective_weight_up blend snapshot + live.

        Args:
            indices: expert indices to snapshot (list/array of ints).
            domain: label for this snapshot generation.
            recycle: if True, zero weight_up and unfreeze at snapshotted positions.
            gate_init: initial gate_bias for FoX gates (default -4.0 → sigmoid≈0.018,
                       trust snapshot 98%).

        Returns:
            ExpertSnapshot: the created snapshot.
        """
        indices = np.array(indices, dtype=np.int32)

        # Build mask
        mask = np.zeros(self.n_experts, dtype=bool)
        mask[indices] = True

        # Deep-copy weight_up for snapshot
        snap_wu = mx.array(np.array(self.weight_up))

        # Gate bias: gate_init at snapshotted positions, 0 elsewhere
        gb = np.zeros(self.n_experts, dtype=np.float32)
        gb[indices] = gate_init
        snap_gb = mx.array(gb)

        # Create snapshot and freeze its weight_up (gate_bias stays trainable)
        snap = ExpertSnapshot(snap_wu, snap_gb, mask, domain=domain)
        snap.freeze(keys=["weight_up"])
        self.snapshots.append(snap)

        # Recycle original slots: zero weight_up for new learning.
        # Keep _frozen=True so mask_peer_frozen_gradients anchors weight_down
        # (routing keys) at snapshotted positions — prevents routing drift.
        # weight_up gradients at frozen positions are also zeroed by the mask,
        # but the FoX gate still receives gradients (it's on the snapshot, not
        # the layer). New learning happens at OTHER (unfrozen) expert positions.
        if recycle:
            wu = np.array(self.weight_up)
            wu[indices] = 0.0
            self.weight_up = mx.array(wu)
            self._frozen[indices] = True

        self._invalidate_cache()
        return snap

    def recycle_experts(self, indices):
        """Reinit dead experts: random weight_down, zero weight_up, clear state."""
        wd = np.array(self.weight_down)
        wu = np.array(self.weight_up)
        gb = np.array(self.gate_bias)

        for i in indices:
            wd[i] = np.random.randn(self.d_in) / math.sqrt(self.d_in)
            wu[i] = 0.0
            gb[i] = 0.0
            self._frozen[i] = False
            self._has_parent[i] = False
            self._parent_idx[i] = -1

        self.weight_down = mx.array(wd)
        self.weight_up = mx.array(wu)
        self.gate_bias = mx.array(gb)
        self._invalidate_cache()

    def expert_importance(self):
        """Per-expert importance as ||weight_up[i]|| (L2 norm)."""
        wu = np.array(self.weight_up)
        return np.linalg.norm(wu, axis=1)

    @property
    def n_frozen(self):
        return int(self._frozen.sum())

    @property
    def n_unfrozen(self):
        return self.n_experts - self.n_frozen

    @property
    def n_gated(self):
        return int(self._has_parent.sum())

    @property
    def frozen_mask(self):
        return self._frozen.copy()

    def gate_values(self):
        """Return sigmoid(gate_bias) for all experts as numpy array."""
        gb = np.array(self.gate_bias)
        return 1.0 / (1.0 + np.exp(-gb))

    # ── Snapshot introspection ─────────────────────────────────

    @property
    def n_snapshots(self):
        """Number of snapshot generations."""
        return len(self.snapshots)

    @property
    def n_preserved(self):
        """Total number of preserved expert positions across all snapshots."""
        return sum(snap.n_active for snap in self.snapshots)

    def snapshot_gate_values(self):
        """Return per-snapshot gate statistics.

        Returns:
            list of dicts with 'domain', 'n_active', 'mean', 'min', 'max'
            for each snapshot generation.
        """
        stats = []
        for snap in self.snapshots:
            gb = np.array(snap.gate_bias)
            gates = 1.0 / (1.0 + np.exp(-gb))
            active_gates = gates[snap._mask]
            if len(active_gates) > 0:
                stats.append({
                    'domain': snap._domain,
                    'n_active': snap.n_active,
                    'mean': float(active_gates.mean()),
                    'min': float(active_gates.min()),
                    'max': float(active_gates.max()),
                })
        return stats

    def __repr__(self):
        frozen = f", frozen={self.n_frozen}" if self.n_frozen > 0 else ""
        gated = f", gated={self.n_gated}" if self.n_gated > 0 else ""
        snaps = f", snapshots={self.n_snapshots}" if self.n_snapshots > 0 else ""
        return (f"PEERLifecycleLinear(d_in={self.d_in}, d_out={self.d_out}, "
                f"experts={self.n_experts}, active={self._n_active_cfg_val}"
                f"{frozen}{gated}{snaps})")



class ParallelPEERLayer(nn.Module):
    """Multiple competing PEER branches with input-dependent gating.

    Each branch is an independent PEERLifecycleLinear that computes additive
    deltas. A learned softmax gate selects per-token how much each branch
    contributes. Between domains, freeze the best branch entirely (complete
    routing isolation) and train remaining/new branches.

    Forward:
        base_out = x @ base_weight.T + base_bias
        gate = softmax(x @ gate_proj.T + gate_bias_vec)  # (B, n_branches)
        delta_i = branch_i.compute_delta(x)               # per branch
        combined = sum(gate_i * delta_i)
        out = base_out + scale * combined
    """

    def __init__(self, d_in, d_out, n_branches=2, n_experts=529,
                 n_active=17, pk=8, d_key=None, base_weight=None,
                 base_bias=None, scale=1.0, use_streams=False):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self._n_branches_val = n_branches
        self.scale = scale
        self._use_streams = use_streams

        # Frozen base linear (shared, computed once)
        if base_weight is not None:
            self.weight = base_weight
        if base_bias is not None:
            self.bias = base_bias
        self._has_base = base_weight is not None
        self._has_bias = base_bias is not None

        # Per-token gate: softmax(x @ gate_proj.T + gate_bias_vec)
        self.gate_proj = mx.random.normal((n_branches, d_in)) * (1.0 / math.sqrt(d_in))
        self.gate_bias_vec = mx.zeros((n_branches,))

        # Independent PEER branches (delta-only, no base linear)
        self.branches = [
            PEERLifecycleLinear(
                d_in=d_in, d_out=d_out,
                n_experts=n_experts, n_active=n_active, pk=pk,
                d_key=d_key, base_weight=None, base_bias=None, scale=1.0,
            )
            for _ in range(n_branches)
        ]

        # Branch-level lifecycle state (numpy, underscore → invisible to MLX)
        self._branch_frozen = np.zeros(n_branches, dtype=bool)
        self._branch_domain = np.full(n_branches, "", dtype=object)

        # Streams set externally via set_streams() for cross-layer sharing.
        # Creating per-layer streams causes 120+ Metal command queues (60 layers
        # × 2 branches), triggering Metal's ~5s watchdog timeout on large models.
        # Instead, create 2 streams globally and share across all layers.
        self._streams = None

    def __call__(self, x):
        # 1. Base linear (once, fused addmm when bias present)
        if self._has_base:
            if self._has_bias:
                x_flat_base = mx.reshape(x, (-1, self.d_in))
                base_out = mx.addmm(self.bias, x_flat_base, self.weight)
                base_out = mx.reshape(base_out, (*x.shape[:-1], self.d_out))
            else:
                base_out = x @ self.weight.T
        else:
            base_out = mx.zeros((*x.shape[:-1], self.d_out))

        # 2. Gate: per-token branch weights
        orig_shape = x.shape
        x_flat = mx.reshape(x, (-1, self.d_in))  # (B, d_in)
        gate_logits = x_flat @ self.gate_proj.T + self.gate_bias_vec  # (B, n_branches)
        gate_weights = mx.softmax(gate_logits, axis=-1)               # (B, n_branches)

        # 3. Branch deltas (optionally dispatched to pre-created streams)
        # Streams are created once in __init__ and reused — never per call.
        # MLX auto-inserts cross-stream dependencies when we combine results.
        deltas = []
        if self._streams is not None:
            for branch, stream in zip(self.branches, self._streams):
                with mx.stream(stream):
                    delta = branch.compute_delta(x)
                    delta_flat = mx.reshape(delta, (-1, self.d_out))
                    deltas.append(delta_flat)
        else:
            for branch in self.branches:
                delta = branch.compute_delta(x)
                delta_flat = mx.reshape(delta, (-1, self.d_out))
                deltas.append(delta_flat)

        # 4. Weighted combination
        # delta_stack: (B, n_branches, d_out)
        delta_stack = mx.stack(deltas, axis=1)
        # gate_weights: (B, n_branches) → (B, n_branches, 1)
        gw = mx.expand_dims(gate_weights, -1)
        combined = mx.sum(gw * delta_stack, axis=1)  # (B, d_out)
        combined = mx.reshape(combined, (*orig_shape[:-1], self.d_out))

        return base_out + self.scale * combined

    def set_streams(self, streams):
        """Set pre-created streams for branch dispatch (shared across layers).

        Args:
            streams: list of mx.Stream (one per branch), or None to disable.
        """
        self._streams = streams

    # ── Branch Lifecycle ─────────────────────────────────────

    def freeze_branch(self, idx, domain_label=""):
        """Freeze all experts in a branch (complete routing isolation)."""
        branch = self.branches[idx]
        all_indices = np.arange(branch.n_experts)
        branch.freeze_experts(all_indices)
        self._branch_frozen[idx] = True
        self._branch_domain[idx] = domain_label

    @property
    def n_active_branches(self):
        """Count of unfrozen branches."""
        return int((~self._branch_frozen).sum())

    def branch_importance(self):
        """Mean expert importance per branch. Returns (n_branches,) numpy array."""
        imp = np.zeros(self._n_branches_val)
        for i, branch in enumerate(self.branches):
            imp[i] = branch.expert_importance().mean()
        return imp

    def __repr__(self):
        frozen = sum(self._branch_frozen)
        return (f"ParallelPEERLayer(d_in={self.d_in}, d_out={self.d_out}, "
                f"branches={self._n_branches_val}, "
                f"frozen_branches={frozen})")


# ── Module-level utilities ──────────────────────────────────

def collect_peer_layers(model):
    """Find all PEERLifecycleLinear layers in a model."""
    results = []

    def _search(module, prefix=""):
        if isinstance(module, PEERLifecycleLinear):
            results.append((prefix, module))
            return
        if isinstance(module, nn.Module):
            children = module.children()
            for name, child in children.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Module):
                    _search(child, full_name)
                elif isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        _search(item, f"{full_name}.{i}")
                elif isinstance(child, dict):
                    for k, v in child.items():
                        _search(v, f"{full_name}.{k}")

    _search(model)
    return results


def mask_peer_frozen_gradients(model, grads):
    """Zero gradients for frozen expert rows across all PEER layers.

    Uses cached mx arrays to avoid numpy→mx conversion every step.
    """
    for name, layer in collect_peer_layers(model):
        if layer.n_frozen == 0:
            continue

        layer._ensure_cache()
        active_2d = mx.expand_dims(layer._cached_active_mx, 1)  # (N, 1)

        parts = name.split(".")
        node = grads
        try:
            for part in parts:
                if part.isdigit():
                    node = node[int(part)]
                else:
                    node = node[part]
            if "weight_down" in node:
                node["weight_down"] = node["weight_down"] * active_2d
            if "weight_up" in node:
                node["weight_up"] = node["weight_up"] * active_2d
        except (KeyError, IndexError, TypeError):
            pass

    return grads


def freeze_top_peer_experts(model, n_per_layer, domain_label=None):
    """Freeze the most important experts in each PEER layer.

    Returns:
        dict: layer_name → list of frozen expert indices.
    """
    frozen_map = {}
    for name, layer in collect_peer_layers(model):
        importance = layer.expert_importance()
        active_mask = ~layer.frozen_mask
        importance[~active_mask] = -1

        n_act = int(active_mask.sum())
        n_to_freeze = min(n_per_layer, n_act)
        if n_to_freeze <= 0:
            frozen_map[name] = []
            continue

        top_indices = np.argsort(importance)[::-1][:n_to_freeze]
        layer.freeze_experts(top_indices)
        frozen_map[name] = top_indices.tolist()

    total_frozen = sum(l.n_frozen for _, l in collect_peer_layers(model))
    total_experts = sum(l.n_experts for _, l in collect_peer_layers(model))
    label = f" ({domain_label})" if domain_label else ""
    print(f"    FREEZE{label}: {n_per_layer}/layer, "
          f"total frozen={total_frozen}/{total_experts}")
    return frozen_map


def clone_frozen_to_children(model, frozen_map):
    """Clone frozen experts into fresh child slots with forget gates.

    For each frozen expert, picks the lowest-importance active slot as child.

    Returns:
        dict: layer_name → list of (parent, child) tuples.
    """
    clone_map = {}
    for name, layer in collect_peer_layers(model):
        frozen_indices = frozen_map.get(name, [])
        if not frozen_indices:
            clone_map[name] = []
            continue

        importance = layer.expert_importance()
        clones = []

        for parent in frozen_indices:
            # Find lowest-importance active, non-parent slot
            active_mask = ~layer._frozen & ~layer._has_parent
            candidates = np.where(active_mask)[0]
            if len(candidates) == 0:
                continue
            cand_importance = importance[candidates]
            child = candidates[np.argmin(cand_importance)]
            layer.clone_with_gate(parent, child)
            clones.append((parent, child))

        clone_map[name] = clones

    total_clones = sum(len(v) for v in clone_map.values())
    total_gated = sum(l.n_gated for _, l in collect_peer_layers(model))
    print(f"    CLONE: {total_clones} parent→child pairs, "
          f"total gated={total_gated}")
    return clone_map


def emancipate_mature_children(model, threshold=0.9):
    """Emancipate children whose gates exceed threshold.

    Returns:
        list of (layer_name, expert_idx, gate_value) tuples.
    """
    emancipated = []
    for name, layer in collect_peer_layers(model):
        gated_indices = np.where(layer._has_parent)[0]
        for idx in gated_indices:
            gb = np.array(layer.gate_bias)
            f = 1.0 / (1.0 + np.exp(-gb[idx]))
            if layer.emancipate(idx, threshold=threshold):
                emancipated.append((name, int(idx), float(f)))

    if emancipated:
        print(f"    EMANCIPATE: {len(emancipated)} children "
              f"(gates: {[f'{g:.2f}' for _, _, g in emancipated]})")
    return emancipated


def recycle_dead_experts(model, threshold=1e-6):
    """Recycle experts with near-zero importance that are active and parentless.

    Returns:
        int: number recycled.
    """
    total = 0
    for name, layer in collect_peer_layers(model):
        importance = layer.expert_importance()
        candidates = np.where(
            (~layer._frozen) & (~layer._has_parent) & (importance < threshold)
        )[0]
        if len(candidates) > 0:
            layer.recycle_experts(candidates)
            total += len(candidates)

    if total > 0:
        print(f"    RECYCLE: {total} dead experts reinitialized")
    return total


def routing_neighbors(expert_idx, sqrt_n, radius=1):
    """Find 2D grid neighbors in the √N × √N product-key grid.

    Product keys form a grid where expert i maps to (i // √N, i % √N).
    Returns neighbor indices within Manhattan distance ≤ radius.

    Args:
        expert_idx: single expert index.
        sqrt_n: √N (grid side length).
        radius: neighborhood radius (default 1 = 8 neighbors).

    Returns:
        numpy array of neighbor indices (excluding self).
    """
    row, col = expert_idx // sqrt_n, expert_idx % sqrt_n
    neighbors = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < sqrt_n and 0 <= c < sqrt_n:
                neighbors.append(r * sqrt_n + c)
    return np.array(neighbors, dtype=np.int32)


def snapshot_top_experts(model, n_per_layer, domain_label="", recycle=True,
                         gate_init=-4.0, include_neighbors=False):
    """Snapshot the most important experts in each PEER layer.

    Like freeze_top_peer_experts but uses snapshot_and_recycle instead:
    frozen snapshot preserves knowledge, original slot recycled for new learning.

    Args:
        model: model containing PEERLifecycleLinear layers.
        n_per_layer: number of experts to snapshot per layer.
        domain_label: label for this snapshot generation.
        recycle: if True, zero and unfreeze snapshotted slots.
        gate_init: initial FoX gate bias (default -4.0).
        include_neighbors: if True, also snapshot routing neighbors.

    Returns:
        dict: layer_name → list of snapshotted expert indices.
    """
    snap_map = {}
    for name, layer in collect_peer_layers(model):
        importance = layer.expert_importance()
        # Only snapshot active (unfrozen) experts
        active_mask = ~layer.frozen_mask
        importance[~active_mask] = -1

        n_act = int(active_mask.sum())
        n_to_snap = min(n_per_layer, n_act)
        if n_to_snap <= 0:
            snap_map[name] = []
            continue

        top_indices = np.argsort(importance)[::-1][:n_to_snap]

        # Optionally include routing neighbors
        if include_neighbors:
            all_indices = set(top_indices.tolist())
            for idx in top_indices:
                nbrs = routing_neighbors(idx, layer._sqrt_n)
                for n in nbrs:
                    if active_mask[n]:
                        all_indices.add(int(n))
            top_indices = np.array(sorted(all_indices), dtype=np.int32)

        layer.snapshot_and_recycle(top_indices, domain=domain_label,
                                   recycle=recycle, gate_init=gate_init)
        snap_map[name] = top_indices.tolist()

    total_snapped = sum(len(v) for v in snap_map.values())
    total_gens = max((l.n_snapshots for _, l in collect_peer_layers(model)), default=0)
    label = f" ({domain_label})" if domain_label else ""
    print(f"    SNAPSHOT{label}: {total_snapped} experts snapshotted, "
          f"gen {total_gens}, recycle={recycle}")
    return snap_map


def version_tree_summary(model):
    """Print version tree state: generations, preserved positions, gate stats."""
    layers = collect_peer_layers(model)
    if not layers:
        print("    No PEER layers found")
        return

    max_gens = max(l.n_snapshots for _, l in layers)
    if max_gens == 0:
        print("    Version tree: no snapshots")
        return

    total_preserved = sum(l.n_preserved for _, l in layers)
    total_experts = sum(l.n_experts for _, l in layers)
    total_frozen = sum(l.n_frozen for _, l in layers)
    total_active = total_experts - total_frozen

    print(f"    Version tree: {max_gens} generation(s), "
          f"{total_preserved} preserved positions, "
          f"{total_active} active, {total_frozen} frozen")

    # Per-generation gate stats (aggregate across layers)
    for gen in range(max_gens):
        gen_gates = []
        gen_count = 0
        for _, layer in layers:
            if gen < len(layer.snapshots):
                snap = layer.snapshots[gen]
                gb = np.array(snap.gate_bias)
                gates = 1.0 / (1.0 + np.exp(-gb))
                gen_gates.extend(gates[snap._mask])
                gen_count += snap.n_active
        if gen_gates:
            arr = np.array(gen_gates)
            domain = layers[0][1].snapshots[gen]._domain if gen < layers[0][1].n_snapshots else "?"
            print(f"      Gen {gen} ({domain}): {gen_count} experts, "
                  f"gate mean={arr.mean():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")


def peer_lifecycle_summary(model):
    """Print lifecycle state summary for all PEER layers."""
    layers = collect_peer_layers(model)
    if not layers:
        print("    No PEER layers found")
        return

    total_experts = sum(l.n_experts for _, l in layers)
    total_frozen = sum(l.n_frozen for _, l in layers)
    total_gated = sum(l.n_gated for _, l in layers)
    total_active = total_experts - total_frozen

    parts = [f"{total_active} active", f"{total_frozen} frozen"]
    if total_gated > 0:
        parts.append(f"{total_gated} gated")
    print(f"    Experts: {', '.join(parts)}, {total_experts} total")

    # Gate distribution
    if total_gated > 0:
        all_gates = []
        for _, l in layers:
            gv = l.gate_values()
            gated_mask = l._has_parent
            if gated_mask.any():
                all_gates.extend(gv[gated_mask])
        if all_gates:
            arr = np.array(all_gates)
            print(f"    Gate values: mean={arr.mean():.3f}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}")

    # Importance
    frozen_imp = []
    active_imp = []
    for _, l in layers:
        imp = l.expert_importance()
        frozen_imp.extend(imp[l._frozen])
        active_imp.extend(imp[~l._frozen])

    if frozen_imp:
        print(f"    Frozen ||v|| mean: {np.mean(frozen_imp):.4f}")
    if active_imp:
        print(f"    Active ||v|| mean: {np.mean(active_imp):.4f}")


def total_peer_params(model):
    """Count total stored PEER parameters across all patched layers."""
    total = 0
    for _, layer in collect_peer_layers(model):
        # weight_down + weight_up + gate_bias
        total += layer.n_experts * layer.d_in
        total += layer.n_experts * layer.d_out
        total += layer.n_experts
        # sub_keys + query_proj
        total += 2 * layer._sqrt_n * layer._d_key
        total += 2 * layer._d_key * layer.d_in
        # Snapshot params: weight_up (frozen) + gate_bias (trainable) per generation
        for snap in layer.snapshots:
            total += layer.n_experts * layer.d_out  # snap.weight_up
            total += layer.n_experts                  # snap.gate_bias
    return total


# ── Parallel PEER utilities ────────────────────────────────


def collect_parallel_peer_layers(model):
    """Find all ParallelPEERLayer instances in a model."""
    results = []

    def _search(module, prefix=""):
        if isinstance(module, ParallelPEERLayer):
            results.append((prefix, module))
            return
        if isinstance(module, nn.Module):
            children = module.children()
            for name, child in children.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Module):
                    _search(child, full_name)
                elif isinstance(child, (list, tuple)):
                    for i, item in enumerate(child):
                        _search(item, f"{full_name}.{i}")
                elif isinstance(child, dict):
                    for k, v in child.items():
                        _search(v, f"{full_name}.{k}")

    _search(model)
    return results


def freeze_best_branch(model, domain_label=""):
    """Freeze the branch with highest mean expert importance across all parallel layers.

    Returns:
        int: index of the frozen branch (same across all layers).
    """
    layers = collect_parallel_peer_layers(model)
    if not layers:
        return -1

    # Aggregate importance across all layers
    n_branches = layers[0][1]._n_branches_val
    total_imp = np.zeros(n_branches)
    for _, layer in layers:
        imp = layer.branch_importance()
        # Only consider unfrozen branches
        for i in range(n_branches):
            if not layer._branch_frozen[i]:
                total_imp[i] += imp[i]
            else:
                total_imp[i] = -1  # skip frozen

    # Find best unfrozen branch
    best_idx = int(np.argmax(total_imp))
    if total_imp[best_idx] <= 0:
        print(f"    FREEZE_BRANCH: no unfrozen branches to freeze")
        return -1

    for _, layer in layers:
        layer.freeze_branch(best_idx, domain_label=domain_label)

    total_frozen = sum(int(l._branch_frozen.sum()) for _, l in layers)
    total_branches = sum(l._n_branches_val for _, l in layers)
    label = f" ({domain_label})" if domain_label else ""
    print(f"    FREEZE_BRANCH{label}: branch {best_idx}, "
          f"total frozen={total_frozen}/{total_branches}")
    return best_idx


def parallel_peer_lifecycle_summary(model):
    """Print lifecycle state for all parallel PEER layers."""
    layers = collect_parallel_peer_layers(model)
    if not layers:
        print("    No ParallelPEERLayer found")
        return

    n_branches = layers[0][1]._n_branches_val
    branch_frozen = layers[0][1]._branch_frozen  # same across layers
    branch_domains = layers[0][1]._branch_domain

    frozen_str = []
    for i in range(n_branches):
        status = "frozen" if branch_frozen[i] else "active"
        domain = f"({branch_domains[i]})" if branch_domains[i] else ""
        frozen_str.append(f"B{i}:{status}{domain}")
    print(f"    Branches: {', '.join(frozen_str)}")

    # Per-branch expert stats
    for i in range(n_branches):
        total_experts = sum(l.branches[i].n_experts for _, l in layers)
        total_frozen = sum(l.branches[i].n_frozen for _, l in layers)
        print(f"      B{i}: {total_experts - total_frozen} active, "
              f"{total_frozen} frozen experts")


def setup_parallel_streams(model, mode="gpu"):
    """Create shared streams and wire them to all ParallelPEERLayer instances.

    Creating per-layer streams causes N_layers × N_branches Metal command queues,
    triggering the ~5s GPU watchdog timeout. Instead, we create just N_branches
    streams globally and share them across all layers.

    Args:
        model: model containing ParallelPEERLayer instances.
        mode: "gpu" — all branches on separate GPU streams (concurrent dispatch).
              "hetero" — branch 0 on GPU, branch 1 on CPU (canonical MLX pattern,
                        slow for matmul-heavy workloads).

    Returns:
        list of created mx.Stream objects.
    """
    layers = collect_parallel_peer_layers(model)
    if not layers:
        return []

    n_branches = layers[0][1]._n_branches_val
    if mode == "hetero":
        streams = [mx.new_stream(mx.gpu if i == 0 else mx.cpu)
                   for i in range(n_branches)]
    else:
        streams = [mx.new_stream(mx.default_device())
                   for _ in range(n_branches)]

    for _, layer in layers:
        layer.set_streams(streams)

    print(f"    STREAMS: {len(streams)} shared streams ({mode}), "
          f"wired to {len(layers)} layers")
    return streams


def total_parallel_peer_params(model):
    """Count total stored parameters across all parallel PEER layers."""
    total = 0
    for _, layer in collect_parallel_peer_layers(model):
        # Gate params
        total += layer._n_branches_val * layer.d_in   # gate_proj
        total += layer._n_branches_val                  # gate_bias_vec
        # Per-branch PEER params
        for branch in layer.branches:
            total += branch.n_experts * branch.d_in     # weight_down
            total += branch.n_experts * branch.d_out    # weight_up
            total += branch.n_experts                    # gate_bias
            total += 2 * branch._sqrt_n * branch._d_key  # sub_keys
            total += 2 * branch._d_key * branch.d_in     # query_proj
    return total
