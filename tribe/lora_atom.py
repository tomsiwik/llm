"""Self-routing LoRA atoms for LLM continual learning.

Each atom is a rank-1 LoRA component where the A vector serves dual purpose:
  1. Projection key: determines how much the atom activates for a given input
  2. LoRA input: the projection value is used in the LoRA delta computation

This means routing is FREE — the same dot product that computes the LoRA
contribution also determines whether the atom should fire. No external router.

Ghost atoms: parasitic rank-1 adapters that attach to host atoms. A ghost
uses its parent's A key (frozen) for projection and its host's routing weight
(established) for activation. Only ghost_B is trainable. Ghosts feed (grow)
when their knowledge direction is compatible with the host's training signal,
and starve (decay) when incompatible. L2 decay provides baseline starvation
pressure. Ghosts that reach a vitality threshold possess their host (merge B),
transferring cross-domain knowledge without any new routing computation.

Ref: MoRAM (2025) — rank-1 decomposition with self-activated routing.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SelfRoutingLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with self-routing LoRA atoms.

    Each atom has:
      - A row (d_in,): key/projection vector
      - B row (d_out,): value vector

    Forward: base_out + gated_LoRA_delta + ghost_delta
      projections = x @ A.T              # (*, n_atoms)
      scores = |projections| / temp
      weights = softmax(scores)           # soft routing
      delta = (weights * projections) @ B # gated LoRA output

      # Ghost contribution (parasitic, no own routing):
      ghost_proj = x @ parent_A.T        # uses frozen parent's key
      host_w = weights[host_idx]          # borrows host's routing weight
      ghost_delta = (host_w * ghost_proj) @ ghost_B
    """

    def __init__(self, d_in, d_out, n_atoms=32, top_k=0, temperature=1.0,
                 base_weight=None, base_bias=None, scale=1.0, max_ghosts=0):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.n_atoms = n_atoms
        self.top_k = top_k
        self.temperature = temperature
        self.scale = scale

        # Frozen base linear
        if base_weight is not None:
            self.weight = base_weight
        if base_bias is not None:
            self.bias = base_bias

        self._has_base = base_weight is not None
        self._has_bias = base_bias is not None

        # Trainable atom parameters
        self.atom_A = mx.random.normal((n_atoms, d_in)) * (1.0 / math.sqrt(d_in))
        self.atom_B = mx.zeros((n_atoms, d_out))

        # Ghost parameters (trainable, only created if max_ghosts > 0)
        self._max_ghosts = max_ghosts
        if max_ghosts > 0:
            self.ghost_B = mx.zeros((max_ghosts, d_out))
            # Pointers (numpy, not parameters, not trained)
            self._ghost_parent_idx = np.zeros(max_ghosts, dtype=np.int32)
            self._ghost_host_idx = np.zeros(max_ghosts, dtype=np.int32)
            self._ghost_alive = np.zeros(max_ghosts, dtype=bool)
            self._ghost_age = np.zeros(max_ghosts, dtype=np.int32)
            # Pre-computed host selection matrix: host_map[g, a] = 1 if ghost g → host a
            self._host_map = np.zeros((max_ghosts, n_atoms), dtype=np.float32)

        # Activation statistics (not parameters, not trained)
        self._activation_counts = np.zeros(n_atoms, dtype=np.float64)
        self._forward_calls = 0

        # Lifecycle state per atom
        self._frozen = np.zeros(n_atoms, dtype=bool)
        # Graduated protection: 1.0 = full gradient, 0.0 = frozen
        self._protection = np.ones(n_atoms, dtype=np.float32)

    def __call__(self, x):
        """Forward pass: base linear + self-routing LoRA delta + ghost delta."""
        # Base linear (frozen)
        if self._has_base:
            out = x @ self.weight.T
            if self._has_bias:
                out = out + self.bias
        else:
            out = mx.zeros((*x.shape[:-1], self.d_out))

        if self.n_atoms == 0:
            return out

        # --- Self-routing LoRA ---
        projections = x @ self.atom_A.T  # (..., n_atoms)
        scores = mx.abs(projections) / max(self.temperature, 1e-6)

        if self.top_k > 0 and self.top_k < self.n_atoms:
            sorted_scores = mx.sort(scores, axis=-1)
            threshold = sorted_scores[..., -self.top_k : -self.top_k + 1]
            mask = (scores >= threshold).astype(mx.float32)
            soft_weights = mx.softmax(scores, axis=-1)
            weights = mask * soft_weights
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-8)
        else:
            weights = mx.softmax(scores, axis=-1)

        gated_proj = weights * projections
        delta = gated_proj @ self.atom_B

        # --- Ghost contribution (parasitic, no own routing) ---
        if self._max_ghosts > 0 and self._ghost_alive.any():
            # Parent's A vectors — frozen, no gradient needed
            parent_A = mx.stop_gradient(
                mx.take(self.atom_A, mx.array(self._ghost_parent_idx), axis=0)
            )  # (max_ghosts, d_in)

            # Ghost projections using parent's routing key
            ghost_proj = mx.stop_gradient(x) @ parent_A.T  # (..., max_ghosts)

            # Host routing weights via pre-computed selection matrix
            # host_map[g, a] = 1 if ghost g is hosted on atom a
            host_map_mx = mx.array(self._host_map)
            host_w = mx.stop_gradient(weights) @ host_map_mx.T  # (..., max_ghosts)

            # Only ghost_B has gradient — the sole trainable ghost parameter
            ghost_delta = (host_w * ghost_proj) @ self.ghost_B  # (..., d_out)
            delta = delta + ghost_delta

        return out + self.scale * delta

    # ── Ghost Lifecycle ─────────────────────────────────────

    def spawn_ghosts(self, frozen_indices, n_per_frozen=1):
        """Spawn parasitic ghosts from frozen atoms.

        Each ghost:
          - parent = frozen atom (uses its A for projection direction)
          - host = random active atom (borrows its routing weight)
          - ghost_B = zero (gradient gives direction, no cold start needed)

        Returns:
            int: number of ghosts spawned.
        """
        if self._max_ghosts == 0:
            return 0

        active_indices = np.where(~self._frozen)[0]
        if len(active_indices) == 0:
            return 0

        n_spawned = 0
        ghost_B_np = np.array(self.ghost_B)

        for frozen_idx in frozen_indices:
            for _ in range(n_per_frozen):
                free_slots = np.where(~self._ghost_alive)[0]
                if len(free_slots) == 0:
                    break
                slot = free_slots[0]

                host = np.random.choice(active_indices)

                self._ghost_parent_idx[slot] = frozen_idx
                self._ghost_host_idx[slot] = host
                self._ghost_alive[slot] = True
                self._ghost_age[slot] = 0
                ghost_B_np[slot] = 0.0

                # Update host selection matrix
                self._host_map[slot] = 0.0
                self._host_map[slot, host] = 1.0

                n_spawned += 1

        self.ghost_B = mx.array(ghost_B_np)
        return n_spawned

    def ghost_vitality(self):
        """Per-ghost ||B|| norms (vitality signal).

        Returns:
            (max_ghosts,) numpy array. Higher = more fed, more viable.
        """
        if self._max_ghosts == 0:
            return np.array([])
        ghost_B_np = np.array(self.ghost_B)
        return np.linalg.norm(ghost_B_np, axis=1)

    def check_possessions(self, threshold=0.3):
        """Check if any ghosts should possess their hosts.

        When ghost ||B|| exceeds threshold, merge ghost_B into host's atom_B.
        Ghost is then killed and its slot freed for respawning.

        Returns:
            list of (ghost_slot, host_idx, parent_idx, vitality) tuples.
        """
        if self._max_ghosts == 0:
            return []

        possessions = []
        ghost_B_np = np.array(self.ghost_B)
        atom_B_np = np.array(self.atom_B)

        for g in range(self._max_ghosts):
            if not self._ghost_alive[g]:
                continue
            vitality = np.linalg.norm(ghost_B_np[g])
            if vitality >= threshold:
                host = self._ghost_host_idx[g]
                parent = self._ghost_parent_idx[g]
                # Possess: additive merge into host
                atom_B_np[host] += ghost_B_np[g]
                # Kill ghost
                ghost_B_np[g] = 0.0
                self._ghost_alive[g] = False
                self._host_map[g] = 0.0
                possessions.append((g, int(host), int(parent), float(vitality)))

        if possessions:
            self.atom_B = mx.array(atom_B_np)
            self.ghost_B = mx.array(ghost_B_np)

        return possessions

    def kill_starved_ghosts(self, starvation_threshold=1e-4):
        """Kill ghosts that have starved to near-zero vitality.

        Returns:
            list of (ghost_slot, parent_idx) tuples for killed ghosts.
        """
        if self._max_ghosts == 0:
            return []

        killed = []
        ghost_B_np = np.array(self.ghost_B)

        for g in range(self._max_ghosts):
            if not self._ghost_alive[g]:
                continue
            vitality = np.linalg.norm(ghost_B_np[g])
            if vitality < starvation_threshold and self._ghost_age[g] > 10:
                parent = self._ghost_parent_idx[g]
                ghost_B_np[g] = 0.0
                self._ghost_alive[g] = False
                self._host_map[g] = 0.0
                killed.append((g, int(parent)))

        if killed:
            self.ghost_B = mx.array(ghost_B_np)

        return killed

    def respawn_ghost(self, slot, parent_idx):
        """Respawn a dead ghost with perturbed parent A direction.

        Uses orthogonal perturbation so the new ghost explores adjacent
        knowledge space, not the same space as its parent.
        """
        if self._max_ghosts == 0:
            return False

        active_indices = np.where(~self._frozen)[0]
        if len(active_indices) == 0:
            return False

        host = np.random.choice(active_indices)

        self._ghost_parent_idx[slot] = parent_idx
        self._ghost_host_idx[slot] = host
        self._ghost_alive[slot] = True
        self._ghost_age[slot] = 0

        self._host_map[slot] = 0.0
        self._host_map[slot, host] = 1.0

        ghost_B_np = np.array(self.ghost_B)
        ghost_B_np[slot] = 0.0
        self.ghost_B = mx.array(ghost_B_np)
        return True

    def ghost_regularization(self):
        """L2 regularization on ghost_B for starvation pressure.

        Add to loss: loss += weight * layer.ghost_regularization()
        Gradient naturally shrinks ghost_B unless training signal counteracts.
        """
        if self._max_ghosts == 0:
            return mx.array(0.0)
        return mx.sum(self.ghost_B ** 2)

    def age_ghosts(self):
        """Increment age of all alive ghosts."""
        self._ghost_age[self._ghost_alive] += 1

    @property
    def n_ghosts_alive(self):
        if self._max_ghosts == 0:
            return 0
        return int(self._ghost_alive.sum())

    # ── Atom Lifecycle ──────────────────────────────────────

    def freeze_atoms(self, indices):
        """Freeze specific atoms (they still fire via self-routing but don't train)."""
        for i in indices:
            self._frozen[i] = True
            self._protection[i] = 0.0

    def unfreeze_atoms(self, indices):
        """Unfreeze specific atoms (make trainable again)."""
        for i in indices:
            self._frozen[i] = False
            self._protection[i] = 1.0

    def recycle_atoms(self, indices):
        """Recycle atoms: reinit A (new routing key), zero B, mark active."""
        A_np = np.array(self.atom_A)
        B_np = np.array(self.atom_B)
        for i in indices:
            A_np[i] = np.random.randn(self.d_in) / math.sqrt(self.d_in)
            B_np[i] = 0.0
            self._frozen[i] = False
        self.atom_A = mx.array(A_np)
        self.atom_B = mx.array(B_np)

    def atom_importance(self):
        """Per-atom importance as ||B_i|| (L2 norm of value vector)."""
        B_np = np.array(self.atom_B)
        return np.linalg.norm(B_np, axis=1)

    @property
    def n_frozen(self):
        return int(self._frozen.sum())

    @property
    def n_active(self):
        return self.n_atoms - self.n_frozen

    @property
    def frozen_mask(self):
        return self._frozen.copy()

    def set_graduated_protection(self, tiers=None):
        """Set graduated gradient scaling based on atom importance tiers.

        Frozen atoms always get 0.0. Active atoms are ranked by ||B|| and
        assigned to tiers with decreasing gradient scale. This COMBINES
        with freeze — frozen atoms are excluded from tiering.

        Args:
            tiers: list of (fraction, scale) tuples. Default:
                [(0.25, 0.1), (0.25, 0.5), (0.5, 1.0)]
                = top 25% active → 0.1 gradient, next 25% → 0.5, bottom 50% → 1.0
        """
        if tiers is None:
            tiers = [(0.25, 0.1), (0.25, 0.5), (0.5, 1.0)]

        active_mask = ~self._frozen
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)

        if n_active == 0:
            return

        # Rank active atoms by importance (||B||) — most important first
        importance = self.atom_importance()
        active_importance = importance[active_indices]
        rank_order = np.argsort(active_importance)[::-1]  # descending
        ranked_indices = active_indices[rank_order]

        # Assign tiers
        cursor = 0
        for frac, scale in tiers:
            n_tier = max(1, int(n_active * frac))
            end = min(cursor + n_tier, n_active)
            for idx in ranked_indices[cursor:end]:
                self._protection[idx] = scale
            cursor = end
            if cursor >= n_active:
                break

        # Ensure frozen atoms stay at 0.0
        self._protection[self._frozen] = 0.0

    # ── Stats ───────────────────────────────────────────────

    def update_stats(self, x):
        """Track activation statistics."""
        x_sg = mx.stop_gradient(x)
        A_sg = mx.stop_gradient(self.atom_A)
        projections = x_sg @ A_sg.T
        scores = mx.abs(projections)
        mean_activation = mx.mean(scores, axis=tuple(range(scores.ndim - 1)))
        mx.eval(mean_activation)
        self._activation_counts += np.array(mean_activation)
        self._forward_calls += 1

    def activation_profile(self):
        if self._forward_calls == 0:
            return np.zeros(self.n_atoms)
        return self._activation_counts / self._forward_calls

    def reset_stats(self):
        self._activation_counts = np.zeros(self.n_atoms, dtype=np.float64)
        self._forward_calls = 0

    def effective_rank(self, x_sample=None):
        if x_sample is not None:
            x_sg = mx.stop_gradient(x_sample)
            A_sg = mx.stop_gradient(self.atom_A)
            projections = x_sg @ A_sg.T
            scores = mx.abs(projections) / max(self.temperature, 1e-6)
            weights = mx.softmax(scores, axis=-1)
            mean_weights = mx.mean(weights, axis=tuple(range(weights.ndim - 1)))
            mx.eval(mean_weights)
            p = np.array(mean_weights)
        else:
            profile = self.activation_profile()
            if profile.sum() == 0:
                return float(self.n_atoms)
            p = profile / profile.sum()
        p = p + 1e-10
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))

    def __repr__(self):
        mode = f"top_k={self.top_k}" if self.top_k > 0 else f"temp={self.temperature}"
        frozen = f", frozen={self.n_frozen}" if self.n_frozen > 0 else ""
        ghosts = f", ghosts={self.n_ghosts_alive}" if self.n_ghosts_alive > 0 else ""
        return (f"SelfRoutingLoRALinear(d_in={self.d_in}, d_out={self.d_out}, "
                f"n_atoms={self.n_atoms}, {mode}{frozen}{ghosts})")


# ── Module-level utilities ──────────────────────────────────

def patch_linear_with_atoms(model_or_layer, linear_attr, n_atoms=32, top_k=0,
                            temperature=1.0, scale=1.0, max_ghosts=0):
    """Replace a Linear layer with a SelfRoutingLoRALinear."""
    linear = getattr(model_or_layer, linear_attr)
    d_out, d_in = linear.weight.shape
    bias = getattr(linear, 'bias', None)

    lora_linear = SelfRoutingLoRALinear(
        d_in=d_in, d_out=d_out, n_atoms=n_atoms, top_k=top_k,
        temperature=temperature, base_weight=linear.weight,
        base_bias=bias, scale=scale, max_ghosts=max_ghosts,
    )

    setattr(model_or_layer, linear_attr, lora_linear)
    return lora_linear


def collect_atom_layers(model):
    """Find all SelfRoutingLoRALinear layers in a model."""
    results = []

    def _search(module, prefix=""):
        if isinstance(module, SelfRoutingLoRALinear):
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


def total_atom_params(model):
    """Count total trainable atom parameters across all patched layers."""
    total = 0
    for _, layer in collect_atom_layers(model):
        total += layer.n_atoms * layer.d_in
        total += layer.n_atoms * layer.d_out
        if layer._max_ghosts > 0:
            total += layer._max_ghosts * layer.d_out
    return total


# ── Lifecycle Utilities ─────────────────────────────────────

def mask_frozen_gradients(model, grads):
    """Scale gradients by per-atom protection mask across all layers.

    Uses the graduated protection mask: 0.0 for frozen, 0.1/0.5/1.0 for
    active atoms at different protection tiers. Falls back to binary
    freeze mask if graduated protection hasn't been set.
    """
    for name, layer in collect_atom_layers(model):
        if layer.n_frozen == 0 and np.all(layer._protection == 1.0):
            continue

        # Use the graduated protection mask (already has 0.0 for frozen)
        prot_mx = mx.array(layer._protection)

        parts = name.split(".")
        node = grads
        try:
            for part in parts:
                if part.isdigit():
                    node = node[int(part)]
                else:
                    node = node[part]
            if "atom_A" in node:
                node["atom_A"] = node["atom_A"] * mx.expand_dims(prot_mx, 1)
            if "atom_B" in node:
                node["atom_B"] = node["atom_B"] * mx.expand_dims(prot_mx, 1)
        except (KeyError, IndexError, TypeError):
            pass

    return grads


def freeze_top_atoms(model, n_freeze_per_layer, domain_label=None):
    """Freeze the most important atoms in each layer.

    Returns:
        dict: layer_name → list of frozen atom indices.
    """
    frozen_map = {}
    for name, layer in collect_atom_layers(model):
        importance = layer.atom_importance()
        active_mask = ~layer.frozen_mask
        importance[~active_mask] = -1

        n_active = int(active_mask.sum())
        n_to_freeze = min(n_freeze_per_layer, n_active)
        if n_to_freeze <= 0:
            frozen_map[name] = []
            continue

        top_indices = np.argsort(importance)[::-1][:n_to_freeze]
        layer.freeze_atoms(top_indices)
        frozen_map[name] = top_indices.tolist()

    total_frozen = sum(layer.n_frozen for _, layer in collect_atom_layers(model))
    total_atoms = sum(layer.n_atoms for _, layer in collect_atom_layers(model))
    label = f" ({domain_label})" if domain_label else ""
    print(f"    FREEZE{label}: {n_freeze_per_layer}/layer, "
          f"total frozen={total_frozen}/{total_atoms}")
    return frozen_map


def apply_graduated_protection(model, tiers=None, domain_label=None):
    """Apply graduated gradient protection to all layers.

    Call AFTER freeze_top_atoms. Tiers the remaining active atoms by importance.

    Args:
        tiers: list of (fraction, scale) tuples. None = default tiers.
    """
    for _, layer in collect_atom_layers(model):
        layer.set_graduated_protection(tiers=tiers)

    label = f" ({domain_label})" if domain_label else ""
    # Summarize tier distribution
    layers = collect_atom_layers(model)
    tier_counts = {}
    for _, layer in layers:
        for p in layer._protection:
            tier_counts[p] = tier_counts.get(p, 0) + 1
    tier_str = ", ".join(f"{v}x:{c}" for v, c in sorted(tier_counts.items()))
    print(f"    PROTECT{label}: {tier_str}")


def spawn_ghosts_from_frozen(model, frozen_map, n_per_frozen=1):
    """Spawn ghosts from newly frozen atoms across all layers.

    Returns:
        int: total ghosts spawned.
    """
    total_spawned = 0
    for name, layer in collect_atom_layers(model):
        frozen_indices = frozen_map.get(name, [])
        if frozen_indices:
            n = layer.spawn_ghosts(frozen_indices, n_per_frozen=n_per_frozen)
            total_spawned += n

    total_alive = sum(layer.n_ghosts_alive for _, layer in collect_atom_layers(model))
    print(f"    SPAWN: {total_spawned} ghosts ({n_per_frozen}/frozen), "
          f"total alive={total_alive}")
    return total_spawned


def ghost_lifecycle_step(model, possession_threshold=0.3, starvation_threshold=1e-4,
                          respawn=True):
    """Run one ghost lifecycle tick: age, check possessions, kill starved, respawn.

    Call periodically during training (e.g., every 20 steps).

    Returns:
        dict with counts of possessions, deaths, respawns.
    """
    total_possessions = 0
    total_deaths = 0
    total_respawns = 0

    for name, layer in collect_atom_layers(model):
        if layer._max_ghosts == 0:
            continue

        layer.age_ghosts()

        # Check possessions
        possessions = layer.check_possessions(threshold=possession_threshold)
        for slot, host, parent, vitality in possessions:
            print(f"      POSSESS: ghost(parent={parent})→host({host}), "
                  f"vitality={vitality:.4f}")
            total_possessions += 1
            # Respawn from same parent with new host
            if respawn:
                if layer.respawn_ghost(slot, parent):
                    total_respawns += 1

        # Kill starved
        killed = layer.kill_starved_ghosts(starvation_threshold=starvation_threshold)
        for slot, parent in killed:
            total_deaths += 1
            if respawn:
                if layer.respawn_ghost(slot, parent):
                    total_respawns += 1

    return {
        'possessions': total_possessions,
        'deaths': total_deaths,
        'respawns': total_respawns,
    }


def ghost_regularization_loss(model, weight=0.01):
    """Total L2 regularization across all ghost_B vectors.

    Add to training loss for starvation pressure.
    """
    reg = mx.array(0.0)
    for _, layer in collect_atom_layers(model):
        reg = reg + layer.ghost_regularization()
    return weight * reg


def lifecycle_summary(model):
    """Print lifecycle state summary."""
    layers = collect_atom_layers(model)
    total_atoms = sum(l.n_atoms for _, l in layers)
    total_frozen = sum(l.n_frozen for _, l in layers)
    total_active = total_atoms - total_frozen
    total_ghosts = sum(l.n_ghosts_alive for _, l in layers)

    frozen_importance = []
    active_importance = []
    ghost_vitalities = []
    for _, l in layers:
        imp = l.atom_importance()
        frozen_importance.extend(imp[l.frozen_mask])
        active_importance.extend(imp[~l.frozen_mask])
        if l._max_ghosts > 0:
            vit = l.ghost_vitality()
            ghost_vitalities.extend(vit[l._ghost_alive])

    parts = [f"{total_active} active", f"{total_frozen} frozen"]
    if total_ghosts > 0:
        parts.append(f"{total_ghosts} ghosts")
    print(f"    Atoms: {', '.join(parts)}, {total_atoms} total")

    if frozen_importance:
        print(f"    Frozen ||B|| mean: {np.mean(frozen_importance):.4f}")
    if active_importance:
        print(f"    Active ||B|| mean: {np.mean(active_importance):.4f}")
    if ghost_vitalities:
        print(f"    Ghost vitality mean: {np.mean(ghost_vitalities):.4f}, "
              f"max: {np.max(ghost_vitalities):.4f}")
