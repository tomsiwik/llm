"""LoRA Lifecycle: snapshot + FoX gate blending for continual learning.

After training on domain k, snapshot the current LoRA (A_k, B_k) as frozen
copies. Initialize fresh (A_{k+1}, B_{k+1}) for the next domain. Forward
blends all frozen snapshots + active adapter via learned FoX gates.

This is the LoRA-scale equivalent of our PEER ExpertSnapshot + version tree,
but at ~1.8M params instead of 80M per layer.
"""

import copy
import math
import torch
import torch.nn as nn


class LoRASnapshot:
    """Frozen snapshot of one generation's LoRA adapter weights."""

    def __init__(self, lora_state, domain, gate_init=-2.0):
        """
        Args:
            lora_state: dict mapping layer_key -> {lora_A: Tensor, lora_B: Tensor, scaling: float}
            domain: label for this snapshot generation.
            gate_init: initial gate_bias (sigmoid(-2.0) ~ 0.12, trust snapshot 88%).
        """
        self.domain = domain
        self.weights = {}  # layer_key -> {lora_A, lora_B, scaling}
        self.gate_biases = nn.ParameterDict()  # layer_key -> scalar gate

        for key, state in lora_state.items():
            self.weights[key] = {
                "lora_A": state["lora_A"].detach().clone(),
                "lora_B": state["lora_B"].detach().clone(),
                "scaling": state["scaling"],
            }
            # Sanitize key for ParameterDict (no dots allowed)
            safe_key = key.replace(".", "__")
            self.gate_biases[safe_key] = nn.Parameter(
                torch.tensor(gate_init, dtype=torch.float32)
            )

    def gate_value(self, key):
        """Return sigmoid(gate_bias) for a layer."""
        safe_key = key.replace(".", "__")
        return torch.sigmoid(self.gate_biases[safe_key])


class LoRALifecycleWrapper:
    """Wraps a PEFT LoRA model with snapshot + FoX gate lifecycle.

    Usage:
        wrapper = LoRALifecycleWrapper(peft_model)
        # Train on domain 0 normally...
        wrapper.snapshot_and_reinit("python")
        # Train on domain 1 normally...
        wrapper.snapshot_and_reinit("javascript")
        # etc.

    The wrapper installs forward hooks to blend snapshot outputs with active
    adapter outputs via learned FoX gates.
    """

    def __init__(self, model, gate_init=-2.0):
        """
        Args:
            model: a PEFT LoRA model (from get_peft_model).
            gate_init: initial FoX gate bias for new snapshots.
        """
        self.model = model
        self.snapshots = []
        self.gate_init = gate_init
        self._hooks = []
        self._snapshot_module = nn.ModuleList()  # holds gate params on device

    def _extract_lora_state(self):
        """Extract current LoRA A/B weights from PEFT model.

        Returns:
            dict: adapter_key -> {lora_A: Tensor, lora_B: Tensor, scaling: float}
        """
        state = {}
        for name, module in self.model.named_modules():
            # PEFT LoRA layers have lora_A and lora_B sub-modules
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # lora_A/lora_B are nn.ModuleDict keyed by adapter name
                for adapter_name in module.lora_A:
                    key = f"{name}.{adapter_name}"
                    A = module.lora_A[adapter_name].weight  # (rank, in_features)
                    B = module.lora_B[adapter_name].weight  # (out_features, rank)
                    scaling = module.scaling.get(adapter_name, 1.0)
                    state[key] = {
                        "lora_A": A,
                        "lora_B": B,
                        "scaling": scaling,
                    }
        return state

    def _reinit_lora(self):
        """Reinitialize active LoRA adapter weights (A=kaiming, B=zeros)."""
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for adapter_name in module.lora_A:
                    A = module.lora_A[adapter_name].weight
                    B = module.lora_B[adapter_name].weight
                    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                    nn.init.zeros_(B)

    def snapshot_and_reinit(self, domain_label):
        """Snapshot current LoRA state, reinitialize for next domain.

        Args:
            domain_label: name of the domain just trained (e.g. "python").
        """
        lora_state = self._extract_lora_state()
        snapshot = LoRASnapshot(lora_state, domain_label, gate_init=self.gate_init)

        # Move gate params to same device as model
        device = next(self.model.parameters()).device
        snapshot.gate_biases = snapshot.gate_biases.to(device)
        self._snapshot_module.append(snapshot.gate_biases)

        self.snapshots.append(snapshot)
        self._reinit_lora()

        # Reinstall hooks
        self._install_hooks()

        n_snap = len(self.snapshots)
        n_gates = sum(len(s.gate_biases) for s in self.snapshots)
        print(f"    SNAPSHOT ({domain_label}): {n_snap} generations, "
              f"{n_gates} gate params")

    def _install_hooks(self):
        """Install forward hooks on LoRA layers to blend snapshots."""
        # Remove old hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for name, module in self.model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue

            for adapter_name in module.lora_A:
                key = f"{name}.{adapter_name}"
                # Capture variables in closure
                self._register_hook(module, adapter_name, key)

    def _register_hook(self, module, adapter_name, key):
        """Register a forward hook on a single LoRA module."""

        def hook_fn(mod, input, output, _key=key, _adapter=adapter_name):
            if not self.snapshots:
                return output

            x = input[0]  # input to the linear layer

            # Compute snapshot contributions
            snapshot_out = torch.zeros_like(output)
            total_snapshot_weight = torch.zeros(1, device=output.device)

            for snap in self.snapshots:
                if _key not in snap.weights:
                    continue
                sw = snap.weights[_key]
                A = sw["lora_A"]  # (rank, in_features)
                B = sw["lora_B"]  # (out_features, rank)
                scaling = sw["scaling"]

                # LoRA forward: x @ A.T @ B.T * scaling
                snap_delta = (x @ A.T @ B.T) * scaling

                gate = snap.gate_value(_key)  # sigmoid(bias), scalar
                snapshot_out = snapshot_out + (1.0 - gate) * snap_delta
                total_snapshot_weight = total_snapshot_weight + (1.0 - gate)

            # The active adapter's output is already in `output` (PEFT computes it).
            # We need to add snapshot contribution.
            # But PEFT already added the active LoRA delta to the base output.
            # We just add the snapshot deltas blended by (1-gate).
            return output + snapshot_out

        h = module.register_forward_hook(hook_fn)
        self._hooks.append(h)

    def extra_parameters(self):
        """Return gate bias parameters for the optimizer.

        These must be added to the optimizer's param groups since PEFT
        doesn't know about them.
        """
        params = []
        for snap in self.snapshots:
            params.extend(snap.gate_biases.parameters())
        return params

    def gate_summary(self):
        """Print gate values across all snapshots."""
        for i, snap in enumerate(self.snapshots):
            gates = []
            for key in snap.weights:
                gates.append(snap.gate_value(key).item())
            if gates:
                import numpy as np
                arr = np.array(gates)
                print(f"      Gen {i} ({snap.domain}): gate mean={arr.mean():.3f}, "
                      f"min={arr.min():.3f}, max={arr.max():.3f}")

    @property
    def n_snapshot_params(self):
        """Total parameters across all snapshots (frozen weights + trainable gates)."""
        total = 0
        for snap in self.snapshots:
            for key, sw in snap.weights.items():
                total += sw["lora_A"].numel() + sw["lora_B"].numel()
            total += len(snap.gate_biases)
        return total

    def save_state(self, path):
        """Save all snapshot state (frozen weights + gate biases) to disk.

        Args:
            path: file path for the .pt checkpoint.
        """
        state = []
        for snap in self.snapshots:
            snap_state = {
                "domain": snap.domain,
                "weights": {
                    k: {"lora_A": v["lora_A"].cpu(), "lora_B": v["lora_B"].cpu(),
                         "scaling": v["scaling"]}
                    for k, v in snap.weights.items()
                },
                "gate_biases": {k: v.cpu() for k, v in snap.gate_biases.items()},
            }
            state.append(snap_state)
        torch.save({"snapshots": state, "gate_init": self.gate_init}, path)

    @classmethod
    def load_state(cls, model, path, device="cuda"):
        """Load a saved lifecycle wrapper from checkpoint.

        Args:
            model: the PEFT model to wrap.
            path: file path to the .pt checkpoint.
            device: device to move tensors to.

        Returns:
            LoRALifecycleWrapper with restored snapshots and gates.
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        wrapper = cls(model, gate_init=data["gate_init"])

        for snap_state in data["snapshots"]:
            lora_state = {
                k: {"lora_A": v["lora_A"], "lora_B": v["lora_B"],
                     "scaling": v["scaling"]}
                for k, v in snap_state["weights"].items()
            }
            snapshot = LoRASnapshot(lora_state, snap_state["domain"],
                                    gate_init=data["gate_init"])
            # Restore learned gate values
            for k, v in snap_state["gate_biases"].items():
                snapshot.gate_biases[k] = nn.Parameter(v)
            snapshot.gate_biases = snapshot.gate_biases.to(device)
            wrapper._snapshot_module.append(snapshot.gate_biases)
            wrapper.snapshots.append(snapshot)

        wrapper._install_hooks()
        return wrapper
