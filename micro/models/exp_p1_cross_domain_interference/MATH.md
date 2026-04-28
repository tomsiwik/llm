# Cross-Domain Interference Matrix

## Type
Verification — confirms LoRA adapters help on-domain and don't catastrophically degrade off-domain.

## Prior work
- Finding #826: uniform composition (all adapters active) achieves 72.2% avg accuracy; hard-routed single adapters achieve 68.9%. Adapters appear complementary rather than interfering.
- Finding #825: uniform composition suffers 10pp math drop and 12pp code drop vs single-adapter baselines — interference exists but is partial.
- LoRA theory (Hu et al. 2021): low-rank updates ΔW = BA occupy a small subspace of the weight matrix. Off-domain inputs are unlikely to activate the adapter's learned subspace strongly.

## Theorem (informal)
If adapter i was trained on domain D_i and evaluated on domain D_j (j ≠ i), the adapter's contribution ΔW_i · x is small when x ∉ D_i because the adapter learned features specific to D_i. The degradation is bounded by the projection of off-domain activations onto the adapter's column space.

## Predictions
1. **On-domain lift ≥ 10pp**: Each adapter meaningfully improves its own benchmark (math→GSM8K, python→HumanEval, medical→MedQA) vs base model.
2. **Off-domain degradation ≤ 3pp**: Each adapter does not catastrophically hurt other benchmarks.
3. **Asymmetry**: On-domain gains >> off-domain losses (the signal is concentrated, noise is diffuse).

## Kill criteria
- **K2067**: Off-domain adapter degrades base model accuracy by ≤3pp on every off-domain benchmark. FAIL if any off-domain delta < -3pp.
- **K2068**: On-domain adapter improves base model accuracy by ≥10pp on its own benchmark. FAIL if any on-domain delta < +10pp.

## Failure modes
- If K2067 FAIL: adapters learned domain-general features that distort the base model's behavior globally — composition requires gating.
- If K2068 FAIL: adapters are too weak (low rank, insufficient training) — not an interference finding, a training quality finding.
