# M2P Distillation: Mathematical Guarantees for Adapter Generation

## The Architecture (proven to be composition-safe)

```
Frozen Grassmannian A-matrices (orthogonal slots, generated once)
  + M2P-generated B-matrices (domain content, generated per context)
  + Scale-calibrated via preservation loss (learned, not fixed)
```

### Three Decoupled Guarantees

| Property | Mechanism | Type of guarantee |
|----------|-----------|-------------------|
| Orthogonality | Frozen Grassmannian A (QR) | Mathematical: A_i^T A_j = 0 by construction |
| Domain quality | M2P generates B from context | Learned: M2P trained on domain data |
| Scale | Preservation loss during M2P training | Learned: gradient teaches correct magnitude |

### Why Composition Cannot Fail (in parameter space)

For adapters Δ_i = B_i A_i and Δ_j = B_j A_j:

  ⟨Δ_i, Δ_j⟩_F = trace(A_i^T B_i^T B_j A_j)
                 = trace(B_j (A_j A_i^T) B_i^T)     [cyclic permutation of trace]
                 = trace(B_j × 0 × B_i^T)            [because A_j A_i^T = 0]
                 = 0

**This holds for ANY B_i, B_j.** M2P can generate whatever B-matrices it wants —
the Grassmannian A-slots guarantee zero parameter interference.

### Where Interference CAN Still Happen (activation space)

h_out = W_base·x + B_1(A_1·x) + B_2(A_2·x)

Read space: A_1 and A_2 read DISJOINT features (guaranteed).
Write space: B_1 and B_2 write to the SAME output space (unconstrained).

If B_1(A_1·x) and B_2(A_2·x) point in opposite directions → destructive.
This is rare in practice (Finding #3: cos=0.0002 empirically) but not
mathematically impossible.

To close this gap: add output-space orthogonality loss to M2P training.

## The Three Distillation Paths

### Path A: Context → Adapter (instant domain expertise)
```
medical_textbook → base model encodes → M2P reads hidden states → generates B
Result: medical adapter in 1.15ms (Finding #339: 66.6% of SFT quality)
```

**Math that makes it scale:**
- M2P capacity bound: M ≥ n_modules × (d_in + d_out) × r / (H × compression)
- At d=3584, r=16, 7 modules, H=3584: M ≥ 7 × (3584+3584) × 16 / 3584 ≈ 224
- SHINE uses M=128 memory tokens → borderline. M=256 gives safety margin.

### Path B: Teacher → Adapter (model distillation)
```
Teacher model processes domain examples → hidden states at all layers
M2P reads teacher hidden states → generates B that makes student ≈ teacher
```

**Math that makes it scale:**
- Standard distillation: minimize KL(teacher ‖ student+adapter)
- M2P distillation: minimize KL(teacher ‖ base+M2P_adapter)
- The gradient flows: ∂KL/∂B = ∂KL/∂h × ∂h/∂B × ∂B/∂θ_M2P
- As long as M2P is differentiable (it is — it's a transformer), this converges.

### Path C: Adapter → Adapter (compression)
```
Full SFT adapter (300 training steps) → its hidden states when active
M2P reads the adapter-modified hidden states → generates compressed version
```

**Math that makes it scale:**
- Compression ratio: 300 steps of gradient descent → 1 M2P forward pass
- Quality bound: M2P can match SFT quality when M2P was trained on similar
  distribution of adapter behaviors (meta-learning)
- PHLoRA (FlexMoRE) shows SVD extraction gets 93-107% quality.
  M2P should match or beat this because it's LEARNED (not fixed SVD).

## Scale-Calibrated Loss Function

```
L_total = L_task + λ × L_preserve + μ × L_ortho_output

L_task = CrossEntropy(base + M2P_adapter, domain_tokens)
  → teaches M2P to generate useful domain adapters

L_preserve = CrossEntropy(base + M2P_adapter, general_tokens)
  → penalizes adapters that destroy general knowledge
  → gradient teaches M2P the correct scale automatically

L_ortho_output = Σ_{i≠j} ‖B_i(A_i x) · B_j(A_j x)‖²
  → penalizes output-space interference (optional, belt-and-suspenders)
```

The scale α is NOT a hyperparameter — it's LEARNED by M2P through the
preservation loss. The M2P discovers that α≈5 is the sweet spot by
experiencing that α=20 triggers catastrophic L_preserve.

## Experiments

### Exp 1: M2P Distillation Path A (context → adapter, toy scale)
Train M2P to generate B-matrices from context on toy GPT.
Use frozen Grassmannian A-slots.
Compare: M2P adapter quality vs SFT adapter quality.
Measure: composition of 5 M2P-generated adapters — does Grassmannian hold?

### Exp 2: Scale-Calibrated M2P Training
Train M2P with L_task + λ×L_preserve.
Verify: M2P learns to generate adapters at scale≈5 automatically.
Verify: no MMLU degradation even when M2P has no explicit scale constraint.

### Exp 3: M2P Distillation Path B (teacher → adapter)
Use a stronger model (Qwen3-8B) as teacher.
M2P generates adapter that makes Qwen3-4B behave like Qwen3-8B on domain.
Compare: M2P distillation vs standard KD vs SFT.

### Exp 4: M2P Composition at Scale (N=5, N=24)
Generate 24 adapters independently via M2P (one per domain context).
All use frozen Grassmannian A-slots.
Measure routing accuracy, composition PPL, MMLU preservation.
This is the END-TO-END test of the decoupled architecture.

### Exp 5: M2P Path C (adapter compression)
Take 5 SFT adapters (trained over 300 steps each).
Feed their behavior through M2P → get compressed versions.
Compare: compressed vs original on quality, size, composition.

## Scaling Properties (what's proven vs hypothesized)

### Proven (mathematical or empirically verified at multiple scales)

1. **Parameter-space orthogonality scales with d/r.** At d=3584, r=16: room for 224
   orthogonal adapters. QR decomposition produces exact orthogonality at any d.
   Verified at d=64, 256, 512, 1024, 2560, 3584 (Findings #3, #126, #318, #341).

2. **M2P quality is d_model-independent at toy scale.** Tested at d=256/512/1024
   with fixed d_M2P=64: quality ratio 97.6% / 100.6% / 99.6% (#359, #361, #362).
   Two valid domains (sort, reverse), synthetic data only.

### Empirically supported but limited evidence

3. **M2P quality degrades gracefully with layer depth.** L=2→99.7%, L=4→93.5%,
   L=8→97.1%, L=16→86.4% (#363). L=36 (Qwen3-4B) is untested.

4. **Data scale (n≥T) eliminates cyclic overfitting.** Verified at d=256/512/1024 (#359).

### Hypothesized (no evidence yet)

5. **M2P works on natural language.** All evidence is on synthetic tasks (arithmetic,
   sort, reverse). Real NLP benchmarks are untested (Level 3 in PoC roadmap).

6. **Multi-cycle promotion converges.** Single-cycle demonstrated (#333). Multi-cycle
   is the next validation target (Level 4A).

7. **Activation-space interference stays bounded at large N.** Measured at N=5 only
   (max|cos|=0.29). Scaling with N is unknown (Level 2B).
