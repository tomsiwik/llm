# Persistent Expert Tree: Mathematical Foundations

## 1. Persistent Binary Tree via Path Copying

### 1.1 Definitions

A **persistent binary tree** of depth D contains:
- 2^D - 1 internal nodes (gates), each with parameters in R^{d+1}
- 2^D leaf nodes (expert capsule groups), each with parameters in R^{2*d*n_c}

Notation:
```
D       = tree depth (default 3)
L       = 2^D leaf groups (default 8)
I       = 2^D - 1 internal gates (default 7)
d       = embedding dimension (default 64)
n_c     = capsules per leaf group (default 32)
V       = number of versions stored
```

A **version** v = (gates_v, leaves_v) is a pair of reference lists:
```
gates_v  = [g_0^v, g_1^v, ..., g_{I-1}^v]    -- gate module references
leaves_v = [l_0^v, l_1^v, ..., l_{L-1}^v]    -- leaf module references
```

Two versions can share the same module instance (structural sharing):
```
id(g_i^{v1}) == id(g_i^{v2})  =>  g_i^{v1} and g_i^{v2} are the same object
```

### 1.2 Path Copying

To create version v+1 from version v by updating leaf j:

1. Compute the ancestor path: P(j) = set of internal nodes on root-to-leaf_j path
   - |P(j)| = D for any j
   - P(j) = {node_0, node_1, ..., node_{D-1}} where node_0 = root

2. Copy path nodes:
```
gates_{v+1}[i] = { copy(gates_v[i])   if i in P(j)
                  { gates_v[i]         otherwise (shared reference)

leaves_{v+1}[i] = { copy(leaves_v[i])  if i == j
                   { leaves_v[i]        otherwise (shared reference)
```

3. New nodes: D gates + 1 leaf = D + 1 per single-leaf update.

### 1.3 Batch Update (Multiple Leaves)

For updating leaves J = {j_1, j_2, ..., j_m}:

```
P(J) = union_{j in J} P(j)    -- union of all ancestor paths
```

New nodes = |P(J)| gates + m leaves.

**Worst case**: all leaves updated, P(J) = all internal nodes.
  New nodes = I + L = 2^{D+1} - 1 (full copy).

**Best case**: all leaves share one subtree.
  For m = L/2 = 2^{D-1} leaves in one subtree:
  New nodes = (2^{D-1} - 1) + D-1 + 2^{D-1} = 2^D - 2 + D - 1
  Saving: D+1 nodes vs full copy (the other subtree is shared).

At D=3, updating 4 of 8 leaves (one subtree):
  P(J) = {root, child, grandchild_0, grandchild_1} = 4 gates
  New = 4 gates + 4 leaves = 8 nodes (out of 15 total). Sharing = 7/15 = 47%.

---

## 2. Memory Analysis

### 2.1 Parameter Sizes

Per internal gate: d + 1 parameters (linear projection + bias).
Per leaf group: 2 * d * n_c parameters (A and B matrices).

At d=64, n_c=32:
```
params_gate = 65
params_leaf = 2 * 64 * 32 = 4,096

Total per layer:
  gates:  7 * 65 = 455
  leaves: 8 * 4,096 = 32,768
  ratio:  leaves/gates = 71.9x
```

### 2.2 Persistent vs Full-Copy Storage

For V versions, each updating m leaves from a common base:

**Full copy**: V * (I * params_gate + L * params_leaf) per layer

**Persistent (path-copy)**:
  Base:    I * params_gate + L * params_leaf  (one full copy)
  Per version delta: |P(J)| * params_gate + m * params_leaf
  Total: base + V * delta

**Overhead vs single mutable tree**:
```
overhead = V * delta / base
         = V * (|P(J)| * params_gate + m * params_leaf) /
               (I * params_gate + L * params_leaf)
```

At D=3, m=4 (half the leaves), V=2 versions:
```
delta = 4 * 65 + 4 * 4096 = 260 + 16384 = 16644
base  = 7 * 65 + 8 * 4096 = 455 + 32768 = 33223
overhead = 2 * 16644 / 33223 = 100.2% per layer
```

The overhead is dominated by leaf parameters (98.4% of delta).

### 2.3 Savings vs Full Copy

```
savings = 1 - (base + V * delta) / ((V+1) * base)
        = 1 - (1 + V * delta/base) / (V+1)
```

At V=2, delta/base = 0.501:
```
savings = 1 - (1 + 2*0.501) / 3 = 1 - 2.002/3 = 33.3%
```

### 2.4 Scaling to Macro (LoRA Adapters)

At macro scale, leaves are LoRA adapters (rank r), not full CapsuleGroups:
```
params_adapter = 2 * d * r    (A and B matrices)
params_base_layer = O(d^2)    (attention + MLP)
```

At d=896, r=16:
```
params_adapter = 2 * 896 * 16 = 28,672
params_base_layer ~ 896^2 * 12 ~ 9.6M (Qwen 0.5B)
adapter_fraction = 28672 / 9.6M = 0.3%
```

Persistent overhead per version (m=4 adapters + 4 gates):
```
delta_macro = 4 * 28672 + 4 * (896+1) = 114688 + 3588 = 118276
overhead_macro = 118276 / 9.6M = 1.2% per version
```

At macro scale, KC2 (<15%) passes easily: even 12 concurrent versions
would stay under 15% overhead.

---

## 3. Cross-Version Composition

### 3.1 Formalization

Given versions v1 and v2, both derived from base v0, define cross-version
composition by a leaf-version map M: {leaf_idx -> version_id}.

```
leaves_composed[i] = leaves_{M(i)}[i]    -- take leaf i from version M(i)
gates_composed[i]  = fresh copy            -- gates are recalibrated
```

Example: M = {0:v1, 1:v1, 2:v1, 3:v1, 4:v2, 5:v2, 6:v2, 7:v2}
  = left subtree from v1, right subtree from v2.

### 3.2 Why Cross-Version Works

Since v1 and v2 share the same base v0, their leaf parameters are
perturbations of the same initialization:

```
leaves_v1[i] = leaves_v0[i] + delta_v1[i]    (training delta)
leaves_v2[i] = leaves_v0[i] + delta_v2[i]
```

Cross-version composition mixes deltas from different training runs.
This works because:

1. **Orthogonality**: LoRA adapters are naturally orthogonal (cos~0.0002
   at d=896). Different training runs produce non-interfering deltas.

2. **Leaf independence**: Each leaf is an independent expert. The tree
   routing (gates) determines which leaf handles each token. The leaf
   parameters don't depend on other leaves' values.

3. **Gate recalibration**: After composition, gates are retrained on mixed
   data (100 steps). This is the same calibration protocol proven to work
   for same-version composition.

### 3.3 Comparison: Cross-Version vs Same-Version

Same-version composition (weight averaging):
```
leaves_avg[i] = (leaves_v1[i] + leaves_v2[i]) / 2
              = leaves_v0[i] + (delta_v1[i] + delta_v2[i]) / 2
```

Cross-version composition (cherry-pick):
```
leaves_cross[i] = { leaves_v1[i]  if i in A
                  { leaves_v2[i]  if i in B
```

Key difference: weight averaging blends ALL leaf parameters (including
leaves that were irrelevant to their domain), while cross-version
cherry-picking preserves each domain's specialized parameters intact.

Prediction: cross-version should be equal or BETTER than same-version
weight averaging, because it avoids the information loss from averaging
specialized parameters with unspecialized ones.

### 3.4 Empirical Validation

Measured across 3 seeds (42, 123, 7):
```
Same-version (calibrated):  mean 0.5563
Cross-version (calibrated): mean 0.5572
Cross vs Same:              +0.16% (within noise)
```

Per-seed: +0.11%, -0.63%, +1.00%. No systematic direction.
Kill criterion (>5%): PASSES with 30x margin.

---

## 4. Rollback Fidelity

### 4.1 Guarantee

With parameter-level snapshots (not structural sharing during training),
rollback is exact:

```
restore(model, snapshot_v0) => output(model, x) == output_v0(x)
```

Measured: max absolute difference = 0.00e+00 across all seeds.
Rollback is numerically exact (bit-identical outputs).

### 4.2 With Structural Sharing (Theoretical)

If using true persistent data structures with structural sharing DURING
training, shared nodes would be mutated by gradient updates. This requires
either:

(a) Copy-on-write: detect when a shared node is about to be modified and
    copy it first. Overhead: O(1) check per parameter update.

(b) Freeze shared nodes: zero gradients for shared parameters during
    fine-tuning. Only version-specific nodes receive gradient updates.

(c) Snapshot-and-restore: use the parameter snapshot approach (as we do).
    No sharing during training; sharing only for storage/inference.

Option (c) is simplest and matches the macro use case (LoRA adapters are
stored and loaded, not structurally shared during training).

---

## 5. Worked Example: D=2, 4 Leaves

### 5.1 Base Tree (v0)
```
        [gate_0]
        /      \
   [gate_1]  [gate_2]
   /    \    /    \
  L0    L1  L2    L3

Params: 3 gates + 4 leaves = 7 nodes
```

### 5.2 Update L0 -> v1 (path copy)
```
Path(0) = {gate_0, gate_1}  (root and left child)

v1.gates  = [gate_0', gate_1', gate_2]   -- gate_2 shared with v0
v1.leaves = [L0', L1, L2, L3]            -- L1, L2, L3 shared with v0

New nodes: 2 gates + 1 leaf = 3 (out of 7)
Shared:    1 gate + 3 leaves = 4 (57%)
```

### 5.3 Update L3 -> v2 (from v0)
```
Path(3) = {gate_0, gate_2}  (root and right child)

v2.gates  = [gate_0'', gate_1, gate_2'']  -- gate_1 shared with v0
v2.leaves = [L0, L1, L2, L3'']            -- L0, L1, L2 shared with v0

New nodes: 2 gates + 1 leaf = 3
```

### 5.4 Cross-Version Compose: L0 from v1, L3 from v2 -> v3
```
v3.gates  = [gate_0''', gate_1''', gate_2''']  -- all recalibrated
v3.leaves = [L0', L1, L2, L3'']

L0' from v1 (domain A specialist)
L3'' from v2 (domain B specialist)
L1, L2 from v0 (shared base)
```

### 5.5 Memory Accounting (d=64, n_c=32)

```
Per gate: 65 params, per leaf: 4096 params

v0: 3*65 + 4*4096 = 195 + 16384 = 16579
v1: 2*65 + 1*4096 = 130 + 4096  = 4226 (new) + 12353 (shared)
v2: 2*65 + 1*4096 = 130 + 4096  = 4226 (new) + 12353 (shared)

Persistent total: 16579 + 4226 + 4226 = 25031
Full-copy total:  3 * 16579 = 49737
Savings: 1 - 25031/49737 = 49.7%

Overhead vs single tree: (25031 - 16579) / 16579 = 51.0%
```
