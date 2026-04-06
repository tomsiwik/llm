# Geometric Adapter Theory: Heightmaps over Ternary Lattices

## The Concept (in simple terms)

Think of a 3D game engine:
- The **mesh** (base model) = ternary lattice {-1, 0, +1}^d — defines structure
- A **displacement map** (adapter) = continuous height per vertex — adds detail
- A **normal map** = surface orientation — encodes how light/information interacts
- Multiple **texture layers** = multiple domain adapters — composable

The adapter is a heightmap H_i ∈ R^{M×N} draped over the frozen ternary grid G.
- H_i(w) = how much weight w changes for domain i
- ∇H_i(w) = rate of change (which weights are "edges" of the domain)
- ∇²H_i(w) = curvature (smooth knowledge regions vs sharp boundaries)

## The Mathematical Framework

This is a **discrete vector bundle** in the simplest terms:
```
Base space:  G ∈ {-1, 0, +1}^d        (frozen ternary grid)
Fiber:       F_w ≅ R^k at each point w  (continuous adaptation space)
Section:     S_i : G → G × F_w          (domain adapter = heightmap)
Distance:    D_w(i,j) = ‖d_i(w) - d_j(w)‖  (domain similarity at weight w)
```

The total adapted weight: W_i = G + H_i (grid + heightmap)

## Three Channels of Information (like 3D textures)

### Channel 1: Value (the B-matrix we already have)
H_i(w) = scalar displacement at each weight.
This is what current LoRA does — it IS the heightmap.

### Channel 2: Gradient (FREE — computed from Channel 1)
∇H_i = spatial rate of change of the adapter.
- Smooth regions: stable, shared knowledge
- Sharp edges: domain-critical weights (boundaries)
- ∇H_i ≈ ∇H_j → domains i and j share structure at these weights

USE: The gradient is a FREE routing signal. No extra parameters needed.
Just compute Sobel/Laplacian of the B-matrix to find domain boundaries.

### Channel 3: Normal (new — encodes inter-domain relationships)
N_i(w) = direction of adaptation relative to other domains.
- Parallel normals: domains agree (align)
- Orthogonal normals: domains independent (compose freely)
- Anti-parallel normals: domains conflict (interference)

USE: This is the "gauge connection" — it tells the router how domains
transform between each other at each weight.

## Connection to Existing Research

1. **Sheaf Neural Networks** (arXiv:2202.02435): encode HOW information
   transforms between neighboring nodes via sheaf structure. The "normal map"
   is essentially a learned sheaf connection.

2. **Geometric Deep Learning** (Bronstein et al.): gauges define local
   coordinate transformations. Our adapter gradient IS a gauge field over
   the weight lattice.

3. **Platonic Representation Hypothesis**: models converge toward shared
   representations. Matching adapter gradients = detecting convergence.

4. **Smooth convolution kernels** (Romero et al.): weight smoothness is
   REQUIRED for performance. Our adapter's Laplacian measures this directly.

## Implications for Pierre

### What we can do NOW (zero cost):
- Compute ∇H_i for existing adapters (just discrete gradient of B-matrix)
- Use gradient similarity as a routing signal: if ∇H_i ≈ ∇H_j, domains share
  structure at those weights → shared adapter regions don't need routing
- Use Laplacian ∇²H_i to identify "smooth" vs "edge" regions in each adapter
  → smooth regions are stable, edge regions are domain-critical

### What this enables:
- **Automatic domain clustering**: domains with similar adapter gradients form
  natural clusters — no need for explicit domain labels
- **Sparse routing**: only route tokens to adapters at EDGE weights (where
  the adapter gradient is high). Smooth regions use a shared/averaged adapter.
- **Composition quality metric**: before composing, check if heightmap curvatures
  are compatible. High curvature mismatch = interference risk.

### The deeper idea (future work):
Instead of training adapters independently and hoping they compose, train them
with a GEOMETRIC LOSS that penalizes:
- Anti-parallel gradients (conflicting domains)
- Excessive curvature (unstable adaptation)
- Non-smooth composition (interference at boundaries)

This would be: training recipe that GUARANTEES composition by construction,
not orthogonality that PREVENTS interference by separation.

## Open Questions
1. Does adapter gradient similarity actually predict domain relatedness? (needs experiment)
2. Is the Laplacian of the B-matrix meaningful or just noise? (needs experiment)
3. Can geometric loss terms improve composition? (needs experiment)
4. What's the compute cost of gradient analysis on M5 Pro? (trivial — just Sobel convolution)
