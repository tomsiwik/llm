# Pierre Product Architecture: The Adapter Flywheel

## The Vision

```
User sessions (ephemeral)
    ↓ generate context adapters (SHINE-style, per-session)
    ↓
Pattern detection (aggregate)
    ↓ 1000 React users → common patterns crystallize
    ↓
Named adapters (persistent)
    ↓ "Modern React" adapter trained from crystallized patterns
    ↓
Base model improvement (rare)
    ↓ best adapters distilled back into base periodically
    ↓
Per-user adapters shrink (natural)
    ↓ base now knows what adapters taught it
    ↓
Repeat
```

## The Four Adapter Tiers

### Tier 1: Base Model (frozen, shared)
- Train once or adopt existing (Qwen3-4B / BitNet-2B)
- Frozen during serving
- Updated RARELY — only when crystallized knowledge is mature
- Cost: zero per-user, one-time training cost

### Tier 2: Domain Adapters (persistent, shared)
- "Modern React", "Medical Billing", "Rust Async"
- Trained from crystallized user patterns
- Shared across all users of that domain
- Updated periodically (weekly/monthly) as patterns evolve
- Cost: one training run per update, ~$2 per adapter (Finding #148)

### Tier 3: Session Adapters (ephemeral, per-user)
- Generated on-the-fly from user context (SHINE-style)
- Lives only for the session duration
- Encodes: "this user is working on THIS specific React component"
- Cost: one forward pass to generate (SHINE: ~1s), zero training

### Tier 4: User Profile Adapters (persistent, per-user)
- Distilled from accumulated session adapters
- "This user prefers functional components, avoids useEffect, uses Zustand"
- Updated after each session (incremental, cheap)
- Cost: lightweight fine-tune, ~seconds per update

## The Inference Stack

```
Token arrives
  ↓
Base model (ternary kernel, fast)     — Tier 1, always active
  + Domain adapter (pre-attached)     — Tier 2, selected at session start  
  + Session adapter (SHINE-generated) — Tier 3, generated once per session
  + User profile adapter (pre-loaded) — Tier 4, loaded at login
  ↓
Output
```

Per-token cost: base matmul + 1 adapter side-path (composed from Tiers 2-4)

## The Cost Structure

### What must be CHEAP:

| Operation | Frequency | Current cost | Target |
|-----------|-----------|-------------|--------|
| Base inference | Every token | 1 matmul | 1 matmul (already optimal) |
| Adapter side-path | Every token | 420 dispatches (73 tok/s) | <60 dispatches (>100 tok/s) |
| Session adapter gen | Once per session | N/A (need SHINE) | <2s, one forward pass |
| Domain adapter load | Once per session | 0.13s (Finding #288) | <0.5s |
| Adapter composition | Once per session | milliseconds (NRE) | milliseconds |
| Domain adapter train | Weekly | ~45 min for 24 domains (#297) | <1 hour |
| Pattern crystallization | Daily | N/A (need pipeline) | Batch job, overnight |
| Base model update | Monthly/quarterly | Days | Can be slow |

### What we've PROVEN is cheap:
- Adapter attach/detach: 0.13s (#288)
- NRE composition: milliseconds
- Ridge router calibration: 14s for N=5 (#276)
- SFT adapter training: ~2 min per domain (#297)
- Adapter storage: ~10KB ternary, ~100KB bf16 per domain

### What we DON'T have yet:
- Session adapter generation (SHINE): estimated 1-2s per session
- Pattern crystallization: need aggregation pipeline
- User profile update: need incremental adapter update
- Multi-tenant serving: need S-LoRA-style batching

## Architectural Requirements

### 1. The base must be FROZEN
We proved: you cannot merge adapters into ternary (#289, #291).
COMPOSITION_THEORY.md: composition must happen in continuous space.
This is actually a FEATURE — frozen base = shared across all users.

### 2. Adapters must compose WITHOUT retraining
Grassmannian orthogonality guarantees non-interference (#3).
NRE merge preserves norms (#275).
Adding adapter N+1 doesn't affect adapters 1..N.

### 3. Routing must be per-token, not per-session
Finding #310: hidden-state probe routes at 98.3% token accuracy.
Finding #313: single-pass matches oracle within 0.61%.
Mixed-domain sessions (React + API docs + testing) need per-token routing.

### 4. Session adapters must be generated WITHOUT training
SHINE generates adapters from context in one forward pass.
No gradient computation, no optimizer state, no training loop.
Just: context → forward pass → adapter weights.

### 5. The system must be CHEAP to host
BitNet-2B: 1.7GB base + ~100MB per domain adapter
With 100 domain adapters: ~11.7GB total
M5 Pro 48GB: fits ~4 concurrent users with unique adapter stacks
Per-user marginal cost: ~3GB (session adapter + KV cache)

## The Baseless Model Problem

We tried and KILLED baseless composition (#94, #96).
But the NEED is real: if the base model were ITSELF composed of adapters,
we could update any piece without full retraining.

### Why it failed:
Fresh adapters on random scaffold produce nothing (#96).
The scaffold needs to be meaningful — it needs pre-trained representations.

### What might work:
- Start with a pre-trained base (Qwen3-4B)
- NEVER retrain the base directly
- Instead: when an adapter becomes "universal" (all users need it),
  promote it to a new "base layer" — a persistent adapter that's
  always active. The original base + promoted adapters = effective base.
- Over time: the effective base grows more capable through adapter promotion
- This IS base improvement WITHOUT base retraining

### The math:
```
Effective_base = Base + Σ promoted_adapters
               = Base + ΔW_react + ΔW_typescript + ΔW_async + ...
```

Each ΔW_i was trained as a domain adapter. When it's promoted, it becomes
permanently active for all users. This is just permanent NRE-composed adapters.

If promoted adapters are Grassmannian-orthogonal, they compose without
interference. The effective base grows richer without retraining.

## What Experiments Validate This Product

### Already validated:
✅ Adapters compose without interference (N=50, #232)
✅ Adapters can be hot-swapped in <0.5s (#288)
✅ Training a new domain adapter costs ~$2, ~2 min (#148, #297)
✅ Per-token routing works at 98.3% accuracy (#310)
✅ Single-pass serving within 0.61% of oracle (#313)

### Needs validation:
❓ Session adapter generation (SHINE-style) on our base
❓ Adapter promotion (permanent attachment without degradation)
❓ Pattern crystallization from user sessions → named adapter
❓ Multi-tenant serving (multiple users, different adapter stacks)
❓ Incremental user profile update (session → persistent learning)
❓ Pierre Pro quality (Qwen3-4B base) vs Pierre Tiny (BitNet-2B)

### The critical path experiments (already registered):
1. Pierre Pro validation (exp_pro_*) → proves quality on modern base
2. Pierre Tiny production (exp_tiny_*) → proves speed on ternary
3. SHINE port (exp_shine_port) → proves session adapter generation
4. Room Model gradient analysis (exp_room_gradient_analysis) → may inform crystallization

## The Business Model

```
Free tier:  Pierre Tiny (BitNet-2B), 5 domain adapters, no session adapters
Pro tier:   Pierre Pro (Qwen3-4B), 25+ domain adapters, session adapters
Enterprise: Custom base, custom domains, user profile adapters, crystallization

Per-user hosting cost:
  Free:  ~2GB (base + 5 adapters) → $0.01/hr at cloud rates
  Pro:   ~5GB (base + 25 adapters + session) → $0.03/hr
  Enterprise: ~10GB → $0.06/hr

Per-adapter training cost: ~$2 (Finding #148)
Per-session adapter: ~$0.001 (one forward pass)
```

The flywheel: more users → more patterns → better adapters → better model → more users.
The moat: Grassmannian orthogonality is the MATHEMATICAL GUARANTEE that adapters compose.
Nobody else has this.
