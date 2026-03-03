# PLAN: Meta-Optimizer Framework Implementation

## Goal

Refactor the ad-hoc `evo_adam_step` / `pareto_adam_step` into a general
meta-optimizer framework with pluggable candidate generators, evaluation
strategies, and selection rules. Run systematic experiments to map the
Pareto frontier across configurations.

## Phase 1: Refactor into Framework (lgme/meta_optimizer.py)

### Core Abstraction

```python
def meta_step(sd, grads, adam_m, adam_v, lr_t, step,
              candidate_fn,    # generates list of candidate weight dicts
              evaluate_fn,     # scores a candidate → dict of {task: loss}
              select_fn,       # picks best candidate from scored list
              frozen_keys=None):
    """One meta-optimizer step."""
    # 1. Compute base Adam update (shared across all candidates)
    updates = compute_adam_update(sd, grads, adam_m, adam_v, lr_t, step, frozen_keys)

    # 2. Generate candidates from base update
    candidates = candidate_fn(originals, updates, step)
    # candidates: list of {key: mx.array} weight dicts

    # 3. Evaluate each candidate on task objectives
    scored = []
    for weights in candidates:
        apply_weights(sd, weights)
        scores = evaluate_fn(sd)  # e.g. {'new': 2.3, 'old': 1.8}
        scored.append((scores, weights))

    # 4. Select best candidate
    best_weights = select_fn(scored)
    apply_weights(sd, best_weights)
```

### Candidate Generators

```python
# Each returns list of candidate weight dicts

def candidates_step_size(originals, updates, step, sizes=[0.5, 1.0, 1.5]):
    """Discrete step sizes along gradient direction."""

def candidates_fisher_informed(originals, updates, step, fisher, sizes=[0.5, 1.0]):
    """Step sizes weighted by inverse Fisher (move more in low-curvature dirs)."""

def candidates_multi_optimizer(originals, grads, adam_m, adam_v, lr, step):
    """Adam, SGD, sign-SGD updates as separate candidates."""

def candidates_parameter_groups(originals, updates, step, groups):
    """Selective updates per parameter group (experts, attention, embeddings)."""

def candidates_mutated(originals, updates, step, rng, n=3, std=0.01):
    """Base update + Gaussian perturbations (current approach)."""
```

### Evaluation Strategies

```python
def eval_single_doc(g, old_doc, new_doc, ...):
    """Current: 1 old + 1 new doc. Fast but noisy."""

def eval_multi_doc(g, old_docs, new_docs, k=3, ...):
    """K old + 1 new doc. Less noisy, K× more expensive."""

def eval_ema_baseline(g, old_doc, new_doc, ema_state, ...):
    """Track running avg of old-task loss. Stable threshold."""

def eval_proxy_then_full(g, candidates, old_doc, new_doc, proxy_fn, top_k=2):
    """Cheap proxy (gradient dot product) to filter, full eval on top-K only."""
```

### Selection Rules

```python
def select_blended(scored, alpha=0.5):
    """Min of alpha * new_loss + (1-alpha) * old_loss (current evo_adam)."""

def select_pareto(scored, slack=0.05):
    """Filter by old_loss constraint, pick best new_loss (current pareto_adam)."""

def select_minimax(scored):
    """Min of max(new_loss, old_loss) — worst-case optimization."""

def select_nash(scored):
    """Nash bargaining: max product of improvements over baseline."""
```

### File: `lgme/meta_optimizer.py`

New file ~150 lines. Replaces the duplicated logic in evo_adam_step and
pareto_adam_step with a single `meta_step` function + pluggable components.

## Phase 2: Implement Key Candidate Strategies

### Priority 1: Fisher-informed candidates
- Compute diagonal Fisher approximation after Phase 1 (already done for EWC)
- Scale gradient by 1/sqrt(F + eps) — step more in low-curvature directions
- Generate 2 candidates: {Fisher-scaled step, standard step}
- Hypothesis: Fisher candidate preserves old task better, standard learns faster

### Priority 2: Parameter-group candidates
- Group keys into: expert_mlp, expert_heads, attention, embeddings
- Generate 4 candidates: {all, experts-only, attention-only, experts+heads}
- Hypothesis: experts-only preserves shared knowledge; attention-only adapts routing

### Priority 3: Multi-optimizer candidates
- Generate from: Adam (momentum+adaptive), SGD (raw gradient), Momentum-only
- Hypothesis: SGD is more stable for old-task, Adam learns new task faster

### Priority 4: Proxy-accelerated evaluation
- Use gradient dot product with old-task gradient as cheap proxy
- Full forward pass only on top-2 candidates
- Reduces cost from N forward passes to 2 + N dot products
- Hypothesis: proxy correlates well enough to not lose quality

## Phase 3: Systematic Experiment

### Experiment Grid

```
Candidate strategies: [step_size, fisher, param_groups, multi_opt, mutated]
Evaluation: [single_doc, multi_doc_3, ema_baseline, proxy_then_full]
Selection: [blended_0.5, pareto_0.05, pareto_0.10, minimax, nash]

Total: 5 × 4 × 5 = 100 configurations
```

Too many for full experiment. Strategy: phase it.

### Round 1: Candidate strategies (fix eval=single_doc, select=pareto_0.05)
- 5 candidate strategies × 5 seeds × 500 steps
- Goal: which candidates produce the best Pareto frontier?

### Round 2: Evaluation strategies (fix best candidate from Round 1)
- 4 eval strategies × 5 seeds × 500 steps
- Goal: does proxy evaluation maintain quality at lower cost?

### Round 3: Selection rules (fix best candidate + eval from above)
- 5 selection rules × 5 seeds × 500 steps
- Goal: which selection rule gives the best tradeoff control?

### Round 4: Full comparison (top-3 configs from above + baselines)
- 3 meta-optimizer configs + GEM + A-GEM + EWC + freeze baseline
- 5 seeds × 500 steps, at TWO model scales (n_embd=16 and n_embd=64)
- Goal: does our Pareto frontier dominate the baselines?

## Phase 4: GEM/A-GEM Baselines

Must implement for credibility:

### GEM (Gradient Episodic Memory)
- Store gradient on memory buffer (set_a[:50]) after Phase 1
- At each Phase 2 step, check if new gradient violates old-task constraints
- If so, project via QP (use scipy.optimize.minimize or closed-form for single task)
- ~30 lines in lgme/gem.py

### A-GEM (Averaged GEM)
- Store average gradient on old-task buffer
- Dot product check + projection (closed form, 5 lines)
- Much cheaper than GEM

### Integration
- Add to train() as alternatives to evo/pareto training
- New configs: (bc) GEM, (bd) A-GEM
- Must sweep GEM memory size for fair Pareto comparison

## Phase 5: Fix Known Bugs

### Adam moment misalignment (Criticism 5)
Option A: maintain separate moment buffers per candidate (expensive)
Option B: after selecting winner, retroactively adjust moments to match
Option C: use the selected step size to scale moment update:
```python
# After selecting candidate with effective step_ratio (e.g., 0.5 for half step):
adam_m[key] = beta1 * adam_m[key] + (1 - beta1) * (step_ratio * g)
```
Start with Option C — cheap and addresses the core issue.

### Baseline noise in Pareto (Criticism 4)
Replace single-sample baseline with EMA:
```python
if 'old_ema' not in state:
    state['old_ema'] = eval_old_fn(sd)
else:
    state['old_ema'] = 0.9 * state['old_ema'] + 0.1 * eval_old_fn(sd)
threshold = state['old_ema'] + slack
```

## Files

| File | Action | Lines |
|------|--------|-------|
| lgme/meta_optimizer.py | NEW | ~150 |
| lgme/gem.py | NEW | ~40 |
| lgme/optimizer.py | Keep existing, add `compute_adam_update` helper | ~10 |
| continual.py | Add GEM/A-GEM configs, wire meta_step | ~40 |

## Success Criteria

1. Meta-optimizer Pareto frontier has ≥1 operating point that dominates
   GEM's best point (better BWT AND better L_B)
2. Fisher-informed candidates improve over ad-hoc candidates
3. Proxy evaluation reduces cost by ≥50% with <5% quality loss
4. Results hold at both model scales
