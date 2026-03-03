# Adaptive Freeze Maps with Expert Inheritance (AFMEI)

> A biologically-inspired mechanism where parameter freezing is continuous, decaying, and heritable across expert generations.

---

## 1. Motivation: The Binary Freeze Trap

Our experiments reveal a fundamental tension in continual learning:

| Strategy | BWT (forgetting) | L_B (learning) | Problem |
|----------|-------------------|----------------|---------|
| Freeze nothing (a) | -0.581±0.046 | **2.234±0.011** | Catastrophic forgetting |
| MoE+freeze+EWC (d) | -0.469±0.019 | 2.240±0.014 | Best known technique — still forgets 80%+ |
| Selective freeze (ab) | -0.224±0.250 | 2.482±0.171 | Promising middle but std=±0.250 |
| Freeze everything (r) | **-0.013±0.028** | 2.904±0.064 | 82% params frozen, can't learn |
| Spawn+guaranteed (y) | -0.651±0.064 | 2.262±0.019 | New expert corrupts old evals |
| Joint oracle (f) | 0.000 | 2.471±0.027 | Requires all data at once |

**After 30 configurations**, no method Pareto-dominates (d) MoE+freeze+EWC.
The textbook approach remains the honest winner. This is the gap AFMEI must close.

**Target**: BWT ≥ -0.20, L_B ≤ 2.35, std ≤ 0.05 (Pareto-dominates ALL existing configs).

**The core problem**: binary freeze (`frozen: bool`) creates a discontinuous stability-plasticity boundary. Parameters are either completely rigid or completely plastic. There is no middle ground.

**Biology doesn't work this way.** In real brains:
- Synapses have variable strength (not on/off)
- Neural pathways consolidate gradually during sleep (not at task boundaries)
- Children inherit modified genes (not carbon copies)
- Acquired traits influence offspring (epigenetic inheritance)

---

## 2. The Mechanism

### 2.1 Continuous Freeze Maps (Epigenetic Methylation)

Replace `frozen: bool` with a **freeze temperature** per parameter:

```
tau_w in [0, 1]
  0.0 = fully frozen (no gradient updates)
  0.5 = partially frozen (gradient scaled by 0.5)
  1.0 = fully trainable
```

The effective gradient update becomes:

```
delta_w = tau_w * (lr * grad_w)
```

**Analogy**: In molecular biology, DNA methylation doesn't delete genes -- it controls their *expression level*. A heavily methylated gene is silenced but still present. Demethylation can reactivate it. The freeze map is our methylation pattern.

**Granularity options:**
- Per-parameter: maximum precision, doubles memory
- Per-weight-matrix: one tau per matrix (e.g., tau for fc1, tau for fc2)
- Per-expert: one tau for all parameters in an expert
- **Recommended**: Per-weight-matrix (good tradeoff)

### 2.2 Temperature Decay (Memory Consolidation)

After training on a task, temperatures decay toward frozen:

```
tau_w(t+1) = tau_w(t) * decay_rate
```

where `decay_rate in (0, 1)` controls consolidation speed.

This is NOT the same as:
- **EWC/SI**: These penalize *distance from old values*. Temperature controls *learning rate per parameter*.
- **Learning rate scheduling**: LR schedules are global. Temperatures are per-parameter and task-dependent.

**Key property**: A parameter with tau=0.1 CAN still change, just very slowly. This allows gradual adaptation without catastrophic overwriting. A parameter that consistently receives gradient signal through multiple tasks accumulates slow, stable updates -- like memory consolidation during sleep.

### 2.3 Expert-Driven Thawing (Cross-Expert Communication)

This is where it gets novel. A new expert can **raise the temperature** of parameters in older experts, selectively unfreezing knowledge that's relevant to the new task:

```
delta_tau = alpha_thaw * gradient_agreement(new_expert, old_expert)
tau_w_new = min(tau_w + delta_tau, 1.0)
```

Gradient agreement:
```
agreement_w = cosine_sim(grad_new_w, grad_old_w)
delta_tau = alpha_thaw * max(0, agreement_w)   # only thaw when gradients agree
```

**Why gradient agreement?** If the new expert's gradient on a parameter points the same direction the old expert learned, the new task *reinforces* that knowledge -- it's safe to unfreeze. If gradients oppose, the parameter stays frozen.

**Who can thaw?**
- Only new (unfrozen) experts can modify the freeze maps of old (frozen) experts
- Children (see below) can modify parent freeze maps
- Self-thawing is allowed (an expert can reduce its own freezing if it detects its knowledge is still useful)

### 2.4 Child Experts (Genetic Recombination)

When two experts have high mutual utility (they're both useful for overlapping input patterns), a **child expert** is created:

```
# Parameter crossover
for each weight matrix w:
    if random() < 0.5:
        child.w = parent1.w + noise
    else:
        child.w = parent2.w + noise

# Freeze map inheritance: child inherits LEAST frozen (most plastic)
child.tau_w = max(parent1.tau_w, parent2.tau_w)

# Child starts with high temperature (young = plastic)
child.tau_w = min(child.tau_w * plasticity_boost, 1.0)
```

**Crossover strategies:**
- **Uniform**: Each parameter randomly from parent1 or parent2
- **Fitness-weighted**: Parameters from parent with lower loss on current data
- **Layer-aware**: FC1 from one parent, FC2 from the other (respects functional grouping)
- **Blend**: `child.w = alpha * parent1.w + (1 - alpha) * parent2.w`

**The child's job**: It inherits combined knowledge but must LEARN to validate and refine it. The child's training:
1. Trains on current task data (like a normal expert)
2. Periodically evaluates on old task data (from parents' specialization)
3. Based on performance, can modify parents' freeze maps:
   - If child outperforms parent on old data: thaw parent (the knowledge wasn't optimal)
   - If child underperforms: child adopts parent's approach (deepen its own freezing)

### 2.5 Evolutionary Selection (Survival of the Fittest)

After N training steps, evaluate child fitness:

```
fitness = -loss_current_task + lambda * mean(loss_old_tasks)
```

Selection criteria:
```
if child_fitness > max(parent1_fitness, parent2_fitness):
    # Child survives, parents decay faster
    parent1.decay_rate *= 1.1  # freeze more aggressively
    parent2.decay_rate *= 1.1
    child is promoted to full expert
else:
    # Child fails to improve on parents
    if child_fitness > min(parent1_fitness, parent2_fitness):
        # Child is useful but not dominant -- keep as supplementary
        child survives with reduced routing allocation
    else:
        # Child adds nothing -- discard
        remove child from expert pool
```

**Population control**: Maximum K experts alive at any time. When population exceeds K, the expert with lowest `activation_count * fitness` is culled. This creates genuine evolutionary pressure on the knowledge representation.

---

## 3. Connection to Our Experimental Findings

| Finding | AFMEI Response |
|---------|---------------|
| lm_head = 43% of forgetting | lm_head gets its own tau per expert. High tau for new expert = learns freely. Low tau for old = protected. |
| 82% frozen = can't learn (L_B=2.90) | Continuous tau means ~40-60% effective capacity available, not 18%. |
| Expert redundancy (cosine 0.58-0.68) | Children from diverse parents create novel combinations. Crossover breaks redundancy. |
| Guaranteed routing hurts BWT (-0.651) | No need for guaranteed routing. Child's fresh tau map means it naturally receives gradient signal through normal routing. |
| Selective freeze high variance (+-0.250) | Continuous decay provides deterministic, smooth consolidation. No random selection of which experts to freeze. |
| EWC adds marginal benefit (+0.03) | EWC is a special case of freeze maps where tau = 1/(1 + lambda * F_w). AFMEI generalizes this. |

---

## 4. Mathematical Formalization

### 4.1 State Variables

Per-expert state:
```
Expert_i = {
    params: {w_fc1, w_fc2, w_lm_head}     # weight matrices
    tau:    {tau_fc1, tau_fc2, tau_lm_head} # freeze temperatures per matrix
    decay:  float                           # consolidation rate
    age:    int                             # number of task boundaries survived
    parent_ids: set                         # which experts this was spawned from
}
```

### 4.2 Training Update Rule

Standard gradient step modified by temperature:
```
grad_w = d(Loss)/d(w)                     # from mx.value_and_grad
delta_w = tau_w * lr * grad_w             # temperature-scaled update
w_new = w - delta_w
```

### 4.3 Consolidation at Task Boundary

```
for each expert i:
    for each weight matrix w:
        tau_w *= decay_i                   # decay toward frozen
    decay_i *= 1.01                        # decay rate itself slowly increases
                                           # (older experts consolidate faster)
```

### 4.4 Thawing Dynamics

After new task arrives and new expert begins training:
```
for each (new_expert, old_expert) pair:
    for each shared weight matrix key k:
        g_new = gradient of new expert loss w.r.t. k
        g_old = gradient of old expert loss w.r.t. k (from EWC fisher or cached)
        agreement = cosine_sim(g_new, g_old)
        if agreement > 0:
            old_expert.tau[k] += alpha_thaw * agreement
            old_expert.tau[k] = min(old_expert.tau[k], 1.0)
```

### 4.5 Child Creation

Trigger: Two experts i, j with:
```
route_overlap(i, j) > theta_overlap      # both activated on same inputs
cosine_sim(params_i, params_j) < theta_redundancy  # NOT redundant (diverse)
```

Child parameters:
```
child.w[k] = crossover(parent_i.w[k], parent_j.w[k], strategy)
child.tau[k] = max(parent_i.tau[k], parent_j.tau[k]) * plasticity_boost
child.decay = mean(parent_i.decay, parent_j.decay)
child.age = 0
child.parent_ids = {i, j}
```

---

## 5. Predicted Behavior

### Phase 1: Initial Training (Task A)
- All experts start with tau=1.0 (fully plastic)
- Normal training, experts specialize
- End of task A: temperatures decay (tau -> tau * 0.7)

### Phase 2: Task Switch (A -> B)
- Old experts' tau ~ 0.7 (mostly frozen but slightly plastic)
- New expert spawned with tau=1.0
- Thawing: new expert's gradients probe old experts
  - Where gradients agree: old expert tau increases slightly (shared knowledge)
  - Where gradients disagree: old expert stays frozen (task-specific knowledge)

### Phase 3: Child Creation (Optional)
- If two experts overlap significantly on task B inputs
- Child created from their crossover
- Child trains with full plasticity
- Child evaluates on task A data:
  - If better than parents: parents consolidate further, child promoted
  - If worse: child discarded or kept as supplementary

### Phase 4: Steady State
```
Expert ecosystem:
  Expert 0: age=3, tau~0.1, highly consolidated (task A specialist)
  Expert 1: age=3, tau~0.15, consolidated (task A specialist, slightly thawed)
  Expert 2: age=1, tau~0.5, intermediate (task B specialist, consolidating)
  Expert 4: age=0, tau~0.9, plastic (child of 0+2, learning to bridge A+B)
```

---

## 6. Adversarial Review (Pre-Emptive)

### Critique 0: "You've tried 30 configs and can't beat EWC. Why will this be different?"
**Response**: This is the most important critique. Honest answer: it might not be.
Every previous "novel" mechanism (SOM topology, Kohonen routing, lateral connections,
niche exclusion, idiotypic regulation, replicator dynamics, Lyapunov regularization,
dual-process composition) either added nothing or actively hurt. The ONLY things that
worked were standard techniques: MoE, parameter freezing, EWC, expert-specific heads.

AFMEI is different because it addresses the specific failure mode we measured:
**the binary freeze boundary**. Our diagnostics show selective freeze (ab) is the
only config that lives in the middle of the Pareto frontier (BWT=-0.224, L_B=2.482),
but its high variance (±0.250) comes from the random choice of WHICH experts to freeze.
AFMEI replaces that random binary decision with a deterministic continuous one.

**Kill criterion**: If Phase 1 (continuous freeze maps alone) can't beat BWT=-0.40
with L_B≤2.40, abandon the approach. Don't compound failed mechanisms.

### Critique 1: "This is just per-parameter learning rates"
**Response**: Partially true -- the gradient scaling is equivalent to adaptive LR. But the key differences are:
1. Temperature *decays* at task boundaries (LR schedules don't)
2. Cross-expert *thawing* based on gradient agreement (no LR scheme does this)
3. *Inheritance* through child experts (LR is not heritable)
The combination creates dynamics that no existing optimizer produces.

**Counter-counter**: Adam already adapts per-parameter. The burden of proof is on us
to show that task-boundary-driven decay + thawing produces a measurably different
result than just Adam + EWC. If the numbers are within error bars, this critique wins.

### Critique 2: "Computational cost doubles"
**Response**: With per-matrix granularity (not per-parameter), the overhead is one float per weight matrix per expert. For 4 experts with 3 matrices each: 12 extra floats. The thawing computation requires one extra gradient computation per task boundary -- comparable to EWC's Fisher computation.

### Critique 3: "Children from random crossover will be broken"
**Response**: Valid concern. Weight averaging of neural networks generally fails (non-convex loss landscape). Our mitigation: children inherit from experts that share a common ancestor (all cloned from the same base) and haven't diverged far. Linear mode connectivity (Frankle et al.) shows this works when models share initialization.

**Honest risk**: Our experts have cosine similarity 0.58-0.68 after training.
That's a lot of divergence. Crossover at this distance may land in a high-loss basin.
We should measure the loss surface between parents before committing to crossover.

### Critique 4: "No convergence guarantees"
**Response**: Temperature decay provides a guaranteed path to stability (all tau -> 0 eventually). The system has a Lyapunov function: `V(t) = sum(tau_w(t))` which monotonically decreases between thawing events. Thawing events are bounded (only occur at task boundaries, magnitude bounded by alpha_thaw). If alpha_thaw * K_tasks < decay^T, the system converges.

### Critique 5: "How is this different from Progressive Neural Networks?"
**Response**: PNN adds new columns but never modifies old ones. AFMEI allows *controlled modification* of old parameters through thawing. PNN has zero backward transfer; AFMEI can have positive backward transfer if thawing helps old experts adapt to shared structure.

### Critique 6: "The real bottleneck is shared attention, not MLP freezing"
**Response**: Our attribution study showed lm_head (43%) and attention (32%+29%=61%
for wo+wv) dominate forgetting. MLP expert freezing addresses lm_head via expert heads,
but shared attention parameters are NOT covered by AFMEI's per-expert freeze maps.
The shared attention params (1,024 of 4,192 = 24%) remain fully trainable and are the
likely source of any remaining forgetting. AFMEI should extend freeze maps to attention
parameters too -- but those are SHARED across experts, creating a harder problem.

---

## 7. Open Questions

1. **Optimal decay rate**: Too fast = can't learn task B. Too slow = catastrophic forgetting. Likely task-dependent.
2. **Thawing magnitude**: How much gradient agreement justifies how much temperature increase? Need empirical calibration.
3. **Child timing**: When to create children? After every task? Only when overlap detected? Population cap?
4. **Crossover geometry**: Random parameter crossover may destroy weight matrix structure. Need layer-aware strategies.
5. **Freeze map granularity**: Per-parameter is ideal but expensive. Per-matrix may lose fine-grained control. Per-row?
6. **Interaction with attention**: Freeze maps on MLP experts are straightforward. How do freeze maps interact with shared attention parameters?

---

## 8. Biological Analogies (Reference)

| AFMEI Concept | Biological Analog | Mechanism |
|---------------|-------------------|-----------|
| Freeze temperature | DNA methylation | Chemical markers controlling gene expression level |
| Temperature decay | Memory consolidation | Hippocampal-cortical transfer during sleep |
| Expert-driven thawing | Demethylation | Environmental signals can reverse methylation |
| Child experts | Sexual reproduction | Genetic recombination creates novel combinations |
| Fresh child tau | Developmental plasticity | Young brains are more plastic than adult brains |
| Child modifying parent maps | Epigenetic inheritance | Offspring traits can influence parental gene expression |
| Evolutionary selection | Natural selection | Fitness-based survival determines which knowledge persists |
| Expert culling | Apoptosis | Programmed cell death removes unneeded neurons |
