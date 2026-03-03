# IDEA: Expert Lifecycle — Capacity-Driven Knowledge Delegation and Self-Organization

## The Vision

Experts are living organisms, not static weight matrices.

They are born (spawned from parents), grow (absorb knowledge), specialize
(narrow their domain), reproduce (combine with complementary experts),
delegate (defer new knowledge to offspring), prune (unlearn what children
know better), and eventually freeze (become read-only memory).

This is not a metaphor — it's a concrete mechanism where the lifecycle
dynamics are the optimization algorithm.

## The Problem This Solves

Current continual learning treats all parameters equally:
- Freeze: binary, all-or-nothing, no granularity
- EWC: parameter importance based on Fisher diagonal — but static, computed
  once at task boundary
- Our meta-optimizer: selects updates that don't hurt old tasks — but still
  treats the model as a monolithic blob

Real intelligence doesn't work this way. Knowledge is **hierarchical and
distributed**: broad concepts (what is a name?) live in different structures
than narrow specializations (what is an English female name?). New information
flows to the right level of specialization. Redundant knowledge is pruned.

## The Lifecycle

### Stage 1: Birth (Bonding)

Two parent experts with complementary knowledge combine to create an offspring.

```
Parent A: English names (broad)
Parent B: Female names (broad)
                  ↓ bond
Child AB: English female names (specialized)
```

The child inherits weights from both parents via genetic crossover —
not a bland average, but SBX/DEAP-style recombination that preserves
complementary knowledge from each parent.

**Key difference from current offspring**: the child is born SMALL. It starts
with fewer parameters or a lower-rank representation of the parent weights.
This is deliberate — a specialized expert doesn't need the full capacity of
a generalist.

Implementation option: child starts with same architecture but with a
**capacity budget** (soft parameter count limit enforced via L1 or sparse
regularization). As it learns, it develops sparse, efficient representations.

### Stage 2: Growth (Fast Learning)

The child is the system's **plasticity zone**. It has:
- High learning rate (tau=1.0, no freeze map)
- Low capacity utilization (lots of room to learn)
- Priority routing for novel inputs

Parents delegate to the child in two cases:

**Case A — "I already know this"**: The input activates the parent strongly
AND the parent's loss on this input is already low. The parent doesn't need
to update — but the child might benefit from seeing this example to build
its specialized representation. The gradient for this input flows primarily
to the child.

```python
parent_loss = compute_loss(parent, input)
if parent_loss < parent_threshold:
    # Parent already knows this — route gradient to child
    child_weight = 0.8
    parent_weight = 0.2  # parent barely updates
```

**Case B — "This would hurt me"**: The input is novel and the gradient
would significantly change the parent's weights (high gradient norm relative
to the parent's frozen stability). The parent protects itself by delegating
to the child.

```python
grad_norm = compute_grad_norm(parent, input)
if grad_norm > parent_disruption_threshold:
    # Novel input too disruptive for parent — child absorbs it
    child_weight = 0.9
    parent_weight = 0.1  # parent barely updates
```

This creates a natural **information funnel**: routine knowledge stays
in parents, novel knowledge flows to children.

### Stage 3: Specialization (Knowledge Crystallization)

As the child accumulates knowledge, its capacity utilization grows. The
child becomes increasingly specialized — its router key converges to a
specific region of input space. This is measured by:

```python
capacity = effective_rank(child_weights) / max_rank
# or: capacity = 1 - sparsity(child_weights)
# or: capacity = child_loss_improvement_rate  (slowing = filling up)
```

The child's learning rate naturally decreases as capacity fills:

```python
child_lr_multiplier = max(0.1, 1.0 - capacity)
```

### Stage 4: Criticality (Capacity Pressure)

When the child approaches full capacity, a cascade begins:

**4a. Freeze**: The child's learning rate drops near zero. It becomes
a specialized read-only expert, like its parents. Its knowledge is
crystallized.

**4b. Forced unlearning in parents**: Here's the novel part. The child
now knows things that the parents also (partially) know. This is
**redundant** — the parameters are wasted.

For each input that activates BOTH the child and a parent:
```python
child_loss = compute_loss(child, input)
parent_loss = compute_loss(parent, input)

if child_loss < parent_loss:
    # Child handles this better — parent should forget this
    parent_unlearn_pressure[input_region] += (parent_loss - child_loss)
```

The parent's freeze map for the relevant parameters INCREASES toward 1.0
(more frozen), but with a twist: parameters that overlap with the child's
specialization are actively PUSHED toward a more general state. This is
**selective unlearning** — the parent doesn't lose all knowledge, just the
specific knowledge that the child has taken over.

Implementation: the parent's weights in the child-overlapping region are
decayed toward their pre-child-birth values (or toward zero, or toward
a more general prior). This frees capacity for the parent to learn new
things in the future.

**4c. Role reversal**: Once the child handles a significant fraction of
the parent's input distribution, routing naturally shifts. The child becomes
the first point of contact for its specialized domain. The parent handles
only out-of-distribution inputs and defers to the child for in-distribution.

This is NOT hardcoded — it emerges from the routing dynamics:
- Child has lower loss on specialized inputs → router scores child higher
- Parent's weights in overlapping region are decayed → parent loss increases
- The routing shift is gradual and reversible

### Stage 5: Reproduction (The Cycle Continues)

A frozen child can now become a parent itself. When a new task arrives
that overlaps with the child's domain:

```
Child AB: English female names (frozen, specialized)
New expert C: Victorian-era names (being learned)
                  ↓ bond (if router keys overlap)
Grandchild ABC: Victorian English female names (specialized²)
```

This creates a **knowledge tree** that grows in specificity:
```
Root: General language model
├── Expert A: English names
│   └── Child AB: English female names
│       └── Grandchild ABC: Victorian English female names
├── Expert B: Female names
│   └── Child BD: French female names
└── Expert C: Company names
    └── Child CE: Tech company names
```

Each node in the tree is frozen. New knowledge flows down to the leaves.
The tree grows organically based on what tasks arrive.

## Why This is Novel

### What exists (precedents)
- **Progressive Neural Networks** (Rusu et al. 2016): add new columns per
  task, never modify old ones. No pruning, no delegation, no lifecycle.
- **Expert Gate** (Aljundi et al. 2017): gate selects experts, but experts
  don't interact, don't spawn children, don't unlearn.
- **PackNet** (Mallya & Lazebnik 2018): prune and reuse capacity, but no
  hierarchical experts, no parent-child dynamics.
- **DEN** (Yoon et al. 2018): dynamically expandable network, adds neurons
  but doesn't create semantic expert relationships.
- **NISPA** (Gurbuz & Dovrolis 2022): neuronal importance based sparsity
  with parameter allocation. Similar capacity notion but no lifecycle.

### What doesn't exist (our contribution)
1. **Capacity-aware experts** that have a measurable "fullness" driving
   their behavior (learn fast when empty, delegate when full)
2. **Knowledge delegation** where parents actively route information to
   children based on disruption potential
3. **Forced unlearning** where children's specialization causes parents
   to prune redundant knowledge, freeing capacity
4. **Role reversal** where children become primary and parents defer
5. **The complete lifecycle** as a unified mechanism: birth → growth →
   specialize → fill → freeze → trigger parent pruning → reproduce

No single paper implements this complete dynamic.

## The Biological Analogy

This isn't just "inspired by biology" — it mirrors specific biological
mechanisms:

| Our System | Biology | Mechanism |
|-----------|---------|-----------|
| Expert bonding | Sexual reproduction | Recombination of complementary genomes |
| Child born small | Offspring are small | Growth requires energy (compute) |
| Fast child learning | Critical period | Young brains learn faster than old ones |
| Parent delegation | Parental care | Parents protect offspring with resources |
| Capacity filling | Neural saturation | Neurons have finite synaptic capacity |
| Forced unlearning | Synaptic pruning | Redundant connections are eliminated |
| Role reversal | Generational transfer | Children surpass parents in specific domains |
| Knowledge tree | Phylogenetic tree | Specialization through branching |

The key insight from neuroscience: **memory consolidation** in the brain
involves transferring knowledge from hippocampus (fast learning, limited
capacity) to cortex (slow learning, large capacity). Our system inverts
this — children are the hippocampus (fast, specialized) and parents are
the cortex (slow, general). But the direction of transfer (cortex prunes
when hippocampus masters) is the same.

## Connection to Meta-Optimizer

The expert lifecycle and meta-optimizer operate at different levels:

```
Meta-optimizer: "How should I update shared parameters THIS step?"
  → Candidate selection, multi-task evaluation, weight-space search

Expert lifecycle: "How should knowledge be DISTRIBUTED across experts?"
  → Birth, growth, delegation, pruning, freezing, reproduction
```

They're complementary:
- Meta-optimizer handles the **per-step** stability-plasticity tradeoff
  for shared parameters (attention, embeddings)
- Expert lifecycle handles the **long-term** knowledge organization
  for expert-specific parameters

Combined: shared params are updated via meta-optimizer (don't forget),
expert params are managed via lifecycle (specialize, delegate, prune).

## Key Metric: Knowledge Efficiency

Define: `knowledge_efficiency = total_useful_knowledge / total_parameters`

Where useful_knowledge = negative average loss across all seen tasks,
and parameters = count of non-zero (or non-frozen) weights.

The lifecycle should INCREASE knowledge efficiency over time:
- Parents prune redundant knowledge → fewer params for same knowledge
- Children specialize → fewer params for specialized knowledge
- The tree structure avoids duplication

Compare against flat MoE (no lifecycle): same number of experts but
no pruning, no delegation, no hierarchy. The flat system should have
lower knowledge efficiency because experts accumulate redundant knowledge.

## Open Questions

1. **How to measure capacity?** Effective rank? Sparsity? Loss improvement
   rate? The choice affects when delegation and pruning trigger.

2. **How to implement forced unlearning?** Decay toward zero? Decay toward
   pre-birth checkpoint? L1 regularization on overlapping parameters?
   The mechanism must be gentle enough to not destroy useful knowledge
   but strong enough to actually free capacity.

3. **When to trigger bonding?** Based on router key proximity? Based on
   task similarity? Based on activation co-occurrence? The bonding trigger
   determines the tree structure.

4. **How deep can the tree go?** At our tiny model scale, even 1 level
   of children may exhaust total parameter budget. The tree depth is
   limited by the model's total capacity. At scale (millions of params),
   deeper trees become viable.

5. **Does this work without the meta-optimizer?** Can the lifecycle alone
   (with standard Adam) prevent forgetting? Or does it need the per-step
   candidate selection for shared parameters?
