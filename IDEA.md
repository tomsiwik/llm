# Living Graph of Micro-Experts (LGME)

> A dynamically growing, graph-structured alternative to monolithic transformer training.
> Formalized against Karpathy's microGPT as the reference implementation.

---

## 1. Motivation

The standard transformer paradigm:

```
[Large Dataset] → [Expensive Batch Training] → [Fixed Model] → [Inference]
```

Problems:
- Training is a one-shot expensive event
- The model is a static artifact — new knowledge requires retraining
- No attribution — outputs cannot be traced to training data
- All parameters participate in every prediction (wasteful)

Proposed paradigm:

```
[Data Stream] → [Novelty Gate] → [Local Update / Node Creation] → [Living Graph] → [Sparse Inference]
```

---

## 2. Formal Definition

### 2.1 The Graph

$$G = (V, E, \Theta, K)$$

| Symbol | Type | Description |
|--------|------|-------------|
| $V = \{v_1, \dots, v_n\}$ | Set | Micro-expert nodes. $n$ is variable (grows/shrinks). |
| $E \subseteq V \times V$ | Set | Directed edges. $(v_i, v_j) \in E$ means $v_i$ can route to $v_j$. |
| $\Theta = \{\theta_1, \dots, \theta_n\}$ | Parameters | Local parameters for each node. $\theta_i \in \mathbb{R}^{p_i}$. |
| $K = \{k_1, \dots, k_n\}$ | Vectors | Router keys. $k_i \in \mathbb{R}^d$ is the "address" of node $v_i$. |

**Constraint:** $G$ must be a DAG (directed acyclic graph) within any single generation step to avoid infinite loops during forward pass. Cross-step cycles (recurrence) are permitted.

### 2.2 Micro-Expert Node

Each node $v_i$ is a small neural network with:

- **Competence set** $C_i \subseteq \{0, \dots, |\text{vocab}|-1\}$: the token subset this node can predict
- **Internal function:**

$$f_i(x; \theta_i) = (h_i, \hat{P}_i)$$

where:

$$h_i = \sigma(W_i^{(1)} x + b_i^{(1)}) \in \mathbb{R}^{d_h}$$

$$\hat{P}_i(c \mid x) = \text{softmax}(W_i^{(2)} h_i) \quad \text{for } c \in C_i$$

- $W_i^{(1)} \in \mathbb{R}^{d_h \times d}$: input projection
- $W_i^{(2)} \in \mathbb{R}^{|C_i| \times d_h}$: output projection
- $\sigma$: activation function (ReLU for parity with microGPT)
- $d_h$: hidden dimension per node (hyperparameter, e.g., 8)

**Parameter count per node:**

$$p_i = d_h \cdot d + d_h + |C_i| \cdot d_h + |C_i| = (d + 1) \cdot d_h + (d_h + 1) \cdot |C_i|$$

For $d = 16$, $d_h = 8$, $|C_i| = 6$: $p_i = 136 + 54 = 190$ parameters.

### 2.3 Context Embedding

Before routing, we need a context representation. We retain the embedding scheme from microGPT:

$$x_t = \text{emb}_{\text{tok}}[\text{token}_t] + \text{emb}_{\text{pos}}[t]$$

where $\text{emb}_{\text{tok}} \in \mathbb{R}^{|\text{vocab}| \times d}$ and $\text{emb}_{\text{pos}} \in \mathbb{R}^{B \times d}$ are shared (not per-node) parameters.

**Note:** These shared embeddings are the **compatibility layer** — they project all tokens into a common space that all nodes can interoperate in.

---

## 3. Router Mechanism

### 3.1 Relevance Scoring

Given context embedding $x \in \mathbb{R}^d$, score each node:

$$r_i(x) = \frac{x \cdot k_i}{\sqrt{d}}$$

Scaled dot-product (same scaling as attention in the reference model, for the same reason: preventing saturation in high dimensions).

### 3.2 Sparse Activation

Select the top-$K$ nodes by relevance:

$$\mathcal{A}(x) = \underset{S \subseteq V, |S| = K}{\arg\max} \sum_{v_i \in S} r_i(x)$$

Compute mixture weights over the active set:

$$\alpha_i = \frac{e^{r_i(x)}}{\sum_{v_j \in \mathcal{A}(x)} e^{r_j(x)}} \quad \text{for } v_i \in \mathcal{A}(x)$$

### 3.3 Graph-Constrained Routing (Optional Extension)

In the basic version, any $K$ nodes can co-activate. In the graph-constrained version, routing respects edge topology:

1. Select top-$K_0$ **root nodes** (nodes with no incoming edges from other active nodes)
2. For each active root, propagate activation along edges:

$$r_j^{(\text{graph})}(x) = r_j(x) + \beta \sum_{(v_i, v_j) \in E, v_i \in \mathcal{A}} \alpha_i \cdot r_j(x)$$

This lets active parent nodes "boost" their children, creating coherent paths through the graph.

---

## 4. Forward Pass (Composition)

### 4.1 Output Composition

The final prediction is a mixture over active nodes:

$$P(c \mid x) = \frac{\sum_{v_i \in \mathcal{A}(x)} \alpha_i \cdot \hat{P}_i(c \mid x) \cdot \mathbb{1}[c \in C_i]}{\sum_{c'} \sum_{v_i \in \mathcal{A}(x)} \alpha_i \cdot \hat{P}_i(c' \mid x) \cdot \mathbb{1}[c' \in C_i]}$$

The denominator re-normalizes because different nodes have different competence sets. If no active node covers token $c$, then $P(c \mid x) = 0$.

**Simplified (if all nodes cover full vocab, i.e., $C_i = \text{vocab}$):**

$$P(c \mid x) = \sum_{v_i \in \mathcal{A}(x)} \alpha_i \cdot \hat{P}_i(c \mid x)$$

### 4.2 Sequential Composition (Multi-Step)

For autoregressive generation (matching microGPT's token-by-token generation):

```
For position t = 0, 1, ..., T:
    x_t = emb_tok[token_t] + emb_pos[t]
    A_t = route(x_t)                          # select active nodes
    P(· | x_t) = compose(A_t, x_t)            # mixture prediction
    token_{t+1} ~ P(· | x_t)                  # sample or argmax
```

### 4.3 Context Propagation Between Steps

A limitation of the basic model: each step routes independently based on $x_t$ alone. The microGPT uses a KV-cache so attention at step $t$ can look back at all prior steps.

To achieve this in LGME, maintain a **running context vector**:

$$\bar{x}_t = \text{RMSNorm}\left(\frac{1}{t+1} \sum_{\tau=0}^{t} x_\tau + \sum_{v_i \in \mathcal{A}_\tau} \alpha_i \cdot h_i\right)$$

This accumulates information from all prior steps and all prior activated nodes. Route using $\bar{x}_t$ instead of $x_t$.

**Alternative:** Each node maintains its own micro-KV-cache (list of its prior activations). This is more faithful to the reference model but increases memory.

---

## 5. Loss Function

### 5.1 Training Loss (Per-Example)

Identical to microGPT — negative log-likelihood:

$$\mathcal{L}(x) = -\frac{1}{T} \sum_{t=1}^{T} \log P(c_t \mid c_{<t})$$

### 5.2 Router Load Balancing Loss (Auxiliary)

Prevent routing collapse (all queries going to a few popular nodes):

$$\mathcal{L}_{\text{balance}} = n \cdot \sum_{i=1}^{n} f_i \cdot m_i$$

where:
- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[v_i \in \mathcal{A}(x_t)]$ (fraction of steps that activated node $i$)
- $m_i = \frac{1}{T} \sum_{t=1}^{T} \alpha_i(x_t)$ (average routing weight to node $i$)

Minimized when load is uniform across nodes. Borrowed from Switch Transformer (Fedus et al., 2021).

### 5.3 Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda_{\text{bal}} \cdot \mathcal{L}_{\text{balance}}$$

---

## 6. Novelty Detection

### 6.1 Surprise Score

For incoming data $x_{\text{new}}$:

$$S(x_{\text{new}}) = \mathcal{L}(x_{\text{new}}) = -\frac{1}{T}\sum_t \log P(c_t \mid c_{<t})$$

This is the model's **surprise** — identical to Shannon information content.

### 6.2 Decision Thresholds

Maintain a running mean $\mu_S$ and standard deviation $\sigma_S$ of surprise scores:

$$\mu_S \leftarrow (1 - \gamma) \cdot \mu_S + \gamma \cdot S(x_{\text{new}})$$

$$\sigma_S \leftarrow (1 - \gamma) \cdot \sigma_S + \gamma \cdot |S(x_{\text{new}}) - \mu_S|$$

Then classify:

$$\text{Decision}(x_{\text{new}}) = \begin{cases} \text{KNOWN} & \text{if } S(x_{\text{new}}) < \mu_S - \sigma_S \\ \text{ADJACENT} & \text{if } \mu_S - \sigma_S \leq S(x_{\text{new}}) \leq \mu_S + 2\sigma_S \\ \text{NOVEL} & \text{if } S(x_{\text{new}}) > \mu_S + 2\sigma_S \end{cases}$$

### 6.3 Garbage Filter

Not all high-surprise input is valuable. Add a structural validity check:

$$\text{Valid}(x) = \begin{cases} 1 & \text{if } \exists v_i \in \mathcal{A}(x) \text{ with } \max_{c} \hat{P}_i(c \mid x) > \delta_{\text{min}} \\ 0 & \text{otherwise} \end{cases}$$

If no active node assigns meaningful probability to any token, the input is likely garbage/adversarial, not genuinely novel.

Learn only if: $\text{Decision} \in \{\text{ADJACENT}, \text{NOVEL}\}$ AND $\text{Valid}(x) = 1$.

---

## 7. Dynamic Learning

### 7.1 KNOWN → Reinforce (Optional)

No parameter update, or a very mild one:

$$\theta_i \leftarrow \theta_i - \eta_{\text{mild}} \cdot \nabla_{\theta_i} \mathcal{L}(x) \quad \text{where } \eta_{\text{mild}} = 0.01 \cdot \eta$$

This is optional. The purpose: prevent slow drift / forgetting of well-known patterns.

### 7.2 ADJACENT → Local Fine-Tuning

Update only the activated nodes:

$$\theta_i \leftarrow \theta_i - \eta \cdot \nabla_{\theta_i} \mathcal{L}(x) \quad \text{for } v_i \in \mathcal{A}(x)$$

Also update their router keys to better capture this data:

$$k_i \leftarrow k_i + \eta_k \cdot \alpha_i \cdot (x - k_i) \quad \text{for } v_i \in \mathcal{A}(x)$$

This pulls each key toward the data it successfully handled (competitive learning / Kohonen update rule).

### 7.3 NOVEL → Node Spawning

**Step 1: Initialize from nearest neighbor.**

$$v_{\text{nearest}} = \underset{v_i \in V}{\arg\min} \|k_i - \bar{x}\|^2$$

$$\theta_{\text{new}} \leftarrow \theta_{\text{nearest}} + \mathcal{N}(0, \epsilon^2 I)$$

Warm start: the new node begins as a noisy copy of the most relevant existing node.

**Step 2: Set competence.**

$$C_{\text{new}} = \{c \mid c \text{ appears in } x_{\text{new}}\} \cup C_{\text{nearest}}$$

The new node covers at minimum the tokens in the novel data plus whatever its parent knew.

**Step 3: Set router key.**

$$k_{\text{new}} = \frac{1}{T} \sum_{t} x_t \quad \text{(mean context embedding of the new data)}$$

**Step 4: Wire edges.**

Connect to the $M$ most co-activated nodes:

$$E \leftarrow E \cup \{(v_j, v_{\text{new}}) \mid v_j \in \text{top-M}(\mathcal{A}(x_{\text{new}}))\}$$

$$E \leftarrow E \cup \{(v_{\text{new}}, v_j) \mid v_j \in \text{top-M}(\mathcal{A}(x_{\text{new}}))\}$$

(Bidirectional: the new node can be reached from and can reach its contextual neighbors.)

**Step 5: Local training.**

$$\theta_{\text{new}} \leftarrow \theta_{\text{new}} - \eta \cdot \nabla_{\theta_{\text{new}}} \mathcal{L}(x_{\text{new}})$$

Only the new node trains. All other parameters frozen.

### 7.4 Consolidation (Merge)

Periodically (every $N_{\text{consol}}$ steps), scan for mergeable pairs:

**Merge condition:**

$$\text{Merge}(v_i, v_j) \iff \cos(k_i, k_j) > \gamma_k \quad \text{AND} \quad \frac{|C_i \cap C_j|}{|C_i \cup C_j|} > \gamma_C$$

Both the router keys and competence sets must be similar.

**Merge operation:**

$$k_{\text{merged}} = \frac{n_i \cdot k_i + n_j \cdot k_j}{n_i + n_j}$$

$$\theta_{\text{merged}} = \frac{n_i \cdot \theta_i + n_j \cdot \theta_j}{n_i + n_j}$$

$$C_{\text{merged}} = C_i \cup C_j$$

Where $n_i$ is the activation count (how many data points node $v_i$ has processed). Edges from both nodes are merged. The two original nodes are deleted.

---

## 8. Equivalence to microGPT (Baseline Behavior)

To validate the architecture, we must show it can replicate microGPT's behavior on the names dataset.

### 8.1 Degenerate Case: Single Node = Linear Model

If $|V| = 1$, $K = 1$ (one node, always active), $C_1 = \text{vocab}$:

$$P(c \mid x) = \hat{P}_1(c \mid x) = \text{softmax}(W^{(2)} \cdot \sigma(W^{(1)} \cdot x))$$

This is a single-hidden-layer MLP — strictly less expressive than the transformer in microGPT.

### 8.2 Replicating Attention via Inter-Node Communication

The key capability microGPT has that a naive LGME lacks is **attention**: the ability for the current token to look back at all previous tokens.

In microGPT (line 129):

$$\text{attn}(Q, K, V)_j = \sum_t \text{softmax}\left(\frac{Q \cdot K_t}{\sqrt{d_h}}\right) \cdot V_{t,j}$$

To replicate this in LGME, we need inter-node communication across time steps.

**Approach: Temporal edges.**

Allow edges not just between nodes but between **(node, timestep)** pairs. At step $t$, node $v_i$ can attend to the outputs of nodes activated at steps $0, \dots, t-1$:

$$\text{attn}_i(t) = \sum_{\tau < t} \sum_{v_j \in \mathcal{A}_\tau} w_{ij\tau} \cdot h_j(\tau)$$

where $w_{ij\tau}$ is a learned or computed attention weight. This is a KV-cache distributed across the graph.

### 8.3 Minimal Replication Configuration

To match microGPT (1 layer, 4 heads, d=16, vocab=27):

| microGPT Component | LGME Equivalent |
|---------------------|-----------------|
| `wte` (27 x 16) | Shared embedding matrix (identical) |
| `wpe` (16 x 16) | Shared position embedding (identical) |
| `attn` (Q,K,V,O projections) | 4 "attention head" nodes, each computing Q·K→weights→V |
| `mlp` (fc1, relu, fc2) | 1 "MLP" node with 64-dim hidden layer |
| `lm_head` (27 x 16) | 1 "output" node or shared output projection |

Total nodes: ~6 (4 attention + 1 MLP + 1 output). All always active ($K = 6$).

This is trivially equivalent — the graph structure just mirrors the transformer block. The architecture becomes interesting when we go beyond this degenerate case.

### 8.4 Target Test

The LGME implementation passes if:

1. With a fixed graph (no dynamic growth), it achieves comparable loss curves on the names dataset
2. Generated names are qualitatively similar (plausible English names)
3. Training and inference produce deterministic results given the same seed

---

## 9. Adversarial Review

### GAP 1: The Top-K Router Is Not Differentiable

**Problem:** $\arg\max$ in the top-K selection (Section 3.2) has zero gradient almost everywhere. We cannot backpropagate through the routing decision.

**Severity:** Critical. Without gradients through the router, the router keys $k_i$ cannot learn from the task loss.

**Mitigations:**
- (a) Use the Kohonen update rule (Section 7.2) instead of backprop for router keys — this is gradient-free.
- (b) Use a differentiable approximation: Gumbel-Softmax over all nodes, then threshold. Adds noise during training.
- (c) Straight-Through Estimator: use hard top-K in forward pass, pretend it was soft in backward pass. Biased but works in practice (used in Switch Transformer).

**Recommendation:** Start with (a) for simplicity. Move to (c) if router doesn't converge.

---

### GAP 2: Compositional Coherence — Can Independently Trained Nodes Cooperate?

**Problem:** Two nodes $v_i$ and $v_j$ trained on different data may produce outputs in incompatible "dialects" of the embedding space. When composed via the mixture in Section 4.1, their outputs may not be coherent.

**Severity:** High. This is the fundamental risk of modular/compositional approaches.

**Analysis:** The shared embedding layer (Section 2.3) partially mitigates this — all nodes receive input in the same coordinate system. But their *outputs* are only constrained by their individual losses, not by how well they compose.

**Mitigations:**
- (a) Regularize node outputs toward a shared representation: $\mathcal{L}_{\text{align}} = \sum_{v_i, v_j \in \mathcal{A}} \|h_i - h_j\|^2$ when co-activated. Expensive.
- (b) Periodically run "composition audits": forward pass through random node combinations, penalize incoherence.
- (c) Initialize all nodes from a shared pretrained seed (Section 7.3, Step 1 already does this). Nodes diverge slowly from a common starting point, maintaining compatibility.

**Recommendation:** Rely on (c) primarily. The warm-start initialization means nodes share a common ancestor and diverge gradually. Monitor in practice.

---

### GAP 3: Context Propagation Is Weaker Than Attention

**Problem:** The running context vector (Section 4.3) is a lossy summary:

$$\bar{x}_t = \text{RMSNorm}\left(\frac{1}{t+1} \sum_{\tau=0}^{t} \text{stuff}\right)$$

Full attention (microGPT) computes a *learned weighted* combination of all past values. The running average is strictly less expressive.

**Severity:** High for sequence modeling. Names are short (max 15 chars) so this may be tolerable. For longer sequences, this would degrade significantly.

**Analysis:** The microGPT attention mechanism has $O(T^2)$ expressive capacity (pairwise interactions between all positions). The running average has $O(T)$ — it compresses all history into a single vector.

**Mitigations:**
- (a) Per-node micro-KV-cache (Section 4.3 alternative). Each active node stores its prior (key, value) pairs. This recovers full attention within each node's activation history. Cost: $O(K \cdot T)$ storage per generation.
- (b) Dedicate specific "memory" nodes whose sole purpose is to maintain context across steps. Route to them always.
- (c) Accept the limitation for now. For the names task (T ≤ 16), the running average may be sufficient.

**Recommendation:** Implement (a) for the baseline replication. It most closely mirrors the reference model.

---

### GAP 4: Node Spawning Criteria Are Fragile

**Problem:** The novelty thresholds $\mu_S \pm k\sigma_S$ (Section 6.2) assume surprise scores are roughly normally distributed. They may not be, especially early in training when the model is poor at everything.

**Severity:** Medium. Bad thresholds → either too many nodes (graph bloat) or too few (underfitting).

**Failure modes:**
- Early training: all data is surprising → spawns a node for every example → 32K nodes (defeats the purpose)
- Late training: thresholds have drifted → novel data incorrectly classified as known

**Mitigations:**
- (a) Warm-up period: disable node spawning for the first $N_{\text{warmup}}$ examples. Let the initial nodes learn before deciding what's novel.
- (b) Hard cap on node creation rate: at most 1 new node per $M$ examples.
- (c) Replace Gaussian thresholds with a percentile-based approach: "novel" = top 5% of surprise scores in a sliding window.

**Recommendation:** Use (a) + (b). Start with a small set of nodes (e.g., one per token in the vocabulary = 27 nodes), train them for 100 steps, then enable dynamic growth with a rate limit.

---

### GAP 5: Consolidation May Destroy Specialized Knowledge

**Problem:** Merging two nodes (Section 7.4) by averaging their parameters assumes their learned functions are similar enough that the average is meaningful. For neural networks, this is generally **not** true — the loss landscape is non-convex.

$$\theta_{\text{merged}} = \frac{n_i \theta_i + n_j \theta_j}{n_i + n_j} \quad \text{← this may land in a high-loss region}$$

**Severity:** Medium-High. Naive weight averaging of neural networks is known to fail except when models share a common initialization and haven't diverged far (Frankle et al., "Linear Mode Connectivity").

**Mitigations:**
- (a) Only merge nodes that are very close in parameter space: $\|\theta_i - \theta_j\| < \epsilon_\theta$ (not just similar keys/competence). This ensures the merge stays in a low-loss basin.
- (b) After merging, run a few local gradient steps on the merged node to "heal" any damage.
- (c) Use soft merging: instead of deleting both nodes, create a merged node and keep the originals with reduced routing priority. Let the router naturally phase them out if the merged node is sufficient.

**Recommendation:** (a) + (b). The warm-start initialization (Section 7.3) helps here too — nodes that were spawned from the same parent are more likely to be mergeable.

---

### GAP 6: No Mechanism for Edge Pruning

**Problem:** Section 7.3 adds edges when spawning nodes. Section 7.4 merges edges during consolidation. But there is no mechanism to *remove* edges that are no longer useful. The graph's edge count grows monotonically.

**Severity:** Low-Medium. Edges are cheap (just pointers), but a dense graph defeats the purpose of sparse routing.

**Mitigation:** Add edge decay. Track edge usage:

$$\text{usage}(e_{ij}) = \frac{\text{times } v_i \text{ and } v_j \text{ co-activated}}{\text{total activations}}$$

Prune edges with usage below a threshold. This naturally removes connections between nodes that turned out to be unrelated.

---

### GAP 7: Comparison to Existing MoE — What Is Actually New?

**Problem:** Sparse Mixture of Experts (Switch Transformer, Mixtral) already does sparse routing to sub-networks. What does LGME add beyond branding?

**Honest assessment of differences:**

| Feature | Standard MoE | LGME |
|---------|-------------|------|
| Number of experts | Fixed at init | Grows dynamically |
| Expert structure | All same architecture | Variable (different $C_i$, $d_h$) |
| Routing | Learned linear → top-K | Learned + graph-constrained |
| Training | Full batch, all experts exist | Incremental, nodes spawn on demand |
| Attribution | None (same as dense model) | Direct (node → training data provenance) |
| Knowledge deletion | Impossible | Delete a node and its edges |

The core novelty is **dynamic growth + novelty gating + attribution**, not the sparse routing itself.

---

### GAP 8: Adversarial Inputs / Poisoning

**Problem:** The novelty gate (Section 6) can be exploited. An adversary could:
- Feed high-novelty garbage → force node spawning → graph bloat (resource exhaustion)
- Feed carefully crafted data that's "adjacent" to critical nodes → slowly corrupt them via local fine-tuning

**Severity:** High in any production context. Less relevant for the names toy problem.

**Mitigations:**
- The garbage filter (Section 6.3) partially addresses the first attack.
- Rate limiting node creation (Gap 4 mitigations) bounds the damage.
- For the corruption attack: maintain a validation set and periodically check node accuracy. Quarantine nodes whose performance degrades.

---

### GAP 9: Theoretical Convergence

**Problem:** Standard SGD on a fixed architecture has well-studied convergence properties. A dynamically growing graph with novelty-gated local updates has **no convergence guarantees** that we can cite.

**Analysis:** The system is closer to online learning / bandit algorithms than to batch optimization. Each decision (route, update, spawn, merge) is locally rational but global optimality is not guaranteed.

**What we can say:**
- Each local update (Section 7.2) decreases loss on the current example (standard SGD guarantee for small enough $\eta$).
- Node spawning is triggered when loss is high, so it targets the regions of highest error.
- Consolidation reduces redundancy.
- But: there's no guarantee the composition of locally-optimal nodes is globally optimal.

**This is the deepest open question in the proposal.**

---

## 10. Implementation Plan (Phase 1: Baseline Replication)

### Goal

Implement LGME in pure Python (matching microGPT's dependency-free constraint) and demonstrate equivalent behavior on the names dataset.

### Phase 1 Constraints (No Dynamic Growth)

To isolate the architecture from the growth mechanism:

- Fixed graph with ~27 initial nodes (one per vocab token) + a small set of "pattern" nodes
- $K = 5$ active nodes per step
- Standard SGD training (same 1000 steps as microGPT)
- No novelty gating, no spawning, no consolidation
- Compare: loss curve, generated name quality, parameter count

### Phase 1 Success Criteria

| Metric | microGPT (reference) | LGME Phase 1 (target) |
|--------|----------------------|----------------------|
| Final loss | ~2.0-2.5 | ~2.0-2.5 |
| Generated names | Plausible English | Plausible English |
| Total params | 4,192 | ~5,000-8,000 (comparable order) |
| Training time | ~10 min (pure Python) | ~10-20 min (acceptable 2x overhead) |

### Phase 2: Dynamic Growth

Enable novelty gating and node spawning. Feed data incrementally. Measure:
- Graph growth curve (nodes over time)
- Per-node attribution (which training examples created which nodes)
- Ability to add new data without retraining existing nodes

---

## 12. The Cognitive Stack

Four structures, layered as a processing pipeline with escalating depth and cost.

### 12.1 The Pipeline

```
                       COST    DEPTH     COGNITIVE ANALOGY
                       ────    ─────     ─────────────────
Input ─→ Bloom Filter   O(1)   Reflex    Subconscious priming
     ─→ Splay Tree      O(1)*  Recall    Working memory / attention
     ─→ HNSW            O(logn) Search   Long-term associative memory
     ─→ ART             O(k·n)  Reason   Conscious novelty resolution

     * amortized
```

Most inputs never reach the bottom. Familiar inputs are handled at Layers 0-1.
Only genuinely novel inputs trigger the expensive ART resonance at Layer 3.

### 12.2 Layer 0: Bloom Filter — Subconscious Priming

Pre-conscious familiarity signal. O(1) check, no false negatives.

Maintain a Bloom filter over character n-grams seen during training:

$$\text{familiar}(x) = \bigwedge_{g \in \text{ngrams}(x)} \text{BF.query}(g)$$

- **BF says "definitely not seen"** → skip straight to HNSW (Layer 2). Fast-track novel inputs.
- **BF says "probably seen"** → proceed to Splay Tree (Layer 1). Familiar path.
- **False positives are harmless** (the splay tree or HNSW will handle it).
- **False negatives are impossible** (guaranteed correctness on the novel detection path).

Implementation: a bit array of size $m$ with $k$ hash functions. For $n$ inserted
items, false positive rate $\approx (1 - e^{-kn/m})^k$. With $m = 2048$, $k = 3$,
$n = 500$ n-grams: FP rate $\approx 3\%$.

### 12.3 Layer 1: Splay Tree — Working Memory

Recently and frequently accessed expert nodes are at the root. O(1) amortized
for the most common access patterns.

Each access splays the activated node to the root:

```
Before accessing [ka]:         After splay:

      [an]                        [ka]       ← now at root
      /   \                       /   \
   [em]   [ma]                 [em]   [an]
   /                                    \
 [ka]                                  [ma]
```

Key properties:
- **Recency:** most recent access is always the root
- **Frequency:** often-accessed nodes cluster near root (keep getting re-splayed)
- **Natural decay:** unused nodes drift to leaves without explicit eviction
- **Entropy-optimal:** amortized cost $O(H(p))$ where $H$ is the access entropy

The splay tree keys are the embedding vectors (or a hash thereof). On a query:
1. Search the tree for the nearest router key
2. If distance $< \theta_{\text{fast}}$: use this expert directly (fast path, System 1)
3. If not found or distance too large: fall through to HNSW (Layer 2)
4. After resolution: splay the winner to root

### 12.4 Layer 2: HNSW — Long-Term Associative Memory

Hierarchical graph for O(log n) approximate nearest neighbor search.

```
Layer 2 (abstract):  [consonant-start] ─────── [vowel-start]
                          │                         │
Layer 1 (category):  [k-names] ── [m-names]   [a-names] ── [e-names]
                      │    │        │            │     │
Layer 0 (specific): [ka-] [ke-]  [ma-] [mi-] [an-] [ar-] [el-]
```

- **Top layers:** sparse, long-range connections (abstract categories)
- **Bottom layers:** dense, fine-grained connections (specific experts)
- **Search:** greedy descent from top layer. At each layer, walk to nearest neighbor,
  then drop down.
- **Insert:** new node enters at a random number of layers (geometric distribution).
  Most nodes are layer 0 only. Few reach layer 1+.

HNSW construction parameter: $M$ = max edges per node per layer.
For $n$ expert nodes, search cost is $O(\log n)$ with high probability.

### 12.5 Layer 3: ART Resonance — Conscious Novelty Resolution

The most expensive layer. Only reached when the input is unfamiliar.

**Match score** between input $x$ and expert $v_i$ with template $w_i$:

$$M(x, v_i) = \frac{\|x \wedge w_i\|}{\|x\|}$$

where $x \wedge w_i$ is element-wise minimum (fuzzy AND): "what fraction of the
input is captured by this expert?"

**Vigilance test** with parameter $\rho \in (0, 1)$:

$$\begin{cases}
M(x, v_i) \geq \rho & \Rightarrow \text{RESONANCE: reinforce } v_i, \text{ update } w_i \\
M(x, v_i) < \rho     & \Rightarrow \text{MISMATCH: disable } v_i, \text{ try next candidate}
\end{cases}$$

**Search-with-reset** (different from top-K):
1. Take candidates from HNSW, ordered by distance
2. Try each: if match $\geq \rho$, resonance (done)
3. If match $< \rho$, disable that expert and try the next
4. If ALL candidates fail: create new expert node

**Reinforcement update** on resonance (the expert learns from the input):

$$w_i^{(\text{new})} = \beta \cdot (x \wedge w_i^{(\text{old})}) + (1 - \beta) \cdot w_i^{(\text{old})}$$

where $\beta$ is a learning rate. The template narrows to capture only the features
that consistently match across inputs.

**ART's guarantee:** old categories are never destroyed by new learning (stability),
and genuinely new patterns always create new categories (plasticity).
This directly addresses Gap 9 (convergence).

### 12.6 Cost Profile — Scales with Novelty

```
Novelty Level     Path Taken                              Cost
────────────────  ──────────────────────────────────────── ───────────
Routine           Bloom(hit) → Splay(root hit)            O(1)
Recently seen     Bloom(hit) → Splay(near root)           O(1) amort.
Familiar type     Bloom(hit) → Splay(miss) → HNSW →      O(log n)
                    ART(resonance on first try)
Novel variant     Bloom(hit) → HNSW → ART(reset,reset,    O(log n + k)
                    resonance on 3rd try)
Truly novel       Bloom(MISS) → HNSW → ART(all fail) →   O(log n + k
                    spawn new node                          + spawn)
```

### 12.7 Graceful Degradation

Each layer is optional. Remove any layer and the system still works, just slower
or less capable:

```
Configuration         Behavior
─────────────────     ──────────────────────────────────────────────
All 4 layers          Full system: fast for familiar, deep for novel
No Bloom filter       No pre-screening. Splay tree handles first pass.
No Splay tree         No recency bias. HNSW handles all lookups.
No HNSW               No hierarchical search. ART scans linearly.
ART only              Everything goes to resonance. Correct but slow.
```

This means we implement bottom-up: ART first (the core), then add each layer
as an optimization on top.

---

## 13. Incremental Surgery Plan

Six surgical cuts. Each is independently testable. Each preserves the existing
loss curve and generated output. The original `microgpt.py` is never modified —
all work happens in a new file that imports/reuses the Value class and helpers.

### 13.1 Overview

```
Cut   What                        Lines added   Behavior change   Enables
────  ──────────────────────────  ───────────   ───────────────   ────────────────
 0    Restructure into functions  ~30           NONE              testable units
 1    Graph wrapper               ~40           NONE              node abstraction
 2    ART vigilance (observe)     ~25           NONE (prints)     novelty scoring
 3    Bloom filter (observe)      ~20           NONE (prints)     familiarity pre-check
 4    Splay tree (cache)          ~45           NONE (cache)      recency acceleration
 5    HNSW index                  ~50           NONE (index)      scalable routing
                                  ─────
                                  ~210 lines total added
```

Compatible mode is maintained throughout: same 4192 params, same loss curve,
same generated names.

### 13.2 Cut 0 — Restructure into Callable Functions

**What:** Extract the forward pass stages from the monolithic `gpt()` into
four standalone functions. No new classes. No new data structures. Pure refactor.

**Touches:** Lines 108-144 of microgpt.py (the `gpt` function).

**Before:**
```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        # ... 25 lines of attention ...
        # ... 5 lines of mlp ...
    logits = linear(x, state_dict['lm_head'])
    return logits
```

**After:**
```python
def embed(token_id, pos_id, sd):
    x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]
    return rmsnorm(x)

def attn(x, li, sd, keys, values):
    x_residual = x
    x = rmsnorm(x)
    q = linear(x, sd[f'layer{li}.attn_wq'])
    k = linear(x, sd[f'layer{li}.attn_wk'])
    v = linear(x, sd[f'layer{li}.attn_wv'])
    keys[li].append(k)
    values[li].append(v)
    x_attn = []
    for h in range(n_head):
        hs = h * head_dim
        q_h = q[hs:hs+head_dim]
        k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
        v_h = [vi[hs:hs+head_dim] for vi in values[li]]
        attn_logits = [sum(q_h[j]*k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                       for t in range(len(k_h))]
        attn_weights = softmax(attn_logits)
        head_out = [sum(attn_weights[t]*v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)]
        x_attn.extend(head_out)
    x = linear(x_attn, sd[f'layer{li}.attn_wo'])
    return [a + b for a, b in zip(x, x_residual)]

def mlp(x, li, sd):
    x_residual = x
    x = rmsnorm(x)
    x = linear(x, sd[f'layer{li}.mlp_fc1'])
    x = [xi.relu() for xi in x]
    x = linear(x, sd[f'layer{li}.mlp_fc2'])
    return [a + b for a, b in zip(x, x_residual)]

def output(x, sd):
    return linear(x, sd['lm_head'])

def gpt(token_id, pos_id, keys, values):      # unchanged interface
    x = embed(token_id, pos_id, state_dict)
    for li in range(n_layer):
        x = attn(x, li, state_dict, keys, values)
        x = mlp(x, li, state_dict)
    return output(x, state_dict)
```

**Verification:** Run both versions with the same seed. Loss at every step must
be identical to 10+ decimal places. The `gpt()` function signature and all call
sites are unchanged — training loop and inference loop are untouched.

### 13.3 Cut 1 — Graph Wrapper

**What:** Wrap `state_dict` and the four functions into a Graph with typed Nodes.
The `gpt()` function becomes a 4-line graph traversal.

**Adds:**
```python
class Node:
    def __init__(self, node_type, forward_fn, param_keys):
        self.node_type = node_type
        self.forward_fn = forward_fn       # one of: attn, mlp, output
        self.param_keys = param_keys       # which state_dict keys this node owns
        self.kv_keys = []                  # KV-cache (for attn nodes)
        self.kv_values = []

class Graph:
    def __init__(self, state_dict, nodes, config):
        self.sd = state_dict
        self.nodes = nodes
        self.config = config
        self.params = [p for mat in state_dict.values()
                       for row in mat for p in row]

    def reset_kv(self):
        for n in self.nodes:
            if n.node_type == 'attn':
                n.kv_keys, n.kv_values = [], []

    def forward(self, token_id, pos_id):
        x = embed(token_id, pos_id, self.sd)
        for node in self.nodes:
            if node.node_type == 'output':
                return node.forward_fn(x, self.sd)
            elif node.node_type == 'attn':
                x = node.forward_fn(x, 0, self.sd,
                        [node.kv_keys], [node.kv_values])
            else:
                x = node.forward_fn(x, 0, self.sd)
        return x

# Construction (replaces direct state_dict usage):
graph = Graph(state_dict, [
    Node('attn',   attn,   ['layer0.attn_wq','layer0.attn_wk',
                             'layer0.attn_wv','layer0.attn_wo']),
    Node('mlp',    mlp,    ['layer0.mlp_fc1','layer0.mlp_fc2']),
    Node('output', output, ['lm_head']),
], config={'n_head': n_head, 'head_dim': head_dim})
```

**Training loop change:** Replace `gpt(token_id, pos_id, keys, values)` →
`graph.forward(token_id, pos_id)`. Replace KV-cache reset → `graph.reset_kv()`.
Replace `params` → `graph.params`. Everything else stays.

**Verification:** Identical loss curve. The graph.forward() calls the same
functions with the same data.

### 13.4 Cut 2 — ART Vigilance (Observation Only)

**What:** After each training step, compute the ART match score and classify the
input as KNOWN/ADJACENT/NOVEL. Print alongside loss. No parameter changes.

**Adds:**
```python
class ART:
    def __init__(self, rho=0.7, beta=0.05):
        self.rho = rho                      # vigilance threshold
        self.mu = 3.3                       # running mean surprise (init: ln(27))
        self.sigma = 0.5                    # running std surprise
        self.beta = beta                    # EMA decay

    def classify(self, loss_val):
        surprise = loss_val
        # Update running stats
        self.mu = (1 - self.beta) * self.mu + self.beta * surprise
        self.sigma = (1 - self.beta) * self.sigma + self.beta * abs(surprise - self.mu)
        # Classify
        if surprise < self.mu - self.sigma:
            return 'KNOWN'
        elif surprise > self.mu + 2 * self.sigma:
            return 'NOVEL'
        else:
            return 'ADJACENT'

art = ART()
```

**Training loop change:** One line after loss computation:
```python
tag = art.classify(loss.data)
print(f"step {step+1:4d} | loss {loss.data:.4f} | {tag}", end='\r')
```

**Verification:** Same loss values. Same generated names. ART is pure observation.
We can count how many steps are KNOWN/ADJACENT/NOVEL to validate the thresholds
before ever using them to drive behavior.

### 13.5 Cut 3 — Bloom Filter (Observation Only)

**What:** Track character bigrams seen during training. Before each step, check if
the current name's bigrams are all familiar. Print the result. No routing changes.

**Adds:**
```python
class BloomFilter:
    def __init__(self, size=2048, num_hashes=3):
        self.bits = [0] * size
        self.k = num_hashes

    def _hashes(self, item):
        # Simple hash family using Python's built-in hash
        return [hash((item, i)) % len(self.bits) for i in range(self.k)]

    def add(self, item):
        for h in self._hashes(item):
            self.bits[h] = 1

    def query(self, item):
        return all(self.bits[h] for h in self._hashes(item))

    def check_name(self, name):
        bigrams = [name[i:i+2] for i in range(len(name)-1)]
        novel = [bg for bg in bigrams if not self.query(bg)]
        for bg in bigrams:
            self.add(bg)
        return novel                        # empty list = all familiar

bf = BloomFilter()
```

**Training loop change:** One line before the forward pass:
```python
novel_bgs = bf.check_name(doc)
tag = art.classify(loss.data)
bf_tag = 'BF:novel' if novel_bgs else 'BF:familiar'
print(f"step {step+1:4d} | loss {loss.data:.4f} | {tag} | {bf_tag}", end='\r')
```

**Verification:** Same loss values. Same generated names. Bloom filter is pure
observation. We can track how often the Bloom filter agrees with ART to validate
its usefulness as a pre-filter.

### 13.6 Cut 4 — Splay Tree (Routing Cache)

**What:** Cache the most recently activated expert path. Before routing, check if
the current input is similar to a recently cached input. If so, reuse the same
path (in compatible mode this is always the same path, so it's a no-op — but the
structure is in place).

**Adds:**
```python
class SplayNode:
    __slots__ = ('key', 'value', 'left', 'right')
    def __init__(self, key, value):
        self.key = key                      # hash of the input embedding
        self.value = value                  # the expert path used
        self.left = self.right = None

class SplayTree:
    def __init__(self):
        self.root = None
        self.hits = 0
        self.misses = 0

    def _splay(self, root, key):
        if root is None or root.key == key:
            return root
        if key < root.key:
            if root.left is None:
                return root
            if key < root.left.key:         # zig-zig
                root.left.left = self._splay(root.left.left, key)
                root = self._rotate_right(root)
            elif key > root.left.key:       # zig-zag
                root.left.right = self._splay(root.left.right, key)
                if root.left.right:
                    root.left = self._rotate_left(root.left)
            return self._rotate_right(root) if root.left else root
        else:
            if root.right is None:
                return root
            if key > root.right.key:        # zag-zag
                root.right.right = self._splay(root.right.right, key)
                root = self._rotate_left(root)
            elif key < root.right.key:      # zag-zig
                root.right.left = self._splay(root.right.left, key)
                if root.right.left:
                    root.right = self._rotate_right(root.right)
            return self._rotate_left(root) if root.right else root

    def _rotate_left(self, x):
        y = x.right; x.right = y.left; y.left = x; return y

    def _rotate_right(self, x):
        y = x.left; x.left = y.right; y.right = x; return y

    def lookup(self, key):
        self.root = self._splay(self.root, key)
        if self.root and self.root.key == key:
            self.hits += 1
            return self.root.value
        self.misses += 1
        return None

    def insert(self, key, value):
        if self.root is None:
            self.root = SplayNode(key, value); return
        self.root = self._splay(self.root, key)
        if self.root.key == key:
            self.root.value = value; return
        node = SplayNode(key, value)
        if key < self.root.key:
            node.right = self.root; node.left = self.root.left
            self.root.left = None
        else:
            node.left = self.root; node.right = self.root.right
            self.root.right = None
        self.root = node

cache = SplayTree()
```

**Training loop change:**
```python
# Before forward pass:
emb_key = hash(tuple(tokens))              # simple hash of token sequence
cached_path = cache.lookup(emb_key)

# After forward pass:
cache.insert(emb_key, 'default_path')      # in compatible mode: always same path

# At end of training:
print(f"splay cache: {cache.hits} hits / {cache.misses} misses")
```

**Verification:** Same loss values. Same generated names. The splay tree is a
cache that, in compatible mode, never changes the computation. But it's measuring
the access pattern: how often the same token sequence repeats, and which patterns
are "hot."

### 13.7 Cut 5 — HNSW Index

**What:** Index all expert nodes' router keys in a simple navigable small-world
graph. In compatible mode with 3 nodes, this is trivially small but exercises
the full insertion and search code path.

**Adds:**
```python
class HNSWNode:
    __slots__ = ('id', 'key', 'neighbors')
    def __init__(self, id, key, max_layers):
        self.id = id
        self.key = key                      # list[float], the router key vector
        self.neighbors = [[] for _ in range(max_layers)]  # per-layer neighbor lists

class HNSW:
    def __init__(self, d, M=4, ef=8):
        self.d = d                          # dimension of key vectors
        self.M = M                          # max neighbors per layer
        self.ef = ef                        # search beam width
        self.nodes = []
        self.max_level = 0
        self.entry = None

    def _distance(self, a, b):
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    def _random_level(self):
        level = 0
        while random.random() < 0.5 and level < 4:
            level += 1
        return level

    def insert(self, key, node_id):
        level = self._random_level()
        hnode = HNSWNode(node_id, key, level + 1)
        if self.entry is None:
            self.entry = hnode; self.max_level = level
            self.nodes.append(hnode); return

        # Greedy search from top to insertion level
        curr = self.entry
        for lv in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                for nb in curr.neighbors[min(lv, len(curr.neighbors)-1)]:
                    if self._distance(key, nb.key) < self._distance(key, curr.key):
                        curr = nb; changed = True

        # Insert and connect at each level
        for lv in range(min(level, self.max_level), -1, -1):
            # Find ef nearest at this level
            candidates = sorted(self.nodes,
                                key=lambda n: self._distance(key, n.key))[:self.ef]
            neighbors = candidates[:self.M]
            if lv < len(hnode.neighbors):
                hnode.neighbors[lv] = neighbors
            for nb in neighbors:
                if lv < len(nb.neighbors) and len(nb.neighbors[lv]) < self.M:
                    nb.neighbors[lv].append(hnode)

        if level > self.max_level:
            self.max_level = level; self.entry = hnode
        self.nodes.append(hnode)

    def search(self, query, k=3):
        if not self.nodes:
            return []
        curr = self.entry
        for lv in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for nb in curr.neighbors[min(lv, len(curr.neighbors)-1)]:
                    if self._distance(query, nb.key) < self._distance(query, curr.key):
                        curr = nb; changed = True
        # At level 0: collect k nearest
        return sorted(self.nodes,
                       key=lambda n: self._distance(query, n.key))[:k]

hnsw = HNSW(d=n_embd)
# Insert the 3 compatible-mode nodes with their mean embedding as router key
for i, node in enumerate(graph.nodes):
    key = [0.0] * n_embd                   # placeholder key (to be refined)
    hnsw.insert(key, i)
```

**Training loop change:** None needed in compatible mode. The HNSW simply exists
and can be queried. Add an optional diagnostic:
```python
# At end of training:
results = hnsw.search([0.0]*n_embd, k=3)
print(f"hnsw nodes: {len(hnsw.nodes)}, levels: {hnsw.max_level}")
```

**Verification:** Same loss values. Same generated names. HNSW is an index
structure sitting alongside the graph, exercised but not yet driving routing.

### 13.8 Wiring Sequence — When Each Layer Starts Driving Behavior

After all 6 cuts are in and verified, we activate the cognitive stack in order:

```
Phase    What activates                   Effect
──────   ─────────────────────────────    ──────────────────────────────
Phase 0  Nothing (compatible mode)        Identical to microGPT.

Phase 1  ART drives train/skip decision  KNOWN inputs → reduced learning rate.
         (Cut 2 becomes active)           NOVEL inputs → full learning rate.
                                          First observable behavior change.

Phase 2  Bloom filter gates ART           BF "definitely novel" → skip to HNSW.
         (Cut 3 feeds into Cut 2)         BF "probably familiar" → splay fast path.
                                          Reduces unnecessary ART invocations.

Phase 3  Splay tree caches routing        Repeated patterns → O(1) path reuse.
         (Cut 4 becomes active)           Cache hit → skip HNSW search entirely.
                                          Measurable speedup on sequential data.

Phase 4  HNSW drives expert selection     Multiple experts per stage enabled.
         (Cut 5 becomes active)           Router selects subsets via HNSW search.
                                          This is where LGME diverges from
                                          monolithic GPT.

Phase 5  ART node spawning enabled        NOVEL → create new expert node.
         (Cut 2 extended)                 Insert into HNSW, splay tree, bloom
                                          filter. The graph grows. This is the
                                          "living" behavior.
```

Each phase is a single boolean flag or threshold change. No structural code changes
between phases.

---

## 15. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $d$ | Embedding dimension (16 in reference model) |
| $d_h$ | Hidden dimension per micro-expert node |
| $K$ | Number of active nodes per routing decision |
| $n$ | Current number of nodes in the graph (variable) |
| $T$ | Sequence length of current input |
| $B$ | Block size / max context window |
| $C_i$ | Competence set of node $v_i$ |
| $\alpha_i$ | Routing weight to node $v_i$ |
| $r_i(x)$ | Relevance score of node $v_i$ for input $x$ |
| $S(x)$ | Surprise / novelty score of input $x$ |
| $\mu_S, \sigma_S$ | Running statistics of surprise scores |
| $n_i$ | Activation count of node $v_i$ (lifetime usage) |
| $\eta$ | Learning rate |
| $\gamma$ | EMA decay factor for running statistics |

---

## Appendix A: Data Model Specification

Exact data structures required for LGME to be 100% compatible with microGPT.

### A.1 microGPT Data Model (Reference)

Every piece of state in the reference implementation, categorized by lifetime.

#### A.1.1 Persistent State (survives all steps)

```
state_dict : dict[str, list[list[Value]]]
├── 'wte'              [27][16]  Value    token embeddings
├── 'wpe'              [16][16]  Value    position embeddings
├── 'lm_head'          [27][16]  Value    output projection
├── 'layer0.attn_wq'   [16][16]  Value    query projection
├── 'layer0.attn_wk'   [16][16]  Value    key projection
├── 'layer0.attn_wv'   [16][16]  Value    value projection
├── 'layer0.attn_wo'   [16][16]  Value    attention output projection
├── 'layer0.mlp_fc1'   [64][16]  Value    MLP up-project (4x expansion)
└── 'layer0.mlp_fc2'   [16][64]  Value    MLP down-project

params : list[Value]  len=4192   flat reference into state_dict (same objects)

optimizer:
├── m   : list[float]  len=4192  first moment  (Adam momentum)
├── v   : list[float]  len=4192  second moment (Adam RMSprop)
├── lr  : float        0.01      base learning rate
├── β1  : float        0.85      momentum decay
├── β2  : float        0.99      variance decay
└── ε   : float        1e-8      denominator stability
```

#### A.1.2 Per-Document State (rebuilt each training step, garbage collected after)

```
doc    : str                          e.g. "emma"
tokens : list[int]    len=2+len(doc)  e.g. [26, 4, 12, 12, 0, 26]
n      : int          ≤ 16           num prediction positions

KV-cache (grows as tokens are processed):
├── keys   : list[list[list[Value]]]  shape [1][num_past][16]
└── values : list[list[list[Value]]]  shape [1][num_past][16]

losses : list[Value]  grows to n      one NLL per position
loss   : Value        scalar          (1/n) * sum(losses) — autograd root
```

#### A.1.3 Per-Token State (within single gpt() call, not retained)

Trace of every intermediate and its exact shape:

```
gpt(token_id: int, pos_id: int, keys, values) → logits: list[Value] len=27

STEP   VARIABLE       SHAPE       OPERATION              MATH
────────────────────────────────────────────────────────────────────
1      tok_emb        [16]        wte[token_id]          table lookup
2      pos_emb        [16]        wpe[pos_id]            table lookup
3      x              [16]        tok + pos              x_i = tok_i + pos_i
4      x              [16]        rmsnorm(x)             x_i / √(mean(x²) + ε)

─── ENTER LAYER 0 ──────────────────────────────────────────────────
5      x_residual     [16]        copy of x              saved for skip connection
6      x              [16]        rmsnorm(x)             normalize before attn

7      q              [16]        linear(x, wq)          q_i = Σ_j wq[i][j] · x[j]
8      k              [16]        linear(x, wk)          k_i = Σ_j wk[i][j] · x[j]
9      v              [16]        linear(x, wv)          v_i = Σ_j wv[i][j] · x[j]
10     keys[0]        append k    KV-cache store          temporal memory
11     values[0]      append v    KV-cache store          temporal memory

─── FOR EACH HEAD h ∈ {0,1,2,3}: ───────────────────────────────────
12     q_h            [4]         q[h*4 : h*4+4]         slice
13     k_h            [t+1][4]    cached keys, sliced     all past + current
14     v_h            [t+1][4]    cached values, sliced   all past + current
15     attn_logits    [t+1]       q_h · k_h[τ] / √4      scaled dot-product
16     attn_weights   [t+1]       softmax(attn_logits)    attention distribution
17     head_out       [4]         Σ_τ weights[τ]·v_h[τ]   weighted value sum
18     x_attn         extend      accumulate heads        → grows to [16]
─── END HEAD LOOP ──────────────────────────────────────────────────

19     x              [16]        linear(x_attn, wo)      output projection
20     x              [16]        x + x_residual          residual connection

21     x_residual     [16]        copy of x               saved for skip connection
22     x              [16]        rmsnorm(x)              normalize before MLP
23     x              [64]        linear(x, mlp_fc1)      up-project
24     x              [64]        relu(x)                 activation
25     x              [16]        linear(x, mlp_fc2)      down-project
26     x              [16]        x + x_residual          residual connection
─── EXIT LAYER 0 ───────────────────────────────────────────────────

27     logits         [27]        linear(x, lm_head)      final projection
```

#### A.1.4 Data Flow Diagram

```
token_id ──→ wte ──┐
                    ├──→ add ──→ rmsnorm ──→ ┐
pos_id ────→ wpe ──┘                         │
                                             ▼
                         ┌─── LAYER 0 ────────────────────────────┐
                         │                                        │
                         │  x_residual ←── x                      │
                         │                 │                       │
                         │            rmsnorm                      │
                         │            ╱   │   ╲                    │
                         │          wq   wk   wv                   │
                         │          │     │    │                   │
                         │          q   [k→cache] [v→cache]        │
                         │          │     │    │                   │
                         │     ┌────┴─────┴────┴────┐              │
                         │     │  4x ATTENTION HEAD  │             │
                         │     │  q_h·k_h/√d → sfmx │             │
                         │     │  → weighted v_h sum │             │
                         │     └─────────┬───────────┘             │
                         │          concat [16]                    │
                         │               │                         │
                         │              wo (linear)                │
                         │               │                         │
                         │           add ← x_residual              │
                         │               │                         │
                         │          x_residual ←── x               │
                         │               │                         │
                         │           rmsnorm                       │
                         │               │                         │
                         │          mlp_fc1 [16→64]                │
                         │               │                         │
                         │             relu                        │
                         │               │                         │
                         │          mlp_fc2 [64→16]                │
                         │               │                         │
                         │           add ← x_residual              │
                         │               │                         │
                         └───────────────┼────────────────────────┘
                                         │
                                         ▼
                                    lm_head [16→27]
                                         │
                                         ▼
                                   logits [27]
```

### A.2 LGME Data Model (Compatible Configuration)

#### A.2.1 Design Principle: Decompose by Computational Role

The microGPT forward pass has 4 distinct computational stages. LGME maps each
stage to a **node type**. In compatible mode, there is exactly one node per stage,
all always active, executed in a fixed topological order.

```
microGPT stage        LGME node type       Params owned
──────────────────────────────────────────────────────────
wte + wpe + add       (shared)             emb_tok, emb_pos      (688)
+ rmsnorm             not a node; shared infrastructure
                      ─────────────────────────────────────
attn block            AttnNode             wq, wk, wv, wo        (1024)
                      ─────────────────────────────────────
mlp block             MLPNode              fc1, fc2               (2048)
                      ─────────────────────────────────────
lm_head               OutputNode           lm_head               (432)
                      ─────────────────────────────────────
                                           TOTAL:                 4192 ✓
```

#### A.2.2 Shared Infrastructure (Not Per-Node)

Identical to microGPT. All nodes read from these; only the optimizer writes to them.

```python
shared = {
    'emb_tok': list[list[Value]],   # [vocab_size][d]  =  [27][16]  →  432 params
    'emb_pos': list[list[Value]],   # [block_size][d]  =  [16][16]  →  256 params
}
```

The embedding step is shared infrastructure because every possible path through the
graph begins with the same operation:

```
x = emb_tok[token_id] + emb_pos[pos_id]
x = rmsnorm(x)
```

This is the **compatibility layer** — it guarantees all nodes receive input in the
same coordinate system.

#### A.2.3 Node Types

**AttnNode** — Self-attention block with residual

```python
class AttnNode:
    node_type = 'attn'
    params: {
        'wq': list[list[Value]],    # [d][d]     =  [16][16]  →  256 params
        'wk': list[list[Value]],    # [d][d]     =  [16][16]  →  256 params
        'wv': list[list[Value]],    # [d][d]     =  [16][16]  →  256 params
        'wo': list[list[Value]],    # [d][d]     =  [16][16]  →  256 params
    }                               #                    total:  1024 params
    kv_cache: {
        'keys':   list[list[Value]],  # [num_past][d], grows per token
        'values': list[list[Value]],  # [num_past][d], grows per token
    }

    def forward(self, x: list[Value]) → list[Value]:  # [d] → [d]
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, self.params['wq'])               # [d]
        k = linear(x, self.params['wk'])               # [d]
        v = linear(x, self.params['wv'])               # [d]
        self.kv_cache['keys'].append(k)
        self.kv_cache['values'].append(v)
        x_attn = []
        for h in range(n_head):                         # 4 heads
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]                     # [4]
            k_h = [ki[hs:hs+head_dim] for ki in self.kv_cache['keys']]
            v_h = [vi[hs:hs+head_dim] for vi in self.kv_cache['values']]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, self.params['wo'])
        x = [a + b for a, b in zip(x, x_residual)]     # residual
        return x                                         # [d]
```

**MLPNode** — Feed-forward block with residual

```python
class MLPNode:
    node_type = 'mlp'
    params: {
        'fc1': list[list[Value]],   # [4*d][d]   =  [64][16]  →  1024 params
        'fc2': list[list[Value]],   # [d][4*d]   =  [16][64]  →  1024 params
    }                               #                    total:  2048 params

    def forward(self, x: list[Value]) → list[Value]:  # [d] → [d]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, self.params['fc1'])              # [4*d]
        x = [xi.relu() for xi in x]                    # [4*d]
        x = linear(x, self.params['fc2'])              # [d]
        x = [a + b for a, b in zip(x, x_residual)]    # residual
        return x                                        # [d]
```

**OutputNode** — Final projection to logits

```python
class OutputNode:
    node_type = 'output'
    params: {
        'lm_head': list[list[Value]],  # [vocab][d] = [27][16]  →  432 params
    }

    def forward(self, x: list[Value]) → list[Value]:   # [d] → [vocab]
        return linear(x, self.params['lm_head'])        # [27]
```

#### A.2.4 The Graph Object

```python
class Graph:
    # --- Structure ---
    nodes:  list[Node]                  # all nodes (AttnNode | MLPNode | OutputNode)
    edges:  dict[int, list[int]]        # adjacency: node_id → [downstream node_ids]
    topo:   list[int]                   # precomputed topological execution order

    # --- Shared params (compatibility layer) ---
    emb_tok: list[list[Value]]          # [27][16]
    emb_pos: list[list[Value]]          # [16][16]

    # --- Router (disabled in compatible mode) ---
    router_keys: list[list[float]]      # [n_nodes][d], one key per node
    top_k:       int                    # K active nodes per step

    # --- All params (flat, for optimizer) ---
    params: list[Value]                 # flat list of ALL trainable Values

    # --- Optimizer ---
    adam_m: list[float]                 # first moment,  len = len(params)
    adam_v: list[float]                 # second moment, len = len(params)

    # --- Config ---
    config: {
        'n_embd':     16,
        'block_size': 16,
        'n_head':     4,
        'head_dim':   4,
        'vocab_size': 27,
        'lr':         0.01,
        'beta1':      0.85,
        'beta2':      0.99,
        'eps':        1e-8,
    }
```

#### A.2.5 Compatible Configuration (Exact microGPT Equivalence)

```
Graph:
├── shared
│   ├── emb_tok    [27][16]   = microGPT wte             432 params
│   └── emb_pos    [16][16]   = microGPT wpe             256 params
│
├── nodes[0]: AttnNode
│   ├── wq         [16][16]   = microGPT layer0.attn_wq  256 params
│   ├── wk         [16][16]   = microGPT layer0.attn_wk  256 params
│   ├── wv         [16][16]   = microGPT layer0.attn_wv  256 params
│   └── wo         [16][16]   = microGPT layer0.attn_wo  256 params
│
├── nodes[1]: MLPNode
│   ├── fc1        [64][16]   = microGPT layer0.mlp_fc1  1024 params
│   └── fc2        [16][64]   = microGPT layer0.mlp_fc2  1024 params
│
├── nodes[2]: OutputNode
│   └── lm_head    [27][16]   = microGPT lm_head         432 params
│
├── edges:  {0: [1], 1: [2]}  (linear chain: attn → mlp → output)
├── topo:   [0, 1, 2]         (fixed execution order)
├── top_k:  3                  (all nodes always active)
│
└── params: list[Value]  len=4192  (identical count to microGPT)
```

#### A.2.6 Compatible Forward Pass

```python
def forward(graph, token_id, pos_id):
    # Shared embedding (identical to microGPT lines 109-112)
    x = [t + p for t, p in zip(graph.emb_tok[token_id],
                                graph.emb_pos[pos_id])]
    x = rmsnorm(x)

    # Execute nodes in topological order
    for node_id in graph.topo:
        node = graph.nodes[node_id]
        if node.node_type == 'output':
            return node.forward(x)          # returns logits [27]
        else:
            x = node.forward(x)             # returns x [16]
```

When the same random seed initializes the weights, this produces **byte-identical**
results to microGPT's `gpt()` function.

#### A.2.7 Extended Configuration (Beyond Compatibility)

When the router is enabled and the graph grows, the forward pass becomes:

```python
def forward_routed(graph, token_id, pos_id):
    # Shared embedding (always the same)
    x = [t + p for t, p in zip(graph.emb_tok[token_id],
                                graph.emb_pos[pos_id])]
    x = rmsnorm(x)

    # Route: select which nodes to activate
    active = route(graph, x, graph.top_k)   # returns K (node, weight) pairs

    # Partition active nodes by type, execute in stage order
    attn_nodes  = [(n, w) for n, w in active if n.node_type == 'attn']
    mlp_nodes   = [(n, w) for n, w in active if n.node_type == 'mlp']
    output_node = [(n, w) for n, w in active if n.node_type == 'output'][0]

    # Stage 1: Attention (mixture of active attn nodes)
    if attn_nodes:
        total_w = sum(w for _, w in attn_nodes)
        x_attn = [Value(0)] * len(x)
        for node, w in attn_nodes:
            out = node.forward(x)
            x_attn = [a + (w/total_w) * o for a, o in zip(x_attn, out)]
        x = x_attn

    # Stage 2: MLP (mixture of active mlp nodes)
    if mlp_nodes:
        total_w = sum(w for _, w in mlp_nodes)
        x_mlp = [Value(0)] * len(x)
        for node, w in mlp_nodes:
            out = node.forward(x)
            x_mlp = [a + (w/total_w) * o for a, o in zip(x_mlp, out)]
        x = x_mlp

    # Stage 3: Output (single output node)
    logits = output_node[0].forward(x)
    return logits
```

This preserves the embed → attn → mlp → output pipeline but allows **multiple
parallel experts per stage**, mixed by router weights.

#### A.2.8 Param Accounting — Why the Counts Match

```
microGPT                          LGME compatible
─────────────────────────────     ─────────────────────────────
wte           27×16  = 432        shared.emb_tok   27×16 = 432
wpe           16×16  = 256        shared.emb_pos   16×16 = 256
layer0.wq     16×16  = 256        nodes[0].wq      16×16 = 256
layer0.wk     16×16  = 256        nodes[0].wk      16×16 = 256
layer0.wv     16×16  = 256        nodes[0].wv      16×16 = 256
layer0.wo     16×16  = 256        nodes[0].wo      16×16 = 256
layer0.fc1    64×16  = 1024       nodes[1].fc1     64×16 = 1024
layer0.fc2    16×64  = 1024       nodes[1].fc2     16×64 = 1024
lm_head       27×16  = 432        nodes[2].lm_head 27×16 = 432
──────────────────────────────    ─────────────────────────────
TOTAL                  4192       TOTAL                   4192
```

Zero overhead in compatible mode. The router keys and graph edges are metadata
(plain floats and ints), not Value nodes — they don't participate in autograd
and don't count toward the trainable parameter total.

#### A.2.9 State Lifecycle Comparison

```
                    microGPT                    LGME
                    ────────────────────        ────────────────────
Permanent state     state_dict (flat dict)      Graph object
                    params (flat list)            ├── nodes[].params
                    m[], v[] (Adam)               ├── shared (emb)
                                                  ├── params (flat list)
                                                  └── adam_m[], adam_v[]

Per-document        keys[layer][time][d]        nodes[i].kv_cache (per AttnNode)
(reset each step)   values[layer][time][d]      same
                    losses[]                     losses[]
                    loss (autograd root)          loss (autograd root)

Per-token           x, q, k, v, logits          same (inside node.forward())
(within gpt())      attn_logits, weights         same
                    head_out, x_attn             same

Autograd graph      ~50K Value nodes/step        ~50K Value nodes/step (identical)
(implicit,          built forward, consumed       built forward, consumed
garbage collected)  backward                      backward
```

The only structural difference: microGPT stores the KV-cache in a flat
`keys[layer_idx]` list, while LGME stores it inside each AttnNode. In compatible
mode with 1 AttnNode, these are the same data under a different address.
