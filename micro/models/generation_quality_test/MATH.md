# Generation Quality Test: Mathematical Framework

## Problem Statement

Given a base model $f_\theta$ and $N=5$ domain-specialized LoRA adapters
$\{\Delta W_i = B_i A_i^T\}_{i=1}^N$, does routed composition produce
measurably better generated text than the base model alone?

## Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| $f_\theta$ | - | Base model (BitNet-2B-4T) |
| $A_i$ | $(d, r)$ | Frozen Grassmannian projection for expert $i$ |
| $B_i$ | $(r, d_{out})$ | Trained ternary B-matrix for expert $i$ |
| $d$ | $2560$ | Hidden dimension |
| $r$ | $16$ | LoRA rank |
| $N$ | $5$ | Number of domain experts |
| $k$ | $2$ | Top-k routing |
| $w_i$ | $\mathbb{R}$ | Routing weight for expert $i$ |

## Three Configurations

### 1. Base Only
$$y = f_\theta(x)$$

### 2. Uniform 1/N Composition
$$y = f_\theta(x) + \frac{\alpha}{N} \sum_{i=1}^N x A_i Q(B_i)$$

where $Q(\cdot)$ is STE ternary quantization and $\alpha = 20$ is the LoRA scale.

### 3. Oracle-Routed Top-2
$$y = f_\theta(x) + \alpha \sum_{i=1}^N w_i \cdot x A_i Q(B_i)$$

where $w_i$ are oracle routing weights: $w_{\text{correct}} = 0.7$,
$w_{\text{secondary}} = 0.3$, all others $= 0$.

Oracle routing is justified because prior experiments showed 99.9% routing
head accuracy on these 5 domains. This gives us the upper bound on routing
benefit.

## Scoring Metrics

### Domain Keyword Density (DKD)
$$\text{DKD}(t, d) = \frac{|\{w \in t : w \in K_d\}|}{|t|}$$

where $t$ is the set of words in generated text and $K_d$ is the domain
keyword set. Higher DKD means the model uses more domain-specific vocabulary.

### N-gram Diversity (NGD)
$$\text{NGD}_n(t) = \frac{|\text{unique n-grams in } t|}{|\text{total n-grams in } t|}$$

Measures text diversity. $n=3$ (trigrams). Higher = more diverse, less repetitive.

### Coherence Score
$$\text{COH}(t) = \max\left(0, 1 - \frac{|\bar{l}_s - 15|}{30}\right)$$

where $\bar{l}_s$ is the average sentence length in words. Peak at 15 words/sentence.

### Cross-Perplexity (XPPL)
$$\text{XPPL}_d(t) = \exp\left(-\frac{1}{|t|}\sum_{j=1}^{|t|} \log p_{f_\theta + \Delta W_d}(t_j | t_{<j})\right)$$

PPL of generated text under the domain adapter. Lower = the domain adapter
"agrees" with the generated text more (it's more domain-appropriate).

### Composite Quality Score
$$Q = 0.4 \cdot \text{DKD} + 0.2 \cdot \text{NGD} + 0.2 \cdot \text{COH} + 0.2 \cdot \text{XPPL}_\text{norm}$$

where $\text{XPPL}_\text{norm} = \max(0, 1 - \text{XPPL}_\text{config} / \text{XPPL}_\text{base})$.

## Kill Criteria

- **K1:** Routed composite score $<$ base composite score on $\geq 3/5$ domains
- **K2:** No domain has $>5\%$ relative improvement (decorative composition)
- **K3:** $>50\%$ of generated texts are incoherent (repetitive, < 5 words, etc.)

## Assumptions

1. Oracle routing is a valid proxy for learned routing (justified by 99.9% accuracy)
2. Domain keyword lists are representative of domain expertise
3. Automated metrics capture meaningful quality differences
4. Temperature 0.7 with top-p 0.9 produces representative samples
5. 10 prompts per domain is sufficient for directional signal (not statistical power)

## Computational Cost

- Base generation: ~2.5 min (10 prompts x 5 domains x 128 tokens)
- Uniform generation: ~3 min (slightly slower due to 5x LoRA computation)
- Routed generation: ~3 min (same as uniform, but with sparse routing)
- Cross-PPL: ~5 min (5 domain models x 50 generated texts each, forward only)
- Total: ~15-20 min

## Memory Budget

| Component | Memory |
|-----------|--------|
| Base model (unpacked bf16) | ~4.8 GB |
| 5 adapter A matrices | ~0.15 GB |
| 5 adapter B matrices | ~0.15 GB |
| KV cache | ~0.1 GB |
| **Total** | **~5.2 GB** |

Well within M5 Pro 48GB budget.
