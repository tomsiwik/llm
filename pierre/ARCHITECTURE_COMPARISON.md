# Pierre vs State-of-the-Art LLM Architectures

Based on Sebastian Raschka's LLM Architecture Gallery (51 models, April 2026).

## The Settled Base Recipe (universal — every modern LLM)

| Component | 2019 (GPT-2) | 2026 (all models) | Why it won |
|-----------|-------------|-------------------|------------|
| Position encoding | Absolute learned | **RoPE** | Relational encoding generalizes to unseen lengths |
| Activation | GELU | **SiLU/Swish** | Cheaper, empirically equivalent |
| Feed-forward | 2-layer dense | **SwiGLU** (3-layer gated) | Better expressivity at fewer parameters |
| Attention | MHA | **GQA** (or MLA at 100B+) | Directly reduces KV-cache, negligible quality loss |
| Normalization | LayerNorm | **RMSNorm** (+ QK-Norm) | Cheaper, one fewer reduction op per layer |
| Dropout | 0.1 | **None** | Single-epoch training = no overfitting |

**Raschka's meta-finding:** "All leading LLMs now share the same base architecture.
The real differentiation is in data and training algorithms, not architecture."

## The 6 Active Innovation Frontiers

### 1. Attention Efficiency — the current #1 frontier

| Technique | Who uses it | What it does |
|-----------|------------|-------------|
| **MLA** (Multi-head Latent Attention) | DeepSeek V3, Kimi K2/K2.5, Sarvam 105B, GLM-5 | Compresses KV into low-rank latent space. Ablations show it beats both MHA and GQA on quality. |
| **Hybrid linear attention** (DeltaNet + full attn, 3:1) | Qwen3-Next, Qwen3.5, Kimi Linear | 75% of layers use O(n) linear recurrence. Full attention layers handle content-based retrieval. |
| **Lightning Attention** (+ MLA) | Ling 2.5 (1T) | Simpler linear variant. 3.5x throughput vs Kimi K2 at same 1T scale. |
| **Sliding window** (alternating local/global) | GPT-OSS (128 tokens!), Gemma 3 (1024), MiMo-V2 | Dramatically reduces KV-cache. GPT-OSS proved 128 tokens is enough in alternating layers. |
| **Partial RoPE** | MiniMax-M2 | Rotation on first `rotary_dim` channels only. Enables length extrapolation. |
| **NoPE** (no position in some layers) | SmolLM3, Kimi Linear (MLA layers) | Causal mask provides ordering. Better length generalization (proven at 100M, unproven at scale). |

**Pierre:** Standard full attention from BitNet-2B base. No attention innovation. **This is our biggest architectural gap.**

### 2. Sparse Activation (MoE)

| Model | Total / Active | Experts | Key choice |
|-------|---------------|---------|------------|
| DeepSeek V3 | 671B / 37B | 256 / 9 (1 shared) | Fine-grained experts, loss-free routing |
| Qwen3 MoE | 235B / 22B | 256 / 8 (no shared) | Dropped shared expert ("no significant improvement") |
| GPT-OSS | 120B | 32 / 4 | Alternating dense + MoE layers |
| Llama 4 Maverick | 400B / 17B | 8 / 2 (no shared) | Fewer, larger experts |
| Grok 2.5 | 270B | 8 (shared = 2x FFN) | Shared expert as doubled dense FFN |
| GLM-4.5 | 355B | Shared, dense prefix | 3 dense layers before MoE begins |

**Pierre:** No MoE. Composable LoRA adapters are our equivalent — domain routing replaces expert routing, but at the model level not layer level. **Conceptually similar, structurally different.**

### 3. Quantization / Efficiency

| Technique | Who | Bits | Impact |
|-----------|-----|------|--------|
| BitNet b1.58 | Microsoft | 1.58 (ternary) | 10x memory reduction, integer addition replaces multiply |
| Standard quantization | Everyone (GPTQ/AWQ) | 4-8 bit | Post-training compression |
| **Pierre** | Us | **1.58 base + bf16 adapters** | **Only system with composable adapters on ternary base** |

### 4. Training Recipe > Architecture

| Model | Architecture | Insight |
|-------|-------------|---------|
| MiniMax M2.5 | Deliberately simple (plain GQA, no MoE tricks) | Beats complex architectures on coding via training recipe alone |
| Nanbeige 4.1 | Near-identical to Llama 3.2 | All gains from SFT + RL post-training |
| Kimi K2 | Identical to DeepSeek V3 | Innovation is Muon optimizer, not architecture |

**Pierre:** Our gains come from composition mathematics (Grassmannian orthogonality, NRE merging) + adapter training recipe (SFT). Architecture is borrowed. **This is valid — training recipe matters more than architecture.**

### 5. Hybrid Architectures (transformer + state-space)

| Model | Mix | Insight |
|-------|-----|---------|
| Nemotron 3 Nano (4B) | Mostly Mamba-2 + some attention | Most extreme hybrid. On-device target. |
| MiniMax M1→M2 | Tried linear attention, **abandoned it** | "Poor accuracy in reasoning and multi-turn tasks" |

**Pierre:** Pure transformer. The MiniMax M2 reversal validates that full attention is still needed for reasoning.

### 6. Novel Structural Choices

| Model | Innovation | Result |
|-------|-----------|--------|
| OLMo 2 | Post-norm (inside residual) + QK-Norm | Smoother loss curves, better training stability |
| Gemma 3 | 5:1 local:global attention ratio | Minimal quality impact, massive KV-cache savings |
| GPT-OSS | Learned attention sinks (per-head bias) | Stabilizes long-context without sink tokens |
| Tiny Aya | Parallel attention + MLP blocks | Reduced serial dependencies → throughput win |
| GLM-4.5 | 3-layer dense prefix before MoE | Stabilizes early feature extraction |
| Step 3.5 Flash | MTP-3 during inference | 3x throughput (100 vs 33 tok/s at 128k) |

## Novel Findings Per Model (what the field learned)

| Finding | Model | Why it matters |
|---------|-------|---------------|
| MLA beats GQA in ablations | DeepSeek V2/V3 | KV-cache compression without quality loss |
| DeltaNet replaces 75% of attention | Qwen3-Next | O(n) attention for most layers is viable |
| Linear attention fails for reasoning | MiniMax M1→M2 | Full attention still needed for multi-step logic |
| 128-token sliding window works | GPT-OSS | Extreme local windows sufficient in alternating layers |
| Shared expert: no consensus | DeepSeek yes, Qwen3 no, Qwen3-Next yes again | Depends on inference optimization constraints |
| Muon optimizer at 1T scale | Kimi K2 | First production use, exceptionally smooth training |
| Training recipe > architecture | MiniMax M2.5, Nanbeige | Simple architectures compete when training is strong |
| QK-Norm stabilizes training | OLMo 2, Gemma 3, Qwen3 | Becoming standard alongside RMSNorm |
| MTP during inference = 3x throughput | Step 3.5 Flash | Multi-token prediction useful beyond training |
| Early multimodal fusion wins | Kimi K2.5 | Vision tokens from pre-training start > bolt-on |
| Width > depth (marginally) | GPT-OSS, Gemma 2 ablation | 52.0 vs 50.8 at 9B scale |
| Parallel attn+MLP blocks | Tiny Aya | Throughput gain from reduced serial dependencies |
| Grassmannian orthogonal adapters | **Pierre** | Zero-interference composition on ternary base |

## Pierre's Ranking

### Position: **Unique niche, not on the main leaderboard**

Pierre doesn't compete with the models above — they're base model architectures at 7B-1T scale. Pierre is a **serving/composition layer** on top of a borrowed 2B ternary base.

The honest comparison:

| Dimension | State-of-the-art | Pierre | Gap |
|-----------|-----------------|--------|-----|
| **Base model** | Custom-trained 7B-1T | Borrowed BitNet-2B | We don't train base models |
| **Attention** | MLA, DeltaNet, sliding window | Standard (from base) | No innovation |
| **Scale** | 1T params, 256k context | 2B params, 256 context | 500x scale gap |
| **MoE/routing** | 256 experts, loss-free routing | 5 domain adapters, ridge regression | Conceptually parallel, structurally different |
| **Quantization** | 4-8 bit post-training | **Native 1.58-bit base** | **Pierre's only architectural edge** |
| **Composition** | Not addressed by any model | **Grassmannian + NRE + null-space** | **Pierre's unique contribution** |
| **Serving speed** | 100+ tok/s at 100B+ | 73 tok/s at 2B | Not comparable (different scales) |
| **Training recipe** | Trillions of tokens, custom optimizers | SFT adapters, 400 samples/domain | Different scope entirely |

### What Pierre contributes that NO model in the gallery has

1. **Mathematical non-interference guarantee for adapter composition** — Grassmannian cos < 0.001
2. **Runtime adapter hot-swap** — attach/detach in <1s, no base model modification
3. **Closed-form zero-training router** — ridge regression, 99.6% accuracy
4. **Norm-rescaled adapter merging** — proven equivalent to Fisher-Rao Karcher mean

### What Pierre should learn from the gallery

1. **MTP during inference** (Step 3.5 Flash) — could 3x our generation throughput
2. **Parallel attention+MLP blocks** (Tiny Aya) — reduce serial dependencies in adapter path
3. **Training recipe matters more than architecture** (MiniMax, Nanbeige) — invest in adapter training quality, not more architectural experiments
4. **MLA for KV-cache** (DeepSeek) — if/when we move to a larger ternary base
