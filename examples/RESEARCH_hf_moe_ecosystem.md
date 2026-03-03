# Hugging Face MoE Ecosystem — Research Notes

## 1. Native Transformers MoE Support

| Model | Total/Active Params | Experts | Routing |
|-------|-------------------|---------|---------|
| Switch Transformers | Up to 1.6T | Up to 2048 | Top-1 |
| Mixtral 8x7B/8x22B | 46.7B/12.9B | 8 | Top-2 noisy gating |
| DBRX | 132B/36B | 16 | Top-4 (fine-grained) |
| DeepSeek-V2 | 236B/21B | 160 routed + 2 shared | Top-6 |
| DeepSeek-V3 | 671B/37B | 256 routed + 1 shared | Top-8, bias-based (no aux loss) |
| Qwen2MoE | 14.3B/2.7B | 60 shared+routed | Top-4 |
| Llama 4 Scout | ~109B/17B | 16 | Top-1 + shared |
| Llama 4 Maverick | 400B/17B | 128 + 1 shared | Top-1 + shared |
| OLMoE-1B-7B | 7B/1B | 64 | Top-k (100% open) |

**Standard routing pattern**: Linear router → softmax logits → top-k → aux load-balancing loss.
**Exception**: DeepSeek uses bias-based routing without aux loss.

## 2. Key Libraries

### mergekit (arcee-ai)
- FrankenMoE: combine fine-tuned dense models into MoE
- Sparse upcycling: clone dense model N times, randomize gates, fine-tune
- Gate init modes: `hidden` (prompt-based), `cheap_embed`, `random`
- **Maps to tribe**: `bond()` is our dynamic version of their one-shot merge

### PEFT + LoRA-MoE (not yet native, third-party fills gap)
- **X-LoRA**: deep layer-wise token-level gating over LoRA adapters
- **MixLoRA**: LoRA experts in FFN block with sparse MoE routing
- **AdaMoLE**: adaptive mixture with dynamic gating
- **MoLA**: different expert counts per layer
- **Maps to tribe**: future "LoRA as knowledge entity" direction

## 3. Must-Study Papers

### Theory on MoE in Continual Learning (2024)
- **Key finding**: gating networks MUST be frozen after convergence in CL
- Ongoing router updates destabilize as new tasks arrive
- **Validates**: our `freeze` mechanism

### LEMoE (EMNLP 2024)
- Freeze prior experts, train only new expert + router
- KV anchor routing for consistency
- **Validates**: our freeze + reincarnate pattern

### MoE-Adapters4CL (CVPR 2024)
- Dynamic expert expansion with OOD detection
- DDAS routes OOD to frozen CLIP baseline
- **Inspires**: fallback expert for out-of-distribution inputs

### DeepSeek Aux-Loss-Free Routing
- Dynamic bias term steers load balance
- No quality trade-off from aux loss
- **Alternative to**: our reactive health_check rebalancing

### MoSE — Mixture of Slimmable Experts (2025)
- Each expert has nested variable widths
- Continuous accuracy-compute trade-offs
- **Beyond**: our binary recycle — could shrink instead of kill

### Drop-Upcycling (ICLR 2025)
- Partial reinit of cloned experts for specialization
- 5.9B active matches 13B dense with 1/4 FLOPs
- **Validates**: our reincarnation-with-memory approach

## 4. Routing Strategy Convergence

The field converges on:
1. **Top-k + aux loss** (Mixtral, DBRX, OLMoE) — classic
2. **Bias-based, no aux loss** (DeepSeek) — avoids quality trade-off
3. **Top-1 + shared expert** (Llama 4, DeepSeek) — minimal routing + always-on base
4. **Fine-grained MoE** (DBRX, DeepSeek) — more small experts, more combinations

### Shared Expert Pattern
Always-on expert + routed specialists. Maps to a "base expert" that handles common patterns while lifecycle manages specialists. Used by DeepSeek, Llama 4, Qwen.

## 5. Tribe Lifecycle Validation

| Our Operation | Published Analogue | Reference |
|--------------|-------------------|-----------|
| `recycle()` | Drop-Upcycling partial reinit | ICLR 2025 |
| `freeze` | LEMoE freeze-old-train-new | EMNLP 2024 |
| `bond()/merge` | mergekit FrankenMoE | arcee-ai |
| `health_check` | DeepSeek bias routing | DeepSeek-V3 |
| `measure_overlap` | REAP saliency scoring | Cerebras |
| `unique_knowledge` | MoE-Pruner router hints | 2024 |

## 6. Key Blog Posts

- [Mixture of Experts Explained](https://huggingface.co/blog/moe) — foundational
- [What is MoE 2.0?](https://huggingface.co/blog/Kseniase/moe2) — S'MoRE, inference opts
- [makeMoE from Scratch](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch) — tutorial
- [FrankenMoE with MergeKit](https://huggingface.co/blog/mlabonne/frankenmoe) — merging
- [Mixture of Tunable Experts](https://huggingface.co/blog/rbrt/mixture-of-tunable-experts) — expert suppression
