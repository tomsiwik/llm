# MoE / Continual Learning — Annotated Algorithm Extracts

Reference implementations from 2022-2024 papers, kept as original PyTorch.
Each file contains `# TRIBE NOTE:` comments mapping innovations to our lifecycle operations.

**Purpose**: Study published approaches, align naming, identify adoptable algorithms.

| # | File | Paper | Year | Maps to (Tribe) |
|---|------|-------|------|-----------------|
| 01 | [switch_routing](01_switch_routing.py) | [Switch Transformer](https://arxiv.org/abs/2101.03961) (Fedus et al.) | 2022 | `route_by_loss()` — learned vs oracle routing |
| 02 | [expert_choice_routing](02_expert_choice_routing.py) | [Expert Choice](https://arxiv.org/abs/2202.09368) (Zhou et al.) | 2022 | `health_check()` — proactive vs reactive load balance |
| 03 | [scattermoe_parallel](03_scattermoe_parallel.py) | [ScatterMoE](https://arxiv.org/abs/2403.08245) (Tan) | 2024 | `batch_route_by_loss()` — parallel vs sequential dispatch |
| 04 | [redo_dormant_reinit](04_redo_dormant_reinit.py) | [ReDo](https://arxiv.org/abs/2310.19731) (Klein et al.) | 2024 | `recycle()` — neuron vs expert reinit |
| 05 | [hc_smoe_merging](05_hc_smoe_merging.py) | [HC-SMoE](https://arxiv.org/abs/2405.15966) (wazenmai) | 2024 | `bond()` — weight similarity vs domain overlap |
| 06 | [mc_smoe_grouping](06_mc_smoe_grouping.py) | [MC-SMoE](https://arxiv.org/abs/2405.17089) (Li et al.) | 2024 | `measure_overlap()` — router correlation vs loss similarity |
| 07 | [reap_saliency](07_reap_saliency.py) | [REAP](https://arxiv.org/abs/2403.16959) (Muzio et al.) | 2024 | `unique_knowledge()` — saliency-based pruning |
| 08 | [dynmoe_grow_shrink](08_dynmoe_grow_shrink.py) | [DynMoE](https://arxiv.org/abs/2405.14297) (Guo et al.) | 2024 | `bond()`/`recycle()` — per-input vs per-generation |
| 09 | [xlora_gating](09_xlora_gating.py) | [X-LoRA](https://arxiv.org/abs/2402.07148) (Buehler) | 2024 | LoRA-as-expert — input-dependent adapter gating |
| 10 | [mole_lora_mixture](10_mole_lora_mixture.py) | [MoLE](https://arxiv.org/abs/2404.07413) (Wu et al.) | 2024 | LoRA-as-expert — learned mixing coefficients |
| 11 | [drop_upcycling](11_drop_upcycling.py) | [Drop-Upcycling](https://arxiv.org/abs/2406.04835) (Taishi) | 2024 | `recycle()` — dropout diversification vs blend+noise |
| 12 | [moe_prompt_cl](12_moe_prompt_cl.py) | [MoE-PromptCL](https://arxiv.org/abs/2405.01032) (Minh) | 2024 | Generations — MoE for continual learning |

## Themes

### Routing (01-03)
How tokens reach experts. Switch = learned top-1, Expert Choice = experts pick tokens, ScatterMoE = efficient parallel dispatch.

### Expert Lifecycle (04-08)
Managing expert populations. ReDo = reinit dormant neurons, HC-SMoE/MC-SMoE = merge redundant experts, REAP = prune unimportant ones, DynMoE = grow/shrink count.

### LoRA as Expert (09-10)
Adapter-based MoE. X-LoRA = deep gating over adapters, MoLE = learned mixing coefficients.

### Diversification + CL (11-12)
Creating diverse experts and applying MoE to continual learning. Drop-Upcycling = diversify via dropout, MoE-PromptCL = task-aware routing.

## Usage

These are **read-only reference extracts** — not runnable. Study the algorithms and `# TRIBE NOTE:` annotations to inform our lifecycle design.

```bash
# Verify all files parse
for f in examples/*.py; do python -c "import ast; ast.parse(open('$f').read())"; done
```
