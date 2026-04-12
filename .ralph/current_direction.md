# Current Direction: P9.B2 TT-LoRA MoE Router

## Status
STARTING -- exp_p9_ttlora_moe_router

## What We're Testing
Can a linear router on base model hidden states correctly select among 5 TT-LoRA
domain experts (math/code/medical/legal/finance)?

## Design
- 5 MMLU domain groups: math (5 subjects), code (4), medical (6), legal (3), finance (4)
- TT-LoRA r=6 on v_proj, 500 steps per domain adapter (~44 min each)
- Linear router: 2560 -> 5 (12,805 params) trained on base model hidden states
- Evaluation: logit-based MCQ accuracy (no generation needed)

## Kill Criteria
- K1360: Router expert selection accuracy >= 90% on 5 domains
- K1361: Routed TT-LoRA MoE outperforms single best TT-LoRA by >= 5pp avg
- K1362: Total system size (5 experts + router) < 2 MB

## Predictions (MATH.md)
- Router accuracy >= 95% (vocabulary separation in 2560-d hidden space)
- MoE advantage ~17.5pp (routing premium with alpha >= 0.9)
- Total size ~652 KB (5 * 154 KB + 25 KB router)

## Prior
- Finding #516: TT-LoRA 84.4% quality at 12.4x compression (154 KB adapter)
- arXiv:2504.21190: TT-LoRA MoE reports 99-100% routing accuracy with <=6 experts
