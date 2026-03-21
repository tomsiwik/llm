# Current Direction: exp_bitnet_scale_n25

## Goal
Test whether ternary LoRA composition on BitNet-2B-4T scales from N=15 (domains only) to N=25 (15 domains + 4 existing capabilities + 6 new capabilities) without degradation.

## Kill Criteria
- K1: composition ratio N=25 > 5x (approaching catastrophe)
- K2: cross-type cosine (capability-domain) > 0.01

## Approach
1. Reuse all 15 trained domain adapters from bitnet_scale_n15
2. Reuse 4 trained capability adapters from capability_expert_taxonomy (reasoning, instruction, conciseness, safety)
3. Train 6 new capability adapters: multilingual, coding-style, summarization, debate, translation, formal-writing
4. Compose all 25 together with 1/N scaling
5. Measure composition ratio, cross-type cosines, per-type degradation

## Key Data Sources (HuggingFace, $0)
- multilingual: Helsinki-NLP/tatoeba_mt (de-en pairs)
- coding-style: bigcode/the-stack-smol (docstring-heavy Python)
- summarization: EdinburghNLP/xsum (summaries)
- debate: argilla/distilabel-capybara-dpo-7k-binarized (argument-style)
- translation: Helsinki-NLP/opus_books (en-fr parallel)
- formal-writing: ccdv/arxiv-summarization (academic writing)

## Expected Runtime
~60-90 min (6 new adapters * ~8 min each + eval overhead)

## Scale
Micro only. Apple Silicon, MLX, $0.
