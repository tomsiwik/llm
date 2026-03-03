"""API-based perplexity evaluation via Together AI (logprobs).

For models too large to run locally (MiniMax, DeepSeek-V3, Mixtral).
Requires TOGETHER_API_KEY in .env or environment.
"""

import math
import os
import time


def api_available() -> bool:
    """Check if Together AI API is configured."""
    return bool(os.environ.get("TOGETHER_API_KEY"))


def _get_client():
    """Get Together AI client."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TOGETHER_API_KEY not set. Add it to .env:\n"
            "  echo 'TOGETHER_API_KEY=your-key' >> .env\n"
            "  Get one at: https://api.together.xyz/settings/api-keys"
        )
    from openai import OpenAI

    return OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key)


def compute_perplexity_api(
    together_model_id: str,
    texts: list[str],
    max_tokens_per_text: int = 512,
) -> tuple[float, float]:
    """Compute perplexity via Together AI logprobs.

    Strategy: send each text as a completion prompt with echo=True + logprobs,
    collect token-level log probabilities, compute PPL = exp(-mean(logprobs)).

    Returns (perplexity, tokens_per_sec).
    """
    client = _get_client()

    all_logprobs = []
    total_tokens = 0
    t0 = time.time()

    for i, text in enumerate(texts):
        # Truncate text to limit cost
        truncated = text[:max_tokens_per_text * 4]  # rough char estimate

        try:
            response = client.chat.completions.create(
                model=together_model_id,
                messages=[{"role": "user", "content": truncated}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=1,
                temperature=0.0,
                echo=True,
            )

            # Extract logprobs from response
            choice = response.choices[0]
            if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
                for token_info in choice.logprobs.content:
                    if token_info.logprob is not None:
                        all_logprobs.append(token_info.logprob)
                        total_tokens += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1}/{len(texts)}] {total_tokens} tokens, "
                      f"{total_tokens/elapsed:.0f} tok/s")

        except Exception as e:
            print(f"    Warning: text {i} failed: {e}")
            continue

    elapsed = time.time() - t0

    if not all_logprobs:
        return float("inf"), 0.0

    # PPL = exp(-mean(log_probs))
    mean_logprob = sum(all_logprobs) / len(all_logprobs)
    ppl = math.exp(-mean_logprob)
    tps = total_tokens / elapsed if elapsed > 0 else 0.0

    return ppl, tps


def compute_perplexity_api_completions(
    together_model_id: str,
    texts: list[str],
    max_tokens_per_text: int = 512,
) -> tuple[float, float]:
    """Fallback: use completions API (non-chat) with echo for prompt logprobs.

    Some models may work better with the completions endpoint.
    """
    client = _get_client()

    all_logprobs = []
    total_tokens = 0
    t0 = time.time()

    for i, text in enumerate(texts):
        truncated = text[:max_tokens_per_text * 4]

        try:
            response = client.completions.create(
                model=together_model_id,
                prompt=truncated,
                max_tokens=1,
                logprobs=1,
                echo=True,
                temperature=0.0,
            )

            choice = response.choices[0]
            if hasattr(choice, "logprobs") and choice.logprobs:
                token_lps = choice.logprobs.token_logprobs or []
                for lp in token_lps:
                    if lp is not None:
                        all_logprobs.append(lp)
                        total_tokens += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1}/{len(texts)}] {total_tokens} tokens, "
                      f"{total_tokens/elapsed:.0f} tok/s")

        except Exception as e:
            print(f"    Warning: text {i} failed: {e}")
            continue

    elapsed = time.time() - t0

    if not all_logprobs:
        return float("inf"), 0.0

    mean_logprob = sum(all_logprobs) / len(all_logprobs)
    ppl = math.exp(-mean_logprob)
    tps = total_tokens / elapsed if elapsed > 0 else 0.0

    return ppl, tps
