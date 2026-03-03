"""Domain data loading from bigcode/the-stack-dedup for CL benchmark.

Each domain is a programming language. Data is tokenized and chunked into
fixed-length sequences for next-token prediction training.
"""

import torch
from torch.utils.data import Dataset

DOMAINS = ["python", "javascript", "rust", "sql", "cpp"]

# Map our domain names to The Stack's language identifiers
_LANG_MAP = {
    "python": "python",
    "javascript": "javascript",
    "rust": "rust",
    "sql": "sql",
    "cpp": "cpp",
}


class TokenizedDataset(Dataset):
    """Pre-tokenized fixed-length sequences for causal LM training."""

    def __init__(self, sequences):
        self.sequences = sequences  # list of (seq_len,) tensors

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def load_domain(lang, tokenizer, n_train=2000, n_eval=200, seq_len=512, seed=42):
    """Load and tokenize a coding language domain from The Stack.

    Args:
        lang: one of DOMAINS (python, javascript, rust, sql, cpp).
        tokenizer: HuggingFace tokenizer.
        n_train: number of training sequences to produce.
        n_eval: number of eval sequences to produce.
        seq_len: token length per sequence.
        seed: random seed for reproducible splits.

    Returns:
        (train_dataset, eval_dataset) as TokenizedDataset instances.
    """
    from datasets import load_dataset
    import numpy as np

    stack_lang = _LANG_MAP[lang]

    # Stream from The Stack to avoid downloading entire dataset
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{stack_lang.lower()}",
        split="train",
        streaming=True,
    )

    # Collect enough text content
    target_n = n_train + n_eval
    texts = []
    for example in ds:
        content = example.get("content", "")
        if len(content.strip()) > 200:  # skip trivially short files
            texts.append(content)
        if len(texts) >= target_n * 2:  # collect 2x to have margin after tokenization
            break

    if len(texts) < 100:
        raise RuntimeError(
            f"Only found {len(texts)} files for {lang}. "
            f"Check dataset access and language filter."
        )

    # Deterministic shuffle and split
    rng = np.random.RandomState(seed)
    rng.shuffle(texts)

    # Tokenize all texts into one long token stream, then chunk
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        # Stop once we have enough tokens
        if len(all_tokens) >= (n_train + n_eval) * seq_len * 2:
            break

    # Chunk into fixed-length sequences
    sequences = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        seq = all_tokens[i:i + seq_len]
        sequences.append(torch.tensor(seq, dtype=torch.long))

    if len(sequences) < n_train + n_eval:
        print(f"  Warning: only {len(sequences)} sequences for {lang} "
              f"(requested {n_train + n_eval})")

    # Split into eval (first) and train (rest)
    eval_seqs = sequences[:n_eval]
    train_seqs = sequences[n_eval:n_eval + n_train]

    return TokenizedDataset(train_seqs), TokenizedDataset(eval_seqs)
