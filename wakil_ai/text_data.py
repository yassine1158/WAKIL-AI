"""Text dataset for the language models: download, split, and sample batches."""
import os
import urllib.request

import numpy as np

# The complete works of Shakespeare (public domain), ~1.1 million characters,
# as prepared by Andrej Karpathy for char-rnn.
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path=None, cache_dir="data"):
    """Returns a text. Default: Tiny Shakespeare, downloaded once and cached.

    path: use your own UTF-8 text file instead (French, Arabic, ...).
    """
    if path is None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "tinyshakespeare.txt")
        if not os.path.exists(path):
            print("Downloading Tiny Shakespeare (~1 MB)...")
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def split_ids(ids, val_fraction=0.1):
    """First 90% for training, last 10% for validation.

    The split is contiguous (not random): two random windows of the same text
    overlap, so a random split would leak training text into validation.
    """
    n = int(len(ids) * (1 - val_fraction))
    return ids[:n], ids[n:]


def get_batch(ids, context, batch_size, rng):
    """Random training examples: X = `context` characters, Y = the next one.

    Returns X with shape (batch_size, context) and Y with shape (batch_size,).
    """
    starts = rng.integers(0, len(ids) - context, size=batch_size)
    X = ids[starts[:, None] + np.arange(context)]
    Y = ids[starts + context]
    return X, Y


def get_sequence_batch(ids, context, batch_size, rng):
    """For Transformers: X = `context` characters, Y = the same window shifted by one.

    Y[:, t] is the character that follows X[:, :t+1], so one window gives
    `context` training examples at once. Shapes: (batch_size, context) for both.
    """
    starts = rng.integers(0, len(ids) - context, size=batch_size)
    positions = starts[:, None] + np.arange(context)
    return ids[positions], ids[positions + 1]
