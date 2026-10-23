"""Character-level tokenizer: every distinct character gets an integer id."""
import json

import numpy as np


class CharTokenizer:

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # string -> id

    @classmethod
    def from_text(cls, text):
        """Builds the vocabulary from all distinct characters of a text, sorted."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """'hi' -> array([46, 47])"""
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Characters not in the vocabulary: {unknown}")
        return np.array([self.stoi[ch] for ch in text], dtype=np.int64)

    def decode(self, ids):
        """array([46, 47]) -> 'hi'"""
        return "".join(self.chars[int(i)] for i in ids)

    def to_json(self):
        return json.dumps(self.chars)

    @classmethod
    def from_json(cls, data):
        return cls(json.loads(data))
