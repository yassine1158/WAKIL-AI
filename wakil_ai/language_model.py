"""Character-level language models: predict the next character from the previous ones.

BigramCounts  - counts which character follows which (no neural network, no training)
CharMLP       - neural network with character embeddings (Bengio et al., 2003):

    context ids X (B, T) --embedding C--> E (B, T, d) --flatten--> (B, T*d)
        --> Z1 = .@W1 + b1 --> A1 = ReLU(Z1) --> logits = A1@W2 + b2 --> softmax
"""
import json
import time

import numpy as np

from .activations import relu, relu_derivative, softmax
from .text_data import get_batch


def cross_entropy_from_ids(probs, targets, eps=1e-12):
    """Same as cross-entropy with one-hot labels, but takes the target ids directly."""
    return float(-np.mean(np.log(probs[np.arange(len(targets)), targets] + eps)))


def sample_next(probs, rng):
    """Picks a character id at random, following the predicted probabilities."""
    return int(rng.choice(len(probs), p=probs / probs.sum()))


class BigramCounts:
    """P(next char | current char) estimated by counting pairs in the training text."""

    def __init__(self, vocab_size, smoothing=1.0):
        self.vocab_size = vocab_size
        self.smoothing = smoothing      # +1 to every count, so no pair has probability 0
        self.probs = np.full((vocab_size, vocab_size), 1.0 / vocab_size)

    def fit(self, ids):
        counts = np.zeros((self.vocab_size, self.vocab_size))
        np.add.at(counts, (ids[:-1], ids[1:]), 1)
        counts += self.smoothing
        self.probs = counts / counts.sum(axis=1, keepdims=True)
        return self

    def loss(self, ids):
        return float(-np.mean(np.log(self.probs[ids[:-1], ids[1:]])))

    def generate(self, prompt_ids, length, rng):
        ids = list(prompt_ids)
        for _ in range(length):
            ids.append(sample_next(self.probs[ids[-1]], rng))
        return ids


class CharMLP:

    def __init__(self, vocab_size, context=8, emb_dim=16, hidden=256, seed=42, dtype=np.float32):
        self.config = {"vocab_size": int(vocab_size), "context": int(context),
                       "emb_dim": int(emb_dim), "hidden": int(hidden)}
        rng = np.random.default_rng(seed)
        fan_in = context * emb_dim
        self.params = {
            "C": rng.standard_normal((vocab_size, emb_dim)).astype(dtype),              # embeddings
            "W1": (rng.standard_normal((fan_in, hidden)) * np.sqrt(2.0 / fan_in)).astype(dtype),
            "b1": np.zeros(hidden, dtype=dtype),
            # small output weights: the untrained model predicts ~uniform probabilities
            "W2": (rng.standard_normal((hidden, vocab_size)) * 0.1 * np.sqrt(1.0 / hidden)).astype(dtype),
            "b2": np.zeros(vocab_size, dtype=dtype),
        }

    @property
    def context(self):
        return self.config["context"]

    def num_parameters(self):
        return int(sum(p.size for p in self.params.values()))

    def forward(self, X):
        """X: (B, T) character ids. Returns (probabilities (B, V), cache)."""
        p = self.params
        E = p["C"][X]                          # (B, T, d): look up one vector per character
        A0 = E.reshape(len(X), -1)             # (B, T*d): concatenate the T vectors
        Z1 = A0 @ p["W1"] + p["b1"]
        A1 = relu(Z1)
        logits = A1 @ p["W2"] + p["b2"]
        probs = softmax(logits)
        return probs, {"X": X, "A0": A0, "Z1": Z1, "A1": A1, "probs": probs, "logits": logits}

    def backward(self, cache, targets):
        """Gradients of the mean cross-entropy. targets: (B,) ids of the true next characters."""
        X, A0, Z1, A1, probs = cache["X"], cache["A0"], cache["Z1"], cache["A1"], cache["probs"]
        p = self.params
        m = len(targets)

        dlogits = probs.copy()
        dlogits[np.arange(m), targets] -= 1    # = P - Y (one-hot), without building Y
        dlogits /= m
        dW2 = A1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dZ1 = (dlogits @ p["W2"].T) * relu_derivative(Z1)
        dW1 = A0.T @ dZ1
        db1 = dZ1.sum(axis=0)
        dE = (dZ1 @ p["W1"].T).reshape(X.shape + (-1,))   # (B, T, d)
        # Each embedding row receives the gradient of every position where its
        # character appeared. np.add.at adds correctly even with repeated ids.
        dC = np.zeros_like(p["C"])
        np.add.at(dC, X, dE)
        return {"C": dC, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update(self, grads, learning_rate):
        for name in self.params:
            self.params[name] -= learning_rate * grads[name]

    def loss(self, X, targets):
        return cross_entropy_from_ids(self.forward(X)[0], targets)

    def generate(self, prompt_ids, length, rng, temperature=1.0, pad_id=0):
        """Writes `length` new characters after the prompt, one at a time.

        temperature < 1: safer, more repetitive text. > 1: more surprising, more mistakes.
        """
        ids = [pad_id] * self.context + list(prompt_ids)
        for _ in range(length):
            window = np.array([ids[-self.context:]])
            logits = self.forward(window)[1]["logits"][0] / temperature
            ids.append(sample_next(softmax(logits[None, :])[0], rng))
        return ids[self.context:]

    def save(self, path, tokenizer, metadata=None):
        np.savez_compressed(path, config=np.array(json.dumps(self.config)),
                            chars=np.array(tokenizer.to_json()),
                            metadata=np.array(json.dumps(metadata or {})), **self.params)

    @classmethod
    def load(cls, path):
        """Returns (model, tokenizer, metadata)."""
        from .tokenizer import CharTokenizer
        with np.load(path) as data:
            model = cls(**json.loads(str(data["config"])))
            model.params = {name: data[name] for name in model.params}
            tokenizer = CharTokenizer.from_json(str(data["chars"]))
            metadata = json.loads(str(data["metadata"]))
        return model, tokenizer, metadata


def estimate_loss(model, ids, batch_size=1000, batches=10, seed=0):
    """Average loss on random windows of `ids` (fixed seed: comparable between calls)."""
    rng = np.random.default_rng(seed)
    return float(np.mean([model.loss(*get_batch(ids, model.context, batch_size, rng)) for _ in range(batches)]))


def train_language_model(model, train_ids, val_ids, steps=20000, batch_size=128, learning_rate=0.1,
                         decay_at=0.75, eval_every=2000, seed=42, verbose=True):
    """Mini-batch gradient descent. The learning rate is divided by 10 after
    `decay_at` of the steps (big steps first, then fine-tuning)."""
    rng = np.random.default_rng(seed)
    history = {"step": [], "train_loss": [], "val_loss": [], "batch_loss": []}
    start = time.time()
    for step in range(1, steps + 1):
        lr = learning_rate if step <= decay_at * steps else learning_rate / 10
        X, Y = get_batch(train_ids, model.context, batch_size, rng)
        probs, cache = model.forward(X)
        history["batch_loss"].append(cross_entropy_from_ids(probs, Y))
        model.update(model.backward(cache, Y), lr)

        if step % eval_every == 0 or step == steps:
            train_loss, val_loss = estimate_loss(model, train_ids), estimate_loss(model, val_ids)
            for key, value in [("step", step), ("train_loss", train_loss), ("val_loss", val_loss)]:
                history[key].append(value)
            if verbose:
                print(f"step {step:6d}/{steps} | lr {lr:.3f} | train loss {train_loss:.4f} "
                      f"| val loss {val_loss:.4f} | {time.time() - start:.0f}s")
    return history
