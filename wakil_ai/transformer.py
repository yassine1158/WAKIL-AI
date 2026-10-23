"""A small GPT-style Transformer, written with NumPy only (forward AND backward).

    token ids (B, T)
      -> token embedding + position embedding                  x: (B, T, d)
      -> N x Block:  x = x + Attention(LayerNorm(x))           (tokens look at earlier tokens)
                     x = x + MLP(LayerNorm(x))                 (each token thinks on its own)
      -> LayerNorm -> Linear(d, vocab)                         logits: (B, T, vocab)

The model predicts the next token at EVERY position at once: position t only
sees positions 0..t (causal mask), so T predictions come from one forward pass.

Every layer has forward(x) and backward(dout). backward() returns the gradient
for its input and stores the gradients of its own parameters in self.grads.
"""
import json

import numpy as np

from .activations import softmax


class Linear:
    def __init__(self, n_in, n_out, rng, dtype, std=0.02):
        self.params = {"W": (rng.standard_normal((n_in, n_out)) * std).astype(dtype),
                       "b": np.zeros(n_out, dtype=dtype)}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        n_in, n_out = self.params["W"].shape
        self.grads = {"W": self.x.reshape(-1, n_in).T @ dout.reshape(-1, n_out),
                      "b": dout.reshape(-1, n_out).sum(axis=0)}
        return dout @ self.params["W"].T


class LayerNorm:
    """Normalizes each token vector to mean 0 / variance 1, then scales and shifts it.
    Keeps the numbers in a stable range as they flow through many layers."""

    def __init__(self, d, dtype, eps=1e-5):
        self.eps = eps
        self.params = {"gamma": np.ones(d, dtype=dtype), "beta": np.zeros(d, dtype=dtype)}

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(x.var(axis=-1, keepdims=True) + self.eps)
        self.xhat = (x - mean) * self.rstd
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, dout):
        d = dout.shape[-1]
        self.grads = {"gamma": (dout * self.xhat).reshape(-1, d).sum(axis=0),
                      "beta": dout.reshape(-1, d).sum(axis=0)}
        dxhat = dout * self.params["gamma"]
        return self.rstd / d * (d * dxhat - dxhat.sum(axis=-1, keepdims=True)
                                - self.xhat * (dxhat * self.xhat).sum(axis=-1, keepdims=True))


class CausalSelfAttention:
    """Multi-head self-attention with a causal mask.

    Each token makes a query ("what am I looking for?"), a key ("what do I contain?")
    and a value ("what do I give if chosen?"). Attention weights = softmax(Q K^T / sqrt(hd)):
    how much each earlier token matters for this one. Output = weighted sum of values.
    """

    def __init__(self, d, n_heads, rng, dtype):
        assert d % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.qkv = Linear(d, 3 * d, rng, dtype)
        self.proj = Linear(d, d, rng, dtype)

    def forward(self, x):
        B, T, d = x.shape
        H, hd = self.n_heads, self.head_dim
        qkv = self.qkv.forward(x).reshape(B, T, 3, H, hd).transpose(2, 0, 3, 1, 4)   # (3, B, H, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)                        # (B, H, T, T)
        future = np.triu(np.ones((T, T), dtype=bool), k=1)                           # True above the diagonal
        scores = np.where(future, -1e9, scores)                                      # can't look at the future
        att = softmax(scores.reshape(-1, T)).reshape(B, H, T, T)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)                     # concatenate the heads
        self.cache = (q, k, v, att)
        return self.proj.forward(out)

    def backward(self, dout):
        q, k, v, att = self.cache
        B, H, T, hd = q.shape
        dout = self.proj.backward(dout).reshape(B, T, H, hd).transpose(0, 2, 1, 3)   # (B, H, T, hd)
        datt = dout @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dout
        dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))           # softmax backward
        dscores /= np.sqrt(hd)                                                       # masked entries: att = 0 -> 0
        dq = dscores @ k
        dk = dscores.transpose(0, 1, 3, 2) @ q
        dqkv = np.stack([dq, dk, dv]).transpose(1, 3, 0, 2, 4).reshape(B, T, 3 * H * hd)
        return self.qkv.backward(dqkv)


class MLP:
    def __init__(self, d, rng, dtype):
        self.fc = Linear(d, 4 * d, rng, dtype)
        self.out = Linear(4 * d, d, rng, dtype)

    def forward(self, x):
        self.h = self.fc.forward(x)
        return self.out.forward(np.maximum(0, self.h))

    def backward(self, dout):
        return self.fc.backward(self.out.backward(dout) * (self.h > 0))


class Block:
    """Pre-norm Transformer block with residual connections (x + ...): the
    gradient can flow straight through the '+', which makes deep stacks trainable."""

    def __init__(self, d, n_heads, rng, dtype):
        self.ln1, self.attn = LayerNorm(d, dtype), CausalSelfAttention(d, n_heads, rng, dtype)
        self.ln2, self.mlp = LayerNorm(d, dtype), MLP(d, rng, dtype)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.mlp.forward(self.ln2.forward(x))

    def backward(self, dout):
        dx = dout + self.ln2.backward(self.mlp.backward(dout))
        return dx + self.ln1.backward(self.attn.backward(dx))


class TinyGPT:

    def __init__(self, vocab_size, block_size=32, d_model=64, n_heads=4, n_layers=2, seed=42, dtype=np.float32):
        self.config = {"vocab_size": int(vocab_size), "block_size": int(block_size), "d_model": int(d_model),
                       "n_heads": int(n_heads), "n_layers": int(n_layers)}
        rng = np.random.default_rng(seed)
        self.params = {"tok_emb": (rng.standard_normal((vocab_size, d_model)) * 0.02).astype(dtype),
                       "pos_emb": (rng.standard_normal((block_size, d_model)) * 0.02).astype(dtype)}
        self.blocks = [Block(d_model, n_heads, rng, dtype) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model, dtype)
        self.head = Linear(d_model, vocab_size, rng, dtype)

    @property
    def context(self):
        return self.config["block_size"]

    def layers(self):
        """(name, layer) for every layer that owns parameters."""
        named = [("model", self), ("ln_f", self.ln_f), ("head", self.head)]
        for i, block in enumerate(self.blocks):
            named += [(f"blocks.{i}.ln1", block.ln1), (f"blocks.{i}.attn.qkv", block.attn.qkv),
                      (f"blocks.{i}.attn.proj", block.attn.proj), (f"blocks.{i}.ln2", block.ln2),
                      (f"blocks.{i}.mlp.fc", block.mlp.fc), (f"blocks.{i}.mlp.out", block.mlp.out)]
        return named

    def named_parameters(self):
        """Yields (full_name, layer, key): the parameter is layer.params[key]."""
        for prefix, layer in self.layers():
            for key in layer.params:
                yield f"{prefix}.{key}", layer, key

    def num_parameters(self):
        return int(sum(layer.params[key].size for _, layer, key in self.named_parameters()))

    def forward(self, idx):
        """idx: (B, T) token ids, T <= block_size. Returns logits (B, T, vocab)."""
        T = idx.shape[1]
        self.idx = idx
        x = self.params["tok_emb"][idx] + self.params["pos_emb"][:T]
        for block in self.blocks:
            x = block.forward(x)
        return self.head.forward(self.ln_f.forward(x))

    def loss(self, idx, targets):
        """Mean cross-entropy over all B*T positions. targets: (B, T) next-token ids."""
        logits = self.forward(idx)
        V = logits.shape[-1]
        self.probs = softmax(logits.reshape(-1, V))
        self.targets = targets.reshape(-1)
        return float(-np.mean(np.log(self.probs[np.arange(len(self.targets)), self.targets] + 1e-12)))

    def backward(self):
        """Backpropagation through the whole model (call after loss())."""
        n = len(self.targets)
        dlogits = self.probs.copy()
        dlogits[np.arange(n), self.targets] -= 1
        dlogits /= n
        B, T = self.idx.shape
        dx = self.ln_f.backward(self.head.backward(dlogits.reshape(B, T, -1)))
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        dtok = np.zeros_like(self.params["tok_emb"])
        np.add.at(dtok, self.idx, dx)
        dpos = np.zeros_like(self.params["pos_emb"])
        dpos[:T] = dx.sum(axis=0)
        self.grads = {"tok_emb": dtok, "pos_emb": dpos}

    def generate(self, prompt_ids, length, rng, temperature=1.0):
        ids = list(prompt_ids)
        for _ in range(length):
            window = np.array([ids[-self.context:]])
            logits = self.forward(window)[0, -1] / temperature
            probs = softmax(logits[None, :])[0]
            ids.append(int(rng.choice(len(probs), p=probs / probs.sum())))
        return ids

    def save(self, path, tokenizer, metadata=None):
        arrays = {name: layer.params[key] for name, layer, key in self.named_parameters()}
        np.savez_compressed(path, config=np.array(json.dumps(self.config)), chars=np.array(tokenizer.to_json()),
                            metadata=np.array(json.dumps(metadata or {})), **arrays)

    @classmethod
    def load(cls, path):
        """Returns (model, tokenizer, metadata)."""
        from .tokenizer import CharTokenizer
        with np.load(path) as data:
            model = cls(**json.loads(str(data["config"])))
            for name, layer, key in model.named_parameters():
                layer.params[key] = data[name]
            tokenizer = CharTokenizer.from_json(str(data["chars"]))
            metadata = json.loads(str(data["metadata"]))
        return model, tokenizer, metadata


class Adam:
    """Adam optimizer: each parameter gets its own adaptive step size.

    m = running average of the gradient (momentum: keeps going in a consistent direction)
    v = running average of the squared gradient (big, noisy gradients -> smaller steps)
    update = lr * m_hat / (sqrt(v_hat) + eps), with bias correction for the first steps.
    """

    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        self.model, self.lr, self.beta1, self.beta2, self.eps = model, lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for name, layer, key in self.model.named_parameters():
            g = layer.grads[key]
            if name not in self.m:
                self.m[name], self.v[name] = np.zeros_like(g), np.zeros_like(g)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            layer.params[key] -= (lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(layer.params[key].dtype)


def estimate_loss(model, ids, batch_size=64, batches=8, seed=0):
    """Average loss on random windows of `ids` (fixed seed: comparable between calls)."""
    from .text_data import get_sequence_batch
    rng = np.random.default_rng(seed)
    return float(np.mean([model.loss(*get_sequence_batch(ids, model.context, batch_size, rng))
                          for _ in range(batches)]))


def train_transformer(model, train_ids, val_ids, steps=5000, batch_size=32, learning_rate=3e-3,
                      decay_at=0.75, eval_every=500, seed=42, verbose=True):
    """Trains with Adam; the learning rate is divided by 10 after `decay_at` of the steps."""
    import time
    from .text_data import get_sequence_batch
    rng = np.random.default_rng(seed)
    optimizer = Adam(model, lr=learning_rate)
    history = {"step": [], "train_loss": [], "val_loss": [], "batch_loss": []}
    start = time.time()
    for step in range(1, steps + 1):
        lr = learning_rate if step <= decay_at * steps else learning_rate / 10
        X, Y = get_sequence_batch(train_ids, model.context, batch_size, rng)
        history["batch_loss"].append(model.loss(X, Y))   # forward
        model.backward()                                  # backward
        optimizer.step(lr)                                # update
        if step % eval_every == 0 or step == steps:
            train_loss, val_loss = estimate_loss(model, train_ids), estimate_loss(model, val_ids)
            for key, value in [("step", step), ("train_loss", train_loss), ("val_loss", val_loss)]:
                history[key].append(value)
            if verbose:
                print(f"step {step:5d}/{steps} | lr {lr:.4f} | train loss {train_loss:.4f} "
                      f"| val loss {val_loss:.4f} | {time.time() - start:.0f}s")
    return history
