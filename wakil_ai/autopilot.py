"""Autopilot: Wakil AI keeps improving on its own, round after round, with no human input.

Digits: the model makes its OWN new training examples by transforming the images
it already has (rotation, shift, size, thicker strokes): the label of a rotated 7
is still 7, so no human is needed. This is data augmentation.

Text: the language model keeps training on its text. Next-character prediction is
self-supervised: the right answer is simply the next character of the text.

Safety: training continues in memory, but the model file is only replaced when the
new version is better on held-out data. A bad round can never overwrite a good model.
"""
import json
import os

import numpy as np

from .data import one_hot


def augment_digit(image, rng):
    """28x28 uint8 digit -> a new, different-looking version of the same digit (784 values in [0, 1])."""
    from PIL import Image
    im = Image.fromarray(np.asarray(image, dtype=np.uint8))
    im = im.rotate(rng.uniform(-45, 45), resample=Image.BILINEAR,
                   translate=(int(rng.integers(-3, 4)), int(rng.integers(-3, 4))))
    size = max(8, int(28 * rng.uniform(0.8, 1.15)))
    im = im.resize((size, size), Image.BILINEAR)
    canvas = Image.new("L", (28, 28))
    canvas.paste(im, ((28 - size) // 2, (28 - size) // 2))
    out = np.asarray(canvas, dtype=np.float32)
    if rng.random() < 0.5:                              # thicker strokes, like a brush or a marker
        out = np.maximum.reduce([out, np.roll(out, 1, 0), np.roll(out, -1, 0), np.roll(out, 1, 1), np.roll(out, -1, 1)])
    return out.reshape(784) / 255.0


class DigitAutopilot:

    def __init__(self, model, path, X_train, y_train, X_val, y_val, extra_X=None, extra_y=None,
                 max_drop=0.005, state=None, seed=0, metadata=None):
        """X_*: MNIST arrays (784 values in [0, 1]). extra_X/extra_y: optional corrections saved earlier.
        max_drop: largest accepted drop of normal accuracy compared with the first run."""
        self.model, self.path, self.metadata = model, path, dict(metadata or {})
        self.rng = np.random.default_rng(seed)
        self.images = (X_train.reshape(-1, 28, 28) * 255).astype(np.uint8)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.extra_X, self.extra_y = extra_X, extra_y
        # a fixed "hard" validation set: transformed validation digits, always the same
        hard_rng = np.random.default_rng(12345)
        val_images = (X_val.reshape(-1, 28, 28) * 255).astype(np.uint8)
        self.hard_X = np.stack([augment_digit(img, hard_rng) for img in val_images])
        self.state = state if state is not None else {}
        normal, hard = self.scores()
        self.state.setdefault("reference_accuracy", normal)     # accuracy when the autopilot first ran
        self.state.setdefault("best_hard_accuracy", hard)
        self.max_drop = max_drop

    def scores(self):
        normal = float(np.mean(self.model.predict(self.X_val) == self.y_val))
        hard = float(np.mean(self.model.predict(self.hard_X) == self.y_val))
        return normal, hard

    def run_round(self, steps=1500, learning_rate=None, batch=64):
        """learning_rate: default 0.05 for the first 4 rounds (learn the new variations), then 0.01
        (settle down and recover the accuracy on normal digits, so the update passes the safety check)."""
        if learning_rate is None:
            learning_rate = 0.05 if self.state.get("digit_rounds", 0) < 4 else 0.01
        for _ in range(steps):
            made = self.rng.integers(0, len(self.y_train), batch // 2)      # examples it makes itself
            kept = self.rng.integers(0, len(self.y_train), batch // 2)      # original examples (no forgetting)
            X = [np.stack([augment_digit(self.images[i], self.rng) for i in made]), self.X_train[kept]]
            y = [self.y_train[made], self.y_train[kept]]
            if self.extra_X is not None and len(self.extra_y):
                extra = self.rng.integers(0, len(self.extra_y), 4)
                X.append(self.extra_X[extra])
                y.append(self.extra_y[extra])
            _, cache = self.model.forward(np.concatenate(X))
            self.model.update(self.model.backward(cache, one_hot(np.concatenate(y), 10)), learning_rate)

        normal, hard = self.scores()
        saved = (normal >= self.state["reference_accuracy"] - self.max_drop
                 and hard > self.state["best_hard_accuracy"])
        if saved:
            self.state["best_hard_accuracy"] = hard
            self.metadata.update({"architecture": " -> ".join(map(str, self.model.layer_sizes)),
                                  "val_accuracy": normal, "hard_val_accuracy": hard,
                                  "autopilot_rounds": self.state.get("digit_rounds", 0) + 1})
            self.model.save(self.path, self.metadata)
        self.state["digit_rounds"] = self.state.get("digit_rounds", 0) + 1
        return {"normal_accuracy": normal, "hard_accuracy": hard, "saved": saved}


class TextAutopilot:

    def __init__(self, model, tokenizer, path, train_ids, val_ids, state=None, seed=0, metadata=None):
        self.model, self.tokenizer, self.path, self.metadata = model, tokenizer, path, dict(metadata or {})
        self.train_ids, self.val_ids = train_ids, val_ids
        self.rng = np.random.default_rng(seed)
        self.is_transformer = hasattr(model, "blocks")
        if self.is_transformer:
            from .transformer import Adam
            self.optimizer = Adam(model)
        self.state = state if state is not None else {}
        self.state.setdefault("best_val_loss", self.val_loss())

    def val_loss(self):
        if self.is_transformer:
            from .transformer import estimate_loss
        else:
            from .language_model import estimate_loss
        return estimate_loss(self.model, self.val_ids)

    def run_round(self, steps=1000, learning_rate=None):
        from .text_data import get_batch, get_sequence_batch
        if learning_rate is None:
            learning_rate = 3e-4 if self.is_transformer else 0.01
        for _ in range(steps):
            if self.is_transformer:
                X, Y = get_sequence_batch(self.train_ids, self.model.context, 32, self.rng)
                self.model.loss(X, Y)
                self.model.backward()
                self.optimizer.step(learning_rate)
            else:
                X, Y = get_batch(self.train_ids, self.model.context, 128, self.rng)
                _, cache = self.model.forward(X)
                self.model.update(self.model.backward(cache, Y), learning_rate)

        loss = self.val_loss()
        saved = loss < self.state["best_val_loss"]
        if saved:
            self.state["best_val_loss"] = loss
            self.metadata.update({"val_loss": loss, "autopilot_rounds": self.state.get("text_rounds", 0) + 1})
            self.model.save(self.path, self.tokenizer, self.metadata)
        self.state["text_rounds"] = self.state.get("text_rounds", 0) + 1
        return {"val_loss": loss, "saved": saved}


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
