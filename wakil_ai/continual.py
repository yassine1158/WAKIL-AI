"""Continual learning: the digit reader keeps learning from the user's corrections.

When the model reads a drawing wrong, the user gives the right label. We store
the example and run a few gradient-descent steps on it. Two safeguards:

- Replay: every batch mixes the user's examples with MNIST examples. Training
  only on the new drawings would make the network forget what it knew
  ("catastrophic forgetting").
- Safety check: before keeping the update, accuracy is measured on held-out
  MNIST images. If it dropped too much, the update is thrown away.
"""
import os

import numpy as np

from .data import one_hot


class DigitTeacher:

    def __init__(self, model, replay_X, replay_y, check_X, check_y, store_path="data/my_digits.npz",
                 max_drop=0.005, seed=0):
        """replay_X/replay_y: examples mixed into every batch (MNIST training images, 784 values in [0, 1]).
        check_X/check_y: held-out examples for the safety check (MNIST validation images).
        max_drop: largest accepted drop of accuracy on the check set (0.005 = 0.5 points)."""
        self.model = model
        self.replay_X, self.replay_y = replay_X, replay_y
        self.check_X, self.check_y = check_X, check_y
        self.store_path, self.max_drop = store_path, max_drop
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((0, 784), dtype=np.float32)
        self.y = np.zeros(0, dtype=np.int64)
        if os.path.exists(store_path):
            with np.load(store_path) as data:
                self.X, self.y = data["X"].astype(np.float32), data["y"].astype(np.int64)

    def __len__(self):
        return len(self.y)

    def add_example(self, image, label):
        """image: 28x28 (or 784) array in [0, 255]. Saved to disk immediately."""
        x = np.asarray(image, dtype=np.float32).reshape(1, 784) / 255.0
        self.X = np.concatenate([self.X, x])
        self.y = np.concatenate([self.y, [int(label)]])
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        np.savez_compressed(self.store_path, X=self.X, y=self.y)

    def check_accuracy(self):
        return float(np.mean(self.model.predict(self.check_X) == self.check_y))

    def user_accuracy(self):
        return float(np.mean(self.model.predict(self.X) == self.y)) if len(self) else float("nan")

    def learn(self, steps=40, learning_rate=0.05, user_per_batch=8, replay_per_batch=24):
        """A few gradient steps on (user examples + replay). Returns a report dict.

        The update is kept only if accuracy on the check set did not drop more than max_drop.
        """
        if len(self) == 0:
            return {"accepted": False, "reason": "no examples yet"}
        before_state = self.model.get_state()
        report = {"check_before": self.check_accuracy(), "user_before": self.user_accuracy()}
        for _ in range(steps):
            u = self.rng.integers(0, len(self), size=user_per_batch)
            r = self.rng.integers(0, len(self.replay_y), size=replay_per_batch)
            X = np.concatenate([self.X[u], self.replay_X[r]])
            Y = one_hot(np.concatenate([self.y[u], self.replay_y[r]]), 10)
            _, cache = self.model.forward(X)
            self.model.update(self.model.backward(cache, Y), learning_rate)
        report["check_after"] = self.check_accuracy()
        report["user_after"] = self.user_accuracy()
        report["accepted"] = report["check_after"] >= report["check_before"] - self.max_drop
        if not report["accepted"]:
            self.model.set_state(before_state)       # safety check failed: undo the update
            report["reason"] = "accuracy on MNIST dropped too much, update cancelled"
        return report
