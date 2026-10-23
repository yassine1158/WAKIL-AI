"""Training loop: mini-batch gradient descent with optional L2 and early stopping."""
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .data import one_hot
from .losses import cross_entropy


@dataclass
class TrainConfig:
    learning_rate: float = 0.1
    batch_size: int = 64
    epochs: int = 20
    weight_decay: float = 0.0          # L2 regularization strength (0 = off)
    patience: Optional[int] = None     # early stopping: stop after N epochs without val improvement
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def evaluate(model, X, y, batch_size=10000):
    """Returns (cross-entropy loss, accuracy) on a dataset, without regularization."""
    probs = np.concatenate([model.predict_proba(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])
    return cross_entropy(probs, one_hot(y, probs.shape[1])), float(np.mean(probs.argmax(axis=1) == y))


def train(model, X_train, y_train, X_val, y_val, config, verbose=True):
    """Trains `model` in place and returns the history dict.

    At the end, the model is set back to the parameters of the epoch with the
    lowest validation loss (the "best checkpoint").
    """
    rng = np.random.default_rng(config.seed)
    num_classes = model.layer_sizes[-1]
    Y_train = one_hot(y_train, num_classes)
    n = len(X_train)
    history = {"batch_loss": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "best_epoch": None, "stopped_early": False}
    best_val_loss, best_state, epochs_without_improvement = np.inf, None, 0

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        order = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            idx = order[start:start + config.batch_size]
            probs, cache = model.forward(X_train[idx])
            loss = cross_entropy(probs, Y_train[idx]) + model.regularization_loss(config.weight_decay)
            grads = model.backward(cache, Y_train[idx], weight_decay=config.weight_decay)
            model.update(grads, config.learning_rate)
            history["batch_loss"].append(loss)

        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)
        for key, value in [("train_loss", train_loss), ("train_acc", train_acc),
                           ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[key].append(value)

        if not np.isfinite(train_loss):
            print("Loss became NaN/inf: the learning rate is probably too high. Stopping.")
            break

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_state, epochs_without_improvement = val_loss, model.get_state(), 0
            history["best_epoch"] = epoch
        else:
            epochs_without_improvement += 1

        if verbose:
            print(f"Epoch {epoch:3d}/{config.epochs} | train loss {train_loss:.4f} acc {train_acc:.2%} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.2%} | {time.time() - start_time:.1f}s"
                  + ("  *best" if improved else ""))

        if config.patience is not None and epochs_without_improvement >= config.patience:
            history["stopped_early"] = True
            if verbose:
                print(f"Early stopping: no validation improvement for {config.patience} epochs.")
            break

    if best_state is not None:
        model.set_state(best_state)
        if verbose:
            print(f"Restored best model from epoch {history['best_epoch']} (val loss {best_val_loss:.4f}).")
    return history
