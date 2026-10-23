"""Loss functions."""
import numpy as np


def cross_entropy(probs, y_onehot, eps=1e-12):
    """Mean cross-entropy: -mean(log(probability given to the correct class)).

    probs:    (m, k) predicted probabilities (rows sum to 1)
    y_onehot: (m, k) one-hot true labels
    """
    return float(-np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1)))
