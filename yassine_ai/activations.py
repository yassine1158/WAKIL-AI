"""Activation functions used by the network, with their derivatives."""
import numpy as np


def relu(z):
    """ReLU(z) = max(0, z). Keeps positive values, sets negatives to 0."""
    return np.maximum(0, z)


def relu_derivative(z):
    """dReLU/dz = 1 where z > 0, else 0."""
    return (z > 0).astype(z.dtype)


def softmax(z):
    """Turns each row of raw scores (logits) into probabilities that sum to 1.

    Subtracting the row maximum first gives the same result (it cancels out
    in the fraction) but prevents overflow in np.exp for large scores.
    """
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)
