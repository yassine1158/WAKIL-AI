"""Gradient check: compares backpropagation with numerical derivatives.

If these tests fail, YassineNet.backward() has a bug.
"""
import numpy as np
import pytest

from yassine_ai import YassineNet, cross_entropy, one_hot


def total_loss(model, X, Y, weight_decay):
    return cross_entropy(model.predict_proba(X), Y) + model.regularization_loss(weight_decay)


@pytest.mark.parametrize("layer_sizes", [(5, 4, 3), (5, 6, 4, 3)])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_backward_matches_numerical_gradient(layer_sizes, weight_decay, eps=1e-5):
    rng = np.random.default_rng(0)
    model = YassineNet(layer_sizes, seed=1, dtype=np.float64)
    X = rng.standard_normal((7, layer_sizes[0]))
    Y = one_hot(rng.integers(0, layer_sizes[-1], size=7), layer_sizes[-1]).astype(np.float64)

    _, cache = model.forward(X)
    analytic = model.backward(cache, Y, weight_decay=weight_decay)

    for name, param in model.params.items():
        numeric = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            loss_plus = total_loss(model, X, Y, weight_decay)
            param[idx] = original - eps
            loss_minus = total_loss(model, X, Y, weight_decay)
            param[idx] = original
            numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
        rel_error = np.linalg.norm(numeric - analytic[name]) / (
            np.linalg.norm(numeric) + np.linalg.norm(analytic[name]) + 1e-12)
        assert rel_error < 1e-6, f"{name}: relative error {rel_error:.2e}"
