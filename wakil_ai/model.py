"""WakilNet: a fully connected neural network written with NumPy only.

Architecture (default): 784 -> 128 (ReLU) -> 10 (softmax).
Any number of hidden layers is supported, e.g. layer_sizes=(784, 256, 128, 10).

Notation for layer i (1..L):
    Z_i = A_{i-1} @ W_i + b_i
    A_i = ReLU(Z_i)        for hidden layers
    A_L = softmax(Z_L)     for the output layer      (A_0 = X, the input)

Training loss = cross-entropy + (weight_decay / 2) * sum of squared weights.
"""
import json

import numpy as np

from .activations import relu, relu_derivative, softmax


class WakilNet:

    def __init__(self, layer_sizes=(784, 128, 10), seed=42, dtype=np.float32):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least an input and an output size")
        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        rng = np.random.default_rng(seed)
        self.params = {}
        for i in range(1, self.num_layers + 1):
            fan_in, fan_out = self.layer_sizes[i - 1], self.layer_sizes[i]
            # He init before a ReLU, Xavier/LeCun init before the softmax
            scale = np.sqrt(2.0 / fan_in) if i < self.num_layers else np.sqrt(1.0 / fan_in)
            self.params[f"W{i}"] = (rng.standard_normal((fan_in, fan_out)) * scale).astype(dtype)
            self.params[f"b{i}"] = np.zeros(fan_out, dtype=dtype)

    @property
    def num_layers(self):
        """Number of layers with weights (hidden layers + output layer)."""
        return len(self.layer_sizes) - 1

    def num_parameters(self):
        return int(sum(p.size for p in self.params.values()))

    # ------------------------------------------------------------------ forward
    def forward(self, X):
        """Returns (probabilities, cache). The cache is needed by backward()."""
        cache = {"A0": X}
        A = X
        for i in range(1, self.num_layers + 1):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = relu(Z) if i < self.num_layers else softmax(Z)
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = A
        return A, cache

    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ----------------------------------------------------------------- backward
    def backward(self, cache, y_onehot, weight_decay=0.0):
        """Backpropagation: gradient of the loss with respect to every parameter.

        Output layer (softmax + cross-entropy):  dZ_L = (P - Y) / m
        For each layer i, from L down to 1:
            dW_i = A_{i-1}^T @ dZ_i  (+ weight_decay * W_i)
            db_i = sum of dZ_i over the batch
            dZ_{i-1} = (dZ_i @ W_i^T) * ReLU'(Z_{i-1})
        """
        m = y_onehot.shape[0]
        grads = {}
        dZ = (cache[f"A{self.num_layers}"] - y_onehot) / m
        for i in range(self.num_layers, 0, -1):
            W = self.params[f"W{i}"]
            grads[f"W{i}"] = cache[f"A{i - 1}"].T @ dZ + weight_decay * W
            grads[f"b{i}"] = dZ.sum(axis=0)
            if i > 1:
                dZ = (dZ @ W.T) * relu_derivative(cache[f"Z{i - 1}"])
        return grads

    def regularization_loss(self, weight_decay):
        """(weight_decay / 2) * sum of all squared weights (biases excluded)."""
        if weight_decay == 0:
            return 0.0
        squared = sum(float(np.sum(self.params[f"W{i}"] ** 2)) for i in range(1, self.num_layers + 1))
        return 0.5 * weight_decay * squared

    # ------------------------------------------------------------------- update
    def update(self, grads, learning_rate):
        """Gradient descent step: W <- W - learning_rate * dL/dW."""
        for name in self.params:
            self.params[name] -= learning_rate * grads[name]

    def get_state(self):
        """A copy of all parameters (used to remember the best epoch)."""
        return {name: value.copy() for name, value in self.params.items()}

    def set_state(self, state):
        self.params = {name: value.copy() for name, value in state.items()}

    # ---------------------------------------------------------------- save/load
    def save(self, path, metadata=None):
        """Saves parameters + architecture + metadata to a compressed .npz file."""
        np.savez_compressed(
            path,
            layer_sizes=np.array(self.layer_sizes),
            metadata=np.array(json.dumps(metadata or {})),
            **self.params,
        )

    @classmethod
    def load(cls, path):
        """Returns (model, metadata). Also reads files saved by the Phase 1 notebook."""
        with np.load(path) as data:
            weight_names = sorted((k for k in data.files if k.startswith("W")), key=lambda k: int(k[1:]))
            if "layer_sizes" in data.files:
                layer_sizes = tuple(int(s) for s in data["layer_sizes"])
            else:  # Phase 1 files: infer the architecture from the weight shapes
                layer_sizes = (data[weight_names[0]].shape[0],) + tuple(data[k].shape[1] for k in weight_names)
            params = {k: data[k] for k in data.files if k[0] in "Wb" and k[1:].isdigit()}
            metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
        model = cls(layer_sizes)
        model.set_state(params)
        return model, metadata
