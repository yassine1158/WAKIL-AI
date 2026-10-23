"""Wakil AI: neural networks implemented and trained from scratch with NumPy."""
from .activations import relu, relu_derivative, softmax
from .data import load_mnist, load_prepared_mnist, one_hot, preprocess
from .losses import cross_entropy
from .model import WakilNet
from .trainer import TrainConfig, evaluate, train

__version__ = "0.2.0"

__all__ = [
    "WakilNet", "TrainConfig", "train", "evaluate",
    "load_mnist", "load_prepared_mnist", "preprocess", "one_hot",
    "relu", "relu_derivative", "softmax", "cross_entropy",
]
