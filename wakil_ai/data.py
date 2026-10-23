"""Loading and preparing the MNIST dataset."""
import os
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
NUM_CLASSES = 10


def load_mnist(cache_dir="data"):
    """Returns (x_train, y_train, x_test, y_test) as uint8 arrays.

    x_train: (60000, 28, 28), x_test: (10000, 28, 28), pixel values 0-255.
    The file (~11 MB) is downloaded once and cached in `cache_dir`.
    Falls back to scikit-learn's OpenML loader if the download fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "mnist.npz")
    try:
        if not os.path.exists(path):
            print("Downloading MNIST (~11 MB)...")
            urllib.request.urlretrieve(MNIST_URL, path)
        with np.load(path) as data:
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]
    except Exception as error:
        print(f"Direct download failed ({error}). Falling back to OpenML (1-2 minutes)...")
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, as_frame=False,
                            parser="liac-arff", return_X_y=True)
        X = X.reshape(-1, 28, 28).astype(np.uint8)
        y = y.astype(np.uint8)
        # OpenML keeps the original order: first 60,000 = train, last 10,000 = test
        return X[:60000], y[:60000], X[60000:], y[60000:]


def preprocess(images):
    """(N, 28, 28) uint8 in [0, 255]  ->  (N, 784) float32 in [0, 1]."""
    images = np.asarray(images)
    return images.reshape(len(images), -1).astype(np.float32) / 255.0


def one_hot(labels, num_classes=NUM_CLASSES):
    """[3, 0] -> [[0,0,0,1,0,...], [1,0,0,0,0,...]]"""
    labels = np.asarray(labels)
    encoded = np.zeros((len(labels), num_classes), dtype=np.float32)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def train_val_split(X, y, val_size=10000, seed=42):
    """Stratified split: same proportion of each digit in train and validation."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=val_size, random_state=seed, stratify=y)


def load_prepared_mnist(val_size=10000, train_limit=None, seed=42, cache_dir="data"):
    """Loads MNIST and returns a dict with X/y arrays for train, val and test.

    train_limit: if set, keep only the first N training examples (useful to
    study overfitting on a small dataset).
    """
    x_train_full, y_train_full, x_test, y_test = load_mnist(cache_dir)
    X_train, X_val, y_train, y_val = train_val_split(
        preprocess(x_train_full), y_train_full, val_size=val_size, seed=seed)
    if train_limit is not None:
        X_train, y_train = X_train[:train_limit], y_train[:train_limit]
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": preprocess(x_test), "y_test": y_test,
        "test_images": x_test,
    }
