import json

import numpy as np

from yassine_ai import TrainConfig, YassineNet, cross_entropy, one_hot, preprocess, relu, softmax, train


def test_softmax_rows_sum_to_one_and_are_stable():
    probs = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs[0], probs[1])  # shifting all logits does not change softmax


def test_relu():
    assert np.array_equal(relu(np.array([-2.0, 0.0, 3.0])), np.array([0.0, 0.0, 3.0]))


def test_cross_entropy_known_values():
    y = one_hot([0], 2)
    assert np.isclose(cross_entropy(np.array([[0.5, 0.5]]), y), np.log(2))
    assert cross_entropy(np.array([[0.99, 0.01]]), y) < 0.02


def test_preprocess_and_one_hot():
    images = np.full((2, 28, 28), 255, dtype=np.uint8)
    X = preprocess(images)
    assert X.shape == (2, 784) and X.dtype == np.float32 and X.max() == 1.0
    assert np.array_equal(one_hot([3], 10)[0], np.eye(10)[3])


def test_shapes_and_parameter_count():
    model = YassineNet((784, 128, 10))
    assert model.num_parameters() == 784 * 128 + 128 + 128 * 10 + 10  # 101,770
    probs = model.predict_proba(np.zeros((5, 784), dtype=np.float32))
    assert probs.shape == (5, 10)


def test_untrained_model_guesses_randomly():
    X = np.random.default_rng(0).random((500, 784), dtype=np.float32)
    y = np.random.default_rng(1).integers(0, 10, 500)
    loss = cross_entropy(YassineNet().predict_proba(X), one_hot(y))
    assert abs(loss - np.log(10)) < 0.5  # close to ln(10) = 2.30


def test_same_seed_gives_same_weights():
    a, b = YassineNet(seed=3), YassineNet(seed=3)
    assert all(np.array_equal(a.params[k], b.params[k]) for k in a.params)


def _blobs(n=400, seed=0):
    """Two easy-to-separate clouds of points in 2D."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = (rng.standard_normal((n, 2)) + np.where(y[:, None] == 1, 2.0, -2.0)).astype(np.float32)
    return X, y


def test_training_learns_a_simple_problem():
    X, y = _blobs()
    model = YassineNet((2, 16, 2), seed=0)
    history = train(model, X[:300], y[:300], X[300:], y[300:],
                    TrainConfig(learning_rate=0.1, batch_size=32, epochs=20), verbose=False)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert np.mean(model.predict(X[300:]) == y[300:]) > 0.95


def test_early_stopping_stops_and_restores_best_epoch():
    # Random labels: there is nothing general to learn, so the model can only
    # memorize the training set and validation loss soon gets worse (overfitting).
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 10)).astype(np.float32)
    y = rng.integers(0, 2, 300)
    model = YassineNet((10, 64, 2), seed=0)
    history = train(model, X[:50], y[:50], X[50:], y[50:],
                    TrainConfig(learning_rate=0.5, batch_size=10, epochs=200, patience=3), verbose=False)
    assert history["stopped_early"]
    assert len(history["val_loss"]) < 200
    best = history["best_epoch"] - 1
    assert history["val_loss"][best] == min(history["val_loss"])


def test_save_and_load_roundtrip(tmp_path):
    model = YassineNet((784, 32, 16, 10), seed=5)
    path = tmp_path / "model.npz"
    model.save(path, {"note": "test"})
    loaded, metadata = YassineNet.load(path)
    X = np.random.default_rng(0).random((20, 784), dtype=np.float32)
    assert loaded.layer_sizes == (784, 32, 16, 10)
    assert metadata == {"note": "test"}
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))


def test_loads_phase1_notebook_files(tmp_path):
    """Files saved by the Phase 1 notebook have W1/b1/W2/b2 + metadata, no layer_sizes."""
    model = YassineNet((784, 128, 10))
    path = tmp_path / "phase1.npz"
    np.savez_compressed(path, metadata=np.array(json.dumps({"name": "v1"})), **model.params)
    loaded, metadata = YassineNet.load(path)
    assert loaded.layer_sizes == (784, 128, 10)
    assert metadata["name"] == "v1"
