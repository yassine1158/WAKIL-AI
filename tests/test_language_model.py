import numpy as np
import pytest

from wakil_ai.language_model import BigramCounts, CharMLP, cross_entropy_from_ids, train_language_model
from wakil_ai.text_data import get_batch, split_ids
from wakil_ai.tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    assert tok.chars == [" ", "d", "e", "h", "l", "o", "r", "w"]
    assert tok.decode(tok.encode("hello")) == "hello"
    assert CharTokenizer.from_json(tok.to_json()).chars == tok.chars
    with pytest.raises(ValueError):
        tok.encode("hello!")


def test_tokenizer_works_for_any_language():
    text = "مرحبا Wakil, ça va ?"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_get_batch_targets_are_the_next_character():
    ids = np.arange(100)
    X, Y = get_batch(ids, context=5, batch_size=16, rng=np.random.default_rng(0))
    assert X.shape == (16, 5) and Y.shape == (16,)
    assert np.array_equal(Y, X[:, -1] + 1)


def test_split_is_contiguous():
    train, val = split_ids(np.arange(100), val_fraction=0.1)
    assert len(train) == 90 and val[0] == 90


def test_bigram_learns_a_deterministic_pattern():
    ids = np.tile(np.arange(4), 50)            # 0 1 2 3 0 1 2 3 ...
    model = BigramCounts(4, smoothing=0.01).fit(ids)
    assert np.allclose(model.probs.sum(axis=1), 1.0)
    assert model.probs[0, 1] > 0.99
    assert model.generate([0], 7, np.random.default_rng(0)) == [0, 1, 2, 3, 0, 1, 2, 3]


def test_cross_entropy_from_ids():
    probs = np.array([[0.5, 0.5], [0.9, 0.1]])
    assert np.isclose(cross_entropy_from_ids(probs, np.array([0, 0])), -(np.log(0.5) + np.log(0.9)) / 2)


def test_untrained_model_predicts_nearly_uniform():
    model = CharMLP(vocab_size=65)
    X = np.random.default_rng(0).integers(0, 65, size=(500, 8))
    Y = np.random.default_rng(1).integers(0, 65, size=500)
    assert abs(model.loss(X, Y) - np.log(65)) < 0.1


def test_backward_matches_numerical_gradient(eps=1e-5):
    rng = np.random.default_rng(0)
    model = CharMLP(vocab_size=5, context=3, emb_dim=2, hidden=4, seed=1, dtype=np.float64)
    model.params["W2"] *= 10                    # larger weights make the check more sensitive
    X = np.array([[0, 1, 1], [2, 2, 2], [4, 0, 3], [1, 1, 0]])   # repeated ids on purpose
    Y = rng.integers(0, 5, size=4)

    _, cache = model.forward(X)
    analytic = model.backward(cache, Y)
    for name, param in model.params.items():
        numeric = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            loss_plus = model.loss(X, Y)
            param[idx] = original - eps
            loss_minus = model.loss(X, Y)
            param[idx] = original
            numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
        rel_error = np.linalg.norm(numeric - analytic[name]) / (
            np.linalg.norm(numeric) + np.linalg.norm(analytic[name]) + 1e-12)
        assert rel_error < 1e-6, f"{name}: relative error {rel_error:.2e}"


def test_mlp_learns_a_repeating_text():
    tok = CharTokenizer.from_text("abcd")
    ids = tok.encode("abcd" * 200)
    model = CharMLP(tok.vocab_size, context=3, emb_dim=4, hidden=32)
    history = train_language_model(model, ids, ids, steps=600, batch_size=32, learning_rate=0.5,
                                   eval_every=600, verbose=False)
    assert history["val_loss"][-1] < 0.05
    generated = tok.decode(model.generate(tok.encode("abc"), 8, np.random.default_rng(0), temperature=0.5))
    assert generated == "abcdabcdabc"


def test_save_and_load_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("to be or not to be")
    model = CharMLP(tok.vocab_size, context=4, emb_dim=3, hidden=8)
    path = tmp_path / "lm.npz"
    model.save(path, tok, {"note": "test"})
    loaded, loaded_tok, metadata = CharMLP.load(path)
    X = np.array([[0, 1, 2, 3]])
    assert loaded_tok.chars == tok.chars and metadata == {"note": "test"}
    assert np.array_equal(loaded.forward(X)[0], model.forward(X)[0])
