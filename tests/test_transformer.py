import numpy as np

from wakil_ai.language_model import load_language_model
from wakil_ai.text_data import get_sequence_batch
from wakil_ai.tokenizer import CharTokenizer
from wakil_ai.transformer import LayerNorm, TinyGPT, train_transformer


def tiny_model(dtype=np.float64):
    return TinyGPT(vocab_size=7, block_size=5, d_model=8, n_heads=2, n_layers=2, seed=1, dtype=dtype)


def test_sequence_batch_targets_are_shifted_by_one():
    X, Y = get_sequence_batch(np.arange(100), context=6, batch_size=4, rng=np.random.default_rng(0))
    assert X.shape == Y.shape == (4, 6)
    assert np.array_equal(Y, X + 1)


def test_layernorm_normalizes_each_vector():
    x = np.random.default_rng(0).normal(5, 3, size=(2, 4, 16))
    y = LayerNorm(16, np.float64).forward(x)
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-7)
    assert np.allclose(y.std(axis=-1), 1, atol=1e-3)


def test_causal_mask_hides_the_future():
    """Changing the last token must not change the predictions for earlier positions."""
    model = tiny_model()
    a = np.array([[1, 2, 3, 4, 5]])
    b = np.array([[1, 2, 3, 4, 0]])
    la, lb = model.forward(a), model.forward(b)
    assert np.allclose(la[0, :4], lb[0, :4])
    assert not np.allclose(la[0, 4], lb[0, 4])


def test_untrained_model_predicts_nearly_uniform():
    model = TinyGPT(vocab_size=65, block_size=16)
    rng = np.random.default_rng(0)
    loss = model.loss(rng.integers(0, 65, (8, 16)), rng.integers(0, 65, (8, 16)))
    assert abs(loss - np.log(65)) < 0.05


def test_backward_matches_numerical_gradient(eps=1e-6):
    rng = np.random.default_rng(0)
    model = tiny_model()
    for _, layer, key in model.named_parameters():
        if key == "W" or key.endswith("emb"):
            layer.params[key] *= 20          # bigger weights: attention is not uniform, check is sensitive
    idx = rng.integers(0, 7, (3, 5))
    idx[0, :3] = 2                           # repeated tokens
    targets = rng.integers(0, 7, (3, 5))

    model.loss(idx, targets)
    model.backward()
    analytic = {name: layer.grads[key].copy() for name, layer, key in model.named_parameters()}
    for name, layer, key in model.named_parameters():
        param = layer.params[key]
        numeric = np.zeros_like(param)
        for i in np.ndindex(param.shape):
            original = param[i]
            param[i] = original + eps
            loss_plus = model.loss(idx, targets)
            param[i] = original - eps
            loss_minus = model.loss(idx, targets)
            param[i] = original
            numeric[i] = (loss_plus - loss_minus) / (2 * eps)
        rel_error = np.linalg.norm(numeric - analytic[name]) / (
            np.linalg.norm(numeric) + np.linalg.norm(analytic[name]) + 1e-12)
        assert rel_error < 1e-6, f"{name}: relative error {rel_error:.2e}"


def test_transformer_learns_a_repeating_text():
    tok = CharTokenizer.from_text("abcde")
    ids = tok.encode("abcde" * 100)
    model = TinyGPT(tok.vocab_size, block_size=8, d_model=16, n_heads=2, n_layers=1)
    history = train_transformer(model, ids, ids, steps=150, batch_size=16, learning_rate=1e-2,
                                eval_every=150, verbose=False)
    assert history["val_loss"][-1] < 0.1
    out = tok.decode(model.generate(tok.encode("abc"), 7, np.random.default_rng(0), temperature=0.5))
    assert out == "abcdeabcde"


def test_save_and_load_roundtrip(tmp_path):
    tok = CharTokenizer.from_text("abcdefg")
    model = tiny_model(np.float32)
    path = tmp_path / "gpt.npz"
    model.save(path, tok, {"note": "test"})
    loaded, loaded_tok, metadata = load_language_model(path)
    assert isinstance(loaded, TinyGPT) and loaded_tok.chars == tok.chars and metadata == {"note": "test"}
    idx = np.array([[0, 1, 2, 3, 4]])
    assert np.array_equal(loaded.forward(idx), model.forward(idx))
