import os

import numpy as np

from wakil_ai.autopilot import DigitAutopilot, TextAutopilot, augment_digit
from wakil_ai.language_model import CharMLP
from wakil_ai.model import WakilNet
from wakil_ai.service import WakilService
from wakil_ai.tokenizer import CharTokenizer


def prototype_digits(n=300, seed=0):
    """10 simple 'digits' (a filled block at a different place each), with their labels."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    X = np.zeros((n, 28, 28), dtype=np.float32)
    for i, label in enumerate(y):
        r, c = 4 + 6 * (label // 4), 4 + 6 * (label % 4)
        X[i, r:r + 8, c:c + 8] = 1.0
    return X.reshape(n, 784), y


def test_augment_digit_makes_a_new_version_of_the_same_image():
    rng = np.random.default_rng(0)
    image = np.zeros((28, 28), dtype=np.uint8)
    image[6:22, 12:16] = 255                       # a "1"
    new = augment_digit(image, rng)
    assert new.shape == (784,) and 0 <= new.min() and new.max() <= 1
    assert not np.allclose(new, image.reshape(784) / 255.0)
    assert new.sum() > 0.3 * image.sum() / 255.0  # the digit is still there


def test_digit_autopilot_saves_only_when_better(tmp_path):
    X, y = prototype_digits()
    path = str(tmp_path / "digits.npz")

    frozen = DigitAutopilot(WakilNet((784, 32, 10)), path, X, y, X[:100], y[:100], metadata={"note": "kept"})
    assert not frozen.run_round(steps=5, learning_rate=0.0)["saved"]      # no learning: nothing saved
    assert not os.path.exists(path)

    pilot = DigitAutopilot(WakilNet((784, 32, 10)), path, X, y, X[:100], y[:100], metadata={"note": "kept"})
    report = pilot.run_round(steps=150, learning_rate=0.1)
    assert report["saved"] and os.path.exists(path)
    assert report["hard_accuracy"] > 0.5
    _, metadata = WakilNet.load(path)
    assert metadata["note"] == "kept" and metadata["autopilot_rounds"] == 1


def test_text_autopilot_saves_only_when_the_validation_loss_improves(tmp_path):
    tok = CharTokenizer.from_text("abcd")
    ids = tok.encode("abcd" * 200)
    path = str(tmp_path / "lm.npz")

    frozen = TextAutopilot(CharMLP(tok.vocab_size, context=3, emb_dim=4, hidden=16), tok, path, ids, ids)
    assert not frozen.run_round(steps=5, learning_rate=0.0)["saved"]
    assert not os.path.exists(path)

    pilot = TextAutopilot(CharMLP(tok.vocab_size, context=3, emb_dim=4, hidden=16), tok, path, ids, ids)
    before = pilot.state["best_val_loss"]
    report = pilot.run_round(steps=300, learning_rate=0.5)
    assert report["saved"] and report["val_loss"] < before and os.path.exists(path)


def test_service_reloads_a_model_improved_by_the_autopilot(tmp_path):
    WakilNet((784, 16, 10)).save(tmp_path / "wakil_ai_mnist.npz", {"architecture": "784 -> 16 -> 10"})
    service = WakilService(models_dir=str(tmp_path), outputs_dir=str(tmp_path))
    assert "improved on its own" not in service.info()

    path = tmp_path / "wakil_ai_mnist.npz"
    WakilNet((784, 16, 10)).save(path, {"architecture": "784 -> 16 -> 10", "autopilot_rounds": 3})
    later = os.path.getmtime(path) + 10
    os.utime(path, (later, later))                 # make sure the file looks newer
    assert "improved on its own" in service.info()
