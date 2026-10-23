"""Tests for the application logic (no Gradio needed)."""
import numpy as np
import pytest

from wakil_ai.image import canvas_to_mnist_format
from wakil_ai.language_model import CharMLP
from wakil_ai.model import WakilNet
from wakil_ai.service import WakilService
from wakil_ai.tokenizer import CharTokenizer


def drawing(background="white"):
    """A 280x280 canvas with a black vertical bar drawn off-center (like a '1')."""
    if background == "white":
        canvas = np.full((280, 280, 4), 255, dtype=np.uint8)
    else:                                       # transparent
        canvas = np.zeros((280, 280, 4), dtype=np.uint8)
    canvas[40:200, 50:80] = [0, 0, 0, 255]
    return canvas


@pytest.mark.parametrize("background", ["white", "transparent"])
def test_canvas_becomes_a_centered_mnist_image(background):
    image = canvas_to_mnist_format(drawing(background))
    assert image.shape == (28, 28)
    assert image.max() == pytest.approx(255, abs=1) and image[0, 0] == 0   # white ink, black background
    rows, cols = np.indices(image.shape)
    cy, cx = (rows * image).sum() / image.sum(), (cols * image).sum() / image.sum()
    assert abs(cy - 14) < 1.5 and abs(cx - 14) < 1.5                     # centered by center of mass
    assert (image > 0).any(axis=1).sum() <= 21                           # fits in a 20-pixel box (+ blur)


def test_empty_canvas_is_rejected():
    with pytest.raises(ValueError):
        canvas_to_mnist_format(np.full((100, 100, 4), 255, dtype=np.uint8))


def test_service_without_models_explains_what_to_do(tmp_path):
    service = WakilService(models_dir=str(tmp_path))
    assert "python train.py" in service.read_digit(drawing())[2]
    assert "train_lm.py" in service.write("ROMEO:")[1]
    assert "not trained yet" in service.info()


@pytest.fixture
def service_with_models(tmp_path):
    WakilNet((784, 16, 10)).save(tmp_path / "wakil_ai_mnist.npz", {"architecture": "784 -> 16 -> 10",
                                                                    "val_accuracy": 0.5})
    tok = CharTokenizer.from_text("ROMEO: hello world\n")
    CharMLP(tok.vocab_size, context=4, emb_dim=2, hidden=8).save(tmp_path / "wakil_lm.npz", tok, {"val_loss": 2.0})
    return WakilService(models_dir=str(tmp_path), outputs_dir=str(tmp_path))


def test_read_digit_returns_probabilities(service_with_models):
    probs, image, message = service_with_models.read_digit(drawing())
    assert set(probs) == {str(d) for d in range(10)}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert image.shape == (28, 28) and message.startswith("Wakil AI reads")
    assert service_with_models.read_digit(np.full((50, 50, 4), 255, dtype=np.uint8))[2].startswith("The canvas is empty")


def test_write_continues_the_prompt_and_drops_unknown_characters(service_with_models):
    text, message = service_with_models.write("ROMEO: é", length=20, seed=1)
    assert text.startswith("ROMEO: ") and len(text) == len("ROMEO: ") + 20
    assert "é" in message
    assert service_with_models.write("ROMEO:", length=20, seed=3)[0] == service_with_models.write("ROMEO:", length=20, seed=3)[0]


def test_info_describes_the_models(service_with_models):
    info = service_with_models.info()
    assert "784 -> 16 -> 10" in info and "MLP (Part 1)" in info and "from scratch" in info
