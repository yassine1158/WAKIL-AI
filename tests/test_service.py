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


def test_teaching_learns_and_saves_a_personal_model(tmp_path):
    rng = np.random.default_rng(0)
    replay_X = rng.random((200, 784), dtype=np.float32)
    replay_y = rng.integers(0, 10, 200)
    model = WakilNet((784, 32, 10))
    model.save(tmp_path / "wakil_ai_mnist.npz", {"architecture": "784 -> 32 -> 10"})
    service = WakilService(models_dir=str(tmp_path), outputs_dir=str(tmp_path), data_dir=str(tmp_path),
                           replay_data=(replay_X, replay_y, replay_X[:50], replay_y[:50]))
    service.teacher = None
    image = canvas_to_mnist_format(drawing())

    message = service.teach(image, 3)
    assert message.startswith("Learned!"), message
    assert (tmp_path / "wakil_ai_mnist_personal.npz").exists() and (tmp_path / "my_digits.npz").exists()
    probs = service.read_digit(drawing())[0]
    assert max(probs, key=probs.get) == "3"

    reloaded = WakilService(models_dir=str(tmp_path), outputs_dir=str(tmp_path), data_dir=str(tmp_path))
    assert "personal version" in reloaded.info()
    assert reloaded.reset_personal_model().startswith("Back to the original model")
    assert not (tmp_path / "wakil_ai_mnist_personal.npz").exists()


def test_teaching_needs_a_drawing_and_a_label(service_with_models):
    assert "Read first" in service_with_models.teach(None, 3)
    assert "Choose the correct digit" in service_with_models.teach(np.zeros((28, 28)), None)


def test_safety_check_cancels_a_harmful_update(tmp_path):
    from wakil_ai.continual import DigitTeacher
    rng = np.random.default_rng(0)
    centers = rng.random((10, 784), dtype=np.float32)
    y = rng.integers(0, 10, 300)
    X = (centers[y] + 0.05 * rng.standard_normal((300, 784))).astype(np.float32)   # an easy problem
    model = WakilNet((784, 16, 10), seed=0)
    for _ in range(100):
        _, cache = model.forward(X)
        model.update(model.backward(cache, np.eye(10, dtype=np.float32)[y]), 0.5)
    teacher = DigitTeacher(model, X, y, X, y, store_path=str(tmp_path / "my.npz"))
    before = {k: v.copy() for k, v in model.params.items()}
    teacher.add_example(np.full(784, 255.0), 0)
    report = teacher.learn(learning_rate=50.0)      # absurd learning rate: destroys the model
    assert not report["accepted"]
    assert all(np.array_equal(before[k], model.params[k]) for k in before)   # update was undone
