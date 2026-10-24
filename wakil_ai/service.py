"""Everything the application needs from the models, with no user-interface code.

app.py (the Gradio interface) only calls this class, so the interface can be
replaced (web page, mobile app, chat bot...) without touching the models.
"""
import json
import os

import numpy as np

from .continual import DigitTeacher
from .image import canvas_to_mnist_format
from .language_model import load_language_model
from .model import WakilNet

DIGIT_MODEL = "wakil_ai_mnist.npz"
PERSONAL_DIGIT_MODEL = "wakil_ai_mnist_personal.npz"   # the model after learning from your corrections
TEXT_MODELS = {"Transformer (TinyGPT)": "wakil_gpt.npz", "MLP (Part 1)": "wakil_lm.npz"}


class WakilService:

    def __init__(self, models_dir="models", outputs_dir="outputs", data_dir="data", replay_data=None):
        """replay_data: optional (replay_X, replay_y, check_X, check_y) for continual learning.
        By default MNIST is loaded the first time the user teaches the model."""
        self.models_dir, self.outputs_dir, self.data_dir = models_dir, outputs_dir, data_dir
        self.replay_data, self.teacher = replay_data, None
        self.digit_model, self.digit_metadata = None, {}
        self.text_models = {}                           # name -> (model, tokenizer, metadata)
        self._file_times = None
        self._refresh()

    def _model_file_times(self):
        names = [PERSONAL_DIGIT_MODEL, DIGIT_MODEL] + list(TEXT_MODELS.values())
        paths = [os.path.join(self.models_dir, name) for name in names]
        return {path: os.path.getmtime(path) for path in paths if os.path.exists(path)}

    def _refresh(self):
        """(Re)loads the models when their files changed, e.g. after the autopilot saved a better one."""
        times = self._model_file_times()
        if times == self._file_times:
            return
        self._file_times = times
        self._load_digit_model()
        self.text_models = {}
        for name, filename in TEXT_MODELS.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.text_models[name] = load_language_model(path)

    # ------------------------------------------------------------------ digits
    def _load_digit_model(self):
        for filename in (PERSONAL_DIGIT_MODEL, DIGIT_MODEL):      # your personal version first
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.digit_model, self.digit_metadata = WakilNet.load(path)
                return
        self.digit_model, self.digit_metadata = None, {}

    def _get_teacher(self):
        if self.teacher is None:
            if self.replay_data is None:
                from .data import load_prepared_mnist
                data = load_prepared_mnist(cache_dir=self.data_dir)
                self.replay_data = (data["X_train"], data["y_train"], data["X_val"][:2000], data["y_val"][:2000])
            self.teacher = DigitTeacher(self.digit_model, *self.replay_data,
                                        store_path=os.path.join(self.data_dir, "my_digits.npz"))
        self.teacher.model = self.digit_model
        return self.teacher

    def teach(self, image, label):
        """The user says the drawing `image` (28x28 MNIST-style array) is the digit `label`.
        Stores the example, learns from it, and saves the personal model. Returns a message."""
        self._refresh()
        if self.digit_model is None:
            return "No digit model yet. Train it first: python train.py"
        if image is None:
            return "Draw a digit and click Read first."
        if label is None or str(label) == "":
            return "Choose the correct digit first."
        label = int(label)
        teacher = self._get_teacher()
        teacher.add_example(image, label)
        report = teacher.learn()
        if not report["accepted"]:
            return (f"Saved your example, but the update was cancelled: {report['reason']} "
                    f"({report['check_before']:.1%} -> {report['check_after']:.1%}).")
        metadata = dict(self.digit_metadata, personal_examples=len(teacher),
                        check_accuracy=report["check_after"])
        self.digit_model.save(os.path.join(self.models_dir, PERSONAL_DIGIT_MODEL), metadata)
        self.digit_metadata = metadata
        self._file_times = self._model_file_times()      # our own save: no need to reload
        probs = self.digit_model.predict_proba(np.asarray(image, dtype=np.float32).reshape(1, 784) / 255.0)[0]
        return (f"Learned! This drawing is now read as **{int(np.argmax(probs))}** "
                f"({probs[label]:.0%} for {label}). Your drawings learned so far: {len(teacher)}, "
                f"read correctly: {report['user_after']:.0%}. "
                f"Safety check on MNIST: {report['check_before']:.1%} -> {report['check_after']:.1%}.")

    def reset_personal_model(self):
        """Goes back to the original model. Your saved drawings are kept."""
        path = os.path.join(self.models_dir, PERSONAL_DIGIT_MODEL)
        if os.path.exists(path):
            os.remove(path)
        self._file_times = None
        self._refresh()
        return "Back to the original model (your saved drawings are kept in data/my_digits.npz)."

    def read_digit(self, image):
        """image: drawing as a numpy array. Returns (probabilities dict, 28x28 image, message)."""
        self._refresh()
        if self.digit_model is None:
            return {}, None, "No digit model yet. Train it first: python train.py"
        if image is None or np.asarray(image).size == 0:
            return {}, None, "Draw a digit first."
        try:
            mnist_image = canvas_to_mnist_format(image)
        except ValueError:
            return {}, None, "The canvas is empty: draw a digit."
        probs = self.digit_model.predict_proba(mnist_image.reshape(1, 784) / 255.0)[0]
        digit = int(np.argmax(probs))
        message = f"Wakil AI reads: {digit} ({probs[digit]:.0%} sure)"
        if probs[digit] < 0.6:
            message += ". Not very sure: try a bigger, centered digit."
        return {str(d): float(p) for d, p in enumerate(probs)}, mnist_image.astype(np.uint8), message

    # -------------------------------------------------------------------- text
    def write(self, prompt, model_name=None, length=300, temperature=0.8, seed=None):
        """Continues `prompt`. Returns (text, message)."""
        self._refresh()
        if not self.text_models:
            return "", "No language model yet. Train one first: python train_lm.py --arch transformer"
        model_name = model_name if model_name in self.text_models else next(iter(self.text_models))
        model, tokenizer, _ = self.text_models[model_name]
        known = "".join(ch for ch in (prompt or "") if ch in tokenizer.stoi)
        message = f"Written by {model_name}, temperature {temperature}."
        if len(known) < len(prompt or ""):
            removed = sorted(set(prompt) - set(known))
            message += f" Removed characters the model never saw: {removed}"
        if not known:
            known = "\n"
        rng = np.random.default_rng(None if seed in (None, "") else int(seed))
        ids = model.generate(tokenizer.encode(known), int(length), rng, float(temperature))
        return tokenizer.decode(ids), message

    # -------------------------------------------------------------------- info
    def info(self):
        """Markdown description of the loaded models."""
        self._refresh()
        lines = ["## Models loaded from `" + self.models_dir + "/`", ""]
        if self.digit_model is None:
            lines.append("- **Digit reader:** not trained yet (`python train.py`)")
        else:
            meta = self.digit_metadata
            line = (f"- **Digit reader:** {meta.get('architecture', self.digit_model.layer_sizes)}, "
                    f"{self.digit_model.num_parameters():,} parameters, "
                    f"validation accuracy {meta.get('val_accuracy', float('nan')):.2%}")
            test_path = os.path.join(self.outputs_dir, "test_metrics.json")
            if os.path.exists(test_path):
                with open(test_path) as f:
                    line += f", **test accuracy {json.load(f)['test_accuracy']:.2%}** (10,000 unseen images)"
            if "autopilot_rounds" in meta:
                line += f" | **improved on its own** by the autopilot ({meta['autopilot_rounds']} rounds)"
            if "personal_examples" in meta:
                line += (f" | **personal version**: learned from {meta['personal_examples']} of your drawings "
                         f"(MNIST check {meta.get('check_accuracy', float('nan')):.1%})")
            lines.append(line)
        for name in TEXT_MODELS:
            if name not in self.text_models:
                lines.append(f"- **{name}:** not trained yet")
                continue
            model, tokenizer, meta = self.text_models[name]
            params = model.num_parameters()
            loss = meta.get("val_loss")
            lines.append(f"- **{name}:** {params:,} parameters, sees {model.context} characters, "
                         f"vocabulary of {tokenizer.vocab_size}"
                         + (f", validation loss {loss:.3f} (perplexity {np.exp(loss):.1f})" if loss else ""))
        lines += ["", "All models are implemented and trained from scratch with NumPy: "
                      "random initial weights, no pretrained model, no external AI API."]
        return "\n".join(lines)
