"""Everything the application needs from the models, with no user-interface code.

app.py (the Gradio interface) only calls this class, so the interface can be
replaced (web page, mobile app, chat bot...) without touching the models.
"""
import json
import os

import numpy as np

from .image import canvas_to_mnist_format
from .language_model import load_language_model
from .model import WakilNet

DIGIT_MODEL = "wakil_ai_mnist.npz"
TEXT_MODELS = {"Transformer (TinyGPT)": "wakil_gpt.npz", "MLP (Part 1)": "wakil_lm.npz"}


class WakilService:

    def __init__(self, models_dir="models", outputs_dir="outputs"):
        self.models_dir, self.outputs_dir = models_dir, outputs_dir
        self.digit_model, self.digit_metadata = None, {}
        path = os.path.join(models_dir, DIGIT_MODEL)
        if os.path.exists(path):
            self.digit_model, self.digit_metadata = WakilNet.load(path)
        self.text_models = {}                           # name -> (model, tokenizer, metadata)
        for name, filename in TEXT_MODELS.items():
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                self.text_models[name] = load_language_model(path)

    # ------------------------------------------------------------------ digits
    def read_digit(self, image):
        """image: drawing as a numpy array. Returns (probabilities dict, 28x28 image, message)."""
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
