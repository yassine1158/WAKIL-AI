"""Wakil Brain: a small intent model, built from scratch, for the Wakil WhatsApp agent.

Given the agent's last message (context) and the customer's new message, it predicts
what the agent should do next:

    reponse   - just answer (question, advice, price, greeting...)
    humain    - hand over to a human advisor (complaint, negotiation, asks for a person...)
    commande  - the customer confirms the order recap -> record the order
    photo     - the customer wants to see a product -> send its photo

Text -> numbers uses the "hashing trick": every word and every 3-letter piece of a
word is hashed (FNV-1a) into one of DIM slots, so there is no vocabulary to store and
unknown or misspelled words still share pieces with known ones. The exact same
featurizer is re-implemented in TypeScript (src/brain.ts in the WhatsApp project);
the exported file contains test cases so both sides can be checked against each other.
"""
import base64
import json
import re
import unicodedata

import numpy as np

LABELS = ["reponse", "humain", "commande", "photo"]
DIM = 2048


def fnv1a(text):
    """32-bit FNV-1a hash of the UTF-8 bytes of `text`."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def words(text):
    """Lowercase, remove accents, every digit -> 0, keep only letters and digits."""
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[0-9]", "0", text)
    return "".join(ch if ch.isalnum() else " " for ch in text).split()


def feature_names(context, text):
    names = []
    for prefix, segment in (("a", context), ("u", text)):
        for word in words(segment):
            names.append(f"{prefix}w:{word}")
            padded = list(f"<{word}>")
            names += [f"{prefix}c:{''.join(padded[i:i + 3])}" for i in range(len(padded) - 2)]
    return names


def featurize(context, text, dim=DIM):
    """(agent's last message, customer's message) -> L2-normalized vector of size dim."""
    x = np.zeros(dim, dtype=np.float32)
    for name in feature_names(context, text):
        x[fnv1a(name) % dim] += 1.0
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def export_brain(model, path, metadata, fixtures):
    """Saves the model as JSON for the TypeScript runtime. Weights: base64 float32 (little-endian)."""
    def encode(array):
        return base64.b64encode(np.asarray(array, dtype="<f4").tobytes()).decode("ascii")

    data = {"format": "wakil-brain-1", "labels": LABELS, "dim": DIM, "layer_sizes": list(model.layer_sizes),
            "params": {name: {"shape": list(value.shape), "data": encode(value)} for name, value in model.params.items()},
            "metadata": metadata, "fixtures": fixtures}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_brain(path):
    """Reads an exported file back into a WakilNet (used by tests and to continue training)."""
    from .model import WakilNet
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    model = WakilNet(tuple(data["layer_sizes"]))
    model.set_state({name: np.frombuffer(base64.b64decode(p["data"]), dtype="<f4").reshape(p["shape"]).copy()
                     for name, p in data["params"].items()})
    return model, data
