import json

import numpy as np

from wakil_ai.intent import DIM, LABELS, export_brain, featurize, fnv1a, load_brain, words
from wakil_ai.model import WakilNet


def test_fnv1a_known_values():
    assert fnv1a("") == 0x811C9DC5
    assert fnv1a("a") == 0xE40C292C
    assert fnv1a("foobar") == 0xBF9CF968


def test_words_normalization():
    assert words("Ça coûte COMBIEN ? 2,5 tonnes!") == ["ca", "coute", "combien", "0", "0", "tonnes"]
    assert words("مرحبا  بك") == ["مرحبا", "بك"]
    assert words("") == [] and words(None) == []


def test_featurize_is_normalized_and_uses_context():
    x = featurize("", "Bonjour")
    assert x.shape == (DIM,) and np.isclose(np.linalg.norm(x), 1.0)
    assert not np.array_equal(featurize("Confirmez-vous ?", "oui"), featurize("", "oui"))
    assert np.array_equal(featurize("", ""), np.zeros(DIM, dtype=np.float32))


def test_export_and_load_roundtrip(tmp_path):
    model = WakilNet((DIM, 8, len(LABELS)), seed=3)
    path = tmp_path / "brain.json"
    probs = model.predict_proba(featurize("", "oui")[None, :])[0]
    export_brain(model, path, {"note": "test"}, [{"context": "", "text": "oui", "probs": probs.tolist()}])
    loaded, data = load_brain(path)
    assert data["labels"] == LABELS and data["metadata"] == {"note": "test"}
    assert np.allclose(loaded.predict_proba(featurize("", "oui")[None, :])[0], data["fixtures"][0]["probs"], atol=1e-6)
    assert json.loads(path.read_text())["format"] == "wakil-brain-1"
