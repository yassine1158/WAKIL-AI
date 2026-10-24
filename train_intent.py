"""Train Wakil Brain (intent model) and export it for the Wakil WhatsApp agent.

    python train_intent.py                                  # synthetic seed examples only
    python train_intent.py --teacher brain-samples.json     # + real examples labelled by Claude
    python train_intent.py --out ../projet_wakil_whatsapp_buss/src/brain-model.json

`brain-samples.json` is the export of the WhatsApp app (Wakil Brain page, admin only):
each real message with the action Claude actually took. Claude is the teacher,
Wakil Brain the student ("distillation").
"""
import argparse
import json
import os
import time

import numpy as np

from wakil_ai.intent import LABELS, export_brain, featurize
from wakil_ai.model import WakilNet
from wakil_ai.trainer import TrainConfig, evaluate, train

FIXTURES = [
    ("Récapitulatif : 2 tonnes de sable, livraison à Cocody. Confirmez-vous ?", "Oui je confirme 👍"),
    ("", "Je veux parler à un conseiller"),
    ("", "Vous avez une photo du gravier 5/15 ?"),
    ("", "C'est combien le sable de mer ? Livrez-vous à Yopougon"),
    ("Bonjour !", "مرحبا، هل لديكم رمل؟"),
    ("", ""),
]


def read_examples(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    rows = data["samples"] if isinstance(data, dict) else data
    # export of the WhatsApp app: the label is what Claude did ("teacher")
    return [{"context": r.get("context", ""), "text": r["text"], "label": r["teacher"], "source": "claude"}
            for r in rows if r.get("teacher") in LABELS and r.get("text")]


def main():
    parser = argparse.ArgumentParser(description="Train Wakil Brain (intent model)")
    parser.add_argument("--seed-data", default="datasets/intents_seed.jsonl")
    parser.add_argument("--teacher", default=None, help="export of real samples labelled by Claude (JSON)")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/wakil_brain.json")
    args = parser.parse_args()

    rows = read_examples(args.seed_data)
    real = read_examples(args.teacher) if args.teacher else []
    print(f"Examples: {len(rows)} synthetic + {len(real)} real (labelled by Claude)")
    rng = np.random.default_rng(args.seed)

    # Validation: real examples when we have enough of them (that's what matters), else synthetic.
    if len(real) >= 50:
        order = rng.permutation(len(real))
        n_val = max(20, len(real) // 5)
        val_rows = [real[i] for i in order[:n_val]]
        train_rows = rows + [real[i] for i in order[n_val:]] * 3     # real examples count 3x more
        val_name = "real (Claude-labelled)"
    else:
        order = rng.permutation(len(rows))
        n_val = len(rows) // 5
        val_rows = [rows[i] for i in order[:n_val]]
        train_rows = [rows[i] for i in order[n_val:]] + real * 3
        val_name = "synthetic (optimistic!)"

    def matrix(subset):
        X = np.stack([featurize(r.get("context", ""), r["text"]) for r in subset])
        y = np.array([LABELS.index(r["label"]) for r in subset])
        return X, y

    X_train, y_train = matrix(train_rows)
    X_val, y_val = matrix(val_rows)
    model = WakilNet((X_train.shape[1], args.hidden, len(LABELS)), seed=args.seed)
    print(f"Wakil Brain: {' -> '.join(map(str, model.layer_sizes))} | {model.num_parameters():,} parameters")
    history = train(model, X_train, y_train, X_val, y_val,
                    TrainConfig(learning_rate=args.lr, batch_size=32, epochs=args.epochs, seed=args.seed),
                    verbose=False)
    val_loss, val_acc = evaluate(model, X_val, y_val)
    print(f"Validation on {val_name} examples: accuracy {val_acc:.1%} (best epoch {history['best_epoch']})")

    predictions = model.predict(X_val)
    for i, label in enumerate(LABELS):
        mask = y_val == i
        if mask.any():
            print(f"  {label:<9} {np.mean(predictions[mask] == i):.0%} of {mask.sum()} examples")

    fixtures = []
    for context, text in FIXTURES:
        probs = model.predict_proba(featurize(context, text)[None, :])[0]
        fixtures.append({"context": context, "text": text, "probs": [round(float(p), 6) for p in probs]})
        print(f"  {text!r:<55} -> {LABELS[int(np.argmax(probs))]} ({probs.max():.0%})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    metadata = {"name": "Wakil Brain - intent model", "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "synthetic_examples": len(rows), "real_examples": len(real), "validation": val_name,
                "val_accuracy": round(val_acc, 4), "pretrained_weights": False}
    export_brain(model, args.out, metadata, fixtures)
    print(f"Exported to {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
