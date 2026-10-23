"""Evaluate a trained model on the 10,000 unseen MNIST test images.

Example:
    python evaluate.py --model models/wakil_ai_mnist.npz
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from wakil_ai import WakilNet, evaluate, load_prepared_mnist
from wakil_ai.plots import plot_confusion_matrix, plot_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wakil AI on the MNIST test set.")
    parser.add_argument("--model", default="models/wakil_ai_mnist.npz")
    parser.add_argument("--plots-dir", default="outputs")
    args = parser.parse_args()

    model, metadata = WakilNet.load(args.model)
    data = load_prepared_mnist()
    X_test, y_test = data["X_test"], data["y_test"]

    test_loss, test_acc = evaluate(model, X_test, y_test)
    probs = model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    wrong = np.where(y_pred != y_test)[0]

    print(f"Model: {metadata.get('architecture', model.layer_sizes)} "
          f"(best val accuracy {metadata.get('val_accuracy', float('nan')):.2%})")
    print(f"TEST loss:     {test_loss:.4f}")
    print(f"TEST accuracy: {test_acc:.2%}  ({len(y_test) - len(wrong)} / {len(y_test)} correct)\n")
    print(classification_report(y_test, y_pred, digits=4))

    os.makedirs(args.plots_dir, exist_ok=True)
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), os.path.join(args.plots_dir, "confusion_matrix.png"))
    sample = np.random.default_rng(7).choice(len(y_test), 12, replace=False)
    plot_predictions(data["test_images"][sample], y_test[sample], probs[sample],
                     os.path.join(args.plots_dir, "test_predictions.png"), "Random unseen test images")
    plot_predictions(data["test_images"][wrong[:12]], y_test[wrong[:12]], probs[wrong[:12]],
                     os.path.join(args.plots_dir, "test_mistakes.png"), "Some test images the model got wrong")

    results = {"test_loss": test_loss, "test_accuracy": test_acc, "test_errors": int(len(wrong)),
               "model": args.model, "architecture": metadata.get("architecture")}
    with open(os.path.join(args.plots_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Figures and test_metrics.json saved in {args.plots_dir}/")


if __name__ == "__main__":
    main()
