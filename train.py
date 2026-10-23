"""Train Wakil AI on MNIST.

Examples:
    python train.py
    python train.py --epochs 30 --hidden 256 128 --weight-decay 1e-4 --patience 5
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed

from wakil_ai import TrainConfig, WakilNet, load_prepared_mnist, train
from wakil_ai.plots import plot_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Wakil AI digit classifier (NumPy only).")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128],
                        help="hidden layer sizes, e.g. --hidden 256 128 (default: 128)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization (default: 0)")
    parser.add_argument("--patience", type=int, default=None,
                        help="early stopping patience in epochs (default: off)")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="train on only the first N images (to study overfitting)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/wakil_ai_mnist.npz", help="where to save the model")
    parser.add_argument("--plots-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(learning_rate=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                         weight_decay=args.weight_decay, patience=args.patience, seed=args.seed)

    data = load_prepared_mnist(train_limit=args.train_limit, seed=args.seed)
    print(f"Train: {data['X_train'].shape}  Validation: {data['X_val'].shape}  "
          f"(the test set is NOT used here; run evaluate.py for that)")

    model = WakilNet(layer_sizes=(784, *args.hidden, 10), seed=args.seed)
    print(f"Architecture: {' -> '.join(map(str, model.layer_sizes))} | "
          f"{model.num_parameters():,} parameters | config: {config.to_dict()}")

    start = time.time()
    history = train(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"], config)
    best = history["best_epoch"] - 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    metadata = {
        "name": "Wakil AI - MNIST digit classifier",
        "architecture": " -> ".join(map(str, model.layer_sizes)),
        "parameters": model.num_parameters(),
        "config": config.to_dict(),
        "train_examples": int(len(data["X_train"])),
        "best_epoch": history["best_epoch"],
        "val_loss": history["val_loss"][best],
        "val_accuracy": history["val_acc"][best],
        "training_seconds": round(time.time() - start, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pretrained_weights": False,
    }
    model.save(args.out, metadata)
    history_path = os.path.join(args.plots_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    plot_history(history, os.path.join(args.plots_dir, "training_curves.png"))

    print(f"\nBest validation accuracy: {metadata['val_accuracy']:.2%} (epoch {history['best_epoch']})")
    print(f"Model saved to {args.out}")
    print(f"Training curves saved to {os.path.join(args.plots_dir, 'training_curves.png')}")


if __name__ == "__main__":
    main()
