"""Train the Wakil AI character-level language model.

Examples:
    python train_lm.py
    python train_lm.py --steps 40000 --context 16 --hidden 512
    python train_lm.py --text my_french_book.txt --out models/wakil_lm_fr.npz
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wakil_ai.language_model import BigramCounts, CharMLP, estimate_loss, train_language_model
from wakil_ai.text_data import load_text, split_ids
from wakil_ai.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a character-level language model (NumPy only).")
    parser.add_argument("--text", default=None, help="UTF-8 text file (default: Tiny Shakespeare)")
    parser.add_argument("--context", type=int, default=8, help="characters the model sees (default: 8)")
    parser.add_argument("--emb-dim", type=int, default=16, help="embedding size per character")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/wakil_lm.npz")
    parser.add_argument("--plots-dir", default="outputs")
    args = parser.parse_args()

    text = load_text(args.text)
    tokenizer = CharTokenizer.from_text(text)
    train_ids, val_ids = split_ids(tokenizer.encode(text))
    print(f"Text: {len(text):,} characters | vocabulary: {tokenizer.vocab_size} characters "
          f"| train {len(train_ids):,} / validation {len(val_ids):,}")

    bigram = BigramCounts(tokenizer.vocab_size).fit(train_ids)
    print(f"Baselines (validation loss): uniform guess {np.log(tokenizer.vocab_size):.4f} "
          f"| bigram counts {bigram.loss(val_ids):.4f}")

    model = CharMLP(tokenizer.vocab_size, context=args.context, emb_dim=args.emb_dim,
                    hidden=args.hidden, seed=args.seed)
    print(f"CharMLP: context {args.context}, embeddings {args.emb_dim}, hidden {args.hidden} "
          f"| {model.num_parameters():,} parameters")

    start = time.time()
    history = train_language_model(model, train_ids, val_ids, steps=args.steps, batch_size=args.batch_size,
                                   learning_rate=args.lr, seed=args.seed)
    val_loss = estimate_loss(model, val_ids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    metadata = {"name": "Wakil AI - character language model", "config": model.config,
                "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
                "seed": args.seed, "val_loss": val_loss, "bigram_val_loss": bigram.loss(val_ids),
                "training_seconds": round(time.time() - start, 1), "text": args.text or "tinyshakespeare",
                "pretrained_weights": False}
    model.save(args.out, tokenizer, metadata)
    with open(os.path.join(args.plots_dir, "lm_history.json"), "w") as f:
        json.dump(history, f)

    plt.figure(figsize=(8, 4))
    batch = np.array(history["batch_loss"])
    plt.plot(np.convolve(batch, np.ones(200) / 200, mode="valid"), alpha=0.6, label="batch loss (moving avg)")
    plt.plot(history["step"], history["train_loss"], "o-", label="train")
    plt.plot(history["step"], history["val_loss"], "o-", label="validation")
    plt.axhline(bigram.loss(val_ids), color="gray", linestyle="--", label="bigram baseline")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Wakil AI language model")
    plt.legend()
    plt.savefig(os.path.join(args.plots_dir, "lm_training_curves.png"), dpi=110, bbox_inches="tight")

    sample = tokenizer.decode(model.generate(tokenizer.encode(text[:1]), 300, np.random.default_rng(args.seed),
                                             temperature=0.8))
    print(f"\nValidation loss: {val_loss:.4f} (bigram: {bigram.loss(val_ids):.4f})")
    print(f"Model saved to {args.out}\n\n--- Sample (temperature 0.8) ---\n{sample}")


if __name__ == "__main__":
    main()
