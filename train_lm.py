"""Train a Wakil AI character-level language model.

Examples:
    python train_lm.py                                   # MLP with embeddings (Phase 3, part 1)
    python train_lm.py --arch transformer                # small Transformer (Phase 3, part 2)
    python train_lm.py --arch transformer --layers 4 --d-model 128 --steps 10000
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

from wakil_ai.language_model import BigramCounts, CharMLP, train_language_model
from wakil_ai.language_model import estimate_loss as estimate_mlp_loss
from wakil_ai.text_data import load_text, split_ids
from wakil_ai.tokenizer import CharTokenizer
from wakil_ai.transformer import TinyGPT, train_transformer
from wakil_ai.transformer import estimate_loss as estimate_transformer_loss

DEFAULTS = {  # per architecture: steps, batch size, learning rate, output file
    "mlp": {"steps": 20000, "batch_size": 128, "lr": 0.1, "out": "models/wakil_lm.npz"},
    "transformer": {"steps": 5000, "batch_size": 32, "lr": 3e-3, "out": "models/wakil_gpt.npz"},
}


def main():
    parser = argparse.ArgumentParser(description="Train a character-level language model (NumPy only).")
    parser.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--text", default=None, help="UTF-8 text file (default: Tiny Shakespeare)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="default: 0.1 (mlp, SGD) / 0.003 (transformer, Adam)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--plots-dir", default="outputs")
    mlp = parser.add_argument_group("mlp options")
    mlp.add_argument("--context", type=int, default=8, help="characters the MLP sees (default: 8)")
    mlp.add_argument("--emb-dim", type=int, default=16)
    mlp.add_argument("--hidden", type=int, default=256)
    gpt = parser.add_argument_group("transformer options")
    gpt.add_argument("--block-size", type=int, default=32, help="characters the Transformer sees (default: 32)")
    gpt.add_argument("--d-model", type=int, default=64)
    gpt.add_argument("--heads", type=int, default=4)
    gpt.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    for key, value in DEFAULTS[args.arch].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    text = load_text(args.text)
    tokenizer = CharTokenizer.from_text(text)
    train_ids, val_ids = split_ids(tokenizer.encode(text))
    print(f"Text: {len(text):,} characters | vocabulary: {tokenizer.vocab_size} characters "
          f"| train {len(train_ids):,} / validation {len(val_ids):,}")
    bigram_loss = BigramCounts(tokenizer.vocab_size).fit(train_ids).loss(val_ids)
    print(f"Baselines (validation loss): uniform guess {np.log(tokenizer.vocab_size):.4f} "
          f"| bigram counts {bigram_loss:.4f}")

    start = time.time()
    if args.arch == "mlp":
        model = CharMLP(tokenizer.vocab_size, context=args.context, emb_dim=args.emb_dim,
                        hidden=args.hidden, seed=args.seed)
        print(f"CharMLP {model.config} | {model.num_parameters():,} parameters")
        history = train_language_model(model, train_ids, val_ids, steps=args.steps, batch_size=args.batch_size,
                                       learning_rate=args.lr, seed=args.seed)
        val_loss = estimate_mlp_loss(model, val_ids)
    else:
        model = TinyGPT(tokenizer.vocab_size, block_size=args.block_size, d_model=args.d_model,
                        n_heads=args.heads, n_layers=args.layers, seed=args.seed)
        print(f"TinyGPT {model.config} | {model.num_parameters():,} parameters")
        history = train_transformer(model, train_ids, val_ids, steps=args.steps, batch_size=args.batch_size,
                                    learning_rate=args.lr, seed=args.seed)
        val_loss = estimate_transformer_loss(model, val_ids)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    metadata = {"name": f"Wakil AI - character language model ({args.arch})", "arch": args.arch,
                "config": model.config, "steps": args.steps, "batch_size": args.batch_size,
                "learning_rate": args.lr, "seed": args.seed, "val_loss": val_loss, "bigram_val_loss": bigram_loss,
                "training_seconds": round(time.time() - start, 1), "text": args.text or "tinyshakespeare",
                "pretrained_weights": False}
    model.save(args.out, tokenizer, metadata)
    with open(os.path.join(args.plots_dir, f"lm_{args.arch}_history.json"), "w") as f:
        json.dump(history, f)

    plt.figure(figsize=(8, 4))
    batch = np.array(history["batch_loss"])
    window = max(1, len(batch) // 100)
    plt.plot(np.convolve(batch, np.ones(window) / window, mode="valid"), alpha=0.6, label="batch loss (moving avg)")
    plt.plot(history["step"], history["train_loss"], "o-", label="train")
    plt.plot(history["step"], history["val_loss"], "o-", label="validation")
    plt.axhline(bigram_loss, color="gray", linestyle="--", label="bigram baseline")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"Wakil AI language model ({args.arch})")
    plt.legend()
    plot_path = os.path.join(args.plots_dir, f"lm_{args.arch}_training_curves.png")
    plt.savefig(plot_path, dpi=110, bbox_inches="tight")

    sample = tokenizer.decode(model.generate(tokenizer.encode(text[:1]), 300, np.random.default_rng(args.seed),
                                             temperature=0.8))
    print(f"\nValidation loss: {val_loss:.4f} (bigram: {bigram_loss:.4f})")
    print(f"Model saved to {args.out} | curves: {plot_path}\n\n--- Sample (temperature 0.8) ---\n{sample}")


if __name__ == "__main__":
    main()
