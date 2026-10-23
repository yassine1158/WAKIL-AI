"""Generate text with a trained Wakil AI language model (MLP or Transformer).

Examples:
    python generate.py --prompt "ROMEO:"
    python generate.py --prompt "To be" --length 500 --temperature 0.5
    python generate.py --model models/wakil_gpt.npz --prompt "ROMEO:"     # the Transformer
"""
import argparse

import numpy as np

from wakil_ai.language_model import load_language_model


def main():
    parser = argparse.ArgumentParser(description="Generate text with Wakil AI.")
    parser.add_argument("--model", default="models/wakil_lm.npz")
    parser.add_argument("--prompt", default="\n", help="text to continue")
    parser.add_argument("--length", type=int, default=400, help="number of characters to write")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="< 1 safer and more repetitive, > 1 more creative and more mistakes")
    parser.add_argument("--seed", type=int, default=None, help="fix it to get the same text every time")
    args = parser.parse_args()

    model, tokenizer, metadata = load_language_model(args.model)
    prompt_ids = tokenizer.encode(args.prompt.replace("\\n", "\n"))
    ids = model.generate(prompt_ids, args.length, np.random.default_rng(args.seed), args.temperature)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
