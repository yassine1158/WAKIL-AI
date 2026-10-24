"""Wakil AI autopilot: keeps improving the models on its own, round after round.

    python autolearn.py                 # runs until you press Ctrl+C
    python autolearn.py --rounds 5      # 5 rounds then stops
    python autolearn.py --only digits   # or: --only text

Nothing to do while it runs. A model file is only replaced when the new version
is better on held-out data. If the app (app.py) is running, it picks up the
improved models automatically.
"""
import argparse
import csv
import os
import time

import numpy as np

from wakil_ai.autopilot import DigitAutopilot, TextAutopilot, load_state, save_state
from wakil_ai.data import load_prepared_mnist
from wakil_ai.language_model import load_language_model
from wakil_ai.model import WakilNet
from wakil_ai.text_data import load_text, split_ids

STATE_PATH = "outputs/autolearn_state.json"
LOG_PATH = "outputs/autolearn_log.csv"


def active_file(*names):
    """The first model file that exists (your personal digit model first, the Transformer first)."""
    for name in names:
        path = os.path.join("models", name)
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Wakil AI autopilot")
    parser.add_argument("--rounds", type=int, default=None, help="number of rounds (default: until Ctrl+C)")
    parser.add_argument("--only", choices=["digits", "text"], default=None)
    parser.add_argument("--digit-steps", type=int, default=1500)
    parser.add_argument("--text-steps", type=int, default=1000)
    args = parser.parse_args()
    state = load_state(STATE_PATH)
    pilots = []

    if args.only in (None, "digits"):
        path = active_file("wakil_ai_mnist_personal.npz", "wakil_ai_mnist.npz")
        if path is None:
            print("No digit model: run python train.py first.")
        else:
            data = load_prepared_mnist()
            extra_X = extra_y = None
            if os.path.exists("data/my_digits.npz"):
                with np.load("data/my_digits.npz") as mine:
                    extra_X, extra_y = mine["X"], mine["y"]
            model, metadata = WakilNet.load(path)
            print(f"Digits: improving {path} with examples it creates itself...")
            pilots.append(("digits", DigitAutopilot(model, path, data["X_train"], data["y_train"],
                                                    data["X_val"], data["y_val"], extra_X, extra_y,
                                                    state=state.setdefault("digits", {}), metadata=metadata)))

    if args.only in (None, "text"):
        path = active_file("wakil_gpt.npz", "wakil_lm.npz")
        if path is None:
            print("No language model: run python train_lm.py first.")
        else:
            model, tokenizer, metadata = load_language_model(path)
            train_ids, val_ids = split_ids(tokenizer.encode(load_text()))
            print(f"Text: improving {path} by reading its text again...")
            pilots.append(("text", TextAutopilot(model, tokenizer, path, train_ids, val_ids,
                                                 state=state.setdefault(f"text:{os.path.basename(path)}", {}),
                                                 metadata=metadata)))

    if not pilots:
        return
    save_state(STATE_PATH, state)
    new_log = not os.path.exists(LOG_PATH)
    print("Autopilot running. Press Ctrl+C to stop (your best models are always kept).\n")
    round_number = 0
    try:
        while args.rounds is None or round_number < args.rounds:
            round_number += 1
            for name, pilot in pilots:
                start = time.time()
                if name == "digits":
                    report = pilot.run_round(steps=args.digit_steps)
                    summary = (f"normal digits {report['normal_accuracy']:.2%} | hard digits "
                               f"{report['hard_accuracy']:.2%} (best {pilot.state['best_hard_accuracy']:.2%})")
                else:
                    report = pilot.run_round(steps=args.text_steps)
                    summary = f"val loss {report['val_loss']:.4f} (best {pilot.state['best_val_loss']:.4f})"
                status = ("SAVED (better)" if report["saved"]
                          else "not saved yet (it saves only when better AND nothing is forgotten)")
                print(f"[round {round_number}] {name:<6} | {summary} | {status} | {time.time() - start:.0f}s")
                save_state(STATE_PATH, state)
                with open(LOG_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    if new_log:
                        writer.writerow(["time", "round", "model", "summary", "saved"])
                        new_log = False
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), round_number, name, summary, report["saved"]])
    except KeyboardInterrupt:
        print("\nStopped. The best models are saved in models/.")


if __name__ == "__main__":
    main()
