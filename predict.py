"""Predict the digit in an image with a trained Yassine AI model.

Examples:
    python predict.py --image my_digit.png
    python predict.py --test-index 0          # an image from the MNIST test set
"""
import argparse

import numpy as np

from yassine_ai import YassineNet


def predict_digit(model, image):
    """image: 784 values (28x28), white digit on black, 0-255 or 0-1.
    Returns (digit, probabilities for 0-9)."""
    x = np.asarray(image, dtype=np.float32).reshape(1, 784)
    if x.max() > 1.0:
        x = x / 255.0
    probs = model.predict_proba(x)[0]
    return int(np.argmax(probs)), probs


def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit with Yassine AI.")
    parser.add_argument("--model", default="models/yassine_ai_mnist.npz")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", nargs="+", help="path(s) to image files (photo or scan of a digit)")
    source.add_argument("--test-index", type=int, nargs="+", help="index(es) in the MNIST test set (0-9999)")
    args = parser.parse_args()

    model, _ = YassineNet.load(args.model)

    if args.image:
        from PIL import Image
        from yassine_ai.image import image_to_mnist_format
        inputs = [(path, image_to_mnist_format(Image.open(path)), None) for path in args.image]
    else:
        from yassine_ai.data import load_mnist
        _, _, x_test, y_test = load_mnist()
        inputs = [(f"test image #{i}", x_test[i], int(y_test[i])) for i in args.test_index]

    for name, image, true_label in inputs:
        digit, probs = predict_digit(model, image)
        top3 = ", ".join(f"{d}: {probs[d]:.1%}" for d in np.argsort(probs)[::-1][:3])
        truth = "" if true_label is None else f" | true: {true_label} {'OK' if digit == true_label else 'WRONG'}"
        print(f"{name}: predicted {digit} ({probs[digit]:.1%}){truth} | top 3 -> {top3}")


if __name__ == "__main__":
    main()
