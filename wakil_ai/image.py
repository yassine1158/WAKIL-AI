"""Converting photos and drawings of a handwritten digit into the MNIST format.

MNIST digits are white on black, fit in a 20x20 box, and are centered by their
center of mass inside a 28x28 image. The model only works well on inputs that
look like that, so every image goes through the same steps.
"""
import numpy as np


def center_digit(ink):
    """ink: 2D array, 0 = background, larger = more ink (any size).
    Returns a (28, 28) float32 array in [0, 255], MNIST style."""
    from PIL import Image

    ink = np.asarray(ink, dtype=np.float32)
    if ink.max() <= 0:
        raise ValueError("No digit found in the image.")
    ink = ink * (255.0 / ink.max())
    ink[ink < 50] = 0                          # paper texture, shadows, anti-aliasing noise
    rows, cols = np.where(ink > 0)
    digit = ink[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    digit = np.asarray(Image.fromarray(digit).resize(new_size, Image.LANCZOS))
    digit = np.clip(digit, 0, 255)

    # put the center of mass of the ink at the center of the 28x28 image
    total = digit.sum()
    cy = (np.arange(digit.shape[0])[:, None] * digit).sum() / total
    cx = (np.arange(digit.shape[1])[None, :] * digit).sum() / total
    top = int(np.clip(round(14 - cy), 0, 28 - digit.shape[0]))
    left = int(np.clip(round(14 - cx), 0, 28 - digit.shape[1]))
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return canvas


def image_to_mnist_format(pil_image):
    """Photo or scan (PIL image) -> (28, 28) MNIST-style array, white digit on black."""
    gray = np.asarray(pil_image.convert("L"), dtype=np.float32)
    if gray.mean() > 127:          # dark digit on light paper -> invert
        gray = 255.0 - gray
    return center_digit(gray)


def canvas_to_mnist_format(array):
    """Drawing from a canvas (numpy array, RGBA / RGB / gray) -> (28, 28) MNIST-style array.

    Works with dark strokes on a white or transparent background.
    """
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        gray, alpha = array, np.full(array.shape, 255.0)
    else:
        gray = array[..., :3].mean(axis=-1)
        alpha = array[..., 3] if array.shape[-1] == 4 else np.full(gray.shape, 255.0)
    ink = (alpha / 255.0) * (255.0 - gray)     # dark and opaque = ink
    if ink.max() > 0 and ink.mean() > 127:     # light strokes on a dark background
        ink = (alpha / 255.0) * gray
    return center_digit(ink)
