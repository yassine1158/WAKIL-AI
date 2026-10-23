"""Converting a photo of a handwritten digit into the MNIST format."""
import numpy as np


def image_to_mnist_format(pil_image):
    """PIL image -> (28, 28) float32 array, white digit on black, values 0-255.

    Steps: grayscale -> invert if the background is light -> remove faint noise
    -> crop to the digit -> resize so the longest side is 20 px -> center in 28x28.
    """
    from PIL import Image

    gray = np.asarray(pil_image.convert("L"), dtype=np.float32)
    if gray.mean() > 127:          # dark digit on light paper -> invert
        gray = 255.0 - gray
    gray[gray < 50] = 0            # paper texture, shadows
    rows, cols = np.where(gray > 0)
    if len(rows) == 0:
        raise ValueError("No digit found in the image.")
    digit = gray[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    digit = np.asarray(Image.fromarray(digit).resize(new_size, Image.LANCZOS))
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - digit.shape[0]) // 2, (28 - digit.shape[1]) // 2
    canvas[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return np.clip(canvas, 0, 255)
