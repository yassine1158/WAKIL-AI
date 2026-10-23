"""Matplotlib figures for training curves, confusion matrices and predictions."""
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, path=None, title="Yassine AI training"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)

    batch_loss = np.array(history["batch_loss"])
    window = min(100, max(1, len(batch_loss) // 10))
    smoothed = np.convolve(batch_loss, np.ones(window) / window, mode="valid")
    axes[0].plot(batch_loss, alpha=0.25, label="batch loss")
    axes[0].plot(np.arange(window - 1, len(batch_loss)), smoothed, label=f"moving average ({window})")
    axes[0].set(xlabel="update step", ylabel="loss", title="Loss during training", ylim=(0, 1.0))
    axes[0].legend()

    for ax, key, name in [(axes[1], "loss", "Loss"), (axes[2], "acc", "Accuracy")]:
        ax.plot(epochs, history[f"train_{key}"], "o-", markersize=3, label="train")
        ax.plot(epochs, history[f"val_{key}"], "o-", markersize=3, label="validation")
        if history.get("best_epoch"):
            ax.axvline(history["best_epoch"], color="gray", linestyle="--", label="best epoch")
        ax.set(xlabel="epoch", ylabel=name.lower(), title=f"{name} per epoch")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path)


def plot_confusion_matrix(cm, path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(cm.shape[1]), yticks=range(cm.shape[0]),
           xlabel="predicted digit", ylabel="true digit", title="Confusion matrix (test set)")
    fig.colorbar(im)
    fig.tight_layout()
    return _finish(fig, path)


def plot_predictions(images, labels, probs, path=None, title="Predictions"):
    preds = probs.argmax(axis=1)
    fig, axes = plt.subplots(1, len(images), figsize=(1.6 * len(images), 2.3), squeeze=False)
    for ax, image, label, pred, p in zip(axes[0], images, labels, preds, probs):
        ax.imshow(image.reshape(28, 28), cmap="gray")
        ax.set_title(f"pred {pred} ({p[pred]:.0%})\ntrue {label}",
                     color="green" if pred == label else "red", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, path)


def _finish(fig, path):
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig
