# Wakil AI

[![tests](https://github.com/yassine1158/WAKIL-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/yassine1158/WAKIL-AI/actions/workflows/tests.yml)

**Neural networks implemented and trained from scratch with NumPy** — no PyTorch, no TensorFlow, no pretrained weights, no AI API.

Wakil AI is a learning project that builds AI step by step, from a single neural network up to a small language model. Every part of the learning process (forward pass, loss, backpropagation, gradient descent) is written by hand and verified by tests.

| Notebook | |
|---|---|
| Phase 1 — neural network from scratch, step by step | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassine1158/WAKIL-AI/blob/main/notebooks/01_phase1_neural_network_from_scratch.ipynb) |
| Phase 2 — the project + overfitting lab | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassine1158/WAKIL-AI/blob/main/notebooks/02_phase2_project_and_overfitting.ipynb) |

| | |
|---|---|
| ✅ Neural network **implemented from scratch** | All the math in NumPy (`wakil_ai/model.py`) |
| ✅ Model **trained from scratch** | Random initial weights, learns only from MNIST |
| ❌ Fine-tuning a pretrained model | Not used |
| ❌ External AI API (ChatGPT, Claude, …) | Not used |

## Results

Digit classifier `784 → 128 (ReLU) → 10 (softmax)`, 101,770 parameters, trained in ~15 s on a CPU:

| Dataset | Images | Accuracy |
|---|---|---|
| Train | 50,000 | 99.50% |
| Validation | 10,000 | 97.74% |
| **Test (never seen during training)** | **10,000** | **97.78%** |

Reproduce with `python train.py && python evaluate.py` (seed 42 gives these exact numbers).

## Quick start

### Option 1 — Google Colab (nothing to install, works on a phone)
Click an **Open in Colab** badge in the table above, then **Runtime → Run all**.

### Option 2 — Your computer (Python 3.9+ and Git)
```bash
git clone https://github.com/yassine1158/WAKIL-AI.git
cd WAKIL-AI
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest              # 15 tests, < 1 second
python train.py               # downloads MNIST (11 MB) once, trains, saves models/wakil_ai_mnist.npz
python evaluate.py            # test accuracy + figures in outputs/
python predict.py --test-index 0 1 2
python predict.py --image my_digit.jpg
```

## Usage

```bash
# Hyperparameters
python train.py --hidden 128 --lr 0.1 --batch-size 64 --epochs 20 --seed 42

# Deeper network, L2 regularization and early stopping
python train.py --hidden 256 128 --weight-decay 1e-4 --patience 5 --epochs 50

# Study overfitting with a small training set
python train.py --train-limit 1000 --epochs 150
```

| Script | What it does | Outputs |
|---|---|---|
| `train.py` | Trains on train set, monitors validation, keeps the best epoch | `models/*.npz`, `outputs/training_curves.png`, `outputs/history.json` |
| `evaluate.py` | Measures the model on the **test set** (only place it is used) | `outputs/test_metrics.json`, confusion matrix, prediction grids |
| `predict.py` | Predicts digits from image files or MNIST test indexes | Printed predictions with top-3 probabilities |

## Project structure

```
WAKIL-AI/
├── wakil_ai/               # the library
│   ├── activations.py        # ReLU, softmax
│   ├── losses.py             # cross-entropy
│   ├── model.py              # WakilNet: init, forward, backward, update, save/load
│   ├── trainer.py            # mini-batch training loop, L2, early stopping
│   ├── data.py               # MNIST download, preprocessing, train/val split
│   ├── image.py              # photo -> MNIST format (28x28, white on black, centered)
│   └── plots.py              # training curves, confusion matrix, prediction grids
├── train.py  evaluate.py  predict.py
├── tests/                    # gradient check + behaviour tests (run by GitHub Actions)
└── notebooks/
    ├── 01_phase1_neural_network_from_scratch.ipynb   # step-by-step tutorial
    └── 02_phase2_project_and_overfitting.ipynb       # using the project + overfitting lab
```

## How it works

For each mini-batch of 64 images `X` with one-hot labels `Y`:

```
Forward      Z1 = X·W1 + b1     A1 = ReLU(Z1)     Z2 = A1·W2 + b2     P = softmax(Z2)
Loss         L  = -mean(Σ Y·log P)  +  (λ/2)·Σ W²
Backward     dZ2 = (P − Y)/m         dW2 = A1ᵀ·dZ2 + λW2     db2 = Σ dZ2
             dZ1 = (dZ2·W2ᵀ) ⊙ ReLU'(Z1)   dW1 = Xᵀ·dZ1 + λW1     db1 = Σ dZ1
Update       W ← W − learning_rate · dW
```

- **Initialization:** He (`√(2/fan_in)`) before ReLU, Xavier/LeCun (`√(1/fan_in)`) before softmax, biases at 0.
- **Correctness:** `tests/test_gradients.py` compares backpropagation with numerical derivatives `(L(w+ε) − L(w−ε)) / 2ε` for every parameter (relative error < 1e-6).
- **Generalization:** the model keeps the epoch with the lowest validation loss; the test set is only used by `evaluate.py`.

## Overfitting, in numbers

Same network trained on only **1,000** images for 150 epochs (from the Phase 2 notebook):

| Setup | Train acc. | Val loss | Val acc. |
|---|---|---|---|
| 1,000 images | 100% | 0.415 (best, epoch 33) → 0.505 (end) | 87.5% |
| 1,000 images + L2 (λ = 0.01) | 99.5% | 0.400 | 88.1% |
| 1,000 images + early stopping | — | 0.415 (stopped at epoch 43) | 87.5% |
| **50,000 images** | 99.5% | **0.080** | **97.7%** |

With little data the model memorizes (100% train) and validation loss climbs back up. Regularization and early stopping help a little; **more data helps the most**.

## Limitations

- Only recognizes single, centered digits in MNIST style (white on black). Photos of real handwriting are a different distribution and get lower accuracy.
- A fully connected network ignores the 2D structure of images; convolutional networks reach > 99% on MNIST.
- NumPy runs on the CPU only. Fine for this size, far too slow for large models.
- This is an educational project. Large language models such as Claude or ChatGPT are millions of times bigger and need huge datasets, GPU clusters, teams and budgets.

## Roadmap

- [x] **Phase 1** — neural network from scratch, trained on MNIST (notebook)
- [x] **Phase 2** — clean project: package, CLI scripts, tests, CI, overfitting study
- [ ] **Phase 3** — tiny character-level language model, then a small Transformer (attention)
- [ ] **Phase 4** — a simple app to use the trained models

## License

MIT — see [LICENSE](LICENSE). MNIST is by Yann LeCun, Corinna Cortes and Christopher Burges.
