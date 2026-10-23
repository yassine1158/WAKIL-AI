# Wakil AI

[![tests](https://github.com/yassine1158/WAKIL-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/yassine1158/WAKIL-AI/actions/workflows/tests.yml)

**Neural networks implemented and trained from scratch with NumPy** — no PyTorch, no TensorFlow, no pretrained weights, no AI API.

Wakil AI is a learning project that builds AI step by step, from a single neural network up to a small language model. Every part of the learning process (forward pass, loss, backpropagation, gradient descent) is written by hand and verified by tests.

| Notebook | |
|---|---|
| Phase 1 — neural network from scratch, step by step | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassine1158/WAKIL-AI/blob/main/notebooks/01_phase1_neural_network_from_scratch.ipynb) |
| Phase 2 — the project + overfitting lab | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassine1158/WAKIL-AI/blob/main/notebooks/02_phase2_project_and_overfitting.ipynb) |
| Phase 3 — a character-level language model (tokenizer, embeddings, generation) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassine1158/WAKIL-AI/blob/main/notebooks/03_phase3_character_language_model.ipynb) |
| Phase 3, part 2 — a Transformer from scratch (self-attention, LayerNorm, Adam) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassine1158/WAKIL-AI/blob/main/notebooks/04_phase3_transformer_from_scratch.ipynb) |
| Phase 4 — the Wakil AI app: draw a digit, generate text (public link from Colab) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassine1158/WAKIL-AI/blob/main/notebooks/05_phase4_app.ipynb) |

| | |
|---|---|
| ✅ Neural network **implemented from scratch** | All the math in NumPy (`wakil_ai/model.py`) |
| ✅ Model **trained from scratch** | Random initial weights, learns only from MNIST |
| ❌ Fine-tuning a pretrained model | Not used |
| ❌ External AI API (ChatGPT, Claude, …) | Not used |

## The app

```bash
pip install -r requirements-app.txt     # adds Gradio, used only for the interface
python app.py                           # http://127.0.0.1:7860   (python app.py --share for a public link)
```

| Tab | What it does |
|---|---|
| ✍️ Read a digit | Draw a digit with the mouse (click 🖌️ Brush first); see the prediction, the probabilities, and the 28×28 image the model actually sees |
| 💬 Write text | Type a beginning; pick the model (Transformer or MLP), length and temperature; Wakil AI continues it |
| ℹ️ Models | Size, accuracy and loss of the trained models |

The interface (`app.py`) only calls `wakil_ai/service.py`, which holds the application logic and is tested without any UI (`tests/test_service.py`). Drawings go through the same preprocessing as MNIST (crop, 20×20 box, centered by center of mass); on MNIST test digits rendered as 280×280 drawings, this pipeline keeps 98.1% accuracy. Real mouse drawings look different from MNIST, so expect more mistakes on them.

## Results

### Digit classifier

`784 → 128 (ReLU) → 10 (softmax)`, 101,770 parameters, trained in ~15 s on a CPU:

| Dataset | Images | Accuracy |
|---|---|---|
| Train | 50,000 | 99.50% |
| Validation | 10,000 | 97.74% |
| **Test (never seen during training)** | **10,000** | **97.78%** |

Reproduce with `python train.py && python evaluate.py` (seed 42 gives the same results on every run; on another computer the last decimals can differ slightly, e.g. 97.76% instead of 97.78% on Windows).

### Language models

Character-level models trained on *Tiny Shakespeare* (1.1M characters, 65-character vocabulary) to predict the next character:

| Model | Sees | Parameters | CPU time | Validation loss | Perplexity |
|---|---|---|---|---|---|
| Uniform guess | — | — | — | 4.17 | 65 |
| Bigram (counting pairs) | 1 char | 4,225 | < 1 s | 2.48 | 12.0 |
| MLP + embeddings (`--arch mlp`) | 8 chars | 50,769 | ~20 s | 1.86 | 6.4 |
| **TinyGPT Transformer** (`--arch transformer`) | **32 chars** | **110,529** | ~3.5 min | **1.72** | **5.6** |

TinyGPT: token + position embeddings (64) → 2 blocks of [LayerNorm → 4-head causal self-attention → residual, LayerNorm → MLP 64→256→64 → residual] → LayerNorm → 65 logits, trained with Adam. Its loss is averaged over all 32 positions, including the first ones that see only 1–2 characters.

Sample from the Transformer (`python generate.py --model models/wakil_gpt.npz --prompt "ROMEO:" --seed 1`, temperature 0.8):

```
ROMEO:
Wenter of the king these of your no monther.

PAULINA:
Why, And heavy my lady, one thing the
boing smoth pexece: the requiling heart.
```

The models have learned the shape of a play (names in capitals, line breaks, punctuation, many real words) but still invent words and make no sense over a sentence. With ~100k parameters and a few CPU minutes, that is the expected level.

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

python -m pytest              # 39 tests, ~2 seconds
python train.py               # downloads MNIST (11 MB) once, trains, saves models/wakil_ai_mnist.npz
python evaluate.py            # test accuracy + figures in outputs/
python predict.py --test-index 0 1 2
python predict.py --image my_digit.jpg

python train_lm.py            # downloads Tiny Shakespeare (1 MB) once, trains the MLP language model (~20 s)
python generate.py --prompt "ROMEO:"

python train_lm.py --arch transformer                  # TinyGPT (~3-8 min on a CPU)
python generate.py --model models/wakil_gpt.npz --prompt "ROMEO:"

pip install -r requirements-app.txt
python app.py                 # the Wakil AI app in your browser
```

## Usage

```bash
# Hyperparameters
python train.py --hidden 128 --lr 0.1 --batch-size 64 --epochs 20 --seed 42

# Deeper network, L2 regularization and early stopping
python train.py --hidden 256 128 --weight-decay 1e-4 --patience 5 --epochs 50

# Study overfitting with a small training set
python train.py --train-limit 1000 --epochs 150

# Language model: longer context, bigger network, your own text
python train_lm.py --context 16 --hidden 512 --steps 40000
python train_lm.py --arch transformer --layers 4 --d-model 128 --block-size 64 --steps 10000
python train_lm.py --text my_book.txt --out models/wakil_lm_book.npz
python generate.py --model models/wakil_lm.npz --prompt "To be" --length 500 --temperature 0.5
```

| Script | What it does | Outputs |
|---|---|---|
| `train.py` | Trains on train set, monitors validation, keeps the best epoch | `models/*.npz`, `outputs/training_curves.png`, `outputs/history.json` |
| `evaluate.py` | Measures the model on the **test set** (only place it is used) | `outputs/test_metrics.json`, confusion matrix, prediction grids |
| `predict.py` | Predicts digits from image files or MNIST test indexes | Printed predictions with top-3 probabilities |
| `train_lm.py` | Trains a character language model (`--arch mlp` or `transformer`), compares it with a bigram baseline | `models/wakil_lm.npz` or `models/wakil_gpt.npz`, training curves, a text sample |
| `generate.py` | Writes text from a prompt with either model; `--temperature` controls creativity | Generated text |
| `app.py` | The Wakil AI web app (Gradio) using all trained models | Runs at http://127.0.0.1:7860 |

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
│   ├── plots.py              # training curves, confusion matrix, prediction grids
│   ├── tokenizer.py          # character tokenizer: text <-> ids
│   ├── text_data.py          # text download, contiguous split, random context windows
│   ├── language_model.py     # bigram baseline, CharMLP with embeddings, training, generation
│   ├── transformer.py        # TinyGPT: attention, LayerNorm, blocks, Adam (forward + backward)
│   └── service.py            # app logic: load models, read a drawing, write text (no UI code)
├── train.py  evaluate.py  predict.py
├── train_lm.py  generate.py
├── app.py                    # the interface only (Gradio)
├── tests/                    # gradient checks + behaviour tests (run by GitHub Actions)
└── notebooks/
    ├── 01_phase1_neural_network_from_scratch.ipynb   # step-by-step tutorial
    ├── 02_phase2_project_and_overfitting.ipynb       # using the project + overfitting lab
    ├── 03_phase3_character_language_model.ipynb      # tokenizer, bigram, embeddings, text generation
    ├── 04_phase3_transformer_from_scratch.ipynb      # attention by hand, TinyGPT, attention heatmaps
    └── 05_phase4_app.ipynb                           # train everything and launch the app
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
- [x] **Phase 3, part 1** — character-level language model: tokenizer, bigram, embeddings, generation
- [x] **Phase 3, part 2** — a small Transformer (self-attention) from scratch
- [x] **Phase 4** — the Wakil AI app (draw a digit, generate text, model info)

## License

MIT — see [LICENSE](LICENSE). MNIST is by Yann LeCun, Corinna Cortes and Christopher Burges. Tiny Shakespeare (public-domain text) as prepared by Andrej Karpathy for char-rnn.
