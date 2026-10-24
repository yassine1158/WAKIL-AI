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
| ✍️ Read a digit | Draw a digit with the mouse (click 🖌️ Brush first); see the prediction, the probabilities, and the 28×28 image the model actually sees; if it's wrong, **teach it** the right answer |
| 💬 Write text | Type a beginning; pick the model (Transformer or MLP), length and temperature; Wakil AI continues it |
| ℹ️ Models | Size, accuracy and loss of the trained models |

The interface (`app.py`) only calls `wakil_ai/service.py`, which holds the application logic and is tested without any UI (`tests/test_service.py`). Drawings go through the same preprocessing as MNIST (crop, 20×20 box, centered by center of mass); on MNIST test digits rendered as 280×280 drawings, this pipeline keeps 98.1% accuracy. Real mouse drawings look different from MNIST, so expect more mistakes on them.

### Wakil AI keeps learning from you (continual learning)

When Wakil AI reads a drawing wrong, choose the right digit and click **Teach Wakil AI**. It learns immediately:

1. Your drawing is saved in `data/my_digits.npz` (it stays on your computer).
2. 40 gradient-descent steps on batches that mix **your drawings (8)** with **MNIST examples (24)**. This "replay" prevents *catastrophic forgetting*: training only on new examples would erase what the network knew.
3. **Safety check:** accuracy is measured on 2,000 held-out MNIST images. If it dropped by more than 0.5 points, the update is cancelled.
4. The result is saved as `models/wakil_ai_mnist_personal.npz`, so the original model is never overwritten (**Reset** goes back to it).

Experiment (`wakil_ai/continual.py`): digits **rotated by 40°**, a style the model never saw. After learning from 50 corrections one at a time, accuracy on 1,000 *other* rotated digits rose from **58.8% to 72.9%**, while accuracy on normal MNIST digits stayed at **97.4%** (97.6% before).

This is learning from human feedback, not "learning alone": the model still needs examples and the right answers.

### Autopilot: Wakil AI improves on its own

```bash
python autolearn.py              # runs until Ctrl+C; nothing to do while it runs
python autolearn.py --rounds 8   # or a fixed number of rounds
```

| Model | How it learns with no human | Result (our run) |
|---|---|---|
| Digits | **Creates its own new examples**: rotates, shifts, resizes and thickens the digits it already has. A rotated 7 is still a 7, so the label stays right (data augmentation). Each batch mixes 50% new examples and 50% originals | 8 rounds, ~1 min: test digits rotated by 40° **54.6% → 82.6%**, normal MNIST test 97.78% → 97.44% |
| Text | **Reads its text again**. Next-character prediction is self-supervised: the answer is the next character of the text, so no human labels are needed | Transformer validation loss **1.722 → 1.707** after 2 rounds |

**Safety:** training continues in memory, but a model file is replaced only when the new version is better on held-out data, and (for digits) the accuracy on normal MNIST has not dropped more than 0.5 points since the first run. The learning rate is 0.05 for the first 4 digit rounds, then 0.01 to settle down; that's why the first rounds usually aren't saved. A running `app.py` reloads improved models automatically. History: `outputs/autolearn_log.csv`.

What does **not** work (we tested it): learning from its own confident answers (*self-training*). On 300 rotated digits, 42 of the 131 drawings it was ≥95% sure about were wrong, and accuracy didn't improve: "confident" doesn't mean "correct", and learning from its own mistakes only reinforces them. Learning something new always needs new information: real data, labels, or transformations that are guaranteed to keep the right answer.

## Wakil Brain: Wakil AI inside the Wakil WhatsApp agent

`train_intent.py` builds **Wakil Brain**, a small intent model (2048 → 32 → 4, 65,700 parameters) for the [Wakil WhatsApp](https://github.com/yassine1158/Projet_wakil_whatsapp_buss) sales agent. From the agent's last message and the customer's message it predicts the next action: **answer**, **hand over to a human**, **record the order**, or **send a product photo**.

```bash
python datasets/build_intent_seed.py                 # 640 synthetic starter examples (SICAR, SANA)
python train_intent.py                               # -> models/wakil_brain.json
python train_intent.py --teacher brain-samples.json --out ../Projet_wakil_whatsapp_buss/src/brain-model.json
```

- Text → numbers with the **hashing trick** (words and 3-letter pieces hashed with FNV-1a into 2,048 slots): no vocabulary file, and misspellings still share pieces with known words. `wakil_ai/intent.py` and the agent's `src/brain.ts` implement exactly the same steps; the exported file carries test cases that check both give the same probabilities.
- In the agent it runs in **shadow mode**: after each real reply, its prediction is compared with what Claude actually did (which tool it called) and saved, but it never changes what the customer receives.
- **Distillation:** the app exports those real messages with Claude's action as the label (`--teacher`). Claude is the teacher, Wakil Brain the student, and every retraining uses more real data.
- First version: 99.2% on held-out *synthetic* examples (optimistic, same templates) and 11/12 on new hand-written sentences. It missed "votre camion a cassé mon portail" (a complaint). Its real level is the agreement with Claude measured on real conversations.

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

python -m pytest              # 50 tests, ~3 seconds
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
python autolearn.py           # autopilot: keeps improving the models on its own
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
│   ├── service.py            # app logic: load models, read a drawing, write text, teach (no UI code)
│   ├── continual.py          # continual learning from corrections: replay + safety check
│   ├── autopilot.py          # self-improvement: self-made digit examples, text re-reading, safe saving
│   └── intent.py             # Wakil Brain: hashed text features, export for the WhatsApp agent
├── train.py  evaluate.py  predict.py
├── train_lm.py  generate.py
├── app.py                    # the interface only (Gradio)
├── autolearn.py              # the autopilot loop
├── train_intent.py           # trains Wakil Brain (intent model) and exports it as JSON
├── datasets/                 # synthetic starter examples for Wakil Brain
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
