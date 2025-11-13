# Comparative Analysis of RNN Architectures for Sentiment Classification

This repo implements and evaluates multiple recurrent architectures (RNN, LSTM, BiLSTM) on the IMDb movie review dataset.  
The experiments are designed to systematically study the effect of:

- Architecture: `rnn`, `lstm`, `bilstm`
- Activation function: `tanh`, `relu`, `sigmoid`
- Optimizer: `adam`, `adamw`, `sgd`, `rmsprop`
- Sequence length: 50, 100, 200
- Gradient clipping: on vs off (max-norm = 1.0)

All runs log metrics into a single CSV so the tables and plots in the report can be reproduced directly.

---

## 1. Setup

### Requirements

- Python 3.10+
- Recommended: GPU (CUDA) for faster training, but CPU also works

Install dependencies:

```bash
pip install -r requirements.txt
````

(Optional) Create and activate a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# or
.venv\Scripts\activate       # Windows
```

---

## 2. Data and Preprocessing

* Dataset: IMDb Movie Reviews (50k reviews: 25k train, 25k test, balanced).
* The dataset is automatically downloaded by the training script.
* Preprocessing:

  * Lowercasing and basic cleaning
  * Tokenization with NLTK
  * Vocabulary: top 10,000 most frequent tokens from the training set
  * Reviews mapped to integer IDs and padded/truncated to a fixed sequence length (`--seq-len`)

You don’t need to manually place any files under `data/` for the default IMDb setup.

---

## 3. Repository Structure

```text
.
├── data/                 # IMDb data and any cached artifacts
├── src/
│   ├── preprocess.py     # Text preprocessing, vocab building, dataset objects
│   ├── models.py         # RNN / LSTM / BiLSTM model definitions
│   ├── train.py          # Training loop and experiment runner
│   ├── evaluate.py       # Metrics aggregation and plotting
│   └── utils.py          # Shared utilities (logging, seeding, etc.)
├── results/
│   ├── metrics.csv       # All experiment metrics appended here
│   └── plots/            # Generated plots (PNG)
├── report.pdf            # Project report (compiled from LaTeX)
├── requirements.txt
└── README.md
```

The report and README assume this structure, so try to keep it unchanged.

---

## 4. Training: Quickstart

### 4.1 Single Baseline Run

Train a single model with a specific configuration (e.g., LSTM + Tanh + Adam, seq_len=50, no grad clipping):

```bash
python -m src.train \
  --mode single \
  --arch lstm \
  --activation tanh \
  --optimizer adam \
  --seq-len 50 \
  --grad-clip 0
```

Key flags:

* `--arch`: `rnn`, `lstm`, `bilstm`
* `--activation`: `tanh`, `relu`, `sigmoid`
* `--optimizer`: `adam`, `adamw`, `sgd`, `rmsprop`
* `--seq-len`: sequence length (e.g., 50, 100, 200)
* `--grad-clip`: 0 (no clipping) or 1 (clip with max-norm 1.0)

By default, the script sets seeds for `torch`, `numpy`, and `random` to make runs reproducible.

---

## 5. Systematic Experiments (as in the Report)

All “grid” experiments vary exactly one factor at a time, starting from a fixed baseline configuration.
Each run appends a row to `results/metrics.csv` with:

`Model, Activation, Optimizer, SeqLength, GradClip, Accuracy, F1, EpochTime, Seed, CPU, RAMGB, RunID`

### 5.1 Group A – Sequence Length (LSTM + Adam + Tanh)

Baseline: `lstm`, `tanh`, `adam`, `grad_clip=0`; vary only sequence length.

```bash
python -m src.train \
  --mode grid \
  --vary seq_len \
  --arch lstm \
  --activation tanh \
  --optimizer adam \
  --grad-clip 0
```

This will run LSTM at `seq_len` in `{50, 100, 200}` and log metrics for each.

---

### 5.2 Group B – Architecture (RNN vs LSTM vs BiLSTM)

Baseline: `seq_len=200`, `tanh`, `adam`, `grad_clip=0`; vary only architecture.

```bash
python -m src.train \
  --mode grid \
  --vary arch \
  --seq-len 200 \
  --activation tanh \
  --optimizer adam \
  --grad-clip 0
```

---

### 5.3 Group C – Optimizer Sweep (BiLSTM)

Baseline: `bilstm`, `tanh`, `seq_len=200`, `grad_clip=0`; vary only optimizer.

```bash
python -m src.train \
  --mode grid \
  --vary optimizer \
  --arch bilstm \
  --activation tanh \
  --seq-len 200 \
  --grad-clip 0
```

Optimizers tested: `adam`, `adamw`, `sgd`, `rmsprop`.

---

### 5.4 Group D – Activation Functions (BiLSTM + AdamW)

Baseline: `bilstm`, `adamw`, `seq_len=200`, `grad_clip=0`; vary only activation.

```bash
python -m src.train \
  --mode grid \
  --vary activation \
  --arch bilstm \
  --optimizer adamw \
  --seq-len 200 \
  --grad-clip 0
```

Activations tested: `tanh`, `relu`, `sigmoid`.

---

### 5.5 Group E – Gradient Clipping (BiLSTM + AdamW + Tanh)

Baseline: `bilstm`, `adamw`, `tanh`, `seq_len=200`; vary only gradient clipping.

```bash
python -m src.train \
  --mode grid \
  --vary grad_clip \
  --arch bilstm \
  --activation tanh \
  --optimizer adamw \
  --seq-len 200
```

---

### 5.6 Group F – Seeds (Stability)

Fix the “good” configuration and vary seeds.
(If `--seed` is exposed as a flag, you can do:)

```bash
# Example: run same config with three seeds
for SEED in 0 1 2; do
  python -m src.train \
    --mode single \
    --arch bilstm \
    --activation tanh \
    --optimizer adamw \
    --seq-len 200 \
    --grad-clip 1 \
    --seed $SEED
done
```

If `--seed` is not exposed, seeds are set inside `train.py` (default 42) and you can modify them there.

---

### 5.7 Group G – Capacity (Hidden Size)

If the script exposes a `--hidden-size` flag, you can sweep capacity like this:

```bash
for HS in 64 128 256; do
  python -m src.train \
    --mode single \
    --arch bilstm \
    --activation tanh \
    --optimizer adamw \
    --seq-len 200 \
    --grad-clip 1 \
    --hidden-size $HS
done
```

Otherwise, hidden sizes 64, 128, 256 are configured directly in `models.py` or via config files.

---

## 6. Evaluation and Plots

All evaluation commands read from `results/metrics.csv` and write plots into `results/plots/`.

### 6.1 Accuracy / F1 vs Sequence Length

For the LSTM + Adam + Tanh sequence-length sweep:

```bash
python -m src.evaluate \
  --plot acc_f1_vs_seq_len \
  --arch lstm \
  --activation tanh \
  --optimizer adam \
  --grad-clip 0
```

This filters runs to the LSTM + Adam + Tanh baseline and plots accuracy and F1 vs `seq_len`.

### 6.2 Best and Worst Loss Curves

Loss curves for the best and worst models (by F1):

```bash
python -m src.evaluate \
  --plot best_worst_losses
```

This produces figures like:

* `results/plots/loss_curves_best.png`
* `results/plots/loss_curves_worst.png`

which are used in the report.

---

## 7. Reproducibility

* Seeds for `torch`, `numpy`, and `random` are fixed in the training script (default 42).
* All important configuration choices are logged into `results/metrics.csv`.
* Hardware info (CPU type, RAM, device) is also logged for each run so results can be interpreted in context.

To fully reproduce the report:

1. Run the Group A–G commands above.
2. Verify that `results/metrics.csv` contains entries for all configurations.
3. Regenerate plots using the `src.evaluate` commands.
4. Recompile `report.tex` to `report.pdf`.

---

## 8. Expected Runtime (Rough)

On a single Colab GPU (x86_64, ~12–13 GB RAM), typical epoch times:

* LSTM, `seq_len=50`: ~0.9–1.0 s / epoch
* LSTM, `seq_len=200`: ~1.7–1.8 s / epoch
* BiLSTM, `seq_len=200`, hidden size 128: ~2.7 s / epoch
* BiLSTM, `seq_len=200`, hidden size 256: ~5.1 s / epoch

CPU runs will be slower by a factor depending on your machine, but relative trends (which config is faster/slower) should match.

---

## 9. Notes

* The default “best” configuration found in the report is:

  * `arch=bilstm`, `activation=tanh`, `optimizer=adamw`, `seq_len=200`, `hidden_size=128`, `grad_clip=0 or 1`
* For CPU-constrained setups, a good trade-off is:

  * `arch=bilstm`, `activation=tanh`, `optimizer=adamw`, `seq_len=200`, `hidden_size=64`, `grad_clip=1`

Everything you need to reproduce the tables and plots in the report is contained in:

* `src/train.py`, `src/evaluate.py`
* `results/metrics.csv`
* `results/plots/`