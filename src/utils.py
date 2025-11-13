# src/utils.py
from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

SEED = 42


# ---------- Reproducibility & system ----------
def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch. Keep deterministic flags stable enough for RNNs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Avoid strict determinism to prevent RNN kernel errors; keep CuDNN stable.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_report() -> Optional[str]:
    """Return GPU name if available, else None (not used in CSV, but handy for logs)."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def hardware_report() -> Tuple[str, Optional[float]]:
    """Return (CPU string, RAM in GB). Keep signature stable for metrics CSV."""
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        ram_gb = None
    return cpu, ram_gb


# Back-compat if some code calls system_info()
def system_info() -> Tuple[str, Optional[float]]:
    return hardware_report()


# ---------- Math/activation ----------
def activation_fn(name: str) -> nn.Module:
    """Return an nn.Module to apply to hidden reps (not the final logits)."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


# ---------- Timing / stats ----------
@dataclass
class EpochStats:
    loss: float
    seconds: float


def epoch_time(start_seconds: float, end_seconds: float) -> float:
    return float(end_seconds - start_seconds)


def format_seconds(sec: float) -> str:
    """Pretty duration like 1h 03m 12s or 42.3s."""
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(round(sec)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def estimate_time_from_loader(epochs: int, train_loader, sample_batches: int = 50) -> Tuple[float, float]:
    """
    Quickly estimate (sec_per_batch, total_sec) by timing a few batches from train_loader.
    Safe to call before training starts.
    """
    n_batches = len(train_loader)
    if n_batches == 0:
        return 0.0, 0.0

    it = iter(train_loader)
    counted = 0
    t0 = time.time()
    with tqdm(total=min(sample_batches, n_batches), desc="warmup timing", unit="batch", dynamic_ncols=True, leave=False) as p:
        for _ in range(min(sample_batches, n_batches)):
            try:
                _ = next(it)
            except StopIteration:
                break
            counted += 1
            p.update(1)
    elapsed = time.time() - t0
    if counted == 0:
        return 0.0, 0.0
    sec_per_batch = elapsed / counted
    total_sec = epochs * n_batches * sec_per_batch
    return float(sec_per_batch), float(total_sec)


# ---------- Metrics ----------
def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Given true labels and probabilities, compute (accuracy, macro-F1).
    NOTE: pass probabilities, not logits. Apply sigmoid before calling this if needed.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    if y_true.size == 0:
        return 0.0, 0.0
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1)


# ---------- IO helpers ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv_row(path: str, header: Sequence[str], row: Sequence[object]) -> None:
    """
    Append a row to CSV at `path`, writing `header` only if the file doesn't exist.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(list(header))
        w.writerow(list(row))


def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_config(cfg: Mapping[str, Any]) -> str:
    """Nicely aligned string for printing hyperparameters."""
    if not cfg:
        return ""
    klen = max(len(k) for k in cfg)
    lines = [f"{k:<{klen}} : {v}" for k, v in cfg.items()]
    return "\n".join(lines)


# ---------- Run IDs ----------
def make_run_id(
    arch: str,
    activation: str,
    optimizer: str,
    seq_len: int,
    grad_clip: bool,
    seed: int,
) -> str:
    """Stable run identifier used for results/experiments/<run_id>/loss.csv"""
    return f"{arch}_{activation}_{optimizer}_L{seq_len}_gc{int(bool(grad_clip))}_seed{seed}"
