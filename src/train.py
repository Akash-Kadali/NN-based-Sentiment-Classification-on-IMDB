# src/train.py
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .preprocess import load_imdb_datasets
from .models import build_model
from .utils import (
    seed_everything,
    get_device,
    compute_metrics,
    ensure_dir,
    hardware_report,
    save_csv_row,
    make_run_id,
)

# Speed up cuDNN autotuning for RNN/LSTM with fixed shapes
torch.backends.cudnn.benchmark = True

# ======== CSV schema (unchanged) ========
HEADER = [
    "Model",
    "Activation",
    "Optimizer",
    "Seq Length",
    "Grad Clipping",
    "Accuracy",
    "F1",
    "Epoch Time (s)",
    "Seed",
    "CPU",
    "RAM(GB)",
    "RunID",
]

# ======== Fixed baseline for controlled sweeps (defaults only) ========
BASELINE: Dict[str, object] = dict(
    arch="lstm",
    activation="tanh",
    optimizer_name="adam",
    seq_len=50,
    grad_clip=0.0,          # use float, 0 disables
    batch_size=32,
    epochs=5,
    lr=1e-3,
    weight_decay=0.0,
    seed=42,
    results_dir="results",
    num_workers=2,
    prefetch_factor=4,
    persistent_workers=True,
    hidden_size=64,
    emb_dim=100,
    num_layers=2,
    dropout=0.4,
)


def make_optimizer(name: str, params, lr: float = 1e-3, weight_decay: float = 0.0):
    n = name.lower()
    if n == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    y_true, y_prob = [], []

    amp_enabled = device.type == "cuda"
    autocast = torch.cuda.amp.autocast

    with autocast(enabled=amp_enabled, dtype=torch.float16 if amp_enabled else None):
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            if logits.dim() == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_true.append(y.numpy())
            y_prob.append(probs)

    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_prob = np.concatenate(y_prob) if y_prob else np.array([])
    acc, f1 = compute_metrics(y_true, y_prob)
    return float(acc), float(f1)


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device: torch.device,
    epochs: int = 5,
    grad_clip: float = 0.0,
    run_id: str = "run",
    results_dir: str = "results",
) -> float:
    """
    Train with AMP + non_blocking transfers. Returns average epoch time in seconds.
    """
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    amp_enabled = device.type == "cuda"
    autocast = torch.cuda.amp.autocast

    per_epoch_rows = []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        for i, (X, y) in enumerate(pbar, 1):
            X = X.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled, dtype=torch.float16 if amp_enabled else None):
                logits = model(X)
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            scaler.step(optimizer)
            scaler.update()

            bs = X.size(0)
            running_loss += loss.detach().float().item() * bs
            seen += bs

            if i % 10 == 0 or i == len(train_loader):
                pbar.set_postfix(
                    loss=f"{running_loss / max(1, seen):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        ep_time = time.time() - start
        epoch_times.append(ep_time)

        # validation once per epoch
        val_acc, val_f1 = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(1, seen)
        print(
            f"Epoch {epoch}/{epochs} | loss {avg_loss:.4f} | "
            f"val_acc {val_acc:.4f} | val_f1 {val_f1:.4f} | {ep_time:.1f}s"
        )
        per_epoch_rows.append(
            {"epoch": epoch, "loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1, "epoch_time_s": ep_time}
        )

    # persist per-epoch loss & val metrics
    exp_dir = Path(results_dir) / "experiments" / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_epoch_rows).to_csv(exp_dir / "loss.csv", index=False)

    return float(np.mean(epoch_times)) if epoch_times else 0.0


def run_experiment(
    arch: str = "lstm",
    activation: str = "tanh",
    optimizer_name: str = "adam",
    seq_len: int = 50,
    grad_clip: float = 0.0,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
    results_dir: str = "results",
    num_workers: int = 2,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    hidden_size: int = 64,
    emb_dim: int = 100,
    num_layers: int = 2,
    dropout: float = 0.4,
) -> None:
    seed_everything(seed)
    device = get_device()
    cpu, ram_gb = hardware_report()

    # data
    train_ds, test_ds, vocab_size, _ = load_imdb_datasets(seq_len=seq_len)
    val_size = int(0.1 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # DataLoader knobs: feed the GPU
    pin = device.type == "cuda"
    # Colab/Linux: good defaults; Windows users can set workers=0 via CLI
    if os.name == "nt":
        num_workers = 0
        persistent_workers = False

    # guard prefetch_factor for workers=0
    pf = prefetch_factor if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=pf,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=pf,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=pf,
    )

    # model & optimizer
    model = build_model(
        arch=arch,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
    )
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay)

    # run id & training
    run_id = make_run_id(arch, activation, optimizer_name, seq_len, bool(grad_clip and grad_clip > 0), seed)
    avg_epoch_time = train_one(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        grad_clip=float(grad_clip) if grad_clip else 0.0,
        run_id=run_id,
        results_dir=results_dir,
    )

    # test
    test_acc, test_f1 = evaluate(model, test_loader, device)

    # save summary row
    ensure_dir(results_dir)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    row = [
        arch.upper(),
        activation.capitalize(),
        optimizer_name.upper(),
        int(seq_len),
        int(grad_clip > 0.0),
        round(float(test_acc), 4),
        round(float(test_f1), 4),
        round(float(avg_epoch_time), 2),
        int(seed),
        cpu,
        ram_gb,
        run_id,
    ]
    save_csv_row(metrics_path, HEADER, row)

    print("\nFinal Test Metrics:")
    print(
        {
            "accuracy": row[5],
            "f1": row[6],
            "avg_epoch_time_sec": row[7],
            "run_id": run_id,
            "device": str(device),
        }
    )


def grid(
    vary: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
    num_workers: int = 2,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> None:
    assert vary in {"arch", "activation", "optimizer", "seq_len", "grad_clip"}
    base = BASELINE.copy()
    base.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    )

    choices = {
        "arch": ["rnn", "lstm", "bilstm"],
        "activation": ["sigmoid", "relu", "tanh"],
        "optimizer": ["adam", "adamw", "sgd", "rmsprop"],
        "seq_len": [50, 100, 200],
        "grad_clip": [0.0, 1.0],
    }

    key_name = "optimizer_name" if vary == "optimizer" else vary
    vals = choices[vary]

    for val in tqdm(vals, desc=f"sweep:{vary}", unit="run", dynamic_ncols=True):
        cfg = dict(base)
        cfg[key_name] = val
        print(f"\n=== Running: {key_name}={val} with others fixed ===")
        run_experiment(**cfg)  # type: ignore[arg-type]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "grid"], default="single")
    p.add_argument("--vary", choices=["arch", "activation", "optimizer", "seq_len", "grad_clip"], default=None)

    # single-run options
    p.add_argument("--arch", default="lstm", choices=["rnn", "lstm", "bilstm"])
    p.add_argument("--activation", default="tanh", choices=["sigmoid", "relu", "tanh"])
    p.add_argument("--optimizer", dest="optimizer_name", default="adam", choices=["adam", "adamw", "sgd", "rmsprop"])
    p.add_argument("--seq-len", dest="seq_len", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=0.0, help="0 disables, >0 enables and sets max_norm")

    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")

    # dataloader/system
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", type=int, default=1, help="1 true, 0 false")

    # model capacity
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=100)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.4)
    return p.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()
    args.persistent_workers = bool(args.persistent_workers)

    if args.mode == "single":
        run_experiment(
            arch=args.arch,
            activation=args.activation,
            optimizer_name=args.optimizer_name,
            seq_len=args.seq_len,
            grad_clip=args.grad_clip,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            results_dir=args.results_dir,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            hidden_size=args.hidden_size,
            emb_dim=args.emb_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        if args.vary is None:
            raise SystemExit("--mode grid requires --vary {arch|activation|optimizer|seq_len|grad_clip}")
        grid(
            vary=args.vary,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
