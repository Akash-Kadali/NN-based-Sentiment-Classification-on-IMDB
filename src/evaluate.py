# src/evaluate.py
from __future__ import annotations

import argparse
import os
from typing import Tuple

import pandas as pd
from tqdm import tqdm

# Headless-safe backend
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .utils import ensure_dir

PLOTS_DIR = os.path.join("results", "plots")
EXP_DIR = os.path.join("results", "experiments")


def _loss_csv_path(run_id: str, experiments_dir: str = EXP_DIR) -> str:
    # train.py saves per-run losses here: results/experiments/<RunID>/loss.csv
    return os.path.join(experiments_dir, run_id, "loss.csv")


def _load_metrics_csv(metrics_csv: str) -> pd.DataFrame:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    expected = {
        "Model",
        "Activation",
        "Optimizer",
        "Seq Length",
        "Grad Clipping",
        "Accuracy",
        "F1",
        "RunID",
    }
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"metrics.csv is missing columns: {missing}")
    # Clean types
    df["Seq Length"] = pd.to_numeric(df["Seq Length"], errors="coerce").astype("Int64")
    df["Grad Clipping"] = pd.to_numeric(df["Grad Clipping"], errors="coerce").astype("Int64")
    for c in ["Accuracy", "F1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Seq Length", "Grad Clipping", "Accuracy", "F1"])
    return df


def _save_line_plot(
    x,
    y_series: Tuple[Tuple[pd.Series, str, str], ...],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: str,
    ylim: Tuple[float, float] | None = None,
):
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure()
    for y, marker, label in y_series:
        plt.plot(x, y, marker=marker, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_acc_f1_vs_seq_len(
    metrics_csv: str = "results/metrics.csv",
    arch: str = "lstm",
    activation: str = "tanh",
    optimizer: str = "adam",
    grad_clip: int = 0,
    out_dir: str = PLOTS_DIR,
) -> None:
    """
    Plots Accuracy and F1 vs Seq Length for a fixed (arch, activation, optimizer, grad_clip).
    Expects metrics.csv columns:
    ['Model','Activation','Optimizer','Seq Length','Grad Clipping','Accuracy','F1',...,'RunID']
    """
    ensure_dir(out_dir)
    try:
        df = _load_metrics_csv(metrics_csv)
    except Exception as e:
        print(f"[WARN] {e}")
        return

    mask = (
        (df["Model"] == arch.upper())
        & (df["Activation"] == activation.capitalize())
        & (df["Optimizer"] == optimizer.upper())
        & (df["Grad Clipping"] == int(bool(grad_clip)))
    )
    dd = df.loc[mask].sort_values("Seq Length")

    if dd.empty:
        print("[INFO] No matching rows for the specified filters.")
        return

    out_path = os.path.join(out_dir, "acc_f1_vs_seq_len.png")
    title = f"Accuracy/F1 vs Sequence Length ({arch}/{activation}/{optimizer}/clip={int(bool(grad_clip))})"
    _save_line_plot(
        dd["Seq Length"],
        (
            (dd["Accuracy"], "o", "Accuracy"),
            (dd["F1"], "s", "F1 (macro)"),
        ),
        xlabel="Sequence Length",
        ylabel="Score",
        title=title,
        out_path=out_path,
        ylim=(0.0, 1.0),
    )


def plot_best_worst_losses(
    metrics_csv: str = "results/metrics.csv",
    experiments_dir: str = EXP_DIR,
    out_dir: str = PLOTS_DIR,
) -> None:
    """
    Finds best and worst runs by F1 from metrics.csv and plots their training loss curves.
    Saves:
      - results/plots/loss_curves_best.png
      - results/plots/loss_curves_worst.png
    """
    ensure_dir(out_dir)
    try:
        df = _load_metrics_csv(metrics_csv)
    except Exception as e:
        print(f"[WARN] {e}")
        return

    if df.empty:
        print("[INFO] metrics.csv is empty.")
        return

    # Choose by F1
    best_row = df.iloc[df["F1"].idxmax()]
    worst_row = df.iloc[df["F1"].idxmin()]

    # Iterate with a tiny tqdm so long runs still show something
    for tag, row in tqdm(
        [("best", best_row), ("worst", worst_row)],
        desc="plot loss curves",
        unit="run",
        dynamic_ncols=True,
    ):
        run_id = str(row["RunID"])
        loss_path = _loss_csv_path(run_id, experiments_dir)
        if not os.path.exists(loss_path):
            print(f"[WARN] Missing loss CSV for {tag} run: {loss_path}")
            continue

        loss_df = pd.read_csv(loss_path)
        if loss_df.empty or "epoch" not in loss_df or "loss" not in loss_df:
            print(f"[WARN] Bad loss CSV format for {tag} run: {loss_path}")
            continue

        out_path = os.path.join(out_dir, f"loss_curves_{tag}.png")
        title = (
            f"Training Loss – {tag.upper()} by F1\n"
            f"{row['Model']}/{row['Activation']}/{row['Optimizer']}/"
            f"L{int(row['Seq Length'])}/clip={int(row['Grad Clipping'])}"
        )
        _save_line_plot(
            loss_df["epoch"],
            ((loss_df["loss"], "o", "Train Loss"),),
            xlabel="Epoch",
            ylabel="Training Loss",
            title=title,
            out_path=out_path,
            ylim=None,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plot", choices=["acc_f1_vs_seq_len", "best_worst_losses"], required=True)
    p.add_argument("--arch", default="lstm")
    p.add_argument("--activation", default="tanh")
    p.add_argument("--optimizer", default="adam")
    p.add_argument("--grad-clip", type=int, default=0)
    p.add_argument("--metrics", default="results/metrics.csv")
    p.add_argument("--experiments-dir", default=EXP_DIR)
    p.add_argument("--out-dir", default=PLOTS_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot == "acc_f1_vs_seq_len":
        plot_acc_f1_vs_seq_len(
            metrics_csv=args.metrics,
            arch=args.arch,
            activation=args.activation,
            optimizer=args.optimizer,
            grad_clip=args.grad_clip,
            out_dir=args.out_dir,
        )
    else:
        plot_best_worst_losses(
            metrics_csv=args.metrics,
            experiments_dir=args.experiments_dir,
            out_dir=args.out_dir,
        )
