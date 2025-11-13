# src/preprocess.py
from __future__ import annotations

import os
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset

# Ensure tokenizer is available
nltk.download("punkt", quiet=True)

# Constants
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
SPECIAL_TOKENS: Dict[str, int] = {"<pad>": 0, "<unk>": 1}


# --------- Text normalization & tokenization ---------
def normalize_text(s: str) -> str:
    """
    Lowercase, remove HTML breaks, strip punctuation/specials, collapse spaces.
    Matches the assignment requirement: remove punctuation & special chars.
    """
    s = s.lower()
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # keep only a-z, 0-9, space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    return word_tokenize(normalize_text(s))


# --------- Safe tar extraction ---------
def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve(strict=False)
        target = target.resolve(strict=False)
    except Exception:
        # Fallback if paths don't exist yet
        directory = Path(os.path.abspath(directory))
        target = Path(os.path.abspath(target))
    return str(target).startswith(str(directory))


def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    dest_path = dest / member.name
    if not _is_within_directory(dest, dest_path):
        raise RuntimeError(f"Blocked path traversal attempt: {member.name}")
    tar.extract(member, path=dest)


# --------- Download helpers with tqdm ---------
def _download_with_progress(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(out_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="download imdb",
            dynamic_ncols=True,
        ) as pbar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def _extract_with_progress(tgz_path: Path, dest: Path) -> None:
    with tarfile.open(tgz_path, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc="extract imdb", unit="file", dynamic_ncols=True) as pbar:
            for m in members:
                _safe_extract_member(tar, m, dest)
                pbar.update(1)


# --------- IMDb data handling ---------
def _read_imdb_split(split_dir: str | Path) -> Tuple[List[str], List[int]]:
    """
    Read pre-defined IMDb split (pos/neg dirs).
    Returns texts (cleaned) and binary labels (pos=1, neg=0).
    """
    split_dir = Path(split_dir)
    texts, labels = [], []
    for label_name, y in (("pos", 1), ("neg", 0)):
        d = split_dir / label_name
        files = sorted(d.glob("*.txt"))
        for path in tqdm(files, desc=f"read {label_name}", unit="file", dynamic_ncols=True, leave=False):
            with path.open("r", encoding="utf-8") as f:
                txt = f.read()
            texts.append(normalize_text(txt))
            labels.append(y)
    return texts, labels


def download_imdb(root: str = "data") -> str:
    """
    Download and extract IMDb dataset if missing. Returns path to 'aclImdb'.
    """
    root_path = Path(root)
    target = root_path / "aclImdb"
    if target.exists():
        return str(target)

    tgz_path = root_path / "aclImdb_v1.tar.gz"
    if not tgz_path.exists():
        _download_with_progress(URL, tgz_path)
    _extract_with_progress(tgz_path, root_path)

    return str(target)


# --------- Vocabulary & encoding ---------
def build_vocab(texts: List[str], max_vocab: int = 10_000) -> Tuple[Dict[str, int], List[str]]:
    """
    Build top-K vocabulary from tokenized training texts.
    Returns (word2idx, idx2word) with PAD=0, UNK=1.
    """
    counter = Counter()
    for t in tqdm(texts, desc="build vocab", unit="doc", dynamic_ncols=True):
        counter.update(tokenize(t))

    # Put special tokens first, then most common words excluding them
    idx2word: List[str] = ["<pad>", "<unk>"]
    for w, _ in counter.most_common(max_vocab - len(idx2word)):
        if w not in SPECIAL_TOKENS:
            idx2word.append(w)

    word2idx: Dict[str, int] = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def encode(text: str, word2idx: Dict[str, int]) -> List[int]:
    unk_id = SPECIAL_TOKENS["<unk>"]
    return [word2idx.get(tok, unk_id) for tok in tokenize(text)]


def pad_trunc(ids: List[int], seq_len: int) -> List[int]:
    if len(ids) >= seq_len:
        return ids[:seq_len]
    return ids + [SPECIAL_TOKENS["<pad>"]] * (seq_len - len(ids))


# --------- Dataset objects & loaders ---------
class IMDBDataset(Dataset):
    """
    Torch Dataset over raw texts + labels.
    Encodes with given vocab and pads/truncates to seq_len.
    """
    def __init__(self, texts: List[str], labels: List[int], word2idx: Dict[str, int], seq_len: int):
        self.seq_len = seq_len
        self.word2idx = word2idx
        self.labels = labels
        self.data = []
        for t in tqdm(texts, desc="encode dataset", unit="doc", dynamic_ncols=True):
            self.data.append(pad_trunc(encode(t, word2idx), seq_len))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = torch.tensor(self.data[i], dtype=torch.long)
        y = torch.tensor(self.labels[i], dtype=torch.float32)
        return x, y


def load_imdb_datasets(
    data_root: str = "data",
    seq_len: int = 50,
    max_vocab: int = 10_000,
) -> Tuple[TensorDataset, TensorDataset, int, Dict[str, int]]:
    """
    Convenience loader that returns TensorDatasets, vocab_size, and word2idx.
    Vocabulary is built on TRAIN ONLY.
    """
    imdb_dir = download_imdb(data_root)
    train_texts, train_labels = _read_imdb_split(Path(imdb_dir) / "train")
    test_texts, test_labels = _read_imdb_split(Path(imdb_dir) / "test")

    word2idx, _ = build_vocab(train_texts, max_vocab=max_vocab)

    def convert(texts: List[str], desc: str) -> torch.Tensor:
        out_ids: List[List[int]] = []
        for t in tqdm(texts, desc=desc, unit="doc", dynamic_ncols=True):
            out_ids.append(pad_trunc(encode(t, word2idx), seq_len))
        return torch.tensor(out_ids, dtype=torch.long)

    X_train = convert(train_texts, desc="to tensor (train)")
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_test = convert(test_texts, desc="to tensor (test)")
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    vocab_size = len(word2idx)

    return train_ds, test_ds, vocab_size, word2idx


# --------- Corpus description for the report ---------
def describe_corpus(data_root: str = "data") -> Dict[str, float]:
    """
    Returns corpus stats required by the report:
    - n_train, n_test
    - avg_len, median_len (token counts after normalization)
    - vocab_topk (fixed to 10_000 as per assignment)
    """
    import numpy as np

    imdb_dir = download_imdb(data_root)
    train_texts, _ = _read_imdb_split(Path(imdb_dir) / "train")
    test_texts, _ = _read_imdb_split(Path(imdb_dir) / "test")

    lengths: List[int] = []
    for t in tqdm(train_texts + test_texts, desc="measure lengths", unit="doc", dynamic_ncols=True):
        lengths.append(len(tokenize(t)))

    return {
        "n_train": float(len(train_texts)),
        "n_test": float(len(test_texts)),
        "avg_len": float(np.mean(lengths)) if lengths else 0.0,
        "median_len": float(np.median(lengths)) if lengths else 0.0,
        "vocab_topk": 10000.0,
    }
