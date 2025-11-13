# src/models.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import activation_fn


def _forget_bias_init(module: nn.Module, value: float = 1.0) -> None:
    """Set LSTM forget-gate bias to a positive value for better training stability."""
    for name, param in module.named_parameters():
        if "bias_ih" in name or "bias_hh" in name:
            # LSTM biases are [b_i | b_f | b_g | b_o]
            hidden = param.numel() // 4
            with torch.no_grad():
                param[hidden : 2 * hidden].fill_(value)


class _BaseClassifier(nn.Module):
    """
    Base text sequence classifier:
      • Embedding (padding_idx=0)
      • RNN core defined in subclasses
      • Activation on the final representation (not logits)
      • Linear layer that outputs a single logit per example
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
        activation: str = "tanh",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        # Activation for hidden representation (keep logits raw for BCEWithLogitsLoss)
        self.act = activation_fn(activation)
        # Subclasses must set self.classifier = nn.Linear(...)
        self.classifier: nn.Linear | None = None

        # Small, sensible init
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()

    def _activate_and_classify(self, rep: torch.Tensor) -> torch.Tensor:
        # rep: (B, H) or (B, 2H)
        rep = self.act(rep)
        rep = self.dropout(rep)
        return self.classifier(rep).squeeze(1)  # logits, shape (B,)

    @staticmethod
    def _lengths_from_padded(x: torch.Tensor) -> torch.Tensor:
        # x is LongTensor (B, T) with 0 as PAD
        lengths = (x != 0).sum(dim=1)
        # clamp in case of empty
        return lengths.clamp(min=1)


class RNNClassifier(_BaseClassifier):
    """
    Vanilla RNN. PyTorch RNN supports nonlinearity in {"tanh","relu"} only.
    We map the cell's nonlinearity, but still honor the requested activation
    on the final representation.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
        activation: str = "tanh",
    ):
        super().__init__(vocab_size, emb_dim, hidden_size, num_layers, dropout, activation)
        nonlin = "relu" if activation.lower() == "relu" else "tanh"
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity=nonlin,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, 1)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) with 0=PAD
        emb = self.embedding(x)  # (B, T, E)
        lengths = self._lengths_from_padded(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, hn = self.rnn(packed)  # hn: (L, B, H)
        rep = hn[-1]  # (B, H) last layer hidden
        return self._activate_and_classify(rep)  # (B,)


class LSTMClassifier(_BaseClassifier):
    """Unidirectional LSTM encoder + linear head."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
        activation: str = "tanh",
    ):
        super().__init__(vocab_size, emb_dim, hidden_size, num_layers, dropout, activation)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, 1)

        # Init
        _forget_bias_init(self.lstm, value=1.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) with 0=PAD
        # flatten_parameters improves cuDNN performance/avoids warnings on some backends
        self.lstm.flatten_parameters()
        emb = self.embedding(x)  # (B, T, E)
        lengths = self._lengths_from_padded(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(packed)  # hn: (L, B, H)
        rep = hn[-1]  # (B, H)
        return self._activate_and_classify(rep)  # (B,)


class BiLSTMClassifier(_BaseClassifier):
    """Bidirectional LSTM encoder + linear head (concatenate last fwd/bwd)."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.4,
        activation: str = "tanh",
    ):
        super().__init__(vocab_size, emb_dim, hidden_size, num_layers, dropout, activation)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)

        # Init
        _forget_bias_init(self.lstm, value=1.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) with 0=PAD
        self.lstm.flatten_parameters()
        emb = self.embedding(x)  # (B, T, E)
        lengths = self._lengths_from_padded(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(packed)  # hn: (L*2, B, H)
        # Last layer's forward and backward states are the last two slices.
        fwd = hn[-2]  # (B, H)
        bwd = hn[-1]  # (B, H)
        rep = torch.cat([fwd, bwd], dim=1)  # (B, 2H)
        return self._activate_and_classify(rep)  # (B,)


def build_model(
    arch: str,
    vocab_size: int,
    emb_dim: int = 100,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.4,
    activation: str = "tanh",
) -> nn.Module:
    """
    Factory for classifiers. Returns a module that outputs raw logits (no sigmoid).
    """
    a = arch.lower()
    if a == "rnn":
        return RNNClassifier(vocab_size, emb_dim, hidden_size, num_layers, dropout, activation)
    if a == "lstm":
        return LSTMClassifier(vocab_size, emb_dim, hidden_size, num_layers, dropout, activation)
    if a == "bilstm":
        return BiLSTMClassifier(vocab_size, emb_dim, hidden_size, num_layers, dropout, activation)
    raise ValueError(f"Unknown arch: {arch}")
