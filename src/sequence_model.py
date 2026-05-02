from __future__ import annotations

import torch
from torch import nn


class NextMovieModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_size: int,
        rnn_type: str = "lstm",
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        if self.rnn_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
            )
        elif self.rnn_type == "gru":
            self.recurrent = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
            )
        elif self.rnn_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True,
            )
        else:
            raise ValueError("rnn_type must be one of: rnn, lstm, gru")

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, num_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.recurrent(emb)
        else:
            _, h_n = self.recurrent(emb)

        hidden = h_n[-1]
        hidden = self.dropout(hidden)
        logits = self.output(hidden)
        return logits
