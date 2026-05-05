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


class SequenceRecommender(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        user_emb_dim=32,
        movie_emb_dim=64,
        hidden_dim=128,
        rnn_type="gru",
        num_layers=1,
        dropout=0.3,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.movie_embedding = nn.Embedding(num_movies, movie_emb_dim, padding_idx=0)

        self.rnn_type = rnn_type.lower()
        input_dim = user_emb_dim + movie_emb_dim
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ValueError("rnn_type must be one of 'rnn', 'lstm', or 'gru'")

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_movies)

    def forward(self, user_ids, padded_seqs, seq_lengths):
        batch_size, seq_len = padded_seqs.size()

        user_emb = self.user_embedding(user_ids)
        user_emb_seq = user_emb.unsqueeze(1).repeat(1, seq_len, 1)
        movie_emb_seq = self.movie_embedding(padded_seqs)

        concat_emb = torch.cat([movie_emb_seq, user_emb_seq], dim=-1)
        concat_emb = self.dropout_layer(concat_emb)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            concat_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.rnn(packed_input)
        output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        idx = (seq_lengths - 1).view(-1, 1).expand(batch_size, output_padded.size(-1)).unsqueeze(1)
        idx = idx.to(output_padded.device)
        last_hidden_state = output_padded.gather(1, idx).squeeze(1)
        last_hidden_state = self.dropout_layer(last_hidden_state)
        logits = self.fc(last_hidden_state)
        return logits

    def predict(self, user_ids, padded_seqs, seq_lengths):
        logits = self.forward(user_ids, padded_seqs, seq_lengths)
        return torch.nn.functional.softmax(logits, dim=-1)
