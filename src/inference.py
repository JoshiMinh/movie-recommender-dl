from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from src.model import NextMovieModel
from src.utils import load_json, read_yaml


def _pad_sequence(seq: List[int], max_seq_len: int) -> List[int]:
    seq = seq[-max_seq_len:]
    return [0] * (max_seq_len - len(seq)) + seq


class RecommenderService:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        metadata = load_json(self.artifact_dir / "metadata.json")

        self.max_seq_len = int(metadata["max_seq_len"])
        self.model_name = str(metadata["model"])
        self.num_items = int(metadata["num_items"])
        self.embedding_dim = int(metadata["embedding_dim"])
        self.hidden_size = int(metadata["hidden_size"])
        self.dropout = float(metadata["dropout"])

        self.item2idx = {int(k): int(v) for k, v in metadata["item2idx"].items()}
        self.idx2item = {int(k): int(v) for k, v in metadata["idx2item"].items()}

        self.model = NextMovieModel(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            rnn_type=self.model_name,
            dropout=self.dropout,
        )
        state = torch.load(self.artifact_dir / "best_model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    @classmethod
    def from_default(cls) -> "RecommenderService":
        cfg = read_yaml("config.yml")
        dataset_cfg = {k: v for k, v in cfg.items() if k in ["dataset", "data_path"]}
        model_cfg = {k: v for k, v in cfg.items() if k in ["model", "embedding_dim", "hidden_size", "max_seq_len", "dropout"]}
        dataset_name = dataset_cfg["dataset"]
        model_name = model_cfg["model"]
        artifact_dir = Path("artifacts") / dataset_name / model_name
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Artifact directory not found: {artifact_dir}. Train a model first."
            )
        return cls(artifact_dir)

    @torch.no_grad()
    def recommend(self, user_sequence: List[int], top_k: int = 3) -> List[int]:
        encoded = [self.item2idx[m] for m in user_sequence if m in self.item2idx]
        if not encoded:
            return []

        model_input = torch.tensor([_pad_sequence(encoded, self.max_seq_len)], dtype=torch.long)
        logits = self.model(model_input)

        # Avoid recommending already-seen items.
        seen = set(encoded)
        logits[0, 0] = float("-inf")
        for idx in seen:
            logits[0, idx] = float("-inf")

        _, top_idx = torch.topk(logits, k=min(top_k, logits.shape[1] - 1), dim=1)
        recs = [self.idx2item.get(int(i), 0) for i in top_idx[0].tolist()]
        return [r for r in recs if r != 0]
