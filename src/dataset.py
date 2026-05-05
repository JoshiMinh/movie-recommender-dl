from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.utils import ROOT_DIR, ensure_dir


DATASET_FILES = {
    "ml-100k": {
        "folder": "ml-100k",
        "interactions": "u.data",
        "zip": "ml-100k.zip",
    },
    "ml-1m": {
        "folder": "ml-1m",
        "interactions": "ratings.dat",
        "zip": "ml-1m.zip",
    },
}


@dataclass
class DataConfig:
    dataset: str
    data_path: str
    max_seq_len: int
    batch_size: int


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def _ensure_dataset_available(dataset: str, data_path: Path) -> Path:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset}. Use ml-100k or ml-1m.")

    spec = DATASET_FILES[dataset]
    dataset_dir = data_path / spec["folder"]
    interactions_file = dataset_dir / spec["interactions"]
    if interactions_file.exists():
        return interactions_file

    zip_path_data = data_path / spec["zip"]
    zip_path_root = ROOT_DIR / spec["zip"]
    zip_path = zip_path_data if zip_path_data.exists() else zip_path_root

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Could not find {spec['interactions']} and no archive found at "
            f"{zip_path_data} or {zip_path_root}. Place MovieLens ZIP locally."
        )

    ensure_dir(data_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    if not interactions_file.exists():
        raise FileNotFoundError(
            f"Extraction completed, but {interactions_file} was not found."
        )
    return interactions_file


def _load_interactions(dataset: str, interactions_path: Path) -> pd.DataFrame:
    if dataset == "ml-100k":
        df = pd.read_csv(
            interactions_path,
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )
    else:
        df = pd.read_csv(
            interactions_path,
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def _build_item_mapping(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    movie_ids = sorted(df["movie_id"].unique().tolist())
    item2idx = {movie_id: idx + 1 for idx, movie_id in enumerate(movie_ids)}
    idx2item = {idx: movie_id for movie_id, idx in item2idx.items()}
    idx2item[0] = 0
    return item2idx, idx2item


def _pad_sequence(seq: List[int], max_seq_len: int) -> List[int]:
    if len(seq) > max_seq_len:
        seq = seq[-max_seq_len:]
    padding = [0] * (max_seq_len - len(seq))
    return padding + seq


def _build_samples(
    user_items: List[int], max_seq_len: int
) -> Tuple[List[List[int]], List[int], List[List[int]], List[int], List[List[int]], List[int]]:
    train_x: List[List[int]] = []
    train_y: List[int] = []
    val_x: List[List[int]] = []
    val_y: List[int] = []
    test_x: List[List[int]] = []
    test_y: List[int] = []

    n = len(user_items)
    if n < 4:
        return train_x, train_y, val_x, val_y, test_x, test_y

    # Train on all but last two targets; reserve final two for val/test.
    for t in range(1, n - 2):
        seq = user_items[:t]
        train_x.append(_pad_sequence(seq, max_seq_len))
        train_y.append(user_items[t])

    val_t = n - 2
    val_x.append(_pad_sequence(user_items[:val_t], max_seq_len))
    val_y.append(user_items[val_t])

    test_t = n - 1
    test_x.append(_pad_sequence(user_items[:test_t], max_seq_len))
    test_y.append(user_items[test_t])

    return train_x, train_y, val_x, val_y, test_x, test_y


def prepare_dataloaders(config: DataConfig) -> Dict[str, object]:
    data_dir = ensure_dir(Path(config.data_path))
    interactions_file = _ensure_dataset_available(config.dataset, data_dir)
    interactions = _load_interactions(config.dataset, interactions_file)

    item2idx, idx2item = _build_item_mapping(interactions)
    interactions["item_idx"] = interactions["movie_id"].map(item2idx)

    train_x_all: List[List[int]] = []
    train_y_all: List[int] = []
    val_x_all: List[List[int]] = []
    val_y_all: List[int] = []
    test_x_all: List[List[int]] = []
    test_y_all: List[int] = []

    for _, user_df in interactions.groupby("user_id"):
        items = user_df["item_idx"].tolist()
        tx, ty, vx, vy, ex, ey = _build_samples(items, config.max_seq_len)
        train_x_all.extend(tx)
        train_y_all.extend(ty)
        val_x_all.extend(vx)
        val_y_all.extend(vy)
        test_x_all.extend(ex)
        test_y_all.extend(ey)

    if not train_x_all:
        raise RuntimeError("No training samples were created. Check dataset or max_seq_len.")

    train_ds = SequenceDataset(np.array(train_x_all), np.array(train_y_all))
    val_ds = SequenceDataset(np.array(val_x_all), np.array(val_y_all))
    test_ds = SequenceDataset(np.array(test_x_all), np.array(test_y_all))

    loaders = {
        "train_loader": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=config.batch_size, shuffle=False),
        "num_items": len(item2idx) + 1,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "stats": {
            "interactions": len(interactions),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "users": interactions["user_id"].nunique(),
            "movies": interactions["movie_id"].nunique(),
        },
    }
    return loaders


# Legacy notebook compatibility helpers

def download_and_extract_movielens(data_dir: str = "data") -> str:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    zip_path = os.path.join(data_dir, "ml-1m.zip")
    extract_path = os.path.join(data_dir, "ml-1m")

    if not os.path.exists(extract_path):
        url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
        print(f"Downloading Movielens 1M from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    return extract_path


def load_data(data_dir: str = "data"):
    extract_path = download_and_extract_movielens(data_dir)

    ratings_file = os.path.join(extract_path, "ratings.dat")
    movies_file = os.path.join(extract_path, "movies.dat")

    ratings_df = pd.read_csv(
        ratings_file,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies_df = pd.read_csv(
        movies_file,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    return ratings_df, movies_df


def process_data(ratings_df, max_seq_len: int = 50):
    unique_movies = ratings_df["movie_id"].unique()
    movie2idx = {mid: idx + 1 for idx, mid in enumerate(unique_movies)}

    unique_users = ratings_df["user_id"].unique()
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}

    ratings_df = ratings_df.copy()
    ratings_df["movie_idx"] = ratings_df["movie_id"].map(movie2idx)
    ratings_df["user_idx"] = ratings_df["user_id"].map(user2idx)
    ratings_df = ratings_df.sort_values(["user_idx", "timestamp"])

    train_data, val_data, test_data = [], [], []

    for user_idx, user_df in ratings_df.groupby("user_idx"):
        movies = user_df["movie_idx"].tolist()
        n_inter = len(movies)

        if n_inter < 3:
            continue

        train_end = int(n_inter * 0.8)
        val_end = int(n_inter * 0.9)

        train_movies = movies[:train_end]
        val_movies = movies[train_end:val_end]
        test_movies = movies[val_end:]

        def get_seqs(interactions):
            seqs, targs = [], []
            for i in range(1, len(interactions)):
                seq = interactions[max(0, i - max_seq_len):i]
                seqs.append(seq)
                targs.append(interactions[i])
            return seqs, targs

        t_seq, t_targ = get_seqs(train_movies)
        v_seq, v_targ = get_seqs(val_movies)
        te_seq, te_targ = get_seqs(test_movies)

        for s, t in zip(t_seq, t_targ):
            train_data.append((user_idx, s, t))
        for s, t in zip(v_seq, v_targ):
            val_data.append((user_idx, s, t))
        for s, t in zip(te_seq, te_targ):
            test_data.append((user_idx, s, t))

    return train_data, val_data, test_data, movie2idx, user2idx


class MovieSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_idx, seq, target = self.data[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def collate_fn(batch):
    user_ids = [item[0] for item in batch]
    sequences = [item[1] for item in batch]
    targets = [item[2] for item in batch]

    sequence_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    return torch.stack(user_ids), padded_sequences, sequence_lengths, torch.stack(targets)
