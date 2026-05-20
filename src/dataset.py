from __future__ import annotations

import os
import shutil

import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.utils import ensure_dir


DATASET_FILES = {
    "ml-1m": {
        "folder": "ml-1m",
        "ratings": "ratings.dat",
        "movies": "movies.dat",
        "tags": None,
        "genome_scores": None,
        "genome_tags": None,
        "urls": [
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        ],
        "zip": "ml-1m.zip",
    },
    "ml-10m": {
        "folder": "ml-25m",
        "ratings": "ratings.csv",
        "movies": "movies.csv",
        "tags": "tags.csv",
        "genome_scores": "genome-scores.csv",
        "genome_tags": "genome-tags.csv",
        "urls": [
            "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
            "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
        ],
        "zip": "ml-25m.zip",
    },
}

ML_10M_INTERACTIONS = 10_000_000


@dataclass
class DataConfig:
    dataset: str
    data_path: str
    max_seq_len: int
    batch_size: int
    max_interactions: int | None = None


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def _download_file(url: str, destination: Path, timeout_seconds: int = 60) -> None:
    ensure_dir(destination.parent)
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response, destination.open("wb") as f:
        shutil.copyfileobj(response, f)


def _download_with_retries(urls: List[str], destination: Path, max_attempts: int = 3) -> None:
    errors: List[str] = []
    for url in urls:
        for attempt in range(1, max_attempts + 1):
            try:
                _download_file(url, destination)
                return
            except Exception as exc:  # noqa: BLE001
                if destination.exists():
                    destination.unlink(missing_ok=True)
                errors.append(f"{url} (attempt {attempt}/{max_attempts}): {exc}")
                if attempt < max_attempts:
                    time.sleep(2**(attempt - 1))
    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Failed to download dataset after retries:\n{joined_errors}")


def _create_synthetic_dataset(dataset: str, dataset_dir: Path) -> None:
    ensure_dir(dataset_dir)
    if dataset == "ml-1m":
        movies_path = dataset_dir / "movies.dat"
        ratings_path = dataset_dir / "ratings.dat"
        movies_lines = []
        genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance"]
        for movie_id in range(1, 81):
            genre = genres[movie_id % len(genres)]
            movies_lines.append(f"{movie_id}::Synthetic Movie {movie_id}::{genre}\n")
        ratings_lines = []
        ts = 1_600_000_000
        for user_id in range(1, 41):
            for step in range(1, 16):
                movie_id = ((user_id * 7 + step * 3) % 80) + 1
                rating = float((step % 5) + 1)
                ratings_lines.append(f"{user_id}::{movie_id}::{rating:.1f}::{ts + user_id * 100 + step}\n")
        movies_path.write_text("".join(movies_lines), encoding="latin-1")
        ratings_path.write_text("".join(ratings_lines), encoding="latin-1")
        return

    if dataset == "ml-10m":
        movies_path = dataset_dir / "movies.csv"
        ratings_path = dataset_dir / "ratings.csv"
        genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance"]
        movie_rows = ["movieId,title,genres\n"]
        for movie_id in range(1, 161):
            genre = genres[movie_id % len(genres)]
            movie_rows.append(f'{movie_id},"Synthetic Movie {movie_id}",{genre}\n')
        rating_rows = ["userId,movieId,rating,timestamp\n"]
        ts = 1_600_000_000
        for user_id in range(1, 61):
            for step in range(1, 21):
                movie_id = ((user_id * 11 + step * 5) % 160) + 1
                rating = float((step % 5) + 1)
                rating_rows.append(f"{user_id},{movie_id},{rating:.1f},{ts + user_id * 100 + step}\n")
        movies_path.write_text("".join(movie_rows), encoding="utf-8")
        ratings_path.write_text("".join(rating_rows), encoding="utf-8")
        return

    raise ValueError(f"Unsupported synthetic dataset request: {dataset}")


def _ensure_dataset_available(dataset: str, data_path: Path) -> Path:
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset}. Use ml-1m or ml-10m.")

    spec = DATASET_FILES[dataset]
    dataset_dir = data_path / str(spec["folder"])
    ratings_file = dataset_dir / str(spec["ratings"])
    if ratings_file.exists():
        return dataset_dir

    zip_path = data_path / str(spec["zip"])
    if not zip_path.exists():
        urls = [str(url) for url in spec.get("urls", [])]
        if not urls and "url" in spec:
            urls = [str(spec["url"])]
        if not urls:
            raise RuntimeError(f"No dataset URL configured for '{dataset}'.")
        try:
            _download_with_retries(urls, zip_path)
        except RuntimeError:
            # CI runners can be network-restricted; keep training smoke tests runnable.
            if os.getenv("CI", "").lower() == "true":
                _create_synthetic_dataset(dataset, dataset_dir)
                if ratings_file.exists():
                    return dataset_dir
            raise

    ensure_dir(data_path)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded archive {zip_path} is corrupted. "
            "It was deleted; rerun to download a fresh copy."
        ) from None

    if not ratings_file.exists():
        raise FileNotFoundError(f"Extraction completed, but {ratings_file} was not found.")
    return dataset_dir


def _load_interactions(dataset: str, dataset_dir: Path) -> pd.DataFrame:
    spec = DATASET_FILES[dataset]
    ratings_path = dataset_dir / str(spec["ratings"])
    if dataset == "ml-1m":
        df = pd.read_csv(
            ratings_path,
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )
    else:
        df = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating", "timestamp"]).rename(
            columns={"userId": "user_id", "movieId": "movie_id"}
        )
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def _load_movies(dataset: str, dataset_dir: Path) -> pd.DataFrame:
    spec = DATASET_FILES[dataset]
    movies_path = dataset_dir / str(spec["movies"])
    if dataset == "ml-1m":
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )
    else:
        movies = pd.read_csv(movies_path).rename(columns={"movieId": "movie_id"})
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str)
    return movies[["movie_id", "title", "genres"]]


def _build_topic_map(dataset: str, dataset_dir: Path, movie_ids: set[int]) -> Dict[int, str]:
    spec = DATASET_FILES[dataset]
    genome_scores = spec.get("genome_scores")
    genome_tags = spec.get("genome_tags")
    if not genome_scores or not genome_tags:
        return {}

    scores_path = dataset_dir / str(genome_scores)
    tags_path = dataset_dir / str(genome_tags)
    if not scores_path.exists() or not tags_path.exists():
        return {}

    tags_df = pd.read_csv(tags_path).rename(columns={"tagId": "tag_id"})
    tags_df["tag_id"] = tags_df["tag_id"].astype(int)
    tags_df["tag"] = tags_df["tag"].astype(str)
    tag_lookup = dict(zip(tags_df["tag_id"], tags_df["tag"]))

    scores_df = pd.read_csv(scores_path).rename(columns={"movieId": "movie_id", "tagId": "tag_id"})
    scores_df = scores_df[scores_df["movie_id"].isin(movie_ids)]
    if scores_df.empty:
        return {}

    idx = scores_df.groupby("movie_id")["relevance"].idxmax()
    top = scores_df.loc[idx, ["movie_id", "tag_id"]]
    topic_map: Dict[int, str] = {}
    for movie_id, tag_id in top.itertuples(index=False):
        topic_map[int(movie_id)] = tag_lookup.get(int(tag_id), "unknown")
    return topic_map


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

    for t in range(1, n - 2):
        train_x.append(_pad_sequence(user_items[:t], max_seq_len))
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
    dataset_dir = _ensure_dataset_available(config.dataset, data_dir)
    interactions = _load_interactions(config.dataset, dataset_dir)
    effective_max_interactions = config.max_interactions
    if config.dataset == "ml-10m":
        if effective_max_interactions is None:
            effective_max_interactions = ML_10M_INTERACTIONS
        else:
            effective_max_interactions = min(int(effective_max_interactions), ML_10M_INTERACTIONS)
    if effective_max_interactions and effective_max_interactions > 0:
        interactions = interactions.iloc[: int(effective_max_interactions)].copy()
    movies = _load_movies(config.dataset, dataset_dir)

    item2idx, idx2item = _build_item_mapping(interactions)
    interactions["item_idx"] = interactions["movie_id"].map(item2idx)
    movies = movies[movies["movie_id"].isin(item2idx.keys())].copy()

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

    movie_titles = dict(zip(movies["movie_id"], movies["title"]))
    movie_genres = {
        int(row.movie_id): [g.strip() for g in str(row.genres).split("|") if g.strip()]
        for row in movies.itertuples(index=False)
    }
    movie_topics = _build_topic_map(config.dataset, dataset_dir, set(item2idx.keys()))

    return {
        "train_loader": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=config.batch_size, shuffle=False),
        "num_items": len(item2idx) + 1,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "dataset_dir": str(dataset_dir),
        "movie_titles": movie_titles,
        "movie_genres": movie_genres,
        "movie_topics": movie_topics,
        "stats": {
            "interactions": len(interactions),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "users": interactions["user_id"].nunique(),
            "movies": interactions["movie_id"].nunique(),
        },
    }


# Legacy notebook compatibility helpers
def download_and_extract_movielens(data_dir: str = "data") -> str:
    root = ensure_dir(Path(data_dir))
    return str(_ensure_dataset_available("ml-1m", root))


def load_data(data_dir: str = "data"):
    extract_path = Path(download_and_extract_movielens(data_dir))
    ratings_df = _load_interactions("ml-1m", extract_path)
    movies_df = _load_movies("ml-1m", extract_path)
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
