from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from torch import nn

from src.settings import DEFAULTS, SUPPORTED_MODELS


ROOT_DIR = Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    top_k: List[int]


def _build_optimizer(model: torch.nn.Module, cfg: TrainingConfig):
    name = cfg.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=0.9,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError("optimizer must be one of: sgd, adam")


@torch.no_grad()
def topk_metrics(logits: torch.Tensor, targets: torch.Tensor, ks: Iterable[int]) -> Dict[str, float]:
    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks)
    _, topk_idx = torch.topk(logits, k=max_k, dim=1)

    metrics: Dict[str, float] = {}
    for k in ks:
        preds = topk_idx[:, :k]
        hits = (preds == targets.unsqueeze(1)).any(dim=1).float()
        hit = hits.mean().item()
        recall = hit

        ndcg_sum = 0.0
        for i in range(preds.size(0)):
            row = preds[i].tolist()
            tgt = int(targets[i].item())
            if tgt in row:
                rank = row.index(tgt) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
        ndcg = ndcg_sum / preds.size(0)

        metrics[f"Hit@{k}"] = hit
        metrics[f"Recall@{k}"] = recall
        metrics[f"NDCG@{k}"] = ndcg
    return metrics


@torch.no_grad()
def evaluate_model(model, loader, ks: List[int] | None = None, device: torch.device | str = "cuda"):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()

    if ks is None:
        top1_correct = 0
        hr10_hits = 0
        total = 0
        for batch in loader:
            user_ids, padded_seqs, seq_lengths, targets = batch
            user_ids = user_ids.to(device)
            padded_seqs = padded_seqs.to(device)
            targets = targets.to(device)

            if hasattr(model, "predict"):
                probs = model.predict(user_ids, padded_seqs, seq_lengths)
            else:
                logits = model(user_ids, padded_seqs, seq_lengths)
                probs = torch.nn.functional.softmax(logits, dim=-1)

            top1_preds = probs.argmax(dim=-1)
            top1_correct += (top1_preds == targets).sum().item()

            top10_preds = torch.topk(probs, k=10, dim=-1).indices
            targets_expanded = targets.unsqueeze(1).expand_as(top10_preds)
            hr10_hits += (top10_preds == targets_expanded).sum().item()
            total += targets.size(0)

        top1_acc = top1_correct / total if total > 0 else 0.0
        hr10 = hr10_hits / total if total > 0 else 0.0
        return top1_acc, hr10

    aggregate = {f"Hit@{k}": 0.0 for k in ks}
    aggregate.update({f"Recall@{k}": 0.0 for k in ks})
    aggregate.update({f"NDCG@{k}": 0.0 for k in ks})

    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        batch_metrics = topk_metrics(logits, y, ks)
        batch_size = x.size(0)
        count += batch_size
        for key, val in batch_metrics.items():
            aggregate[key] += val * batch_size

    if count == 0:
        return {k: 0.0 for k in aggregate}
    return {k: v / count for k, v in aggregate.items()}


def format_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    if not results:
        return "No results to display."

    metric_keys = sorted(next(iter(results.values())).keys())
    header = ["Model"] + metric_keys
    widths = [len(h) for h in header]
    rows = []
    for model_name, metrics in results.items():
        row = [model_name] + [f"{metrics[k]:.4f}" for k in metric_keys]
        rows.append(row)
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    def fmt_row(cols):
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    line = "-+-".join("-" * w for w in widths)
    text_lines = [fmt_row(header), line]
    text_lines.extend(fmt_row(r) for r in rows)
    return "\n".join(text_lines)


def train_one_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    cfg: TrainingConfig,
    device: torch.device,
    save_path: str | Path,
) -> Tuple[torch.nn.Module, Dict[str, float], Dict[str, List[float]]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, cfg)
    best_metric = -1.0
    best_val_metrics: Dict[str, float] = {}
    history: Dict[str, List[float]] = {"train_loss": []}

    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    model.to(device)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader.dataset))
        history["train_loss"].append(avg_train_loss)

        val_metrics = evaluate_model(model, val_loader, cfg.top_k, device)
        metric_name = f"Hit@{max(cfg.top_k)}"
        current_metric = val_metrics.get(metric_name, 0.0)

        if current_metric > best_metric:
            best_metric = current_metric
            best_val_metrics = val_metrics
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_val_metrics, history


@torch.no_grad()
def collect_predictions(model, loader, top_k: int, device: torch.device | str = "cuda") -> Dict[str, List[int]]:
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    targets: List[int] = []
    pred_top1: List[int] = []
    pred_topk: List[List[int]] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        top_idx = torch.topk(logits, k=min(top_k, logits.size(1)), dim=1).indices
        targets.extend(y.detach().cpu().tolist())
        pred_top1.extend(top_idx[:, 0].detach().cpu().tolist())
        pred_topk.extend(top_idx.detach().cpu().tolist())
    return {"targets": targets, "pred_top1": pred_top1, "pred_topk": pred_topk}


def compute_label_accuracy(
    targets_idx: List[int],
    pred_top1_idx: List[int],
    pred_topk_idx: List[List[int]],
    idx2item: Dict[int, int],
    movie_labels: Dict[int, List[str]],
) -> Dict[str, Dict[str, float]]:
    per_label_hits_top1 = defaultdict(int)
    per_label_hits_topk = defaultdict(int)
    per_label_support = defaultdict(int)

    for target_i, pred1_i, predk_i in zip(targets_idx, pred_top1_idx, pred_topk_idx):
        target_movie = idx2item.get(int(target_i))
        pred1_movie = idx2item.get(int(pred1_i))
        predk_movies = {idx2item.get(int(i)) for i in predk_i}
        target_labels = movie_labels.get(int(target_movie), [])
        pred1_labels = set(movie_labels.get(int(pred1_movie), []))
        predk_labels = set()
        for m in predk_movies:
            if m is not None:
                predk_labels.update(movie_labels.get(int(m), []))
        for label in target_labels:
            per_label_support[label] += 1
            if label in pred1_labels:
                per_label_hits_top1[label] += 1
            if label in predk_labels:
                per_label_hits_topk[label] += 1

    result: Dict[str, Dict[str, float]] = {}
    for label, support in per_label_support.items():
        if support > 0:
            result[label] = {
                "support": float(support),
                "top1_accuracy": per_label_hits_top1[label] / support,
                "topk_hit": per_label_hits_topk[label] / support,
            }
    return result


@dataclass
class ExperimentInput:
    dataset: str
    data_path: str
    model: str
    embedding_dim: int
    hidden_size: int
    max_seq_len: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    top_k: List[int]
    max_interactions: int | None = None
    is_baseline: bool = False
    run_label: str = ""


def _dataset_artifact_dir(dataset: str) -> Path:
    return ensure_dir(Path("artifacts") / dataset)


def _safe_topics(movie_topics: Dict[int, str]) -> Dict[int, List[str]]:
    return {int(movie_id): [topic] for movie_id, topic in movie_topics.items() if topic}


def run_experiment(exp: ExperimentInput, device: str | None = None) -> Dict[str, object]:
    from src.dataset import DataConfig, prepare_dataloaders
    from src.model import NextMovieModel

    dataset_dir = _dataset_artifact_dir(exp.dataset)
    model_key = f"{exp.model}_{exp.optimizer}"

    data_cfg = DataConfig(
        dataset=exp.dataset,
        data_path=exp.data_path,
        max_seq_len=exp.max_seq_len,
        batch_size=exp.batch_size,
        max_interactions=exp.max_interactions,
    )
    bundle = prepare_dataloaders(data_cfg)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(resolved_device)

    model = NextMovieModel(
        num_items=int(bundle["num_items"]),
        embedding_dim=exp.embedding_dim,
        hidden_size=exp.hidden_size,
        rnn_type=exp.model,
        dropout=exp.dropout,
    )
    train_cfg = TrainingConfig(
        epochs=exp.epochs,
        learning_rate=exp.learning_rate,
        optimizer=exp.optimizer,
        weight_decay=exp.weight_decay,
        top_k=exp.top_k,
    )

    model_path = dataset_dir / f"{model_key}.pth"
    model, best_val, history = train_one_model(
        model=model,
        train_loader=bundle["train_loader"],
        val_loader=bundle["val_loader"],
        cfg=train_cfg,
        device=torch_device,
        save_path=model_path,
    )

    test_metrics = evaluate_model(model, bundle["test_loader"], exp.top_k, torch_device)
    pred = collect_predictions(model, bundle["test_loader"], top_k=max(exp.top_k), device=torch_device)
    idx2item = {int(k): int(v) for k, v in bundle["idx2item"].items()}
    genre_metrics = compute_label_accuracy(
        pred["targets"], pred["pred_top1"], pred["pred_topk"], idx2item, bundle["movie_genres"]
    )
    topic_map = _safe_topics(bundle["movie_topics"])
    topic_metrics = (
        compute_label_accuracy(pred["targets"], pred["pred_top1"], pred["pred_topk"], idx2item, topic_map)
        if topic_map
        else {}
    )

    metadata = {
        "run_id": model_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "is_baseline": exp.is_baseline,
        "run_label": exp.run_label,
        "params": asdict(exp),
        "dataset_stats": bundle["stats"],
        "best_val_metrics": best_val,
        "test_metrics": test_metrics,
        "history": history,
        "genre_metrics": genre_metrics,
        "topic_metrics": topic_metrics,
        "movie_titles": bundle["movie_titles"],
        "item2idx": bundle["item2idx"],
        "idx2item": bundle["idx2item"],
    }
    save_json(dataset_dir / f"{model_key}_metadata.json", metadata)
    return metadata


def list_runs(dataset: str) -> List[Dict[str, object]]:
    root = _dataset_artifact_dir(dataset)
    if not root.exists():
        return []
    runs: List[Dict[str, object]] = []
    for model_name in SUPPORTED_MODELS:
        candidates = sorted(root.glob(f"{model_name}_*_metadata.json"), reverse=True)
        for candidate in candidates:
            runs.append(load_json(candidate))
            break
    return runs


def build_comparison_table(runs: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in runs:
        metrics = run.get("test_metrics", {})
        params = run.get("params", {})
        rows.append(
            {
                "Run ID": run.get("run_id"),
                "Label": run.get("run_label"),
                "Baseline": bool(run.get("is_baseline", False)),
                "Model": params.get("model"),
                "Optimizer": params.get("optimizer"),
                "LR": params.get("learning_rate"),
                "Hit@5": metrics.get("Hit@5", 0.0),
                "Hit@10": metrics.get("Hit@10", 0.0),
                "NDCG@10": metrics.get("NDCG@10", 0.0),
                "Created": run.get("created_at"),
            }
        )
    return pd.DataFrame(rows)


def _pad_sequence(seq: List[int], max_seq_len: int) -> List[int]:
    seq = seq[-max_seq_len:]
    return [0] * (max_seq_len - len(seq)) + seq


class RecommenderService:
    def __init__(self, artifact_dir: str | Path, model_key: str | None = None):
        from src.model import NextMovieModel

        self.artifact_dir = Path(artifact_dir)
        if model_key:
            metadata_path = self.artifact_dir / f"{model_key}_metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        else:
            metadata_candidates = sorted(self.artifact_dir.glob("*_metadata.json"))
            if not metadata_candidates:
                raise FileNotFoundError(f"No metadata files found in {self.artifact_dir}")
            metadata_path = metadata_candidates[0]
        metadata = load_json(metadata_path)

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
        model_file = self.artifact_dir / f"{self.model_name}_{metadata['optimizer']}.pth"
        state = torch.load(model_file, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    @classmethod
    def from_default(cls) -> "RecommenderService":
        dataset = DEFAULTS.dataset
        dataset_dir = Path("artifacts") / dataset

        if dataset_dir.exists():
            if DEFAULTS.model in SUPPORTED_MODELS:
                preferred_meta = dataset_dir / f"{DEFAULTS.model}_{DEFAULTS.optimizer}_metadata.json"
                if preferred_meta.exists():
                    return cls(dataset_dir, model_key=f"{DEFAULTS.model}_{DEFAULTS.optimizer}")

            for model_name in ("rnn", "lstm", "gru"):
                meta_matches = sorted(dataset_dir.glob(f"{model_name}_*_metadata.json"))
                if meta_matches:
                    model_key = meta_matches[0].name.replace("_metadata.json", "")
                    return cls(dataset_dir, model_key=model_key)

        raise FileNotFoundError(
            f"No artifact directory found under {dataset_dir}. "
            "Train a model first (expected artifacts/<dataset>/<model>_<optimizer>.pth and metadata)."
        )

    @torch.no_grad()
    def recommend(self, user_sequence: List[int], top_k: int = 3) -> List[int]:
        encoded = [self.item2idx[m] for m in user_sequence if m in self.item2idx]
        if not encoded:
            return []
        model_input = torch.tensor([_pad_sequence(encoded, self.max_seq_len)], dtype=torch.long)
        logits = self.model(model_input)
        logits[0, 0] = float("-inf")
        for idx in set(encoded):
            logits[0, idx] = float("-inf")
        _, top_idx = torch.topk(logits, k=min(top_k, logits.shape[1] - 1), dim=1)
        recs = [self.idx2item.get(int(i), 0) for i in top_idx[0].tolist()]
        return [r for r in recs if r != 0]
