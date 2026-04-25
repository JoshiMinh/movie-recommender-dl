from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.evaluation.metrics import evaluate_model
from src.utils import ensure_dir


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


def train_one_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    cfg: TrainingConfig,
    device: torch.device,
    save_path: str | Path,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, cfg)
    best_metric = -1.0
    best_val_metrics: Dict[str, float] = {}

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
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader.dataset))
        val_metrics = evaluate_model(model, val_loader, cfg.top_k, device)
        metric_name = f"Hit@{max(cfg.top_k)}"
        current_metric = val_metrics.get(metric_name, 0.0)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | train_loss={avg_train_loss:.4f} "
            f"| {metric_name}={current_metric:.4f}"
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_val_metrics
