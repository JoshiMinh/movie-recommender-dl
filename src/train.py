from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import collate_fn
from src.metrics import evaluate_model
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
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / max(1, len(train_loader.dataset))
        history["train_loss"].append(avg_train_loss)
        
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
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_val_metrics, history


# Legacy notebook compatibility wrapper

def train_with_oom_fallback(create_model_fn, train_dataset, val_dataset,
                            optimizer_name='adam', lr=1e-3, num_epochs=10,
                            device='cuda', start_batch_size=256):
    batch_size = start_batch_size

    while batch_size >= 16:
        print(f"Starting training with batch_size={batch_size}, optimizer={optimizer_name}, lr={lr}")
        try:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            model = create_model_fn().to(device)
            criterion = nn.CrossEntropyLoss()

            if optimizer_name.lower() == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name.lower() == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            else:
                raise ValueError("optimizer_name must be 'adam' or 'sgd'")

            train_losses, val_losses = [], []

            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0

                for user_ids, padded_seqs, seq_lengths, targets in train_loader:
                    user_ids = user_ids.to(device)
                    padded_seqs = padded_seqs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    logits = model(user_ids, padded_seqs, seq_lengths)
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_train = epoch_loss / len(train_loader)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for user_ids, padded_seqs, seq_lengths, targets in val_loader:
                        user_ids = user_ids.to(device)
                        padded_seqs = padded_seqs.to(device)
                        targets = targets.to(device)

                        logits = model(user_ids, padded_seqs, seq_lengths)
                        loss = criterion(logits, targets)
                        val_loss += loss.item()

                avg_val = val_loss / len(val_loader)

                train_losses.append(avg_train)
                val_losses.append(avg_val)
                print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

            return model, train_losses, val_losses, batch_size

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"CUDA OOM detected at batch_size={batch_size}. Halving batch size...")
                batch_size //= 2
            else:
                raise e

    raise RuntimeError("Failed to train without OOM even at minimum batch size.")
