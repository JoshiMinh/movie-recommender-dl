from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Config
from src.dataset import DATASET_FILES, DataConfig, prepare_dataloaders
from src.model import NextMovieModel
from src.utils import (
    ROOT_DIR,
    RecommenderService,
    TrainingConfig,
    ensure_dir,
    evaluate_model,
    format_comparison_table,
    read_yaml,
    save_json,
    train_one_model,
)


class RecommendRequest(BaseModel):
    user_sequence: list[int] = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=50)


class RecommendResponse(BaseModel):
    recommendations: list[int]


app = FastAPI(title="Movie Recommender DL", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        config = Config.get()
        artifact_dir = ROOT_DIR / "artifacts" / config.dataset.dataset / config.model.model
        service = RecommenderService(artifact_dir)
        recs = service.recommend(payload.user_sequence, top_k=payload.top_k)
        return RecommendResponse(recommendations=recs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def run_training(
    config_path: str,
    model_override: str | None = None,
    optimizer_override: str | None = None,
    device_override: str | None = None,
) -> Dict[str, Dict[str, float]]:
    # Load configuration
    config = Config.load(config_path)
    dataset_cfg = config.dataset
    model_cfg = config.model
    train_cfg = config.train

    selected_model = (model_override or model_cfg.model).lower()
    if selected_model == "all":
        model_names = ["rnn", "lstm", "gru"]
    elif selected_model in {"rnn", "lstm", "gru"}:
        model_names = [selected_model]
    else:
        raise ValueError("Model must be rnn, lstm, gru, or all")

    data_cfg = DataConfig(
        dataset=dataset_cfg.dataset,
        data_path=dataset_cfg.data_path,
        max_seq_len=model_cfg.max_seq_len,
        batch_size=train_cfg.batch_size,
    )

    data_bundle = prepare_dataloaders(data_cfg)
    print("Data stats:", data_bundle["stats"])

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    optimizer_name = (optimizer_override or train_cfg.optimizer).lower()

    train_config = TrainingConfig(
        epochs=train_cfg.epochs,
        learning_rate=train_cfg.learning_rate,
        optimizer=optimizer_name,
        weight_decay=train_cfg.weight_decay,
        top_k=train_cfg.top_k,
    )

    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    num_items = int(data_bundle["num_items"])

    comparison: Dict[str, Dict[str, float]] = {}

    for model_name in model_names:
        print(f"\n=== Training {model_name.upper()} with {optimizer_name.upper()} ===")
        model = NextMovieModel(
            num_items=num_items,
            embedding_dim=model_cfg.embedding_dim,
            hidden_size=model_cfg.hidden_size,
            rnn_type=model_name,
            dropout=model_cfg.dropout,
        )

        artifact_dir = ensure_dir(ROOT_DIR / "artifacts" / dataset_cfg.dataset / f"{model_name}_{optimizer_name}")
        model_path = artifact_dir / "best_model.pt"

        try:
            model, best_val, history = train_one_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=train_config,
                device=device,
                save_path=model_path,
            )
        except KeyboardInterrupt:
            print("\nTraining cancelled by user (Ctrl+C).")
            if comparison:
                print("\n=== Partial Test Comparison ===")
                print(format_comparison_table(comparison))
            return comparison

        test_metrics = evaluate_model(model, test_loader, train_config.top_k, device)
        comparison[model_name] = test_metrics

        metadata = {
            "dataset": dataset_cfg.dataset,
            "model": model_name,
            "optimizer": optimizer_name,
            "num_items": num_items,
            "embedding_dim": model_cfg.embedding_dim,
            "hidden_size": model_cfg.hidden_size,
            "max_seq_len": model_cfg.max_seq_len,
            "dropout": model_cfg.dropout,
            "item2idx": data_bundle["item2idx"],
            "idx2item": data_bundle["idx2item"],
            "best_val_metrics": best_val,
            "test_metrics": test_metrics,
            "history": history,
        }
        save_json(artifact_dir / "metadata.json", metadata)
        print(f"Saved best model and metadata to: {artifact_dir}")

    print("\n=== Test Comparison (RNN vs LSTM vs GRU) ===")
    print(format_comparison_table(comparison))
    return comparison


def run_ui() -> None:
    ui_path = ROOT_DIR / "src" / "streamlit.py"
    venv_python = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    python_exec = str(venv_python) if venv_python.exists() else sys.executable
    subprocess.run(
        [
            python_exec,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.fileWatcherType",
            "poll",
        ],
        check=True,
    )


def update_dataset(config_path: str, dataset_name: str) -> None:
    cfg = read_yaml(config_path)
    cfg["dataset"] = dataset_name
    with Path(config_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # Reload config after update
    Config.load(config_path)


def prompt_model(default_model: str) -> str:
    allowed = ["rnn", "lstm", "gru", "all", "a"]
    raw = input(f"Model (rnn / lstm / gru / all) [{default_model}]: ").strip().lower()
    model = raw or default_model.lower()
    if model == "a":
        return "all"
    if model not in allowed:
        print(f"Invalid model '{model}', using default '{default_model}'.")
        return default_model.lower()
    return model


def prompt_device() -> str:
    current = "cuda" if torch.cuda.is_available() else "cpu"
    allowed = ["cpu", "cuda", "auto"]
    raw = input(f"Device (cpu / cuda / auto) [{current}]: ").strip().lower()
    device = raw or "auto"
    if device == "auto":
        return current
    if device not in allowed:
        print(f"Invalid device '{device}', using auto '{current}'.")
        return current
    return device


def get_current_dataset(config_path: str) -> str:
    Config.load(config_path)
    return Config.get().dataset.dataset


def set_dataset_interactive(config_path: str) -> None:
    current = get_current_dataset(config_path)
    supported = sorted(DATASET_FILES.keys())
    print(f"\nCurrent dataset: {current}")
    print(f"Supported datasets: {', '.join(supported)}")
    choice = input(f"Set dataset [{current}]: ").strip().lower()
    if not choice:
        print(f"Dataset unchanged: {current}")
        return
    if choice not in supported:
        print(f"Invalid dataset '{choice}'. Supported: {', '.join(supported)}")
        return
    update_dataset(config_path, choice)
    print(f"Dataset updated to: {choice}")


def interactive_menu() -> None:
    config_path = str(ROOT_DIR / "config.yml")
    Config.load(config_path)

    while True:
        config = Config.get()
        dataset_cfg = config.dataset
        model_cfg = config.model
        train_cfg = config.train
        print("\n=== Movie Recommender ===")
        print(f"Dataset: {dataset_cfg.dataset}")
        print(f"Config: {model_cfg.model} + {train_cfg.optimizer}")
        print("\n1. Set/Get Dataset")
        print("2. Train Models (all or one)")
        print("3. Run Streamlit")
        print("4. Exit")

        choice = input("Choose an option [1-4]: ").strip()

        try:
            if choice == "1":
                set_dataset_interactive(config_path)

            elif choice == "2":
                selected_model = prompt_model(model_cfg.model)
                selected_device = prompt_device()
                run_training(
                    config_path=config_path,
                    model_override=selected_model,
                    optimizer_override=train_cfg.optimizer,
                    device_override=selected_device,
                )

            elif choice == "3":
                run_ui()

            elif choice == "4":
                print("Exiting.")
                return

            else:
                print("Invalid option, select 1-4.")

        except FileNotFoundError as exc:
            print(f"Error: {exc}")
        except Exception as exc:
            print(f"Operation failed: {exc}")


def main() -> None:
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C).")


if __name__ == "__main__":
    main()
