from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

from src.inference import RecommenderService
from src.config import Config
from src.dataset import DataConfig, prepare_dataloaders
from src.metrics import evaluate_model, format_comparison_table
from src.model import NextMovieModel
from src.train import TrainingConfig, train_one_model
from src.utils import ROOT_DIR, ensure_dir, read_yaml, save_json


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


def run_demo(
    config_path: str,
    model_override: str | None = None,
) -> None:
    Config.load(config_path)
    config = Config.get()
    dataset_cfg = config.dataset
    model_cfg = config.model

    model_name = (model_override or model_cfg.model).lower()
    if model_name == "all":
        model_name = "lstm"

    artifact_dir = ROOT_DIR / "artifacts" / dataset_cfg.dataset / model_name
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Artifact {artifact_dir} does not exist. Run training first."
        )

    service = RecommenderService(artifact_dir)
    sample = [1, 5, 20]
    recs = service.recommend(sample, top_k=3)

    print("Demo request:", {"user_sequence": sample})
    print("Demo response:", {"recommendations": recs})


def run_ui() -> None:
    ui_path = ROOT_DIR / "src" / "streamlit.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)


def list_datasets(config_path: str) -> List[str]:
    from src.dataset import DATASET_FILES
    Config.load(config_path)
    config = Config.get()
    data_path = Path(config.dataset.data_path)
    if not data_path.is_absolute():
        data_path = ROOT_DIR / data_path

    if not data_path.exists():
        return []

    # Filter DATASET_FILES to those available as directory or zip
    available = []
    for name, spec in DATASET_FILES.items():
        # Check if directory exists
        if (data_path / spec["folder"]).is_dir():
            available.append(name)
        # Check if zip exists in data_path or root
        elif (data_path / spec["zip"]).exists() or (ROOT_DIR / spec["zip"]).exists():
            available.append(name)

    return sorted(list(set(available)))


def update_dataset(config_path: str, dataset_name: str) -> None:
    cfg = read_yaml(config_path)
    cfg["dataset"] = dataset_name
    with Path(config_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # Reload config after update
    Config.load(config_path)


def prompt_model(default_model: str, allow_all: bool = True) -> str:
    allowed = ["rnn", "lstm", "gru"] + (["all", "a"] if allow_all else [])
    prompt_suffix = " / ".join(allowed)
    raw = input(f"Model ({prompt_suffix}) [{default_model}]: ").strip().lower()
    model = raw or default_model.lower()
    if model == "a":
        return "all"
    if model not in allowed:
        print(f"Invalid model '{model}', using default '{default_model}'.")
        return default_model.lower()
    return model


def prompt_optimizer(default_opt: str) -> str:
    allowed = ["sgd", "adam", "auto"]
    raw = input(f"Optimizer (sgd / adam / auto) [{default_opt}]: ").strip().lower()
    opt = raw or default_opt.lower()
    if opt == "auto":
        return "adam"
    if opt not in allowed:
        print(f"Invalid optimizer '{opt}', using default '{default_opt}'.")
        return default_opt.lower()
    return opt


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
        print("\n0. Automated Workflow (train all models + optimizers)")
        print("1. Streamlit UI")
        print("2. Exit")

        choice = input("Choose an option [0-2]: ").strip()

        try:
            if choice == "0":
                print("\nStarting automated workflow...")
                selected_device = prompt_device()
                for opt in ["adam", "sgd"]:
                    print(f"\n>>> TRAINING: {opt.upper()} OPTIMIZER <<<")
                    run_training(
                        config_path=config_path,
                        model_override="all",
                        optimizer_override=opt,
                        device_override=selected_device,
                    )

            elif choice == "1":
                run_ui()

            elif choice == "2":
                print("Exiting.")
                return

            else:
                print("Invalid option, select 0-2.")

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
