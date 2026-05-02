from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

from src.inference import RecommenderService
from src.dataset import DataConfig, prepare_dataloaders
from src.metrics import evaluate_model, format_comparison_table
from src.sequence_model import NextMovieModel
from src.trainer import TrainingConfig, train_one_model
from src.utils import ROOT_DIR, ensure_dir, read_yaml, save_json


def load_configs(config_path: str) -> Tuple[dict, dict, dict]:
    """Load all configurations from single config.yml file."""
    cfg = read_yaml(config_path)
    # Split into dataset, model, and train configs
    dataset_cfg = {"dataset": cfg["dataset"], "data_path": cfg["data_path"]}
    model_cfg = {k: v for k, v in cfg.items() if k in ["model", "embedding_dim", "hidden_size", "max_seq_len", "dropout"]}
    train_cfg = {k: v for k, v in cfg.items() if k in ["batch_size", "epochs", "learning_rate", "optimizer", "top_k", "weight_decay"]}
    return dataset_cfg, model_cfg, train_cfg


def run_training(
    config_path: str,
    model_override: str | None = None,
    optimizer_override: str | None = None,
    device_override: str | None = None,
) -> Dict[str, Dict[str, float]]:
    dataset_cfg, model_cfg, train_cfg = load_configs(config_path)

    selected_model = (model_override or model_cfg["model"]).lower()
    if selected_model == "all":
        model_names = ["rnn", "lstm", "gru"]
    elif selected_model in {"rnn", "lstm", "gru"}:
        model_names = [selected_model]
    else:
        raise ValueError("Model must be rnn, lstm, gru, or all")

    data_cfg = DataConfig(
        dataset=dataset_cfg["dataset"],
        data_path=dataset_cfg["data_path"],
        max_seq_len=int(model_cfg["max_seq_len"]),
        batch_size=int(train_cfg["batch_size"]),
    )

    data_bundle = prepare_dataloaders(data_cfg)
    print("Data stats:", data_bundle["stats"])

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    optimizer_name = (optimizer_override or train_cfg["optimizer"]).lower()

    train_config = TrainingConfig(
        epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        optimizer=optimizer_name,
        weight_decay=float(train_cfg["weight_decay"]),
        top_k=[int(k) for k in train_cfg["top_k"]],
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
            embedding_dim=int(model_cfg["embedding_dim"]),
            hidden_size=int(model_cfg["hidden_size"]),
            rnn_type=model_name,
            dropout=float(model_cfg["dropout"]),
        )

        artifact_dir = ensure_dir(ROOT_DIR / "artifacts" / dataset_cfg["dataset"] / f"{model_name}_{optimizer_name}")
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
            "dataset": dataset_cfg["dataset"],
            "model": model_name,
            "optimizer": optimizer_name,
            "num_items": num_items,
            "embedding_dim": int(model_cfg["embedding_dim"]),
            "hidden_size": int(model_cfg["hidden_size"]),
            "max_seq_len": int(model_cfg["max_seq_len"]),
            "dropout": float(model_cfg["dropout"]),
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
    dataset_cfg, model_cfg, _ = load_configs(config_path)

    model_name = (model_override or model_cfg["model"]).lower()
    if model_name == "all":
        model_name = "lstm"

    artifact_dir = ROOT_DIR / "artifacts" / dataset_cfg["dataset"] / model_name
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
    ui_path = ROOT_DIR / "src" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)


def list_datasets(config_path: str) -> List[str]:
    from src.dataset import DATASET_FILES
    cfg = read_yaml(config_path)
    dataset_cfg = {"data_path": cfg["data_path"]}
    data_path = Path(dataset_cfg.get("data_path", "./data/"))
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

    while True:
        dataset_cfg, model_cfg, train_cfg = load_configs(config_path)
        print("\n=== Movie Recommender Console ===")
        print(f"Current dataset: {dataset_cfg['dataset']}")
        print(f"Default model: {model_cfg['model']}")
        print(f"Default optimizer: {train_cfg['optimizer']}")
        print("1. Select dataset")
        print("2. Run Train")
        print("3. Run Comprehensive Comparison (RNN/LSTM/GRU + Adam/SGD)")
        print("4. Demo")
        print("5. Demo with UI")
        print("6. Exit")

        choice = input("Choose an option [1-6]: ").strip()

        try:
            if choice == "1":
                datasets = list_datasets(config_path)
                if not datasets:
                    print("No datasets found under data path in config.yml.")
                    continue

                print("\nAvailable datasets:")
                for idx, name in enumerate(datasets, start=1):
                    marker = " (current)" if name == dataset_cfg["dataset"] else ""
                    print(f"{idx}. {name}{marker}")

                raw_index = input("Select dataset number: ").strip()
                if not raw_index.isdigit() or int(raw_index) < 1 or int(raw_index) > len(datasets):
                    print("Invalid dataset selection.")
                    continue

                selected_dataset = datasets[int(raw_index) - 1]
                update_dataset(config_path, selected_dataset)
                print(f"Dataset updated to '{selected_dataset}'.")

            elif choice == "2":
                selected_model = prompt_model(model_cfg["model"], allow_all=True)
                selected_opt = prompt_optimizer(train_cfg["optimizer"])
                selected_device = prompt_device()
                run_training(
                    config_path=config_path,
                    model_override=selected_model,
                    optimizer_override=selected_opt,
                    device_override=selected_device,
                )

            elif choice == "3":
                print("\nStarting comprehensive comparison across models and optimizers...")
                selected_device = prompt_device()
                for opt in ["adam", "sgd"]:
                    print(f"\n>>> TESTING OPTIMIZER: {opt.upper()} <<<")
                    run_training(
                        config_path=config_path,
                        model_override="all",
                        optimizer_override=opt,
                        device_override=selected_device,
                    )

            elif choice == "4":
                selected_model = prompt_model(model_cfg["model"], allow_all=False)
                run_demo(
                    config_path=config_path,
                    model_override=selected_model,
                )

            elif choice == "5":
                run_ui()

            elif choice == "6":
                print("Exiting.")
                return

            else:
                print("Invalid option, select 1-6.")

        except FileNotFoundError as exc:
            print(f"Error: {exc}")
        except Exception as exc:
            print(f"Operation failed: {exc}")


def main() -> None:
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C).")
