from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

from src.api.inference import RecommenderService
from src.data.dataset import DataConfig, prepare_dataloaders
from src.evaluation.metrics import evaluate_model, format_comparison_table
from src.models.sequence_model import NextMovieModel
from src.training.trainer import TrainingConfig, train_one_model
from src.utils import ROOT_DIR, ensure_dir, read_yaml, save_json


def load_configs(dataset_cfg_path: str, model_cfg_path: str, train_cfg_path: str) -> Tuple[dict, dict, dict]:
    dataset_cfg = read_yaml(dataset_cfg_path)
    model_cfg = read_yaml(model_cfg_path)
    train_cfg = read_yaml(train_cfg_path)
    return dataset_cfg, model_cfg, train_cfg


def run_training(
    dataset_cfg_path: str,
    model_cfg_path: str,
    train_cfg_path: str,
    model_override: str | None = None,
) -> Dict[str, Dict[str, float]]:
    dataset_cfg, model_cfg, train_cfg = load_configs(
        dataset_cfg_path,
        model_cfg_path,
        train_cfg_path,
    )

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_config = TrainingConfig(
        epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        optimizer=str(train_cfg["optimizer"]),
        weight_decay=float(train_cfg["weight_decay"]),
        top_k=[int(k) for k in train_cfg["top_k"]],
    )

    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    num_items = int(data_bundle["num_items"])

    comparison: Dict[str, Dict[str, float]] = {}

    for model_name in model_names:
        print(f"\n=== Training {model_name.upper()} ===")
        model = NextMovieModel(
            num_items=num_items,
            embedding_dim=int(model_cfg["embedding_dim"]),
            hidden_size=int(model_cfg["hidden_size"]),
            rnn_type=model_name,
            dropout=float(model_cfg["dropout"]),
        )

        artifact_dir = ensure_dir(ROOT_DIR / "artifacts" / dataset_cfg["dataset"] / model_name)
        model_path = artifact_dir / "best_model.pt"

        try:
            model, best_val = train_one_model(
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
            "num_items": num_items,
            "embedding_dim": int(model_cfg["embedding_dim"]),
            "hidden_size": int(model_cfg["hidden_size"]),
            "max_seq_len": int(model_cfg["max_seq_len"]),
            "dropout": float(model_cfg["dropout"]),
            "item2idx": data_bundle["item2idx"],
            "idx2item": data_bundle["idx2item"],
            "best_val_metrics": best_val,
            "test_metrics": test_metrics,
        }
        save_json(artifact_dir / "metadata.json", metadata)
        print(f"Saved best model to: {model_path}")

    print("\n=== Test Comparison (RNN vs LSTM vs GRU) ===")
    print(format_comparison_table(comparison))
    return comparison


def run_demo(
    dataset_cfg_path: str,
    model_cfg_path: str,
    train_cfg_path: str,
    model_override: str | None = None,
) -> None:
    dataset_cfg, model_cfg, _ = load_configs(
        dataset_cfg_path,
        model_cfg_path,
        train_cfg_path,
    )

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


def list_datasets(dataset_cfg_path: str) -> List[str]:
    dataset_cfg = read_yaml(dataset_cfg_path)
    data_path = Path(dataset_cfg.get("data_path", "./data/"))
    if not data_path.is_absolute():
        data_path = ROOT_DIR / data_path

    if not data_path.exists():
        return []

    return sorted([entry.name for entry in data_path.iterdir() if entry.is_dir()])


def update_dataset(dataset_cfg_path: str, dataset_name: str) -> None:
    dataset_cfg = read_yaml(dataset_cfg_path)
    dataset_cfg["dataset"] = dataset_name
    with Path(dataset_cfg_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_cfg, f, sort_keys=False)


def prompt_model(default_model: str, allow_all: bool = True) -> str:
    allowed = ["rnn", "lstm", "gru"] + (["all"] if allow_all else [])
    prompt_suffix = " / ".join(allowed)
    raw = input(f"Model ({prompt_suffix}) [{default_model}]: ").strip().lower()
    model = raw or default_model.lower()
    if model not in allowed:
        print(f"Invalid model '{model}', using default '{default_model}'.")
        return default_model.lower()
    return model


def interactive_menu() -> None:
    dataset_cfg_path = str(ROOT_DIR / "config" / "dataset.yaml")
    model_cfg_path = str(ROOT_DIR / "config" / "model.yaml")
    train_cfg_path = str(ROOT_DIR / "config" / "train.yaml")

    while True:
        dataset_cfg, model_cfg, _ = load_configs(dataset_cfg_path, model_cfg_path, train_cfg_path)
        print("\n=== Movie Recommender Console ===")
        print(f"Current dataset: {dataset_cfg['dataset']}")
        print(f"Default model: {model_cfg['model']}")
        print("1. Select dataset")
        print("2. Run Train")
        print("3. Demo")
        print("4. Demo with UI")
        print("5. Exit")

        choice = input("Choose an option [1-5]: ").strip()

        try:
            if choice == "1":
                datasets = list_datasets(dataset_cfg_path)
                if not datasets:
                    print("No datasets found under data path in config/dataset.yaml.")
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
                update_dataset(dataset_cfg_path, selected_dataset)
                print(f"Dataset updated to '{selected_dataset}'.")

            elif choice == "2":
                selected_model = prompt_model(model_cfg["model"], allow_all=True)
                run_training(
                    dataset_cfg_path=dataset_cfg_path,
                    model_cfg_path=model_cfg_path,
                    train_cfg_path=train_cfg_path,
                    model_override=selected_model,
                )

            elif choice == "3":
                selected_model = prompt_model(model_cfg["model"], allow_all=False)
                run_demo(
                    dataset_cfg_path=dataset_cfg_path,
                    model_cfg_path=model_cfg_path,
                    train_cfg_path=train_cfg_path,
                    model_override=selected_model,
                )

            elif choice == "4":
                run_ui()

            elif choice == "5":
                print("Exiting.")
                return

            else:
                print("Invalid option, select 1-5.")

        except FileNotFoundError as exc:
            print(f"Error: {exc}")
        except Exception as exc:
            print(f"Operation failed: {exc}")


def main() -> None:
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C).")
