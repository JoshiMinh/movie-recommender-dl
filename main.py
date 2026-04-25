from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from src.api.inference import RecommenderService
from src.data.dataset import DataConfig, prepare_dataloaders
from src.evaluation.metrics import evaluate_model, format_comparison_table
from src.models.sequence_model import NextMovieModel
from src.training.trainer import TrainingConfig, train_one_model
from src.utils import ensure_dir, read_yaml, save_json


def load_configs(dataset_cfg_path: str, model_cfg_path: str, train_cfg_path: str):
    dataset_cfg = read_yaml(dataset_cfg_path)
    model_cfg = read_yaml(model_cfg_path)
    train_cfg = read_yaml(train_cfg_path)
    return dataset_cfg, model_cfg, train_cfg


def run_training(args) -> Dict[str, Dict[str, float]]:
    dataset_cfg, model_cfg, train_cfg = load_configs(
        args.dataset_config,
        args.model_config,
        args.train_config,
    )

    selected_model = (args.model or model_cfg["model"]).lower()
    if selected_model == "all":
        model_names = ["rnn", "lstm", "gru"]
    elif selected_model in {"rnn", "lstm", "gru"}:
        model_names = [selected_model]
    else:
        raise ValueError("--model must be rnn, lstm, gru, or all")

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

        artifact_dir = ensure_dir(Path("artifacts") / dataset_cfg["dataset"] / model_name)
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


def run_demo(args) -> None:
    dataset_cfg, model_cfg, _ = load_configs(
        args.dataset_config,
        args.model_config,
        args.train_config,
    )

    model_name = (args.model or model_cfg["model"]).lower()
    if model_name == "all":
        model_name = "lstm"

    artifact_dir = Path("artifacts") / dataset_cfg["dataset"] / model_name
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Artifact {artifact_dir} does not exist. Run training first."
        )

    service = RecommenderService(artifact_dir)
    sample = [1, 5, 20]
    recs = service.recommend(sample, top_k=3)

    print("Demo request:", {"user_sequence": sample})
    print("Demo response:", {"recommendations": recs})


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning Movie Recommender")
    parser.add_argument("--train", action="store_true", help="Train model(s)")
    parser.add_argument("--demo", action="store_true", help="Run local recommendation demo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model type: rnn | lstm | gru | all",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="config/dataset.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model.yaml",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="config/train.yaml",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()

        if not args.train and not args.demo:
            raise ValueError("Use at least one mode: --train and/or --demo")

        if args.train:
            run_training(args)

        if args.demo:
            run_demo(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user (Ctrl+C).")


if __name__ == "__main__":
    main()
