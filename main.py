from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.dataset import DATASET_FILES, DataConfig, prepare_dataloaders
from src.model import NextMovieModel
from src.settings import DEFAULTS, ROOT_DIR, SUPPORTED_DATASETS, SUPPORTED_MODELS
from src.utils import (
    RecommenderService,
    TrainingConfig,
    ensure_dir,
    evaluate_model,
    format_comparison_table,
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
        service = RecommenderService.from_default()
        recs = service.recommend(payload.user_sequence, top_k=payload.top_k)
        return RecommendResponse(recommendations=recs)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def run_training(
    dataset: str,
    model_override: str | None = None,
    optimizer_override: str | None = None,
    device_override: str | None = None,
) -> Dict[str, Dict[str, float]]:
    dataset_name = dataset.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    selected_model = (model_override or DEFAULTS.model).lower()
    if selected_model == "all":
        model_names = list(SUPPORTED_MODELS)
    elif selected_model in SUPPORTED_MODELS:
        model_names = [selected_model]
    else:
        raise ValueError("Model must be rnn, lstm, gru, or all")

    data_cfg = DataConfig(
        dataset=dataset_name,
        data_path=DEFAULTS.data_path,
        max_seq_len=DEFAULTS.max_seq_len,
        batch_size=DEFAULTS.batch_size,
    )

    data_bundle = prepare_dataloaders(data_cfg)
    print("Data stats:", data_bundle["stats"])

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    optimizer_name = (optimizer_override or DEFAULTS.optimizer).lower()

    train_config = TrainingConfig(
        epochs=DEFAULTS.epochs,
        learning_rate=DEFAULTS.learning_rate,
        optimizer=optimizer_name,
        weight_decay=DEFAULTS.weight_decay,
        top_k=list(DEFAULTS.top_k),
    )

    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    num_items = int(data_bundle["num_items"])

    comparison: Dict[str, Dict[str, float]] = {}
    trained_model_keys: list[str] = []

    for model_name in model_names:
        print(f"\n=== Training {model_name.upper()} with {optimizer_name.upper()} ===")
        model = NextMovieModel(
            num_items=num_items,
            embedding_dim=DEFAULTS.embedding_dim,
            hidden_size=DEFAULTS.hidden_size,
            rnn_type=model_name,
            dropout=DEFAULTS.dropout,
        )

        artifact_dir = ensure_dir(ROOT_DIR / "artifacts" / dataset_name)
        model_key = f"{model_name}_{optimizer_name}"
        model_path = artifact_dir / f"{model_key}.pth"

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
            "dataset": dataset_name,
            "model": model_name,
            "optimizer": optimizer_name,
            "num_items": num_items,
            "embedding_dim": DEFAULTS.embedding_dim,
            "hidden_size": DEFAULTS.hidden_size,
            "max_seq_len": DEFAULTS.max_seq_len,
            "dropout": DEFAULTS.dropout,
            "item2idx": data_bundle["item2idx"],
            "idx2item": data_bundle["idx2item"],
            "best_val_metrics": best_val,
            "test_metrics": test_metrics,
            "history": history,
        }
        save_json(artifact_dir / f"{model_key}_metadata.json", metadata)
        trained_model_keys.append(model_key)
        print(f"Saved best model and metadata to: {artifact_dir}")

    validate_artifact_parity(dataset_name, trained_model_keys)
    print("\n=== Test Comparison (RNN vs LSTM vs GRU) ===")
    print(format_comparison_table(comparison))
    return comparison


def validate_artifact_parity(dataset: str, trained_model_keys: list[str]) -> None:
    dataset_dir = ROOT_DIR / "artifacts" / dataset
    expected_pth = sorted(f"{key}.pth" for key in trained_model_keys)
    actual_pth = sorted(p.name for p in dataset_dir.glob("*.pth"))
    actual_subset = sorted(name for name in actual_pth if name in expected_pth)

    if actual_subset != expected_pth:
        raise RuntimeError(
            "Artifact parity check failed. "
            f"Expected checkpoints: {expected_pth}. "
            f"Found matching checkpoints: {actual_subset} under {dataset_dir}."
        )

    missing_metadata = [
        f"{key}_metadata.json"
        for key in trained_model_keys
        if not (dataset_dir / f"{key}_metadata.json").exists()
    ]
    if missing_metadata:
        raise RuntimeError(
            "Artifact parity check failed. Missing metadata files: "
            f"{missing_metadata} under {dataset_dir}."
        )


def _install_windows_asyncio_guard() -> None:
    if sys.platform != "win32":
        return

    original_call_connection_lost = asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost

    def _safe_call_connection_lost(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            original_call_connection_lost(self, *args, **kwargs)
        except ConnectionResetError:
            return
        except OSError as err:
            if getattr(err, "winerror", None) == 10054:
                return
            raise

    asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost


def run_ui() -> None:
    _install_windows_asyncio_guard()
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


def set_dataset_interactive(current: str) -> str:
    supported = sorted(DATASET_FILES.keys())
    print(f"\nCurrent dataset: {current}")
    print(f"Supported datasets: {', '.join(supported)}")
    choice = input(f"Set dataset [{current}]: ").strip().lower()
    if not choice:
        print(f"Dataset unchanged: {current}")
        return current
    if choice not in supported:
        print(f"Invalid dataset '{choice}'. Supported: {', '.join(supported)}")
        return current
    print(f"Dataset updated to: {choice}")
    return choice


def interactive_menu() -> None:
    dataset_name = DEFAULTS.dataset

    while True:
        print("\n=== Movie Recommender ===")
        print(f"Dataset: {dataset_name}")
        print(f"Config: {DEFAULTS.model} + {DEFAULTS.optimizer}")
        print("\n1. Set/Get Dataset")
        print("2. Train Models (all or one)")
        print("3. Run Streamlit")
        print("4. Exit")

        choice = input("Choose an option [1-4]: ").strip()

        try:
            if choice == "1":
                dataset_name = set_dataset_interactive(dataset_name)

            elif choice == "2":
                selected_model = prompt_model(DEFAULTS.model)
                selected_device = prompt_device()
                run_training(
                    dataset=dataset_name,
                    model_override=selected_model,
                    optimizer_override=DEFAULTS.optimizer,
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
