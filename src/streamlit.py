from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st
import torch

from src.model import NextMovieModel
from src.settings import (
    DEFAULT_DATASET,
    MAX_HISTORY_ITEMS,
    MAX_INFERENCE_TOP_K,
    SUPPORTED_DATASETS,
    SUPPORTED_MODELS,
)
from src.utils import list_runs


def _install_windows_asyncio_guard() -> None:
    if sys.platform != "win32":
        return
    original_call_connection_lost = asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost

    def _safe_call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
        try:
            original_call_connection_lost(self, exc)
        except ConnectionResetError:
            return
        except OSError as err:
            if getattr(err, "winerror", None) == 10054:
                return
            raise

    asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost


def _recommend(run_meta: Dict[str, object], user_sequence: List[int], top_k: int) -> List[int]:
    dataset = str(run_meta["params"]["dataset"])
    params = run_meta["params"]
    model_name = str(params["model"])
    optimizer = str(params["optimizer"])
    run_dir = Path("artifacts") / dataset
    state = torch.load(run_dir / f"{model_name}_{optimizer}.pth", map_location="cpu")

    idx2item = {int(k): int(v) for k, v in run_meta["idx2item"].items()}
    item2idx = {int(k): int(v) for k, v in run_meta["item2idx"].items()}
    model = NextMovieModel(
        num_items=int(max(idx2item.keys())) + 1,
        embedding_dim=int(params["embedding_dim"]),
        hidden_size=int(params["hidden_size"]),
        rnn_type=str(params["model"]),
        dropout=float(params["dropout"]),
    )
    model.load_state_dict(state)
    model.eval()

    encoded = [item2idx[m] for m in user_sequence if m in item2idx]
    if not encoded:
        return []

    max_seq_len = int(params["max_seq_len"])
    seq = encoded[-max_seq_len:]
    x = [0] * (max_seq_len - len(seq)) + seq
    model_input = torch.tensor([x], dtype=torch.long)
    logits = model(model_input)
    logits[0, 0] = float("-inf")
    for seen in set(encoded):
        logits[0, seen] = float("-inf")
    top_idx = torch.topk(logits, k=min(top_k, logits.shape[1] - 1), dim=1).indices[0].tolist()
    return [idx2item.get(int(i), 0) for i in top_idx if idx2item.get(int(i), 0) != 0]


def _latest_runs_by_model(runs: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    by_model: Dict[str, Dict[str, object]] = {}
    for model_name in SUPPORTED_MODELS:
        hit = next((r for r in runs if str(r.get("params", {}).get("model", "")).lower() == model_name), None)
        if hit:
            by_model[model_name] = hit
    return by_model


def main() -> None:
    _install_windows_asyncio_guard()
    st.set_page_config(page_title="Movie Recommender Demo", page_icon="🎬", layout="wide")
    st.title("Movie Recommender Demo")
    st.caption("Inference-only Streamlit demo over existing experiment artifacts.")

    with st.sidebar:
        st.header("Demo Controls")
        dataset = st.selectbox(
            "Dataset",
            list(SUPPORTED_DATASETS),
            index=list(SUPPORTED_DATASETS).index(DEFAULT_DATASET),
        )

    runs = list_runs(dataset)
    latest_by_model = _latest_runs_by_model(runs)
    available_models = list(latest_by_model.keys())
    model_options = ["all"] + available_models

    with st.sidebar:
        selected_model = st.selectbox("Model", model_options if model_options else ["all"], index=0)

    if not runs:
        st.warning(f"No experiment runs found for dataset '{dataset}'.")
        st.info("Run training first so inference metadata (movie titles and index maps) is available.")
        return

    if not available_models:
        st.warning(f"No model artifacts available for dataset '{dataset}'.")
        return

    base_run = runs[0]
    titles = {int(k): str(v) for k, v in base_run.get("movie_titles", {}).items()}
    if not titles:
        st.warning("Movie title metadata is missing for this dataset.")
        return

    options = sorted(titles.keys(), key=lambda x: titles[x].lower())
    history = st.multiselect(
        "Movie Input (watch history)",
        options=options,
        format_func=lambda m: f"{titles[m]} ({m})",
        max_selections=MAX_HISTORY_ITEMS,
    )
    recommend_amount = st.slider("Recommend amount", 1, MAX_INFERENCE_TOP_K, 5)

    if st.button("Run Prediction Recommend", type="primary"):
        if not history:
            st.info("Select at least one movie.")
            return

        targets: Dict[str, Dict[str, object]] = {}
        if selected_model == "all":
            targets = latest_by_model
        elif selected_model in latest_by_model:
            targets = {selected_model: latest_by_model[selected_model]}
        else:
            st.warning(f"Selected model '{selected_model}' has no available run for dataset '{dataset}'.")
            return

        for model_name, run_meta in targets.items():
            recs = _recommend(run_meta, history, recommend_amount)
            st.subheader(f"Model: {model_name.upper()}")
            if not recs:
                st.info("No recommendations found.")
                continue
            for i, movie_id in enumerate(recs, start=1):
                st.write(f"{i}. {titles.get(movie_id, f'Movie {movie_id}')} ({movie_id})")


if __name__ == "__main__":
    main()
