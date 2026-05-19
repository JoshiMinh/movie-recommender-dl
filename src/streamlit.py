from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import torch

from src.config import Config
from src.utils import ExperimentInput, build_comparison_table, list_runs, run_experiment
from src.model import NextMovieModel


Config.load()
st.set_page_config(page_title="MovieLens Recommender Lab", page_icon="🎬", layout="wide")


def _config_defaults() -> dict:
    cfg = Config.get()
    return {
        "dataset": cfg.dataset.dataset,
        "data_path": cfg.dataset.data_path,
        "model": cfg.model.model,
        "embedding_dim": cfg.model.embedding_dim,
        "hidden_size": cfg.model.hidden_size,
        "max_seq_len": cfg.model.max_seq_len,
        "dropout": cfg.model.dropout,
        "batch_size": cfg.train.batch_size,
        "epochs": cfg.train.epochs,
        "learning_rate": cfg.train.learning_rate,
        "optimizer": cfg.train.optimizer,
        "weight_decay": cfg.train.weight_decay,
        "top_k": cfg.train.top_k,
    }


def _history_frame(runs: List[Dict[str, object]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in runs:
        hist = run.get("history", {})
        train_losses = hist.get("train_loss", [])
        for idx, loss in enumerate(train_losses):
            rows.append(
                {
                    "Epoch": idx + 1,
                    "Loss": loss,
                    "Run": run.get("run_label") or run.get("run_id"),
                    "Model": run.get("params", {}).get("model", ""),
                    "Optimizer": run.get("params", {}).get("optimizer", ""),
                }
            )
    return pd.DataFrame(rows)


def _recommend(run_meta: Dict[str, object], user_sequence: List[int], top_k: int) -> List[int]:
    run_id = str(run_meta["run_id"])
    dataset = str(run_meta["params"]["dataset"])
    run_dir = Path("artifacts") / "experiments" / dataset / run_id
    state = torch.load(run_dir / "best_model.pt", map_location="cpu")

    idx2item = {int(k): int(v) for k, v in run_meta["idx2item"].items()}
    item2idx = {int(k): int(v) for k, v in run_meta["item2idx"].items()}
    params = run_meta["params"]
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
    for seen in encoded:
        logits[0, seen] = float("-inf")
    logits[0, 0] = float("-inf")
    top_idx = torch.topk(logits, k=min(top_k, logits.shape[1] - 1), dim=1).indices[0].tolist()
    return [idx2item.get(int(i), 0) for i in top_idx if idx2item.get(int(i), 0) != 0]


def _label_frame(metric_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = [
        {
            "label": label,
            "support": int(values.get("support", 0)),
            "top1_accuracy": values.get("top1_accuracy", 0.0),
            "topk_hit": values.get("topk_hit", 0.0),
        }
        for label, values in metric_map.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["label", "support", "top1_accuracy", "topk_hit"])
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def main() -> None:
    defaults = _config_defaults()
    st.title("MovieLens Recommender Lab")
    st.caption("Remote MovieLens (1M/25M), training, evaluation, comparisons, and interactive inference in one app.")

    with st.sidebar:
        st.header("Run Setup")
        dataset = st.selectbox("Dataset", ["ml-1m", "ml-25m"], index=0)
        mode = st.selectbox("Mode", ["Quick (1M-friendly)", "Full"], index=0)
        model = st.selectbox("Model", ["rnn", "lstm", "gru"], index=1)
        optimizer = st.selectbox("Optimizer", ["adam", "sgd"], index=0)
        lr = st.number_input("Learning Rate", value=float(defaults["learning_rate"]), format="%.6f")
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=int(defaults["epochs"]))
        batch_size = st.number_input("Batch Size", min_value=16, max_value=4096, value=int(defaults["batch_size"]))
        max_interactions = 1_000_000 if mode.startswith("Quick") else 0
        if dataset == "ml-25m" and mode.startswith("Quick"):
            st.info("Quick mode limits ratings to first 1,000,000 interactions.")

        col_a, col_b = st.columns(2)
        run_baseline = col_a.button("Run Baseline", use_container_width=True)
        run_optimized = col_b.button("Run Optimized", use_container_width=True)

    if run_baseline or run_optimized:
        label = "baseline" if run_baseline else "optimized"
        with st.spinner(f"Running {label} experiment..."):
            exp = ExperimentInput(
                dataset=dataset,
                data_path=str(defaults["data_path"]),
                model=defaults["model"] if run_baseline else model,
                embedding_dim=int(defaults["embedding_dim"]),
                hidden_size=int(defaults["hidden_size"]),
                max_seq_len=int(defaults["max_seq_len"]),
                dropout=float(defaults["dropout"]),
                batch_size=int(defaults["batch_size"] if run_baseline else batch_size),
                epochs=int(defaults["epochs"] if run_baseline else epochs),
                learning_rate=float(defaults["learning_rate"] if run_baseline else lr),
                optimizer=str(defaults["optimizer"] if run_baseline else optimizer),
                weight_decay=float(defaults["weight_decay"]),
                top_k=[5, 10],
                max_interactions=max_interactions if max_interactions > 0 else None,
                is_baseline=bool(run_baseline),
                run_label=label,
            )
            result = run_experiment(exp)
        st.success(f"Completed run: {result['run_id']}")

    runs = list_runs(dataset)
    comp = build_comparison_table(runs)

    tab_data, tab_arch, tab_opt, tab_base, tab_label, tab_demo = st.tabs(
        [
            "Data & Run Control",
            "Architecture Comparison",
            "Optimizer/LR Comparison",
            "Baseline vs Optimized",
            "Label Analytics",
            "Interactive Demo",
        ]
    )

    with tab_data:
        st.subheader("Run Registry")
        if comp.empty:
            st.warning("No runs yet. Start by running baseline and optimized experiments.")
        else:
            st.dataframe(comp, use_container_width=True, hide_index=True)
            stats_cols = st.columns(4)
            stats_cols[0].metric("Total Runs", len(comp))
            stats_cols[1].metric("Baselines", int(comp["Baseline"].sum()))
            stats_cols[2].metric("Best Hit@10", f"{comp['Hit@10'].max():.4f}")
            stats_cols[3].metric("Best NDCG@10", f"{comp['NDCG@10'].max():.4f}")

    with tab_arch:
        st.subheader("Architecture Results")
        if comp.empty:
            st.info("No data available.")
        else:
            arch_df = comp.groupby("Model", as_index=False)[["Hit@5", "Hit@10", "NDCG@10"]].max()
            st.dataframe(arch_df, use_container_width=True, hide_index=True)
            st.bar_chart(arch_df.set_index("Model")[["Hit@10", "NDCG@10"]])
            hist_df = _history_frame(runs)
            if not hist_df.empty:
                st.line_chart(hist_df.pivot_table(index="Epoch", columns="Model", values="Loss", aggfunc="mean"))

    with tab_opt:
        st.subheader("Optimizer and Learning Rate")
        if comp.empty:
            st.info("No data available.")
        else:
            opt_df = comp.groupby(["Optimizer", "LR"], as_index=False)[["Hit@10", "NDCG@10"]].max()
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
            st.bar_chart(opt_df.set_index(opt_df["Optimizer"] + "@" + opt_df["LR"].astype(str))[["Hit@10", "NDCG@10"]])
            hist_df = _history_frame(runs)
            if not hist_df.empty:
                st.line_chart(hist_df.pivot_table(index="Epoch", columns="Optimizer", values="Loss", aggfunc="mean"))

    with tab_base:
        st.subheader("Baseline vs Optimized")
        base = [r for r in runs if r.get("is_baseline")]
        opt = [r for r in runs if not r.get("is_baseline")]
        if not base or not opt:
            st.info("Need at least one baseline and one optimized run.")
        else:
            base_latest = base[0]
            opt_latest = opt[0]
            base_metrics = base_latest.get("test_metrics", {})
            opt_metrics = opt_latest.get("test_metrics", {})
            compare_rows = []
            for metric in sorted(set(base_metrics.keys()).intersection(opt_metrics.keys())):
                b = float(base_metrics.get(metric, 0.0))
                o = float(opt_metrics.get(metric, 0.0))
                compare_rows.append({"metric": metric, "baseline": b, "optimized": o, "delta": o - b})
            compare_df = pd.DataFrame(compare_rows)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            st.bar_chart(compare_df.set_index("metric")[["baseline", "optimized"]])

    with tab_label:
        st.subheader("Genre and Topic Accuracy")
        if not runs:
            st.info("No runs yet.")
        else:
            selected_id = st.selectbox(
                "Run",
                options=[r["run_id"] for r in runs],
                format_func=lambda rid: f"{rid} ({next((x['run_label'] for x in runs if x['run_id']==rid), '')})",
            )
            selected = next(r for r in runs if r["run_id"] == selected_id)

            gdf = _label_frame(selected.get("genre_metrics", {}))
            tdf = _label_frame(selected.get("topic_metrics", {}))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Genre Metrics**")
                st.dataframe(gdf, use_container_width=True, hide_index=True)
                if not gdf.empty:
                    st.bar_chart(gdf.head(20).set_index("label")[["top1_accuracy", "topk_hit"]])
            with col2:
                st.markdown("**Topic Metrics**")
                if tdf.empty:
                    st.info("No topic data available for this run/dataset.")
                else:
                    st.dataframe(tdf, use_container_width=True, hide_index=True)
                    st.bar_chart(tdf.head(20).set_index("label")[["top1_accuracy", "topk_hit"]])

    with tab_demo:
        st.subheader("Interactive Recommendation Demo")
        if not runs:
            st.warning("Run an experiment first.")
        else:
            selected_id = st.selectbox("Inference Run", options=[r["run_id"] for r in runs], key="inference_run_id")
            selected = next(r for r in runs if r["run_id"] == selected_id)
            titles = {int(k): str(v) for k, v in selected.get("movie_titles", {}).items()}
            if not titles:
                st.warning("Movie title metadata missing.")
            else:
                options = sorted(titles.keys(), key=lambda x: titles[x].lower())
                history = st.multiselect("Watch History", options=options, format_func=lambda m: f"{titles[m]} ({m})")
                k = st.slider("Top-K", 1, 20, 5)
                if st.button("Predict Next", type="primary"):
                    recs = _recommend(selected, history, k)
                    if not recs:
                        st.info("No recommendations found.")
                    else:
                        for i, movie_id in enumerate(recs, start=1):
                            st.write(f"{i}. {titles.get(movie_id, f'Movie {movie_id}')} ({movie_id})")


if __name__ == "__main__":
    main()
