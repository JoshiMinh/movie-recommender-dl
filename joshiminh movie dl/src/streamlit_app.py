from __future__ import annotations

import csv
import io
import re
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.api.inference import RecommenderService
from src.utils import read_yaml


st.set_page_config(
    page_title="Movie Recommender Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_configs() -> Tuple[dict, dict]:
    dataset_cfg = read_yaml(_root_dir() / "config" / "dataset.yaml")
    model_cfg = read_yaml(_root_dir() / "config" / "model.yaml")
    return dataset_cfg, model_cfg


@lru_cache(maxsize=8)
def load_movie_titles(dataset_name: str, data_path: str) -> Dict[int, str]:
    data_root = Path(data_path)
    if not data_root.is_absolute():
        data_root = _root_dir() / data_root

    dataset_root = data_root / dataset_name
    if dataset_name == "ml-100k":
        item_file = dataset_root / "u.item"
        if not item_file.exists():
            return {}

        df = pd.read_csv(
            item_file,
            sep="|",
            encoding="latin-1",
            header=None,
            usecols=[0, 1],
            names=["movie_id", "title"],
        )
        return dict(zip(df["movie_id"].astype(int), df["title"].astype(str)))

    movies_file = dataset_root / "movies.dat"
    if not movies_file.exists():
        return {}

    df = pd.read_csv(
        movies_file,
        sep="::",
        engine="python",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "title"],
        encoding="latin-1",
    )
    return dict(zip(df["movie_id"].astype(int), df["title"].astype(str)))


@lru_cache(maxsize=16)
def load_artifact_metadata(dataset_name: str, model_name: str) -> dict:
    metadata_path = _root_dir() / "artifacts" / dataset_name / model_name / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Artifact metadata not found: {metadata_path}")

    import json

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def load_service(dataset_name: str, model_name: str) -> RecommenderService:
    artifact_dir = _root_dir() / "artifacts" / dataset_name / model_name
    return RecommenderService(artifact_dir)


def list_available_models(dataset_name: str) -> List[str]:
    artifact_root = _root_dir() / "artifacts" / dataset_name
    if not artifact_root.exists():
        return []

    models: List[str] = []
    for candidate in sorted(artifact_root.iterdir()):
        if candidate.is_dir() and (candidate / "best_model.pt").exists() and (candidate / "metadata.json").exists():
            models.append(candidate.name)
    return models


def parse_sequence_input(raw_value: str, title_to_id: Dict[str, int]) -> Tuple[List[int], List[str]]:
    tokens: List[str] = []
    reader = csv.reader(io.StringIO(raw_value), skipinitialspace=True)
    for row in reader:
        for cell in row:
            parts = [part.strip() for part in cell.split(";") if part.strip()]
            tokens.extend(parts)

    if not tokens and raw_value.strip():
        tokens = [raw_value.strip()]

    sequence: List[int] = []
    unresolved: List[str] = []

    normalized_titles = {re.sub(r"\s+", " ", title.lower()).strip(): movie_id for title, movie_id in title_to_id.items()}

    for token in tokens:
        clean_token = token.strip().strip('"').strip("'")
        if not clean_token:
            continue

        if clean_token.isdigit():
            sequence.append(int(clean_token))
            continue

        id_match = re.search(r"\((\d+)\)\s*$", clean_token)
        if id_match:
            sequence.append(int(id_match.group(1)))
            continue

        normalized_token = re.sub(r"\s+", " ", clean_token.lower()).strip()
        movie_id = normalized_titles.get(normalized_token)
        if movie_id is not None:
            sequence.append(movie_id)
            continue

        matches = [movie_id for title, movie_id in title_to_id.items() if normalized_token in title.lower()]
        if len(matches) == 1:
            sequence.append(matches[0])
        else:
            unresolved.append(clean_token)

    return sequence, unresolved


def format_movie(movie_id: int, titles: Dict[int, str]) -> str:
    title = titles.get(movie_id)
    return f"{title} ({movie_id})" if title else f"Movie {movie_id}"


def render_metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_sample_sequence(sample_sequence: List[int]) -> None:
    st.session_state["watch_sequence"] = ", ".join(str(movie_id) for movie_id in sample_sequence)


def append_movie_to_sequence(movie_id: int) -> None:
    existing = st.session_state.get("watch_sequence", "").strip()
    st.session_state["watch_sequence"] = f"{existing}, {movie_id}" if existing else str(movie_id)


def main() -> None:
    dataset_cfg, model_cfg = load_configs()
    dataset_name = dataset_cfg["dataset"]
    data_path = dataset_cfg["data_path"]

    available_models = list_available_models(dataset_name)
    default_model = model_cfg["model"] if model_cfg["model"] in available_models else (available_models[0] if available_models else model_cfg["model"])

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

        :root {
            --bg: #050506;
            --panel: rgba(15, 15, 18, 0.75);
            --panel-border: rgba(255, 255, 255, 0.08);
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #e11d48;
            --accent-glow: rgba(225, 29, 72, 0.15);
            --card-bg: rgba(255, 255, 255, 0.03);
        }

        .stApp {
            background-color: var(--bg);
            background-image: 
                radial-gradient(circle at 0% 0%, rgba(225, 29, 72, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 100% 100%, rgba(225, 29, 72, 0.08) 0%, transparent 40%);
            color: var(--text);
            font-family: "Inter", -apple-system, sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            max-width: 1100px;
        }

        .hero {
            padding: 2.5rem 2rem;
            border: 1px solid var(--panel-border);
            border-radius: 28px;
            background: var(--panel);
            backdrop-filter: blur(20px);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .hero::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), transparent);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.2em;
            font-size: 0.75rem;
            color: var(--accent);
            margin-bottom: 0.75rem;
            font-weight: 800;
        }

        .hero h1 {
            margin: 0;
            font-family: "Outfit", sans-serif;
            font-size: clamp(2.5rem, 5vw, 4rem);
            font-weight: 800;
            line-height: 1.1;
            color: white;
            letter-spacing: -0.02em;
        }

        .hero p {
            margin: 1.25rem 0 0;
            max-width: 65ch;
            font-size: 1.1rem;
            color: var(--muted);
            line-height: 1.6;
        }

        .metric-card {
            padding: 1.25rem;
            border-radius: 20px;
            border: 1px solid var(--panel-border);
            background: var(--card-bg);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            border-color: rgba(225, 29, 72, 0.3);
            background: rgba(225, 29, 72, 0.02);
        }

        .metric-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--muted);
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .metric-value {
            font-family: "Outfit", sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.25rem;
        }

        .metric-help {
            color: var(--muted);
            font-size: 0.8rem;
        }

        .section-title {
            font-family: "Outfit", sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: white;
            margin: 2.5rem 0 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .section-title::after {
            content: "";
            height: 1px;
            flex-grow: 1;
            background: var(--panel-border);
        }

        .recommendation-card {
            padding: 1.25rem;
            border-radius: 20px;
            border: 1px solid var(--panel-border);
            background: var(--card-bg);
            margin-bottom: 1rem;
            transition: transform 0.2s ease;
        }

        .recommendation-card:hover {
            transform: translateX(4px);
            border-color: rgba(225, 29, 72, 0.4);
        }

        .recommendation-rank {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--accent);
            margin-bottom: 0.4rem;
            font-weight: 700;
        }

        .recommendation-title {
            font-size: 1.15rem;
            color: white;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .recommendation-subtitle {
            font-size: 0.9rem;
            color: var(--muted);
        }

        /* Streamlit Overrides */
        .stMultiSelect [data-baseweb="select"] > div {
            border-radius: 16px;
            background: var(--panel);
            border: 1px solid var(--panel-border);
        }

        .stMultiSelect [data-baseweb="tag"] {
            background: var(--accent) !important;
            border-radius: 8px;
            color: white !important;
        }

        .stSelectbox [data-baseweb="select"] > div {
            border-radius: 14px;
            background: var(--panel);
            border: 1px solid var(--panel-border);
        }

        [data-testid="stExpander"] {
            border-radius: 20px;
            background: var(--panel);
            border: 1px solid var(--panel-border) !important;
        }

        [data-testid="stMetricValue"] {
            color: white;
        }

        .stButton button {
            border-radius: 14px;
            border: none;
            background: var(--accent);
            color: white;
            padding: 0.6rem 1.5rem;
            font-weight: 700;
            transition: all 0.2s ease;
            width: 100%;
        }

        .stButton button:hover {
            background: #f43f5e;
            box-shadow: 0 0 20px rgba(225, 29, 72, 0.4);
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: var(--accent);
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background-color: var(--accent) !important;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 12px;
            background-color: var(--panel);
            border: 1px solid var(--panel-border);
            color: var(--muted);
            padding: 0 1.5rem;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--accent) !important;
            color: white !important;
            border-color: var(--accent) !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--panel-border);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Deep learning powered</div>
          <h1>Recommender Studio</h1>
          <p>Explore movie recommendations using state-of-the-art sequential models. Build your watch history below and let the neural engine predict your next favorite film.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    titles = load_movie_titles(dataset_name, data_path)
    
    tab_studio, tab_dashboard = st.tabs(["🎯 Recommender Studio", "📊 Performance Dashboard"])

    with tab_studio:
        # --- CONFIGURATION PANEL ---
        st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
        
        m_col_select, m_col1, m_col2, m_col3 = st.columns([1.5, 1, 1, 1])
        
        with m_col_select:
            model_options = available_models or [default_model]
            selected_model = st.selectbox(
                "Neural Checkpoint",
                options=model_options,
                index=model_options.index(default_model) if default_model in model_options else 0,
                key="model_selector"
            )
        
        try:
            selected_metadata = load_artifact_metadata(dataset_name, selected_model)
            
            with m_col1:
                render_metric_card("Dataset", dataset_name, "Active source")
            with m_col2:
                render_metric_card("Architecture", selected_metadata.get("model", "N/A").upper(), "Network type")
            with m_col3:
                render_metric_card("Embedding", f"{selected_metadata.get('hidden_size', 'N/A')}d", "Vector width")

            # --- SEQUENCE BUILDER ---
            st.markdown('<div class="section-title">Sequence Builder</div>', unsafe_allow_html=True)
            
            movie_options = sorted(titles.keys(), key=lambda x: titles[x].lower())
            
            builder_container = st.container()
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
            
            if ctrl_col1.button("✨ Load Sample", use_container_width=True):
                st.session_state["sequence_multiselect"] = list(titles.keys())[:5]
                st.rerun()

            with builder_container:
                selected_ids = st.multiselect(
                    "Your Watch History",
                    options=movie_options,
                    format_func=lambda x: titles.get(x, f"Movie {x}"),
                    placeholder="Search and select movies you've watched...",
                    help="The order of selection represents your watch sequence.",
                    key="sequence_multiselect"
                )
            
            with ctrl_col2:
                run_clicked = st.button("🚀 Generate", use_container_width=True, type="primary")

            with ctrl_col3:
                top_k = st.slider("Predictions", min_value=1, max_value=20, value=5, label_visibility="collapsed")
            
            if run_clicked:
                if not selected_ids:
                    st.error("Please add at least one movie to your history.")
                else:
                    with st.spinner("Analyzing patterns..."):
                        service = load_service(dataset_name, selected_model)
                        recommendations = service.recommend(selected_ids, top_k=top_k)
                        
                        st.markdown('<div class="section-title">Predicted Next Watches</div>', unsafe_allow_html=True)
                        
                        if not recommendations:
                            st.info("No new recommendations found for this sequence.")
                        else:
                            grid_cols = st.columns(2)
                            for idx, movie_id in enumerate(recommendations):
                                with grid_cols[idx % 2]:
                                    t = titles.get(movie_id, "Unknown Movie")
                                    st.markdown(
                                        f"""
                                        <div class="recommendation-card">
                                          <div class="recommendation-rank">Recommendation #{idx+1}</div>
                                          <div class="recommendation-title">{t}</div>
                                          <div class="recommendation-subtitle">ID: {movie_id}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
        except Exception as e:
            st.error(f"Error loading model: {e}")

    with tab_dashboard:
        st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
        
        if not available_models:
            st.warning("No trained models found. Run training in the console first.")
        else:
            comparison_data = []
            histories = {}
            
            for m_name in available_models:
                meta = load_artifact_metadata(dataset_name, m_name)
                metrics = meta.get("test_metrics", {})
                comparison_data.append({
                    "Model": m_name.upper(),
                    "Optimizer": meta.get("optimizer", "unknown").upper(),
                    "Hit@5": metrics.get("Hit@5", 0.0),
                    "Hit@10": metrics.get("Hit@10", 0.0),
                    "NDCG@10": metrics.get("NDCG@10", 0.0),
                })
                if "history" in meta:
                    histories[m_name] = meta["history"]

            # Comparison Table
            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(
                df_comp,
                use_container_width=True,
                hide_index=True,
            )

            # Loss Curves
            st.markdown('<div class="section-title">Training Loss Curves</div>', unsafe_allow_html=True)
            if not histories:
                st.info("No training history available for current models. Retrain to see loss curves.")
            else:
                loss_df_list = []
                for m_name, hist in histories.items():
                    if "train_loss" in hist:
                        for epoch, loss in enumerate(hist["train_loss"]):
                            loss_df_list.append({
                                "Epoch": epoch + 1,
                                "Loss": loss,
                                "Model": m_name.upper()
                            })
                
                if loss_df_list:
                    loss_df = pd.DataFrame(loss_df_list)
                    # Use native streamlit line chart for wide support
                    chart_data = loss_df.pivot(index="Epoch", columns="Model", values="Loss")
                    st.line_chart(chart_data)
                else:
                    st.info("Training loss data is empty.")

if __name__ == "__main__":
    main()
