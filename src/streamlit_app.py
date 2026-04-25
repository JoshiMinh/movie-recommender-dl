from __future__ import annotations

import re
from functools import lru_cache
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
    tokens = [token.strip() for token in re.split(r"[,\n;]+", raw_value) if token.strip()]
    sequence: List[int] = []
    unresolved: List[str] = []

    normalized_titles = {title.lower(): movie_id for title, movie_id in title_to_id.items()}

    for token in tokens:
        if token.isdigit():
            sequence.append(int(token))
            continue

        movie_id = normalized_titles.get(token.lower())
        if movie_id is not None:
            sequence.append(movie_id)
            continue

        matches = [movie_id for title, movie_id in title_to_id.items() if token.lower() in title.lower()]
        if len(matches) == 1:
            sequence.append(matches[0])
        else:
            unresolved.append(token)

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


def main() -> None:
    dataset_cfg, model_cfg = load_configs()
    dataset_name = dataset_cfg["dataset"]
    data_path = dataset_cfg["data_path"]

    available_models = list_available_models(dataset_name)
    default_model = model_cfg["model"] if model_cfg["model"] in available_models else (available_models[0] if available_models else model_cfg["model"])

    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe6;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-border: rgba(26, 29, 35, 0.09);
            --text: #17181d;
            --muted: #5e6370;
            --accent: #b04e2e;
            --accent-2: #2a6f97;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(176, 78, 46, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(42, 111, 151, 0.13), transparent 24%),
                linear-gradient(180deg, #fbf7f1 0%, #f4efe6 100%);
            color: var(--text);
        }

        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            background: var(--panel);
            backdrop-filter: blur(14px);
            box-shadow: 0 20px 50px rgba(17, 17, 17, 0.06);
            margin-bottom: 1rem;
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: var(--accent-2);
            margin-bottom: 0.45rem;
            font-weight: 700;
        }

        .hero h1 {
            margin: 0;
            font-family: Georgia, Cambria, serif;
            font-size: clamp(2.1rem, 4vw, 3.8rem);
            line-height: 1.03;
            color: var(--text);
        }

        .hero p {
            margin: 0.85rem 0 0;
            max-width: 70ch;
            font-size: 1.03rem;
            color: var(--muted);
        }

        .metric-card {
            padding: 1rem 1rem 0.9rem;
            border-radius: 20px;
            border: 1px solid var(--panel-border);
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 12px 30px rgba(17, 17, 17, 0.05);
            height: 100%;
        }

        .metric-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin-bottom: 0.55rem;
        }

        .metric-value {
            font-family: Georgia, Cambria, serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.35rem;
        }

        .metric-help {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .section-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin: 1.4rem 0 0.65rem;
            font-weight: 700;
        }

        .recommendation-card {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid var(--panel-border);
            background: rgba(255, 255, 255, 0.84);
            box-shadow: 0 14px 30px rgba(17, 17, 17, 0.05);
            margin-bottom: 0.7rem;
        }

        .recommendation-rank {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--accent);
            margin-bottom: 0.3rem;
            font-weight: 700;
        }

        .recommendation-title {
            font-size: 1.02rem;
            color: var(--text);
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .recommendation-subtitle {
            font-size: 0.88rem;
            color: var(--muted);
        }

        .stTextArea textarea {
            border-radius: 18px;
            border: 1px solid rgba(26, 29, 35, 0.12);
            background: rgba(255, 255, 255, 0.92);
        }

        .stButton button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, var(--accent) 0%, #d07b43 100%);
            color: white;
            padding: 0.65rem 1rem;
            font-weight: 700;
        }

        .stButton button:hover {
            box-shadow: 0 10px 20px rgba(176, 78, 46, 0.24);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Local deep learning recommender</div>
          <h1>Movie Recommender Studio</h1>
          <p>Paste a watch sequence, choose a trained model, and get the next-movie predictions from the local checkpoint. Titles are resolved from MovieLens metadata when available.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    titles = load_movie_titles(dataset_name, data_path)
    title_to_id = {title: movie_id for movie_id, title in titles.items()}

    if "watch_sequence" not in st.session_state:
        st.session_state.watch_sequence = "1, 5, 20"

    model_options = available_models or [default_model]

    selected_model = st.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
        help="Choose a trained artifact from artifacts/<dataset>/<model>/",
    )

    selected_metadata = load_artifact_metadata(dataset_name, selected_model)

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        render_metric_card("Dataset", dataset_name, "Configured in config/dataset.yaml")
    with col_b:
        render_metric_card("Model", selected_metadata["model"].upper(), "Loaded from artifacts")
    with col_c:
        render_metric_card("Sequence Length", str(selected_metadata["max_seq_len"]), "Model input window")
    with col_d:
        render_metric_card("Hidden Size", str(selected_metadata["hidden_size"]), "Sequence encoder width")

    st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)
    controls_left, controls_right = st.columns([1.25, 0.85])

    with controls_left:
        st.text_area(
            "Watch sequence",
            key="watch_sequence",
            height=120,
            help="Enter movie IDs, exact titles, or title fragments separated by commas, semicolons, or new lines.",
        )
        top_k = st.slider("Recommendations to return", min_value=1, max_value=20, value=3, step=1)

    with controls_right:
        st.markdown(
            """
            <div class="recommendation-card">
              <div class="recommendation-rank">How it works</div>
              <div class="recommendation-title">1. Pick a trained checkpoint</div>
              <div class="recommendation-subtitle">The app reads metadata and model weights from the local artifacts folder.</div>
            </div>
            <div class="recommendation-card">
              <div class="recommendation-rank">Input</div>
              <div class="recommendation-title">2. Enter a user sequence</div>
              <div class="recommendation-subtitle">Short sequences are padded automatically to match the model window.</div>
            </div>
            <div class="recommendation-card">
              <div class="recommendation-rank">Output</div>
              <div class="recommendation-title">3. Review ranked suggestions</div>
              <div class="recommendation-subtitle">Already-seen items are excluded from the recommendation list.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    button_col, sample_col = st.columns([0.3, 0.7])
    with button_col:
        run_clicked = st.button("Generate recommendations", use_container_width=True)
    with sample_col:
        sample_sequence = [movie_id for movie_id in list(titles.keys())[:3]] if titles else [1, 5, 20]
        st.button(
            "Use a sample sequence",
            use_container_width=True,
            on_click=apply_sample_sequence,
            args=(sample_sequence,),
        )

    sequence_input = st.session_state.watch_sequence
    sequence, unresolved = parse_sequence_input(sequence_input, title_to_id)

    if unresolved:
        st.warning("Some entries could not be matched: " + ", ".join(unresolved))

    if sequence:
        st.markdown('<div class="section-title">Parsed Sequence</div>', unsafe_allow_html=True)
        st.write(
            ", ".join(format_movie(movie_id, titles) for movie_id in sequence)
        )

    if run_clicked:
        if not sequence:
            st.error("Enter at least one valid movie ID or title before generating recommendations.")
            return

        try:
            service = load_service(dataset_name, selected_model)
            recommendations = service.recommend(sequence, top_k=top_k)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            return

        st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
        if not recommendations:
            st.info("No unseen recommendations were produced for the supplied sequence.")
            return

        for rank, movie_id in enumerate(recommendations, start=1):
            title = titles.get(movie_id, "Unknown title")
            st.markdown(
                f"""
                <div class="recommendation-card">
                  <div class="recommendation-rank">Rank {rank}</div>
                  <div class="recommendation-title">{title}</div>
                  <div class="recommendation-subtitle">Movie ID {movie_id}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
