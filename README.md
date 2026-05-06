# Movie Recommendation System - Deep Learning Edition

An end-to-end sequence recommendation project for predicting the next movie a user is likely to watch.
The system is built around recurrent neural networks (RNN, LSTM, GRU), trained on MovieLens data,
and delivered with notebook, CLI, Streamlit, and FastAPI workflows.

## Overview

This repository implements next-item recommendation as a sequence modeling problem.
Given a user's historical watch sequence, the model predicts a probability distribution over movies
and returns the top-k candidates.

The codebase is designed for:
- Reproducible experimentation.
- Architecture and optimizer comparison.
- Practical local deployment for interactive demos.
- Easy extension for future recommender research.

## Key Features

- Sequential recommendation using temporal interaction history.
- Three recurrent backbones: RNN, LSTM, GRU.
- Config-driven training with a single config file.
- Automated model comparison across optimizers.
- Interactive Streamlit app for recommendation testing.
- FastAPI endpoint for programmatic inference.
- OOM-aware training utility in notebook workflow.
- Saved artifacts and metadata for reproducibility.

## Repository Structure

```text
movie-recommender-dl/
├── main.py
├── config.yml
├── Movie_Recommendation_Pipeline.ipynb
├── REPORT.md
├── README.md
├── requirements.txt
├── artifacts/
├── data/
└── src/
    ├── app.py
    ├── config.py
    ├── dataset.py
    ├── inference.py
    ├── metrics.py
    ├── model.py
    ├── streamlit.py
    ├── train.py
    └── utils.py
```

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
pip install -r requirements.txt
```

Optional CUDA setup (if you want GPU acceleration):

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify PyTorch device availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start

### 1. Run the Interactive CLI

```bash
python main.py
```

Current CLI options:
- `0`: automated workflow (all models with Adam and SGD).
- `1`: launch Streamlit UI.
- `2`: exit.

### 2. Run Streamlit Directly

```bash
streamlit run src/streamlit.py
```

### 3. Run FastAPI Service

```bash
uvicorn src.app:app --reload
```

### 4. Run the Notebook Pipeline

Open and run:
- `Movie_Recommendation_Pipeline.ipynb`

The notebook includes:
- Data loading and preprocessing.
- RNN/LSTM/GRU architecture comparison.
- Optimizer and learning-rate tuning.
- Interactive widget-based recommendation demo.

## Configuration

The project uses a single config file: `config.yml`.

Current defaults:

```yaml
dataset: ml-1m
data_path: ./data/
model: lstm
embedding_dim: 64
hidden_size: 128
max_seq_len: 10
dropout: 0.2
batch_size: 256
epochs: 5
learning_rate: 0.001
optimizer: adam
top_k: [5, 10]
weight_decay: 1e-05
```

Notes:
- `model` supports `rnn`, `lstm`, `gru`, and CLI override `all`.
- `max_seq_len` controls truncation length for user history.
- `top_k` drives evaluation and ranking output.

## Data Pipeline and Leakage Prevention

The training and evaluation flow follows sequence-aware best practices:
- Per-user chronological split.
- Train/validation/test separation by time.
- Sequence construction without cross-split leakage.
- Padding index reserved for variable-length batching.

This is critical for realistic next-item evaluation.

## Modeling Approach

Core architecture components:
- Item embedding layer.
- Recurrent encoder (RNN/LSTM/GRU).
- Dense output layer with softmax-style ranking over item space.

Why sequence models:
- User preferences are temporal.
- Order and recency of watched movies matter.
- Recurrent units offer a strong baseline with manageable complexity.

## Evaluation Metrics

The project tracks ranking and exact-hit quality:

1. Top-1 Accuracy
Formula: $\text{Top-1} = \frac{\#\text{correct argmax predictions}}{\#\text{samples}}$

2. Hit@10
Formula: $\text{Hit@10} = \frac{\#\text{targets in top-10}}{\#\text{samples}}$

3. NDCG@10
Measures ranking quality with stronger weight on higher-ranked relevant items.

## Consolidated Report Findings

The prior report is merged here for a single source of truth.

### Requirement Compliance

All core project requirements were implemented:
- Next-item recommendation on MovieLens.
- User/item embedding based recurrent model.
- RNN, LSTM, GRU architecture support.
- Chronological split and leakage prevention.
- Hyperparameter and optimizer comparison.
- Metric reporting and visualization.
- Interactive demo and deployable interfaces.

### Architecture Comparison (Reported)

| Architecture | Top-1 Accuracy | Hit@10 | Training Time | Memory |
|---|---:|---:|---:|---:|
| RNN | 22.3% | 58.7% | 12 min | 1240 MB |
| LSTM | 26.8% | 63.2% | 18 min | 1580 MB |
| GRU | 25.4% | 61.9% | 15 min | 1420 MB |

Interpretation:
- LSTM delivered the best overall quality.
- GRU was a good efficiency-quality compromise.
- Vanilla RNN worked as a useful baseline.

### Optimization Findings (Reported)

Learning rate comparison:

| Learning Rate | Final Val Loss | Convergence | Stability |
|---|---:|---|---|
| 1e-2 | 3.72 | Fast | Unstable |
| 1e-3 | 2.48 | Moderate | Stable |
| 1e-4 | 2.51 | Slow | Over-damped |

Optimizer comparison:

| Optimizer | Final Val Loss | Behavior |
|---|---:|---|
| Adam | 2.42 | Better final quality, adaptive updates |
| SGD + momentum | 2.68 | Stable but lower final quality |

Regularization takeaway:
- Dropout around 0.3 gave the best reported generalization tradeoff in report experiments.

## Production Readiness Summary

The repository includes practical deployment paths:
- Streamlit app for interactive recommendations and comparison dashboards.
- FastAPI app for service-style inference integration.
- Artifact and metadata saving for versioned models.

From the report, observed serving characteristics were suitable for local production-style demos,
with low-latency single-request inference and acceptable batch throughput.

## Limitations

- Cold-start behavior remains a challenge for users/items with limited history.
- Popular-item bias can reduce novelty.
- Current recurrent architecture can be outperformed by attention-based models on some datasets.

## Future Work

Recommended next steps:
1. Add attention or Transformer-based sequential models.
2. Incorporate side information (genre, metadata, demographics).
3. Add debiasing or calibration for popularity effects.
4. Evaluate with additional benchmarks and temporal holdouts.
5. Add online/continual learning support.

## Reproducibility

To keep experiments repeatable:
- Use fixed random seeds.
- Keep `config.yml` committed alongside artifacts.
- Save model checkpoints and metadata under `artifacts/`.
- Track dataset variant (`ml-100k` vs `ml-1m`) in every run.

## Citation

```bibtex
@software{movie_recommender_2026,
  author = {JoshiMinh and Jade2308},
  title = {Movie Recommendation System - Deep Learning Edition},
  year = {2026},
  url = {https://github.com/JoshiMinh/movie-recommender-dl}
}
```

## License

MIT License. See `LICENSE`.

## Authors

- JoshiMinh
- Jade2308

## Support and Contributions

- Open an issue for bugs, questions, or feature requests.
- Pull requests are welcome.
- For major changes, include a short experiment summary and metrics.

---

Last updated: May 6, 2026