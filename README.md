# MovieLens Recommender Lab (Streamlit-First)

This project is a Streamlit-first movie recommendation system using sequence models (RNN/LSTM/GRU) and MovieLens data.

## What It Does

- Pulls **MovieLens 1M** or **MovieLens 25M** from official URLs (remote-first; no preloaded dataset required).
- Runs the same preprocessing/training/evaluation pipeline for both datasets.
- Recomputes baseline runs from current code defaults.
- Runs optimized experiments (model/optimizer/LR/epochs/batch size).
- Persists experiment artifacts and metrics for comparison.
- Compares model quality with:
  - ranking metrics (`Hit@K`, `NDCG@K`)
  - label-based accuracy by **genres**
  - label-based accuracy by **themes/topics** (MovieLens genome tags when available).

## Streamlit Tabs

The app provides:

1. `Data & Run Control`
2. `Architecture Comparison`
3. `Optimizer/LR Comparison`
4. `Baseline vs Optimized`
5. `Label Analytics`
6. `Interactive Demo`

Notebook-era outputs are recreated inside the app:
- architecture comparison table + curves
- optimizer/LR comparison table + curves
- interactive recommendation workflow

## Run

```bash
pip install -r requirements.txt
streamlit run src/streamlit.py
```

## Notes

- Default quick workflow is intended for fast `ml-1m` iteration.
- For `ml-25m`, use full mode for complete runs, or quick mode for faster testing.
- Experiment outputs are saved under `artifacts/experiments/`.
- The `data/` folder is a disposable cache now. You can delete it at any time; datasets will be pulled again automatically on next run.
