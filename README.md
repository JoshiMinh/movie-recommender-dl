# MovieLens Recommender Demo

This project provides MovieLens sequence-model training utilities and an inference-only Streamlit demo.

## Streamlit Demo

The Streamlit app is focused on recommendation inference, not training.

UI controls:
- dataset picker (`ml-1m`, `ml-25m`)
- model picker (`all`, `rnn`, `lstm`, `gru`) based on available runs
- movie history input
- run prediction button

When `all` is selected, recommendations are shown separately per model.

## Run

```bash
pip install -r requirements.txt
streamlit run src/streamlit.py
```

## Notes

- Inference metadata is loaded from `artifacts/<dataset>/<model>_<optimizer>_metadata.json`.
- Checkpoints are loaded from `artifacts/<dataset>/<model>_<optimizer>.pth`.
- If no runs are available for a dataset, Streamlit shows a warning instead of failing.
- Default runtime values are centralized in `src/settings.py`.
