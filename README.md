# movie-recommender-dl

Deep learning next-movie recommendation using user watch sequence:

- Input: `[m1, m2, m3]`
- Output: predicted next movie `m4`

Implements three sequence models with a shared embedding concept:

- RNN
- LSTM
- GRU

Everything runs 100% locally.

## Project Structure

```text
movie-recommender-dl/
├── config/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/
├── src/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   ├── cli.py
│   └── streamlit_app.py
├── main.py
├── requirements.txt
└── README.md
```

## Dataset Config (Single Switch)

Edit only `config/dataset.yaml` to swap datasets:

```yaml
dataset: ml-100k  # switch to ml-1m later
data_path: ./data/
```

No code changes are needed to switch between MovieLens 100K and 1M.

## Setup

```bash
pip install -r requirements.txt
```

## Run Console Menu

Launch the interactive console menu:

```bash
python main.py
```

Menu options:

- Select dataset
- Run Train
- Demo
- Demo with UI

Example output format:

```json
{
  "recommendations": [42, 18, 7]
}
```

## Streamlit UI

Launch the interactive local UI:

```bash
streamlit run src/streamlit_app.py
```

Or use the console menu (`Demo with UI`) from:

```bash
python main.py
```

The UI lets you paste movie IDs or titles, switch between trained models, and view ranked recommendations with MovieLens titles when metadata is available.

## Local API (FastAPI)

Start API server:

```bash
uvicorn src.api.app:app --reload
```

Request:

```http
POST /recommend
Content-Type: application/json

{
  "user_sequence": [1, 5, 20],
  "top_k": 3
}
```

Response:

```json
{
  "recommendations": [42, 18, 7]
}
```

## Configurable Parameters

In `config/model.yaml`:

- `model` (`rnn` / `lstm` / `gru`)
- `embedding_dim`
- `hidden_size`
- `max_seq_len`
- `dropout`

In `config/train.yaml`:

- `epochs`
- `batch_size`
- `learning_rate`
- `optimizer` (`sgd` / `adam`)
- `weight_decay`
- `top_k`

## Notes

- The loader sorts interactions by timestamp per user.
- Training samples are built as sequence-to-next-item pairs.
- Zero-padding is used for short sequences.
- Best checkpoints are saved under `artifacts/<dataset>/<model>/best_model.pt`.
- Metadata and ID mappings are saved to `artifacts/<dataset>/<model>/metadata.json`.
