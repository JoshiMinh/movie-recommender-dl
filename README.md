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
│   └── training/
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

## Train

Train all models and print comparison table:

```bash
python main.py --train --model all
```

Train a single model:

```bash
python main.py --train --model lstm
```

## Demo

```bash
python main.py --demo
```

Example output format:

```json
{
  "recommendations": [42, 18, 7]
}
```

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
