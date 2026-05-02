# PROJECT SPECIFICATION (STRICT): Movie Recommendation System using Deep Learning

## ROLE
You are an **Expert PyTorch Machine Learning Engineer**.  
Your task is to implement **Project 5: Movie Recommendation System** strictly following all requirements below.

---

## 1) PROJECT OVERVIEW

### 1.1 Goal
Build a **next-item movie recommendation system** that predicts the next movie a user is likely to watch.

### 1.2 Core Logic (Critical)
- This is a **sequential recommendation** task.
- Learn from **chronological interaction behavior** (implicit feedback).
- Although MovieLens provides ratings, do **not** model explicit rating regression.
- Treat each user’s interactions as an ordered sequence of watched items.

### 1.3 Dataset
Use **MovieLens 1M** with the following columns:
- `user_id`
- `movie_id`
- `rating`
- `timestamp`

(You may read `rating`, but training target is next-item classification, not rating prediction.)

### 1.4 Tech Stack
- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib

No non-standard dependencies (e.g., fuzzywuzzy/rapidfuzz/other external matching packages).

### 1.5 Hardware Target
- Optimize for **NVIDIA RTX 3060 (CUDA)**.
- Use GPU whenever available.

### 1.6 Reproducibility (Mandatory)
Set fixed global seed = `42` before any pipeline logic:
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)` (if CUDA)
- deterministic settings for reproducibility where appropriate.

---

## 2) REQUIRED PROJECT STRUCTURE

Implement exactly these modules/files:

1. `dataset.py`  
   Data loading, per-user chronological split, leakage-safe sequence construction, DataLoader + padding.

2. `model.py`  
   `SequenceRecommender` model with selectable recurrent backbone: `RNN`, `LSTM`, `GRU`.

3. `train.py`  
   Training/validation loops, optimizer comparison, LR tuning, OOM fallback, metric logging.

4. `evaluate.py`  
   Test-time evaluation and comparison tables (architecture + optimizer) with required metrics.

5. `Report_and_Demo.ipynb`  
   End-to-end orchestration, visualizations, result tables, and interactive recommendation demo.

Code must be modular, object-oriented, and PEP-8 compliant.

---

## 3) IMPLEMENTATION REQUIREMENTS (DETAILED)

## Step 1 — Data Preprocessing (`dataset.py`)

### 3.1.1 Per-user chronological split (Critical)
For **each user independently**, sort interactions by `timestamp` and split:
- Train: first 80%
- Validation: next 10%
- Test: final 10%

**Do not use random split.**

### 3.1.2 Leakage prevention (Critical)
Sequence samples must be created **independently inside each split**:
- Build train sequences only from train interactions.
- Build val sequences only from val interactions.
- Build test sequences only from test interactions.

No sequence may cross split boundaries.

### 3.1.3 Sequence construction
Use **sliding window** to generate `(input_sequence, target_movie)`:
- Input = list of historical movie IDs
- Target = next movie ID
- Maximum input length = **50** (truncate from the left if longer).

### 3.1.4 Vocabulary/indexing
Define consistent user/movie index mapping for model embeddings and targets:
- Ensure movie target indices align with classifier output dimension.
- Keep mapping artifacts accessible for inference/demo.

### 3.1.5 DataLoader + padding
Implement custom `collate_fn`:
- Right-pad sequences using `torch.nn.utils.rnn.pad_sequence(batch_first=True, padding_value=0)`
- Return:
  - `user_ids`
  - `padded_sequences`
  - `sequence_lengths` (true lengths before padding)
  - `targets`

`sequence_lengths` is mandatory for packed sequence handling.

---

## Step 2 — Model Architecture (`model.py`)

Create class `SequenceRecommender(nn.Module)` with configurable recurrent type (`rnn`, `lstm`, `gru`).

### 3.2.1 Embeddings (Mandatory)
- User embedding: `nn.Embedding(num_users, user_emb_dim, padding_idx=...)` if needed
- Movie embedding: `nn.Embedding(num_movies, movie_emb_dim, padding_idx=0)` (recommended for padded token)

### 3.2.2 User + movie fusion (Mandatory)
For each sample:
1. Get user embedding vector.
2. Repeat user embedding across sequence time dimension.
3. Concatenate with movie embedding at each timestep:
   `concat_t = [movie_emb_t ; user_emb]`

Pass concatenated sequence to recurrent layer.

### 3.2.3 Recurrent layer
Selectable backbone:
- `nn.RNN` or `nn.LSTM` or `nn.GRU`
- Include dropout support (e.g., candidates: `0.1`, `0.3`, `0.5`)

### 3.2.4 Packed sequence handling (Mandatory)
Use:
- `pack_padded_sequence(..., enforce_sorted=False)`
- recurrent forward on packed input
- `pad_packed_sequence(...)` if needed

### 3.2.5 Last valid hidden state (Critical)
Extract the hidden state corresponding to each sequence’s **last real timestep** (ignoring padding), using `sequence_lengths`.

Do **not** take the final padded timestep blindly.

### 3.2.6 Output head
- Final dense layer: `nn.Linear(hidden_dim, num_movies)`
- `forward(...)` must return **raw logits**.

### 3.2.7 Inference method
Implement separate `predict(...)` method:
- Applies `Softmax(dim=-1)` on logits
- Returns probabilities and/or top-k predictions

Softmax is for inference pipeline; not for training loss computation.

---

## Step 3 — Training & Optimization (`train.py`)

### 3.3.1 Device policy
- Use `cuda` if available, otherwise `cpu`.
- Move model and all tensors properly to device.

### 3.3.2 Loss (Mandatory)
Use:
- `nn.CrossEntropyLoss` on **raw logits** and class targets

Do **not** apply softmax before `CrossEntropyLoss`.

### 3.3.3 Batch size + OOM fallback (Mandatory)
- Start with `batch_size = 256`.
- Wrap training start/epoch execution in `try/except RuntimeError`.
- If CUDA OOM is detected:
  - clear cache if needed (`torch.cuda.empty_cache()`)
  - halve batch size: 256 → 128 → 64 → ...
  - retry automatically until success or minimum threshold.

### 3.3.4 Optimizer comparison
Compare at least:
- `SGD`
- `Adam`

### 3.3.5 Learning rate tuning
Evaluate multiple LRs (e.g.):
- `1e-2`
- `1e-3`
- `1e-4`

Apply same protocol for fair comparison.

### 3.3.6 Logging
Track and store per epoch:
- Training Loss
- Validation Loss

Optionally log additional validation metrics during training.

---

## Step 4 — Evaluation (`evaluate.py`)

### 3.4.1 Required metrics
Compute on test set:
1. **Top-1 Accuracy**
2. **Hit Ratio @10** (target appears in top-10 predicted items)

### 3.4.2 Architecture comparison (Mandatory)
Train/evaluate and report:
- RNN vs LSTM vs GRU

### 3.4.3 Optimizer comparison (Mandatory)
On the best architecture, compare:
- SGD vs Adam

### 3.4.4 Comparison tables (Mandatory)
Produce clear tables:
- Architecture performance table
- Optimizer performance table
- Include metric values and key hyperparameters.

---

## Step 5 — Notebook Report & Demo (`Report_and_Demo.ipynb`)

The notebook must include:

### 3.5.1 End-to-end pipeline
- Data loading
- Preprocessing
- Training
- Evaluation
- Inference demo

### 3.5.2 Visualizations
Plot **Training vs Validation Loss curves** for all major experiments.

### 3.5.3 Result summaries
Show final comparison tables clearly in notebook cells.

### 3.5.4 Interactive recommendation demo
Input example:
- `["Titanic", "Avatar", "Inception"]`

Mapping logic requirements:
- Implement simple string normalization:
  - lowercase
  - trim spaces
  - remove release year patterns like `(1997)`
- Match normalized titles against MovieLens metadata
- Convert to movie IDs
- Run best model inference
- Output recommended movie title string

No fuzzy matching libraries allowed.

---

## 4) EXPERIMENTAL PROTOCOL (FAIRNESS REQUIREMENTS)

- Keep data split fixed across all experiments.
- Keep random seed fixed for comparability.
- When comparing architecture/optimizer, control other hyperparameters as much as possible.
- Clearly state chosen best model and why (based on required metrics).

---

## 5) CODE QUALITY & ENGINEERING REQUIREMENTS

- PEP-8 compliant.
- Clear class/function docstrings.
- Type hints where practical.
- Avoid monolithic notebook-only logic; core logic must live in `.py` modules.
- Handle edge cases:
  - users with too few interactions
  - empty sequence generation in a split
  - padding index correctness
  - stable top-k computation when movie count is large

---

## 6) DEFINITION OF DONE (CHECKLIST)

A submission is complete only if **all** are satisfied:

- [ ] Per-user chronological split 80/10/10 implemented.
- [ ] No leakage across splits in sequence generation.
- [ ] Sliding window implemented with max sequence length = 50.
- [ ] Right-padding + sequence lengths returned by DataLoader.
- [ ] User embedding + movie embedding concatenated per timestep.
- [ ] Configurable RNN/LSTM/GRU architecture implemented.
- [ ] `pack_padded_sequence(..., enforce_sorted=False)` used correctly.
- [ ] Last valid hidden state extracted correctly (ignoring padding).
- [ ] `forward()` returns raw logits; `predict()` applies softmax.
- [ ] CrossEntropyLoss used correctly on logits.
- [ ] OOM fallback halves batch size from 256 automatically.
- [ ] SGD vs Adam comparison completed.
- [ ] LR tuning completed (at least 3 LR values).
- [ ] Top-1 Accuracy and Hit Ratio@10 reported on test set.
- [ ] Architecture comparison table + optimizer comparison table included.
- [ ] Notebook includes loss curves and interactive title-based demo.

---

## 7) FINAL REQUIRED OUTPUTS

Provide:
1. Full source code for:
   - `dataset.py`
   - `model.py`
   - `train.py`
   - `evaluate.py`
2. `Report_and_Demo.ipynb` with executable cells.
3. Final metrics and comparison tables.
4. Brief conclusion:
   - best architecture
   - best optimizer
   - best learning rate
   - observed trade-offs.

Implement everything exactly as specified.