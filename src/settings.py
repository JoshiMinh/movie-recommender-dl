from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

SUPPORTED_DATASETS = ("ml-1m", "ml-25m")
SUPPORTED_MODELS = ("rnn", "lstm", "gru")

DEFAULT_DATASET = "ml-1m"
DEFAULT_DATA_PATH = str(ROOT_DIR / "data")
DEFAULT_MODEL = "all"

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_MAX_SEQ_LEN = 10
DEFAULT_DROPOUT = 0.2

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_OPTIMIZER = "adam"
DEFAULT_TOP_K = (5, 10)
DEFAULT_WEIGHT_DECAY = 1e-5

DEFAULT_INFERENCE_TOP_K = 5
MAX_INFERENCE_TOP_K = 20
MAX_HISTORY_ITEMS = 30


@dataclass(frozen=True)
class DefaultConfig:
    dataset: str = DEFAULT_DATASET
    data_path: str = DEFAULT_DATA_PATH
    model: str = DEFAULT_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    dropout: float = DEFAULT_DROPOUT
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    optimizer: str = DEFAULT_OPTIMIZER
    top_k: tuple[int, ...] = DEFAULT_TOP_K
    weight_decay: float = DEFAULT_WEIGHT_DECAY


DEFAULTS = DefaultConfig()
