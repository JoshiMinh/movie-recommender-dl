"""Centralized configuration management for the movie recommender."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.utils import read_yaml


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    dataset: str
    data_path: str | Path
    max_seq_len: int
    batch_size: int


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model: str
    embedding_dim: int
    hidden_size: int
    max_seq_len: int
    dropout: float


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    top_k: list[int]
    weight_decay: float


class Config:
    """Centralized configuration loader."""

    _instance: Config | None = None
    _data: Dict[str, Any] | None = None

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Config:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yml. If None, uses ROOT_DIR/config.yml
            
        Returns:
            Config instance (singleton).
        """
        if config_path is None:
            config_path = ROOT_DIR / "config.yml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        cls._data = read_yaml(config_path)
        cls._instance = cls()
        return cls._instance

    @classmethod
    def get(cls) -> Config:
        """Get the loaded config instance."""
        if cls._instance is None:
            cls.load()
        return cls._instance

    @property
    def raw(self) -> Dict[str, Any]:
        """Get raw config dictionary."""
        if self._data is None:
            raise RuntimeError("Config not loaded. Call Config.load() first.")
        return self._data

    @property
    def dataset(self) -> DatasetConfig:
        """Get dataset configuration."""
        cfg = self.raw
        return DatasetConfig(
            dataset=cfg["dataset"],
            data_path=cfg["data_path"],
            max_seq_len=int(cfg["max_seq_len"]),
            batch_size=int(cfg["batch_size"]),
        )

    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        cfg = self.raw
        return ModelConfig(
            model=cfg["model"],
            embedding_dim=int(cfg["embedding_dim"]),
            hidden_size=int(cfg["hidden_size"]),
            max_seq_len=int(cfg["max_seq_len"]),
            dropout=float(cfg["dropout"]),
        )

    @property
    def train(self) -> TrainConfig:
        """Get training configuration."""
        cfg = self.raw
        return TrainConfig(
            batch_size=int(cfg["batch_size"]),
            epochs=int(cfg["epochs"]),
            learning_rate=float(cfg["learning_rate"]),
            optimizer=cfg["optimizer"],
            top_k=[int(k) for k in cfg["top_k"]],
            weight_decay=float(cfg["weight_decay"]),
        )
