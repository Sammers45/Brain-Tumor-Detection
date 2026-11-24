"""Load the YAML config and resolve paths relative to the project root."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


def resolve(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    """Make runs repeatable (as far as TensorFlow on CPU allows)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:  # the data stage does not need TensorFlow
        pass
