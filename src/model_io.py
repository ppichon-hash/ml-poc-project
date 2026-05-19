"""Helpers for loading serialized models."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path so joblib can reconstruct KMeansWrapper
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

class KMeansWrapper:
    def __init__(self, kmeans, scaler):
        self.kmeans = kmeans
        self.scaler = scaler
    
    def predict(self, X):
        import pandas as pd
        X_scaled = self.scaler.transform(X.astype(float))
        return self.kmeans.predict(X_scaled)


def load_model(model_path: Path) -> Any:
    """Load a serialized model from disk.

    Supported formats are `.joblib`, `.pkl`, and `.pickle`.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    suffix = model_path.suffix.lower()

    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "Loading `.joblib` files requires the `joblib` package. "
                "Add it to requirements.txt if needed."
            ) from exc

        return joblib.load(model_path)

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as file_handle:
            return pickle.load(file_handle)

    raise ValueError(
        f"Unsupported model format for {model_path}. Use .joblib, .pkl, or .pickle."
    )
