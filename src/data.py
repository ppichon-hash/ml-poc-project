from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR

FEATURE_COLS = [
    # Conditions environnementales (observables / planifiables par Vinci)
    "lum", "atm",
    # Infrastructure routière
    "circ", "nbv", "prof", "surf", "infra", "situ", "vma",
    # Temporel / saisonnalité
    "mois", "saison",
]


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv(DATA_DIR / "processed_dataset.csv")

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df["target"]

    # Remplacer les NaN par la médiane de chaque colonne
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
