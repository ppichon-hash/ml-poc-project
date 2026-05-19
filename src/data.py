from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from config import DATA_DIR

# Colonnes utilisees par les modeles ML
FEATURE_COLS = [
    "rarity_encoded",
    "set_age",
    "price_range",
    "is_holo",
    "hp_normalized",
    "has_evolution",
    "is_reverse",
    "nb_attacks",
    "market_price",
    "low_price",
    "high_price",
]


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv(DATA_DIR / "pokemon_cards.csv")

    # Garder uniquement les colonnes presentes dans le dataset
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputation des NaN par la mediane (fit sur train, transform sur les deux)
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

    return X_train, X_test, y_train, y_test
