from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansWrapper:
    def __init__(self, kmeans: KMeans, scaler: StandardScaler) -> None:
        self.kmeans = kmeans
        self.scaler = scaler

    def fit(self, X) -> "KMeansWrapper":
        X_arr = _to_array(X)
        self.scaler.fit(X_arr)
        self.kmeans.fit(self.scaler.transform(X_arr))
        return self

    def predict(self, X) -> np.ndarray:
        X_arr = _to_array(X)
        return self.kmeans.predict(self.scaler.transform(X_arr))


def _to_array(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.fillna(0).astype(float).values
    return np.nan_to_num(np.array(X, dtype=float))
