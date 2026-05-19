"""KMeansWrapper: scales features then predicts cluster labels."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansWrapper:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self._cluster_labels: dict[int, str] = {
            0: "Cartes communes",
            1: "Cartes rares",
            2: "Cartes collector",
        }

    def fit(self, X) -> "KMeansWrapper":
        X_scaled = self.scaler.fit_transform(X.astype(float))
        self.kmeans.fit(X_scaled)
        self._order_clusters(X)
        return self

    def _order_clusters(self, X) -> None:
        """Remap cluster ids so 0=cheap, 1=mid, 2=expensive by market_price."""
        import pandas as pd
        X_scaled = self.scaler.transform(X.astype(float))
        raw_labels = self.kmeans.predict(X_scaled)
        df = pd.DataFrame(X).copy()
        df["_cluster"] = raw_labels
        if "market_price" in df.columns:
            price_col = "market_price"
        else:
            price_col = df.columns[0]
        means = df.groupby("_cluster")[price_col].mean().sort_values()
        self._remap = {old: new for new, old in enumerate(means.index)}

    def predict(self, X) -> np.ndarray:
        X_scaled = self.scaler.transform(X.astype(float))
        raw = self.kmeans.predict(X_scaled)
        if hasattr(self, "_remap"):
            return np.array([self._remap[c] for c in raw])
        return raw

    def fit_predict(self, X) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    @property
    def cluster_labels(self) -> dict[int, str]:
        return self._cluster_labels
