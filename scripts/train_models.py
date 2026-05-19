"""Train Random Forest, XGBoost, and KMeans on Pokemon card data."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import DATA_DIR, MODELS_DIR
from data import FEATURE_COLS, load_dataset_split
from kmeans_wrapper import KMeansWrapper

X_train, X_test, y_train, y_test = load_dataset_split()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Target distribution (train):\n{y_train.value_counts().to_string()}")

# --- Random Forest ---
print("\nTraining Random Forest...")
sample_w = compute_sample_weight("balanced", y_train)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_path = MODELS_DIR / "random_forest.joblib"
joblib.dump(rf, rf_path)
print(f"  Saved: {rf_path}")

# --- XGBoost ---
print("Training XGBoost...")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_train.astype(float), y_train)
xgb_path = MODELS_DIR / "xgboost.joblib"
joblib.dump(xgb, xgb_path)
print(f"  Saved: {xgb_path}")

# --- KMeans ---
print("Training KMeans (3 clusters)...")
from sklearn.impute import SimpleImputer
df_full = pd.read_csv(DATA_DIR / "pokemon_cards.csv")
available_cols = [c for c in FEATURE_COLS if c in df_full.columns]
X_all_raw = df_full[available_cols]
imputer_km = SimpleImputer(strategy="median")
X_all = pd.DataFrame(imputer_km.fit_transform(X_all_raw), columns=X_all_raw.columns)
kmeans_wrapper = KMeansWrapper(n_clusters=3, random_state=42)
kmeans_wrapper.fit(X_all)
kmeans_path = MODELS_DIR / "kmeans.joblib"
joblib.dump(kmeans_wrapper, kmeans_path)
print(f"  Saved: {kmeans_path}")

# Quick validation
from sklearn.metrics import accuracy_score, classification_report
y_pred_rf = rf.predict(X_test)
y_pred_xgb = xgb.predict(X_test.astype(float))
print(f"\nRandom Forest accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))
print(f"XGBoost accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

clusters = kmeans_wrapper.predict(X_test.astype(float))
print(f"KMeans cluster counts: {pd.Series(clusters).value_counts().to_dict()}")
print("\nAll models trained and saved successfully.")
