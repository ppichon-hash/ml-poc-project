"""Entraîne et sauvegarde les 3 modèles ML sur le dataset BAAC autoroutes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb

from config import DATA_DIR, MODELS_DIR, MODEL_METRICS_FILE
from data import FEATURE_COLS, load_dataset_split
from kmeans_wrapper import KMeansWrapper
from metrics import compute_metrics

X_train, X_test, y_train, y_test = load_dataset_split()
print(f"Train : {X_train.shape}  Test : {X_test.shape}")
print(f"Target (train) :\n{y_train.value_counts().to_string()}")

scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
metrics_rows: list[dict] = []

# ── Random Forest ──────────────────────────────────────────────────────────────
print("\nEntraînement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_path = MODELS_DIR / "random_forest.joblib"
joblib.dump(rf, rf_path)
print(f"  Sauvegardé : {rf_path}")

y_pred_rf = rf.predict(X_test)
m = compute_metrics(y_test, y_pred_rf)
print(f"  Accuracy : {m['accuracy']:.4f}  F1 : {m['f1']:.4f}")
metrics_rows.append({"model_key": "random_forest", "model_name": "Random Forest",
                     "model_path": str(rf_path), **{k: round(v, 4) for k, v in m.items()}})

# ── XGBoost (hyperparamètres optimisés par RandomizedSearchCV) ────────────────
print("Entraînement XGBoost (tuned)...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.6,
    min_child_weight=3,
    gamma=0,
    reg_alpha=0.5,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_train.astype(float), y_train)
xgb_path = MODELS_DIR / "xgboost.joblib"
joblib.dump(xgb, xgb_path)
print(f"  Sauvegardé : {xgb_path}")

y_pred_xgb = xgb.predict(X_test.astype(float))
m = compute_metrics(y_test, y_pred_xgb)
print(f"  Accuracy : {m['accuracy']:.4f}  F1 : {m['f1']:.4f}")
metrics_rows.append({"model_key": "xgboost", "model_name": "XGBoost",
                     "model_path": str(xgb_path), **{k: round(v, 4) for k, v in m.items()}})

# ── LightGBM (hyperparamètres optimisés par RandomizedSearchCV) ───────────────
print("Entraînement LightGBM (tuned)...")
lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.6,
    min_child_samples=10,
    reg_alpha=0.5,
    reg_lambda=0.1,
    scale_pos_weight=scale_pos,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgbm.fit(X_train, y_train)
lgbm_path = MODELS_DIR / "lightgbm.joblib"
joblib.dump(lgbm, lgbm_path)
print(f"  Sauvegardé : {lgbm_path}")

y_pred_lgbm = lgbm.predict(X_test)
m = compute_metrics(y_test, y_pred_lgbm)
print(f"  Accuracy : {m['accuracy']:.4f}  F1 : {m['f1']:.4f}")
metrics_rows.append({"model_key": "lightgbm", "model_name": "LightGBM",
                     "model_path": str(lgbm_path), **{k: round(v, 4) for k, v in m.items()}})

# ── KMeans ────────────────────────────────────────────────────────────────────
print("Entraînement KMeans (3 clusters)...")
df_full = pd.read_csv(DATA_DIR / "processed_dataset.csv")
available = [c for c in FEATURE_COLS if c in df_full.columns]
X_all = df_full[available].fillna(df_full[available].median()).astype(float)

scaler  = StandardScaler()
kmeans  = KMeans(n_clusters=3, random_state=42, n_init=10)
wrapper = KMeansWrapper(kmeans=kmeans, scaler=scaler)
wrapper.fit(X_all)

kmeans_path = MODELS_DIR / "kmeans.joblib"
joblib.dump(wrapper, kmeans_path)
print(f"  Sauvegardé : {kmeans_path}")

# Pseudo-accuracy : mapper chaque cluster vers le label majoritaire
clusters_test = wrapper.predict(X_test)
cluster_s = pd.Series(clusters_test)
label_s   = y_test.reset_index(drop=True)
mapping   = {c: int(label_s[cluster_s == c].mode().iloc[0])
             for c in cluster_s.unique() if not label_s[cluster_s == c].empty}
y_pred_km = cluster_s.map(mapping)
m = compute_metrics(label_s, y_pred_km)
print(f"  Pseudo-accuracy : {m['accuracy']:.4f}")
metrics_rows.append({"model_key": "kmeans", "model_name": "KMeans",
                     "model_path": str(kmeans_path), **{k: round(v, 4) for k, v in m.items()}})

# ── Métriques ──────────────────────────────────────────────────────────────────
mdf = pd.DataFrame(metrics_rows)
mdf.to_csv(MODEL_METRICS_FILE, index=False)
print(f"\nMétriques sauvegardées : {MODEL_METRICS_FILE}")
print(mdf.to_string(index=False))
print("\nTous les modèles entraînés avec succès.")
