"""
Prépare le dataset BAAC (accidents corporels sur autoroutes) pour le ML.

Pipeline :
  1. Charge les 4 fichiers × 5 années (2020-2024)
  2. Concatène les 5 années
  3. Fusionne les 4 tables sur Num_Acc
  4. Filtre catr == 1 (autoroutes uniquement — contexte Vinci)
  5. Supprime grav == -1
  6. Crée la target binaire (0 = léger, 1 = grave)
  7. Garde les features demandées et nettoie les valeurs -1/NaN
  8. Sauvegarde data/processed_dataset.csv

Usage : python scripts/prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np

DATA_DIR = ROOT / "data"

YEARS = [2020, 2021, 2022, 2023, 2024]

FEATURE_COLS = [
    "lum", "atm", "col",
    "circ", "nbv", "prof",
    "surf", "infra", "situ",
    "vma", "catv",
    "mois", "jour",
]

TARGET_COL = "target"


def _load_year(year: int) -> dict[str, pd.DataFrame]:
    """Charge les 4 fichiers BAAC pour une année donnée."""
    tables: dict[str, pd.DataFrame] = {}
    for table in ["caracteristiques", "lieux", "usagers", "vehicules"]:
        path = DATA_DIR / f"{table}-{year}.csv"
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)

        # 2022 : renommer Accident_Id → Num_Acc dans caracteristiques
        if "Accident_Id" in df.columns:
            df = df.rename(columns={"Accident_Id": "Num_Acc"})

        df["annee"] = year
        tables[table] = df
        print(f"  {table}-{year}.csv : {len(df):,} lignes")

    return tables


def main() -> None:
    print("=== Préparation du dataset BAAC Autoroutes ===\n")

    all_caract: list[pd.DataFrame] = []
    all_lieux:  list[pd.DataFrame] = []
    all_usag:   list[pd.DataFrame] = []
    all_veh:    list[pd.DataFrame] = []

    for year in YEARS:
        print(f"Chargement {year}...")
        t = _load_year(year)
        all_caract.append(t["caracteristiques"])
        all_lieux.append(t["lieux"])
        all_usag.append(t["usagers"])
        all_veh.append(t["vehicules"])

    caract = pd.concat(all_caract, ignore_index=True)
    lieux  = pd.concat(all_lieux,  ignore_index=True)
    usag   = pd.concat(all_usag,   ignore_index=True)
    veh    = pd.concat(all_veh,    ignore_index=True)

    print(f"\nTotaux bruts :")
    print(f"  caract : {len(caract):,}  lieux : {len(lieux):,}  usagers : {len(usag):,}  véhicules : {len(veh):,}")

    # ── Fusion sur Num_Acc ──────────────────────────────────────────────────────
    # usagers → un usager par ligne, garder grav (target) + colonnes utiles
    usag_keep = ["Num_Acc", "grav", "catu"]
    usag_sel  = usag[[c for c in usag_keep if c in usag.columns]].copy()

    # lieux : une ligne par accident
    lieux_keep = ["Num_Acc", "catr", "circ", "nbv", "prof", "surf", "infra", "situ", "vma"]
    lieux_sel  = lieux[[c for c in lieux_keep if c in lieux.columns]].drop_duplicates("Num_Acc")

    # véhicules : garder catv (type véhicule dominant par accident)
    veh_keep = ["Num_Acc", "catv"]
    veh_sel  = veh[[c for c in veh_keep if c in veh.columns]].copy()
    # catv : prendre la valeur la plus fréquente par accident
    veh_sel = (
        veh_sel.groupby("Num_Acc")["catv"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .reset_index()
    )

    # caract : une ligne par accident, colonnes utiles
    caract_keep = ["Num_Acc", "lum", "atm", "col", "mois", "jour", "annee", "dep", "lat", "long"]
    caract_sel  = caract[[c for c in caract_keep if c in caract.columns]].copy()

    # Merge progressif
    df = usag_sel.merge(caract_sel, on="Num_Acc", how="inner")
    df = df.merge(lieux_sel,  on="Num_Acc", how="left")
    df = df.merge(veh_sel,    on="Num_Acc", how="left")

    print(f"\nAprès fusion : {len(df):,} lignes (une par usager)")

    # ── Filtre autoroutes (catr == 1) ──────────────────────────────────────────
    df = df[df["catr"] == 1].copy()
    print(f"Après filtre autoroutes (catr=1) : {len(df):,} lignes")

    # ── Filtre grav != -1 ──────────────────────────────────────────────────────
    df["grav"] = pd.to_numeric(df["grav"], errors="coerce")
    df = df[df["grav"].notna() & (df["grav"] != -1)].copy()
    print(f"Après filtre grav != -1 : {len(df):,} lignes")

    # ── Target binaire ──────────────────────────────────────────────────────────
    # 0 = indemne (1) ou blessé léger (2)
    # 1 = hospitalisé (3) ou tué (4)
    df[TARGET_COL] = df["grav"].apply(lambda g: 0 if g in (1, 2) else 1)
    print(f"\nDistribution target :")
    print(f"  0 (léger)  : {(df[TARGET_COL]==0).sum():,}  ({(df[TARGET_COL]==0).mean()*100:.1f}%)")
    print(f"  1 (grave)  : {(df[TARGET_COL]==1).sum():,}  ({(df[TARGET_COL]==1).mean()*100:.1f}%)")

    # ── Nettoyage valeurs -1 ──────────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -1, col] = np.nan

    # ── lat/long ──────────────────────────────────────────────────────────────
    for coord in ["lat", "long"]:
        if coord in df.columns:
            df[coord] = df[coord].astype(str).str.replace(",", ".", regex=False)
            df[coord] = pd.to_numeric(df[coord], errors="coerce")

    # ── Colonnes finales ──────────────────────────────────────────────────────
    keep_final = ["Num_Acc", "annee", "dep", "lat", "long"] + FEATURE_COLS + [TARGET_COL]
    df = df[[c for c in keep_final if c in df.columns]].copy()

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    out = DATA_DIR / "processed_dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nDataset sauvegardé : {out}  ({len(df):,} lignes)")

    # ── Rapport par année ──────────────────────────────────────────────────────
    print("\nAccidents par année (autoroutes) :")
    for yr in YEARS:
        sub = df[df["annee"] == yr]
        pct = sub[TARGET_COL].mean() * 100 if len(sub) > 0 else 0
        print(f"  {yr} : {len(sub):5,} usagers  ({pct:.1f}% graves)")

    print("\n=== Préparation terminée ===")


if __name__ == "__main__":
    main()
