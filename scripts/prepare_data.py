"""
Prépare le dataset BAAC (accidents corporels sur autoroutes) pour le ML.

Pipeline :
  1. Charge les 4 fichiers × 5 années (2020-2024)
  2. Concatène les 5 années
  3. Fusionne les 4 tables sur Num_Acc
  4. Filtre catr == 1 (autoroutes uniquement — contexte Vinci)
  5. Supprime grav == -1
  6. Crée la target binaire (0 = léger, 1 = grave)
  7. Enrichit avec features usager (sexe, age, ceinture) + heure
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
    # Conditions environnementales
    "lum", "atm", "col",
    # Infrastructure routière
    "circ", "nbv", "prof", "surf", "infra", "situ", "vma",
    # Véhicule
    "catv",
    # Temporel
    "mois", "jour", "heure", "saison",
    # Usager (nouvelles features)
    "catu", "sexe", "age", "secu",
]

TARGET_COL = "target"


def _parse_heure(hrmn_series: pd.Series) -> pd.Series:
    """Convertit 'HH:MM' ou entier HHMM en heure décimale (0-23.99)."""
    def _parse_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if ":" in s:
            parts = s.split(":")
            try:
                return int(parts[0]) + int(parts[1]) / 60.0
            except (ValueError, IndexError):
                return np.nan
        # Format entier HHMM
        try:
            v = int(float(s))
            return (v // 100) + (v % 100) / 60.0
        except ValueError:
            return np.nan
    return hrmn_series.apply(_parse_one)


def _mois_to_saison(mois: pd.Series) -> pd.Series:
    """Convertit le mois (1-12) en saison (1=hiver, 2=printemps, 3=été, 4=automne)."""
    mapping = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
               6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    return mois.map(mapping)


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

    # ── Préparation usagers ─────────────────────────────────────────────────────
    # Garder : grav, catu, sexe, an_nais (→ age), secu1
    usag_keep = ["Num_Acc", "grav", "catu", "sexe", "an_nais", "secu1"]
    usag_sel = usag[[c for c in usag_keep if c in usag.columns]].copy()

    # Nettoyer -1 → NaN dans les colonnes usager
    for col in ["catu", "sexe", "an_nais", "secu1"]:
        if col in usag_sel.columns:
            usag_sel[col] = pd.to_numeric(usag_sel[col], errors="coerce")
            usag_sel.loc[usag_sel[col] == -1, col] = np.nan

    # ── Préparation caracteristiques ────────────────────────────────────────────
    caract_keep = ["Num_Acc", "lum", "atm", "col", "mois", "jour", "hrmn", "annee", "dep", "lat", "long"]
    caract_sel = caract[[c for c in caract_keep if c in caract.columns]].copy()
    # Extraire l'heure décimale
    if "hrmn" in caract_sel.columns:
        caract_sel["heure"] = _parse_heure(caract_sel["hrmn"])
        caract_sel = caract_sel.drop(columns=["hrmn"])

    # ── Préparation lieux ───────────────────────────────────────────────────────
    lieux_keep = ["Num_Acc", "catr", "circ", "nbv", "prof", "surf", "infra", "situ", "vma"]
    lieux_sel = lieux[[c for c in lieux_keep if c in lieux.columns]].drop_duplicates("Num_Acc")

    # ── Préparation véhicules ───────────────────────────────────────────────────
    veh_keep = ["Num_Acc", "catv"]
    veh_sel = veh[[c for c in veh_keep if c in veh.columns]].copy()
    veh_sel = (
        veh_sel.groupby("Num_Acc")["catv"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .reset_index()
    )

    # ── Merge progressif ────────────────────────────────────────────────────────
    df = usag_sel.merge(caract_sel, on="Num_Acc", how="inner")
    df = df.merge(lieux_sel,  on="Num_Acc", how="left")
    df = df.merge(veh_sel,    on="Num_Acc", how="left")

    print(f"\nAprès fusion : {len(df):,} lignes (une par usager)")

    # ── Filtre autoroutes (catr == 1) ───────────────────────────────────────────
    df = df[df["catr"] == 1].copy()
    print(f"Après filtre autoroutes (catr=1) : {len(df):,} lignes")

    # ── Filtre grav != -1 ────────────────────────────────────────────────────────
    df["grav"] = pd.to_numeric(df["grav"], errors="coerce")
    df = df[df["grav"].notna() & (df["grav"] != -1)].copy()
    print(f"Après filtre grav != -1 : {len(df):,} lignes")

    # ── Target binaire ───────────────────────────────────────────────────────────
    # 0 = indemne (1) ou blessé léger (2)
    # 1 = hospitalisé (3) ou tué (4)
    df[TARGET_COL] = df["grav"].apply(lambda g: 0 if g in (1, 2) else 1)
    print(f"\nDistribution target :")
    print(f"  0 (léger)  : {(df[TARGET_COL]==0).sum():,}  ({(df[TARGET_COL]==0).mean()*100:.1f}%)")
    print(f"  1 (grave)  : {(df[TARGET_COL]==1).sum():,}  ({(df[TARGET_COL]==1).mean()*100:.1f}%)")

    # ── Feature engineering ──────────────────────────────────────────────────────
    # Saison depuis mois
    df["mois"] = pd.to_numeric(df["mois"], errors="coerce")
    df["saison"] = _mois_to_saison(df["mois"])

    # Âge conducteur depuis an_nais
    if "an_nais" in df.columns:
        df["age"] = df["annee"] - df["an_nais"]
        # Filtrer ages aberrants (< 14 ou > 110)
        df.loc[(df["age"] < 14) | (df["age"] > 110), "age"] = np.nan
        df = df.drop(columns=["an_nais"])

    # secu1 → secu : simplifier (1=ceinture portée, 0=non portée/autres, NaN=inconnu)
    if "secu1" in df.columns:
        df["secu"] = df["secu1"].apply(
            lambda x: 1 if x == 1 else (0 if pd.notna(x) and x in [0, 2, 3, 4, 5, 6, 7, 8, 9] else np.nan)
        )
        df = df.drop(columns=["secu1"])

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
    print(f"\nDataset sauvegardé : {out}  ({len(df):,} lignes, {len(df.columns)} colonnes)")
    print(f"Features : {[c for c in FEATURE_COLS if c in df.columns]}")

    # ── Rapport par année ──────────────────────────────────────────────────────
    print("\nAccidents par année (autoroutes) :")
    for yr in YEARS:
        sub = df[df["annee"] == yr]
        pct = sub[TARGET_COL].mean() * 100 if len(sub) > 0 else 0
        print(f"  {yr} : {len(sub):5,} usagers  ({pct:.1f}% graves)")

    # ── Corrélations avec target ───────────────────────────────────────────────
    print("\nCorrélation features / target (Pearson) :")
    for col in [c for c in FEATURE_COLS if c in df.columns]:
        r = df[[col, TARGET_COL]].dropna().corr().iloc[0, 1]
        print(f"  {col:10s} : {r:+.4f}")

    print("\n=== Préparation terminée ===")


if __name__ == "__main__":
    main()
