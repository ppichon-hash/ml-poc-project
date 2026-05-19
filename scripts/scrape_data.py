"""
Scraping de donnees reelles de cartes Pokemon depuis 3 sources :
  1. API Pokemon TCG  (JSON, gratuite, sans cle)
  2. PriceCharting    (HTML scraping)
  3. Cardmarket       (HTML scraping, peut etre bloque par anti-bot)

Usage : python scripts/scrape_data.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "scraping.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
}

DELAY = 1.0  # secondes entre chaque requete

# Mapping rarete -> score numerique
RARITY_MAP: dict[str, int] = {
    "Common": 1,
    "Promo": 2,
    "Uncommon": 2,
    "Rare": 3,
    "Amazing Rare": 3,
    "Rare Holo": 4,
    "Rare Holo EX": 4,
    "Rare Holo GX": 4,
    "Rare Holo V": 4,
    "Rare Prime": 4,
    "Rare Break": 4,
    "Rare ACE": 4,
    "Double Rare": 4,
    "Classic Collection": 4,
    "Shiny Rare": 4,
    "Trainer Gallery Rare Holo": 4,
    "Ultra Rare": 5,
    "Rare Ultra": 5,
    "Rare Holo VMAX": 5,
    "Rare Holo VSTAR": 5,
    "Rare Holo LV.X": 5,
    "LEGEND": 5,
    "Rare Shining": 5,
    "Rare Holo Star": 5,
    "Illustration Rare": 5,
    "Shiny Holo Rare": 5,
    "Rare Secret": 6,
    "Secret Rare": 6,
    "Rare Shiny": 6,
    "Rare Rainbow": 6,
    "Rainbow Rare": 6,
    "Hyper Rare": 6,
    "Special Illustration Rare": 6,
}


def encode_rarity(rarity: str | None) -> int:
    if not rarity:
        return 2
    return RARITY_MAP.get(rarity, 3)


# ---------------------------------------------------------------------------
# Source 1 — API Pokemon TCG
# ---------------------------------------------------------------------------

def _best_tcg_price(price_dict: dict) -> dict:
    """Extrait les prix TCGPlayer en privilegiant holofoil > normal > reverseHolofoil."""
    result: dict[str, float | None] = {"low": None, "mid": None, "high": None, "market": None}
    for variant in ("holofoil", "normal", "reverseHolofoil", "1stEditionHolofoil",
                    "1stEditionNormal", "unlimitedHolofoil"):
        if variant in price_dict:
            p = price_dict[variant]
            result = {
                "low":    p.get("low"),
                "mid":    p.get("mid"),
                "high":   p.get("high"),
                "market": p.get("market"),
            }
            if any(v is not None for v in result.values()):
                break
    return result


def scrape_tcg_api() -> pd.DataFrame | None:
    log.info("=" * 60)
    log.info("SOURCE 1 : API Pokemon TCG  (api.pokemontcg.io)")
    log.info("=" * 60)

    BASE_URL = "https://api.pokemontcg.io/v2/cards"
    all_cards: list[dict] = []
    page = 1
    page_size = 250

    while True:
        try:
            resp = requests.get(
                BASE_URL,
                params={"page": page, "pageSize": page_size},
                headers={**HEADERS, "Accept": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            log.error(f"  Erreur reseau page {page} : {exc}")
            break
        except Exception as exc:
            log.error(f"  Erreur JSON page {page} : {exc}")
            break

        cards = data.get("data", [])
        if not cards:
            log.info("  Pagination terminee (page vide)")
            break

        for card in cards:
            # Prix TCGPlayer
            tcg_prices_raw = card.get("tcgplayer", {}).get("prices", {})
            tcg = _best_tcg_price(tcg_prices_raw)
            has_holo    = "holofoil" in tcg_prices_raw or "1stEditionHolofoil" in tcg_prices_raw
            has_reverse = "reverseHolofoil" in tcg_prices_raw

            # Prix Cardmarket (inclus dans l'API TCG)
            cm = card.get("cardmarket", {}).get("prices", {})

            # Type principal
            types = card.get("types") or []
            card_type = types[0] if types else None

            # HP (string dans l'API)
            try:
                hp = int(card.get("hp") or 0) or None
            except (ValueError, TypeError):
                hp = None

            rarity = card.get("rarity")

            all_cards.append({
                "id":               card.get("id"),
                "name":             card.get("name"),
                "set_name":         card.get("set", {}).get("name"),
                "set_release_date": card.get("set", {}).get("releaseDate"),
                "rarity":           rarity,
                "hp":               hp,
                "type":             card_type,
                "evolves_from":     card.get("evolvesFrom"),
                "nb_attacks":       len(card.get("attacks") or []),
                "is_holo":          int(has_holo or bool(rarity and "Holo" in rarity)),
                "is_reverse":       int(has_reverse),
                # TCGPlayer
                "price_tcgplayer_low":    tcg["low"],
                "price_tcgplayer_mid":    tcg["mid"],
                "price_tcgplayer_high":   tcg["high"],
                "price_tcgplayer_market": tcg["market"],
                # Cardmarket (via API TCG)
                "price_cardmarket_low":      cm.get("lowPrice"),
                "price_cardmarket_trend":    cm.get("trendPrice"),
                "price_cardmarket_avg30":    cm.get("avg30"),
                "price_cardmarket_avg7":     cm.get("avg7"),
                "price_cardmarket_avg1":     cm.get("avg1"),
                "price_cardmarket_avg_sell": cm.get("averageSellPrice"),
            })

        total = data.get("totalCount", "?")
        log.info(f"  Page {page:3d} OK  —  {len(all_cards):>6,} / {total} cartes")

        if isinstance(total, int) and len(all_cards) >= total:
            break

        page += 1
        time.sleep(DELAY)

    if not all_cards:
        log.warning("  Aucune carte recuperee depuis TCG API")
        return None

    df = pd.DataFrame(all_cards)
    out = DATA_DIR / "raw_tcg_api.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info(f"  Sauvegarde : {out}  ({len(df):,} cartes)")
    return df


# ---------------------------------------------------------------------------
# Source 2 — PriceCharting
# ---------------------------------------------------------------------------

def _parse_price(text: str) -> float | None:
    """Nettoie un texte et retourne un float prix, ou None."""
    cleaned = text.strip().replace("$", "").replace(",", "").replace("N/A", "")
    try:
        val = float(cleaned)
        return val if val > 0 else None
    except ValueError:
        return None


def scrape_pricecharting() -> pd.DataFrame | None:
    log.info("=" * 60)
    log.info("SOURCE 2 : PriceCharting  (pricecharting.com)")
    log.info("=" * 60)

    BASE = "https://www.pricecharting.com"
    all_rows: list[dict] = []

    # Recuperer la liste des sets Pokemon
    try:
        resp = requests.get(
            f"{BASE}/category/pokemon-cards",
            headers={**HEADERS, "Accept": "text/html,*/*"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.error(f"  Impossible d'acceder a PriceCharting : {exc}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Liens vers les pages de sets
    set_links: list[str] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "/console/pokemon" in href or "/game/pokemon" in href:
            full = href if href.startswith("http") else BASE + href
            if full not in set_links:
                set_links.append(full)

    log.info(f"  {len(set_links)} sets detectes")

    if not set_links:
        log.warning("  Aucun set trouve — structure HTML peut-etre differente")
        return None

    # Limiter a 60 sets pour ne pas surcharger le serveur
    for i, set_url in enumerate(set_links[:60]):
        try:
            time.sleep(DELAY)
            r = requests.get(
                set_url,
                headers={**HEADERS, "Accept": "text/html,*/*"},
                timeout=30,
            )
            r.raise_for_status()
            s = BeautifulSoup(r.text, "lxml")

            # Chercher le tableau des cartes
            table = (
                s.find("table", id="games_table")
                or s.find("table", class_="sort-table")
                or s.find("table")
            )
            if not table:
                continue

            set_name = set_url.rstrip("/").split("/")[-1].replace("-", " ").title()
            rows_before = len(all_rows)

            for tr in table.find_all("tr")[1:]:
                cells = tr.find_all("td")
                if len(cells) < 2:
                    continue

                card_name = cells[0].get_text(strip=True)
                if not card_name or card_name.lower() in ("name", "title"):
                    continue

                # Chercher le premier prix valide dans les cellules suivantes
                price_val: float | None = None
                for cell in cells[1:5]:
                    price_val = _parse_price(cell.get_text(strip=True))
                    if price_val is not None:
                        break

                all_rows.append({
                    "name_pc":            card_name,
                    "set_name_pc":        set_name,
                    "price_pricecharting": price_val,
                })

            added = len(all_rows) - rows_before
            log.info(f"  [{i+1:2d}/{min(len(set_links), 60)}]  {set_name:<35s}  +{added:3d} cartes")

        except requests.exceptions.RequestException as exc:
            log.error(f"  Erreur reseau {set_url}: {exc}")
        except Exception as exc:
            log.error(f"  Erreur parsing {set_url}: {exc}")

    if not all_rows:
        log.warning("  Aucune donnee recuperee depuis PriceCharting")
        return None

    df = pd.DataFrame(all_rows)
    out = DATA_DIR / "raw_pricecharting.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info(f"  Sauvegarde : {out}  ({len(df):,} lignes)")
    return df


# ---------------------------------------------------------------------------
# Source 3 — Cardmarket
# ---------------------------------------------------------------------------

def scrape_cardmarket() -> pd.DataFrame | None:
    log.info("=" * 60)
    log.info("SOURCE 3 : Cardmarket  (cardmarket.com)")
    log.info("=" * 60)

    # Note : Cardmarket utilise Cloudflare. Le scraping HTML est souvent bloque.
    # Les prix Cardmarket sont deja inclus dans l'API TCG (Source 1).
    # On tente quand meme pour les donnees supplementaires.

    try:
        resp = requests.get(
            "https://www.cardmarket.com/en/Pokemon/Products/Singles",
            headers={**HEADERS, "Accept": "text/html,*/*"},
            timeout=30,
        )

        # Detection des protections anti-bot
        is_blocked = (
            resp.status_code in (403, 429, 503)
            or "cloudflare" in resp.text.lower()
            or "just a moment" in resp.text.lower()
            or "challenge" in resp.text.lower()
            or len(resp.text) < 2000
        )

        if is_blocked:
            log.warning(
                f"  Cardmarket protege par anti-bot (HTTP {resp.status_code}) — "
                f"les prix CM proviennent de l'API TCG (source 1)"
            )
            return None

        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        rows: list[dict] = []

        # Tentative de parsing (si on passe la protection)
        for item in soup.select(".productRow, .col-sellerProductInfo, [class*='product-row']")[:300]:
            name = item.get_text(separator=" ", strip=True)[:120]
            if len(name) > 3:
                rows.append({"name_cm": name, "price_cardmarket_scraped": None})

        if rows:
            df = pd.DataFrame(rows)
            out = DATA_DIR / "raw_cardmarket.csv"
            df.to_csv(out, index=False, encoding="utf-8")
            log.info(f"  Sauvegarde : {out}  ({len(df):,} lignes)")
            return df

    except requests.exceptions.RequestException as exc:
        log.error(f"  Erreur reseau Cardmarket : {exc}")
    except Exception as exc:
        log.error(f"  Erreur inattendue Cardmarket : {exc}")

    log.warning("  Cardmarket indisponible — source ignoree")
    return None


# ---------------------------------------------------------------------------
# Fusion & feature engineering
# ---------------------------------------------------------------------------

def build_final_dataset(
    df_tcg: pd.DataFrame,
    df_pc: pd.DataFrame | None,
    df_cm: pd.DataFrame | None,
) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("FUSION ET FEATURE ENGINEERING")
    log.info("=" * 60)

    df = df_tcg.copy()

    # --- Jointure PriceCharting ---
    if df_pc is not None and not df_pc.empty:
        # Normaliser les noms pour la jointure (lower + strip)
        df["_name_key"] = df["name"].str.lower().str.strip()
        df_pc["_name_key"] = df_pc["name_pc"].str.lower().str.strip()
        # Aggreger par nom (plusieurs sets sur PC)
        agg_pc = (
            df_pc.dropna(subset=["price_pricecharting"])
            .groupby("_name_key")["price_pricecharting"]
            .mean()
            .reset_index()
        )
        df = df.merge(agg_pc, on="_name_key", how="left")
        df.drop(columns=["_name_key"], inplace=True)
        matched = df["price_pricecharting"].notna().sum()
        log.info(f"  PriceCharting joint : {matched:,} cartes avec prix ({matched/len(df)*100:.1f}%)")
    else:
        df["price_pricecharting"] = np.nan

    # --- market_price : moyenne pondéree des sources disponibles ---
    price_sources = [
        "price_tcgplayer_market",
        "price_tcgplayer_mid",
        "price_cardmarket_avg_sell",
        "price_cardmarket_avg7",
        "price_pricecharting",
    ]
    available = [c for c in price_sources if c in df.columns]
    df["market_price"] = df[available].mean(axis=1, skipna=True)

    # --- high_price ---
    high_cols = [c for c in ["price_tcgplayer_high", "price_tcgplayer_market",
                              "price_cardmarket_avg_sell"] if c in df.columns]
    df["high_price"] = df[high_cols].max(axis=1, skipna=True)

    # --- low_price ---
    low_cols = [c for c in ["price_tcgplayer_low", "price_cardmarket_low",
                             "price_pricecharting"] if c in df.columns]
    df["low_price"] = df[low_cols].min(axis=1, skipna=True)

    # low ne doit pas depasser market
    df["low_price"] = df[["low_price", "market_price"]].min(axis=1)

    # --- trend_price : priorite cm_trend > cm_avg7 > cm_avg30 > market ---
    df["trend_price"] = df.get("price_cardmarket_trend", pd.Series(np.nan, index=df.index))
    for fallback in ("price_cardmarket_avg7", "price_cardmarket_avg30"):
        if fallback in df.columns:
            df["trend_price"] = df["trend_price"].fillna(df[fallback])
    df["trend_price"] = df["trend_price"].fillna(df["market_price"])

    # --- Annee de sortie du set ---
    def _parse_year(val: object) -> int | None:
        if pd.isna(val):
            return None
        try:
            return int(str(val)[:4])
        except (ValueError, TypeError):
            return None

    df["year"] = df["set_release_date"].apply(_parse_year)

    # --- Features engineerees ---
    df["rarity_encoded"]    = df["rarity"].apply(encode_rarity)
    df["set_age"]           = df["year"].apply(lambda y: (2024 - y) if pd.notna(y) else np.nan)
    df["price_range"]       = (df["high_price"] - df["low_price"]).clip(lower=0)
    df["hp_normalized"]     = df["hp"] / df["hp"].max()
    df["has_evolution"]     = df["evolves_from"].notna().astype(int)
    df["price_source_count"] = df[available].notna().sum(axis=1)
    df["price_volatility"]  = np.where(
        df["market_price"] > 0,
        df["price_range"] / df["market_price"],
        np.nan,
    )

    # --- Target ---
    df["target"] = (
        (df["trend_price"] > df["market_price"] * 1.1)
        & df["trend_price"].notna()
        & df["market_price"].notna()
    ).astype(int)

    # --- Filtrage : garder uniquement les cartes avec au moins un prix ---
    df_final = df[df["market_price"].notna()].copy()
    df_final = df_final.reset_index(drop=True)

    log.info(f"  Cartes avec prix : {len(df_final):,} / {len(df):,}")
    log.info(f"  Target : {df_final['target'].value_counts().to_dict()}")

    return df_final


# ---------------------------------------------------------------------------
# Rapport final
# ---------------------------------------------------------------------------

def print_report(df_tcg: pd.DataFrame, df_pc: pd.DataFrame | None,
                 df_cm: pd.DataFrame | None, df_final: pd.DataFrame) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("RAPPORT DE SCRAPING")
    print(sep)
    print(f"  Source 1 — API TCG      : {len(df_tcg):>6,} cartes brutes")
    print(f"  Source 2 — PriceCharting: {len(df_pc):>6,} lignes" if df_pc is not None
          else "  Source 2 — PriceCharting: indisponible")
    print(f"  Source 3 — Cardmarket   : {len(df_cm):>6,} lignes" if df_cm is not None
          else "  Source 3 — Cardmarket   : indisponible (prix CM via API TCG)")
    print(f"\n  Dataset final           : {len(df_final):>6,} cartes avec prix")
    print(f"  Target = 1 (valorisation): {df_final['target'].sum():>6,} ({df_final['target'].mean()*100:.1f}%)")
    print(f"  Target = 0 (stable)      : {(df_final['target']==0).sum():>6,}")

    print(f"\n  Colonnes de prix disponibles :")
    price_cols = [c for c in df_final.columns if "price" in c.lower()]
    for col in price_cols:
        pct = df_final[col].notna().mean() * 100
        bar = "#" * int(pct / 5)
        print(f"    {col:<40s} {pct:5.1f}%  {bar}")

    print(f"\n  Taux de donnees manquantes (colonnes > 5%) :")
    miss = df_final.isnull().mean() * 100
    for col, pct in miss[miss > 5].sort_values(ascending=False).items():
        print(f"    {col:<40s} {pct:5.1f}% manquant")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Debut du scraping — Pokemon Card Investor")
    log.info(f"Delai entre requetes : {DELAY}s")

    # Source 1 — indispensable
    df_tcg = scrape_tcg_api()
    if df_tcg is None:
        log.error("Source 1 (API TCG) indisponible — impossible de continuer")
        sys.exit(1)

    # Source 2 — optionnelle
    df_pc = scrape_pricecharting()

    # Source 3 — optionnelle
    df_cm = scrape_cardmarket()

    # Fusion
    df_final = build_final_dataset(df_tcg, df_pc, df_cm)

    # Sauvegarde dataset final
    out = DATA_DIR / "pokemon_cards.csv"
    df_final.to_csv(out, index=False, encoding="utf-8")
    log.info(f"Dataset final sauvegarde : {out}  ({len(df_final):,} cartes)")

    print_report(df_tcg, df_pc, df_cm, df_final)


if __name__ == "__main__":
    main()
