"""Pokemon Card Investor — Streamlit application."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, MODEL_METRICS_FILE, MODELS_DIR

# ── Palette ───────────────────────────────────────────────────────────────────

RED    = "#E3350D"
YELLOW = "#FFCB05"
DARK   = "#1A1A2E"
DARK2  = "#16213E"
DARK3  = "#0F3460"
WHITE  = "#F8F9FA"
GREEN  = "#00B894"
ORANGE = "#FDCB6E"

# ── ML constants ──────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rarity_encoded", "set_age", "price_range", "is_holo",
    "hp_normalized", "has_evolution", "is_reverse", "nb_attacks",
    "market_price", "low_price", "high_price",
]

FEATURE_LABELS = {
    "rarity_encoded": "Rareté",
    "set_age":        "Âge du set (ans)",
    "price_range":    "Écart de prix ($)",
    "is_holo":        "Holographique",
    "hp_normalized":  "HP normalisé",
    "has_evolution":  "A une évolution",
    "is_reverse":     "Reverse Holo",
    "nb_attacks":     "Nb d'attaques",
    "market_price":   "Prix marché ($)",
    "low_price":      "Prix bas ($)",
    "high_price":     "Prix haut ($)",
}

FEATURE_EXPLAIN = {
    "market_price":  ("Plus le prix actuel est élevé, plus la carte a une cote reconnue.",       lambda v: f"${v:.2f}"),
    "price_range":   ("Un grand écart prix bas/haut signale une forte volatilité.",               lambda v: f"${v:.2f}"),
    "low_price":     ("Prix plancher constaté sur le marché.",                                    lambda v: f"${v:.2f}"),
    "high_price":    ("Prix plafond constaté sur le marché.",                                     lambda v: f"${v:.2f}"),
    "set_age":       ("Les sets anciens ont eu le temps de se valoriser.",                        lambda v: f"{int(v)} ans"),
    "rarity_encoded":("Plus la rareté est élevée, plus la carte est convoitée des collectionneurs.", lambda v: f"{int(v)}/6"),
    "hp_normalized": ("Les cartes puissantes attirent les joueurs compétitifs.",                  lambda v: f"{v*100:.0f}%"),
    "is_holo":       ("Les cartes holographiques se distinguent visuellement et sont plus rares.", lambda v: "Oui" if v else "Non"),
    "is_reverse":    ("Le reverse holo ajoute une variante rare à la carte.",                     lambda v: "Oui" if v else "Non"),
    "has_evolution": ("Les cartes évoluées sont souvent jouées dans les decks compétitifs.",      lambda v: "Oui" if v else "Non"),
    "nb_attacks":    ("Plus d'attaques signifie plus de polyvalence tactique.",                   lambda v: str(int(v))),
}

RARITY_SIMPLIFIED_ORDER = ["Common", "Uncommon", "Rare", "Holo Rare", "Ultra Rare", "Secret Rare"]
RARITY_ENCODED_MAP      = {r: i + 1 for i, r in enumerate(RARITY_SIMPLIFIED_ORDER)}
RARITY_COLORS_MAP = {
    "Common":      "#74b9ff",
    "Uncommon":    "#a29bfe",
    "Rare":        "#6c5ce7",
    "Holo Rare":   YELLOW,
    "Ultra Rare":  ORANGE,
    "Secret Rare": RED,
    "Other":       "#b2bec3",
}

CLUSTER_NAMES  = {0: "Cartes communes", 1: "Cartes rares", 2: "Cartes collector"}
CLUSTER_COLORS = {0: "#74b9ff", 1: YELLOW, 2: RED}
CLUSTER_EMOJIS = {0: "🃏", 1: "⭐", 2: "💎"}

POKEMON_TYPES = [
    "Fire", "Water", "Grass", "Lightning", "Psychic",
    "Fighting", "Darkness", "Metal", "Dragon", "Colorless", "Fairy",
]

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {{
    background-color: {DARK};
    color: {WHITE};
}}
[data-testid="stHeader"] {{ background-color: {DARK}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {DARK2} 0%, {DARK3} 100%);
    border-right: 2px solid {RED};
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background: {DARK2};
    border: 1px solid {DARK3};
    border-left: 4px solid {RED};
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
}}
[data-testid="stMetricValue"] {{
    color: {YELLOW} !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {WHITE} !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* ── Buttons ── */
.stButton > button {{
    background: {RED} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.6rem !important;
    transition: background 0.25s ease, color 0.25s ease, box-shadow 0.25s ease !important;
    letter-spacing: 0.3px !important;
}}
.stButton > button:hover {{
    background: {YELLOW} !important;
    color: {DARK} !important;
    box-shadow: 0 4px 18px rgba(255,203,5,0.45) !important;
}}

/* ── Titres avec dégradé ── */
h1 {{
    background: linear-gradient(90deg, {RED}, {YELLOW});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900 !important;
}}
h2 {{ color: {WHITE} !important; font-weight: 700 !important; }}
h3 {{ color: {YELLOW} !important; font-weight: 600 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {DARK2};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {WHITE};
    border-radius: 8px;
    font-weight: 600;
    padding: 6px 22px;
}}
.stTabs [aria-selected="true"] {{
    background: {RED} !important;
    color: {WHITE} !important;
}}

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    background: {DARK2} !important;
    color: {WHITE} !important;
    border-color: {DARK3} !important;
    border-radius: 8px !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: {RED} !important;
}}
.stCheckbox span[data-testid="stWidgetLabel"] {{ color: {WHITE} !important; }}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {DARK2};
    border: 1px solid {DARK3};
    border-radius: 12px;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* ── Alerts ── */
[data-testid="stAlert"] {{ border-radius: 12px !important; }}

/* ── Progress ── */
[data-testid="stProgress"] > div > div {{ background: {RED} !important; }}

/* ── Divider ── */
hr {{ border-color: {DARK3} !important; }}
</style>
"""

# ── Cache ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des données Pokémon…")
def load_data() -> pd.DataFrame:
    try:
        return pd.read_csv(DATA_DIR / "pokemon_cards.csv")
    except FileNotFoundError:
        st.error(f"Fichier introuvable : {DATA_DIR / 'pokemon_cards.csv'}. Lancez scripts/generate_data.py.")
        st.stop()


@st.cache_resource(show_spinner="Chargement des modèles ML…")
def load_models() -> dict:
    models: dict = {}
    for key, fname in [("rf", "random_forest.joblib"), ("xgb", "xgboost.joblib"), ("kmeans", "kmeans.joblib")]:
        p = MODELS_DIR / fname
        try:
            if p.exists():
                models[key] = joblib.load(p)
        except Exception as exc:
            st.warning(f"Impossible de charger {fname} : {exc}")
    return models


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    try:
        if MODEL_METRICS_FILE.exists():
            return pd.read_csv(MODEL_METRICS_FILE)
    except Exception:
        pass
    return None

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_layout(title: str = "", **kwargs) -> dict:
    base: dict = dict(
        paper_bgcolor=DARK2,
        plot_bgcolor=DARK2,
        font=dict(color=WHITE, family="sans-serif", size=12),
        xaxis=dict(gridcolor="#2d3561", zerolinecolor="#2d3561", linecolor="#2d3561"),
        yaxis=dict(gridcolor="#2d3561", zerolinecolor="#2d3561", linecolor="#2d3561"),
        margin=dict(t=55, b=40, l=45, r=20),
        legend=dict(bgcolor=DARK2, bordercolor=DARK3, borderwidth=1),
    )
    if title:
        base["title"] = dict(text=title, font=dict(color=YELLOW, size=15))
    base.update(kwargs)
    return base


def _banner(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {RED} 0%, #8B0000 55%, {DARK2} 100%);
        border-radius: 16px; padding: 30px 36px; margin-bottom: 28px;
        border-left: 6px solid {YELLOW};
        box-shadow: 0 6px 24px rgba(227,53,13,0.35);
    ">
        <h1 style="
            background: linear-gradient(90deg, {YELLOW}, #fff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin: 0; font-size: 2rem; font-weight: 900;
        ">{title}</h1>
        <p style="color:{WHITE}; margin:10px 0 0; font-size:1.05rem; opacity:0.88;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def _card_html(emoji: str, title: str, body: str, color: str = DARK3) -> str:
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}cc, {DARK2});
        border: 1px solid {color}; border-radius: 14px; padding: 22px 20px;
        height: 100%; box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    ">
        <div style="font-size:2.2rem; margin-bottom:10px;">{emoji}</div>
        <div style="color:{YELLOW}; font-weight:800; font-size:1rem; margin-bottom:10px;">{title}</div>
        <div style="color:{WHITE}; font-size:0.88rem; line-height:1.65;">{body}</div>
    </div>
    """


def _simplify_rarity(r) -> str:
    if pd.isna(r):
        return "Other"
    s = str(r).lower()
    if any(x in s for x in ("secret", "hyper", "rainbow", "shiny ultra", "mega hyper")):
        return "Secret Rare"
    if any(x in s for x in ("ultra", "illustration rare", "special illustration", "radiant")):
        return "Ultra Rare"
    if any(x in s for x in ("holo", "vmax", "vstar", "amazing", "prism", "break",
                             "legend", "lv.x", "star", "prime", "ace spec", "double rare")):
        return "Holo Rare"
    if "rare" in s:
        return "Rare"
    if "uncommon" in s:
        return "Uncommon"
    if "common" in s:
        return "Common"
    return "Other"


def _simulate_price_history(
    market_price: float,
    low_price: float,
    high_price: float,
    rarity_encoded: int,
    set_age: int,
    release_date: str | None,
) -> pd.DataFrame:
    np.random.seed(int(market_price * 100) % 9999)

    if rarity_encoded >= 5:
        base_growth = 0.025 if set_age > 10 else 0.012
    elif rarity_encoded >= 3:
        base_growth = 0.012 if set_age > 5 else 0.005
    else:
        base_growth = 0.001

    noise_scale = max(0.015, (high_price - low_price) / max(market_price, 0.01) * 0.2)

    today = pd.Timestamp.today().normalize()
    if release_date:
        try:
            start = pd.Timestamp(release_date)
        except Exception:
            start = today - pd.DateOffset(years=max(set_age, 1))
    else:
        start = today - pd.DateOffset(years=max(set_age, 1))

    if start >= today:
        start = today - pd.DateOffset(months=12)

    dates = pd.date_range(start=start, end=today, freq="MS")
    if len(dates) < 2:
        dates = pd.date_range(start=start, periods=2, freq="MS")

    n = len(dates)
    start_price = max(market_price / ((1 + base_growth) ** n), low_price * 0.3, 0.01)

    prices = [start_price]
    for _ in range(n - 1):
        noise = np.random.normal(0, noise_scale)
        prices.append(max(low_price * 0.2, prices[-1] * (1 + base_growth + noise)))

    if prices[-1] > 0:
        scale  = market_price / prices[-1]
        prices = [round(p * scale, 2) for p in prices]

    return pd.DataFrame({"date": dates[: len(prices)], "prix": prices})


# ── Page 1 — Accueil ──────────────────────────────────────────────────────────

def page_accueil(df: pd.DataFrame) -> None:
    st.markdown(f"""
    <div style="
        text-align:center; padding:48px 32px 38px;
        background: linear-gradient(135deg, {DARK2} 0%, {DARK3} 50%, {DARK2} 100%);
        border-radius: 20px; margin-bottom: 32px;
        border: 1px solid {DARK3}; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    ">
        <div style="font-size:4rem; margin-bottom:14px;">🎴</div>
        <h1 style="
            font-size:2.8rem; font-weight:900; margin:0;
            background: linear-gradient(90deg, {RED}, {YELLOW});
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            background-clip:text;
        ">Pokemon Card Investor</h1>
        <p style="color:{WHITE}; font-size:1.15rem; margin-top:14px; opacity:0.85;">
            Analysez, prédisez et optimisez vos investissements dans les cartes Pokémon
        </p>
        <div style="margin-top:22px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
            <span style="background:{RED}22; border:1px solid {RED}; border-radius:20px;
                         padding:6px 16px; color:{WHITE}; font-size:0.85rem;">🤖 Machine Learning</span>
            <span style="background:{YELLOW}22; border:1px solid {YELLOW}; border-radius:20px;
                         padding:6px 16px; color:{WHITE}; font-size:0.85rem;">📊 {len(df):,} cartes analysées</span>
            <span style="background:{DARK3}; border:1px solid {DARK3}; border-radius:20px;
                         padding:6px 16px; color:{WHITE}; font-size:0.85rem;">🏆 3 modèles ML</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🃏 Cartes analysées",  f"{len(df):,}")
    c2.metric("📦 Sets différents",   df["set_name"].nunique())
    c3.metric("💰 Prix moyen",         f"${df['market_price'].mean():.2f}")
    c4.metric("📈 Cartes valorisées", f"{df['target'].mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "**Comment utiliser cette app :**\n\n"
        "- **📊 Dashboard** — explorez l'évolution des cotes, les tendances par rareté et les top cartes\n"
        "- **🔮 Prédiction** — entrez les caractéristiques d'une carte et obtenez une prédiction ML instantanée\n"
        "- **⚖️ Comparaison** — comparez les 3 modèles (Random Forest, XGBoost, KMeans) et leur performance"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2>Les 3 modules de l'application</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(_card_html(
            "📊", "Dashboard des cotes",
            "Recherchez une carte, visualisez l'évolution simulée de son prix depuis sa sortie, "
            "explorez les tendances par rareté et type, et consultez les top valorisations.",
            DARK3,
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(_card_html(
            "🔮", "Prédiction ML",
            "Configurez les caractéristiques d'une carte et obtenez instantanément une prédiction "
            "de valorisation par XGBoost avec niveau de confiance et explications des facteurs clés.",
            "#1a3a2e",
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(_card_html(
            "⚖️", "Comparaison modèles",
            "Comparez Random Forest, XGBoost et KMeans sur accuracy, F1-score, précision et recall. "
            "Visualisez la feature importance et les 3 profils de cartes identifiés par KMeans.",
            "#2a1a3e",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2>Aperçu du dataset</h2>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        rarity_counts = df["rarity"].apply(_simplify_rarity).value_counts()
        rarity_counts = rarity_counts.reindex(RARITY_SIMPLIFIED_ORDER + ["Other"]).dropna()
        fig = px.bar(
            x=rarity_counts.index, y=rarity_counts.values,
            color=rarity_counts.index,
            color_discrete_map=RARITY_COLORS_MAP,
            labels={"x": "Rareté", "y": "Nb de cartes"},
            title="Distribution par rareté",
        )
        fig.update_layout(**_fig_layout(), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        t_counts = df["target"].value_counts().sort_index()
        fig_d = px.pie(
            values=t_counts.values,
            names=["❌ Ne se valorisera pas", "✅ Se valorisera"],
            color_discrete_sequence=[DARK3, GREEN],
            title="Objectif : cartes qui vont se valoriser",
            hole=0.52,
        )
        fig_d.update_layout(**_fig_layout(), showlegend=True)
        fig_d.update_traces(textfont_size=13)
        st.plotly_chart(fig_d, use_container_width=True)


# ── Page 2 — Dashboard ────────────────────────────────────────────────────────

def page_dashboard(df: pd.DataFrame) -> None:
    _banner("📊 Dashboard des cotes", "Explorez les tendances du marché et l'évolution des prix")

    # ── Section 1 : Recherche carte ──────────────────────────────────────────
    st.markdown(f"<h3>🔍 Recherche d'une carte</h3>", unsafe_allow_html=True)

    all_names   = sorted(df["name"].dropna().unique())
    default_idx = all_names.index("Charizard") if "Charizard" in all_names else 0
    chosen_name = st.selectbox("Choisissez un Pokémon", options=all_names, index=default_idx)

    matches = df[df["name"] == chosen_name].copy()
    if matches.empty:
        st.warning("Aucune carte trouvée.")
        return

    if len(matches) > 1:
        chosen_set = st.selectbox(f"{len(matches)} éditions disponibles — choisir un set :", matches["set_name"].tolist())
        card = matches[matches["set_name"] == chosen_set].iloc[0]
    else:
        card = matches.iloc[0]

    rarity_color = RARITY_COLORS_MAP.get(_simplify_rarity(card["rarity"]), DARK3)
    rarity_label = card["rarity"] if not pd.isna(card["rarity"]) else "N/A"
    holo_icon    = "✨" if card["is_holo"] else "🃏"

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {rarity_color}22, {DARK2});
        border: 2px solid {rarity_color}; border-radius: 14px;
        padding: 22px 28px; margin: 14px 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    ">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h2 style="color:{YELLOW}; margin:0 0 5px;">{card['name']}</h2>
                <p style="color:{WHITE}; opacity:0.8; margin:0;">
                    {card['set_name']} &nbsp;•&nbsp; {int(card['year'])}
                    &nbsp;•&nbsp; <span style="color:{rarity_color}; font-weight:700;">{rarity_label}</span>
                </p>
            </div>
            <div style="font-size:2.8rem;">{holo_icon}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    cc1.metric("⭐ Rareté",  rarity_label)
    cc2.metric("❤️ HP",      int(card["hp"]) if not pd.isna(card["hp"]) else "N/A")
    cc3.metric("🔥 Type",    card["type"] if not pd.isna(card["type"]) else "N/A")
    cc4.metric("✨ Holo",    "Oui" if card["is_holo"] else "Non")
    cc5.metric("🔄 Reverse", "Oui" if card["is_reverse"] else "Non")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("💰 Prix actuel", f"${card['market_price']:.2f}")
    mc2.metric("📉 Prix bas",    f"${card['low_price']:.2f}")
    mc3.metric("📈 Prix haut",   f"${card['high_price']:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2 : Évolution simulée ────────────────────────────────────────
    st.markdown(f"<h3>📈 Évolution simulée de la cote</h3>", unsafe_allow_html=True)

    try:
        raw_rd   = card["set_release_date"] if "set_release_date" in card.index else None
        rel_date = str(raw_rd) if raw_rd is not None and not pd.isna(raw_rd) else None
        history  = _simulate_price_history(
            float(card["market_price"]), float(card["low_price"]), float(card["high_price"]),
            int(card["rarity_encoded"]), int(card["set_age"]), rel_date,
        )
        p0, p1  = history["prix"].iloc[0], history["prix"].iloc[-1]
        var_pct = (p1 - p0) / max(p0, 0.01) * 100

        is_up     = var_pct > 2
        is_down   = var_pct < -2
        trend_lbl = "📈 En hausse" if is_up else "📉 En baisse" if is_down else "➡️ Stable"
        line_clr  = GREEN if is_up else RED if is_down else ORANGE
        fill_clr  = ("rgba(0,184,148,0.1)" if is_up else
                     "rgba(227,53,13,0.1)" if is_down else "rgba(253,203,110,0.1)")

        ti1, ti2, ti3, ti4 = st.columns(4)
        ti1.metric("Tendance",             trend_lbl)
        ti2.metric("Prix initial estimé",  f"${p0:.2f}", f"{var_pct:+.1f}%")
        ti3.metric("Min période",          f"${history['prix'].min():.2f}")
        ti4.metric("Max période",          f"${history['prix'].max():.2f}")

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=history["date"], y=history["prix"],
            mode="lines+markers",
            line=dict(color=line_clr, width=3),
            marker=dict(size=5, color=line_clr, line=dict(color=DARK2, width=1)),
            fill="tozeroy", fillcolor=fill_clr,
            hovertemplate="<b>%{x|%b %Y}</b><br>Prix : $%{y:.2f}<extra></extra>",
        ))
        fig_line.update_layout(
            **_fig_layout(title=f"Évolution simulée — {card['name']} ({card['set_name']})"),
            xaxis_title="Date", yaxis_title="Prix ($)", hovermode="x unified",
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.caption("Note : évolution simulée à titre indicatif à partir des données de prix et de rareté.")
    except Exception as exc:
        st.error(f"Erreur lors du calcul de l'évolution : {exc}")

    st.divider()

    # ── Section 3 : Comparaison par rareté ───────────────────────────────────
    st.markdown(f"<h3>🌈 Comparaison par rareté et type</h3>", unsafe_allow_html=True)

    df_plot = df.copy()
    df_plot["rarity_simple"] = df_plot["rarity"].apply(_simplify_rarity)

    tab1, tab2, tab3 = st.tabs(["📦 Prix par rareté", "🔥 Prix par type", "🏆 Top 10 valorisées"])

    with tab1:
        q97 = df_plot["market_price"].quantile(0.97)
        fig_box = px.box(
            df_plot[df_plot["market_price"] < q97],
            x="rarity_simple", y="market_price",
            color="rarity_simple",
            color_discrete_map=RARITY_COLORS_MAP,
            category_orders={"rarity_simple": RARITY_SIMPLIFIED_ORDER + ["Other"]},
            labels={"rarity_simple": "Rareté", "market_price": "Prix marché ($)"},
            title="Distribution des prix par rareté (hors top 3%)",
        )
        fig_box.update_layout(**_fig_layout(), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        avg_type = (
            df_plot.dropna(subset=["type"])
            .groupby("type")["market_price"].mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        avg_type.columns = ["Type", "Prix moyen ($)"]
        fig_bar = px.bar(
            avg_type, x="Prix moyen ($)", y="Type", orientation="h",
            color="Prix moyen ($)",
            color_continuous_scale=["#74b9ff", YELLOW, RED],
            title="Prix moyen par type de carte",
        )
        fig_bar.update_layout(**_fig_layout(), showlegend=False, yaxis_categoryorder="total ascending")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        top10 = (
            df_plot.nlargest(10, "market_price")[
                ["name", "set_name", "rarity_simple", "type", "market_price"]
            ].reset_index(drop=True)
        )
        top10.index += 1
        fig_top = px.bar(
            top10.reset_index(), x="market_price", y="name", orientation="h",
            color="rarity_simple",
            color_discrete_map=RARITY_COLORS_MAP,
            labels={"market_price": "Prix ($)", "name": "Carte", "rarity_simple": "Rareté"},
            title="Top 10 cartes par prix de marché",
            hover_data=["set_name", "type"],
        )
        fig_top.update_layout(**_fig_layout(), yaxis_categoryorder="total ascending")
        st.plotly_chart(fig_top, use_container_width=True)

        display = top10[["name", "set_name", "rarity_simple", "market_price"]].copy()
        display.columns = ["Carte", "Set", "Rareté", "Prix ($)"]
        st.dataframe(
            display.style.background_gradient(subset=["Prix ($)"], cmap="YlOrRd"),
            use_container_width=True,
        )

    st.divider()

    # ── Section 4 : Stats globales ────────────────────────────────────────────
    st.markdown(f"<h3>📊 Stats globales du dataset</h3>", unsafe_allow_html=True)

    s1, s2, s3 = st.tabs(["🥧 Distribution rarétés", "🎯 Distribution target", "🔥 Corrélations"])

    with s1:
        rare_counts = df_plot["rarity_simple"].value_counts().reindex(RARITY_SIMPLIFIED_ORDER + ["Other"]).dropna()
        fig_pie = px.pie(
            values=rare_counts.values, names=rare_counts.index,
            color=rare_counts.index,
            color_discrete_map=RARITY_COLORS_MAP,
            title="Répartition des cartes par rareté",
        )
        fig_pie.update_layout(**_fig_layout())
        st.plotly_chart(fig_pie, use_container_width=True)

    with s2:
        t_counts = df["target"].value_counts().sort_index()
        fig_donut = px.pie(
            values=t_counts.values,
            names=["❌ Ne se valorisera pas", "✅ Se valorisera"],
            color_discrete_sequence=[RED, GREEN],
            title="Distribution de la target (variable à prédire)",
            hole=0.5,
        )
        fig_donut.update_layout(**_fig_layout())
        st.plotly_chart(fig_donut, use_container_width=True)

    with s3:
        corr_cols = FEATURE_COLS + ["target"]
        corr      = df[corr_cols].corr()
        fig_heat  = go.Figure(go.Heatmap(
            z=corr.values,
            x=[FEATURE_LABELS.get(c, c) for c in corr.columns],
            y=[FEATURE_LABELS.get(c, c) for c in corr.index],
            colorscale=[[0, "#74b9ff"], [0.5, DARK3], [1, RED]],
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_heat.update_layout(
            **_fig_layout(title="Corrélation entre les features et la target"),
            height=500,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


# ── Page 3 — Prédiction ───────────────────────────────────────────────────────

def page_prediction(df: pd.DataFrame, models: dict) -> None:
    _banner("🔮 Prédiction du prix", "Configurez une carte et obtenez une prédiction ML instantanée")

    if "xgb" not in models or "rf" not in models:
        st.error("Modèles non chargés. Lancez `python scripts/train_models.py` d'abord.")
        return

    st.markdown(f"<h3>Caractéristiques de la carte</h3>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            f"<div style='background:{DARK2}; border:1px solid {DARK3}; border-radius:12px; padding:20px;'>",
            unsafe_allow_html=True,
        )
        rarity      = st.selectbox("⭐ Rareté", RARITY_SIMPLIFIED_ORDER)
        p_type      = st.selectbox("🔥 Type", POKEMON_TYPES)
        hp          = st.number_input("❤️ HP", min_value=30, max_value=340, value=100, step=10)
        set_names   = sorted(df["set_name"].dropna().unique())
        chosen_set  = st.selectbox("📦 Extension", set_names)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(
            f"<div style='background:{DARK2}; border:1px solid {DARK3}; border-radius:12px; padding:20px;'>",
            unsafe_allow_html=True,
        )
        is_holo      = st.checkbox("✨ Carte Holographique ?")
        is_reverse   = st.checkbox("🔄 Carte Reverse Holo ?")
        has_evolution = st.checkbox("🔁 A une évolution ?", value=True)
        nb_attacks   = st.number_input("⚔️ Nombre d'attaques", min_value=1, max_value=4, value=2)
        set_year     = st.slider("📅 Année de sortie", min_value=1999, max_value=2025, value=2020)
        st.markdown("</div>", unsafe_allow_html=True)

    # Compute derived features
    rarity_enc   = RARITY_ENCODED_MAP.get(rarity, 1)
    set_age      = max(0, 2025 - set_year)
    hp_norm      = round(hp / 340.0, 4)

    # Estimate market price from the selected set's median
    set_median   = float(df[df["set_name"] == chosen_set]["market_price"].median()) if chosen_set else 5.0
    market_price = set_median
    spread       = market_price * 0.15
    low_price    = max(0.01, market_price - spread)
    high_price   = market_price + spread
    price_range  = round(high_price - low_price, 2)

    input_df = pd.DataFrame([{
        "rarity_encoded": rarity_enc,
        "set_age":        set_age,
        "price_range":    price_range,
        "is_holo":        int(is_holo),
        "hp_normalized":  hp_norm,
        "has_evolution":  int(has_evolution),
        "is_reverse":     int(is_reverse),
        "nb_attacks":     nb_attacks,
        "market_price":   market_price,
        "low_price":      round(low_price, 2),
        "high_price":     round(high_price, 2),
    }])

    st.markdown("<br>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("🔮 Prédire la valorisation", type="primary", use_container_width=True)

    if predict_clicked:
        try:
            xgb = models["xgb"]
            rf  = models["rf"]

            xgb_proba = xgb.predict_proba(input_df.astype(float))[0]
            rf_proba  = rf.predict_proba(input_df)[0]
            avg_proba = float((xgb_proba[1] + rf_proba[1]) / 2)
            prediction = 1 if avg_proba >= 0.5 else 0

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<h3>Résultat de la prédiction</h3>", unsafe_allow_html=True)

            if prediction == 1:
                st.success(f"✅ **Cette carte devrait prendre de la valeur !**  Confiance : {avg_proba*100:.1f}%")
            else:
                st.error(f"❌ **Cette carte ne devrait pas prendre de valeur** au-dessus de la moyenne.  Confiance : {(1-avg_proba)*100:.1f}%")

            pct_int   = int(avg_proba * 100)
            bar_color = GREEN if pct_int >= 70 else YELLOW if pct_int >= 50 else RED

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("⚡ XGBoost",       f"{xgb_proba[1]*100:.1f}%", "Valorisation" if xgb_proba[1] >= 0.5 else "Stable")
            mc2.metric("🌲 Random Forest",  f"{rf_proba[1]*100:.1f}%",  "Valorisation" if rf_proba[1] >= 0.5 else "Stable")
            mc3.metric("🤝 Consensus",       f"{avg_proba*100:.1f}%",   "Valorisation" if prediction == 1 else "Stable")

            st.markdown(
                f"<p style='color:{WHITE}; font-weight:600; margin-top:16px;'>Probabilité de valorisation</p>",
                unsafe_allow_html=True,
            )
            st.progress(float(avg_proba))
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; margin-top:4px; margin-bottom:20px;">
                <span style="color:{WHITE}; font-size:0.82rem;">0 % — Stable</span>
                <span style="color:{bar_color}; font-weight:800; font-size:1.15rem;">{pct_int} %</span>
                <span style="color:{WHITE}; font-size:0.82rem;">100 % — Valorisation sûre</span>
            </div>
            """, unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_proba * 100,
                number=dict(suffix="%", font=dict(color=YELLOW, size=40)),
                title=dict(text="Probabilité de valorisation", font=dict(color=WHITE, size=14)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=WHITE),
                    bar=dict(color=bar_color, thickness=0.3),
                    bgcolor=DARK3,
                    steps=[
                        dict(range=[0,  30], color="#3a1a1a"),
                        dict(range=[30, 50], color="#3a2a1a"),
                        dict(range=[50, 70], color="#1a2a2a"),
                        dict(range=[70,100], color="#1a3a1a"),
                    ],
                    threshold=dict(line=dict(color=WHITE, width=3), thickness=0.85, value=50),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor=DARK2, font=dict(color=WHITE), margin=dict(t=60, b=20), height=280,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Pourquoi ? — top 3 features du RF
            with st.expander("💡 Pourquoi cette prédiction ? Les 3 facteurs clés"):
                importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
                for feat, imp in importances.head(3).items():
                    val   = float(input_df[feat].iloc[0])
                    label = FEATURE_LABELS.get(feat, feat)
                    expl_tuple = FEATURE_EXPLAIN.get(feat, ("", lambda v: str(v)))
                    expl, fmt  = expl_tuple
                    bar_w = min(int(imp * 600), 100)
                    st.markdown(f"""
                    <div style="background:{DARK2}; border:1px solid {DARK3}; border-radius:10px;
                                padding:14px 18px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span style="color:{YELLOW}; font-weight:700;">{label}</span>
                            <span style="color:{WHITE}; font-weight:600;">{fmt(val)}</span>
                        </div>
                        <div style="background:{DARK3}; border-radius:4px; height:8px; margin-bottom:8px;">
                            <div style="background:{RED}; border-radius:4px; height:8px; width:{bar_w}%;"></div>
                        </div>
                        <p style="color:{WHITE}; font-size:0.85rem; opacity:0.8; margin:0;">{expl}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("🔧 Features envoyées au modèle"):
                disp = input_df.copy()
                disp.columns = [FEATURE_LABELS.get(c, c) for c in disp.columns]
                st.dataframe(disp, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Erreur lors de la prédiction : {exc}")


# ── Page 4 — Comparaison ──────────────────────────────────────────────────────

def page_comparaison(df: pd.DataFrame, models: dict) -> None:
    _banner("⚖️ Comparaison des modèles ML", "Random Forest vs XGBoost vs KMeans — performances et explications")

    metrics_df = load_metrics()

    t1, t2, t3, t4 = st.tabs(["📊 Tableau des performances", "📈 Graphiques", "🔍 Feature Importance", "💡 Explications"])

    with t1:
        if metrics_df is not None:
            best_acc = metrics_df["accuracy"].max()

            for _, row in metrics_df.iterrows():
                is_best    = row.get("accuracy") == best_acc
                border_col = YELLOW if is_best else DARK3
                badge_html = (
                    f" &nbsp;<span style='background:#1a3a1a; color:{GREEN}; border-radius:12px;"
                    f" padding:3px 10px; font-size:0.78rem;'>🏆 Meilleur modèle</span>"
                    if is_best else ""
                )
                metric_blocks = ""
                for key, lbl in [("accuracy", "Accuracy"), ("f1", "F1-score"), ("precision", "Précision"), ("recall", "Recall")]:
                    v    = row.get(key, 0) * 100
                    clr  = GREEN if v >= 90 else ORANGE if v >= 80 else RED
                    metric_blocks += f"""
                    <div style="text-align:center; background:{DARK3}; border-radius:10px; padding:12px;">
                        <div style="color:{WHITE}; font-size:0.82rem; opacity:0.75; margin-bottom:4px;">{lbl}</div>
                        <div style="color:{clr}; font-size:1.4rem; font-weight:800;">{v:.1f}%</div>
                    </div>"""

                st.markdown(f"""
                <div style="background:{DARK2}; border:2px solid {border_col}; border-radius:14px;
                            padding:22px 24px; margin-bottom:16px; box-shadow:0 3px 12px rgba(0,0,0,0.3);">
                    <div style="margin-bottom:14px;">
                        <span style="color:{YELLOW}; font-size:1.15rem; font-weight:800;">
                            {row.get('model_name', row['model_key'])}
                        </span>{badge_html}
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:14px;">
                        {metric_blocks}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Lancez `python scripts/main.py` pour générer les métriques.")

    with t2:
        if metrics_df is not None:
            metric_keys = ["accuracy", "f1", "precision", "recall"]
            metric_lbls = ["Accuracy", "F1-score", "Précision", "Recall"]
            colors      = [RED, YELLOW, "#74b9ff"]

            fig_bar = go.Figure()
            for i, (_, row) in enumerate(metrics_df.iterrows()):
                vals = [row.get(k, 0) for k in metric_keys]
                fig_bar.add_trace(go.Bar(
                    name=row["model_name"],
                    x=metric_lbls, y=vals,
                    marker_color=colors[i % len(colors)],
                    text=[f"{v*100:.1f}%" for v in vals],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                **_fig_layout(title="Comparaison des métriques par modèle"),
                barmode="group",
            )
            fig_bar.update_yaxes(range=[0.5, 1.05], tickformat=".0%")
            st.plotly_chart(fig_bar, use_container_width=True)

            cats    = metric_lbls + [metric_lbls[0]]
            fig_rad = go.Figure()
            for i, (_, row) in enumerate(metrics_df.iterrows()):
                vals = [row.get(k, 0) for k in metric_keys]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats,
                    fill="toself",
                    name=row["model_name"],
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)] + "33",
                ))
            fig_rad.update_layout(
                paper_bgcolor=DARK2,
                font=dict(color=WHITE),
                polar=dict(
                    bgcolor=DARK2,
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2d3561"),
                    angularaxis=dict(gridcolor="#2d3561"),
                ),
                legend=dict(bgcolor=DARK2),
                title=dict(text="Radar des performances", font=dict(color=YELLOW)),
            )
            st.plotly_chart(fig_rad, use_container_width=True)
        else:
            st.info("Lancez `python scripts/main.py` pour générer les métriques.")

    with t3:
        if "rf" in models:
            try:
                importances = pd.Series(
                    models["rf"].feature_importances_, index=FEATURE_COLS
                ).sort_values()
                labels     = [FEATURE_LABELS.get(f, f) for f in importances.index]
                q70, q40   = importances.quantile(0.7), importances.quantile(0.4)
                colors_imp = [
                    RED if v >= q70 else YELLOW if v >= q40 else "#74b9ff"
                    for v in importances.values
                ]
                fig_imp = go.Figure(go.Bar(
                    x=importances.values, y=labels, orientation="h",
                    marker_color=colors_imp,
                    text=[f"{v:.3f}" for v in importances.values],
                    textposition="outside",
                ))
                fig_imp.update_layout(**_fig_layout(title="Feature Importance — Random Forest"))
                fig_imp.update_xaxes(title="Importance (Gini)")
                fig_imp.update_yaxes(gridcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown(f"<h4 style='color:{YELLOW};'>Interprétation des 3 features les plus importantes</h4>",
                            unsafe_allow_html=True)
                top3 = pd.Series(models["rf"].feature_importances_, index=FEATURE_COLS).nlargest(3)
                for feat, val in top3.items():
                    lbl  = FEATURE_LABELS.get(feat, feat)
                    expl = FEATURE_EXPLAIN.get(feat, ("",))[0]
                    st.markdown(f"- **{lbl}** (score {val:.3f}) — {expl}")
            except Exception as exc:
                st.error(f"Erreur feature importance : {exc}")
        else:
            st.warning("Modèle Random Forest non chargé.")

    with t4:
        e1, e2, e3 = st.columns(3)
        with e1:
            with st.expander("🌲 Random Forest — Comment ça marche ?", expanded=True):
                st.markdown(f"""
                <p style="color:{WHITE}; line-height:1.75;">
                    Imaginez <strong style="color:{YELLOW};">200 experts indépendants</strong>
                    (les "arbres") qui chacun analysent la carte et votent.
                    La décision finale est celle de la majorité.
                </p>
                <p style="color:{WHITE}; line-height:1.75;">
                    <strong>✅ Avantages :</strong> très stable, résistant aux données aberrantes,
                    facile à interpréter via la feature importance.<br>
                    <strong>⚠️ Limite :</strong> parfois moins précis que XGBoost sur des données complexes.
                </p>
                """, unsafe_allow_html=True)
        with e2:
            with st.expander("⚡ XGBoost — Comment ça marche ?", expanded=True):
                st.markdown(f"""
                <p style="color:{WHITE}; line-height:1.75;">
                    XGBoost apprend de ses erreurs à chaque round. Chaque nouvel arbre corrige
                    les cas mal classés par le précédent.
                    C'est le <strong style="color:{YELLOW};">favori des compétitions ML</strong>.
                </p>
                <p style="color:{WHITE}; line-height:1.75;">
                    <strong>✅ Avantages :</strong> haute performance, gère bien le déséquilibre de classes.<br>
                    <strong>⚠️ Limite :</strong> nécessite plus de réglage, moins interprétable.
                </p>
                """, unsafe_allow_html=True)
        with e3:
            with st.expander("🔵 KMeans — Quels profils identifiés ?", expanded=True):
                if "kmeans" in models:
                    try:
                        km     = models["kmeans"]
                        X      = df[FEATURE_COLS].astype(float)
                        labels = km.predict(X)
                        df_c   = df.copy()
                        df_c["cluster"] = labels
                        for cid, cname in CLUSTER_NAMES.items():
                            sub = df_c[df_c["cluster"] == cid]
                            if sub.empty:
                                continue
                            dominant_rarity = sub["rarity"].mode()[0] if not sub["rarity"].mode().empty else "N/A"
                            st.markdown(f"""
                            <div style="background:{DARK3}; border-radius:8px; padding:12px; margin-bottom:8px;">
                                <strong style="color:{CLUSTER_COLORS[cid]};">
                                    {CLUSTER_EMOJIS[cid]} {cname}
                                </strong><br>
                                <span style="color:{WHITE}; font-size:0.83rem; line-height:1.6;">
                                    {len(sub):,} cartes<br>
                                    Prix moyen : <strong>${sub['market_price'].mean():.2f}</strong><br>
                                    Rareté dominante : {dominant_rarity}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as exc:
                        st.error(f"Erreur KMeans : {exc}")
                else:
                    st.info("Modèle KMeans non chargé.")


# ── Entry point ───────────────────────────────────────────────────────────────

PAGES = {
    "🏠 Accueil":             ("accueil",     page_accueil,     False),
    "📊 Dashboard des cotes": ("dashboard",   page_dashboard,   False),
    "🔮 Prédiction":          ("prediction",  page_prediction,  True),
    "⚖️ Comparaison modèles": ("comparaison", page_comparaison, True),
}


def build_app() -> None:
    st.set_page_config(
        page_title="Pokemon Card Investor",
        page_icon="🎴",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    try:
        df     = load_data()
        models = load_models()
    except Exception as exc:
        st.error(f"Erreur critique au chargement : {exc}")
        st.stop()

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:24px 0 16px;">
            <div style="font-size:3.5rem;">🎴</div>
            <div style="color:{YELLOW}; font-size:1.15rem; font-weight:900; margin-top:8px;">
                Pokemon Card Investor
            </div>
            <div style="color:{WHITE}; font-size:0.78rem; opacity:0.65; margin-top:4px;">
                ML · {len(df):,} cartes · {len(models)} modèles chargés
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{RED}; margin:0 0 16px;'>", unsafe_allow_html=True)

        page_name = st.radio(
            "Navigation",
            options=list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown(f"""
        <div style="background:{DARK3}; border-radius:10px; padding:14px; margin-top:4px;">
            <p style="color:{YELLOW}; font-weight:800; margin:0 0 8px; font-size:0.82rem;">📊 À propos du dataset</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">🃏 {len(df):,} cartes analysées</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">📦 {df['set_name'].nunique()} sets différents</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">📅 {df['year'].min()} — {df['year'].max()}</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">💰 Prix moyen ${df['market_price'].mean():.2f}</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">📈 {df['target'].mean()*100:.1f}% cartes valorisées</p>
        </div>
        """, unsafe_allow_html=True)

    _, fn, needs_models = PAGES[page_name]
    if needs_models:
        fn(df, models)
    else:
        fn(df)


if __name__ == "__main__":
    build_app()
