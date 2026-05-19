"""Vinci Autoroutes — Prédiction des accidents graves. Streamlit app."""
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

# ── Palette Vinci Autoroutes ───────────────────────────────────────────────────
NAVY   = "#003087"
ORANGE = "#FF6B00"
WHITE  = "#FFFFFF"
LIGHT  = "#F4F6FA"
DARK   = "#0A1628"
DARK2  = "#0D1F3C"
GREY   = "#6B7280"
GREEN  = "#10B981"
RED    = "#EF4444"
YELLOW = "#F59E0B"

# ── Codes BAAC → labels lisibles ─────────────────────────────────────────────
LUM_LABELS  = {1: "Plein jour", 2: "Crépuscule/Aube", 3: "Nuit sans éclairage",
               4: "Nuit - éclairage éteint", 5: "Nuit - éclairage allumé"}
ATM_LABELS  = {1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige/Grêle",
               5: "Brouillard/Fumée", 6: "Vent fort/Tempête", 7: "Éblouissant", 8: "Couvert", 9: "Autre"}
COL_LABELS  = {1: "Frontale", 2: "Par l'arrière", 3: "Côté", 4: "En chaîne",
               5: "Multiples - autres", 6: "Sans collision"}
SURF_LABELS = {1: "Normale", 2: "Mouillée", 3: "Flaques", 4: "Inondée",
               5: "Enneigée", 6: "Boueuse", 7: "Verglacée", 8: "Corps gras/Huile", 9: "Autre"}
CATV_LABELS = {1: "Vélo", 2: "Cyclomoteur", 3: "Voiturette", 7: "Voiture",
               10: "Utilitaire léger", 13: "Poids lourd", 14: "Poids lourd + remorque",
               15: "Tracteur routier", 17: "Autocar", 33: "Tramway", 37: "Voiture + remorque"}
INFRA_LABELS = {
    0: "Aucune infrastructure particulière",
    1: "Souterrain / Tunnel",
    2: "Pont / Autopont",
    3: "Bretelle d'échangeur",
    5: "Carrefour aménagé",
    7: "Zone de péage",
    8: "Chantier / travaux",
    9: "Autre infrastructure",
}
MOIS_LABELS = {1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
               7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"}

FEATURE_COLS = [
    # Features environnementales — observables / planifiables par Vinci
    "lum", "atm",
    # Infrastructure routière
    "circ", "nbv", "prof", "surf", "infra", "situ", "vma",
    # Temporel / saisonnalité
    "mois", "saison",
]
FEATURE_LABELS = {
    "lum": "Luminosité", "atm": "Météo",
    "circ": "Régime circulation", "nbv": "Nb de voies", "prof": "Profil de la route",
    "surf": "État de surface", "infra": "Infrastructure", "situ": "Situation",
    "vma": "Vitesse max (km/h)",
    "mois": "Mois", "saison": "Saison",
}

CLUSTER_NAMES  = {0: "Accidents mineurs", 1: "Accidents modérés", 2: "Accidents graves"}
CLUSTER_COLORS = {0: GREEN, 1: YELLOW, 2: RED}

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {LIGHT};
    color: {DARK};
}}
[data-testid="stHeader"] {{ background-color: {NAVY}; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, {DARK2} 100%);
    border-right: 3px solid {ORANGE};
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
[data-testid="stSidebar"] [data-baseweb="radio"] label {{
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 4px;
    transition: background 0.2s;
}}
[data-testid="stSidebar"] [data-baseweb="radio"] label:hover {{
    background: rgba(255,107,0,0.3);
}}

[data-testid="metric-container"] {{
    background: {WHITE};
    border: 1px solid #E5E7EB;
    border-top: 4px solid {NAVY};
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,48,135,0.10);
}}
[data-testid="stMetricValue"] {{
    color: {NAVY} !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {GREY} !important;
    font-weight: 500 !important;
}}

.stButton > button {{
    background: {ORANGE} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
}}
.stButton > button:hover {{
    background: {NAVY} !important;
    box-shadow: 0 4px 16px rgba(0,48,135,0.35) !important;
}}

h1 {{ color: {NAVY} !important; font-weight: 900 !important; }}
h2 {{ color: {NAVY} !important; font-weight: 700 !important; }}
h3 {{ color: {ORANGE} !important; font-weight: 600 !important; }}

.stTabs [data-baseweb="tab-list"] {{
    background: {WHITE};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #E5E7EB;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {NAVY};
    border-radius: 8px;
    font-weight: 600;
    padding: 6px 20px;
}}
.stTabs [aria-selected="true"] {{
    background: {NAVY} !important;
    color: {WHITE} !important;
}}

.stSelectbox > div > div,
.stNumberInput > div > div > input {{
    background: {WHITE} !important;
    border-color: #D1D5DB !important;
    border-radius: 8px !important;
    color: {DARK} !important;
}}

[data-testid="stExpander"] {{
    background: {WHITE};
    border: 1px solid #E5E7EB;
    border-radius: 12px;
}}

hr {{ border-color: #E5E7EB !important; }}
[data-testid="stAlert"] {{ border-radius: 12px !important; }}
</style>
"""


# ── Cache ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des données BAAC…")
def load_data() -> pd.DataFrame:
    path = DATA_DIR / "processed_dataset.csv"
    if not path.exists():
        st.error("Dataset introuvable. Lancez : python scripts/prepare_data.py")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Chargement des modèles…")
def load_models() -> dict:
    models: dict = {}
    for key, fname in [("rf", "random_forest.joblib"), ("xgb", "xgboost.joblib"),
                       ("kmeans", "kmeans.joblib")]:
        p = MODELS_DIR / fname
        if p.exists():
            try:
                models[key] = joblib.load(p)
            except Exception as exc:
                st.warning(f"Impossible de charger {fname} : {exc}")
    return models


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    if MODEL_METRICS_FILE.exists():
        return pd.read_csv(MODEL_METRICS_FILE)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_layout(title: str = "", **kwargs) -> dict:
    base = dict(
        paper_bgcolor=WHITE,
        plot_bgcolor=LIGHT,
        font=dict(color=DARK, family="sans-serif", size=12),
        xaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB", linecolor="#D1D5DB"),
        yaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB", linecolor="#D1D5DB"),
        margin=dict(t=55, b=40, l=45, r=20),
        legend=dict(bgcolor=WHITE, bordercolor="#E5E7EB", borderwidth=1),
    )
    if title:
        base["title"] = dict(text=title, font=dict(color=NAVY, size=15))
    base.update(kwargs)
    return base


def _banner(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {NAVY} 0%, #005BAC 60%, {DARK2} 100%);
        border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
        border-left: 6px solid {ORANGE};
        box-shadow: 0 6px 24px rgba(0,48,135,0.25);
    ">
        <h1 style="color:{WHITE} !important; margin:0; font-size:1.8rem;">{title}</h1>
        <p style="color:rgba(255,255,255,0.85); margin:8px 0 0; font-size:1rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def _info_card(emoji: str, title: str, body: str, border: str = NAVY) -> str:
    return f"""
    <div style="background:{WHITE}; border:1px solid #E5E7EB; border-top:4px solid {border};
                border-radius:14px; padding:22px 20px; height:100%;
                box-shadow:0 2px 8px rgba(0,48,135,0.08);">
        <div style="font-size:2rem; margin-bottom:10px;">{emoji}</div>
        <div style="color:{NAVY}; font-weight:800; font-size:1rem; margin-bottom:10px;">{title}</div>
        <div style="color:{GREY}; font-size:0.88rem; line-height:1.65;">{body}</div>
    </div>"""


# ── Page 1 — Contexte Vinci ──────────────────────────────────────────────────

def page_contexte(df: pd.DataFrame) -> None:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {NAVY} 0%, #005BAC 50%, {DARK2} 100%);
        border-radius: 20px; padding: 48px 40px; margin-bottom: 32px;
        box-shadow: 0 8px 32px rgba(0,48,135,0.30);
        border: 1px solid rgba(255,107,0,0.4);
    ">
        <div style="font-size:3.5rem; margin-bottom:16px;">🛣️</div>
        <h1 style="font-size:2.4rem; color:{WHITE} !important; margin:0; font-weight:900;"> Vinci Autoroutes </h1>
        <h2 style="font-size:1.3rem; color:{ORANGE} !important; margin:10px 0 16px; font-weight:700;">
            Prédiction des accidents graves
        </h2>
        <p style="color:rgba(255,255,255,0.85); font-size:1.05rem; max-width:680px; line-height:1.7; margin:0;">
            Système d'aide à la décision basé sur le Machine Learning pour anticiper
            la gravité des accidents sur le réseau autoroutier et optimiser le déploiement
            des équipes d'intervention.
        </p>
        <div style="margin-top:24px; display:flex; gap:10px; flex-wrap:wrap;">
            <span style="background:rgba(255,107,0,0.2); border:1px solid {ORANGE}; border-radius:20px;
                         padding:6px 16px; color:{WHITE}; font-size:0.85rem;">🤖 Machine Learning</span>
            <span style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.3);
                         border-radius:20px; padding:6px 16px; color:{WHITE}; font-size:0.85rem;">
                📊 {len(df['Num_Acc'].unique() if 'Num_Acc' in df.columns else df):,} accidents analysés</span>
            <span style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.3);
                         border-radius:20px; padding:6px 16px; color:{WHITE}; font-size:0.85rem;">
                📅 2020 – 2024</span>
            <span style="background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.3);
                         border-radius:20px; padding:6px 16px; color:{WHITE}; font-size:0.85rem;">
                🏆 3 modèles ML</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    n_acc = df["Num_Acc"].nunique() if "Num_Acc" in df.columns else len(df)
    pct_grave = df["target"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚗 Usagers analysés",    f"{len(df):,}")
    c2.metric("🚧 Accidents uniques",   f"{n_acc:,}")
    c3.metric("⚠️ Accidents graves",    f"{pct_grave:.1f}%")
    c4.metric("📅 Période couverte",    "2020 – 2024")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"<h2>Objectif business</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{WHITE}; border:1px solid #E5E7EB; border-left:5px solid {ORANGE};
                border-radius:12px; padding:24px 28px; margin-bottom:24px;
                box-shadow:0 2px 8px rgba(0,48,135,0.08);">
        <p style="color:{DARK}; font-size:1rem; line-height:1.8; margin:0;">
            Matmut gère <strong style="color:{NAVY};">plus de 4 600 km d'autoroutes</strong> en France
            et doit chaque jour décider du niveau d'intervention à déployer lors d'un accident.
            Ce système prédit en temps réel si un accident est susceptible d'être
            <strong style="color:{RED};">grave (hospitalisation ou décès)</strong>
            ou <strong style="color:{GREEN};">léger (indemne ou blessé léger)</strong>,
            permettant d'optimiser le déploiement des équipes de secours et de réduire les temps d'intervention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(_info_card(
            "📊", "Dashboard accidents",
            "Visualisez la carte des accidents sur autoroutes avec leur gravité, "
            "les tendances mensuelles, par météo, par luminosité et les zones à risque.",
            NAVY,
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(_info_card(
            "🚨", "Prédiction en temps réel",
            "Renseignez les conditions de l'accident (météo, luminosité, type de véhicule…) "
            "et obtenez une évaluation du risque avec recommandation d'intervention Vinci.",
            ORANGE,
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(_info_card(
            "⚖️", "Comparaison des modèles",
            "Comparez Random Forest, XGBoost et KMeans : accuracy, F1-score, feature importance "
            "et explication pédagogique de chaque algorithme.",
            GREEN,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<h2>Aperçu du dataset BAAC (autoroutes)</h2>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        by_year = df.groupby("annee")["target"].agg(["count", "mean"]).reset_index()
        by_year.columns = ["Année", "Usagers", "Taux grave"]
        fig = px.bar(
            by_year, x="Année", y="Usagers",
            color="Taux grave",
            color_continuous_scale=[GREEN, YELLOW, RED],
            title="Usagers par année (autoroutes)",
            labels={"Usagers": "Nb usagers", "Taux grave": "Taux gravité"},
        )
        fig.update_layout(**_fig_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        t_counts = df["target"].value_counts().sort_index()
        fig_d = px.pie(
            values=t_counts.values,
            names=["✅ Accident léger", "⚠️ Accident grave"],
            color_discrete_sequence=[GREEN, RED],
            title="Répartition des gravités",
            hole=0.52,
        )
        fig_d.update_layout(**_fig_layout())
        fig_d.update_traces(textfont_size=13)
        st.plotly_chart(fig_d, use_container_width=True)


# ── Page 2 — Dashboard accidents ──────────────────────────────────────────────

def page_dashboard(df: pd.DataFrame) -> None:
    _banner("📊 Dashboard accidents autoroutes", "Analyse des accidents corporels sur le réseau Vinci (2020–2024)")

    # ── Filtres ──────────────────────────────────────────────────────────────
    with st.expander("🔽 Filtres", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        annees  = sorted(df["annee"].dropna().unique().astype(int)) if "annee" in df.columns else []
        mois_l  = sorted(df["mois"].dropna().unique().astype(int)) if "mois" in df.columns else []
        atm_l   = sorted(df["atm"].dropna().unique().astype(int)) if "atm" in df.columns else []
        lum_l   = sorted(df["lum"].dropna().unique().astype(int)) if "lum" in df.columns else []

        sel_annees = fc1.multiselect("Année", annees, default=annees)
        sel_mois   = fc2.multiselect("Mois", mois_l, default=mois_l,
                                      format_func=lambda m: MOIS_LABELS.get(int(m), str(m)))
        sel_atm    = fc3.multiselect("Météo", atm_l, default=atm_l,
                                      format_func=lambda a: ATM_LABELS.get(int(a), str(a)))
        sel_lum    = fc4.multiselect("Luminosité", lum_l, default=lum_l,
                                      format_func=lambda l: LUM_LABELS.get(int(l), str(l)))

    mask = pd.Series([True] * len(df), index=df.index)
    if sel_annees and "annee" in df.columns:
        mask &= df["annee"].isin(sel_annees)
    if sel_mois and "mois" in df.columns:
        mask &= df["mois"].isin(sel_mois)
    if sel_atm and "atm" in df.columns:
        mask &= df["atm"].isin(sel_atm)
    if sel_lum and "lum" in df.columns:
        mask &= df["lum"].isin(sel_lum)
    dff = df[mask].copy()

    if dff.empty:
        st.warning("Aucun accident avec ces filtres.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Usagers filtrés", f"{len(dff):,}")
    m2.metric("🚧 Accidents",
              f"{dff['Num_Acc'].nunique():,}" if "Num_Acc" in dff.columns else "N/A")
    m3.metric("⚠️ Taux gravité", f"{dff['target'].mean()*100:.1f}%")
    m4.metric("💀 Accidents graves",
              f"{dff['target'].sum():,}")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ Carte", "📅 Évolution", "🌡️ Météo & Lumière", "🔥 Heatmap", "🏆 Top dép.", "🚗 Véhicules"
    ])

    # ── Onglet 1 : Carte Folium ──────────────────────────────────────────────
    with tab1:
        has_coords = "lat" in dff.columns and "long" in dff.columns
        if not has_coords:
            st.info("Coordonnées GPS non disponibles dans ce dataset.")
        else:
            try:
                import folium
                from streamlit_folium import folium_static

                geo = dff.dropna(subset=["lat", "long"]).copy()
                geo = geo[(geo["lat"].between(42, 51)) & (geo["long"].between(-5, 8))]

                # Échantillon de 2 000 points max pour la lisibilité
                if len(geo) > 2000:
                    geo = geo.sample(2000, random_state=42)

                m = folium.Map(location=[46.8, 2.3], zoom_start=6,
                               tiles="CartoDB positron")

                for _, row in geo.iterrows():
                    color = "red" if row["target"] == 1 else "green"
                    folium.CircleMarker(
                        location=[row["lat"], row["long"]],
                        radius=4,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"Gravité: {'Grave' if row['target']==1 else 'Léger'} | Année: {row.get('annee','')}",
                    ).add_to(m)

                st.caption(f"🔴 Accident grave  🟢 Accident léger  — {len(geo):,} points affichés")
                folium_static(m, width=900, height=500)
            except ImportError:
                st.info("Installez `streamlit-folium` pour la carte interactive.")

    # ── Onglet 2 : Évolution mensuelle ───────────────────────────────────────
    with tab2:
        if "annee" in dff.columns and "mois" in dff.columns:
            evo = (dff.groupby(["annee", "mois"])
                   .agg(nb=("target", "count"), graves=("target", "sum"))
                   .reset_index())
            evo["date"] = pd.to_datetime(dict(year=evo["annee"], month=evo["mois"], day=1))
            evo["taux"] = evo["graves"] / evo["nb"] * 100

            c1, c2 = st.columns(2)
            with c1:
                fig_evo = px.line(
                    evo, x="date", y="nb", color="annee",
                    title="Évolution mensuelle du nombre d'accidents",
                    labels={"nb": "Nb usagers", "date": "Date", "annee": "Année"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_evo.update_layout(**_fig_layout())
                st.plotly_chart(fig_evo, use_container_width=True)

            with c2:
                fig_tx = px.line(
                    evo, x="date", y="taux", color="annee",
                    title="Taux de gravité mensuel (%)",
                    labels={"taux": "Taux graves (%)", "date": "Date", "annee": "Année"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_tx.update_traces(line_width=2)
                fig_tx.update_layout(**_fig_layout())
                st.plotly_chart(fig_tx, use_container_width=True)

    # ── Onglet 3 : Météo & Luminosité ────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            if "atm" in dff.columns:
                atm_g = (dff.groupby("atm")["target"]
                         .agg(nb="count", graves="sum").reset_index())
                atm_g["label"] = atm_g["atm"].map(lambda x: ATM_LABELS.get(int(x), str(x)))
                atm_g["taux"]  = atm_g["graves"] / atm_g["nb"] * 100
                fig_atm = px.bar(
                    atm_g.sort_values("taux", ascending=False),
                    x="label", y="taux",
                    color="taux",
                    color_continuous_scale=[GREEN, YELLOW, RED],
                    title="Taux de gravité par météo (%)",
                    labels={"label": "Météo", "taux": "Taux graves (%)"},
                )
                fig_atm.update_layout(**_fig_layout(), showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_atm, use_container_width=True)

        with c2:
            if "lum" in dff.columns:
                lum_g = (dff.groupby("lum")["target"]
                         .agg(nb="count", graves="sum").reset_index())
                lum_g["label"] = lum_g["lum"].map(lambda x: LUM_LABELS.get(int(x), str(x)))
                lum_g["taux"]  = lum_g["graves"] / lum_g["nb"] * 100
                fig_lum = px.bar(
                    lum_g.sort_values("taux", ascending=False),
                    x="label", y="taux",
                    color="taux",
                    color_continuous_scale=[GREEN, YELLOW, RED],
                    title="Taux de gravité par luminosité (%)",
                    labels={"label": "Luminosité", "taux": "Taux graves (%)"},
                )
                fig_lum.update_layout(**_fig_layout(), showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_lum, use_container_width=True)

    # ── Onglet 4 : Heatmap mois × jour ──────────────────────────────────────
    with tab4:
        if "mois" in dff.columns and "jour" in dff.columns:
            heat = (dff.groupby(["mois", "jour"])["target"]
                    .mean().reset_index()
                    .pivot(index="mois", columns="jour", values="target"))
            fig_h = go.Figure(go.Heatmap(
                z=heat.values,
                x=[str(int(c)) for c in heat.columns],
                y=[MOIS_LABELS.get(int(r), str(r)) for r in heat.index],
                colorscale=[[0, GREEN], [0.5, YELLOW], [1, RED]],
                colorbar=dict(title="Taux gravité"),
                hovertemplate="Mois: %{y}<br>Jour: %{x}<br>Taux grave: %{z:.1%}<extra></extra>",
            ))
            fig_h.update_layout(
                **_fig_layout(title="Heatmap gravité : mois × jour du mois"),
                height=450,
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # ── Onglet 5 : Top départements ──────────────────────────────────────────
    with tab5:
        if "dep" in dff.columns:
            top_dep = (dff.groupby("dep")["target"]
                       .agg(nb="count", graves="sum")
                       .reset_index())
            top_dep["taux"] = top_dep["graves"] / top_dep["nb"] * 100
            top10 = top_dep.nlargest(10, "graves").reset_index(drop=True)
            top10["dep"] = top10["dep"].astype(str)

            fig_dep = px.bar(
                top10, x="graves", y="dep", orientation="h",
                color="taux",
                color_continuous_scale=[YELLOW, RED],
                title="Top 10 départements — accidents graves sur autoroutes",
                labels={"graves": "Nb accidents graves", "dep": "Département", "taux": "Taux (%)"},
                text="graves",
            )
            fig_dep.update_layout(**_fig_layout(), yaxis_categoryorder="total ascending")
            st.plotly_chart(fig_dep, use_container_width=True)

    # ── Onglet 6 : Véhicules ─────────────────────────────────────────────────
    with tab6:
        if "catv" in dff.columns:
            catv_g = (dff.groupby("catv")["target"]
                      .agg(nb="count", graves="sum").reset_index())
            catv_g["label"] = catv_g["catv"].map(
                lambda x: CATV_LABELS.get(int(x), f"Code {int(x)}") if pd.notna(x) else "Inconnu"
            )
            catv_g["taux"] = catv_g["graves"] / catv_g["nb"] * 100
            catv_g = catv_g[catv_g["nb"] >= 10].sort_values("nb", ascending=False).head(12)

            c1, c2 = st.columns(2)
            with c1:
                fig_cv = px.bar(
                    catv_g, x="label", y="nb",
                    color="taux",
                    color_continuous_scale=[GREEN, YELLOW, RED],
                    title="Accidents par type de véhicule",
                    labels={"label": "Type", "nb": "Nb usagers"},
                )
                fig_cv.update_layout(**_fig_layout(), coloraxis_showscale=False)
                st.plotly_chart(fig_cv, use_container_width=True)

            with c2:
                fig_tx = px.bar(
                    catv_g.sort_values("taux", ascending=False),
                    x="label", y="taux",
                    color="taux",
                    color_continuous_scale=[GREEN, YELLOW, RED],
                    title="Taux de gravité par type de véhicule (%)",
                    labels={"label": "Type", "taux": "Taux graves (%)"},
                )
                fig_tx.update_layout(**_fig_layout(), coloraxis_showscale=False)
                st.plotly_chart(fig_tx, use_container_width=True)


# ── Page 3 — Prédiction gravité ───────────────────────────────────────────────

def page_prediction(df: pd.DataFrame, models: dict) -> None:
    _banner("🚨 Prédiction de la gravité", "Évaluez le risque et recevez une recommandation d'intervention Vinci")

    if "rf" not in models and "xgb" not in models:
        st.error("Modèles non chargés. Lancez `python scripts/train_models.py` d'abord.")
        return

    # Médianes pour les features non saisies
    medians = {col: float(df[col].median()) for col in FEATURE_COLS if col in df.columns}

    st.markdown("<h3>Conditions environnementales</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div style='background:{WHITE}; border:1px solid #E5E7EB; border-radius:12px; padding:20px;'>"
            f"<p style='font-weight:700; color:{NAVY}; margin-bottom:12px;'>🌦️ Météo & Visibilité</p>",
            unsafe_allow_html=True)
        lum_choice  = st.selectbox("☀️ Luminosité",
                                    options=list(LUM_LABELS.keys()),
                                    format_func=lambda k: LUM_LABELS[k])
        atm_choice  = st.selectbox("🌧️ Conditions météo",
                                    options=list(ATM_LABELS.keys()),
                                    format_func=lambda k: ATM_LABELS[k])
        surf_choice = st.selectbox("🛣️ État de la chaussée",
                                    options=list(SURF_LABELS.keys()),
                                    format_func=lambda k: SURF_LABELS[k])
        mois_choice = st.selectbox("📅 Mois",
                                    options=list(MOIS_LABELS.keys()),
                                    format_func=lambda k: MOIS_LABELS[k])
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"<div style='background:{WHITE}; border:1px solid #E5E7EB; border-radius:12px; padding:20px;'>"
            f"<p style='font-weight:700; color:{NAVY}; margin-bottom:12px;'>🏗️ Infrastructure routière</p>",
            unsafe_allow_html=True)
        infra_choice = st.selectbox("🏗️ Type d'infrastructure",
                                     options=list(INFRA_LABELS.keys()),
                                     format_func=lambda k: INFRA_LABELS[k])
        CIRC_LABELS = {1: "Sens unique", 2: "Bidirectionnel", 3: "Voies séparées", 4: "Variable"}
        circ_choice = st.selectbox("🔄 Régime de circulation",
                                    options=list(CIRC_LABELS.keys()),
                                    format_func=lambda k: CIRC_LABELS[k])
        nbv_choice  = st.slider("🛣️ Nombre de voies", min_value=1, max_value=6, value=2)
        PROF_LABELS = {1: "Plat", 2: "Pente", 3: "Sommet de côte", 4: "Bas de côte"}
        prof_choice = st.selectbox("📐 Profil de la route",
                                    options=list(PROF_LABELS.keys()),
                                    format_func=lambda k: PROF_LABELS[k])
        vma_choice  = st.selectbox("🚀 Vitesse max autorisée (km/h)",
                                    options=[70, 80, 90, 110, 130], index=4)
        st.markdown("</div>", unsafe_allow_html=True)

    # Calcul saison depuis mois
    saison_map = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
                  6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}
    saison_choice = saison_map.get(mois_choice, 1)

    input_row = {col: medians.get(col, 0) for col in FEATURE_COLS}
    input_row.update({
        "lum": lum_choice, "atm": atm_choice,
        "surf": surf_choice, "infra": infra_choice,
        "circ": circ_choice, "nbv": float(nbv_choice),
        "prof": prof_choice, "vma": float(vma_choice),
        "mois": mois_choice, "saison": saison_choice,
    })
    input_df = pd.DataFrame([input_row])[FEATURE_COLS]

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        clicked = st.button("🚨 Évaluer le risque", type="primary", use_container_width=True)

    if clicked:
        try:
            probas: list[float] = []
            if "xgb" in models:
                probas.append(float(models["xgb"].predict_proba(input_df.astype(float))[0][1]))
            if "rf" in models:
                probas.append(float(models["rf"].predict_proba(input_df)[0][1]))
            avg_proba = float(np.mean(probas)) if probas else 0.5
            prediction = int(avg_proba >= 0.5)

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.error(f"⚠️ **ACCIDENT GRAVE — Déployer équipe renforcée**  "
                         f"(probabilité grave : {avg_proba*100:.1f}%)")
            else:
                st.success(f"✅ **Accident léger — Intervention standard**  "
                           f"(probabilité grave : {avg_proba*100:.1f}%)")

            if len(probas) > 1:
                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("⚡ XGBoost",       f"{probas[0]*100:.1f}%", "Grave" if probas[0] >= 0.5 else "Léger")
                pm2.metric("🌲 Random Forest",  f"{probas[1]*100:.1f}%", "Grave" if probas[1] >= 0.5 else "Léger")
                pm3.metric("🤝 Consensus",      f"{avg_proba*100:.1f}%", "Grave" if prediction == 1 else "Léger")

            st.markdown(
                f"<p style='font-weight:600; margin-top:16px; color:{NAVY};'>Probabilité d'accident grave</p>",
                unsafe_allow_html=True)
            st.progress(float(avg_proba))

            # Gauge
            bar_color = RED if avg_proba >= 0.6 else YELLOW if avg_proba >= 0.4 else GREEN
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_proba * 100,
                number=dict(suffix="%", font=dict(color=NAVY, size=40)),
                title=dict(text="Probabilité de gravité", font=dict(color=DARK, size=14)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=DARK),
                    bar=dict(color=bar_color, thickness=0.3),
                    bgcolor=LIGHT,
                    steps=[
                        dict(range=[0, 30],  color="#D1FAE5"),
                        dict(range=[30, 60], color="#FEF3C7"),
                        dict(range=[60, 100], color="#FEE2E2"),
                    ],
                    threshold=dict(line=dict(color=NAVY, width=3), thickness=0.85, value=50),
                ),
            ))
            fig_gauge.update_layout(
                paper_bgcolor=WHITE, font=dict(color=DARK),
                margin=dict(t=60, b=20), height=280,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Recommandation Vinci
            with st.expander("📋 Recommandation Vinci"):
                if prediction == 1:
                    st.markdown(f"""
                    <div style="background:#FEE2E2; border-left:5px solid {RED}; border-radius:8px; padding:18px;">
                        <strong style="color:{RED};">NIVEAU D'INTERVENTION : ÉLEVÉ</strong><br><br>
                        🚑 Déployer ambulance et équipe médicale<br>
                        🚒 Alerter les pompiers<br>
                        🚔 Sécurisation complète de la zone (3 km)<br>
                        📡 Activer les panneaux d'information dynamiques<br>
                        ⛔ Fermeture préventive des voies concernées<br>
                        🚁 Évaluer la nécessité d'un hélitreuillage
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#D1FAE5; border-left:5px solid {GREEN}; border-radius:8px; padding:18px;">
                        <strong style="color:{GREEN};">NIVEAU D'INTERVENTION : STANDARD</strong><br><br>
                        🚗 Patrouille autoroutière<br>
                        🔸 Baliser la zone (500 m)<br>
                        📞 Prévenir les secours de garde<br>
                        ℹ️ Information PMV si nécessaire<br>
                        🔄 Surveillance du trafic aval
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Erreur lors de la prédiction : {exc}")


# ── Page 4 — Comparaison modèles ──────────────────────────────────────────────

def page_comparaison(df: pd.DataFrame, models: dict) -> None:
    _banner("⚖️ Comparaison des modèles ML",
            "Random Forest vs XGBoost vs KMeans — performances et explications")

    metrics_df = load_metrics()
    METRIC_KEYS = ["accuracy", "f1", "precision", "recall"]
    METRIC_LBLS = ["Accuracy", "F1-score", "Précision", "Recall"]
    MODEL_COLORS = [NAVY, ORANGE, GREEN]

    t1, t2, t3, t4 = st.tabs(["📊 Performances", "📈 Graphiques", "🔍 Feature Importance", "💡 Explications"])

    # ── Tab 1 : Tableau ──────────────────────────────────────────────────────
    with t1:
        if metrics_df is None:
            st.info("Lancez `python scripts/train_models.py` pour générer les métriques.")
        else:
            mdf = metrics_df.copy()
            best_acc = mdf["accuracy"].max()

            for _, row in mdf.iterrows():
                is_best    = abs(row["accuracy"] - best_acc) < 1e-9
                border_col = ORANGE if is_best else "#E5E7EB"
                badge      = (f"&nbsp;<span style='background:#D1FAE5; color:{GREEN}; "
                              f"border-radius:12px; padding:3px 12px; font-size:0.78rem; "
                              f"font-weight:700;'>🏆 Meilleur modèle</span>" if is_best else "")

                # Header card (no indented content to avoid Markdown code-block misparse)
                st.markdown(
                    f'<div style="background:{WHITE};border:2px solid {border_col};border-radius:14px;'
                    f'padding:22px 24px 12px;margin-bottom:4px;box-shadow:0 2px 8px rgba(0,48,135,0.08);">'
                    f'<span style="color:{NAVY};font-size:1.2rem;font-weight:800;">'
                    f'{row.get("model_name", row["model_key"])}</span>{badge}</div>',
                    unsafe_allow_html=True,
                )
                # Metric tiles via native Streamlit columns (avoids HTML-in-Markdown issue)
                metric_cols = st.columns(4)
                for mc, (key, lbl) in zip(metric_cols, zip(METRIC_KEYS, METRIC_LBLS)):
                    v   = float(row.get(key, 0)) * 100
                    clr = GREEN if v >= 70 else YELLOW if v >= 55 else RED
                    with mc:
                        st.markdown(
                            f'<div style="text-align:center;background:{LIGHT};border-radius:10px;'
                            f'padding:14px 8px;margin-bottom:16px;">'
                            f'<div style="color:{GREY};font-size:0.8rem;margin-bottom:6px;">{lbl}</div>'
                            f'<div style="color:{clr};font-size:1.5rem;font-weight:800;">{v:.1f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── Tab 2 : Graphiques ───────────────────────────────────────────────────
    with t2:
        if metrics_df is None:
            st.info("Lancez `python scripts/train_models.py` pour générer les métriques.")
        else:
            mdf = metrics_df.copy()
            fig_bar = go.Figure()
            for i, (_, row) in enumerate(mdf.iterrows()):
                vals = [float(row.get(k, 0)) for k in METRIC_KEYS]
                fig_bar.add_trace(go.Bar(
                    name=row["model_name"],
                    x=METRIC_LBLS, y=vals,
                    marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                    text=[f"{v*100:.1f}%" for v in vals],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                **_fig_layout(title="Comparaison des métriques"),
                barmode="group",
            )
            fig_bar.update_yaxes(range=[0, 1.12], tickformat=".0%")
            st.plotly_chart(fig_bar, use_container_width=True)

            RADAR_FILLS = ["rgba(0,48,135,0.18)", "rgba(255,107,0,0.18)", "rgba(16,185,129,0.18)"]
            cats    = METRIC_LBLS + [METRIC_LBLS[0]]
            fig_rad = go.Figure()
            for i, (_, row) in enumerate(mdf.iterrows()):
                vals = [float(row.get(k, 0)) for k in METRIC_KEYS]
                fig_rad.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=cats, fill="toself",
                    name=row["model_name"],
                    line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2),
                    fillcolor=RADAR_FILLS[i % len(RADAR_FILLS)],
                ))
            fig_rad.update_layout(
                paper_bgcolor=WHITE,
                font=dict(color=DARK, size=12),
                margin=dict(t=60, b=40, l=40, r=40),
                polar=dict(
                    bgcolor=LIGHT,
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="#E5E7EB", tickformat=".0%",
                                    tickfont=dict(size=9, color=GREY)),
                    angularaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=11)),
                ),
                legend=dict(bgcolor=WHITE, bordercolor="#E5E7EB", borderwidth=1),
                title=dict(text="Radar des performances", font=dict(color=NAVY, size=15)),
            )
            st.plotly_chart(fig_rad, use_container_width=True)

    # ── Tab 3 : Feature Importance ───────────────────────────────────────────
    with t3:
        if "rf" not in models:
            st.warning("Modèle Random Forest non chargé.")
        else:
            rf  = models["rf"]
            available_feats = [c for c in FEATURE_COLS if c in df.columns]
            try:
                importances = pd.Series(
                    rf.feature_importances_[:len(available_feats)],
                    index=available_feats,
                ).sort_values()
                labels = [FEATURE_LABELS.get(f, f) for f in importances.index]
                q70    = importances.quantile(0.7)
                q40    = importances.quantile(0.4)
                colors_imp = [
                    RED if v >= q70 else YELLOW if v >= q40 else "#74b9ff"
                    for v in importances.values
                ]

                fig_imp = go.Figure(go.Bar(
                    x=importances.values, y=labels, orientation="h",
                    marker_color=colors_imp,
                    text=[f"{v*100:.1f}%" for v in importances.values],
                    textposition="outside",
                ))
                fig_imp.update_layout(
                    **_fig_layout(title="Feature Importance — Random Forest"),
                    height=420,
                )
                fig_imp.update_xaxes(title="Importance", tickformat=".0%")
                fig_imp.update_yaxes(gridcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown(f"""
                <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:8px;">
                    <span style="color:{RED}; font-weight:700;">■ Fort impact</span>
                    <span style="color:{YELLOW}; font-weight:700;">■ Impact modéré</span>
                    <span style="color:#74b9ff; font-weight:700;">■ Faible impact</span>
                </div>""", unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Erreur feature importance : {exc}")

    # ── Tab 4 : Explications ─────────────────────────────────────────────────
    with t4:
        rf_acc  = float(metrics_df[metrics_df["model_key"] == "random_forest"]["accuracy"].iloc[0]) * 100 if metrics_df is not None else 0
        xgb_acc = float(metrics_df[metrics_df["model_key"] == "xgboost"]["accuracy"].iloc[0]) * 100 if metrics_df is not None else 0
        km_acc  = float(metrics_df[metrics_df["model_key"] == "kmeans"]["accuracy"].iloc[0]) * 100 if metrics_df is not None else 0

        for emoji, name, acc, color, body in [
            ("🌲", "Random Forest", rf_acc, NAVY,
             "Imaginez <strong>100 experts indépendants</strong> (les arbres) qui analysent chacun "
             "les conditions de l'accident et votent. La décision finale est celle de la majorité.<br><br>"
             "<strong>✅ Avantages :</strong> robuste, résistant au surapprentissage, feature importance lisible.<br>"
             "<strong>⚠️ Limite :</strong> moins réactif sur des données très déséquilibrées."),
            ("⚡", "XGBoost", xgb_acc, ORANGE,
             "XGBoost apprend de ses erreurs à chaque itération. Chaque nouvel arbre corrige "
             "les cas mal classés par le précédent — c'est le <strong>favori des compétitions ML</strong>.<br><br>"
             "<strong>✅ Avantages :</strong> haute performance, gère bien les données déséquilibrées.<br>"
             "<strong>⚠️ Limite :</strong> boîte noire, nécessite plus de réglage."),
            ("🔵", "KMeans (non supervisé)", km_acc, GREEN,
             "KMeans regroupe automatiquement les accidents similaires <strong>SANS connaître leur gravité</strong>. "
             "Il détecte 3 profils naturels d'accidents sur autoroute.<br><br>"
             "<strong>✅ Avantages :</strong> non supervisé, découvre des patterns cachés.<br>"
             "<strong>⚠️ Limite :</strong> ne prédit pas directement, moins précis en classification."),
        ]:
            st.markdown(f"""
            <div style="background:{WHITE}; border:1px solid #E5E7EB; border-left:5px solid {color};
                        border-radius:12px; padding:22px 26px; margin-bottom:16px;
                        box-shadow:0 2px 8px rgba(0,48,135,0.06);">
                <div style="font-size:1.6rem; margin-bottom:8px;">{emoji}</div>
                <div style="color:{NAVY}; font-size:1.1rem; font-weight:800; margin-bottom:10px;">
                    {name}
                    <span style="background:{LIGHT}; color:{color}; border-radius:8px;
                                 padding:2px 12px; font-size:0.78rem; margin-left:8px; font-weight:700;">
                        Accuracy : {acc:.1f}%
                    </span>
                </div>
                <p style="color:{GREY}; line-height:1.75; margin:0;">{body}</p>
            </div>
            """, unsafe_allow_html=True)

        # Clusters KMeans
        if "kmeans" in models:
            st.markdown(f"<h4 style='color:{ORANGE};'>Les 3 profils d'accidents identifiés par KMeans</h4>",
                        unsafe_allow_html=True)
            try:
                km = models["kmeans"]
                available = [c for c in FEATURE_COLS if c in df.columns]
                X_km = df[available].fillna(0).astype(float)
                labels_km = km.predict(X_km)
                df_c = df.copy()
                df_c["cluster"] = labels_km

                cl1, cl2, cl3 = st.columns(3)
                for col_w, (cid, cname) in zip([cl1, cl2, cl3], CLUSTER_NAMES.items()):
                    sub = df_c[df_c["cluster"] == cid]
                    if sub.empty:
                        continue
                    pct_grave = sub["target"].mean() * 100
                    with col_w:
                        st.markdown(f"""
                        <div style="background:{WHITE}; border:2px solid {CLUSTER_COLORS[cid]};
                                    border-radius:14px; padding:20px;
                                    box-shadow:0 3px 10px rgba(0,48,135,0.08);">
                            <div style="color:{CLUSTER_COLORS[cid]}; font-weight:800;
                                        font-size:1rem; margin-bottom:12px;">{cname}</div>
                            <div style="color:{GREY}; font-size:0.85rem; line-height:1.8;">
                                🚧 <strong>{len(sub):,}</strong> usagers<br>
                                ⚠️ Taux gravité : <strong>{pct_grave:.1f}%</strong><br>
                                ⚡ Vit. moy. : <strong>{sub['vma'].mean():.0f} km/h</strong><br>
                                🌧️ Météo dom. : <strong>{ATM_LABELS.get(int(sub['atm'].mode().iloc[0]), '?') if not sub['atm'].mode().empty else '?'}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Erreur clusters : {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

PAGES = {
    "🛣️ Contexte Vinci":        ("contexte",    page_contexte,    False),
    "📊 Dashboard accidents":   ("dashboard",   page_dashboard,   False),
    "🚨 Prédiction gravité":    ("prediction",  page_prediction,  True),
    "⚖️ Comparaison modèles":   ("comparaison", page_comparaison, True),
}


def build_app() -> None:
    st.set_page_config(
        page_title="Vinci Autoroutes — IA Accidents",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    df     = load_data()
    models = load_models()

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:28px 0 18px;">
            <div style="font-size:3rem;">🛣️</div>
            <div style="color:{WHITE}; font-size:1.05rem; font-weight:900; margin-top:8px;"> Vinci Autoroutes </div>
            <div style="color:rgba(255,255,255,0.6); font-size:0.75rem; margin-top:4px;">
                Prédiction accidents graves
            </div>
        </div>
        <hr style="border-color:{ORANGE}; margin:0 0 14px;">
        """, unsafe_allow_html=True)

        page_name = st.radio("Navigation", list(PAGES.keys()),
                             label_visibility="collapsed")

        st.divider()

        n_acc = df["Num_Acc"].nunique() if "Num_Acc" in df.columns else len(df)
        pct_g = df["target"].mean() * 100
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:14px; margin-top:4px;">
            <p style="color:{ORANGE}; font-weight:800; margin:0 0 8px; font-size:0.82rem;">📊 Dataset BAAC</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">🚗 {len(df):,} usagers</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">🚧 {n_acc:,} accidents</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">📅 2020 – 2024</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">🏎️ Autoroutes (catr=1)</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0; font-weight:700;">
                ⚠️ {pct_g:.1f}% accidents graves</p>
            <p style="color:{WHITE}; font-size:0.78rem; margin:3px 0;">🤖 {len(models)} modèles chargés</p>
        </div>
        """, unsafe_allow_html=True)

    _, fn, needs_models = PAGES[page_name]
    if needs_models:
        fn(df, models)
    else:
        fn(df)


if __name__ == "__main__":
    build_app()
