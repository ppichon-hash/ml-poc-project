"""Generate a synthetic Pokemon TCG dataset and train the three models."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

SETS = [
    ("Base Set", 1999), ("Jungle", 1999), ("Fossil", 1999),
    ("Team Rocket", 2000), ("Neo Genesis", 2000), ("Neo Discovery", 2001),
    ("Expedition", 2002), ("Ruby & Sapphire", 2003), ("FireRed LeafGreen", 2004),
    ("Delta Species", 2005), ("Diamond & Pearl", 2007), ("Platinum", 2009),
    ("HeartGold SoulSilver", 2010), ("Black & White", 2011), ("Boundaries Crossed", 2012),
    ("XY", 2014), ("Primal Clash", 2015), ("Evolutions", 2016),
    ("Sun & Moon", 2017), ("Burning Shadows", 2017), ("Ultra Prism", 2018),
    ("Team Up", 2019), ("Hidden Fates", 2019), ("Sword & Shield", 2020),
    ("Vivid Voltage", 2020), ("Chilling Reign", 2021), ("Evolving Skies", 2021),
    ("Astral Radiance", 2022), ("Lost Origin", 2022), ("Paldea Evolved", 2023),
    ("Obsidian Flames", 2023), ("Paradox Rift", 2023), ("Twilight Masquerade", 2024),
    ("Stellar Crown", 2024),
]

RARITIES = ["Common", "Uncommon", "Rare", "Holo Rare", "Ultra Rare", "Secret Rare"]
RARITY_WEIGHTS = [0.35, 0.25, 0.18, 0.12, 0.07, 0.03]
RARITY_PRICE_BASE = {"Common": 0.5, "Uncommon": 1.0, "Rare": 3.0, "Holo Rare": 8.0, "Ultra Rare": 25.0, "Secret Rare": 80.0}
RARITY_ENCODED = {"Common": 1, "Uncommon": 2, "Rare": 3, "Holo Rare": 4, "Ultra Rare": 5, "Secret Rare": 6}

TYPES = ["Fire", "Water", "Grass", "Electric", "Psychic", "Fighting", "Dark", "Metal", "Dragon", "Normal", "Fairy"]
TYPE_WEIGHTS = [0.12, 0.12, 0.11, 0.11, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06, 0.05]

POKEMON_NAMES = [
    "Pikachu", "Charizard", "Blastoise", "Venusaur", "Mewtwo", "Mew", "Gengar",
    "Alakazam", "Machamp", "Golem", "Snorlax", "Dragonite", "Lapras", "Vaporeon",
    "Jolteon", "Flareon", "Eevee", "Gyarados", "Arcanine", "Ninetales",
    "Raichu", "Lucario", "Garchomp", "Salamence", "Metagross", "Tyranitar",
    "Scizor", "Espeon", "Umbreon", "Lugia", "Ho-Oh", "Celebi", "Entei",
    "Suicune", "Raikou", "Kyogre", "Groudon", "Rayquaza", "Deoxys", "Jirachi",
    "Dialga", "Palkia", "Giratina", "Arceus", "Darkrai", "Shaymin",
    "Zekrom", "Reshiram", "Kyurem", "Xerneas", "Yveltal", "Zygarde",
    "Solgaleo", "Lunala", "Marshadow", "Zeraora", "Necrozma",
    "Zacian", "Zamazenta", "Eternatus", "Calyrex", "Regieleki",
    "Miraidon", "Koraidon", "Palafin", "Gholdengo", "Iron Bundle",
]

set_indices = np.random.choice(len(SETS), N)
set_names = [SETS[i][0] for i in set_indices]
set_years = np.array([SETS[i][1] for i in set_indices])

rarities = np.random.choice(RARITIES, N, p=RARITY_WEIGHTS)
types_ = np.random.choice(TYPES, N, p=TYPE_WEIGHTS)
names = np.random.choice(POKEMON_NAMES, N)
hp = np.clip(np.random.normal(80, 40, N).astype(int), 30, 340)
is_holo = (rarities == "Holo Rare") | (rarities == "Ultra Rare") | (rarities == "Secret Rare")
is_holo = is_holo.astype(int)
is_reverse = np.random.choice([0, 1], N, p=[0.7, 0.3])
nb_attacks = np.random.choice([1, 2, 3], N, p=[0.2, 0.6, 0.2])
has_evolution = np.random.choice([0, 1], N, p=[0.4, 0.6])

set_age = 2024 - set_years
rarity_encoded = np.array([RARITY_ENCODED[r] for r in rarities])
hp_normalized = hp / 340.0

base_price = np.array([RARITY_PRICE_BASE[r] for r in rarities])
vintage_mult = 1 + set_age * 0.05
holo_mult = 1 + is_holo * 0.8
popularity_noise = np.random.lognormal(0, 0.6, N)
pikachu_mult = np.where(names == "Pikachu", 2.5, 1.0)
charizard_mult = np.where(names == "Charizard", 4.0, 1.0)
market_price = np.clip(
    base_price * vintage_mult * holo_mult * popularity_noise * pikachu_mult * charizard_mult,
    0.05, 2000.0
)

spread = market_price * np.random.uniform(0.05, 0.25, N)
low_price = np.clip(market_price - spread, 0.01, None)
high_price = market_price + spread

trend_noise = np.random.normal(0, 0.15, N)
trend_mult = 1 + trend_noise + (rarity_encoded > 3) * 0.1 + (set_age > 10) * 0.05
trend_price = np.clip(market_price * trend_mult, 0.01, None)

price_range = high_price - low_price

set_avg_prices = {}
temp_df = pd.DataFrame({"set_name": set_names, "trend_price": trend_price})
for sn in temp_df["set_name"].unique():
    set_avg_prices[sn] = temp_df[temp_df["set_name"] == sn]["trend_price"].mean()
set_avg = np.array([set_avg_prices[s] for s in set_names])
target = (trend_price > set_avg).astype(int)

df = pd.DataFrame({
    "name": names,
    "set_name": set_names,
    "rarity": rarities,
    "hp": hp,
    "type": types_,
    "year": set_years,
    "market_price": np.round(market_price, 2),
    "low_price": np.round(low_price, 2),
    "high_price": np.round(high_price, 2),
    "trend_price": np.round(trend_price, 2),
    "is_holo": is_holo,
    "is_reverse": is_reverse,
    "nb_attacks": nb_attacks,
    "has_evolution": has_evolution,
    "rarity_encoded": rarity_encoded,
    "set_age": set_age,
    "price_range": np.round(price_range, 2),
    "hp_normalized": np.round(hp_normalized, 4),
    "target": target,
})

out = ROOT / "data" / "pokemon_cards.csv"
df.to_csv(out, index=False)
print(f"Dataset saved: {out} ({len(df)} rows)")
print(df["target"].value_counts().to_string())
print(df[["market_price", "trend_price", "low_price", "high_price"]].describe().to_string())
