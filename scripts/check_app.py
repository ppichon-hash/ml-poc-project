"""Validation script for app.py — no emoji in prints for cp1252 compat."""
import sys
sys.path.insert(0, "src")
import importlib.util
import unittest.mock as mock

sys.modules["streamlit"] = mock.MagicMock()

spec = importlib.util.spec_from_file_location("config_mod", "src/config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
sys.modules["config"] = config

spec2 = importlib.util.spec_from_file_location("app", "src/app.py")
m = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(m)

import pandas as pd
df = pd.read_csv("data/pokemon_cards.csv")

pages = [k for k in dir(m) if k.startswith("page_")]
print("Pages:", pages)
print("NAV_PAGES count:", len(m.NAV_PAGES))
print("NAV_PAGES labels:", list(m.NAV_PAGES.keys()))
print("build_app callable:", callable(m.build_app))
print("inject_css callable:", callable(m.inject_css))

h = m._simulate_price_history(10.0, 12.0, 8.0, 14.0)
print("History OK shape:", h.shape)

rows = m._marketplace_rows("Pikachu", 15.0)
print("Marketplace sites sorted:", [r["site"] for r in rows])
print("Marketplace prices:", [r["prix_estime"] for r in rows])

print("CSS constants - RED:", m.RED, "YELLOW:", m.YELLOW, "DARK:", m.DARK)
print("FEATURE_COLS count:", len(m.FEATURE_COLS))
print("RARITY_ORDER:", m.RARITY_ORDER)
print("ALL CHECKS PASSED")
