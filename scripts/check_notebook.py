import json

with open("notebooks/exploration_pokemon.ipynb") as f:
    nb = json.load(f)

print("Cellules:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = c["source"]
    if isinstance(src, list):
        src = "".join(src)
    print(f"  {i+1:2}. {src[:60].strip()}")
