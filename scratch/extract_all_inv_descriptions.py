import pandas as pd

df = pd.read_csv("datasets/crudo/inventario.csv", sep=";", dtype=str, skiprows=1)
descs = sorted(list(set(df['Descripcion'].dropna().str.strip().str.upper())))
descs = [d for d in descs if d != "" and d != "NAN"]

print(f"Total descripciones únicas no vacías en inventario.csv: {len(descs)}")
for i, d in enumerate(descs):
    print(f"{i+1:3d}: {d}")
