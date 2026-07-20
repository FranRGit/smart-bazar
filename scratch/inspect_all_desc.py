import pandas as pd
import os

DIR_LIMPIO = 'datasets/limpio/'
df_detalle = pd.read_csv(os.path.join(DIR_LIMPIO, 'detalle_ventas.csv'))
descripciones = df_detalle['Descripcion'].dropna().unique()
with open("scratch/all_desc.txt", "w", encoding="utf-8") as f:
    for d in sorted(descripciones):
        f.write(f"{d}\n")
