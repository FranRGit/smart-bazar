import pandas as pd
import os

DIR_LIMPIO = 'datasets/limpio/'
df_detalle = pd.read_csv(os.path.join(DIR_LIMPIO, 'detalle_ventas.csv'))
descripciones = df_detalle['Descripcion'].value_counts()
print(f"Total descripciones únicas: {len(descripciones)}")
for desc, count in descripciones.head(60).items():
    print(f"{count:4d}: {desc}")
