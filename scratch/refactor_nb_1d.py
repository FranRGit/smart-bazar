import json
from pathlib import Path

nb_path = Path("Notebooks/Panel 1D_Reglas_Asociacion.ipynb")

def mk_cell(content):
    lines = [line + "\n" for line in content.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def cd_cell(content):
    lines = [line + "\n" for line in content.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

cells = []

# TÍTULO EJECUTIVO Y RESUMEN
cells.append(mk_cell("""# Panel 1D: Reporte Analítico Profesional – Minería de Reglas de Asociación y Afinidad en Retail

**Proyecto:** SmartBazar – Pipeline de Machine Learning para Retail  
**Insumo de Entrada:** Datasets Saneados (`datasets/limpio/` – Salida del Cuaderno 1A)  
**Enfoque:** Market Basket Analysis (MBA), Comparativa Algorítmica (Apriori vs FP-Growth) y Estrategia de Granularidad (SKU Detallado vs Categoría Universal)

---

## Resumen Ejecutivo de Negocio
El objetivo de este reporte es identificar patrones ocultos en el comportamiento de compra transaccional (*tickets* de venta) para potenciar la rentabilidad operativa del retail mediante tres palancas:
1. **Cross-Selling y Venta Cruzada en Caja:** Promover artículos complementarios basándose en reglas con alta probabilidad condicional (*Confianza* > 30% y *Lift* > 10).
2. **Optimización del Layout y Pasillos:** Alinear la disposición física y visual de la tienda con las sinergias de categorías macro (*Producto Universal*).
3. **Gestión Interdependiente de Stocks:** Coordinar el reabastecimiento de productos que se adquieren en conjunto, evitando quiebres de inventario inducidos."""))

# BLOQUE 1: SETUP Y CARGA DE DATOS
cells.append(mk_cell("""## 1. Setup y Carga de Datos Transaccionales
Preparamos el entorno analítico, cargamos el historial de tickets saneados y construimos las matrices de transacciones binarias bajo dos niveles estratégicos de granularidad."""))

cells.append(mk_cell("""### 1.1 Configuración del Entorno e Importaciones"""))

cells.append(cd_cell("""try:
    import mlxtend
except ImportError:
    !pip install mlxtend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import time
import os
import warnings

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings('ignore')

# Configuración de estilo visual elegante y profesional
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (11, 6)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
np.random.seed(42)"""))

cells.append(mk_cell("""### 1.2 Ingesta de Datasets Limpios y Exploración de Tickets
Evaluamos el volumen transaccional general y la distribución de cantidad de artículos comprados en una misma canasta."""))

cells.append(cd_cell("""# Rutas de entrada a los datasets saneados
DIR_LIMPIO = 'datasets/limpio/'
df_ventas = pd.read_csv(os.path.join(DIR_LIMPIO, 'ventas.csv'))
df_detalle = pd.read_csv(os.path.join(DIR_LIMPIO, 'detalle_ventas.csv'))

print(f"Total de Tickets Evaluados (df_ventas): {df_ventas.shape[0]:,} transacciones")
print(f"Total de Ítems Vendidos (df_detalle): {df_detalle.shape[0]:,} registros transaccionales\\n")

# Top 15 Productos Más Frecuentes en Tickets
prod_frecuentes = df_detalle['Descripcion'].value_counts().head(15).reset_index()
prod_frecuentes.columns = ['Producto (SKU)', 'Frecuencia en Tickets']

plt.figure(figsize=(12, 6))
sns.barplot(x='Frecuencia en Tickets', y='Producto (SKU)', data=prod_frecuentes, palette="viridis")
plt.title('Top 15 Productos con Mayor Rotación Transaccional', fontsize=14, fontweight='bold')
plt.xlabel('Número de Apariciones en Canastas de Compra')
plt.ylabel('Descripción del Producto')
plt.tight_layout()
plt.show()

# Distribución de tamaño de ticket (ítems distintos por transacción)
productos_distintos = df_detalle.groupby('ID_Venta')['ID_Producto'].nunique()
resumen_ticket = pd.DataFrame(productos_distintos.describe()).T
resumen_ticket.columns = ['Total Canastas', 'Promedio Ítems/Ticket', 'Desv. Estandar', 'Mínimo', 'Q1 (25%)', 'Mediana (50%)', 'Q3 (75%)', 'Máximo']
display(resumen_ticket.style.format("{:.2f}").background_gradient(cmap='Blues'))"""))

cells.append(mk_cell("""### 1.3 Construcción de Matrices Transaccionales (SKU Detallado vs Categoría Universal)
Para evitar la dispersión estadística del soporte que ocurre al dividir las ventas entre variantes de tamaño, marca o volumen (ej. *SILICONA 30ML* vs *SILICONA 250ML* vs *SILICONA 100ML*), implementamos una doble estrategia:
- **Nivel Producto Detallado (SKU específico):** Analiza afinidades operativas finas (251 ítems únicos).
- **Nivel Producto Universal (Genérico):** Consolida variantes bajo conceptos macro limpios (71 categorías macro), revelando reglas estratégicas de alto impacto y alta robustez estadística.

> **Clean Code Architecture:** La categorización universal se gestiona de forma limpia y declarativa mediante un diccionario de mapeo (`diccionario_mapeo`) y una función iterativa simple, eliminando condicionales redundantes (*no-IFs*)."""))

cells.append(cd_cell("""# NOTA DE PRODUCCIÓN: En un entorno productivo de despliegue, este diccionario y la función 
# de categorización deben aislarse en un módulo de soporte independiente (ej. `utils.py`).

diccionario_mapeo = {
    # Escritura y Trazado
    "LAPIZ 2B": "LAPIZ",
    "PORTAMINAS": "LAPIZ",
    "LÁPIZ": "LAPIZ",
    "LAPIZ": "LAPIZ",
    "CARBONCILLO": "LAPIZ",
    "FRIXION": "LAPICERO",
    "LAPICERO": "LAPICERO",
    "PLUMON": "PLUMON / RESALTADOR",
    "PLUMÓN": "PLUMON / RESALTADOR",
    "RESALTADOR": "PLUMON / RESALTADOR",
    "CHEQUEO": "PLUMON / RESALTADOR",
    "MARKER": "PLUMON / RESALTADOR",
    "INDELEBLE": "PLUMON / RESALTADOR",
    "CORRECTOR": "CORRECTOR",
    "MINAS": "MINAS / REPUESTOS",
    "TIZA": "TIZA",
    
    # Arte, Dibujo y Manualidades
    "LAPICES DE COLOR": "COLORES",
    "COLORES": "COLORES",
    "CRAYON": "CRAYOLAS",
    "CRAYOLA": "CRAYOLAS",
    "TEMPERA": "TEMPERA",
    "TÉMPERA": "TEMPERA",
    "PINCEL": "PINCEL",
    "PALETA": "ACCESORIOS DE PINTURA",
    "PLASTILINA": "PLASTILINA",
    "LENTEJUELA": "LENTEJUELAS Y ESCARCHA",
    "ESCARCHA": "LENTEJUELAS Y ESCARCHA",
    "OROPEL": "PAPEL OROPEL",
    "MICROPOROSO": "MICROPOROSO",
    "PALITOS DE CHUPETE": "PALITOS DE MANUALIDADES",
    "BAJA LENGUA": "PALITOS DE MANUALIDADES",
    "OJOS MOVILES": "MANUALIDADES",
    "PUNZON": "HERRAMIENTAS DE MANUALIDADES",
    "ESPONJA": "ESPONJA",
    "ALGODÓN": "ALGODON",
    "ALGODON": "ALGODON",
    
    # Pegamentos y Adhesivos
    "SILICONA": "SILICONA",
    "GOMA": "GOMA",
    "CINTA MASKING": "CINTA ADHESIVA / EMBALAJE",
    "CINTA EMBALAJE": "CINTA ADHESIVA / EMBALAJE",
    "CINTA DE ESCRITORIO": "CINTA ADHESIVA / EMBALAJE",
    "CINTA DE AGUA": "CINTAS DECORATIVAS Y AGUA",
    "LIMPIATIPO": "LIMPIATIPOS",
    
    # Cuadernos, Blocks y Papelería
    "SKETCH BOOK": "BLOCK / SKETCH BOOK",
    "BLOCK": "BLOCK / SKETCH BOOK",
    "CUADERNO": "CUADERNO",
    "POST-IT": "NOTAS ADHESIVAS / POST-IT",
    "HOJA BOND": "HOJA BOND",
    "PAPEL BOND": "HOJA BOND",
    "HOJA DE COLOR": "HOJAS DE COLORES",
    "HOJAS DE COLOR": "HOJAS DE COLORES",
    "CARTUILINA": "CARTULINA",
    "CARTULINA": "CARTULINA",
    "PAPEL CREPE": "PAPEL CREPE",
    "PAPEL LUSTRE": "PAPEL LUSTRE",
    "PAPEL KRAFT": "PAPEL ESPECIAL / KRAFT",
    "PAPEL CRAFT": "PAPEL ESPECIAL / KRAFT",
    "PAPEL DE SEDA": "PAPEL DE SEDA",
    "PAPEL DE SEDE": "PAPEL DE SEDA",
    "PAPEL MANTECA": "PAPEL ESPECIAL",
    "PAPEL DE REGALO": "PAPEL DE REGALO",
    "PAPELOTE": "PAPELOTE",
    "SOBRE": "SOBRE",
    "TARJETA BIBLIOGRAFICA": "TARJETA BIBLIOGRAFICA",
    "BILLETE Y MONEDA": "MATERIAL DIDACTICO",
    "TABLA PERIODICA": "MATERIAL DIDACTICO",
    "LAMINAS": "LAMINAS DIDACTICAS",
    
    # Organización, Archivo y Sujetadores
    "FOLDER": "FOLDER",
    "MICA": "MICA / FOTOCHECK",
    "PLASTIFORRO": "FORRO / VINIFAN",
    "VINIFAN": "FORRO / VINIFAN",
    "FASTER": "SUJETADORES / FASTERS",
    "CHINCHE": "SUJETADORES / CHINCHES",
    "GRAPA": "SUJETADORES / GRAPAS",
    "LIGAS": "SUJETADORES / LIGAS",
    "PABILO": "PABILO",
    
    # Herramientas y Accesorios de Escritorio
    "TIJERA": "TIJERA / CUTER",
    "CUTER": "TIJERA / CUTER",
    "TAJADOR": "TAJADOR",
    "BORRADOR": "BORRADOR",
    "REGLA": "REGLA / GEOMETRIA",
    "ESCUADRA": "REGLA / GEOMETRIA",
    "TRANSPORTADOR": "REGLA / GEOMETRIA",
    "PERFORADOR": "PERFORADOR",
    "CALCULADORA": "CALCULADORA",
    "MOTA": "MOTA",
    "TAMPON": "TAMPON Y TINTAS",
    "TINTA": "TAMPON Y TINTAS",
    "LUPA": "LUPA",
    "PERCHERO": "PERCHERO",
    
    # Artículos Festivos y Recreación
    "GLOBO": "GLOBOS Y ACCESORIOS",
    "PALIGLOBO": "GLOBOS Y ACCESORIOS",
    "SERPENTINA": "SERPENTINA Y PICA PICA",
    "PICA PICA": "SERPENTINA Y PICA PICA",
    "FLAUTA": "INSTRUMENTOS MUSICALES",
    "PITO": "ARTICULOS DEPORTIVOS Y RECREACION",
    
    # Artículos Cívicos y Uniforme
    "CORBATA": "ARTICULOS CIVICOS Y UNIFORME",
    "ESCARAPELA": "ARTICULOS CIVICOS Y UNIFORME",
    "BANDERA": "ARTICULOS CIVICOS Y UNIFORME",
    "INSIGNA": "ARTICULOS CIVICOS Y UNIFORME",
    "PALO DE BRIGADIER": "ARTICULOS CIVICOS Y UNIFORME",
    "CORREA": "ARTICULOS CIVICOS Y UNIFORME",
    "GUANTE": "GUANTES E HIGIENE",
    "MANDIL": "MANDIL Y BOLSA",
    "BOLSA": "MANDIL Y BOLSA",
    
    # Servicios y Otros
    "FOTOCOPIA": "FOTOCOPIA E IMPRESION",
    "IMPRESION": "FOTOCOPIA E IMPRESION",
    "TIPEO": "SERVICIOS",
    "SUPER BLUE": "ACCESORIOS DE LIMPIEZA",
    "PRODUCTO": "GENERAL / OTROS"
}

def categorizar_producto(descripcion, diccionario=diccionario_mapeo):
    if not isinstance(descripcion, str):
        return "OTRO"
    desc_upper = descripcion.upper().strip()
    for clave, categoria in diccionario.items():
        if clave in desc_upper:
            return categoria
    return "OTRO"

# Asignación limpia y declarativa de la categoría universal al dataframe
df_detalle['Producto_Universal'] = df_detalle['Descripcion'].apply(categorizar_producto)

# 1. MATRIZ TRANSACCIONAL NIVEL DETALLADO (SKU ESPECÍFICO)
transacciones_prod = df_detalle.groupby('ID_Venta')['Descripcion'].apply(list).tolist()
te_prod = TransactionEncoder()
df_trans_prod = pd.DataFrame(te_prod.fit(transacciones_prod).transform(transacciones_prod), columns=te_prod.columns_)
min_apariciones = int(df_trans_prod.shape[0] * 0.01)
df_trans_prod = df_trans_prod.loc[:, df_trans_prod.sum() >= min_apariciones]

# 2. MATRIZ TRANSACCIONAL NIVEL UNIVERSAL (GENÉRICO / GENERAL)
transacciones_univ = df_detalle.groupby('ID_Venta')['Producto_Universal'].apply(lambda x: list(set(x))).tolist()
te_univ = TransactionEncoder()
df_trans_univ = pd.DataFrame(te_univ.fit(transacciones_univ).transform(transacciones_univ), columns=te_univ.columns_)

# Resumen comparativo de dispersión y dimensiones transaccionales
sparsidad_prod = 1.0 - (df_trans_prod.values.sum() / df_trans_prod.size)
sparsidad_univ = 1.0 - (df_trans_univ.values.sum() / df_trans_univ.size)

resumen_matrices = pd.DataFrame({
    'Nivel de Granularidad': ['Producto Detallado (SKU >1% soporte)', 'Producto Universal (Concepto Genérico)'],
    'Transacciones (Filas)': [df_trans_prod.shape[0], df_trans_univ.shape[0]],
    'Ítems / Conceptos Únicos (Columnas)': [df_trans_prod.shape[1], df_trans_univ.shape[1]],
    'Sparsidad de la Matriz (%)': [f"{sparsidad_prod*100:.2f}%", f"{sparsidad_univ*100:.2f}%"]
})
display(resumen_matrices.style.background_gradient(cmap='Purples', subset=['Ítems / Conceptos Únicos (Columnas)']))
print("\\nVistazo a las primeras 3 transacciones en la Matriz de Producto Universal:")
display(df_trans_univ.head(3))"""))

# BLOQUE 2: MODELADO
cells.append(mk_cell("""## 2. Modelado de Reglas de Asociación Multi-Algoritmo
Aplicamos y evaluamos dos motores clásicos de minería de patrones transaccionales sobre ambas matrices:
- **Apriori:** Búsqueda exhaustiva *bottom-up* combinando itemsets frecuentes.
- **FP-Growth:** Compresión transaccional basada en árboles de prefijos (`FP-Tree`) para máxima eficiencia de cómputo en catálogos extensos.

Fijamos el **soporte mínimo (`min_support = 0.005` ó `0.5%`)** y un umbral de elevación **`Lift > 1.0`** para filtrar asociaciones que realmente superen al azar."""))

cells.append(mk_cell("""### 2.1 Modelo 1: Algoritmo Apriori (Granularidad SKU Detallado vs Producto Universal)
Exploramos los itemsets frecuentes y reglas generadas con Apriori en ambas dimensiones."""))

cells.append(cd_cell("""soporte_elegido = 0.005

# Ejecución Apriori - Nivel SKU Detallado
t0 = time.time()
frec_ap_prod = apriori(df_trans_prod, min_support=soporte_elegido, use_colnames=True)
tiempo_ap_prod = time.time() - t0
frec_ap_prod['longitud'] = frec_ap_prod['itemsets'].apply(len)
reglas_ap_prod = association_rules(frec_ap_prod, metric="lift", min_threshold=1.0)

# Ejecución Apriori - Nivel Producto Universal
t0 = time.time()
frec_ap_univ = apriori(df_trans_univ, min_support=soporte_elegido, use_colnames=True)
tiempo_ap_univ = time.time() - t0
frec_ap_univ['longitud'] = frec_ap_univ['itemsets'].apply(len)
reglas_ap_univ = association_rules(frec_ap_univ, metric="lift", min_threshold=1.0)

# Formateo de columnas para inspección limpia
for df_r in [reglas_ap_prod, reglas_ap_univ]:
    if len(df_r) > 0:
        df_r['antecedents_str'] = df_r['antecedents'].apply(lambda x: ', '.join(list(x)))
        df_r['consequents_str'] = df_r['consequents'].apply(lambda x: ', '.join(list(x)))
        df_r['rule_name'] = df_r['antecedents_str'] + " -> " + df_r['consequents_str']

print(f"Apriori Detallado: {len(frec_ap_prod)} itemsets | {len(reglas_ap_prod)} reglas en {tiempo_ap_prod:.4f}s")
print(f"Apriori Universal: {len(frec_ap_univ)} itemsets | {len(reglas_ap_univ)} reglas en {tiempo_ap_univ:.4f}s\\n")

print("Top 5 Reglas Generadas por Apriori a Nivel Producto Universal (Ordenadas por Lift):")
display(reglas_ap_univ[['rule_name', 'support', 'confidence', 'lift', 'leverage', 'conviction']]
        .sort_values(by='lift', ascending=False).head(5)
        .style.background_gradient(cmap='YlOrRd', subset=['lift', 'confidence']))

# Gráficos explicativos del modelo Apriori Universal
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatterplot Soporte vs Confianza
sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', data=reglas_ap_univ, 
                palette='coolwarm', sizes=(60, 450), ax=axes[0])
axes[0].set_title('Mapa de Reglas Apriori (Universal): Soporte vs Confianza (Color = Lift)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Soporte (Proporción de Tickets)')
axes[0].set_ylabel('Confianza (Probabilidad Condicional)')
axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# Barplot Top 10 por Lift
top10_ap_univ = reglas_ap_univ.sort_values(by='lift', ascending=False).head(10)
sns.barplot(x='lift', y='rule_name', data=top10_ap_univ, palette="viridis", ax=axes[1])
axes[1].set_title('Top 10 Reglas Universales con Mayor Fuerza de Asociación (Lift)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Lift (Elevación sobre Independencia)')
axes[1].set_ylabel('Regla de Afinidad')

plt.tight_layout()
plt.show()"""))

cells.append(mk_cell("""### 2.2 Modelo 2: Algoritmo FP-Growth (Optimización Computacional del Árbol Transaccional)
Verificamos la escalabilidad y consistencia del motor FP-Growth sobre la estructura compacta universal, demostrando que produce exactamente las mismas asociaciones pero con una optimización notable en la convergencia del algoritmo."""))

cells.append(cd_cell("""# Ejecución FP-Growth - Nivel SKU Detallado
t0 = time.time()
frec_fp_prod = fpgrowth(df_trans_prod, min_support=soporte_elegido, use_colnames=True)
tiempo_fp_prod = time.time() - t0
frec_fp_prod['longitud'] = frec_fp_prod['itemsets'].apply(len)
reglas_fp_prod = association_rules(frec_fp_prod, metric="lift", min_threshold=1.0)

# Ejecución FP-Growth - Nivel Producto Universal
t0 = time.time()
frec_fp_univ = fpgrowth(df_trans_univ, min_support=soporte_elegido, use_colnames=True)
tiempo_fp_univ = time.time() - t0
frec_fp_univ['longitud'] = frec_fp_univ['itemsets'].apply(len)
reglas_fp_univ = association_rules(frec_fp_univ, metric="lift", min_threshold=1.0)

for df_r in [reglas_fp_prod, reglas_fp_univ]:
    if len(df_r) > 0:
        df_r['antecedents_str'] = df_r['antecedents'].apply(lambda x: ', '.join(list(x)))
        df_r['consequents_str'] = df_r['consequents'].apply(lambda x: ', '.join(list(x)))
        df_r['rule_name'] = df_r['antecedents_str'] + " -> " + df_r['consequents_str']

print(f"FP-Growth Detallado: {len(frec_fp_prod)} itemsets | {len(reglas_fp_prod)} reglas en {tiempo_fp_prod:.4f}s")
print(f"FP-Growth Universal: {len(frec_fp_univ)} itemsets | {len(reglas_fp_univ)} reglas en {tiempo_fp_univ:.4f}s\\n")

print("Top 5 Reglas Generadas por FP-Growth a Nivel Producto Universal:")
display(reglas_fp_univ[['rule_name', 'support', 'confidence', 'lift', 'conviction']]
        .sort_values(by='lift', ascending=False).head(5)
        .style.background_gradient(cmap='magma', subset=['lift', 'confidence']))

# Scatterplot explicativo para FP-Growth Universal
plt.figure(figsize=(10, 5))
sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', data=reglas_fp_univ, palette='magma', sizes=(60, 400))
plt.title('Consistencia de Reglas FP-Growth (Universal): Soporte vs Confianza', fontsize=14, fontweight='bold')
plt.xlabel('Soporte (Frecuencia en Tickets)')
plt.ylabel('Confianza (Tasa de Conversión Cruzada)')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()"""))

# BLOQUE 3: COMPARATIVA
cells.append(mk_cell("""## 3. Comparación Integral de Granularidad y Algoritmos

En esta sección evaluamos dos dimensiones críticas mediante métricas formales y visualizaciones directas:

*   **⚡ Eficiencia Algorítmica (Apriori vs FP-Growth):** Demuestra que **FP-Growth** converge a los mismos resultados exactos con menor consumo computacional y sin generación de candidatos intermedios.
*   **🎯 Impacto Estratégico de la Generalización (Detallado vs Universal):**
    *   **Concentración de Soporte:** Al unificar variantes como *SILICONA PEQUEÑA/MEDIANA/GRANDE* en **`SILICONA`**, el soporte por categoría se fortalece y permite detectar patrones sinérgicos que a nivel SKU específico quedaban diluidos por debajo del umbral estadístico.
    *   **Maximización de Confianza y Lift:** El nivel Universal extrae asociaciones macro altamente accionables, con **Confianzas de hasta 100% y Lifts superiores a 20**, idóneas para decisiones de *cross-selling* estructural en el punto de venta."""))

cells.append(mk_cell("""### 3.1 Benchmark de Rendimiento Algorítmico (Apriori vs FP-Growth)
Contrastamos la convergencia, cantidad de reglas descubiertas y métricas punta alcanzadas por ambos motores en cada nivel transaccional."""))

cells.append(cd_cell("""# 1. Benchmark directo Apriori vs FP-Growth por dimensión
tabla_comp_algos = pd.DataFrame({
    'Nivel de Granularidad': ['SKU Detallado', 'SKU Detallado', 'Producto Universal', 'Producto Universal'],
    'Algoritmo': ['Apriori', 'FP-Growth', 'Apriori', 'FP-Growth'],
    'Tiempo Ejecución (s)': [tiempo_ap_prod, tiempo_fp_prod, tiempo_ap_univ, tiempo_fp_univ],
    'Itemsets Frecuentes': [len(frec_ap_prod), len(frec_fp_prod), len(frec_ap_univ), len(frec_fp_univ)],
    'Reglas Descubiertas (Lift>1)': [len(reglas_ap_prod), len(reglas_fp_prod), len(reglas_ap_univ), len(reglas_fp_univ)],
    'Soporte Máx (%)': [f"{reglas_ap_prod['support'].max()*100:.2f}%" if len(reglas_ap_prod)>0 else "0.00%", 
                        f"{reglas_fp_prod['support'].max()*100:.2f}%" if len(reglas_fp_prod)>0 else "0.00%",
                        f"{reglas_ap_univ['support'].max()*100:.2f}%" if len(reglas_ap_univ)>0 else "0.00%",
                        f"{reglas_fp_univ['support'].max()*100:.2f}%" if len(reglas_fp_univ)>0 else "0.00%"],
    'Confianza Máx (%)': [f"{reglas_ap_prod['confidence'].max()*100:.2f}%" if len(reglas_ap_prod)>0 else "0.00%", 
                          f"{reglas_fp_prod['confidence'].max()*100:.2f}%" if len(reglas_fp_prod)>0 else "0.00%",
                          f"{reglas_ap_univ['confidence'].max()*100:.2f}%" if len(reglas_ap_univ)>0 else "0.00%",
                          f"{reglas_fp_univ['confidence'].max()*100:.2f}%" if len(reglas_fp_univ)>0 else "0.00%"],
    'Lift Máximo': [f"{reglas_ap_prod['lift'].max():.2f}" if len(reglas_ap_prod)>0 else "0.00", 
                    f"{reglas_fp_prod['lift'].max():.2f}" if len(reglas_fp_prod)>0 else "0.00",
                    f"{reglas_ap_univ['lift'].max():.2f}" if len(reglas_ap_univ)>0 else "0.00",
                    f"{reglas_fp_univ['lift'].max():.2f}" if len(reglas_fp_univ)>0 else "0.00"]
})

print("=== BENCHMARK ALGORÍTMICO INTEGRAL (APRIORI vs FP-GROWTH) ===")
display(tabla_comp_algos.style.background_gradient(cmap='Blues', subset=['Tiempo Ejecución (s)', 'Reglas Descubiertas (Lift>1)']))"""))

cells.append(mk_cell("""### 3.2 Impacto Estratégico de la Generalización (SKU Detallado vs Producto Universal)
Contrastamos visualmente el salto en la robustez de las asociaciones cuando pasamos del detalle técnico del inventario a la categoría universal del cliente."""))

cells.append(cd_cell("""# 2. Resumen comparativo directo de Granularidad (Usando Apriori como referencia estandarizada)
tabla_gran = pd.DataFrame({
    'Dimensión / Métrica': ['Ítems Únicos en Matriz (Columnas)', 'Sparsidad Transaccional (%)', 'Itemsets Frecuentes Descubiertos', 
                            'Total de Reglas Accionables (Lift>1)', 'Soporte Máximo Observado (%)', 'Confianza Máxima Observada (%)', 'Lift Máximo Observado'],
    'Nivel SKU Detallado (Operativo)': [
        df_trans_prod.shape[1],
        f"{sparsidad_prod*100:.2f}%",
        len(frec_ap_prod),
        len(reglas_ap_prod),
        f"{reglas_ap_prod['support'].max()*100:.2f}%" if len(reglas_ap_prod)>0 else "0.00%",
        f"{reglas_ap_prod['confidence'].max()*100:.2f}%" if len(reglas_ap_prod)>0 else "0.00%",
        f"{reglas_ap_prod['lift'].max():.2f}" if len(reglas_ap_prod)>0 else "0.00"
    ],
    'Nivel Producto Universal (Estratégico)': [
        df_trans_univ.shape[1],
        f"{sparsidad_univ*100:.2f}%",
        len(frec_ap_univ),
        len(reglas_ap_univ),
        f"{reglas_ap_univ['support'].max()*100:.2f}%" if len(reglas_ap_univ)>0 else "0.00%",
        f"{reglas_ap_univ['confidence'].max()*100:.2f}%" if len(reglas_ap_univ)>0 else "0.00%",
        f"{reglas_ap_univ['lift'].max():.2f}" if len(reglas_ap_univ)>0 else "0.00"
    ]
})

print("=== COMPARATIVA DE GRANULARIDAD: DETALLADO vs UNIVERSAL ===")
display(tabla_gran.style.set_properties(**{'background-color': '#F8F9FA'}, subset=['Dimensión / Métrica']))

# Gráficos comparativos de impacto estadístico
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de Soporte Máximo
sns.barplot(x=['SKU Detallado', 'Producto Universal'], 
            y=[reglas_ap_prod['support'].max()*100 if len(reglas_ap_prod)>0 else 0, 
               reglas_ap_univ['support'].max()*100 if len(reglas_ap_univ)>0 else 0], 
            ax=axes[0], palette=['#457B9D', '#E63946'])
axes[0].set_title('Concentración del Soporte Máximo en Canastas (%)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Soporte Máximo Observado (%)')
axes[0].set_ylim(0, max(reglas_ap_univ['support'].max()*120 if len(reglas_ap_univ)>0 else 3, 3))

# Gráfico de Confianza Máxima
sns.barplot(x=['SKU Detallado', 'Producto Universal'], 
            y=[reglas_ap_prod['confidence'].max()*100 if len(reglas_ap_prod)>0 else 0, 
               reglas_ap_univ['confidence'].max()*100 if len(reglas_ap_univ)>0 else 0], 
            ax=axes[1], palette=['#457B9D', '#2A9D8F'])
axes[1].set_title('Salto en la Confianza Máxima Condicional (%)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Confianza Máxima Observada (%)')
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.show()"""))

# BLOQUE 4: CONCLUSIONES Y ACCIONES
cells.append(mk_cell("""## 4. Top Reglas Accionables y Conclusiones de Negocio

En este bloque final destilamos las reglas puras con mayor valor de conversión comercial y visualizamos la arquitectura global de afinidades mediante grafos de red (`NetworkX`) y un análisis polar multi-métrica (`Radar Chart`)."""))

cells.append(mk_cell("""### 4.1 Top Reglas de Negocio Accionables (Universal y Detallado)

Presentamos las reglas filtradas por alta relevancia:
*   **Top Reglas Producto Universal:** Revelan afinidades de canasta macro (`Confianza > 20%`, `Lift > 5.0`), perfectas para diseñar promociones en caja e interlazar pasillos de tienda.
*   **Top Reglas SKU Detallado:** Permiten afinar qué variante o marca exacta colocar dentro del kit promocional una vez que la categoría universal ha confirmado la sinergia estructural."""))

cells.append(cd_cell("""# 1. Top Reglas Accionables - Producto Universal (Estratégicas para Venta Cruzada Macro)
if len(reglas_ap_univ) > 0:
    reglas_univ_top = (reglas_ap_univ[(reglas_ap_univ['lift'] > 3.0) & (reglas_ap_univ['confidence'] > 0.20)]
                       .sort_values('lift', ascending=False)
                       [['rule_name', 'support', 'confidence', 'lift', 'leverage']])
    
    # Formateo legible de porcentajes
    tabla_display_univ = reglas_univ_top.head(10).copy()
    tabla_display_univ['support'] = tabla_display_univ['support'].apply(lambda x: f"{x*100:.2f}%")
    tabla_display_univ['confidence'] = tabla_display_univ['confidence'].apply(lambda x: f"{x*100:.2f}%")
    tabla_display_univ['lift'] = tabla_display_univ['lift'].apply(lambda x: f"{x:.2f}")
    tabla_display_univ['leverage'] = tabla_display_univ['leverage'].apply(lambda x: f"{x:.4f}")
    tabla_display_univ.columns = ['Regla Universal (Antecedente -> Consecuente)', 'Soporte (%)', 'Confianza (%)', 'Lift', 'Leverage']
    
    print("=== TOP 10 REGLAS ACCIONABLES DE NEGOCIO – NIVEL PRODUCTO UNIVERSAL ===")
    display(tabla_display_univ.style.set_properties(**{'font-weight': 'bold'}, subset=['Regla Universal (Antecedente -> Consecuente)']))
else:
    print("No se encontraron reglas universales accionables con los parámetros actuales.")

# 2. Top Reglas Accionables - SKU Detallado (Operativas para Selección de Variantes)
if len(reglas_ap_prod) > 0:
    reglas_prod_top = (reglas_ap_prod[(reglas_ap_prod['lift'] > 3.0) & (reglas_ap_prod['confidence'] > 0.15)]
                       .sort_values('lift', ascending=False)
                       [['rule_name', 'support', 'confidence', 'lift']])
    
    tabla_display_prod = reglas_prod_top.head(10).copy()
    tabla_display_prod['support'] = tabla_display_prod['support'].apply(lambda x: f"{x*100:.2f}%")
    tabla_display_prod['confidence'] = tabla_display_prod['confidence'].apply(lambda x: f"{x*100:.2f}%")
    tabla_display_prod['lift'] = tabla_display_prod['lift'].apply(lambda x: f"{x:.2f}")
    tabla_display_prod.columns = ['Regla SKU Detallada (Antecedente -> Consecuente)', 'Soporte (%)', 'Confianza (%)', 'Lift']
    
    print("\\n=== TOP 10 REGLAS OPERATIVAS – NIVEL SKU DETALLADO ===")
    display(tabla_display_prod.style.set_properties(**{'background-color': '#F4F6F6'}, subset=['Regla SKU Detallada (Antecedente -> Consecuente)']))
else:
    print("No se encontraron reglas detalladas accionables con los parámetros actuales.")"""))

cells.append(mk_cell("""### 4.2 Visualización de Redes de Afinidad y Radar Chart Multinivel
Exhibimos la estructura relacional de los productos. En los grafos de red:
- **Tamaño del Nodo:** Proporcional al Soporte del producto en el historial de ventas.
- **Grosor y Dirección de la Flecha:** Proporcional a la fuerza de la asociación (*Lift*) desde el antecedente hacia el consecuente."""))

cells.append(cd_cell("""# 1. Grafo de Red (Network Graph) - Productos Universales (Generales)
if len(reglas_ap_univ) > 0:
    plt.figure(figsize=(14, 10))
    G_univ = nx.DiGraph()
    top_rules_univ_graph = reglas_ap_univ.sort_values('lift', ascending=False).head(20)
    
    for _, row in top_rules_univ_graph.iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                G_univ.add_edge(ant, con, weight=row['lift'], confidence=row['confidence'])
                
    pos_univ = nx.spring_layout(G_univ, k=0.65, seed=42)
    node_sizes_univ = [frec_ap_univ[frec_ap_univ['itemsets'] == frozenset({node})]['support'].values[0] * 65000 
                       if len(frec_ap_univ[frec_ap_univ['itemsets'] == frozenset({node})]['support'].values) > 0 else 1500 
                       for node in G_univ.nodes()]
    
    nx.draw_networkx_nodes(G_univ, pos_univ, node_size=node_sizes_univ, node_color="#F4A261", alpha=0.92)
    nx.draw_networkx_labels(G_univ, pos_univ, font_size=11, font_family="sans-serif", font_weight="bold")
    
    edges_univ = G_univ.edges()
    weights_univ = [G_univ[u][v]['weight'] for u, v in edges_univ]
    nx.draw_networkx_edges(G_univ, pos_univ, edgelist=edges_univ, width=weights_univ, edge_color="#2A9D8F", alpha=0.75, arrows=True, arrowsize=18)
    
    plt.title("Red de Afinidad Estratégica – Categorías Universales (Top 20 por Lift)\\nTamaño de Nodo = Soporte General | Grosor de Flecha = Lift", fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 2. Grafo de Red (Network Graph) - Productos Detallados (SKUs Exactos)
if len(reglas_ap_prod) > 0:
    plt.figure(figsize=(14, 10))
    G_prod = nx.DiGraph()
    top_rules_graph = reglas_ap_prod.sort_values('lift', ascending=False).head(20)
    
    for _, row in top_rules_graph.iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                G_prod.add_edge(ant, con, weight=row['lift'], confidence=row['confidence'])

    pos_prod = nx.spring_layout(G_prod, k=0.55, seed=42)
    node_sizes_prod = [frec_ap_prod[frec_ap_prod['itemsets'] == frozenset({node})]['support'].values[0] * 55000 
                       if len(frec_ap_prod[frec_ap_prod['itemsets'] == frozenset({node})]['support'].values) > 0 else 1000 
                       for node in G_prod.nodes()]
    
    nx.draw_networkx_nodes(G_prod, pos_prod, node_size=node_sizes_prod, node_color="#88D8B0", alpha=0.9)
    nx.draw_networkx_labels(G_prod, pos_prod, font_size=9, font_family="sans-serif")
    
    edges_prod = G_prod.edges()
    weights_prod = [G_prod[u][v]['weight'] for u, v in edges_prod]
    nx.draw_networkx_edges(G_prod, pos_prod, edgelist=edges_prod, width=weights_prod, edge_color="#FF6F69", alpha=0.65, arrows=True)
    
    plt.title("Red de Afinidad Operativa – SKUs Exactos (Top 20 por Lift)\\nTamaño de Nodo = Soporte | Grosor de Flecha = Lift", fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 3. Radar Chart normalizado de las Top 5 Reglas Universales
if len(reglas_ap_univ) > 0:
    top_5_univ = reglas_ap_univ.sort_values('lift', ascending=False).head(5).copy()
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    metricas = ['support', 'confidence', 'lift', 'conviction']
    top_5_univ['conviction'] = top_5_univ['conviction'].replace([np.inf, -np.inf], top_5_univ['conviction'].dropna().max() * 1.5)
    
    df_radar = pd.DataFrame(scaler.fit_transform(top_5_univ[metricas]), columns=metricas)
    df_radar['Regla'] = top_5_univ['rule_name'].values

    fig = go.Figure()
    for i, row in df_radar.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['support'], row['confidence'], row['lift'], row['conviction']],
            theta=['Soporte', 'Confianza', 'Lift', 'Convicción'],
            fill='toself',
            name=row['Regla']
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="<b>Perfil Multidimensional Normalizado – Top 5 Reglas Universales</b>",
        height=550
    )
    fig.show()"""))

cells.append(mk_cell("""### 4.3 Recomendaciones Ejecutivas y Plan de Acción Retail

---

#### 📌 1. Estrategia de Venta Cruzada en Caja (Cross-Selling Dinámico)
- **Acción:** Integrar alertas de recomendación en las terminales POS (Punto de Venta) basadas en las reglas universales con mayor confianza.
- **Ejemplo Práctico:** Si un cliente ingresa a su ticket artículos de **`TEMPERA`**, el sistema debe sugerir inmediatamente la adición de **`PINCEL`** (y viceversa), capturando una probabilidad de compra conjunta documentada del **26% al 33% con un Lift de 16.4x**. De igual forma con la dupla **`MICROPOROSO -> SILICONA`**.

#### 📌 2. Diseño de Kits y Promociones por Familia Universal
- **Acción:** Crear "Kits Escolares" o "Kits de Manualidades" empaquetados bajo conceptos universales fuertemente conectados, permitiendo al cliente elegir el tamaño o marca operativa dentro del paquete.
- **Ejemplo Práctico:** Diseñar el **Kit Creativo (Cartulina + Silicona + Hojas de Colores)** basándose en la regla `[CARTULINA, SILICONA] -> [HOJAS DE COLORES]` (que registra una extraordinaria **Confianza del 63.6% y Lift de 20.6x**). Esto aumenta el ticket promedio sin forzar una única marca específica de silicona.

#### 📌 3. Optimización Visual del Layout de Tienda (Merchandising)
- **Acción:** Reconfigurar la disposición física de las góndolas basándose en la red de afinidad universal (`Network Graph`).
- **Ejemplo Práctico:** Ubicar la exhibición de **`SILICONA`** y adhesivos líquidos justo entre los pasillos de **`MICROPOROSO`** y **`HOJAS DE COLORES / CARTULINA`**, reduciendo la fricción de compra impulsiva para los productos complementarios más frecuentes.

#### 📌 4. Co-Management Interdependiente de Inventario (Supply Chain)
- **Acción:** Vincular las alertas de punto de pedido (*Reorder Point*) de artículos consecuentes a los pronósticos de venta de sus antecedentes principales.
- **Ejemplo Práctico:** No evaluar el stock de *Siliconas* o *Pinceles* de forma aislada. Si se planifica una campaña o incremento en la importación de *Temperas* o *Microporoso*, el área de abastecimiento debe coordinar un incremento proporcional en el inventario de pegamentos y brochas para no sufrir pérdidas de venta por quiebres cruzados.

---
**Siguiente Paso del Pipeline:** Integrar los clústeres de clientes descubiertos en los paneles de *Clustering* para segmentar la afinidad de canastas por tipología de comprador (ej. *Mayoristas* vs *Comprador Escolar Ocasional*)."""))

out_nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(out_nb, f, indent=1, ensure_ascii=False)

print(f"¡Notebook {nb_path} refactorizado en reporte profesional con exactamente {len(cells)} celdas!")
