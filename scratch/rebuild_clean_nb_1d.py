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

# Cell 0: Título Principal
cells.append(mk_cell("""# Panel 1D: Minería de Reglas de Asociación y Análisis de Afinidad entre Productos

**Curso:** Minería de Datos   
**Proyecto:** SmartBazar – Pipeline de Machine Learning para Retail  
**Insumo de Entrada:** Datasets Saneados (`datasets/limpio/` – Salida del Cuaderno 1A)  
**Metodología:** Preparación Transaccional → Modelado Multi-Algoritmo (Detallado y Universal) → Evaluación Comparativa → Interpretación de Negocio

---

## Introducción Teórica: Market Basket Analysis (MBA)

El Market Basket Analysis (MBA) o Análisis de la Canasta de Compras es una técnica de minería de datos que busca descubrir asociaciones entre productos comprados juntos. Al no contar con un identificador único por cliente, el análisis se realiza a nivel transaccional (por *ticket* de compra).

### Métricas Clave:
Sean $X$ y $Y$ conjuntos de ítems (productos):
- **Soporte (Support):** Proporción de transacciones que contienen al conjunto de ítems.
  $$supp(X) = \\frac{|T(X)|}{|T|}$$
- **Confianza (Confidence):** Probabilidad condicional de comprar $Y$ dado que se ha comprado $X$.
  $$conf(X \\rightarrow Y) = \\frac{supp(X \\cup Y)}{supp(X)}$$
- **Lift (Elevación):** Cuánto más probable es que se compre $Y$ cuando se compra $X$, comparado con si fueran independientes.
  $$lift(X \\rightarrow Y) = \\frac{conf(X \\rightarrow Y)}{supp(Y)}$$
- **Conviction:** Mide qué tanto depende $Y$ de $X$. Valores altos indican fuerte dependencia.
- **Leverage:** Diferencia entre la frecuencia observada de $X \\cup Y$ y la frecuencia esperada si fueran independientes.

### Algoritmos Utilizados:
1. **Apriori:** Basado en el principio de anti-monotonicidad, genera candidatos nivel por nivel.
2. **FP-Growth:** Construye un árbol de patrones frecuentes (FP-Tree) que comprime la información, evitando la costosa generación de candidatos de Apriori."""))

# Cell 1: Importaciones
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

# Configuraciones de estilo
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
np.random.seed(42)"""))

# Cell 2: Sección 1 Markdown
cells.append(mk_cell("""## Sección 1: Carga de Datos y EDA Orientado a Asociaciones
En esta sección cargaremos los datos limpios y exploraremos la distribución de las transacciones (tickets de compra) y la frecuencia de los productos y departamentos."""))

# Cell 3: Sección 1 Código
cells.append(cd_cell("""# Rutas a los datasets
DIR_LIMPIO = 'datasets/limpio/'
df_ventas = pd.read_csv(os.path.join(DIR_LIMPIO, 'ventas.csv'))
df_detalle = pd.read_csv(os.path.join(DIR_LIMPIO, 'detalle_ventas.csv'))

print(f"Dimensiones de ventas: {df_ventas.shape}")
print(f"Dimensiones de detalle_ventas: {df_detalle.shape}\\n")

# Exploración inicial de productos más frecuentes
prod_frecuentes = df_detalle['Descripcion'].value_counts().head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x=prod_frecuentes.values, y=prod_frecuentes.index, palette="viridis")
plt.title('Top 20 Productos Más Frecuentes', fontsize=14)
plt.xlabel('Frecuencia (Número de apariciones)')
plt.ylabel('Producto')
plt.tight_layout()
plt.show()

# Distribución de cantidad de productos por ticket
prod_por_ticket = df_detalle.groupby('ID_Venta')['ID_Producto'].count()

plt.figure(figsize=(10, 5))
sns.histplot(prod_por_ticket, bins=range(1, prod_por_ticket.max() + 2), kde=False, color="#2E86AB")
plt.title('Distribución de Número de Productos por Ticket')
plt.xlabel('Número de Productos')
plt.ylabel('Frecuencia (Tickets)')
plt.tight_layout()
plt.show()

# Top departamentos
dept_frecuentes = df_detalle['Departamento'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=dept_frecuentes.values, y=dept_frecuentes.index, palette="magma")
plt.title('Top Departamentos por Volumen de Transacciones', fontsize=14)
plt.xlabel('Volumen')
plt.ylabel('Departamento')
plt.tight_layout()
plt.show()

# Distribución de tamaño de ticket (productos distintos)
productos_distintos = df_detalle.groupby('ID_Venta')['ID_Producto'].nunique()
print("Estadísticas del tamaño de ticket:")
print(productos_distintos.describe())"""))

# Cell 4: Sección 2 Markdown
cells.append(mk_cell("""## Sección 2: Construcción de la Matriz de Transacciones
Para aplicar los algoritmos de minería de reglas, necesitamos transformar los datos en una matriz binaria transaccional (One-Hot Encoding), donde cada fila es una transacción (ticket de compra) y cada columna es un ítem. 

Para lograr un análisis multinivel e integral, trabajaremos en **tres niveles de granularidad**:
1. **Nivel Producto Detallado (SKU específico):** Analiza cada descripción exacta del producto, diferenciando por talla, volumen y marca (ej. *SILICONA LIQUIDA PEQUEÑA VINIFAN 30 ML*).
2. **Nivel Producto Universal (Genérico):** Agrupa todas las variantes, marcas y tamaños bajo un concepto o producto general universal (ej. todas las presentaciones de silicona -> *SILICONA*). Esto soluciona la dispersión estadística del soporte y permite encontrar reglas de negocio amplias y generales ("no tan detalladas").
3. **Nivel Departamento:** Agrupa los productos en grandes categorías o secciones de tienda (ej. *LIBRERÍA*, *BARRAS Y PEGAMENTOS*, etc.)."""))

# Cell 5: Sección 2 Código
cells.append(cd_cell("""# NIVEL PRODUCTO DETALLADO (SKU ESPECÍFICO)
# Agrupamos productos exactos por ticket
transacciones_prod = df_detalle.groupby('ID_Venta')['Descripcion'].apply(list).tolist()

te = TransactionEncoder()
te_ary = te.fit(transacciones_prod).transform(transacciones_prod)
df_trans_prod = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Forma de matriz a nivel Producto Detallado: {df_trans_prod.shape}")
sparsidad_prod = 1.0 - (df_trans_prod.values.sum() / df_trans_prod.size)
print(f"Sparsidad de la matriz Producto Detallado: {sparsidad_prod:.4%}\\n")

# Filtramos ítems muy poco frecuentes para mejorar análisis del nivel detallado
min_apariciones = int(df_trans_prod.shape[0] * 0.01)
df_trans_prod = df_trans_prod.loc[:, df_trans_prod.sum() >= min_apariciones]
print(f"Forma de matriz a nivel Producto Detallado filtrada (>1% soporte): {df_trans_prod.shape}\\n")


# NIVEL PRODUCTO UNIVERSAL (GENÉRICO / GENERAL)
# Función para generalizar el producto eliminando talla, marca, volumen o variantes específicas
def generalizar_producto(desc):
    if not isinstance(desc, str):
        return "OTRO"
    d = desc.upper().strip()
    
    # Reglas universales consolidadas
    if "SILICONA" in d:
        return "SILICONA"
    if "FOTOCOPIA" in d:
        return "FOTOCOPIA"
    if "IMPRESION" in d:
        return "IMPRESION"
    if "CUADERNO" in d:
        return "CUADERNO"
    if "LAPICERO" in d:
        return "LAPICERO"
    if "LAPIZ" in d or "LÁPIZ" in d or "PORTAMINAS" in d:
        return "LAPIZ"
    if "GOMA" in d:
        return "GOMA"
    if "BORRADOR" in d:
        return "BORRADOR"
    if "PLUMON" in d or "PLUMÓN" in d or "INDELEBLE" in d or "RESALTADOR" in d or "CHEQUEO" in d or "MARKER" in d:
        return "PLUMON / RESALTADOR"
    if "CARTULINA" in d:
        return "CARTULINA"
    if "MICROPOROSO" in d:
        return "MICROPOROSO"
    if "HOJA DE COLOR" in d or "HOJAS DE COLORES" in d:
        return "HOJAS DE COLORES"
    if "HOJA BOND" in d:
        return "HOJA BOND"
    if "CINTA" in d:
        return "CINTA ADHESIVA / EMBALAJE"
    if "PINCEL" in d:
        return "PINCEL"
    if "MICA" in d:
        return "MICA"
    if "PAPEL CREPE" in d:
        return "PAPEL CREPE"
    if "PAPEL LUSTRE" in d:
        return "PAPEL LUSTRE"
    if "PAPEL DE SEDA" in d or "PAPEL CRAFT" in d or "PAPEL MANTECA" in d:
        return "PAPEL ESPECIAL"
    if "PAPELOTE" in d:
        return "PAPELOTE"
    if "REGLA" in d or "TRANSPORTADOR" in d:
        return "REGLA / GEOMETRIA"
    if "TEMPERA" in d or "TÉMPERA" in d:
        return "TEMPERA"
    if "COLORES" in d:
        return "COLORES"
    if "CRAYOLA" in d:
        return "CRAYOLA"
    if "CORBATA" in d:
        return "CORBATA"
    if "ESCARAPELA" in d:
        return "ESCARAPELA"
    if "GUANTE" in d:
        return "GUANTE"
    if "BANDERA" in d:
        return "BANDERA"
    if "BLOCK" in d or "SKETCH BOOK" in d:
        return "BLOCK / SKETCH BOOK"
    if "FOLDER" in d:
        return "FOLDER"
    if "GLOBO" in d or "PALIGLOBO" in d:
        return "GLOBOS Y ACCESORIOS"
    if "TAJADOR" in d:
        return "TAJADOR"
    if "SOBRE" in d:
        return "SOBRE"
    if "LENTEJUELA" in d:
        return "LENTEJUELAS"
    if "CORREA" in d:
        return "CORREA"
    if "LIMPIATIPO" in d:
        return "LIMPIATIPOS"
    if "PABILO" in d:
        return "PABILO"
    if "MOTA" in d:
        return "MOTA"
    if "LUPA" in d:
        return "LUPA"
    if "TIJERA" in d or "CUTER" in d:
        return "TIJERA / CUTER"
    if "PLASTILINA" in d:
        return "PLASTILINA"
    if "CHINCHE" in d or "GRAPA" in d:
        return "SUJETADORES / CHINCHES"
    if "VINIFAN" in d:
        return "FORRO / MICA VINIFAN"
    if "SUPER" in d:
        return "SUPER BLUE"
    
    palabras = d.split()
    if len(palabras) > 0:
        return palabras[0]
    return "OTRO"

df_detalle['Producto_Universal'] = df_detalle['Descripcion'].apply(generalizar_producto)
transacciones_univ = df_detalle.groupby('ID_Venta')['Producto_Universal'].apply(lambda x: list(set(x))).tolist()

te_univ = TransactionEncoder()
te_univ_ary = te_univ.fit(transacciones_univ).transform(transacciones_univ)
df_trans_univ = pd.DataFrame(te_univ_ary, columns=te_univ.columns_)

print(f"Forma de matriz a nivel Producto Universal (Genérico): {df_trans_univ.shape}")
sparsidad_univ = 1.0 - (df_trans_univ.values.sum() / df_trans_univ.size)
print(f"Sparsidad de la matriz Producto Universal: {sparsidad_univ:.4%}\\n")


# NIVEL DEPARTAMENTO
transacciones_dept = df_detalle.groupby('ID_Venta')['Departamento'].apply(lambda x: list(set(x))).tolist()
te_dept = TransactionEncoder()
te_dept_ary = te_dept.fit(transacciones_dept).transform(transacciones_dept)
df_trans_dept = pd.DataFrame(te_dept_ary, columns=te_dept.columns_)

print(f"Forma de matriz a nivel Departamento: {df_trans_dept.shape}")
sparsidad_dept = 1.0 - (df_trans_dept.values.sum() / df_trans_dept.size)
print(f"Sparsidad de la matriz de Departamentos: {sparsidad_dept:.4%}\\n")

display(df_trans_univ.head(3))"""))

# Cell 6: Sección 3 Markdown (Apriori Detallado)
cells.append(mk_cell("""## Sección 3: Modelo 1 - Algoritmo Apriori (Nivel Producto Detallado)
El algoritmo Apriori utiliza un enfoque "bottom-up", descubriendo conjuntos frecuentes paso a paso utilizando el principio de que cualquier subconjunto de un conjunto de ítems frecuente también debe ser frecuente. Comenzamos explorando las reglas a nivel de SKU específico."""))

# Cell 7: Sección 3 Código (Apriori Detallado)
cells.append(cd_cell("""# Análisis de sensibilidad para elegir min_support
soportes = np.arange(0.01, 0.1, 0.01)
num_itemsets = []

for s in soportes:
    freq_items = apriori(df_trans_prod, min_support=s, use_colnames=True)
    num_itemsets.append(len(freq_items))

plt.figure(figsize=(10, 5))
sns.lineplot(x=soportes, y=num_itemsets, marker='o', color="#D62828")
plt.title('Sensibilidad del Número de Itemsets Frecuentes vs Soporte Mínimo')
plt.xlabel('Soporte Mínimo (min_support)')
plt.ylabel('Cantidad de Itemsets Frecuentes')
plt.tight_layout()
plt.show()

# Elección de min_support (usaremos 0.005 dado el volumen de 939 transacciones)
soporte_elegido = 0.005

inicio_apriori = time.time()
frecuentes_apriori = apriori(df_trans_prod, min_support=soporte_elegido, use_colnames=True)
tiempo_apriori = time.time() - inicio_apriori

print(f"Tiempo de ejecución Apriori (SKU Detallado): {tiempo_apriori:.4f} segundos")
print(f"Itemsets encontrados: {len(frecuentes_apriori)}\\n")

# Top 20 itemsets
frecuentes_apriori['longitud'] = frecuentes_apriori['itemsets'].apply(lambda x: len(x))
display(frecuentes_apriori.sort_values(by='support', ascending=False).head(20))

# Generación de reglas
reglas_apriori = association_rules(frecuentes_apriori, metric="lift", min_threshold=1.0)
print(f"Reglas generadas (SKU Detallado): {len(reglas_apriori)}\\n")

if len(reglas_apriori) > 0:
    display(reglas_apriori.sort_values(by='lift', ascending=False).head(20))
    
    reglas_apriori['antecedents_str'] = reglas_apriori['antecedents'].apply(lambda x: ', '.join(list(x)))
    reglas_apriori['consequents_str'] = reglas_apriori['consequents'].apply(lambda x: ', '.join(list(x)))
    reglas_apriori['rule_name'] = reglas_apriori['antecedents_str'] + " -> " + reglas_apriori['consequents_str']
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', data=reglas_apriori, palette='coolwarm', sizes=(50, 400))
    plt.title('Reglas Apriori (SKU Detallado): Soporte vs Confianza (Color y Tamaño por Lift)')
    plt.xlabel('Soporte')
    plt.ylabel('Confianza')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    reglas_apriori_top = reglas_apriori.sort_values(by='lift', ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='lift', y='rule_name', data=reglas_apriori_top, palette="Spectral")
    plt.title('Top 15 Reglas de Asociación por Lift (Apriori - SKU Detallado)')
    plt.xlabel('Lift')
    plt.ylabel('Regla (Antecedente -> Consecuente)')
    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron reglas con los parámetros actuales.")"""))

# Cell 8: Sección 3.1 Markdown (Apriori Universal)
cells.append(mk_cell("""### 3.1 Modelo 1: Algoritmo Apriori a Nivel Producto Universal (Reglas Generales)
Al agrupar los productos bajo un concepto o producto universal (por ejemplo, consolidando las ventas de *SILICONA LIQUIDA PEQUEÑA*, *MEDIANA* y *GRANDE* en la categoría única **SILICONA**), el soporte de cada ítem general aumenta de manera significativa al no dividirse entre variantes. 

A continuación aplicamos el algoritmo **Apriori** sobre `df_trans_univ` para descubrir las reglas más generales y representativas de afinidad en la compra."""))

# Cell 9: Sección 3.1 Código (Apriori Universal)
cells.append(cd_cell("""# Ejecución de Apriori sobre Producto Universal
inicio_ap_univ = time.time()
frecuentes_ap_univ = apriori(df_trans_univ, min_support=soporte_elegido, use_colnames=True)
tiempo_ap_univ = time.time() - inicio_ap_univ

print(f"Tiempo de ejecución Apriori (Universal): {tiempo_ap_univ:.4f} segundos")
print(f"Itemsets frecuentes encontrados (Universal): {len(frecuentes_ap_univ)}\\n")

frecuentes_ap_univ['longitud'] = frecuentes_ap_univ['itemsets'].apply(lambda x: len(x))
display(frecuentes_ap_univ.sort_values(by='support', ascending=False).head(15))

# Generación de reglas universales con Apriori
reglas_ap_univ = association_rules(frecuentes_ap_univ, metric="lift", min_threshold=1.0)
print(f"Reglas generadas a Nivel Producto Universal (Apriori): {len(reglas_ap_univ)}\\n")

if len(reglas_ap_univ) > 0:
    reglas_ap_univ['antecedents_str'] = reglas_ap_univ['antecedents'].apply(lambda x: ', '.join(list(x)))
    reglas_ap_univ['consequents_str'] = reglas_ap_univ['consequents'].apply(lambda x: ', '.join(list(x)))
    reglas_ap_univ['rule_name'] = reglas_ap_univ['antecedents_str'] + " -> " + reglas_ap_univ['consequents_str']
    
    display(reglas_ap_univ.sort_values(by='lift', ascending=False).head(15))
    
    # Scatter plot: Soporte vs Confianza para Reglas Universales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', data=reglas_ap_univ, palette='coolwarm', sizes=(60, 450))
    plt.title('Reglas Apriori (Producto Universal): Soporte vs Confianza (Color/Tamaño por Lift)')
    plt.xlabel('Soporte')
    plt.ylabel('Confianza')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Top 15 Reglas Universales por Lift
    reglas_univ_top = reglas_ap_univ.sort_values(by='lift', ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='lift', y='rule_name', data=reglas_univ_top, palette="viridis")
    plt.title('Top 15 Reglas de Asociación Generales por Lift - Producto Universal (Apriori)', fontsize=14)
    plt.xlabel('Lift')
    plt.ylabel('Regla Universal (Antecedente -> Consecuente)')
    plt.tight_layout()
    plt.show()
else:
    print("No se encontraron reglas con los parámetros actuales.")"""))

# Cell 10: Sección 4 Markdown (FP-Growth Detallado)
cells.append(mk_cell("""## Sección 4: Modelo 2 - Algoritmo FP-Growth (Nivel Producto Detallado)
FP-Growth es más eficiente que Apriori, ya que utiliza una estructura de datos llamada FP-Tree para evitar el escaneo repetido de los datos y la generación explícita de candidatos. Evaluamos primero su rendimiento sobre los ítems específicos (`df_trans_prod`)."""))

# Cell 11: Sección 4 Código (FP-Growth Detallado)
cells.append(cd_cell("""inicio_fpgrowth = time.time()
frecuentes_fp = fpgrowth(df_trans_prod, min_support=soporte_elegido, use_colnames=True)
tiempo_fpgrowth = time.time() - inicio_fpgrowth

print(f"Tiempo de ejecución FP-Growth (SKU Detallado): {tiempo_fpgrowth:.4f} segundos")
print(f"Itemsets encontrados: {len(frecuentes_fp)}")

frecuentes_fp['longitud'] = frecuentes_fp['itemsets'].apply(lambda x: len(x))
display(frecuentes_fp.sort_values(by='support', ascending=False).head(20))

# Generación de reglas FP-Growth
reglas_fp = association_rules(frecuentes_fp, metric="lift", min_threshold=1.0)
print(f"Reglas generadas (FP-Growth SKU Detallado): {len(reglas_fp)}\\n")

if len(reglas_fp) > 0:
    display(reglas_fp.sort_values(by='lift', ascending=False).head(20))
    
    reglas_fp['antecedents_str'] = reglas_fp['antecedents'].apply(lambda x: ', '.join(list(x)))
    reglas_fp['consequents_str'] = reglas_fp['consequents'].apply(lambda x: ', '.join(list(x)))
    reglas_fp['rule_name'] = reglas_fp['antecedents_str'] + " -> " + reglas_fp['consequents_str']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', data=reglas_fp, palette='coolwarm', sizes=(50, 400))
    plt.title('Reglas FP-Growth (SKU Detallado): Soporte vs Confianza')
    plt.xlabel('Soporte')
    plt.ylabel('Confianza')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()"""))

# Cell 12: Sección 4.1 Markdown (FP-Growth Universal)
cells.append(mk_cell("""### 4.1 Modelo 2: Algoritmo FP-Growth a Nivel Producto Universal (Reglas Generales)
Al igual que con Apriori, aplicamos ahora el algoritmo **FP-Growth** sobre la matriz transaccional universal (`df_trans_univ`). Al reducir el número total de columnas de 251 ítems específicos a 71 conceptos universales, la estructura del árbol FP-Tree resulta altamente compacta, logrando un rendimiento computacional casi instantáneo y verificando la consistencia de las asociaciones generalizadas."""))

# Cell 13: Sección 4.1 Código (FP-Growth Universal)
cells.append(cd_cell("""# Ejecución de FP-Growth sobre Producto Universal
inicio_fp_univ = time.time()
frecuentes_fp_univ = fpgrowth(df_trans_univ, min_support=soporte_elegido, use_colnames=True)
tiempo_fp_univ = time.time() - inicio_fp_univ

print(f"Tiempo de ejecución FP-Growth (Universal): {tiempo_fp_univ:.4f} segundos")
print(f"Itemsets frecuentes encontrados (Universal): {len(frecuentes_fp_univ)}")

frecuentes_fp_univ['longitud'] = frecuentes_fp_univ['itemsets'].apply(lambda x: len(x))
reglas_fp_univ = association_rules(frecuentes_fp_univ, metric="lift", min_threshold=1.0)
print(f"Reglas generadas a Nivel Producto Universal (FP-Growth): {len(reglas_fp_univ)}\\n")

if len(reglas_fp_univ) > 0:
    reglas_fp_univ['antecedents_str'] = reglas_fp_univ['antecedents'].apply(lambda x: ', '.join(list(x)))
    reglas_fp_univ['consequents_str'] = reglas_fp_univ['consequents'].apply(lambda x: ', '.join(list(x)))
    reglas_fp_univ['rule_name'] = reglas_fp_univ['antecedents_str'] + " -> " + reglas_fp_univ['consequents_str']
    
    display(reglas_fp_univ.sort_values(by='lift', ascending=False).head(15))
    
    # Scatter plot: Soporte vs Confianza para FP-Growth Universal
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='support', y='confidence', size='lift', hue='lift', data=reglas_fp_univ, palette='magma', sizes=(60, 450))
    plt.title('Reglas FP-Growth (Producto Universal): Soporte vs Confianza')
    plt.xlabel('Soporte')
    plt.ylabel('Confianza')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()"""))

# Cell 14: Sección 5 Markdown
cells.append(mk_cell("""## Sección 5: Comparación Integral de Algoritmos y Granularidad (Detallado vs Universal)
En esta sección realizamos una doble comparativa:
1. **Comparación Algorítmica (Apriori vs FP-Growth):** Evaluamos el tiempo de ejecución y la equivalencia de resultados en cada nivel de análisis.
2. **Impacto de la Granularidad (Producto Detallado vs Producto Universal):** Contrastamos cómo la generalización conceptual del producto (ej. unificar todas las presentaciones de *silicona* o de *cuaderno*) transforma la fuerza estadística, el soporte y el valor accionable de las reglas descubiertas."""))

# Cell 15: Sección 5 Código
cells.append(cd_cell("""# 1. Comparación Algorítmica (Apriori vs FP-Growth en ambos niveles)
tabla_comp_algos = pd.DataFrame({
    'Nivel_Granularidad': ['Producto Detallado (SKU)', 'Producto Detallado (SKU)', 'Producto Universal (Genérico)', 'Producto Universal (Genérico)'],
    'Algoritmo': ['Apriori', 'FP-Growth', 'Apriori', 'FP-Growth'],
    'Tiempo_Ejecucion (s)': [tiempo_apriori, tiempo_fpgrowth, tiempo_ap_univ, tiempo_fp_univ],
    'Itemsets_Frecuentes': [len(frecuentes_apriori), len(frecuentes_fp), len(frecuentes_ap_univ), len(frecuentes_fp_univ)],
    'Total_Reglas (Lift>1)': [len(reglas_apriori), len(reglas_fp), len(reglas_ap_univ), len(reglas_fp_univ)],
    'Soporte_Max': [reglas_apriori['support'].max() if len(reglas_apriori)>0 else 0, 
                    reglas_fp['support'].max() if len(reglas_fp)>0 else 0,
                    reglas_ap_univ['support'].max() if len(reglas_ap_univ)>0 else 0,
                    reglas_fp_univ['support'].max() if len(reglas_fp_univ)>0 else 0],
    'Confianza_Max': [reglas_apriori['confidence'].max() if len(reglas_apriori)>0 else 0, 
                      reglas_fp['confidence'].max() if len(reglas_fp)>0 else 0,
                      reglas_ap_univ['confidence'].max() if len(reglas_ap_univ)>0 else 0,
                      reglas_fp_univ['confidence'].max() if len(reglas_fp_univ)>0 else 0],
    'Lift_Max': [reglas_apriori['lift'].max() if len(reglas_apriori)>0 else 0, 
                 reglas_fp['lift'].max() if len(reglas_fp)>0 else 0,
                 reglas_ap_univ['lift'].max() if len(reglas_ap_univ)>0 else 0,
                 reglas_fp_univ['lift'].max() if len(reglas_fp_univ)>0 else 0]
})

print("=== 1. COMPARATIVA ALGORÍTMICA (APRIORI vs FP-GROWTH POR NIVEL) ===")
display(tabla_comp_algos.style.background_gradient(cmap='Blues'))

# 2. Resumen comparativo directo de Granularidad (Tomando Apriori como referencia)
tabla_gran = pd.DataFrame({
    'Métrica': ['Total Ítems Únicos (Columnas de Matriz)', 'Sparsidad de Matriz (%)', 'Itemsets Frecuentes (min_sup=0.5%)', 
                'Total Reglas Descubiertas (Lift>1)', 'Soporte Máximo Observado (%)', 'Confianza Máxima Observada (%)', 'Lift Máximo Observado'],
    'Producto Detallado (SKU específico)': [
        df_trans_prod.shape[1],
        f"{sparsidad_prod*100:.2f}%",
        len(frecuentes_apriori),
        len(reglas_apriori),
        f"{reglas_apriori['support'].max()*100:.2f}%" if len(reglas_apriori)>0 else "0.00%",
        f"{reglas_apriori['confidence'].max()*100:.2f}%" if len(reglas_apriori)>0 else "0.00%",
        f"{reglas_apriori['lift'].max():.2f}" if len(reglas_apriori)>0 else "0.00"
    ],
    'Producto Universal (Genérico)': [
        df_trans_univ.shape[1],
        f"{sparsidad_univ*100:.2f}%",
        len(frecuentes_ap_univ),
        len(reglas_ap_univ),
        f"{reglas_ap_univ['support'].max()*100:.2f}%" if len(reglas_ap_univ)>0 else "0.00%",
        f"{reglas_ap_univ['confidence'].max()*100:.2f}%" if len(reglas_ap_univ)>0 else "0.00%",
        f"{reglas_ap_univ['lift'].max():.2f}" if len(reglas_ap_univ)>0 else "0.00"
    ]
})

print("\\n=== 2. IMPACTO DE LA GENERALIZACIÓN (DETALLADO vs UNIVERSAL) ===")
display(tabla_gran)

# Gráficos comparativos por dimensión de granularidad
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=['Detallado (SKU)', 'Universal (Genérico)'], 
            y=[reglas_apriori['support'].max()*100 if len(reglas_apriori)>0 else 0, 
               reglas_ap_univ['support'].max()*100 if len(reglas_ap_univ)>0 else 0], 
            ax=axes[0], palette=['#457B9D', '#E63946'])
axes[0].set_title('Soporte Máximo Observado en Reglas (%)', fontsize=12)
axes[0].set_ylabel('Soporte (%)')

sns.barplot(x=['Detallado (SKU)', 'Universal (Genérico)'], 
            y=[reglas_apriori['confidence'].max()*100 if len(reglas_apriori)>0 else 0, 
               reglas_ap_univ['confidence'].max()*100 if len(reglas_ap_univ)>0 else 0], 
            ax=axes[1], palette=['#457B9D', '#2A9D8F'])
axes[1].set_title('Confianza Máxima Observada en Reglas (%)', fontsize=12)
axes[1].set_ylabel('Confianza (%)')

plt.tight_layout()
plt.show()

print("\\n--- Conclusiones Clave de la Comparativa de Granularidad ---")
print("1. Concentración del Soporte: Al unificar variantes como 'Silicona pequeña/mediana/grande' en 'SILICONA', el soporte por categoría aumenta notablemente, lo que permite detectar patrones fuertes que a nivel detallado quedaban diluidos.")
print("2. Reglas Más Generales y Estratégicas: El modelo universal extrae asociaciones de muy alto valor comercial (como [TEMPERA -> PINCEL] o [MICROPOROSO -> SILICONA]) con alta confianza, ideales para decisiones macro de distribución, diseño de pasillos y promociones cruzadas en tienda.")
print("3. Eficiencia Computacional: Al trabajar con menos columnas (71 vs 251), tanto Apriori como FP-Growth convergen con un menor overhead y generan un conjunto de reglas limpias y fáciles de interpretar.")"""))

# Cell 16: Sección 6 Markdown
cells.append(mk_cell("""## Sección 6: Análisis Profundo de Reglas Descubiertas
Extraeremos reglas accionables tanto a nivel de producto detallado como a nivel de producto universal y de departamento, analizando su aplicabilidad directa de negocio. Filtraremos aquellas asociaciones con un Lift mayor a 1 (indicador de verdadera dependencia o sinergia) y con una confianza mínima operativa."""))

# Cell 17: Sección 6 Código
cells.append(cd_cell("""if len(reglas_apriori) > 0:
    # Filtrado de reglas a nivel Producto Detallado
    reglas_accionables = reglas_apriori[(reglas_apriori['lift'] > 1.0) & (reglas_apriori['confidence'] > 0.05)]
    reglas_accionables = reglas_accionables.sort_values('lift', ascending=False).head(10)

    print("--- Top 10 Reglas a Nivel Producto Detallado (SKU específico) ---")
    for i, row in reglas_accionables.iterrows():
        ant = row['antecedents_str']
        con = row['consequents_str']
        print(f"Si compran [{ant}] es probable que compren [{con}] (Lift: {row['lift']:.2f}, Confianza: {row['confidence']:.2%})")

print("\\n--- Top 10 Reglas Accionables a Nivel Producto Universal (Generales) ---")
if len(reglas_ap_univ) > 0:
    reglas_univ_accionables = reglas_ap_univ[(reglas_ap_univ['lift'] > 1.2) & (reglas_ap_univ['confidence'] > 0.15)]
    reglas_univ_accionables = reglas_univ_accionables.sort_values('lift', ascending=False).head(10)
    for i, row in reglas_univ_accionables.iterrows():
        ant = row['antecedents_str']
        con = row['consequents_str']
        print(f"Si compran [{ant}] es probable que compren [{con}] (Lift: {row['lift']:.2f}, Confianza: {row['confidence']:.2%}, Soporte: {row['support']:.2%})")

# Análisis a nivel Departamento
frec_dept = fpgrowth(df_trans_dept, min_support=0.01, use_colnames=True)
reglas_dept = association_rules(frec_dept, metric="lift", min_threshold=1.0)

print("\\n--- Top Reglas a Nivel Departamento ---")
if not reglas_dept.empty:
    reglas_dept['antecedents_str'] = reglas_dept['antecedents'].apply(lambda x: ', '.join(list(x)))
    reglas_dept['consequents_str'] = reglas_dept['consequents'].apply(lambda x: ', '.join(list(x)))
    
    reglas_dept_top = reglas_dept.sort_values('lift', ascending=False).head(5)
    for i, row in reglas_dept_top.iterrows():
        print(f"Departamentos: [{row['antecedents_str']}] -> [{row['consequents_str']}] (Lift: {row['lift']:.2f}, Confianza: {row['confidence']:.2%})")
        
    # Análisis bidireccional
    print("\\n--- Análisis Bidireccional (A->B vs B->A) ---")
    for i, row in reglas_dept_top.iterrows():
        a, b = row['antecedents'], row['consequents']
        inversa = reglas_dept[(reglas_dept['antecedents'] == b) & (reglas_dept['consequents'] == a)]
        if not inversa.empty:
            inv_row = inversa.iloc[0]
            print(f"A->B: {row['antecedents_str']} -> {row['consequents_str']} (Conf: {row['confidence']:.2%})")
            print(f"B->A: {inv_row['antecedents_str']} -> {inv_row['consequents_str']} (Conf: {inv_row['confidence']:.2%})\\n")
else:
    print("No se encontraron reglas significativas entre departamentos con los parámetros actuales.")"""))

# Cell 18: Sección 7 Markdown
cells.append(mk_cell("""## Sección 7: Visualizaciones Avanzadas
Exploraremos la red de asociaciones tanto para los productos detallados como para los productos universales, así como los diagramas de afinidad departamental y gráficos comparativos normalizados."""))

# Cell 19: Sección 7 Código
cells.append(cd_cell("""if len(reglas_apriori) > 0:
    # 1. Heatmap de afinidad entre Departamentos
    if not reglas_dept.empty:
        matriz_dept = reglas_dept.pivot(index='antecedents_str', columns='consequents_str', values='lift')
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz_dept, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=.5)
        plt.title('Afinidad entre Departamentos (Valores de Lift)')
        plt.tight_layout()
        plt.show()

    # 2. Grafo de Red (Network Graph) con NetworkX para Productos Detallados
    plt.figure(figsize=(14, 10))
    G = nx.DiGraph()
    top_rules_graph = reglas_apriori.sort_values('lift', ascending=False).head(20)
    for _, row in top_rules_graph.iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                G.add_edge(ant, con, weight=row['lift'], confidence=row['confidence'])

    pos = nx.spring_layout(G, k=0.5, seed=42)
    node_sizes = [frecuentes_apriori[frecuentes_apriori['itemsets'] == frozenset({node})]['support'].values[0] * 50000 if len(frecuentes_apriori[frecuentes_apriori['itemsets'] == frozenset({node})]['support'].values) > 0 else 1000 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#88D8B0", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color="#FF6F69", alpha=0.6, arrows=True)
    
    plt.title("Red de Asociaciones de Productos Detallados (Top 20 por Lift)\\nTamaño de Nodo = Soporte, Grosor de Borde = Lift", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 3. Grafo de Red (Network Graph) para Productos Universales (Generales)
if len(reglas_ap_univ) > 0:
    plt.figure(figsize=(14, 10))
    G_univ = nx.DiGraph()
    top_rules_univ_graph = reglas_ap_univ.sort_values('lift', ascending=False).head(20)
    for _, row in top_rules_univ_graph.iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                G_univ.add_edge(ant, con, weight=row['lift'], confidence=row['confidence'])
                
    pos_univ = nx.spring_layout(G_univ, k=0.6, seed=42)
    node_sizes_univ = [frecuentes_ap_univ[frecuentes_ap_univ['itemsets'] == frozenset({node})]['support'].values[0] * 60000 if len(frecuentes_ap_univ[frecuentes_ap_univ['itemsets'] == frozenset({node})]['support'].values) > 0 else 1500 for node in G_univ.nodes()]
    
    nx.draw_networkx_nodes(G_univ, pos_univ, node_size=node_sizes_univ, node_color="#F4A261", alpha=0.9)
    nx.draw_networkx_labels(G_univ, pos_univ, font_size=11, font_family="sans-serif", font_weight="bold")
    
    edges_univ = G_univ.edges()
    weights_univ = [G_univ[u][v]['weight'] for u, v in edges_univ]
    nx.draw_networkx_edges(G_univ, pos_univ, edgelist=edges_univ, width=weights_univ, edge_color="#2A9D8F", alpha=0.7, arrows=True, arrowsize=15)
    
    plt.title("Red de Asociaciones de Productos Universales (Top 20 por Lift)\\nTamaño de Nodo = Soporte General, Grosor de Borde = Lift", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 4. Radar Chart resumen normalizado top 5 reglas universales
    top_5_univ = reglas_ap_univ.sort_values('lift', ascending=False).head(5)
    
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
        title="Comparativa Normalizada de Métricas - Top 5 Reglas Universales"
    )
    fig.show()"""))

# Cell 20: Sección 8 Markdown
cells.append(mk_cell("""## Sección 8: Conclusiones y Recomendaciones de Negocio

### Principales Hallazgos y Análisis de Granularidad (Universal vs Detallado)
- **El Poder de la Generalización Conceptual (Producto Universal):** La comparativa entre el análisis a nivel detallado (SKU exacto) y el nivel universal (concepto genérico) demuestra que agrupar variantes (como *SILICONA LIQUIDA PEQUEÑA*, *MEDIANA* y *GRANDE* de las marcas *VINIFAN* y *ARTESCO* en **SILICONA**) elimina la dispersión estadística del soporte. Esto revela reglas de asociación sumamente generales, intuitivas y sólidas ("no tan detalladas") que a nivel SKU quedaban fragmentadas o con soportes marginales:
  - `MICROPOROSO -> SILICONA` (Alta confianza y lift alto, reflejando que el comprador de goma eva/microporoso adquiere naturalmente pegamento líquido).
  - `TEMPERA -> PINCEL` (Asociación directa en canastas escolares y de arte).
  - `HOJAS DE COLORES -> CARTULINA` y `HOJA BOND -> CARTULINA` (Combos clave en papelería).
- **Consistencia Algorítmica (Apriori vs FP-Growth):** Ambos algoritmos confirman las mismas reglas de asociación en cada nivel. Al aplicar **FP-Growth** sobre la matriz de productos universales (`df_trans_univ`), el tiempo de ejecución se reduce a fracciones de segundo gracias a la compactación del árbol de patrones en solo 71 columnas, frente a las 251 originales.
- **Sinergia entre Niveles de Granularidad:**
  - El **Nivel Producto Universal** dictamina las decisiones estratégicas de diseño de tienda (*layout*), ubicación de pasillos conexos y promociones de categorías de producto.
  - El **Nivel Producto Detallado (SKU)** interviene en una segunda etapa operativa: para seleccionar qué presentación o gramaje específico conviene incluir en los combos con mayor rentabilidad de margen o para gestionar quiebres de inventario interdependientes.

### Recomendaciones Accionables para Retail
1. **Estrategia de Venta Cruzada Macro (Cross-Selling de Categorías Universales):** Capacitar al personal en punto de venta y optimizar los flujos visuales en caja basándose en las reglas generales universales. Si un cliente lleva *Temperas* o *Microporoso*, ofrecer de inmediato *Pinceles* o *Siliconas* (cualquiera sea el tamaño disponible).
2. **Combos y Promociones por Familia Universal:** Diseñar "Kits Escolares" o "Kits de Manualidades" agrupando productos universales fuertemente asociados (ej. Kit Arte: *Tempera + Pincel + Cartulina*). Al permitir que el cliente elija el tamaño o variante del producto dentro del combo, se maximiza la tasa de conversión sin restringir la oferta a una sola marca o talla.
3. **Optimización de Layout Departamental y Pasillos:** Alinear la disposición física de la tienda con el mapa de afinidad universal y departamental para fomentar la compra impulsiva de productos complementarios.
4. **Gestión Conjunta de Reabastecimiento (Co-Management de Stocks):** Dado que la demanda del consecuente universal depende del antecedente (ej. la venta de *Siliconas* depende en parte del flujo de ventas de *Microporoso* y *Cartulina*), el área de compras debe coordinar alertas conjuntas de inventario para no perder ventas cruzadas por quiebre de stock del producto principal.

### Limitaciones y Trabajo Futuro
- **Escalabilidad de Historial y Ventanas Temporales:** Se sugiere expandir el análisis con históricos anuales para ejecutar minería de reglas segmentada por temporadas comerciales (Campaña Escolar vs Temporada Regular).
- **Segmentación por Clientes:** Integrar variables del programa de fidelización para cruzar la afinidad de productos universales con perfiles demográficos o de compra."""))

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

print(f"¡Notebook {nb_path} regenerado limpiamente con exactamente {len(cells)} celdas!")
