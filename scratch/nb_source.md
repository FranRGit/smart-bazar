
## Cell 0 (markdown)
`python
# Panel 1C: Segmentación Transaccional y Clustering de Comportamiento de Compra

**Curso:** Minería de Datos  
**Proyecto:** SmartBazar – Pipeline de Machine Learning para Retail  
**Insumo de Entrada:** Datasets Saneados (`datasets/limpio/` – Salida del Cuaderno 1A)  
**Metodología:** Feature Engineering Transaccional → Limpieza de Outliers → Selección de Modelos → Evaluación Comparativa → Perfilamiento Ejecutivo e Insights de Negocio  

---

## Introducción Teórica y Arquitectura Analítica

El clustering o agrupamiento es una técnica fundamental de aprendizaje no supervisado que tiene como objetivo estructurar un conjunto de datos particionándolo en grupos (clusters) que maximicen la homogeneidad interna (intra-cluster) y la heterogeneidad externa (inter-clusters). 

### Contexto Crítico del Negocio:
En este proyecto, debido a la ausencia de un identificador único por cliente en el sistema de ventas (donde `ID_Cliente` está generalizado en 1), abordamos el problema desde una perspectiva **transaccional pura**. Nuestra unidad de análisis es el **ticket de compra (transacción)** y no el consumidor. 

Compararemos tres enfoques algorítmicos con fundamentos matemáticos distintos:
1. **K-Means:** Algoritmo basado en prototipos y distancias euclidianas. Minimiza la inercia (Suma de Errores Cuadráticos Intra-cluster, WCSS), asumiendo clusters de geometría esférica e igual varianza.
2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Agrupamiento basado en densidad local. Identifica clusters de formas arbitrarias a través de parámetros de proximidad física ($\epsilon$ y *MinPts*) y tiene la capacidad clave de aislar puntos atípicos como ruido.
3. **GMM (Gaussian Mixture Model):** Enfoque probabilístico de clustering suave (*soft-clustering*). Asume que los datos siguen una mezcla latente de distribuciones gaussianas multivariadas y estima las probabilidades de pertenencia mediante el algoritmo de Esperanza-Maximización (EM).

### Arquitectura Analítica en Fases:
1. **Feature Engineering Transaccional:** Creación de variables sintéticas que capturen patrones del ticket (gasto, volumen, diversidad y dispersión de precios).
2. **Tratamiento y Limpieza de Outliers (IQR):** Filtrado de transacciones atípicas para estabilizar los centroides y evitar distorsiones en los algoritmos de distancia.
3. **Estandarización de Escalas:** Escalado z-score para mitigar el sesgo de magnitud de las variables.
4. **Evaluación de Modelos:** Comparación empírica multimodelo usando índices estructurales (Coeficiente de Silueta, Calinski-Harabasz y Davies-Bouldin).
5. **Perfilamiento y Estrategia Comercial:** Traducción de centroides matemáticos en perfiles de clientes y planes de acción específicos para la tienda.

`

## Cell 1 (code)
`python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

import scipy.stats as stats
import warnings
import os
import time

warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.figsize'] = (11, 5)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
`

## Cell 2 (markdown)
`python
## 1. Carga de Datos

En esta etapa inicial, importamos los conjuntos de datos saneados procedentes de la etapa de preparación (EDA). Estos corresponden a las transacciones de venta, el detalle de artículos de cada ticket y la maestra de inventarios.

`

## Cell 3 (code)
`python
DIR_LIMPIO = 'datasets/limpio/'
df_ventas = pd.read_csv(os.path.join(DIR_LIMPIO, 'ventas.csv'))
df_detalle = pd.read_csv(os.path.join(DIR_LIMPIO, 'detalle_ventas.csv'))
df_inventario = pd.read_csv(os.path.join(DIR_LIMPIO, 'inventario.csv'))

df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'])
df_detalle['Fecha'] = pd.to_datetime(df_detalle['Fecha'])

print(f"Ventas shape: {df_ventas.shape}")
print(f"Detalles shape: {df_detalle.shape}")
print(f"Inventario shape: {df_inventario.shape}\n")
display(df_ventas.head(3))
`

## Cell 4 (markdown)
`python
## 2. Feature Engineering a Nivel de Ticket

Para modelar adecuadamente el comportamiento de compra en ausencia de perfiles de clientes históricos, enriquecemos las transacciones agregando el detalle de cada ticket. Construimos las siguientes variables transaccionales de alto valor:

*   **`total_monto`:** El valor monetario total del ticket (indicador de la magnitud de la compra).
*   **`n_items`:** Cantidad de unidades físicas adquiridas en la transacción.
*   **`diversidad_productos`:** Cantidad de productos únicos (SKUs) en el ticket.
*   **`ticket_promedio_item`:** Costo promedio de los artículos en la transacción ($\frac{\text{total\_monto}}{\text{n\_items}}$).
*   **`ratio_diversidad`:** Medida de homogeneidad de la compra ($\frac{\text{diversidad\_productos}}{\text{n\_items}}$). Un valor cercano a 1 indica compras sumamente variadas; valores bajos representan compras de un mismo producto en volumen.
*   **`n_departamentos`:** Número de departamentos del negocio tocados en la transacción.
*   **`max_subtotal`:** Precio del artículo más caro comprado en el ticket.
*   **`std_subtotal`:** Desviación estándar de los subtotales en el ticket (indica si los productos comprados tienen precios similares o muy heterogéneos).

`

## Cell 5 (code)
`python
# Agrupación a nivel de ticket
ticket_features = df_detalle.groupby('ID_Venta').agg(
    n_items=('Cantidad', 'sum'),
    diversidad_productos=('ID_Producto', 'nunique'),
    n_departamentos=('Departamento', 'nunique'),
    max_subtotal=('Subtotal', 'max'),
    std_subtotal=('Subtotal', 'std')
).reset_index()

# Completar NaNs con 0 para tickets de un solo ítem
ticket_features['std_subtotal'] = ticket_features['std_subtotal'].fillna(0)

# Merge con ventas
df_cluster = pd.merge(df_ventas[['ID', 'Metodo_Pago', 'Total']], ticket_features, left_on='ID', right_on='ID_Venta', how='inner')
df_cluster.rename(columns={'Total': 'total_monto'}, inplace=True)

# Features adicionales
df_cluster['ticket_promedio_item'] = df_cluster['total_monto'] / df_cluster['n_items']
df_cluster['ratio_diversidad'] = df_cluster['diversidad_productos'] / df_cluster['n_items']

features_numericas = ['total_monto', 'n_items', 'diversidad_productos', 'ticket_promedio_item', 
                      'ratio_diversidad', 'n_departamentos', 'max_subtotal', 'std_subtotal']

display(df_cluster[features_numericas].describe())

# Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df_cluster[features_numericas].corr(), annot=True, cmap='viridis', fmt=".2f", square=True)
plt.title('Matriz de Correlación de Variables Transaccionales')
plt.tight_layout()
plt.show()
`

## Cell 6 (code)
`python
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(features_numericas):
    sns.histplot(df_cluster[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribución de {col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frecuencia')
    
plt.tight_layout()
plt.show()
`

## Cell 7 (markdown)
`python
## 3. Detección de Valores Atípicos (Outliers)

Para la estabilidad de los algoritmos de agrupamiento, en especial **K-Means** (que es altamente sensible a puntos extremos al calcular la media de los centroides), implementamos una limpieza rigurosa basada en el método del Rango Intercuartil (IQR):

$$ \text{Rango Intercuartil (IQR)} = Q_3 - Q_1 $$

Definimos los límites de aceptación para las variables críticas `total_monto` y `n_items` mediante:

$$ \text{Límite Superior} = Q_3 + 1.5 \times \text{IQR} $$
$$ \text{Límite Inferior} = Q_1 - 1.5 \times \text{IQR} $$

**Decisión Analítica:** Las observaciones que excedan estos límites serán excluidas del entrenamiento (limpieza de outliers) para evitar que sesguen la posición de los centroides. Esto nos permitirá segmentar con precisión el comportamiento transaccional cotidiano del negocio, aislando compras masivas o mayoristas excepcionales.

`

## Cell 8 (code)
`python
def get_iqr_bounds(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

lower_monto, upper_monto = get_iqr_bounds(df_cluster, 'total_monto')
lower_items, upper_items = get_iqr_bounds(df_cluster, 'n_items')

mask_normal = (
    (df_cluster['total_monto'] <= upper_monto) & 
    (df_cluster['n_items'] <= upper_items)
)

df_normal = df_cluster[mask_normal].copy()
df_mayoristas = df_cluster[~mask_normal].copy()

print(f"Total tickets originales: {len(df_cluster)}")
print(f"Tickets Normales (Para Clustering): {len(df_normal)} ({(len(df_normal)/len(df_cluster))*100:.2f}%)")
print(f"Tickets Mayoristas/Outliers (Aislados): {len(df_mayoristas)} ({(len(df_mayoristas)/len(df_cluster))*100:.2f}%)\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(x=df_normal['total_monto'], ax=axes[0], color='lightgreen')
axes[0].set_title('Boxplot Limpio: total_monto')

sns.boxplot(x=df_normal['n_items'], ax=axes[1], color='lightsalmon')
axes[1].set_title('Boxplot Limpio: n_items')

plt.tight_layout()
plt.show()
`

## Cell 9 (markdown)
`python
## 4. Estandarización de Variables

Dado que nuestros descriptores transaccionales operan en escalas radicalmente distintas (por ejemplo, el monto monetario frente a la cantidad de departamentos o el ratio de diversidad), es indispensable estandarizar los datos. Los algoritmos como K-Means y DBSCAN utilizan distancias euclidianas, por lo que variables con mayores magnitudes dominarían el cálculo del agrupamiento.

Aplicamos la transformación Z-Score mediante `StandardScaler`:

$$ Z = \frac{X - \mu}{\sigma} $$

Donde $\mu$ es la media y $\sigma$ es la desviación estándar de cada variable. Tras esta transformación, todas las características tendrán media 0 y varianza 1, garantizando que cada métrica aporte de manera equitativa a la estructura de distancia del espacio multidimensional.

`

## Cell 10 (code)
`python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_normal[features_numericas])

df_scaled = pd.DataFrame(X_scaled, columns=features_numericas)

print("Medias (aproximadas a 0):")
print(df_scaled.mean().round(4))
print("\nDesviaciones Estándar (aproximadas a 1):")
print(df_scaled.std().round(4))
`

## Cell 11 (markdown)
`python
## 5. Modelo 1 - K-Means

K-Means particiona el espacio de datos en $K$ clusters disjuntos, asignando cada observación a su centroide más cercano. El algoritmo optimiza de manera iterativa la inercia interna (Within-Cluster Sum of Squares, WCSS):

$$ \text{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2 $$

Determinaremos el número óptimo de agrupaciones ($K$) mediante un análisis combinado del **Método del Codo (Elbow Method)** y el **Coeficiente de Silueta**. Este último evalúa la cohesión intra-cluster y la separación inter-cluster:

$$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$

`

## Cell 12 (code)
`python
k_values = range(2, 11)
inertia_scores = []
silhouette_scores = []
calinski_scores = []
davies_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertia_scores.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
    davies_scores.append(davies_bouldin_score(X_scaled, labels))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0,0].plot(k_values, inertia_scores, marker='o', color='b')
axes[0,0].set_title('Método del Codo (Inertia)')

axes[0,1].plot(k_values, silhouette_scores, marker='o', color='g')
axes[0,1].set_title('Silhouette Score')

axes[1,0].plot(k_values, calinski_scores, marker='o', color='r')
axes[1,0].set_title('Calinski-Harabasz Score')

axes[1,1].plot(k_values, davies_scores, marker='o', color='purple')
axes[1,1].set_title('Davies-Bouldin Score')

for ax in axes.flatten():
    ax.set_xticks(k_values)
    ax.grid(True)

plt.tight_layout()
plt.show()
`

## Cell 13 (code)
`python
# Configurar K óptimo (basado en Silhouette y Codo)
K_OPTIMO = 3

kmeans_final = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_scaled)
df_normal['Cluster_KMeans'] = labels_kmeans

sil_score_kmeans = silhouette_score(X_scaled, labels_kmeans)
print(f"Silhouette Score final para K-Means (K={K_OPTIMO}): {sil_score_kmeans:.4f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_monto', y='n_items', hue='Cluster_KMeans', data=df_normal, palette='Set2', s=60, alpha=0.8)
plt.title('Dispersión: Monto Total vs Cantidad de Items (K-Means)')
plt.tight_layout()
plt.show()
`

## Cell 14 (markdown)
`python
## 6. Modelo 2 - DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clasifica los datos según la densidad local de su entorno. Sus parámetros fundamentales son:
*   **`eps` ($\epsilon$):** El radio de vecindad máximo para considerar que dos puntos son cercanos.
*   **`min_samples`:** La cantidad mínima de puntos necesarios dentro de la vecindad para clasificar un punto como "núcleo" (*core point*).

**Ventaja Analítica:** DBSCAN no impone una forma esférica a los clusters y es robusto contra el ruido, catalogando automáticamente las observaciones de baja densidad local como atípicas (etiqueta `-1`).

`

## Cell 15 (code)
`python
# Determinar EPS óptimo con KNN
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, 9], axis=0)

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.title('Gráfico de K-Distancia (k=10)')
plt.xlabel('Puntos ordenados')
plt.ylabel('Distancia al k-ésimo vecino')
plt.grid(True)
plt.tight_layout()
plt.show()

# Búsqueda de cuadrícula (Grid Search)
eps_values = [1.5, 2.0, 2.5, 3.0]
min_samples_values = [5, 10, 15]

best_eps, best_min = None, None
best_sil = -1

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        if len(set(labels)) > 1 and len(set(labels)) < len(X_scaled) / 2:
            # Ignorar ruido (-1) en silhouette
            mask = labels != -1
            if np.sum(mask) > 0 and len(set(labels[mask])) > 1:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                if sil > best_sil:
                    best_sil = sil
                    best_eps, best_min = eps, min_samples

print(f"Mejores parámetros para DBSCAN: eps={best_eps}, min_samples={best_min} (Silhouette {best_sil:.4f})")
`

## Cell 16 (code)
`python
# Entrenar modelo final DBSCAN
dbscan_final = DBSCAN(eps=best_eps if best_eps else 2.5, min_samples=best_min if best_min else 10)
labels_dbscan = dbscan_final.fit_predict(X_scaled)
df_normal['Cluster_DBSCAN'] = labels_dbscan

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_db = list(labels_dbscan).count(-1)
print(f"Clusters encontrados (sin ruido): {n_clusters_db}")
print(f"Porcentaje de puntos clasificados como ruido: {(n_noise_db / len(labels_dbscan))*100:.2f}%")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_monto', y='n_items', hue='Cluster_DBSCAN', data=df_normal, palette='tab10', s=60, alpha=0.8)
plt.title('Dispersión DBSCAN (Ruido en Gris si el color es -1)')
plt.tight_layout()
plt.show()
`

## Cell 17 (markdown)
`python
## 7. Modelo 3 - Gaussian Mixture Model (GMM)

GMM es un modelo probabilístico de clustering suave (*soft-clustering*) que asume que el conjunto de datos se origina a partir de una combinación lineal de $N$ distribuciones normales multivariadas independientes. En lugar de una asignación binaria rígida, estima para cada ticket una probabilidad de pertenencia a cada grupo.

El entrenamiento se ejecuta mediante el algoritmo de Maximización de la Esperanza (EM). Evaluamos el número óptimo de componentes gaussianas mediante criterios de información que penalizan la complejidad del modelo:
*   **BIC (Criterio de Información Bayesiano)**
*   **AIC (Criterio de Información de Akaike)**

`

## Cell 18 (code)
`python
n_components = range(2, 11)
bic_scores = []
aic_scores = []

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

plt.figure(figsize=(10, 6))
plt.plot(n_components, bic_scores, marker='o', label='BIC')
plt.plot(n_components, aic_scores, marker='s', label='AIC')
plt.title('BIC y AIC para GMM')
plt.xlabel('Número de componentes')
plt.ylabel('Puntuación de Información')
plt.legend()
plt.tight_layout()
plt.show()
`

## Cell 19 (code)
`python
# Elegimos N=3 basado en resultados previos y regularización del codo
N_GMM = 3
gmm_final = GaussianMixture(n_components=N_GMM, random_state=42)
labels_gmm = gmm_final.fit_predict(X_scaled)
df_normal['Cluster_GMM'] = labels_gmm

prob_gmm = gmm_final.predict_proba(X_scaled)
max_prob_gmm = prob_gmm.max(axis=1)

sil_score_gmm = silhouette_score(X_scaled, labels_gmm)
print(f"Silhouette Score final para GMM: {sil_score_gmm:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(x='total_monto', y='n_items', hue='Cluster_GMM', data=df_normal, palette='viridis', s=60, alpha=0.8, ax=axes[0])
axes[0].set_title('Dispersión GMM')

sns.histplot(max_prob_gmm, bins=20, kde=True, color='teal', ax=axes[1])
axes[1].set_title('Confianza de asignación de clústeres (Probabilidad Máxima)')

plt.tight_layout()
plt.show()
`

## Cell 20 (markdown)
`python
## 8. Comparación de Modelos y Selección de Algoritmo

Para garantizar el rigor del pipeline analítico, comparamos cuantitativamente los tres modelos entrenados mediante métricas intrínsecas:
1.  **Coeficiente de Silueta:** Medida de qué tan similar es un objeto a su propio cluster en comparación con otros clusters.
2.  **Índice Calinski-Harabasz:** Ratio de la suma de dispersión entre clusters y la dispersión intra-cluster (a mayor valor, mejor estructura).
3.  **Índice Davies-Bouldin:** Similitud promedio de cada cluster con su cluster más similar (a menor valor, mejor separación).

`

## Cell 21 (code)
`python
resultados = {
    'Modelo': ['K-Means', 'DBSCAN', 'GMM'],
    'Silueta': [silhouette_score(X_scaled, labels_kmeans), 
                silhouette_score(X_scaled, labels_dbscan) if n_clusters_db > 1 else 0,
                silhouette_score(X_scaled, labels_gmm)],
    'Calinski_Harabasz': [calinski_harabasz_score(X_scaled, labels_kmeans), 
                          calinski_harabasz_score(X_scaled, labels_dbscan) if n_clusters_db > 1 else 0,
                          calinski_harabasz_score(X_scaled, labels_gmm)],
    'Davies_Bouldin': [davies_bouldin_score(X_scaled, labels_kmeans),
                       davies_bouldin_score(X_scaled, labels_dbscan) if n_clusters_db > 1 else 0,
                       davies_bouldin_score(X_scaled, labels_gmm)],
    'N_Clusters': [K_OPTIMO, n_clusters_db, N_GMM]
}

df_comparacion = pd.DataFrame(resultados)
display(df_comparacion.style.highlight_max(subset=['Silueta', 'Calinski_Harabasz'], color='lightgreen')\
        .highlight_min(subset=['Davies_Bouldin'], color='lightgreen'))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(x='Modelo', y='Silueta', data=df_comparacion, ax=axes[0], palette='pastel')
axes[0].set_title('Puntuación Silueta (Mayor es mejor)')

sns.barplot(x='Modelo', y='Calinski_Harabasz', data=df_comparacion, ax=axes[1], palette='pastel')
axes[1].set_title('Calinski-Harabasz (Mayor es mejor)')

sns.barplot(x='Modelo', y='Davies_Bouldin', data=df_comparacion, ax=axes[2], palette='pastel')
axes[2].set_title('Davies-Bouldin (Menor es mejor)')

plt.tight_layout()
plt.show()
`

## Cell 22 (markdown)
`python
### Selección del Mejor Modelo

Tras contrastar los índices de rendimiento estructural y la coherencia del negocio, seleccionamos el modelo de **K-Means con $K=3$**. 

#### Justificación Técnica:
*   **Estabilidad y Métricas:** K-Means con $K=3$ ofrece el Coeficiente de Silueta y el Índice Calinski-Harabasz más robustos del pipeline.
*   **Interpretabilidad Comercial:** Los tres grupos resultantes representan segmentos transaccionales claramente diferenciados y accionables, lo que facilita el desarrollo de estrategias y la integración del modelo de cara al usuario final en el dashboard de Streamlit.

`

## Cell 23 (markdown)
`python
## 9. Perfilamiento de Clusters y Revelaciones del Modelo

En esta sección, procedemos a desglosar y caracterizar matemáticamente los tres grupos definitivos obtenidos por el modelo de K-Means. A través de este análisis de centroides, transformaremos las métricas transaccionales puras en conocimiento de negocio aplicable.

`

## Cell 24 (code)
`python
# Analizar usando el modelo ganador K-Means
df_normal['Cluster_Final'] = df_normal['Cluster_KMeans']

perfil_cluster = df_normal.groupby('Cluster_Final')[features_numericas].mean().round(2)
display(perfil_cluster)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(x='Cluster_Final', data=df_normal, palette='Set2', ax=axes[0])
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}\n({p.get_height()/len(df_normal)*100:.1f}%)', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 10), textcoords='offset points')
axes[0].set_title('Volumen de Tickets por Cluster')

sns.histplot(data=df_normal, x='Metodo_Pago', hue='Cluster_Final', multiple='stack', palette='Set2', ax=axes[1])
axes[1].set_title('Distribución de Métodos de Pago por Cluster')

plt.tight_layout()
plt.show()
`

## Cell 25 (markdown)
`python
### Perfilamiento Avanzado de Clientes (Tickets)

El modelo matemático identificó con éxito tres comportamientos transaccionales independientes y consistentes. 

Al analizar los centroides de los clusters, destaca un hallazgo de enorme valor comercial: **la variable `n_departamentos` promedia exactamente 1.0 en todos los clusters**. Esto nos da una revelación contundente:


> **Comportamiento Transaccional Exclusivo:** Los clientes de la tienda **no cruzan departamentos en un mismo ticket**. Es decir, quien va a sacar copias no compra útiles escolares en la misma transacción, y viceversa. Existe una barrera operativa o conductual que separa los "Servicios" de los "Productos".

---

#### Caracterización Detallada de los Clusters:

1.  **Cluster 1 (19.8% de los tickets) - "Servicio de Fotocopiadora" (Volumen de Servicios)**
    *   **Volumen de Items (`n_items`):** Alto (promedio de **6.55** unidades por ticket).
    *   **Ticket Promedio Monetario (`total_monto`):** Extremadamente bajo (promedio de **S/. 2.47** por ticket).
    *   **Diversidad de Productos (`ratio_diversidad`):** Muy baja (promedio de **0.31**).
    *   **Gasto Promedio por Item (`ticket_promedio_item`):** Muy bajo (promedio de **S/. 0.47**).
    *   **Interpretación:** Este grupo representa fielmente a clientes que acuden al local exclusivamente por el servicio de copiado o impresión. El modelo agrupó matemáticamente estas transacciones porque muestran un patrón inconfundible: compran muchas unidades de un mismo servicio (de ahí la baja diversidad) haciendo que el costo unitario por ítem sea muy bajo (S/. 0.47), a pesar de que en la caja terminen pagando un promedio de S/. 2.47 en total.

2.  **Cluster 0 (66.1% de los tickets) - "Útiles Express / Cotidianos" (Baja Fricción)**
    *   **Volumen de Items (`n_items`):** Muy bajo (promedio de **1.28** unidades por ticket).
    *   **Diversidad de Productos (`ratio_diversidad`):** Muy alta (promedio de **0.92**).
    *   **Gasto Promedio por Item (`ticket_promedio_item`):** Moderado (promedio de **S/. 1.51**).
    *   **Interpretación:** Es el segmento mayoritario de la tienda. Representa compras cotidianas y de paso rápido: clientes que adquieren uno o dos productos escolares o de oficina específicos (lapicero, borrador, cuaderno) y de manera independiente (de ahí el elevado ratio de diversidad, ya que casi no repiten el mismo artículo en el ticket).

3.  **Cluster 2 (14.1% de los tickets) - "Útiles Premium / Alto Valor" (Rentabilidad)**
    *   **Volumen de Items (`n_items`):** Bajo (promedio de **1.50** unidades por ticket).
    *   **Ticket Promedio Monetario (`total_monto`):** El más alto de la tienda (promedio de **S/. 6.92**, alcanzando un máximo de **S/. 6.69** tras la remoción de outliers).
    *   **Interpretación:** Este grupo genera un alto margen para la tienda. Consiste en clientes que compran pocos productos (1 o 2 por ticket) pero de alto valor unitario (como calculadoras científicas, archivadores premium, mochilas, estuches de plumones finos, etc.).

`

## Cell 26 (markdown)
`python
## 10. Conclusiones y Acciones Comerciales

El análisis transaccional mediante K-Means ha revelado que la tienda opera de manera binaria (o bien compran útiles, o bien sacan copias, pero nunca ambas cosas en una misma visita de caja). Aprovechando este hallazgo y el perfilamiento de los tres grupos, diseñamos las siguientes estrategias de optimización para SmartBazar:

### Plan de Acción Estratégico

#### 1. Para "Fotocopiadora" (Cluster 1 - 19.8% de tickets)
* **Reto:** Alto volumen de atención y tráfico recurrente, pero anclado a un solo departamento (Fotocopiadora) con un ticket promedio bajo (S/. 2.47).
* **Estrategia de Escalamiento:** Aumentar agresivamente la captación de clientes para este segmento (ej. promociones para estudiantes o convenios con negocios locales) para multiplicar el volumen diario de tickets.
* **Estrategia de Venta Cruzada Inter-Departamental:** El objetivo clave es lograr que estos clientes adquieran productos de otro departamento de la tienda, elevando su ticket promedio. Para ello, se colocarán físicamente consumibles económicos (folders de plástico de S/. 0.50, clips, micas, grapas o sobres) en la misma área de fotocopiado. El personal aplicará la venta sugestiva sistemática: *"¿Desea un folder o mica para guardar sus copias?"*.
* **Fidelización:** Implementar un sistema de recompensas (ej. "Saca 100 copias, las siguientes 10 son gratis") para asegurar la recurrencia y dominar la demanda de la zona.

#### 2. Para "Útiles Express / Cotidianos" (Cluster 0 - 66.1% de tickets)
* **Reto:** Son la gran mayoría de transacciones. Se debe capitalizar este altísimo tráfico elevando ligeramente el gasto sin generar cuellos de botella en la atención.
* **Agilidad Extrema en Caja:** Mantener la velocidad de flujo promocionando activamente cobros mediante códigos QR (Yape / Plin) y optimizando el espacio en el mostrador.
* **Estrategia de Compra por Impulso (Upselling rápido):** Aprovechando el momento de pago, colocar productos atractivos de bajo costo directamente en la zona de caja (ej. post-its llamativos, resaltadores en promoción, borradores especiales). El objetivo es que el cliente agregue un ítem extra en el último segundo, subiendo su ticket de S/. 1.81 a S/. 2.50 sin fricción.

#### 3. Para "Útiles Premium / Alto Valor" (Cluster 2 - 14.1% de tickets)
* **Reto:** Segmento de alta rentabilidad y margen, concentrado en la venta de muy pocos artículos por cliente.
* **Estrategia de Exhibición Visual Caliente:** Los productos de alto valor (calculadoras, mochilas, archivadores premium) deben tener una exhibición destacada y aspiracional. Deben estar ubicados a la altura de los ojos, bien iluminados y, de ser necesario, en vitrinas frontales.
* **Estrategia de Empaquetado (Kits Temáticos):** Agrupar productos de este cluster para crear kits de mayor valor percibido (ej. "Kit de Dibujo Técnico", "Kit Ejecutivo"). Ofrecer una ligera ventaja por llevar el kit completo versus los ítems por separado para incentivar la venta cruzada de alto valor.

---

### Exportación de Resultados

Con este perfilamiento robusto y orientado a la realidad del negocio, asignamos a cada transacción su correspondiente etiqueta de cluster y exportamos el dataset limpio para alimentar la visualización en tiempo real del **Panel 1A (Clustering)** en el Dashboard de SmartBazar.

`

## Cell 27 (code)
`python
ruta_salida = os.path.join(DIR_LIMPIO, 'ventas_clusterizadas.csv')
df_normal.to_csv(ruta_salida, index=False)
print(f"Archivo final exportado exitosamente a: {ruta_salida}")
`
