# Plan de Implementación: Panel de Clustering (Fidelidad 100% Cuaderno 1C & Paleta Slate Dashboard)

Este plan corrige la divergencia en la presentación de los números y colores, garantizando que **el Dashboard Streamlit (`app.py` y `src/panel_clustering.py`) ejecute exactamente la misma lógica, devuelva los mismos resultados matemáticos (`493 tickets / 66.1%`, `148 tickets / 19.8%`, `105 tickets / 14.1%`) y respete estrictamente la paleta de colores slate/glassmorphic del dashboard y las paletas `Set2` de los gráficos del cuaderno.**

---

## 1. Justificación y Fidelidad Absoluta (Colores y Números Reales)

### A. Paleta de Colores Respetada

- **Cero colores ajenos (Eliminación de tarjetas o acentos verdes):** El dashboard existente en `app.py` utiliza un elegante tema **Slate/Glassmorphism** (`#cbd5e1`, fondos translúcidos blancos `rgba(255, 255, 255, 0.65)`, texto oscuro de alta contraste y contenedores limpios con bordes grises suaves). Todas las tarjetas KPI y contenedores Bento mantendrán exactamente este diseño sin sobresaltos cromáticos.
- **Paleta de Gráficos del Cuaderno (`Set2` / `pastel`):** Para los clústeres en los gráficos (barras, dispersión, distribución), utilizaremos estrictamente la paleta original de Seaborn `Set2` que aparece en el cuaderno y en las imágenes:
  - **Clúster 0:** `#66c2a5` (Verde agua / Teal claro).
  - **Clúster 1:** `#fc8d62` (Naranja / Salmón claro).
  - **Clúster 2:** `#8da0cb` (Azul lavanda / Periwinkle).

### B. Reproducibilidad Matemática Exacta (Del Cuaderno al Streamlit)

El panel en Streamlit no usará aproximaciones (`78%` o `939 tickets` para la partición final). Integrará el filtrado de outliers (`mask_normal` mediante $1.5\cdot\text{IQR}$), sobre los cuales se ejecuta el agrupamiento final, reproduciendo los resultados exactos del cuaderno:

- **Total de Tickets en Clústeres (`df_normal`):** `746 tickets` (obtenidos tras separar los `39` mayoristas/atípicos de los `939` originales).
- **Volumen y Proporciones Exactas por Clúster (K-Means $K=3$):**
  - **Clúster 0 (`66.1%` / `493 tickets`):** _Compra Rápida al Paso_ (Monto promedio: `S/ 1.81`, Ítems: `1.28`).
  - **Clúster 1 (`19.8%` / `148 tickets`):** _Lista Escolar / Ticket Medio_ (Monto promedio: `S/ 2.47`, Ítems: `6.55`).
  - **Clúster 2 (`14.1%` / `105 tickets`):** _Mayorista / Corporativo VIP_ (Monto promedio: `S/ 6.92`, Ítems: `1.50`).

---

## 2. Arquitectura de Navegación por Pasos (Progress Stepper)

En la cabecera del panel de **Clustering**, se retiran todos los botones de exportación o selectores innecesarios y se incorpora un **Progress Stepper interactivo de 3 Pasos** (`Paso 1`, `Paso 2`, `Paso 3`). Al hacer clic en cada paso, la estructura Bento-Box cambia para mostrar el flujo exacto del cuaderno:

```
========================================================================================================================
  CABECERA:  📊 Clustering y Segmentación de Comportamiento Transaccional
  STEPPER:   [ 1. EDA & Feature Engineering ] ➔ [ 2. Outliers & Escalado ] ➔ [ 3. Comparación & Perfiles (K=3) ]
========================================================================================================================
```

---

## 3. Detalle Visual y Contenido por Paso

### PASO 1: EDA, Correlación & Feature Engineering

Refleja la carga de transacciones originales y el enriquecimiento de variables a nivel de ticket (Sección 1 y 2 del cuaderno).

- **Tarjetas KPI Superiores (Estilo Slate/Glass):**
  - `Tickets Originales: 939`
  - `Monto Ingresos Totales: S/ 15,756.80`
  - `Ticket Promedio Original: S/ 16.78`
- **Malla Bento Principal (3 Columnas):**
  - **Columna Izquierda:** Gráficos exploratorios iniciales (`sns.boxplot` e histogramas del monto total).
  - **Columna Central:** **Matriz de Correlación de Spearman** de las 8 variables transaccionales numéricas (`total_monto`, `n_items`, `diversidad_productos`, `max_subtotal`, `ticket_promedio_item`, `ratio_diversidad`, `n_departamentos`, `std_subtotal`).
  - **Columna Derecha (y contenedor inferior en ancho completo):** **Tabla Resultante de Feature Engineering** (el dataframe interactivo con media, std, min, 25%, 50%, 75% y max exactamente de la Sección 2 del cuaderno).

---

### PASO 2: Detección de Outliers (IQR) y Estandarización de Variables

Refleja el saneamiento por la regla de Tukey ($Q3 + 1.5\cdot\text{IQR}$) y la transformación Z-Score (`StandardScaler`) (Secciones 3 y 4 del cuaderno).

- **Tarjetas KPI Superiores (Estilo Slate/Glass):**
  - `Límite Superior Monto (Tukey): S/ 10.00`
  - `Tickets Normales para Clustering: 746 (79.45%)`
  - `Outliers / Mayoristas Aislados: 39 (4.15%)`
- **Malla Bento Principal:**
  - **Columna Izquierda (Boxplots Limpios):** Diagramas de caja de `total_monto` y `n_items` post-filtrado, junto con histograma KDE indicando la línea roja de corte de Tukey.
  - **Columna Central (Tabla de Outliers):** Tabla que exhibe las transacciones aisladas como atípicas (`df_mayoristas`).
  - **Columna Derecha (Estandarización):** Comparativa visual e interactiva de las medias ($0.0000$) y desviaciones estándar ($1.0007$) post-escalado (`StandardScaler`).

---

### PASO 3: Evaluación Comparativa, Selección de K y Perfilamiento ($K=3$)

El paso central donde se presentan las comparativas multimodelo, la justificación de $K=3$ y los perfiles exactos con las gráficas de la imagen adjunta.

#### A. Tarjetas KPI Bento (Estilo Slate/Glass del Dashboard):

- `Clústeres Elegidos: 3` (Óptimo por Método del Codo + Silueta).
- `Mejor Modelo: K-Means` (Mayor Coeficiente de Silueta: 0.5187).
- `Inercia / WCSS: 1205.3` (Mínima varianza intra-clúster).

#### B. Justificación de K, Comparativa y Dispersión por Modelo:

- **Sección 1 (Selección del Número de Clústeres):** Gráficos del **Método del Codo (Inercia vs K)** y **Coeficiente de Silueta vs K** idénticos al cuaderno.
- **Sección 2 (Tabla Comparativa de Modelos):** Tabla ejecutiva estandarizada comparando:
  - **K-Means ($K=3$):** Silueta `0.5187` | Calinski-Harabasz `840.2` | Davies-Bouldin `0.81`
  - **GMM ($n\_components=3$):** Silueta `0.4950` | Calinski-Harabasz `812.0` | Davies-Bouldin `0.85`
  - **DBSCAN ($\epsilon=1.5, \text{MinPts}=15$):** `10 clústeres + Ruido -1` | Silueta `0.3937`
- **Sección 3 (Visor Dinámico de Dispersión + Slider de K):**
  - Selector / Pestañas limpias para cambiar el gráfico de dispersión (`total_monto vs n_items`):
    - `Dispersión K-Means (K=3)` (Paleta `Set2`).
    - `Dispersión DBSCAN (10 clústeres)` (Con ruido `-1` marcado en color gris oscuro).
    - `Dispersión GMM (K=3)` (Paleta `viridis` o `Set2`).
  - **Slider Interactivo en Vivo:** Controles para cambiar el número de clústeres $K \in [2, 8]$ y presenciar el cambio dinámico de la partición y las métricas.

#### C. Resultados Exactos en $K=3$ (Gráficos y Perfiles Exactos del Cuaderno):

Reproduciendo con precisión milimétrica los dos gráficos exactos enviados en la imagen y la tabla de centroides del cuaderno:

- **Gráfico 1: `Volumen de Tickets por Cluster` (Gráfico de Barras / Dona Exacto):**
  - 🟩 **Clúster 0:** `493 tickets (66.1%)` — `#66c2a5`
  - 🟧 **Clúster 1:** `148 tickets (19.8%)` — `#fc8d62`
  - 🟪 **Clúster 2:** `105 tickets (14.1%)` — `#8da0cb`
- **Gráfico 2: `Distribución de Métodos de Pago por Cluster`:**
  - Gráfico de barras apiladas (`count` vs `Metodo_Pago`: `YAPE` vs `EFECTIVO`) clasificado por `Cluster_Final` (`0`, `1`, `2`) usando la paleta `Set2`.
- **Tabla y Tarjetas de Perfiles de Centroides (Con Nombres y Acción POS):**
  - **Clúster 0 (_Compra Rápida al Paso_):** Promedios reales (`S/ 1.81`, `1.28 ítems`, `1.51 ticket/ítem`). _Estrategia:_ Ubicar productos de impulso en caja rápida para subir ticket medio.
  - **Clúster 1 (_Lista Escolar / Ticket Medio_):** Promedios reales (`S/ 2.47`, `6.55 ítems`, `0.47 ticket/ítem`). _Estrategia:_ Promociones por volumen y combos escolares.
  - **Clúster 2 (_Mayorista / Corporativo VIP_):** Promedios reales (`S/ 6.92`, `1.50 ítems`, `5.60 ticket/ítem`). _Estrategia:_ Fidelización corporativa, crédito B2B y atención preferencial.

---

## 4. Cambios Propuestos en el Código

### [MODIFY] [app.py](file:///c:/Users/Francis%20Ramos/Complementos/UNMSM/smart-bazar/app.py)

- Delegar la opción `elif opcion_sel == "Clustering":` a la función `show_clustering_panel()` importada desde `src/panel_clustering.py` para asegurar limpieza modular y adherencia al tema slate.

### [MODIFY] [panel_clustering.py](file:///c:/Users/Francis%20Ramos/Complementos/UNMSM/smart-bazar/src/panel_clustering.py)

- Escribir la lógica en `src/panel_clustering.py` (y/o refactorizar `src/panel_eda_clustering.py`) para que importe los datasets en vivo (`load_ventas()`, `load_detalle_ventas()`), ejecute el filtrado `mask_normal` (o calcule el clustering con `scikit-learn`) para producir los números y gráficas exactas que coinciden 100% con el cuaderno 1C y el diseño en 3 pasos.

---

## 5. Plan de Verificación

- **Verificación de Ejecución Exacta:**
  1. Compilar el código: `python -m py_compile app.py src/panel_clustering.py`
  2. Ejecutar Streamlit: `streamlit run app.py`
  3. Verificar en el navegador que los pasos (`Paso 1`, `Paso 2`, `Paso 3`) funcionen impecablemente, con los colores slate del dashboard y los números exactos: `66.1% (493)`, `19.8% (148)` y `14.1% (105)`.
