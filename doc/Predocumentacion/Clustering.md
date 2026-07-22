# Vista Clustering: segmentación transaccional no supervisada

## 1. Objetivo y procedencia

La vista agrupa **tickets**, no clientes, porque no utiliza un identificador confiable de comprador recurrente. El objetivo es descubrir patrones de composición y valor sin una variable objetivo conocida. Los números de clúster (0, 1, 2) son etiquetas técnicas arbitrarias; no tienen significado ordinal.

- **En vivo:** agregado, escalado o modelo ajustado durante la ejecución, sujeto a `st.cache_data`.
- **Precalculado:** CSV limpio producido antes de abrir la vista.
- **Preentrenado:** modelo cargado desde disco. **No existe aquí:** K-Means, DBSCAN y GMM se ajustan en runtime.
- **Hardcodeado:** métrica, perfil o conclusión escrita literalmente en el módulo.

## 2. Ruta activa y mapa de dependencias

`app.py` importa `show_clustering_panel` como `show_clustering` y lo llama cuando el radio vale `Clustering` ([`app.py`, líneas 33 y 498-499](../app.py)).

```text
app.py
  -> show_clustering_panel()
       -> get_processed_data()
            -> data_loader.load_ventas()
            -> data_loader.load_detalle_ventas()
            -> feature engineering -> Tukey -> StandardScaler -> KMeans(K=3)
       -> get_model_evaluations(X_scaled)
            -> tabla K hardcodeada -> DBSCAN -> GMM
       -> tres pestañas de resultados
```

| Archivo/artefacto | Símbolos relevantes | Papel real |
|---|---|---|
| [`src/panel_clustering.py`](../src/panel_clustering.py) | `get_processed_data`, `get_model_evaluations`, `show_clustering_panel` | Ruta activa, cálculos y renderizado. |
| [`src/data_loader.py`](../src/data_loader.py) | `load_ventas`, `load_detalle_ventas`, `get_datasets_path` | Prefiere los CSV limpios; solo cae al crudo si no encuentra el archivo limpio. |
| [`Panel 1C`](../Notebooks/Panel%201C_Clustering.ipynb) | celdas 3-27 | Desarrollo offline de referencia: evaluación real de K, búsqueda DBSCAN, AIC/BIC, comparación y exportación. No se ejecuta desde Streamlit. |
| `datasets/limpio/ventas.csv` y `detalle_ventas.csv` | cabecera y líneas de ticket | Insumos precalculados actuales. `inventario.csv` se carga en el notebook, pero no influye en el runtime. |
| `datasets/limpio/ventas_clusterizadas.csv` | exportación del notebook | No se lee en esta vista; las etiquetas se recalculan. |
| [`src/panel_eda_clustering.py`](../src/panel_eda_clustering.py) | `show_panel` | Implementación anterior/inactiva. No está importada por `app.py`; sus controles y resultados no afectan esta vista. |

Las funciones principales usan `@st.cache_data`. En el primer cálculo ajustan modelos; reruns con los mismos argumentos reutilizan el resultado serializado. El slider no es argumento de esas funciones, por lo que solo su K-Means adicional se recalcula para la gráfica.

## 3. Construcción del vector de ocho características

`get_processed_data()` convierte fechas, pero luego no usa ninguna variable temporal. Agrupa `detalle_ventas` por `ID_Venta` y realiza un `inner merge` con `ventas.ID`: tickets sin correspondencia en cualquiera de las tablas desaparecen.

| Variable | Definición exacta | Interpretación |
|---|---|---|
| `total_monto` | `ventas.Total` renombrado | valor declarado en cabecera |
| `n_items` | suma de `Cantidad` | unidades compradas, no número de líneas |
| `diversidad_productos` | `nunique(ID_Producto)` | variedad de SKU distintos |
| `ticket_promedio_item` | `total_monto / n_items` | monto medio por unidad |
| `ratio_diversidad` | `diversidad_productos / n_items` | cercanía a una unidad por SKU; puede superar supuestos si los datos son inconsistentes |
| `n_departamentos` | `nunique(Departamento)` | amplitud departamental del ticket |
| `max_subtotal` | máximo `Subtotal` de sus líneas | línea de mayor valor |
| `std_subtotal` | desviación estándar muestral de subtotales | heterogeneidad de valores dentro del ticket; se imputa 0 cuando hay una sola línea |

No hay guardia para `n_items = 0`: las dos divisiones pueden producir infinito o `NaN`, y `StandardScaler`/los modelos pueden fallar. Un `ID` duplicado puede multiplicar filas al unir. Las fechas inválidas se convierten en `NaT` sin efecto porque no se usan.

## 4. Outliers por regla de Tukey

Para monto e ítems se calculan por separado:

```text
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR
normal = (total_monto <= limite_monto) AND (n_items <= limite_items)
```

Solo se aplican límites superiores; no se filtran extremos inferiores. Un ticket que exceda cualquiera de los dos límites pasa a `df_mayoristas`. “Outlier” significa observación extrema bajo esta regla, no error ni cliente corporativo demostrado.

Con los CSV versionados al redactar este documento hay 939 tickets; los límites son S/ 10 y 16 ítems, quedan 746 normales y 193 aislados (20.55 %). La separación proviene de 58 tickets con monto mayor a 10, 171 con más de 16 ítems y 36 que cumplen ambas condiciones. Este snapshot permite detectar una discrepancia importante con los textos fijos de la vista.

## 5. Estandarización Z-Score

`StandardScaler.fit_transform()` se ajusta **solo con los tickets normales** y transforma las ocho columnas:

```text
z_ij = (x_ij - media_j) / desviacion_poblacional_j
```

Esto evita que soles, unidades y ratios dominen la distancia únicamente por su escala. Scikit-learn usa desviación poblacional (`ddof=0`); luego la tabla verifica con `DataFrame.std()` muestral (`ddof=1`), por eso muestra un valor ligeramente mayor que 1. Una variable constante se transforma a cero y no aporta distancia.

## 6. Algoritmos ejecutados

### K-Means

Se ajusta `KMeans(n_clusters=3, random_state=42, n_init=10)` sobre las ocho variables estandarizadas. La inicialización por defecto es K-Means++, se prueban diez inicializaciones y se conserva la de menor inercia. El algoritmo alterna:

1. asignar cada ticket al centroide euclidiano más cercano;
2. reemplazar cada centroide por la media de sus miembros;
3. repetir hasta convergencia.

Minimiza la inercia o WCSS:

```text
WCSS = sum_k sum_{x en C_k} ||x - centroide_k||²
```

Favorece grupos aproximadamente compactos/esféricos y es sensible a outliers; de ahí el filtro previo. `random_state=42` hace reproducible el resultado para el mismo dato y versión de librería, pero los identificadores 0/1/2 siguen siendo nominales.

### DBSCAN

`DBSCAN(eps=1.5, min_samples=15)` sí se ajusta en runtime. Un punto es núcleo si su vecindad de radio 1.5 en el espacio Z-Score de ocho dimensiones contiene al menos 15 observaciones; núcleos conectados forman clústeres, puntos alcanzables son borde y el resto recibe `-1` (ruido). No necesita fijar K y puede encontrar formas no esféricas, pero es sensible a escala, densidades variables y parámetros.

### Gaussian Mixture Model (GMM)

`GaussianMixture(n_components=3, random_state=42)` ajusta tres gaussianas con covarianza completa mediante Expectation-Maximization:

```text
p(x) = sum_k pi_k * Normal(x | mu_k, Sigma_k)
```

En E estima probabilidades posteriores; en M actualiza pesos, medias y covarianzas. `fit_predict()` muestra solo el componente de máxima probabilidad. Aunque GMM es probabilístico, la vista no presenta `predict_proba`, incertidumbre, AIC ni BIC.

## 7. Métricas: significado y realidad del código

| Métrica | Fórmula/lectura | Situación en la vista |
|---|---|---|
| Inercia/WCSS | suma de distancias cuadradas al centroide; menor siempre al aumentar K | valores K=2..8 **hardcodeados** |
| Silueta | por punto `s=(b-a)/max(a,b)`, donde `a` es cohesión y `b` separación; mayor y cercano a 1 es mejor | tabla/curvas **hardcodeadas** |
| Calinski-Harabasz | `(dispersión entre grupos/(K-1)) / (dispersión interna/(n-K))`; mayor es mejor | tabla comparativa **hardcodeada** |
| Davies-Bouldin | promedio de la peor razón `(dispersión_i+dispersión_j)/distancia_centroides`; menor es mejor | tabla comparativa **hardcodeada** |

Aunque las funciones de `sklearn.metrics` están importadas, no se llaman en la ruta activa. `get_model_evaluations()` contiene literalmente la curva K y los valores de silueta retornados para DBSCAN/GMM; estos dos últimos ni siquiera se usan después. El cuaderno 1C sí calcula métricas reales para K=2..10, busca parámetros DBSCAN, evalúa AIC/BIC y construye la comparación. La vista sustituyó esas evaluaciones por constantes.

## 8. Inventario completo de resultados

### Pestaña 1 — EDA & Feature Engineering

| Salida | Cálculo/datos | Procedencia |
|---|---|---|
| Tickets originales | `len(df_cluster)` tras el inner join | **En vivo sobre precalculado** |
| Ingresos totales | `sum(total_monto)` | **En vivo sobre precalculado** |
| Ticket promedio | `mean(total_monto)` | **En vivo sobre precalculado** |
| Histograma + KDE de monto | 30 bins y densidad suavizada sobre todos los tickets, antes de Tukey | **En vivo sobre precalculado** |
| Correlación Spearman | correlación de rangos de las 8 variables sobre todos los tickets | **En vivo sobre precalculado** |
| Tabla descriptiva | `describe().T`: count, media, desviación, cuartiles, mínimo y máximo | **En vivo sobre precalculado** |
| Justificación sobre ID de cliente | narrativa fija; `ID_Cliente` ni siquiera entra al pipeline | **Hardcodeado** |

Spearman mide asociación monótona mediante rangos y es menos sensible a magnitudes extremas que Pearson; no implica causalidad. Esta matriz usa datos **sin filtrar**, mientras que los modelos usan solo `df_normal`.

### Pestaña 2 — Outliers & Escalado

| Salida | Cálculo/datos | Procedencia |
|---|---|---|
| Límite superior monto y Q3 | cuantiles e IQR actuales | **En vivo** |
| Tickets normales y porcentaje | longitud de la máscara conjunta monto/ítems | **En vivo** |
| “39 (4.15 %) outliers” | literal, no `len(df_mayoristas)` | **Hardcodeado y desactualizado** |
| Boxplots de monto e ítems | muestran únicamente `df_normal`; no dibujan el corte | **En vivo** |
| Histograma con línea Tukey | todos los montos y línea del límite de monto; no muestra el límite de ítems | **En vivo** |
| Tabla mayoristas | ocho mayores montos de `df_mayoristas` | **En vivo** |
| Tabla Z-Score | media y desviación muestral de cada columna escalada | **En vivo** |
| Texto “media 0, desviación 1.0007” y explicación B2B | literal; no valida la tabla ni la naturaleza corporativa | **Hardcodeado** |

La leyenda “muestra saneada (`total_monto <= S/ 10`)” omite la segunda condición `n_items <= upper_items`. Con los archivos actuales, el KPI dinámico indica 746 normales, mientras el KPI fijo sigue diciendo 39 aislados aunque `df_mayoristas` contiene 193.

### Pestaña 3 — Comparación & Perfiles

| Salida/control | Cálculo/datos | Procedencia |
|---|---|---|
| K elegido = 3 y mejor modelo K-Means | texto literal | **Hardcodeado** |
| Inercia K=3 | `metrics_k[3]`, diccionario literal | **Hardcodeado** |
| Curvas de codo y silueta K=2..8 | listas literales; la estrella en K=3 también es fija | **Hardcodeado** |
| Tabla K-Means/GMM/DBSCAN | HTML con parámetros y tres métricas literales | **Hardcodeado** |
| Slider K=2..8 | solo cambia el K-Means del visor; para K distinto de 3 ajusta un modelo nuevo | Control **En vivo** |
| Dispersión K-Means | color de etiqueta calculada con 8D; ejes muestran monto/ítems sin escalar | **En vivo** |
| Dispersión DBSCAN | etiquetas DBSCAN vivas, incluido ruido `-1`; proyección 2D | **En vivo** |
| Dispersión GMM | componente GMM vivo; proyección 2D | **En vivo** |
| Volumen por clúster | conteo y porcentaje de `Cluster_Final` K-Means K=3 | **En vivo** |
| Pago por clúster | frecuencias apiladas de `Metodo_Pago` por K-Means K=3 | **En vivo** |
| Tabla de perfiles comerciales | porcentajes, centroides, nombres y estrategias escritos en HTML | **Hardcodeado** |
| “n_departamentos = 1.00 en todos” | conclusión literal | **Hardcodeado y contradicho por los datos actuales** |

El slider no actualiza KPIs, codo, silueta, tabla comparativa, perfiles, volumen ni métodos de pago: todos permanecen referidos a K=3 o a constantes. Los tres scatterplots son proyecciones; dos puntos cercanos en monto/ítems pueden estar separados por las otras seis variables.

En los CSV actuales existen siete tickets con más de un departamento y cuatro sobreviven al filtro de Tukey (por ejemplo, `VTA-0000001`). Por tanto, la afirmación de una barrera absoluta y promedio exactamente 1 en todos los clústeres no debe presentarse como resultado vigente. Tampoco debe llamarse “centroide” a la tabla comercial sin recalcular `groupby('Cluster_Final').mean()`.

## 9. Diferencias frente al cuaderno 1C

| Cuaderno offline | Runtime Streamlit |
|---|---|
| Correlación Pearson por defecto y ocho histogramas | Spearman y solo histograma de monto |
| Evalúa K=2..10 con cuatro métricas reales | muestra K=2..8 con inercia/silueta fijas |
| Gráfico k-distancia y grid search de DBSCAN | fija `eps=1.5`, `min_samples=15` |
| Evalúa GMM 2..10 con AIC/BIC y muestra confianza posterior | fija tres componentes y solo muestra etiquetas |
| Calcula comparación con funciones métricas | muestra tabla HTML fija |
| Calcula `perfil_cluster` y exporta `ventas_clusterizadas.csv` | muestra perfil fijo y no exporta |

## 10. Fallos, supuestos y forma de explicarlo

- Si faltan columnas, archivos o dependencias no hay manejo local de errores; falla la vista.
- CSV limpio significa “preprocesado”, no necesariamente correcto ni actualizado.
- El filtrado Tukey y StandardScaler se recalculan si se invalida la caché; ninguna métrica fija se sincroniza automáticamente.
- No se valida estabilidad de clústeres, sensibilidad a parámetros ni deriva del dataset.
- Un segmento comercial requiere perfilar medias/medianas y validarlo con negocio; el algoritmo solo entrega particiones.

Para una exposición universitaria: explicar primero la conversión línea -> vector de ticket, luego Tukey, después Z-Score y finalmente las diferencias geométricas entre K-Means, DBSCAN y GMM. Cerrar separando resultados realmente calculados de la narrativa fija: el principal aprendizaje de minería de datos es que una cifra solo es defendible si puede trazarse hasta datos, transformación y algoritmo reproducibles.

