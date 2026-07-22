# Clasificación: predicción del método de pago

## 1. Propósito de la vista

La vista **Clasificación** estudia un problema supervisado binario: predecir si una venta será pagada con **YAPE** o **EFECTIVO** a partir del monto, la composición resumida del ticket y el día de compra.

La variable objetivo se codifica como:

$$y=\begin{cases}
1,&\text{YAPE}\\
0,&\text{EFECTIVO}
\end{cases}$$

El flujo tiene dos etapas distintas:

1. **Offline, en el notebook:** carga datos, construye variables, divide train/test, entrena Random Forest y XGBoost, evalúa, elige un ganador y exporta artefactos.
2. **Runtime, en Streamlit:** carga el ganador ya entrenado y los resultados ya calculados. El selector permite comparar métricas históricas de ambos modelos, pero la inferencia usa siempre el único modelo ganador guardado.

### Etiquetas de procedencia

| Etiqueta | Significado |
|---|---|
| **En vivo** | Se calcula durante la interacción actual. |
| **Precalculado** | Se calculó en el notebook y fue exportado a JSON. |
| **Preentrenado** | Lo produce el modelo persistido, entrenado anteriormente. |
| **Hardcodeado** | Está escrito directamente en el código de la vista. |

## 2. Ruta activa, archivos y artefactos

```text
Notebooks/Panel_2_Prediccion_Metodo_Pago.ipynb
  -> cargar ventas y detalle
  -> construir features por ID_Venta
  -> split estratificado 80/20
  -> entrenar Random Forest y XGBoost
  -> elegir por F1
  -> exportar models/*.json

app.py: opción "Clasificación"
  -> show_predictivo (alias de src.panel_predictivo.show_panel)
     -> render()
        -> cargar_modelo()
        -> _cargar_resultados()
        -> KPIs / matriz / importancias
        -> predict() + predict_proba() al pulsar el botón
```

| Archivo | Funciones o contenido | Influencia |
|---|---|---|
| `app.py`, líneas 35 y 505-508 | importación y enrutamiento | Llama a `src.panel_predictivo.show_panel`. |
| `src/panel_predictivo.py`, líneas 31-44 | rutas y esquema esperado | Ancla `models/` al repositorio y define las siete columnas esperadas. |
| `src/panel_predictivo.py`, líneas 108-159 | `cargar_modelo()` | Reconstruye el encoder y carga el XGBoost nativo o, alternativamente, un Random Forest serializado. |
| `src/panel_predictivo.py`, líneas 162-183 | `_cargar_resultados()` | Lee métricas, matrices e importancias, y comprueba el checksum si existe. |
| `src/panel_predictivo.py`, líneas 189-384 | `render()` / `show_panel` | Produce toda la vista e inferencia. |
| `Notebooks/Panel_2_Prediccion_Metodo_Pago.ipynb` | `cargar_ventas()`, `construir_features_detalle()`, `construir_dataset_modelo()`, `preparar_datos()`, `entrenar_y_evaluar()`, `guardar_modelo_seguro()` | Fuente del entrenamiento y de los artefactos. No se ejecuta al abrir Streamlit. |
| `models/modelo_metodo_pago.json` | metadatos, columnas y clases | Declara que el ganador actual es XGBoost, las clases del encoder y el booster que se debe abrir. |
| `models/modelo_metodo_pago_booster.json` | árboles XGBoost | Modelo preentrenado utilizado en inferencia. |
| `models/resultados_panel4.json` | métricas, matrices, importancias y recomendación | Snapshot precalculado mostrado en KPIs y gráficos. |

Aunque algunos textos antiguos del notebook mencionan un modelo «.pkl», el flujo actual no usa un `.pkl` para XGBoost: guarda el booster en formato nativo JSON para reducir incompatibilidades de pickle.

## 3. Construcción offline del dataset

### 3.1 Carga

El notebook busca recursivamente en Google Drive el primer `ventas.csv` y el primer `detalle-ventas.csv`, usando `glob(...)[0]`. Los lee como archivos separados por `;`, elimina columnas `Unnamed` del detalle y normaliza `Fecha` al día, descartando cualquier hora.

Esta búsqueda es un riesgo de reproducibilidad: si Drive contiene varias copias, el primer resultado puede no ser el dataset deseado. Además, el runtime no vuelve a consultar los CSV; trabaja con los artefactos ya exportados.

### 3.2 Resumen del detalle a una fila por ticket

`construir_features_detalle(detalle)` agrupa por `ID_Venta` y crea:

| Variable | Cálculo del notebook | Interpretación y matiz |
|---|---|---|
| `n_items` | suma de `Cantidad` | Unidades totales, no número de líneas. Las fotocopias pueden producir valores muy altos. |
| `n_productos_distintos` | `nunique(ID_Producto)` | Número de IDs no nulos diferentes; los IDs ausentes no cuentan. |
| `departamento_principal` | moda de `Departamento`, tomando `.iloc[0]` | Departamento más frecuente por **líneas**, no por unidades o soles. En un empate, se toma la primera moda que entrega pandas. |
| `pct_fotocopiadora` | media de `(Departamento == "FOTOCOPIADORA")` | Fracción de líneas de detalle de fotocopiadora; una línea con 100 copias pesa igual que una línea con una unidad. |

Después, `construir_dataset_modelo()` hace un `inner merge` entre `ventas.ID` y `detalle.ID_Venta`. Solo permanecen tickets presentes en ambos archivos. Añade:

| Variable | Cálculo |
|---|---|
| `Total` | monto de la cabecera de venta. |
| `dia_semana` | `Fecha.dt.dayofweek`: lunes=0, ..., domingo=6. |
| `es_fin_de_semana` | 1 si `dia_semana` es 5 o 6; 0 en otro caso. |
| `target` | YAPE=1, EFECTIVO=0. |

La columna `Medio` del detalle se excluye deliberadamente porque replica `Metodo_Pago`. Usarla sería **fuga de información**: el predictor recibiría prácticamente la respuesta que intenta aprender.

El dataset registrado por el notebook tiene 939 tickets y nueve columnas contando ID y target. El snapshot actual de `datasets/limpio/ventas.csv` contiene 633 pagos en EFECTIVO (67.41 %) y 306 en YAPE (32.59 %).

### 3.3 Codificación y partición

`preparar_datos(dataset, test_size=0.2)` ajusta `LabelEncoder` sobre `departamento_principal`. Las clases exportadas actualmente son:

```text
FOTOCOPIADORA -> 0
UTILES        -> 1
```

Luego selecciona siete predictores, en este orden contractual:

```text
Total
n_items
n_productos_distintos
departamento_principal_enc
pct_fotocopiadora
dia_semana
es_fin_de_semana
```

`train_test_split(test_size=0.2, random_state=42, stratify=y)` produce 751 tickets de entrenamiento y 188 de prueba. `stratify=y` mantiene aproximadamente la proporción de YAPE/EFECTIVO en ambos grupos. El conjunto de prueba no se usa para ajustar árboles, aunque sí se usa una sola vez para comparar y escoger al ganador.

No se estandarizan variables. Esto es razonable para árboles: sus cortes dependen del orden de los valores y no de distancias euclidianas. Tampoco hay tratamiento de outliers, imputación, selección de variables ni validación de combinaciones imposibles.

## 4. Algoritmos entrenados

### 4.1 Random Forest

Un árbol clasifica mediante divisiones sucesivas, por ejemplo `Total <= 8.5`. En clasificación suele elegir cortes que reduzcen la impureza Gini:

$$Gini(S)=1-\sum_{k}p_k^2$$

Random Forest entrena muchos árboles sobre muestras *bootstrap* y subconjuntos aleatorios de variables. La clase final surge del voto de los árboles y `predict_proba()` promedia sus probabilidades. Esta combinación reduce la varianza de un árbol individual.

Parámetros explícitos del notebook:

| Parámetro | Valor | Función |
|---|---:|---|
| `n_estimators` | 200 | Cantidad de árboles. |
| `max_depth` | 6 | Limita complejidad y sobreajuste. |
| `class_weight` | `balanced` | Pondera cada clase inversamente a su frecuencia. |
| `random_state` | 42 | Hace reproducible el muestreo aleatorio. |

Los demás parámetros conservan los defaults de la versión de scikit-learn usada al entrenar.

### 4.2 XGBoost

XGBoost es *gradient boosting*: construye árboles secuencialmente, de modo que cada nuevo árbol corrige los errores del conjunto anterior. En forma simplificada:

$$F_m(x)=F_{m-1}(x)+\eta h_m(x)$$

donde $h_m$ es el nuevo árbol y $\eta$ es la tasa de aprendizaje. Para clasificación binaria minimiza log-loss regularizada, usando gradientes y segundas derivadas para decidir cortes.

Parámetros explícitos:

| Parámetro | Valor | Función |
|---|---:|---|
| `n_estimators` | 200 | Número máximo de rondas/árboles. |
| `max_depth` | 4 | Profundidad de cada árbol. |
| `learning_rate` | 0.05 | Reduce el aporte de cada árbol. |
| `scale_pos_weight` | $N_{EFECTIVO}/N_{YAPE}$ en train | Da mayor peso a la clase positiva minoritaria, YAPE. |
| `eval_metric` | `logloss` | Métrica interna de entrenamiento. |
| `random_state` | 42 | Reproducibilidad. |

No se configura *early stopping*, conjunto de validación, búsqueda de hiperparámetros ni calibración de probabilidades. Los defaults restantes dependen de XGBoost; el artefacto declara versión de entrenamiento 3.3.0.

### 4.3 Criterio para elegir ganador

`entrenar_y_evaluar()` ajusta cada modelo, obtiene clases con `predict()`, probabilidades YAPE con `predict_proba()[:, 1]` y calcula métricas sobre los mismos 188 tickets de test.

El ganador se elige exclusivamente por mayor **F1 de YAPE**. En empate gana XGBoost por el operador `>=`. En el snapshot actual, XGBoost obtiene F1=0.400 y Random Forest F1≈0.387, por lo que se exporta XGBoost aunque Random Forest tenga mayor accuracy y ROC-AUC.

## 5. Métricas de evaluación

Con YAPE como clase positiva:

- **TP:** YAPE real predicho como YAPE.
- **TN:** EFECTIVO real predicho como EFECTIVO.
- **FP:** EFECTIVO real predicho como YAPE.
- **FN:** YAPE real predicho como EFECTIVO.

$$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$$

$$Precision=\frac{TP}{TP+FP}$$

$$Recall=\frac{TP}{TP+FN}$$

$$F1=2\frac{Precision\cdot Recall}{Precision+Recall}$$

- **Accuracy** mide aciertos globales, pero una clase mayoritaria puede hacerla optimista.
- **Precision** responde: «de todo lo predicho como YAPE, ¿cuánto fue realmente YAPE?».
- **Recall** responde: «de todos los YAPE reales, ¿cuántos detectó?».
- **F1** equilibra precision y recall mediante media armónica.
- **ROC-AUC** mide la capacidad de ordenar un YAPE por encima de un EFECTIVO a través de todos los umbrales. 0.5 es comparable al azar y 1.0 es separación perfecta.

La vista solo muestra F1, accuracy y ROC-AUC como KPIs. Precision y recall están en el JSON y son indispensables para interpretar los errores.

### Snapshot exportado actual

| Modelo | Accuracy | Precision YAPE | Recall YAPE | F1 YAPE | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.6117 | 0.3966 | 0.3770 | 0.3866 | 0.5818 |
| XGBoost | 0.5851 | 0.3768 | 0.4262 | 0.4000 | 0.5545 |

Ambos AUC están cerca de 0.5 y los F1 son moderados/bajos. XGBoost recupera más YAPE, pero a costa de más falsos positivos y menor accuracy. No es correcto presentar una ventaja pequeña en F1 como prueba de un modelo fuerte.

## 6. Matrices de confusión

`sklearn.metrics.confusion_matrix()` produce filas reales y columnas predichas en el orden `[0, 1]`:

### Random Forest — **Precalculado**

|  | Pred. EFECTIVO | Pred. YAPE |
|---|---:|---:|
| Real EFECTIVO | TN=92 | FP=35 |
| Real YAPE | FN=38 | TP=23 |

Acertó 115 de 188 tickets. Detectó 23 de 61 YAPE y dejó escapar 38.

### XGBoost — **Precalculado**

|  | Pred. EFECTIVO | Pred. YAPE |
|---|---:|---:|
| Real EFECTIVO | TN=84 | FP=43 |
| Real YAPE | FN=35 | TP=26 |

Acertó 110 de 188. Detectó tres YAPE adicionales respecto a Random Forest, pero convirtió ocho EFECTIVO adicionales en falsos YAPE. Esa compensación explica su mayor recall y F1, junto con menor precision y accuracy.

El heatmap de Streamlit no recalcula estas matrices con el modelo cargado: lee los cuatro valores de `resultados_panel4.json`.

## 7. Importancia de variables y diferencia con SHAP

El notebook sí calcula valores SHAP con `shap.TreeExplainer` para análisis interactivo del ganador, pero **no exporta esos valores**. Al construir `resultados_panel4.json`, exporta `modelo.feature_importances_` de ambos modelos.

La pestaña llamada **“Importancia SHAP”** en Streamlit grafica esas importancias internas. Por tanto, el rótulo es incorrecto:

- En Random Forest corresponden a importancia normalizada por disminución de impureza agregada entre árboles.
- En XGBoost corresponden a la importancia interna normalizada del booster; con árboles y configuración por defecto suele basarse en *gain*.
- No son valores SHAP, no muestran signo, no indican si una variable aumenta o disminuye la probabilidad y no ofrecen explicaciones por ticket.

### Valores exportados — **Precalculado**

| Variable | Random Forest | XGBoost |
|---|---:|---:|
| `Total` | 0.4386 | 0.2210 |
| `n_items` | 0.2611 | 0.2006 |
| `n_productos_distintos` | 0.0638 | 0.1168 |
| `departamento_principal_enc` | 0.0200 | 0.1851 |
| `pct_fotocopiadora` | 0.0258 | 0.1152 |
| `dia_semana` | 0.1535 | 0.1613 |
| `es_fin_de_semana` | 0.0372 | 0.0000 |

La barra horizontal ordena de menor a mayor y escribe el valor al extremo. «Importante» significa que el modelo usó esa variable para dividir, no que exista causalidad. Tampoco se deben comparar literalmente magnitudes entre algoritmos con definiciones de importancia diferentes.

## 8. Carga e integridad de artefactos

### `cargar_modelo()` — **Preentrenado**

1. Abre `modelo_metodo_pago.json`.
2. Reconstruye `LabelEncoder.classes_` con `departamentos_clases`.
3. Si `tipo_modelo == "XGBoost"`, crea `XGBClassifier()` y carga el booster nativo indicado.
4. Si el ganador fuera Random Forest, decodifica el pickle Base64, valida SHA-256 y deserializa.
5. Devuelve modelo, encoder, columnas y tipo.

Para XGBoost se compara la versión instalada con la de entrenamiento y se muestra una advertencia si difieren, pero se intenta continuar. El booster XGBoost no tiene una verificación SHA-256 equivalente en esta ruta.

### `_cargar_resultados()` — **Precalculado**

Lee `resultados_panel4.json`, elimina temporalmente `checksum_sha256`, serializa el resto con claves ordenadas y compara el hash. Una diferencia solo produce advertencia; los números se siguen mostrando.

Ambas funciones usan caché de Streamlit (`cache_resource` para el modelo y `cache_data` para resultados). Reemplazar archivos mientras el servidor está activo puede no actualizar la vista hasta invalidar la caché.

Se manejan explícitamente archivos ausentes y algunos `ValueError`. Un JSON mal formado, claves faltantes, dependencias ausentes o errores inesperados de XGBoost no tienen una recuperación específica. Si las columnas del metadata difieren de `_COLUMNAS_ESPERADAS`, aparece una advertencia, pero la vista continúa.

## 9. Inventario resultado por resultado

| Elemento visible | Procedencia | Cálculo / fuente | Interpretación y discrepancias |
|---|---|---|---|
| **Selector XGBoost / Random Forest** | **Hardcodeado** | Elige una clave dentro del JSON. XGBoost es el valor inicial. | Cambia KPIs, matriz e importancias; no cambia el modelo de inferencia. |
| **F1-Score** | **Precalculado** | `resultados["metricas"][modelo]["f1"]`. | Evalúa YAPE en el test único. El subtítulo `class_weight='balanced'` se muestra también para XGBoost, que en realidad usa `scale_pos_weight`. |
| **Accuracy** | **Precalculado** | Aciertos / 188 en test. | Debe leerse junto con desbalance y matriz. |
| **ROC-AUC** | **Precalculado** | AUC sobre probabilidades del test. | Mide ranking, no exactitud de probabilidades ni desempeño a un único umbral. |
| **Matriz de Confusión** | **Precalculado** | Arreglo 2×2 del JSON para el selector. | Los rótulos de filas/columnas son correctos; no se recalcula al abrir. |
| **Importancia Global de Variables** | **Precalculado** | `feature_importances_` exportado. | La pestaña dice SHAP, pero no usa SHAP. No comunica dirección ni explicación local. |
| **Aviso de modelo ganador** | **En vivo + Preentrenado** | Aparece cuando el selector difiere de `tipo_modelo`. | Explica correctamente que el selector no cambia inferencia. |
| **Método Predicho** | **Preentrenado, inferencia en vivo** | `modelo.predict(fila)[0]`, normalmente umbral interno 0.5. | Es una clase estimada, no un pago observado ni garantía. |
| **Probabilidad de YAPE** | **Preentrenado, inferencia en vivo** | `modelo.predict_proba(fila)[0,1]`. | Si predice EFECTIVO, se sigue mostrando probabilidad de YAPE. No está calibrada. |
| **Preferencia por F1-Score** | **Hardcodeado** | Texto fijo. | Dice 66.3 %/33.7 %, pero el snapshot actual tiene 67.41 %/32.59 %. Accuracy no queda «invalidada»; simplemente es insuficiente sola. |
| **Lectura honesta del desempeño** | **Hardcodeado + Precalculado** | Inserta `modelo_recomendado` del JSON en una explicación fija. | El badge dice `SHAP INSIGHT`, aunque el texto no se deriva de SHAP. |

## 10. Construcción de una inferencia

Los controles generan una única fila con el mismo orden de columnas del metadata:

| Entrada | Rango / transformación runtime | Diferencia respecto al entrenamiento |
|---|---|---|
| Total | S/ 0.1–250.0; defecto 15.0 | El notebook registró valores hasta alrededor de S/ 188.2; la interfaz permite cierta extrapolación. |
| Nº Items | 1–100; defecto 2 | En entrenamiento existen cantidades mucho mayores (el notebook reporta hasta 2,352), que la simulación no permite representar. |
| Nº Productos Distintos | 1–50; defecto 1 | El dataset puede contener 0 cuando falta `ID_Producto`; runtime no permite 0. No valida que distintos ≤ ítems. |
| Departamento | valores de `departamentos_validos` | Actualmente solo FOTOCOPIADORA y UTILES. Se transforma con el encoder persistido. |
| `pct_fotocopiadora` | 1.0 si el departamento elegido es FOTOCOPIADORA; si no, 0.0 | El entrenamiento admite tickets mixtos con fracciones intermedias; la UI solo simula extremos. |
| Día | lista lunes-domingo | Se convierte a índice 0–6. |
| Fin de semana | derivado automáticamente | 1 para sábado/domingo. |

El botón «Predecir Método de Pago» no reentrena. Solo llama al ganador que `cargar_modelo()` dejó en memoria. En el estado actual, aunque el usuario seleccione Random Forest para ver sus gráficos, la predicción seguirá saliendo de XGBoost.

## 11. Limitaciones y validez

1. **Una sola división train/test.** No hay validación cruzada ni intervalo de variabilidad entre particiones.
2. **Selección y reporte en el mismo test.** El test participa en elegir al ganador; falta un conjunto final independiente para una estimación completamente imparcial.
3. **Sin ajuste sistemático.** Los hiperparámetros son decisiones manuales; no hay GridSearch, RandomizedSearch ni optimización bayesiana.
4. **Desempeño limitado.** F1≈0.40 y AUC≈0.55 muestran señal débil. El modelo no debería automatizar decisiones sensibles.
5. **Probabilidades no calibradas.** Un 70 % mostrado no ha sido validado para significar siete aciertos de cada diez casos similares.
6. **Pocas variables de contexto.** No incluye promociones, temporada, hora, cliente, conectividad de Yape ni secuencia histórica.
7. **Agregaciones discutibles.** Departamento y porcentaje se basan en líneas, mientras `n_items` usa unidades.
8. **Sin control de dominio.** La UI permite combinaciones incoherentes y algunas fuera del rango de entrenamiento; los árboles extrapolan de forma escalonada, no aprenden tendencias fuera de sus cortes.
9. **LabelEncoder como número.** Con dos departamentos funciona como separación binaria. Si se agregan más, introduce un orden artificial; debería evaluarse one-hot encoding o un pipeline.
10. **Drift.** Las preferencias de pago pueden cambiar. La vista no detecta deriva, no registra predicciones reales ni reentrena automáticamente.
11. **Reproducibilidad de Drive.** `glob()[0]` puede entrenar con una copia distinta si existen varios archivos del mismo nombre.
12. **Versiones no fijadas.** `requirements.txt` no fija XGBoost/scikit-learn; el metadata advierte diferencias, pero no garantiza equivalencia futura.

## 12. Guion académico para explicar la construcción

1. «Defino YAPE como clase positiva porque es la clase minoritaria que deseo detectar».
2. «Agrego el detalle a nivel ticket para que cada fila de $X$ corresponda exactamente a una etiqueta de pago».
3. «Elimino `Medio` porque sería fuga de datos y creo siete variables disponibles antes o durante la venta».
4. «Uso un split estratificado 80/20 y balanceo distinto para cada algoritmo: pesos de clase en Random Forest y `scale_pos_weight` en XGBoost».
5. «Random Forest reduce varianza combinando árboles independientes; XGBoost aprende árboles secuenciales que corrigen errores».
6. «Comparo errores sobre YAPE con precision, recall y F1, además de accuracy y ROC-AUC. Elijo XGBoost por F1, no porque gane todas las métricas».
7. «Exporto modelo, encoder, columnas y resultados para separar entrenamiento offline de inferencia runtime».
8. «En producción el selector es analítico: compara snapshots. Solo el ganador persistido responde al botón».
9. «La gráfica rotulada SHAP es una discrepancia: muestra importancia interna. Los SHAP verdaderos solo se calcularon dentro del notebook».

La conclusión académicamente defendible es: «hay una señal predictiva moderada en monto, composición y día, pero las métricas actuales exigen tratar la salida como apoyo exploratorio, no como certeza operativa».
