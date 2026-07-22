# Predicciones: pronóstico semanal de ingresos

## 1. Objetivo y procedencia de los resultados

La vista responde: **¿cuánto venderá SmartBazar en cada próxima semana?** La variable objetivo es el ingreso semanal en soles, obtenido al sumar <code>Total</code>; no pronostica tickets ni unidades.

| Etiqueta | Significado |
|---|---|
| **En vivo** | Se calcula al ejecutar con el CSV, la fecha del sistema o controles actuales. |
| **Preentrenado** | Es inferencia de un modelo persistido; la vista no vuelve a ajustarlo. |
| **Precalculado** | Parámetro producido offline y cargado desde un artefacto. |
| **Hardcodeado** | Opción, regla, periodo o texto escrito directamente en Python. |

La salida principal es **En vivo sobre Prophet Preentrenado**: se reconstruyen fechas y regresores, pero los parámetros aprendidos vienen de un archivo.

## 2. Ruta activa, archivos y funciones

<code>app.py</code> incluye **Predicciones** en el menú (líneas 478-491) y llama a <code>panel_forecast.show_panel()</code> (514-515). Aunque importa además el alias <code>show_forecast</code>, ese alias no se usa.

| Archivo | Papel |
|---|---|
| <code>src/panel_forecast.py</code> | Preparación semanal, inferencia, métricas, gráficos y tablas. |
| <code>src/data_loader.py</code> | <code>load_ventas(limpio=True)</code> elige primero el CSV limpio y, si falta, el crudo. |
| <code>datasets/limpio/ventas.csv</code> | Fuente activa mientras exista; requiere <code>Fecha</code> y <code>Total</code>. |
| <code>models/final_prophet_model.joblib</code> | Prophet final entrenado offline. |
| <code>models/ma_model_params.joblib</code> | Ventana y último promedio de la media móvil. Actualmente: 3 y 254.033333333333. |
| <code>Notebooks/Panel 3_ Series temporales.ipynb</code> | Experimentación, entrenamiento final y exportación; no se ejecuta al abrir la vista. |
| <code>src/colab_3.py</code> | Aplicación antigua independiente; no está importada por <code>app.py</code>. |

| Función activa | Responsabilidad |
|---|---|
| <code>parsear_fechas_cronologicas()</code> | Resuelve fechas ambiguas día/mes intentando conservar cronología. |
| <code>_load_sales_data()</code> | Convierte, valida, elimina inválidos y filtra fechas futuras. |
| <code>_build_school_dates()</code>, <code>_build_regressors()</code> | Construyen feriados y periodos escolares semanales. |
| <code>_prepare_weekly_data()</code> | Completa días, agrega semanas y añade regresores. |
| <code>_load_model()</code>, <code>_load_ma_params()</code> | Cargan y cachean artefactos. |
| <code>_fit_forecast_frame()</code> | Ejecuta Prophet en historia y horizonte. |
| <code>_fit_moving_average_frame()</code> | Calcula baseline histórico y futuro constante. |
| <code>_safe_metrics()</code>, <code>_build_backtest_frame()</code> | Calculan errores agregados e individuales. |
| <code>show_panel()</code> | Orquesta las tres pestañas. |

## 3. Construcción de la serie

### 3.1 Carga y fechas

<code>_load_sales_data()</code>, líneas 398-414:

1. carga ventas limpias;
2. exige <code>Fecha</code> y <code>Total</code>;
3. genera dos interpretaciones de cada fecha, con día primero y mes primero;
4. favorece la interpretación que no retrocede respecto de la fila anterior y, si persiste la ambigüedad, el salto cronológico menor;
5. convierte <code>Total</code> a número y elimina fecha/total nulos;
6. descarta fechas posteriores al día actual;
7. normaliza la hora para obtener <code>Fecha_Diaria</code>.

El parser depende del orden de filas. El filtro por hoy hace que el mismo CSV pueda producir más semanas con el paso del tiempo, pero **no reentrena Prophet**. No se filtran montos negativos ni ceros.

### 3.2 Agregación

<code>_prepare_weekly_data()</code>, líneas 484-504, suma por día, crea un calendario diario continuo, rellena días ausentes con cero y remuestrea con <code>W-SUN</code>:

\[
ingreso_d=\sum_i Total_i,
\qquad
y_t=\sum_{d\in semana\ terminada\ el\ domingo\ t} ingreso_d
\]

<code>ds</code> es el domingo de cierre y <code>y</code> el ingreso. Primera y última semanas pueden ser parciales; el promedio posterior incluye los ceros imputados.

### 3.3 Regresores

| Variable | Algoritmo | Procedencia |
|---|---|---|
| <code>is_nat</code> | 1 si existe al menos un feriado peruano en esa semana; se remuestrea con máximo. | **En vivo** mediante <code>holidays</code> |
| <code>is_school</code> | 1 si la semana intersecta 1-30 de marzo o 20 de julio-3 de agosto. | **Hardcodeado** por año |

Los periodos escolares son una aproximación, no un calendario oficial. Se generan años hasta dos años posteriores al último dato. Si <code>holidays</code> falla, la excepción se oculta y todos los feriados quedan en cero.

## 4. Algoritmos y entrenamiento offline

### 4.1 Experimento del notebook

El notebook reserva cronológicamente las últimas cuatro semanas y compara:

- Prophet sobre <code>log1p(y)</code> con feriados y periodo escolar;
- Random Forest Regressor con semana, regresores y rezagos 1-3;
- XGBoost Regressor con las mismas características;
- media móvil de tres semanas;
- Prophet base sin regresores.

Random Forest y XGBoost fueron comparadores offline: la vista activa no los carga.

### 4.2 Prophet exportado

Luego se crea un Prophet nuevo, entrenado con todas las semanas: crecimiento lineal, estacionalidades semanal y anual desactivadas, dos regresores y objetivo <code>log1p(y)</code>. Se exporta a <code>final_prophet_model.joblib</code>. En Streamlit solo se llama a <code>predict()</code>.

Una representación simplificada es:

\[
\log(1+y_t)=g(t)+\beta_1 is\_nat_t+\beta_2 is\_school_t+\varepsilon_t
\]

Prophet aprende la tendencia segmentada \(g(t)\). La salida y sus límites vuelven a soles así:

\[
\hat y_t=\max(0,\exp(\widehat{\log(1+y_t)})-1)
\]

El recorte evita ingresos negativos. Al aplicarlo también a los límites, la banda puede quedar asimétrica.

### 4.3 Media móvil

El artefacto exporta <code>window=3</code> y <code>last_mean</code>. Para el histórico:

\[
\hat y_t^{MA}=\frac{1}{k}\sum_{j=1}^{k}y_{t-j},\quad k\le3
\]

Se implementa con <code>rolling(...).mean().shift(1)</code>. La primera fila, sin pasado, se rellena con el último promedio exportado, decisión poco natural cronológicamente. Para el futuro se repite <code>last_mean</code> en todo el horizonte; <code>ma_future</code> se calcula pero no se muestra.

## 5. Ejecución y controles

<code>show_panel()</code> carga Prophet, reconstruye la serie, carga la media móvil y calcula todos los resultados en cada rerun. <code>_prepare_weekly_data()</code> usa caché de datos sin argumentos y los modelos caché de recursos; cambiar CSV o artefactos puede requerir limpiar caché/reiniciar.

| Control | Efecto real |
|---|---|
| Horizonte 4, 8, 12, 16, 24 o 52 | Cambia <code>periods</code> y filas futuras. Opciones **Hardcodeadas**. |
| Resumen / últimas 8 / últimas 12 | Modifica la ventana de errores, no el rango del gráfico principal. Puede limitarse a aproximadamente un cuarto de la historia. |
| «Ejecutar modelo» | Solo oculta el mensaje informativo en ese rerun. Inferencia y métricas se calculan se pulse o no; nunca entrena. |

## 6. Pestaña «Pronóstico y Proyecciones»

<code>_fit_forecast_frame()</code> produce:

- <code>forecast_hist</code>: predicción sobre fechas del CSV;
- <code>forecast_future</code>: calendario de la historia interna del modelo más el horizonte.

Ambos reconstruyen regresores y calculan valor central/límites en soles. <code>make_future_dataframe()</code> parte de la historia guardada en el artefacto, mientras «futuro» se filtra usando el máximo del CSV actual. Se supone que ambos periodos están alineados; si no, pueden aparecer fechas históricas como futuras o un horizonte insuficiente.

### Gráfico «Evolución de Ventas Semanales»

| Traza | Resultado | Tipo |
|---|---|---|
| Ventas reales | <code>weekly_df.y</code> | **En vivo** |
| Prophet ajuste | <code>forecast_hist.yhat_original</code> | **Preentrenado**; dentro de muestra si coincide con entrenamiento |
| Prophet futuro | Salida desde la última fecha observada, incluyéndola | **Preentrenado** |
| Intervalo | <code>yhat_lower/upper</code> invertidos | **Preentrenado**; se dibuja también sobre historia |
| Media móvil | Promedio rezagado histórico | **En vivo + Precalculado** |

El resumen llama al intervalo «80 %». El notebook usa el valor predeterminado de Prophet, 0.80; si se reemplaza el artefacto por otro ancho, ese texto fijo no cambiará.

### Tabla futura

Muestra fecha, pronóstico central, mínimo y máximo para filas posteriores al último dato, hasta el horizonte. Es **inferencia con modelo Preentrenado**. Si no hay filas posteriores, usa la última salida como fallback: podría mostrar una fecha no futura y solo una fila.

## 7. Pestaña «Evaluación de Modelos»

### Comparativa

El gráfico toma siempre las últimas cuatro filas y muestra real, Prophet y media móvil. «Ventana de prueba» es impreciso: no se crea train/test en Streamlit. Si el CSV coincide con el usado para el modelo final, Prophet se evalúa dentro de muestra. El holdout real del notebook no fue exportado a esta vista.

### Métricas agregadas

\[
RMSE=\sqrt{\frac1n\sum_t(y_t-\hat y_t)^2}
\]

\[
MAPE=\frac{100}{n}\sum_t\left|\frac{y_t-\hat y_t}{d_t}\right|,
\quad d_t=1\ si\ y_t=0;\ d_t=y_t\ en\ otro\ caso
\]

RMSE está en soles y penaliza errores grandes. Reemplazar cero por 1 evita división, pero para una semana sin ventas el término MAPE se convierte en \(100|\hat y_t|\), que deja de ser un porcentaje convencional.

Las tarjetas son **En vivo**. Para la media móvil, <code>recent_eval</code> vuelve a calcular una ventana fija 3 en vez de tomar siempre la del artefacto; hoy coinciden, pero podrían divergir.

### Ganador dinámico

\[
score=RMSE+MAPE
\]

Gana el menor. Es un criterio débil: suma soles con porcentaje. La tarjeta dice «menor varianza», pero no calcula varianza.

### Barras semanales

<code>_build_backtest_frame()</code> calcula MAPE individual y:

\[
\sqrt{(y_t-\hat y_t)^2}=|y_t-\hat y_t|
\]

La barra llamada «RMSE semanal» es, con una observación, **error absoluto**. Las barras usan la media móvil del artefacto; el gráfico de líneas siempre usa cuatro semanas, aunque se hayan elegido 8 o 12.

## 8. Pestaña «Resumen de Ventas»

| Indicador | Fórmula/fuente | Tipo y lectura |
|---|---|---|
| Ventas Históricas | \(\sum_t y_t\) | **En vivo**; suma válida, no futura |
| Última semana | último \(y_t\) | **En vivo**; puede ser parcial |
| Ventas Promedio | \(\sum y_t/n\) | **En vivo**; incluye calendario rellenado |
| Número de semanas | <code>len(weekly_df)</code> | **En vivo** |
| Próxima semana | primera fila futura central | **Preentrenado** |
| Intervalo próximo | límites de esa fila | **Preentrenado** |

<code>_currency_2()</code> reemplaza la coma de miles por punto y conserva el punto decimal; valores con miles pueden verse ambiguos, por ejemplo <code>S/ 1.234.56</code>. Es formato, no cálculo.

## 9. Discrepancias y límites

| Elemento | Realidad |
|---|---|
| Ejecutar modelo | El cálculo siempre ocurre; no entrena. |
| Vista 8/12 semanas | Solo afecta errores y puede devolver menos; no recorta el gráfico. |
| Ventana de prueba | Evaluación reciente sin holdout activo. |
| Ganador por varianza | No hay varianza; suma unidades incompatibles. |
| RMSE semanal | Equivale a error absoluto. |
| Media móvil futura | Se calcula constante, no se visualiza. |
| Random Forest/XGBoost | Solo experimentación del notebook. |
| CSV actualizado | Cambia la serie, no los parámetros de Prophet. |

Errores y operación:

- si falta Prophet, media móvil o sus dependencias, se muestra error y la vista termina;
- columnas ausentes o serie vacía impiden preparar fechas/última fila;
- incompatibilidad de regresores puede fallar durante <code>predict()</code>;
- cachés pueden ocultar cambios externos;
- no se valida deriva ni alineación modelo/CSV;
- no se mide cobertura del intervalo ni rendimiento fuera de muestra productivo;
- la última semana parcial puede distorsionar métricas.

<code>src/colab_3.py</code> no es la fuente activa: contiene una app autónoma, rutas relativas y frecuencia <code>W-MON</code> en varias secciones, mientras <code>panel_forecast.py</code> usa <code>W-SUN</code>.

## 10. Guion académico

1. La unidad de análisis es semana e ingreso total.
2. Se normalizan fechas, completan días sin venta y agregan semanas.
3. Feriados y campañas escolares aportan conocimiento de dominio.
4. Prophet aprende <code>log1p(y)</code> y la salida vuelve con <code>expm1</code>.
5. La media móvil de tres semanas es baseline interpretable.
6. Streamlit carga modelos; no entrena.
7. Las métricas visibles no son el holdout del notebook.
8. Botón, ganador y «RMSE semanal» deben explicarse con sus limitaciones reales.

