# Vista EDA: auditoría, limpieza e ingeniería de características

## 1. Qué problema resuelve

La vista **EDA** (Exploratory Data Analysis) permite estudiar la calidad y la forma de los datos antes de aplicar modelos de minería. Su unidad de análisis cambia según la pestaña: filas de ventas, líneas de detalle, SKU de inventario o tickets agregados. No entrena ningún modelo predictivo.

La procedencia de los resultados se marca así:

- **En vivo:** se calcula al ejecutar Streamlit a partir del CSV leído, aunque `st.cache_data` puede reutilizar el cálculo.
- **Precalculado:** el dato ya fue limpiado/exportado a `datasets/limpio/` por un proceso anterior.
- **Preentrenado:** resultado de un modelo persistido. **No existe en esta vista.**
- **Hardcodeado:** cifra o conclusión escrita literalmente en Python; no se recalcula con el CSV.

## 2. Ruta activa, archivos y funciones

`app.py` importa `show_panel` como `show_eda` y, al elegir `EDA`, llama esa función ([`app.py`, líneas 32 y 496-497](../app.py)). La ruta activa es:

```text
app.py -> src.panel_eda.show_panel()
       -> _load_raw_datasets()   -> datasets/crudo/*.csv
       -> _load_clean_datasets() -> datasets/limpio/*.csv
       -> _render_tab_1 ... _render_tab_5
```

| Archivo o artefacto | Funciones/celdas relevantes | Influencia real |
|---|---|---|
| [`src/panel_eda.py`](../src/panel_eda.py) | `show_panel`, `_load_raw_datasets`, `_load_clean_datasets`, `_render_tab_1` a `_render_tab_5`, `show_dataset_modal`, `_safe_render` | Implementación completa que se ejecuta en la vista. |
| [`src/data_loader.py`](../src/data_loader.py) | `clean_ventas_df`, `clean_detalle_df`, `clean_inventario_df`, `export_cleaned_datasets` | Explica una vía para producir los CSV limpios, pero **la vista EDA no importa ni llama este módulo**. |
| [`Panel 1A`](../Notebooks/Panel%201A_%20Auditoria_y_Limpieza_Datos.ipynb) | celdas 3-19 | Proceso offline de auditoría, saneamiento y exportación de `datasets/limpio/`. Es el origen conceptual de las pestañas 1-4. |
| [`Panel 1B`](../Notebooks/Panel%201B_%20EDA_%20e_Ingenieria_de_Caracteristicas.ipynb) | celdas 1-15 | EDA univariado/bivariado e ingeniería por ticket. Es el origen conceptual de la pestaña 5. |
| `datasets/crudo/*.csv` | `ventas`, `detalle_ventas`, `inventario` | Fuente de las pestañas 1-4. No se modifica desde la vista. |
| `datasets/limpio/*.csv` | los mismos tres archivos | Fuente precalculada de la pestaña 5. |

### Carga y caché

`show_panel()` carga los seis DataFrames antes de renderizar las pestañas. Si falta o no se puede interpretar cualquier archivo, no existe un `try/except` de recuperación y falla toda la vista. `_load_raw_datasets()` usa rutas relativas al directorio desde el cual se lanzó la aplicación; solo cae a `datasets/limpio/` cuando no existe la carpeta cruda completa. Prueba `;` y luego `,` para ventas/detalle, y trata especialmente el inventario crudo con cabecera desplazada. `_load_clean_datasets()` asume CSV con coma y UTF-8.

Ambas cargas usan `@st.cache_data`: los indicadores son “en vivo” respecto de la copia cargada en caché, no una consulta continua al disco. Para reflejar un CSV reemplazado puede ser necesario invalidar la caché o reiniciar la aplicación.

### Qué ocurrió offline y qué ocurre en la vista

El cuaderno 1A normaliza fechas a `YYYY-MM-DD`, convierte numéricos, recorta cantidades/subtotales negativos, marca y trunca stock negativo e imputa `Stock_Minimo`; después exporta los CSV limpios. `src/data_loader.py` contiene una alternativa equivalente en intención. La vista **no ejecuta esa limpieza**: en las pestañas 2-4 hace copias temporales para diagnosticar el crudo, y en la pestaña 5 consume el resultado ya exportado.

Hay dos diferencias importantes de linaje:

1. El cuaderno 1A lee el inventario crudo con coma, igual que `_load_raw_datasets`; `load_raw_inventario()` de `data_loader.py` fija `sep=';'`, por lo que esa vía no es compatible con el archivo crudo actual sin corregir el separador.
2. No puede demostrarse desde la aplicación si los CSV limpios actuales fueron generados por el notebook o por el helper. La documentación solo puede afirmar que son artefactos precalculados.

## 3. Pestaña 1: Resumen de Auditoría y Limpieza

### Inventario de resultados

| Resultado/control | Cálculo o algoritmo | Procedencia |
|---|---|---|
| Tarjetas y modal de los tres datasets | `df.shape`; el botón abre `show_dataset_modal()` con `df.head(5)` | **En vivo**, crudo |
| Resumen dimensional | Número de filas y columnas de cada DataFrame | **En vivo**, crudo |
| Selector de dataset | Cambia `target_df` entre ventas, detalle e inventario | Control; no transforma datos |
| Total registros | `len(target_clean)` después de excluir columnas cuyo nombre empieza por `Unnamed` | **En vivo**, crudo |
| Registros válidos | `len(target_clean.dropna())`: filas completas, sin ningún valor nulo | **En vivo**, crudo |
| Nulos en origen | `target_clean.isnull().sum().sum()`: **celdas** nulas, no filas | **En vivo**, crudo |
| Cobertura | `filas_completas / filas_totales * 100` | **En vivo**, crudo |
| Tabla “Auditoría por Campo” | `DataFrame` construido con listas literales de tipos, nulos, completitud y acciones | **Hardcodeado** |
| Nota sobre `Stock_Minimo` | Texto fijo mostrado al seleccionar inventario | **Hardcodeado** |

La cobertura mide **completitud de filas**, no completitud de celdas. Por ejemplo, una fila con un solo nulo deja de ser válida igual que otra con diez. En un DataFrame vacío el código devuelve 100 %, decisión técnica que puede resultar engañosa. Además, “Nulos en origen” y “Registros válidos” no comparten unidad de medida.

La tabla de auditoría no inspecciona los DataFrames. Sus cifras (por ejemplo, 314 nulos) y acciones se seguirán mostrando aunque cambien los archivos. También atribuye imputación por mediana de fecha, moda de pago y `CLI-0000` a cliente; esas operaciones no aparecen ni en el cuaderno 1A ni en las funciones de limpieza actuales. Debe presentarse como una ficha narrativa histórica, no como evidencia calculada.

## 4. Pestaña 2: fenómeno temporal y sesgo horario

El algoritmo elimina ventas sin `ID` o `Fecha`, intenta interpretar `Fecha` con `pd.to_datetime(..., format='mixed', errors='coerce')`, deriva el día de semana y considera que existe hora cuando el texto original contiene `:`. Solo sobre esas filas extrae `dt.hour`. Para inventario busca la primera columna disponible entre `Fecha Ingreso`, `Fecha_Ingreso`, `Fecha` y `FechaIngreso`, la convierte y deriva `Dia_Ingreso`.

| Resultado | Cómo se obtiene | Procedencia |
|---|---|---|
| Conteo de ingresos por día | `countplot` de `Dia_Ingreso`, orden lunes-domingo; cuenta filas por categoría | **En vivo**, crudo |
| Histograma horario | 24 intervalos sobre `Hora` de ventas cuyo texto contiene `:` | **En vivo**, crudo |
| Curva KDE | Estimación de densidad `f_h(x) = (1/nh) Σ K((x-x_i)/h)` añadida por Seaborn; suaviza la distribución, no predice ventas | **En vivo**, crudo |
| Tabla día/registros/porcentaje | `value_counts`; porcentaje = conteo / total de filas de inventario | **En vivo**, crudo |
| 201 domingos, 121 lunes, 76 %, causa operativa y acción ISO | Texto literal de las tarjetas | **Hardcodeado** |

El denominador de la tabla incluye filas cuya fecha no pudo convertirse; por eso los porcentajes pueden no sumar 100 %. Si no hay ventas con hora, el segundo eje queda vacío. Las tarjetas infieren que se digitó por lotes y recomiendan eliminar la hora, pero la pestaña no prueba causalidad ni escribe el dato normalizado: esa transformación ocurrió offline en el cuaderno 1A/CSV limpio.

## 5. Pestaña 3: desfase de Kardex y stock negativo

`Stock_Actual` se convierte a número; valores no interpretables o ausentes se reemplazan temporalmente por 0. Se filtran valores `< 0`, se ordenan de menor a mayor y se toman los diez déficits más severos.

| Resultado | Cómo se obtiene | Procedencia |
|---|---|---|
| Total de SKU negativos | `len(df_negativos)` | **En vivo**, crudo |
| Barras Top 10 | eje X = stock negativo; eje Y = descripción; línea de referencia en cero | **En vivo**, crudo |
| Tabla Top 5 | primeras cinco filas del Top 10 con ID, descripción, departamento y stock | **En vivo**, crudo |
| Productos y déficits citados, explicación de causa | Texto literal | **Hardcodeado** |
| `Alerta_Kardex_Negativo` y `.clip(lower=0)` descritos | Acción offline del cuaderno/helper; no se ejecuta en `_render_tab_3` | **Precalculado + Hardcodeado** |

El truncamiento aplica la regla física `Stock_saneado = max(Stock_original, 0)` y conserva una bandera booleana antes de reemplazar. Es una regla de saneamiento, no un algoritmo que reconstruya el stock real. Tampoco demuestra que el desfase de tiempos sea la causa; esa es una hipótesis de negocio. Los valores no numéricos se vuelven cero y dejan de ser visibles como problema de calidad.

## 6. Pestaña 4: imputación de `Stock_Minimo`

El ROP (*reorder point*) indica el nivel que activa reposición. La vista convierte `Stock_Minimo` a número, calcula nulos y analiza los valores presentes por departamento. No modifica el DataFrame mostrado.

| Resultado | Cómo se obtiene | Procedencia |
|---|---|---|
| Tasa de omisión | `nulos / total_filas * 100` | **En vivo**, crudo |
| Distribución por departamento | `countplot` de valores no nulos, con color por `Departamento` | **En vivo**, crudo |
| Tabla pre-imputación | `pd.crosstab(Departamento, Stock_Minimo)`; el centinela `-999` se etiqueta “NULO / AUSENTE”; añade totales | **En vivo**, crudo |
| 83 % de útiles, moda/mediana 5 y 100 % nulo en fotocopiadora | Textos literales | **Hardcodeado** |
| Regla `UTILES -> 5`, resto -> 2 | Definida en el notebook 1A y en `clean_inventario_df`; la tarjeta solo la describe | **Precalculado + Hardcodeado** |

Formalmente, para una fila con valor ausente:

```text
Stock_Minimo_imputado = 5, si Departamento normalizado = UTILES
                        2, en cualquier otro caso
```

No es imputación estadística aprendida: es una **regla de negocio determinista** inspirada en moda/mediana observadas. En el notebook se implementa con `np.where`; el helper usa `DataFrame.apply`. En la pestaña no se asigna el resultado, pese a que la tarjeta diga “fórmula aplicada”. Si faltan las columnas requeridas se informa que no hay gráfico; si existen pero todos los valores son nulos, el área puede quedar sin gráfico ni aviso específico.

## 7. Pestaña 5: EDA e ingeniería de características

Esta pestaña calcula sobre los CSV **limpios precalculados**.

### Estadísticos univariados

Para `Total`, `Cantidad`, `Precio_Unitario`, `Subtotal`, `Precio_Venta` y `Stock_Actual`, `_get_univariate_stats()` convierte a número, elimina nulos y devuelve:

- media `x̄ = Σx_i/n`, mediana, y la primera moda devuelta por Pandas;
- desviación estándar muestral `s = sqrt(Σ(x_i-x̄)^2/(n-1))`;
- `IQR = Q3-Q1` y `CV = 100*s/x̄` cuando la media no es cero;
- asimetría de SciPy y curtosis de Fisher, donde una normal ideal tiene curtosis 0.

La tabla es **En vivo sobre datos precalculados**. Series vacías se omiten; media cero produce `CV = NaN`; series constantes o muy cortas pueden producir asimetría/curtosis `NaN`.

### Gráficos univariados

| Gráfico | Datos y lectura | Procedencia |
|---|---|---|
| Boxplot + histograma/KDE de `Total` | distribución monetaria por ticket; el boxplot marca media y mediana | **En vivo sobre precalculado** |
| Boxplot + histograma/KDE de `Subtotal` | distribución por línea de detalle | **En vivo sobre precalculado** |
| Boxplot + histograma/KDE de `Precio_Venta` | distribución de precios del catálogo | **En vivo sobre precalculado** |

El boxplot representa Q1, mediana, Q3 y bigotes según 1.5 IQR; los puntos fuera de los bigotes son observaciones atípicas, no errores automáticos. La KDE depende del ancho de banda elegido internamente por Seaborn.

### Vector transaccional por ticket

Se normaliza `Departamento` a mayúsculas; `Cantidad` y `Subtotal` no numéricos se vuelven 0. Después `groupby('ID_Venta')` convierte la relación 1:N en una fila por ticket:

| Característica | Fórmula |
|---|---|
| `Cantidad_Items_Total` | suma de unidades `Cantidad` |
| `Variedad_Items_Ticket` | número de `ID_Producto` distintos; si no existe esa columna, cuenta líneas de subtotal |
| `Subtotal_Total` | suma de subtotales de las líneas |
| `Gasto_Utiles` / `Gasto_Fotocopiadora` | suma condicional del subtotal por departamento |
| `Precio_Promedio_Item` | `Subtotal_Total / Cantidad_Items_Total`, o 0 si no hay unidades |
| `Ratio_Utiles` | `Gasto_Utiles / Subtotal_Total`, o 0 si el subtotal es cero |
| `Departamento_Dominante` | `UTILES` en empate o mayor gasto; si no, `FOTOCOPIADORA` |

La tabla muestra solo ocho tickets, pero la leyenda de dimensionalidad usa toda la matriz. No se une contra la cabecera de ventas, por lo que `Subtotal_Total` no se reconcilia con `ventas.Total`. El valor `MIXTO` declarado como `default` es prácticamente inalcanzable: las dos condiciones son complementarias y los empates se asignan a `UTILES`. A diferencia del notebook 1B, el runtime tampoco calcula `Gasto_Otros` ni exporta `ticket_features.csv`.

### Análisis bivariado

| Resultado | Algoritmo | Procedencia |
|---|---|---|
| Dispersión variedad vs. subtotal | cada punto es un ticket; color = departamento dominante | **En vivo sobre precalculado** |
| Línea de tendencia | regresión lineal OLS global `y = β0 + β1x`, minimizando `Σ(y_i-ŷ_i)^2`; no se muestran `R²`, error ni significancia | **En vivo sobre precalculado** |
| Boxplot de gasto por departamento | compara mediana, IQR, bigotes y atípicos de `Subtotal_Total` | **En vivo sobre precalculado** |
| Preguntas/hipótesis redactadas | texto fijo que orienta la interpretación | **Hardcodeado** |

El notebook 1B contiene además matrices Pearson/Spearman, tabla bivariada por grupo, detección IQR y exportación de características; ninguna de esas salidas forma parte de la ruta EDA actual.

## 8. Cómo explicarlo en una exposición

1. Separar **diagnóstico crudo** (pestañas 1-4) de **análisis sobre datos ya saneados** (pestaña 5).
2. Aclarar la unidad de análisis: registro, celda, SKU, línea o ticket.
3. Presentar las reglas de stock como decisiones de negocio auditables, no como modelos aprendidos.
4. Explicar que agregación e ingeniería convierten líneas heterogéneas en vectores comparables para Clustering.
5. Distinguir evidencia calculada de textos fijos: una conclusión hardcodeada debe volver a validarse si cambian los CSV.

