import json
import os

notebook_path = r"c:\Users\RootAccess\Documents\Proyecto_Mineria\smart-bazar\Notebooks\Panel 3_ Series temporales.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper function to convert text to list of lines with trailing newlines
def to_notebook_source(text):
    lines = text.strip().split('\n')
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]

# Definitions of parameter markdown blocks to append
additions = {
    "### Carga y Normalización de Datos de Ventas": """

#### Parámetros de las funciones aplicadas:
*   **`pd.read_csv`**:
    *   `filepath_or_buffer` (`'ventas.csv'`): Ruta del archivo CSV a cargar.
    *   `sep` (`';'`): Delimitador de columnas utilizado en el archivo.
*   **`parser.parse`**:
    *   `timestr`: Cadena de caracteres que representa la fecha a analizar.
    *   `dayfirst` (`True`): Indica que el día aparece antes que el mes (formato DD/MM/AAAA).
*   **`pd.to_datetime`**:
    *   `arg`: Estructura (lista o serie) a convertir a objetos de tipo fecha de pandas.
*   **`df.dropna`**:
    *   `subset` (`['Fecha_Estandar']`): Lista de columnas donde se buscarán valores nulos para descartar filas.
    *   `inplace` (`True`): Modifica el DataFrame original en el lugar en lugar de retornar una copia.
*   **`series.dt.normalize`**:
    *   No tiene parámetros adicionales. Ajusta todas las horas a la medianoche (00:00:00) para homogeneizar las fechas.""",

    "### Transformación a Datos Semanales y Adición de Regresores": """

#### Parámetros de las funciones aplicadas:
*   **`df.groupby`**:
    *   `by` (`'Fecha_Diaria'`): Columna o índice utilizada para agrupar los registros.
*   **`series.sum`**:
    *   Calcula el total acumulado de las ventas del grupo.
*   **`df.reset_index`**:
    *   Restablece el índice numérico por defecto, convirtiendo el antiguo índice en columna.
*   **`df.set_index`**:
    *   `keys` (`'Fecha_Diaria'`): Columna seleccionada para ser el nuevo índice del DataFrame.
    *   `inplace` (`True`): Aplica los cambios directamente en el DataFrame actual.
*   **`pd.date_range`**:
    *   `start`: Fecha inicial de la secuencia.
    *   `end`: Fecha final de la secuencia.
    *   `freq` (`'D'`): Especifica un intervalo diario entre las fechas generadas.
*   **`df.reindex`**:
    *   `labels` (`full_range`): Nuevo índice que conformará la estructura.
    *   `fill_value` (`0`): Valor con el cual rellenar las nuevas filas que no tengan datos previos.
*   **`df.resample`**:
    *   `rule` (`'W-SUN'`): Frecuencia de remuestreo basada en semanas que culminan en domingo.
*   **`holidays.country_holidays`**:
    *   `country` (`'PE'`): Código internacional del país (Perú).
    *   `years` (`[2026]`): Años específicos de los cuales obtener la lista de días festivos.
*   **`pd.merge`**:
    *   `left`, `right`: DataFrames de origen a fusionar.
    *   `on` (`'ds'`): Nombre de la columna común que sirve de clave para la combinación.
    *   `how` (`'left'`): Tipo de fusión (mantiene todas las filas de la tabla izquierda).
*   **`df.fillna`**:
    *   `value` (`0`): Valor numérico para reemplazar valores nulos (`NaN`) resultantes de la fusión.""",

    "### Creación de Características 'Lag' y División de Conjuntos": """

#### Parámetros de las funciones aplicadas:
*   **`df.copy`**:
    *   Crea una copia profunda del DataFrame para evitar la modificación accidental del original.
*   **`series.shift`**:
    *   `periods` (`i`): Número de periodos (filas) a desplazar. En este caso se usa para referenciar ventas de semanas pasadas (lags).
*   **`series.astype`**:
    *   `dtype` (`int`): Convierte el tipo de datos de la serie al tipo especificado (entero).""",

    "### Entrenamiento y Evaluación de Prophet, Random Forest y XGBoost": """

#### Parámetros de las funciones aplicadas:
*   **`np.log1p`**:
    *   `x`: Variable a transformar mediante logaritmo natural de $1 + x$, utilizada para estabilizar la varianza de la variable objetivo.
*   **`Prophet`**:
    *   `growth` (`'linear'`): Configura una tendencia lineal para el crecimiento a largo plazo.
    *   `weekly_seasonality` (`False`): Desactiva el cálculo automático de estacionalidad semanal.
    *   `yearly_seasonality` (`False`): Desactiva el cálculo automático de estacionalidad anual.
*   **`m_prophet.add_regressor`**:
    *   `name`: Nombre del regresor exógeno que se añade como predictor (ej. `'is_nat'`, `'is_school'`).
*   **`m_prophet.fit`**:
    *   `df` (`train_p`): DataFrame de entrenamiento que contiene obligatoriamente las columnas `ds` (fecha), `y` (ventas) y regresores adicionales.
*   **`m_prophet.predict`**:
    *   `df` (`weekly_df`): DataFrame que contiene las fechas y regresores para los cuales se generará el pronóstico.
*   **`np.expm1`**:
    *   `x`: Inversa del logaritmo aplicada a las predicciones ($e^x - 1$) para devolverlas a su escala original de ventas.
*   **`series.clip`**:
    *   `lower` (`0`): Limita los valores inferiores a 0 para evitar que el pronóstico arroje ventas negativas imposibles.
*   **`RandomForestRegressor`**:
    *   `n_estimators` (`100`): Número de árboles que conformarán el bosque aleatorio.
    *   `random_state` (`42`): Garantiza que los resultados del bosque sean reproducibles en cada ejecución.
*   **`rf.fit` / `xgb.fit`**:
    *   `X` (`X_train`), `y` (`y_train`): Datos de entrenamiento de las variables predictoras y objetivo, respectivamente.
*   **`rf.predict` / `xgb.predict`**:
    *   `X` (`X_test`): Características de prueba para predecir el comportamiento de ventas.
*   **`XGBRegressor`**:
    *   `n_estimators` (`100`): Número de etapas consecutivas de estimación mediante árboles.
    *   `learning_rate` (`0.1`): Factor de reducción aplicado a la contribución de cada nuevo árbol para evitar sobreajuste.
    *   `random_state` (`42`): Semilla aleatoria de reproducibilidad.
*   **`mean_squared_error`**:
    *   `y_true` (`true`), `y_pred` (`pred`): Vectores de datos reales y predicciones correspondientes.
*   **`mean_absolute_percentage_error`**:
    *   `y_true` (`true`), `y_pred` (`pred`): Vectores de datos reales y predicciones para calcular el error porcentual absoluto medio.""",

    "### Visualización de Métricas de Rendimiento": """

#### Parámetros de las funciones aplicadas:
*   **`display`**:
    *   `objs` (`metrics_df`): DataFrame de pandas a renderizar con formato enriquecido en la interfaz del notebook.""",

    "### Análisis de Componentes del Modelo Prophet": """

#### Parámetros de las funciones aplicadas:
*   **`m_prophet.plot_components`**:
    *   `fcst` (`forecast_p_full`): DataFrame que contiene las predicciones detalladas de Prophet para desglosar la tendencia y regresores.""",

    "### Comparación Visual de Predicciones de Modelos": """

#### Parámetros de las funciones aplicadas:
*   **`plt.figure`**:
    *   `figsize` (`(12, 6)`): Define las dimensiones (ancho, alto) de la figura del gráfico.
*   **`plt.plot`**:
    *   `x` (`dates_test`), `y` (ej. `y_test`, `y_pred_p`): Coordenadas de los puntos a graficar.
    *   `fmt` (ej. `'ko-'`, `'g--'`): Estilo visual y color de la serie en el gráfico.
    *   `label`: Nombre asignado a la serie dentro de la leyenda.
    *   `linewidth` (`2`): Grosor de la línea que une los puntos.""",

    "### Establecimiento de Benchmark con Media Móvil Simple": """

#### Parámetros de las funciones aplicadas:
*   **`series.tail`**:
    *   `n` (`window`): Número de últimos registros a extraer de la serie.
*   **`series.mean`**:
    *   Calcula el promedio aritmético simple de los datos seleccionados.
*   **`np.full`**:
    *   `shape` (`test_size`): Tamaño de la dimensión del arreglo resultante.
    *   `fill_value` (`last_mean`): Valor constante para poblar cada celda del arreglo.
*   **`pd.concat`**:
    *   `objs`: Secuencia de DataFrames a unir.
    *   `ignore_index` (`True`): Restablece el índice consecutivo del nuevo DataFrame combinado.
*   **`joblib.dump`**:
    *   `value` (`ma_model_params`): Estructura u objeto de Python a serializar.
    *   `filename` (`'ma_model_params.joblib'`): Archivo de destino en el almacenamiento.""",

    "### Evaluación de Impacto de Regresores con Prophet Base": """

#### Parámetros de las funciones aplicadas:
*   **`Prophet`**:
    *   `growth` (`'linear'`), `weekly_seasonality` (`False`), `yearly_seasonality` (`False`): Configuración idéntica al modelo principal, pero sin incluir regresores exógenos, sirviendo como modelo base.
*   **`m_prophet_base.fit`**:
    *   `df` (`train_p_base`): Datos de entrenamiento del modelo Prophet base.
*   **`m_prophet_base.predict`**:
    *   `df` (`weekly_df`): Fechas históricas y de prueba a estimar por el modelo base.""",

    "### Visualización del Impacto de Regresores en Prophet": """

#### Parámetros de las funciones aplicadas:
*   **`plt.figure`**:
    *   `figsize` (`(12, 6)`): Configuración del tamaño del canvas de visualización.
*   **`plt.plot`**:
    *   `x`, `y`: Vectores de tiempo y ventas a graficar.
    *   `label`: Texto de la etiqueta de la leyenda.
    *   `marker` (ej. `'s'`, `'x'`): Símbolo geométrico para representar los puntos individuales.
    *   `linestyle` (ej. `':'`): Tipo de trazado (punteado, discontinuo, continuo).""",

    "### Generación y Visualización del Pronóstico Futuro": """

#### Parámetros de las funciones aplicadas:
*   **`final_model.make_future_dataframe`**:
    *   `periods` (`4`): Cantidad de semanas futuras a predecir.
    *   `freq` (`'W-SUN'`): Frecuencia semanal terminada en domingo para alinearse con los datos históricos.
*   **`series.rolling`**:
    *   `window` (`3`): Ventana de semanas consecutivas para calcular la media móvil histórica.
    *   `min_periods` (`1`): Número mínimo de semanas con datos necesarios para emitir un promedio parcial.
*   **`plt.fill_between`**:
    *   `x` (`ds`): Coordenadas de tiempo en el eje X.
    *   `y1` (`yhat_lower`), `y2` (`yhat_upper`): Límites inferior y superior que enmarcan la zona sombreada.
    *   `color` (`'magenta'`): Color del sombreado.
    *   `alpha` (`0.2`): Opacidad del sombreado para transparencia visual.
    *   `label`: Etiqueta del intervalo de confianza para la leyenda.""",

    "### Exportación del Modelo Prophet Final": """

#### Parámetros de las funciones aplicadas:
*   **`joblib.dump`**:
    *   `value` (`final_model`): Instancia del modelo Prophet final entrenado.
    *   `filename` (`'final_prophet_model.joblib'`): Nombre asignado al archivo serializado en disco."""
}

modified_cells_count = 0
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        source_str = "".join(cell.get('source', []))
        # Check if any of our headers is in this markdown cell
        matched_header = None
        for header in additions:
            if header in source_str:
                matched_header = header
                break
        
        if matched_header:
            # Check if parameters are already documented in this cell to avoid duplicate appends
            if "#### Parámetros de las funciones aplicadas:" in source_str:
                print(f"Cell with header '{matched_header}' already has parameter documentation. Skipping.")
                continue
            
            # Append the parameters documentation
            new_source_str = source_str + additions[matched_header]
            cell['source'] = to_notebook_source(new_source_str)
            print(f"Updated cell: {matched_header}")
            modified_cells_count += 1

if modified_cells_count > 0:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully modified {modified_cells_count} markdown cells and saved the notebook.")
else:
    print("No cells were modified.")
