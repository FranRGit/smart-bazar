import json
import os

# =====================================================================
# CUADERNO 1A: AUDITORÍA DE CALIDAD Y LIMPIEZA PROFUNDA DE DATOS
# =====================================================================
nb_1a = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Panel 1A: Auditoría de Calidad y Limpieza Profunda de Datos (Data Quality & Sanitization)\n",
        "\n",
        "Este cuaderno constituye la **Fase 1 del Pipeline Analítico** del proyecto **SmartBazar**. Su propósito fundamental es prevenir el antipatrón *Garbage In, Garbage Out* sometiendo los datos crudos (`datasets/crudo/`) a una exhaustiva auditoría y saneamiento antes de utilizarlos en análisis exploratorios o modelos de Machine Learning.\n",
        "\n",
        "### Objetivos de la Auditoría y Limpieza:\n",
        "1. **Inspección de Datos Crudos:** Visualizar las primeras filas (`head()`), estructura (`info()`), conteo de valores nulos (`isnull().sum()`) y medias/estadísticos (`describe()`).\n",
        "2. **Levantamiento de Observaciones Críticas de Negocio:**\n",
        "   - **Fechas y Horario (Registro por Lote):** Normalizar los distintos formatos de fecha y **excluir la variable hora** al identificarse que las ventas se digitan los fines de semana (por lo que la hora refleja la digitación y no la transacción real).\n",
        "   - **Detalle de Ventas:** Eliminar columnas vacías generadas por exportación (`Unnamed`) y sanear descripciones.\n",
        "   - **Inventario:** Eliminar filas vacías, corregir **stocks negativos** provocados por ventas sin ingreso registrado en kardex e imputar el **Stock Mínimo** diferenciado por departamento (evitando el valor por defecto genérico de 5).\n",
        "3. **Demostración Before vs. After y Exportación:** Verificación final de calidad y exportación del insumo oficial a `datasets/limpio/`."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import os\n",
        "\n",
        "pd.set_option('display.max_columns', 20)\n",
        "pd.set_option('display.width', 1000)\n",
        "print(\"Entorno configurado para auditoría de calidad de datos.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Auditoría de Entrada: Inspección de Datos Crudos (`datasets/crudo/`)\n",
        "Cargamos los archivos tal como salieron del sistema fuente para diagnosticar su calidad inicial (*Garbage In*)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "crudo_dir = 'datasets/crudo'\n",
        "\n",
        "df_ventas_raw = pd.read_csv(os.path.join(crudo_dir, 'ventas.csv'), sep=';', encoding='utf-8-sig')\n",
        "df_detalle_raw = pd.read_csv(os.path.join(crudo_dir, 'detalle_ventas.csv'), sep=';', encoding='utf-8-sig', skiprows=1)\n",
        "df_inv_raw = pd.read_csv(os.path.join(crudo_dir, 'inventario.csv'), sep=';', encoding='utf-8-sig', skiprows=1)\n",
        "\n",
        "print(\"=== AUDITORÍA INICIAL: VENTAS CRUDAS ===\")\n",
        "print(\"Dimensiones:\", df_ventas_raw.shape)\n",
        "print(\"\\nPrimeras 5 filas crudas (df.head()):\")\n",
        "print(df_ventas_raw.head())\n",
        "print(\"\\nValores nulos en ventas crudas:\")\n",
        "print(df_ventas_raw.isnull().sum())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"=== AUDITORÍA INICIAL: DETALLE DE VENTAS Y COLUMNAS FANTASMA ===\")\n",
        "print(\"Dimensiones crudas:\", df_detalle_raw.shape)\n",
        "print(\"Columnas presentes:\", df_detalle_raw.columns.tolist())\n",
        "print(\"\\nConteo de nulos por columna:\")\n",
        "print(df_detalle_raw.isnull().sum())\n",
        "print(\"\\nPrimeras 3 filas del detalle crudo:\")\n",
        "print(df_detalle_raw.head(3))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"=== AUDITORÍA INICIAL: INVENTARIO Y FILAS VACÍAS ===\")\n",
        "print(\"Dimensiones crudas:\", df_inv_raw.shape)\n",
        "print(\"\\nConteo de nulos por columna:\")\n",
        "print(df_inv_raw.isnull().sum())\n",
        "print(\"\\nEstadísticas descriptivas (media, min, max) de variables numéricas crudas:\")\n",
        "print(df_inv_raw[['Stock_Minimo', 'Stock_Actual', 'Costo_Unitario', 'Precio_Venta']].describe(include='all'))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Levantamiento de Observaciones y Saneamiento Específico\n",
        "\n",
        "### Observación 1: Fechas con Formatos Mixtos y Hora no Representativa\n",
        "- **Diagnóstico:** En `ventas.csv` coexisten fechas con formato corto (`DD/MM/YY`) y formato largo (`M/D/YYYY HH:MM:SS`). Más importante aún, **la hora no representa el momento de compra real**, ya que el bazar digita los comprobantes por lotes los fines de semana.\n",
        "- **Acción de Limpieza:** Normalizar la columna `Fecha` al formato canónico `YYYY-MM-DD` y **descartar el uso de la hora** para clustering o análisis de demanda horaria."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"Muestras de Fecha antes de normalizar:\", df_ventas_raw['Fecha'].unique()[:5])\n",
        "\n",
        "# Limpieza de Ventas\n",
        "df_ventas_clean = df_ventas_raw.dropna(subset=['ID', 'Total']).copy()\n",
        "df_ventas_clean['Fecha'] = pd.to_datetime(df_ventas_clean['Fecha'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')\n",
        "df_ventas_clean['Metodo_Pago'] = df_ventas_clean['Metodo_Pago'].astype(str).str.strip().str.upper()\n",
        "df_ventas_clean['Total'] = pd.to_numeric(df_ventas_clean['Total'], errors='coerce').fillna(0.0)\n",
        "\n",
        "print(\"\\nMuestras de Fecha normalizada (YYYY-MM-DD sin ruido horario):\")\n",
        "print(df_ventas_clean[['ID', 'Fecha', 'Metodo_Pago', 'Total']].head())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Observación 2: Saneamiento de Detalle de Ventas\n",
        "- **Diagnóstico:** Existen 4 columnas vacías (`Unnamed: 0`, `Unnamed: 10`, `Unnamed: 11`, `Unnamed: 12`) y descripciones con espacios en blanco o faltantes.\n",
        "- **Acción de Limpieza:** Filtrar columnas `Unnamed`, sanear `ID_Venta` y convertir numéricos preservando solo transacciones válidas."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df_detalle_clean = df_detalle_raw.loc[:, ~df_detalle_raw.columns.str.contains('^Unnamed')].copy()\n",
        "df_detalle_clean = df_detalle_clean.dropna(subset=['ID_Venta']).copy()\n",
        "df_detalle_clean['ID_Venta'] = df_detalle_clean['ID_Venta'].astype(str).str.strip()\n",
        "df_detalle_clean['Fecha'] = pd.to_datetime(df_detalle_clean['Fecha'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')\n",
        "df_detalle_clean['Descripcion'] = df_detalle_clean['Descripcion'].fillna('SIN DESCRIPCION').astype(str).str.strip()\n",
        "\n",
        "for col in ['Cantidad', 'Precio_Unitario', 'Subtotal']:\n",
        "    df_detalle_clean[col] = pd.to_numeric(df_detalle_clean[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)\n",
        "    df_detalle_clean[col] = df_detalle_clean[col].clip(lower=0)\n",
        "\n",
        "print(\"Dimensiones detalle saneado:\", df_detalle_clean.shape)\n",
        "print(\"Nulos restantes en detalle:\", df_detalle_clean.isnull().sum().sum())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### Observación 3: Inventario - Filas Vacías, Stocks Negativos y Stock Mínimo Diferenciado\n",
        "- **Diagnóstico 3A:** 511 filas en `inventario.csv` son completamente nulas/vacías.\n",
        "- **Diagnóstico 3B (Stocks Negativos):** Artículos de alta rotación (ej. `HOJA DE COLORES` con -184) presentan stock negativo por ventas realizadas antes del registro de entrada en almacén.\n",
        "- **Diagnóstico 3C (Stock Mínimo):** El 89% tiene `Stock_Minimo` en nulo y lo existente está sesgado al valor 5.\n",
        "- **Acción de Limpieza:**\n",
        "  1. Eliminar las 511 filas vacías.\n",
        "  2. Ajustar stocks negativos a `0` físico creando la bandera de auditoría `Alerta_Kardex_Negativo`.\n",
        "  3. Imputar `Stock_Minimo` por departamento: **5** para Útiles/Librería y **2** para Fotocopiadora/Servicios."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df_inv_clean = df_inv_raw.loc[:, ~df_inv_raw.columns.str.contains('^Unnamed')].copy()\n",
        "df_inv_clean = df_inv_clean.dropna(subset=['ID']).copy()\n",
        "\n",
        "for col in ['Costo_Unitario', 'Precio_Venta']:\n",
        "    df_inv_clean[col] = pd.to_numeric(df_inv_clean[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)\n",
        "for col in ['Stock_Minimo', 'Stock_Actual']:\n",
        "    df_inv_clean[col] = pd.to_numeric(df_inv_clean[col], errors='coerce')\n",
        "\n",
        "# Identificar stocks negativos\n",
        "negativos = df_inv_clean[df_inv_clean['Stock_Actual'] < 0]\n",
        "print(f\"Artículos detectados con Stock_Actual negativo (desfase de kardex): {len(negativos)}\")\n",
        "print(negativos[['ID', 'Descripcion', 'Departamento', 'Stock_Actual']].head(5))\n",
        "\n",
        "# Saneamiento\n",
        "df_inv_clean['Stock_Minimo'] = df_inv_clean.apply(\n",
        "    lambda r: 5 if pd.isna(r['Stock_Minimo']) and str(r['Departamento']).upper() == 'UTILES'\n",
        "    else (2 if pd.isna(r['Stock_Minimo']) else r['Stock_Minimo']),\n",
        "    axis=1\n",
        ").astype(int)\n",
        "\n",
        "df_inv_clean['Alerta_Kardex_Negativo'] = df_inv_clean['Stock_Actual'] < 0\n",
        "df_inv_clean['Stock_Actual'] = df_inv_clean['Stock_Actual'].fillna(0).clip(lower=0).astype(int)\n",
        "\n",
        "print(\"\\nDistribución final de Stock_Minimo saneado por Departamento:\")\n",
        "print(df_inv_clean.groupby('Departamento')['Stock_Minimo'].value_counts())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Auditoría Post-Limpieza (Before vs. After)\n",
        "Validamos estadísticamente la calidad final de los datasets limpios (`df.head()`, media de variables, nulos en 0)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"=== RESUMEN DE CALIDAD POST-LIMPIEZA ===\")\n",
        "print(f\"Ventas:       {df_ventas_clean.shape[0]} registros | Nulos: {df_ventas_clean.isnull().sum().sum()}\")\n",
        "print(f\"Detalle:      {df_detalle_clean.shape[0]} registros | Nulos: {df_detalle_clean.isnull().sum().sum()}\")\n",
        "print(f\"Inventario:   {df_inv_clean.shape[0]} registros | Nulos: {df_inv_clean.isnull().sum().sum()}\")\n",
        "\n",
        "print(\"\\nEstadísticas descriptivas limpias de Inventario (medias correctas):\")\n",
        "print(df_inv_clean[['Stock_Minimo', 'Stock_Actual', 'Costo_Unitario', 'Precio_Venta']].describe().round(2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Exportación del Insumo Saneado a `datasets/limpio/`\n",
        "Guardamos los archivos sin BOM ni columnas basura para alimentar al Cuaderno 1B (EDA y Clustering) y demás paneles del equipo."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "limpio_dir = 'datasets/limpio'\n",
        "os.makedirs(limpio_dir, exist_ok=True)\n",
        "\n",
        "df_ventas_clean.to_csv(os.path.join(limpio_dir, 'ventas.csv'), index=False, encoding='utf-8')\n",
        "df_detalle_clean.to_csv(os.path.join(limpio_dir, 'detalle_ventas.csv'), index=False, encoding='utf-8')\n",
        "df_detalle_clean.to_csv(os.path.join(limpio_dir, 'detalle-ventas.csv'), index=False, encoding='utf-8')\n",
        "df_inv_clean.to_csv(os.path.join(limpio_dir, 'inventario.csv'), index=False, encoding='utf-8')\n",
        "\n",
        "print(\"[OK] Datasets limpios y auditados exportados exitosamente a datasets/limpio/\")"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}

# =====================================================================
# CUADERNO 1B: EXTRACCIÓN DE CARACTERÍSTICAS, EDA DE NEGOCIO Y CLUSTERING
# =====================================================================
nb_1b = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Panel 1B: Extracción de Características, EDA de Negocio y Segmentación K-Means\n",
        "\n",
        "Este cuaderno constituye la **Fase 2 del Panel 1** y parte **exclusivamente de los datos limpios y auditados (`datasets/limpio/`)** generados en el Cuaderno 1A.\n",
        "\n",
        "### Componentes del Cuaderno:\n",
        "1. **Análisis Exploratorio de Negocio (Macro-EDA):** Comportamiento de ventas por día, preferencia de métodos de pago (**66.3% Efectivo vs 33.7% Yape**) y salud del inventario con alertas de quiebre reales.\n",
        "2. **Extracción de Características por Ticket (`Feature Engineering`):**\n",
        "   - Al haberse determinado que la hora de compra no es representativa (por registro en lote los fines de semana), se construye una matriz de características de **comportamiento real del comprador**:\n",
        "     * `Total` (Monto pagado S/).\n",
        "     * `n_items` (Volumen físico de unidades compradas).\n",
        "     * `diversidad_productos` (Cantidad de productos distintos en el ticket).\n",
        "3. **Detección Estadística de Outliers (1.5·IQR):** Identificación visual y analítica de compras mayoristas/institucionales.\n",
        "4. **Segmentación K-Means:** Evaluación óptima con **Método del Codo** y **Coeficiente de Silueta** ($K=3$) y perfilamiento de clientes."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import silhouette_score\n",
        "\n",
        "sns.set_theme(style='whitegrid', palette='muted')\n",
        "plt.rcParams['figure.figsize'] = (10, 5)\n",
        "print(\"Librerías de análisis y clustering importadas.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Carga de Datos Limpios (`datasets/limpio/`)\n",
        "Verificamos la carga de la data saneada con cero valores nulos y tipos normalizados."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "limpio_dir = 'datasets/limpio'\n",
        "df_ventas = pd.read_csv(f'{limpio_dir}/ventas.csv')\n",
        "df_detalle = pd.read_csv(f'{limpio_dir}/detalle_ventas.csv')\n",
        "df_inv = pd.read_csv(f'{limpio_dir}/inventario.csv')\n",
        "\n",
        "df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'])\n",
        "print(f\"[OK] Ventas limpias: {df_ventas.shape}\")\n",
        "print(f\"[OK] Detalle limpio: {df_detalle.shape}\")\n",
        "print(f\"[OK] Inventario limpio: {df_inv.shape}\")\n",
        "print(\"Primeras filas de ventas limpias:\")\n",
        "print(df_ventas.head(3))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Análisis Exploratorio de Negocio (Macro-EDA)\n",
        "### 2.1 KPIs Operativos y Preferencia de Pago"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "total_ingresos = df_ventas['Total'].sum()\n",
        "ticket_promedio = df_ventas['Total'].mean()\n",
        "n_transacciones = len(df_ventas)\n",
        "\n",
        "print(f\"--- RESUMEN FINANCIERO DEL BAZAR ---\")\n",
        "print(f\"Ingresos Totales Acumulados: S/ {total_ingresos:,.2f}\")\n",
        "print(f\"Total de Tickets Emitidos:    {n_transacciones:,}\")\n",
        "print(f\"Ticket Promedio (Media):      S/ {ticket_promedio:,.2f}\")\n",
        "\n",
        "pago_dist = df_ventas['Metodo_Pago'].value_counts()\n",
        "pago_pct = df_ventas['Metodo_Pago'].value_counts(normalize=True) * 100\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
        "sns.barplot(x=pago_dist.index, y=pago_dist.values, hue=pago_dist.index, ax=ax1, palette=['#2b5c8f', '#d95f02'], legend=False)\n",
        "ax1.set_title('Volumen por Método de Pago')\n",
        "ax1.set_ylabel('Transacciones')\n",
        "\n",
        "ax2.pie(pago_pct, labels=pago_pct.index, autopct='%1.1f%%', colors=['#2b5c8f', '#d95f02'], startangle=140)\n",
        "ax2.set_title('Proporción de Uso de Método de Pago')\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 2.2 Demanda por Día de la Semana\n",
        "Al haberse descartado la hora por la digitación de fin de semana, analizamos el comportamiento diario de las ventas registradas."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df_ventas['Dia_Semana'] = df_ventas['Fecha'].dt.day_name()\n",
        "orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
        "\n",
        "plt.figure(figsize=(10, 4))\n",
        "sns.countplot(data=df_ventas, x='Dia_Semana', order=[d for d in orden_dias if d in df_ventas['Dia_Semana'].unique()], color='#2ecc71')\n",
        "plt.title('Distribución de Tickets por Día de la Semana')\n",
        "plt.xlabel('Día de la Semana')\n",
        "plt.ylabel('Cantidad de Ventas')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 2.3 Valorización de Almacén y Alertas Reales de Reposición\n",
        "Sobre el inventario saneado identificamos los artículos con `Stock_Actual <= Stock_Minimo` que requieren orden de compra urgente."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df_inv['Valor_Inventario'] = df_inv['Stock_Actual'] * df_inv['Costo_Unitario']\n",
        "valor_almacen = df_inv['Valor_Inventario'].sum()\n",
        "alertas = df_inv[df_inv['Stock_Actual'] <= df_inv['Stock_Minimo']]\n",
        "\n",
        "print(f\"Valor Total en Almacén (Saneado): S/ {valor_almacen:,.2f}\")\n",
        "print(f\"Artículos con Alerta de Reposición Urgente: {len(alertas)} de {len(df_inv)} ({len(alertas)/len(df_inv):.1%})\")\n",
        "print(\"\\nTop 5 Ítems con Stock Crítico:\")\n",
        "print(alertas[['Descripcion', 'Departamento', 'Stock_Minimo', 'Stock_Actual']].head(5).to_string(index=False))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Extracción de Características por Ticket (`Feature Engineering`)\n",
        "En reemplazo de la variable horaria (ruido artificial), creamos una matriz representativa del **comportamiento real de compra** de cada ticket:\n",
        "- `Total`: Monto en Soles.\n",
        "- `n_items`: Unidades físicas totales adquiridas.\n",
        "- `diversidad_productos`: Cantidad de productos únicos dentro de la misma compra."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "agg_ticket = df_detalle.groupby('ID_Venta').agg(\n",
        "    n_items=('Cantidad', 'sum'),\n",
        "    diversidad_productos=('ID_Producto', 'nunique')\n",
        ").reset_index()\n",
        "\n",
        "df_cluster = df_ventas.merge(agg_ticket, left_on='ID', right_on='ID_Venta', how='inner')\n",
        "features = ['Total', 'n_items', 'diversidad_productos']\n",
        "X = df_cluster[features].copy()\n",
        "\n",
        "print(\"Primeras filas de las Características Extraídas por Ticket:\")\n",
        "print(X.head())\n",
        "print(\"\\nEstadísticas descriptivas de las Características:\")\n",
        "print(X.describe().round(2))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Detección Estadística de Outliers (Regla 1.5·IQR)\n",
        "Calculamos los cuartiles Q1, Q3 y el umbral superior para aislar compras institucionales o mayoristas."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "q1 = X['Total'].quantile(0.25)\n",
        "q3 = X['Total'].quantile(0.75)\n",
        "iqr = q3 - q1\n",
        "umbral_superior = q3 + 1.5 * iqr\n",
        "\n",
        "outliers = df_cluster[df_cluster['Total'] > umbral_superior]\n",
        "print(f\"Q1: S/ {q1:.2f} | Q3: S/ {q3:.2f} | IQR: S/ {iqr:.2f}\")\n",
        "print(f\"Umbral de Outlier Superior: S/ {umbral_superior:.2f}\")\n",
        "print(f\"Tickets Outliers Detectados: {len(outliers)} ({len(outliers)/len(df_cluster):.2%})\")\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
        "sns.boxplot(y=df_cluster['Total'], ax=ax1, color='#e74c3c')\n",
        "ax1.set_title('Boxplot de Ventas con Umbral Outlier')\n",
        "ax1.axhline(umbral_superior, color='black', linestyle='--', label=f'Umbral S/ {umbral_superior:.1f}')\n",
        "ax1.legend()\n",
        "\n",
        "sns.histplot(df_cluster['Total'], bins=40, kde=True, ax=ax2, color='#34495e')\n",
        "ax2.set_title('Distribución de Montos con Umbral')\n",
        "ax2.axvline(umbral_superior, color='red', linestyle='--', label='Umbral Outlier')\n",
        "ax2.legend()\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 5. Segmentación de Clientes con K-Means\n",
        "### 5.1 Estandarización y Selección de K (Codo y Silueta)\n",
        "Estandarizamos las variables con `StandardScaler` y justificamos la elección de $K=3$ analizando la inercia y el score de silueta."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "scaler = StandardScaler()\n",
        "X_scaled = scaler.fit_transform(X)\n",
        "\n",
        "inertias = []\n",
        "silhouettes = []\n",
        "k_range = range(2, 9)\n",
        "\n",
        "for k in k_range:\n",
        "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
        "    labels = km.fit_predict(X_scaled)\n",
        "    inertias.append(km.inertia_)\n",
        "    silhouettes.append(silhouette_score(X_scaled, labels))\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n",
        "ax1.plot(k_range, inertias, 'bo-', marker='o', linewidth=2)\n",
        "ax1.set_title('Método del Codo (Inercia vs K)')\n",
        "ax1.set_xlabel('Número de Clústeres (K)')\n",
        "ax1.set_ylabel('Inercia')\n",
        "ax1.grid(True)\n",
        "\n",
        "ax2.plot(k_range, silhouettes, 'ro-', marker='s', linewidth=2)\n",
        "ax2.set_title('Coeficiente de Silueta vs K')\n",
        "ax2.set_xlabel('Número de Clústeres (K)')\n",
        "ax2.set_ylabel('Score de Silueta')\n",
        "ax2.grid(True)\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 5.2 Ajuste Final ($K=3$) y Perfilamiento Ejecutivo del Negocio\n",
        "Entrenamos el modelo con 3 clústeres e interpretamos cada segmento para la estrategia comercial del bazar."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "K_OPTIMO = 3\n",
        "kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10)\n",
        "df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)\n",
        "\n",
        "print(f\"Coeficiente de Silueta Final para K={K_OPTIMO}: {silhouette_score(X_scaled, df_cluster['Cluster']):.4f}\")\n",
        "\n",
        "plt.figure(figsize=(9, 5))\n",
        "sns.scatterplot(data=df_cluster, x='Total', y='n_items', hue='Cluster', palette='Set1', style='Cluster', s=80, alpha=0.85)\n",
        "plt.title(f'Segmentación K-Means de Comportamiento de Compra (K={K_OPTIMO})')\n",
        "plt.xlabel('Monto Total del Ticket (S/)')\n",
        "plt.ylabel('Cantidad Total de Ítems Adquiridos')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "perfiles = df_cluster.groupby('Cluster')[['Total', 'n_items', 'diversidad_productos']].mean().round(2)\n",
        "perfiles['Cantidad_Tickets'] = df_cluster['Cluster'].value_counts()\n",
        "perfiles['Participacion_%'] = (perfiles['Cantidad_Tickets'] / len(df_cluster) * 100).round(1)\n",
        "print(\"=== PERFILAMIENTO EJECUTIVO DE LOS CLÚSTERES DE CLIENTES ===\")\n",
        "print(perfiles)"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}

# GUARDAR CUADERNOS DIVIDIDOS Y VERSIÓN CONSOLIDADA
with open('Panel 1A_ Auditoria_y_Limpieza_Datos.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_1a, f, indent=2, ensure_ascii=False)

with open('Panel 1B_ EDA_y_Clustering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_1b, f, indent=2, ensure_ascii=False)

# Actualizar el cuaderno consolidado Panel 1_ EDA_Clustering.ipynb integrando las 2 partes
nb_consolidado = {
  "cells": nb_1a["cells"] + nb_1b["cells"][2:], # Evitamos duplicar la importación inicial de librerías
  "metadata": nb_1a["metadata"],
  "nbformat": 4,
  "nbformat_minor": 2
}

with open('Panel 1_ EDA_Clustering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_consolidado, f, indent=2, ensure_ascii=False)

print("Cuadernos Panel 1A, Panel 1B y Panel 1 (consolidado) generados con éxito.")
