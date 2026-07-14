import json
import os

notebook_content = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Panel 1: Análisis Exploratorio General (EDA), Limpieza Estructural y Segmentación (Clustering)\n",
        "\n",
        "Este cuaderno actúa como el **módulo de entrada (Macro-EDA)** dentro del pipeline analítico del proyecto **SmartBazar**:\n",
        "1. **Limpieza Estructural y Flujo de Datos:** Carga los archivos originales transaccionales desde la carpeta `datasets/crudo/`, elimina errores estructurales de exportación (BOM UTF-8, columnas fantasma `Unnamed`, delimitadores vacíos) y **exporta la versión canónica y limpia** hacia `datasets/limpio/`. Este conjunto limpio sirve como insumo estandarizado para los demás paneles y modelos del equipo.\n",
        "2. **EDA Detallado del Negocio:** Explora visual y estadísticamente el comportamiento de las ventas, las horas pico, los métodos de pago predominantes y las alertas de inventario (quiebres de stock).\n",
        "3. **Detección de Outliers & Segmentación K-Means:** Identifica transacciones atípicas mediante la regla **1.5·IQR** y segmenta los tickets de venta en perfiles de consumo utilizando **K-Means**, validando el número óptimo de clústeres mediante el **Método del Codo** y el **Coeficiente de Silueta**."
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
        "import os\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import silhouette_score\n",
        "\n",
        "sns.set_theme(style='whitegrid', palette='muted')\n",
        "plt.rcParams['figure.figsize'] = (10, 5)\n",
        "print(\"Librerías importadas correctamente.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 1. Carga de Datos Crudos (`datasets/crudo/`) y Saneamiento Estructural\n",
        "Leemos las fuentes transaccionales originales solucionando los artefactos típicos de exportación (punto y coma inicial, filas vacías de delimitadores, formato canónico de fechas y numéricos)."
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
        "# 1. Ventas\n",
        "df_ventas = pd.read_csv(os.path.join(crudo_dir, 'ventas.csv'), sep=';', encoding='utf-8-sig')\n",
        "df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'], format='mixed', errors='coerce')\n",
        "df_ventas['Metodo_Pago'] = df_ventas['Metodo_Pago'].astype(str).str.strip().str.capitalize()\n",
        "\n",
        "# 2. Detalle de Ventas (saltando fila vacía inicial y columnas Unnamed)\n",
        "df_detalle = pd.read_csv(os.path.join(crudo_dir, 'detalle_ventas.csv'), sep=';', encoding='utf-8-sig', skiprows=1)\n",
        "df_detalle = df_detalle.loc[:, ~df_detalle.columns.str.contains('^Unnamed')]\n",
        "df_detalle['Fecha'] = pd.to_datetime(df_detalle['Fecha'], format='mixed', errors='coerce')\n",
        "df_detalle = df_detalle.dropna(subset=['ID_Venta'])\n",
        "df_detalle['ID_Venta'] = df_detalle['ID_Venta'].astype(str).str.strip()\n",
        "for col in ['Cantidad', 'Precio_Unitario', 'Subtotal']:\n",
        "    if col in df_detalle.columns:\n",
        "        df_detalle[col] = pd.to_numeric(df_detalle[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)\n",
        "\n",
        "# 3. Inventario\n",
        "df_inventario = pd.read_csv(os.path.join(crudo_dir, 'inventario.csv'), sep=';', encoding='utf-8-sig', skiprows=1)\n",
        "df_inventario = df_inventario.loc[:, ~df_inventario.columns.str.contains('^Unnamed')]\n",
        "for col in ['Costo_Unitario', 'Precio_Venta']:\n",
        "    if col in df_inventario.columns:\n",
        "        df_inventario[col] = pd.to_numeric(df_inventario[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)\n",
        "for col in ['Stock_Minimo', 'Stock_Actual']:\n",
        "    if col in df_inventario.columns:\n",
        "        df_inventario[col] = pd.to_numeric(df_inventario[col], errors='coerce').fillna(0).astype(int)\n",
        "df_inventario = df_inventario.dropna(subset=['ID'])\n",
        "\n",
        "print(f\"[OK] Ventas cargadas: {df_ventas.shape}\")\n",
        "print(f\"[OK] Detalle cargado: {df_detalle.shape}\")\n",
        "print(f\"[OK] Inventario cargado: {df_inventario.shape}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 2. Exportación de Datos Limpios a `datasets/limpio/`\n",
        "Una vez normalizada la estructura de los datos, guardamos los archivos en `datasets/limpio/` en formato UTF-8 estándar. Estos archivos son el **insumo oficial** para los modelos analíticos específicos (Paneles 2, 3 y las reglas de asociación)."
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
        "df_ventas.to_csv(os.path.join(limpio_dir, 'ventas.csv'), index=False, encoding='utf-8')\n",
        "df_detalle.to_csv(os.path.join(limpio_dir, 'detalle_ventas.csv'), index=False, encoding='utf-8')\n",
        "df_detalle.to_csv(os.path.join(limpio_dir, 'detalle-ventas.csv'), index=False, encoding='utf-8') # Compatibilidad\n",
        "df_inventario.to_csv(os.path.join(limpio_dir, 'inventario.csv'), index=False, encoding='utf-8')\n",
        "\n",
        "print(\"[OK] Datasets limpios exportados exitosamente a datasets/limpio/\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 3. Análisis Exploratorio General del Negocio (Macro-EDA)\n",
        "### 3.1 KPIs Principales y Proporción por Método de Pago\n",
        "Analizamos el ingreso total acumulado, ticket promedio y la preferencia de pago de los clientes."
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
        "print(f\"--- KPIs FINANCIEROS Y OPERATIVOS ---\")\n",
        "print(f\"Ingresos Totales Registrados: S/ {total_ingresos:,.2f}\")\n",
        "print(f\"Total de Ventas / Tickets:     {n_transacciones:,}\")\n",
        "print(f\"Ticket Promedio por Venta:     S/ {ticket_promedio:,.2f}\")\n",
        "\n",
        "# Distribución por Método de Pago\n",
        "pago_dist = df_ventas['Metodo_Pago'].value_counts()\n",
        "pago_pct = df_ventas['Metodo_Pago'].value_counts(normalize=True) * 100\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
        "sns.barplot(x=pago_dist.index, y=pago_dist.values, hue=pago_dist.index, ax=ax1, palette=['#2b5c8f', '#d95f02'], legend=False)\n",
        "ax1.set_title('Transacciones por Método de Pago')\n",
        "ax1.set_ylabel('Cantidad de Transacciones')\n",
        "\n",
        "ax2.pie(pago_pct, labels=pago_pct.index, autopct='%1.1f%%', colors=['#2b5c8f', '#d95f02'], startangle=140)\n",
        "ax2.set_title('Porcentaje de Uso de Método de Pago')\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 3.2 Patrones Temporales de Venta (Día de la Semana y Hora de Compra)\n",
        "Identificar en qué horarios y días se concentra la demanda permite optimizar el personal en mostrador y la provisión de cambio/efectivo en caja."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df_ventas['Hora'] = df_ventas['Fecha'].dt.hour\n",
        "df_ventas['Dia_Semana'] = df_ventas['Fecha'].dt.day_name()\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))\n",
        "sns.countplot(data=df_ventas, x='Hora', ax=ax1, color='#3498db')\n",
        "ax1.set_title('Distribución de Ventas por Hora del Día')\n",
        "ax1.set_xlabel('Hora del Día (24h)')\n",
        "ax1.set_ylabel('Número de Ventas')\n",
        "\n",
        "orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
        "sns.countplot(data=df_ventas, x='Dia_Semana', order=[d for d in orden_dias if d in df_ventas['Dia_Semana'].unique()], ax=ax2, color='#2ecc71')\n",
        "ax2.set_title('Ventas según Día de la Semana')\n",
        "ax2.set_xlabel('Día')\n",
        "ax2.set_ylabel('Número de Ventas')\n",
        "ax2.tick_params(axis='x', rotation=30)\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 3.3 Salud del Inventario y Alertas de Quiebre de Stock\n",
        "Evaluamos el catálogo de productos para detectar artículos en situación de riesgo (`Stock_Actual < Stock_Minimo`)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "df_inventario['Valor_Inventario'] = df_inventario['Stock_Actual'] * df_inventario['Costo_Unitario']\n",
        "valor_almacen = df_inventario['Valor_Inventario'].sum()\n",
        "\n",
        "alertas_quiebre = df_inventario[df_inventario['Stock_Actual'] <= df_inventario['Stock_Minimo']]\n",
        "\n",
        "print(f\"Valor Total Monetario en Almacén: S/ {valor_almacen:,.2f}\")\n",
        "print(f\"Productos con Alerta de Quiebre o Stock Crítico: {len(alertas_quiebre)} de {len(df_inventario)} artículos ({len(alertas_quiebre)/len(df_inventario):.1%})\")\n",
        "\n",
        "if len(alertas_quiebre) > 0:\n",
        "    print(\"\\n--- Top 5 Productos en Alerta Crítica de Reposición ---\")\n",
        "    print(alertas_quiebre[['Descripcion', 'Departamento', 'Stock_Minimo', 'Stock_Actual']].head(5).to_string(index=False))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 4. Detección Estadística de Outliers (Regla 1.5·IQR)\n",
        "Identificamos transacciones con montos que se alejan de la distribución típica del bazar (ej. compras mayoristas institucionales o institutos cercanos)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "q1 = df_ventas['Total'].quantile(0.25)\n",
        "q3 = df_ventas['Total'].quantile(0.75)\n",
        "iqr = q3 - q1\n",
        "umbral_superior = q3 + 1.5 * iqr\n",
        "\n",
        "outliers = df_ventas[df_ventas['Total'] > umbral_superior]\n",
        "print(f\"Q1: S/ {q1:.2f} | Q3: S/ {q3:.2f} | IQR: S/ {iqr:.2f}\")\n",
        "print(f\"Umbral Superior de Outlier: S/ {umbral_superior:.2f}\")\n",
        "print(f\"Total de Tickets Outliers Detectados: {len(outliers)} ({len(outliers)/len(df_ventas):.2%})\")\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\n",
        "sns.boxplot(y=df_ventas['Total'], ax=ax1, color='#e74c3c')\n",
        "ax1.set_title('Boxplot del Monto Total (S/)')\n",
        "ax1.axhline(umbral_superior, color='black', linestyle='--', label=f'Umbral Outlier (S/{umbral_superior:.1f})')\n",
        "ax1.legend()\n",
        "\n",
        "sns.histplot(df_ventas['Total'], bins=40, kde=True, ax=ax2, color='#34495e')\n",
        "ax2.set_title('Distribución Histograma con Umbral de Outliers')\n",
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
        "## 5. Segmentación de Tickets de Venta (Clustering K-Means)\n",
        "Agrupamos las transacciones combinando el monto de venta (`Total`), el número de artículos adquiridos (`n_items`) y la hora de compra (`hora_compra`)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Construcción del dataset agregado por ticket\n",
        "agg_detail = df_detalle.groupby('ID_Venta').agg(\n",
        "    n_items=('Cantidad', 'sum')\n",
        ").reset_index()\n",
        "\n",
        "cluster_df = df_ventas.merge(agg_detail, left_on='ID', right_on='ID_Venta', how='inner')\n",
        "features = ['Total', 'n_items', 'Hora']\n",
        "X = cluster_df[features].dropna()\n",
        "\n",
        "scaler = StandardScaler()\n",
        "X_scaled = scaler.fit_transform(X)\n",
        "print(f\"Dimensiones de matriz escalada para K-Means: {X_scaled.shape}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 5.1 Selección Óptima de K (Método del Codo y Coeficiente de Silueta)\n",
        "Evaluamos el rango $K \\in [2, 8]$ para justificar de forma cuantitativa la elección de clústeres."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
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
        "ax1.set_ylabel('Inercia (SSE)')\n",
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
        "### 5.2 Entrenamiento y Perfilamiento Final ($K=3$)\n",
        "Ajustamos el modelo final con 3 clústeres e interpretamos su significado en la operativa del bazar."
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
        "cluster_df['Cluster'] = kmeans.fit_predict(X_scaled)\n",
        "\n",
        "score_final = silhouette_score(X_scaled, cluster_df['Cluster'])\n",
        "print(f\"Coeficiente de Silueta para K={K_OPTIMO}: {score_final:.4f}\")\n",
        "\n",
        "# Scatterplot\n",
        "plt.figure(figsize=(9, 5))\n",
        "sns.scatterplot(data=cluster_df, x='Total', y='n_items', hue='Cluster', palette='Set1', style='Cluster', s=80, alpha=0.85)\n",
        "plt.title(f'Segmentación K-Means de Tickets (K={K_OPTIMO})')\n",
        "plt.xlabel('Monto Total del Ticket (S/)')\n",
        "plt.ylabel('Cantidad Total de Artículos Adquiridos')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Tabla Perfiles Promedio por Clúster\n",
        "perfiles = cluster_df.groupby('Cluster')[['Total', 'n_items', 'Hora']].mean().round(2)\n",
        "perfiles['Cantidad_Tickets'] = cluster_df['Cluster'].value_counts()\n",
        "perfiles['Participacion_%'] = (perfiles['Cantidad_Tickets'] / len(cluster_df) * 100).round(1)\n",
        "print(\"=== PERFILAMIENTO DE LOS CLÚSTERES DE TICKETS ===\")\n",
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

out_path = 'Panel 1_ EDA_Clustering.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2, ensure_ascii=False)

print(f"Notebook {out_path} actualizado exitosamente.")
