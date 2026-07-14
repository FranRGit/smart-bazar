import pandas as pd
import numpy as np

print("=== 1. AUDITORÍA PROFUNDA DE VENTAS.CSV (CRUDO) ===")
df_v = pd.read_csv('datasets/crudo/ventas.csv', sep=';', encoding='utf-8-sig')
print("Dimensiones:", df_v.shape)
print("\nPrimeras 5 filas:")
print(df_v.head())
print("\nInformación de nulos y tipos de datos:")
print(df_v.info())
print("\nResumen descriptivo:")
print(df_v.describe(include='all'))
print("\nValores únicos de Metodo_Pago:", df_v['Metodo_Pago'].unique())
print("Muestras de Fecha cruda:", df_v['Fecha'].head(10).tolist())

print("\n=== 2. AUDITORÍA PROFUNDA DE DETALLE_VENTAS.CSV (CRUDO) ===")
df_d_raw = pd.read_csv('datasets/crudo/detalle_ventas.csv', sep=';', encoding='utf-8-sig')
print("Dimensiones crudas (sin skiprows):", df_d_raw.shape)
print("Primeras 3 filas crudas:\n", df_d_raw.head(3))

df_d = pd.read_csv('datasets/crudo/detalle_ventas.csv', sep=';', encoding='utf-8-sig', skiprows=1)
print("\nDimensiones con skiprows=1:", df_d.shape)
print("Columnas:", df_d.columns.tolist())
print("Conteo de nulos por columna:")
print(df_d.isnull().sum())
print("\nFilas con ID_Venta nulo o vacío:")
nulos_id = df_d[df_d['ID_Venta'].isnull()]
print(f"Cantidad de filas sin ID_Venta: {len(nulos_id)}")
print(nulos_id.head(5))

print("\n=== 3. AUDITORÍA PROFUNDA DE INVENTARIO.CSV (CRUDO) ===")
df_i = pd.read_csv('datasets/crudo/inventario.csv', sep=';', encoding='utf-8-sig', skiprows=1)
df_i = df_i.loc[:, ~df_i.columns.str.contains('^Unnamed')]
print("Dimensiones:", df_i.shape)
print("Conteo de nulos por columna:")
print(df_i.isnull().sum())
print("\nDistribución de Stock_Minimo crudo:")
print(df_i['Stock_Minimo'].value_counts(dropna=False).head(10))
print("\nResumen numérico de Stock_Minimo y Stock_Actual:")
print(df_i[['Stock_Minimo', 'Stock_Actual', 'Costo_Unitario', 'Precio_Venta']].describe(include='all'))
