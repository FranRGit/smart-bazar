import pandas as pd
import os
import streamlit as st

def get_datasets_path(subfolder=""):
    """Gets the path to the datasets folder or a subfolder (crudo/limpio)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_path = os.path.join(base_dir, "datasets")
    if not os.path.exists(datasets_path):
        datasets_path = "datasets"
    if subfolder:
        target_path = os.path.join(datasets_path, subfolder)
        if os.path.exists(target_path):
            return target_path
    return datasets_path

def _find_file(base_path, candidates):
    for c in candidates:
        p = os.path.join(base_path, c)
        if os.path.exists(p):
            return p
    return os.path.join(base_path, candidates[0])

def load_raw_ventas(base_path=None):
    """Loads the raw sales file from datasets/crudo."""
    if base_path is None:
        base_path = get_datasets_path("crudo")
    path = _find_file(base_path, ["ventas.csv"])
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed', errors='coerce')
    df['Metodo_Pago'] = df['Metodo_Pago'].astype(str).str.strip().str.capitalize()
    return df

def load_raw_detalle_ventas(base_path=None):
    """Loads the raw sales detail file from datasets/crudo, skipping empty headers and Unnamed columns."""
    if base_path is None:
        base_path = get_datasets_path("crudo")
    path = _find_file(base_path, ["detalle_ventas.csv", "detalle-ventas.csv"])
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig', skiprows=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed', errors='coerce')
    df = df.dropna(subset=['ID_Venta'])
    df['ID_Venta'] = df['ID_Venta'].astype(str).str.strip()
    for col in ['Cantidad', 'Precio_Unitario', 'Subtotal']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)
    return df

def load_raw_inventario(base_path=None):
    """Loads the raw inventory catalogue from datasets/crudo."""
    if base_path is None:
        base_path = get_datasets_path("crudo")
    path = _find_file(base_path, ["inventario.csv", "Inventario.csv"])
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig', skiprows=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for col in ['Costo_Unitario', 'Precio_Venta']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)
    for col in ['Stock_Minimo', 'Stock_Actual']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df = df.dropna(subset=['ID'])
    return df

def clean_ventas_df(df):
    """Saneamiento profundo de ventas: normalización canónica de fecha (sin hora por digitación en lote) y pago."""
    df = df.copy()
    df = df.dropna(subset=['ID', 'Total'])
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
    df['Metodo_Pago'] = df['Metodo_Pago'].astype(str).str.strip().str.upper()
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0.0)
    return df

def clean_detalle_df(df):
    """Saneamiento profundo de detalle: eliminación de columnas Unnamed, normalización de fechas y numéricos."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(subset=['ID_Venta'])
    df['ID_Venta'] = df['ID_Venta'].astype(str).str.strip()
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed', errors='coerce').dt.strftime('%Y-%m-%d')
    df['Descripcion'] = df['Descripcion'].fillna('SIN DESCRIPCION').astype(str).str.strip()
    for col in ['Cantidad', 'Precio_Unitario', 'Subtotal']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)
            df[col] = df[col].clip(lower=0)
    return df

def clean_inventario_df(df):
    """Saneamiento profundo de inventario: corrección de stocks negativos e imputación inteligente de stock mínimo."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(subset=['ID'])
    for col in ['Costo_Unitario', 'Precio_Venta']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)
    for col in ['Stock_Minimo', 'Stock_Actual']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Imputación inteligente de Stock_Minimo por departamento
    df['Stock_Minimo'] = df.apply(
        lambda row: 5 if pd.isna(row['Stock_Minimo']) and str(row['Departamento']).upper() == 'UTILES'
        else (2 if pd.isna(row['Stock_Minimo']) else row['Stock_Minimo']),
        axis=1
    ).astype(int)
    
    # Corrección de stocks negativos (venta sin entrada registrada en kardex)
    df['Stock_Actual_Saneado'] = df['Stock_Actual'].fillna(0).clip(lower=0).astype(int)
    df['Alerta_Kardex_Negativo'] = df['Stock_Actual'] < 0
    df['Stock_Actual'] = df['Stock_Actual_Saneado']
    return df

def export_cleaned_datasets():
    """Exports structurally and logically cleaned datasets to datasets/limpio/."""
    limpio_dir = os.path.join(get_datasets_path(), "limpio")
    os.makedirs(limpio_dir, exist_ok=True)
    
    df_v = clean_ventas_df(load_raw_ventas())
    df_d = clean_detalle_df(load_raw_detalle_ventas())
    df_i = clean_inventario_df(load_raw_inventario())
    
    df_v.to_csv(os.path.join(limpio_dir, "ventas.csv"), index=False, encoding='utf-8')
    df_d.to_csv(os.path.join(limpio_dir, "detalle_ventas.csv"), index=False, encoding='utf-8')
    df_d.to_csv(os.path.join(limpio_dir, "detalle-ventas.csv"), index=False, encoding='utf-8')
    df_i.to_csv(os.path.join(limpio_dir, "inventario.csv"), index=False, encoding='utf-8')
    
    return df_v, df_d, df_i

def load_ventas(limpio=True):
    """Loads cleaned sales dataset from datasets/limpio if available."""
    if limpio:
        limpio_path = os.path.join(get_datasets_path("limpio"), "ventas.csv")
        if os.path.exists(limpio_path):
            df = pd.read_csv(limpio_path, encoding='utf-8')
            return df
    return load_raw_ventas()

def load_detalle_ventas(limpio=True):
    """Loads cleaned sales detail dataset from datasets/limpio if available."""
    if limpio:
        limpio_path = os.path.join(get_datasets_path("limpio"), "detalle_ventas.csv")
        if os.path.exists(limpio_path):
            df = pd.read_csv(limpio_path, encoding='utf-8')
            return df
    return load_raw_detalle_ventas()

def load_inventario(limpio=True):
    """Loads cleaned inventory dataset from datasets/limpio if available."""
    if limpio:
        limpio_path = os.path.join(get_datasets_path("limpio"), "inventario.csv")
        if os.path.exists(limpio_path):
            return pd.read_csv(limpio_path, encoding='utf-8')
    return load_raw_inventario()

if __name__ == "__main__":
    print("Testing data loader and exporting clean datasets to datasets/limpio/...")
    try:
        df_v, df_d, df_i = export_cleaned_datasets()
        print(f"Cleaned ventas.csv exported successfully. Shape: {df_v.shape}")
        print(f"Cleaned detalle_ventas.csv exported successfully. Shape: {df_d.shape}")
        print(f"Cleaned inventario.csv exported successfully. Shape: {df_i.shape}")
    except Exception as e:
        print(f"Error during export: {e}")

