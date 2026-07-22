"""
Panel 1 – Análisis Exploratorio de Datos (EDA) & Auditoría de Calidad
Auditoría, limpieza de datos, resolución de fenómenos de Kardex e Ingeniería de Características
fiel a los cuadernos 1A y 1B de SmartBazar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


# ═══════════════════════════════════════════════════════════════
#  ESTILOS CSS & GLASSMORPHISM INYECTADOS
# ═══════════════════════════════════════════════════════════════
def _inject_custom_css():
    st.markdown("""
    <style>
    /* ── Flip Cards para Datasets ── */
    .flip-card-container {
        perspective: 1000px;
        margin-bottom: 0.8rem;
    }
    .flip-card {
        background-color: transparent;
        width: 100%;
        height: 220px;
        perspective: 1000px;
    }
    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        transform-style: preserve-3d;
    }
    .flip-card:hover .flip-card-inner {
        transform: rotateY(180deg);
    }
    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        border-radius: 16px;
        padding: 1.4rem 1rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .flip-card-front {
        color: #000000;
    }
    .flip-card-back {
        background: #000000;
        color: #ffffff;
        transform: rotateY(180deg);
        padding: 1.2rem;
    }
    .flip-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 0.5rem;
        color: #000000;
    }
    .flip-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #5D5F5F;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .flip-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        line-height: 1.45;
        color: #f1f5f9;
        text-align: left;
    }
    .flip-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        letter-spacing: 0.05em;
    }

    /* ── Tarjeta Resumen Columna Derecha ── */
    .summary-shape-card {
        background: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .summary-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        color: #000000;
        margin-bottom: 0.4rem;
    }
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(93, 95, 95, 0.15);
    }
    .summary-item:last-child {
        border-bottom: none;
    }
    .summary-ds-name {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.92rem;
        color: #000000;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .summary-ds-meta {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        color: #5D5F5F;
        background: rgba(0,0,0,0.06);
        padding: 4px 10px;
        border-radius: 8px;
    }

    /* ── Tarjetas KPI Calidad de Datos ── */
    .kpi-quality-card {
        background: #ffffff;
        border: 1px solid #ffffff;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.05);
        text-align: center;
    }

    /* ── Contenedores nativos (border=True) en Panel EDA (Fondo Blanco Puro sin contornos grises) ── */
    section.main [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"],
    section.main [data-testid="stBorderWrapper"],
    [data-testid="stMain"] [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stMain"] [data-testid="stBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #ffffff !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.05) !important;
        padding: 1.3rem !important;
        overflow: hidden !important;
        transition: all 0.3s ease !important;
    }

    section.main [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    section.main [data-testid="stBorderWrapper"]:hover {
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.08) !important;
        border-color: #ffffff !important;
    }
    .kpi-quality-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: #5D5F5F;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: block;
        margin-bottom: 0.3rem;
    }
    .kpi-quality-val {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #000000;
        display: block;
    }

    /* ── Estilos para Radio Buttons en la vista principal (Horizontal & Soft Glass) ── */
    [data-testid="stMainBlockContainer"] div[data-testid="stRadio"] div[role="radiogroup"],
    .main div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.8rem !important;
        width: 100% !important;
    }
    
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label {
        display: inline-flex !important;
        align-items: center !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 999px !important;
        padding: 0.45rem 1.1rem !important;
        color: #334155 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
    }

    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #e2e8f0 !important;
        border-color: #64748b !important;
        color: #0f172a !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06) !important;
    }

    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p,
    .main div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPERS VISUALES & CARGA DE DATOS (`datasets/crudo/`)
# ═══════════════════════════════════════════════════════════════
def _apply_chart_style(fig, ax, *, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Aplica el estilo claro y minimalista en tarjetas de fondo blanco puro sin contorno gris."""
    fig.patch.set_facecolor("#ffffff")
    if isinstance(ax, np.ndarray):
        for sub_ax in ax.flat:
            _style_single_ax(sub_ax)
    elif isinstance(ax, (list, tuple)):
        for sub_ax in ax:
            _style_single_ax(sub_ax)
    else:
        _style_single_ax(ax, title, xlabel, ylabel)
    fig.tight_layout()


def _style_single_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#1c1b1b")
    ax.xaxis.label.set_color("#1c1b1b")
    ax.yaxis.label.set_color("#1c1b1b")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.grid(True, alpha=0.25, color="#e2e8f0")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#000000")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _safe_render(fig):
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


@st.cache_data
def _load_raw_datasets():
    """Carga los datasets originales en crudo desde datasets/crudo para inspección fiel al cuaderno."""
    crudo_dir = os.path.join("datasets", "crudo")
    
    if not os.path.exists(crudo_dir):
        crudo_dir = os.path.join("datasets", "limpio")
        
    df_ventas = pd.read_csv(os.path.join(crudo_dir, "ventas.csv"), sep=";", encoding="utf-8-sig", encoding_errors="ignore")
    if df_ventas.shape[1] == 1 and "," in df_ventas.columns[0]:
        df_ventas = pd.read_csv(os.path.join(crudo_dir, "ventas.csv"), sep=",", encoding="utf-8-sig")

    df_detalle = pd.read_csv(os.path.join(crudo_dir, "detalle_ventas.csv"), sep=";", encoding="utf-8-sig", skiprows=1 if "crudo" in crudo_dir else 0, encoding_errors="ignore")
    if df_detalle.shape[1] == 1:
        df_detalle = pd.read_csv(os.path.join(crudo_dir, "detalle_ventas.csv"), sep=",", encoding="utf-8-sig", header=1 if "crudo" in crudo_dir else 0)

    if os.path.exists(os.path.join(crudo_dir, "inventario.csv")):
        try:
            if "crudo" in crudo_dir:
                df_inv = pd.read_csv(os.path.join(crudo_dir, "inventario.csv"), sep=",", encoding="utf-8-sig", header=1, encoding_errors="ignore")
                if df_inv.shape[1] == 1:
                    df_inv = pd.read_csv(os.path.join(crudo_dir, "inventario.csv"), sep=";", encoding="utf-8-sig", header=1)
            else:
                df_inv = pd.read_csv(os.path.join(crudo_dir, "inventario.csv"), encoding="utf-8")
        except Exception:
            df_inv = pd.read_csv(os.path.join(crudo_dir, "inventario.csv"), encoding="utf-8")
    else:
        df_inv = pd.DataFrame()

    df_inv = df_inv.loc[:, ~df_inv.columns.str.contains('^Unnamed')].copy()
    return df_ventas, df_detalle, df_inv


@st.cache_data
def _load_clean_datasets():
    """Carga los datasets limpios y transformados (salida de Panel 1A) para EDA univariado/bivariado."""
    limpio_dir = os.path.join("datasets", "limpio")
    df_ventas = pd.read_csv(os.path.join(limpio_dir, "ventas.csv"), encoding="utf-8")
    df_detalle = pd.read_csv(os.path.join(limpio_dir, "detalle_ventas.csv"), encoding="utf-8")
    df_inv = pd.read_csv(os.path.join(limpio_dir, "inventario.csv"), encoding="utf-8")
    return df_ventas, df_detalle, df_inv


# ═══════════════════════════════════════════════════════════════
#  COMPONENTES MODALES (@st.dialog) PARA VISTA PREVIA
# ═══════════════════════════════════════════════════════════════
@st.dialog("🔍 Vista Previa de Dataset (Top 5 Registros)")
def show_dataset_modal(nombre_dataset: str, df: pd.DataFrame, descripcion: str):
    st.markdown(f"### `{nombre_dataset}`")
    st.markdown(f"**Descripción funcional:** {descripcion}")
    st.divider()
    st.markdown(f"**Estructura dimensional:** `{df.shape[0]} filas × {df.shape[1]} columnas`")
    st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    st.caption("Nota: Se presentan los 5 primeros registros extraídos directamente del archivo crudo del proyecto.")


# ═══════════════════════════════════════════════════════════════
#  TAB 1: RESUMEN DE AUDITORÍA Y LIMPIEZA
# ═══════════════════════════════════════════════════════════════
def _render_tab_1(df_ventas_raw, df_detalle_raw, df_inv_raw):
    # ── 1. Contenedor principal 60/40 (Datasets y Resumen) ──
    col_izq, col_der = st.columns([6, 4], gap="large")

    with col_izq:
        st.markdown("#### 📁 Datasets Utilizados (Interacción Flip & Modal)")
        c_v, c_d, c_i = st.columns(3)

        desc_ventas = "Registro transaccional maestro de cabecera de tickets con ID de venta, fecha/hora, método de pago (Efectivo/Yape) y monto total capturado en caja mostrador."
        desc_detalle = "Desglose a nivel de SKU por cada ticket emitido, especificando descripción, cantidad del ítem, precio unitario cobrado y subtotal por línea."
        desc_inv = "Catálogo maestro y Kardex del almacén, conteniendo stock actual, stock mínimo (ROP), costos unitarios, precios de venta y asignación por departamento."

        with c_v:
            st.markdown("""<div class="flip-card-container">
<div class="flip-card">
    <div class="flip-card-inner">
        <div class="flip-card-front">
            <span style="font-size:2.4rem;">🧾</span>
            <div class="flip-title">ventas.csv</div>
            <div class="flip-subtitle">Cabecera de Comprobantes</div>
        </div>
        <div class="flip-card-back">
            <span class="flip-badge">Maestro de Ventas</span>
            <div class="flip-desc">Registro transaccional maestro con ID de venta, fecha/hora, método de pago y monto total.</div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)
            if st.button("👁️ Ver Tabla (`ventas`)", key="btn_modal_ventas", use_container_width=True):
                show_dataset_modal("ventas.csv", df_ventas_raw, desc_ventas)

        with c_d:
            st.markdown("""<div class="flip-card-container">
<div class="flip-card">
    <div class="flip-card-inner">
        <div class="flip-card-front">
            <span style="font-size:2.4rem;">📦</span>
            <div class="flip-title">detalle_ventas.csv</div>
            <div class="flip-subtitle">Desglose por SKU</div>
        </div>
        <div class="flip-card-back">
            <span class="flip-badge">Líneas de Detalle</span>
            <div class="flip-desc">Desglose a nivel SKU por ticket, especificando descripción, cantidad, precio unitario y subtotal.</div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)
            if st.button("👁️ Ver Tabla (`detalle`)", key="btn_modal_detalle", use_container_width=True):
                show_dataset_modal("detalle_ventas.csv", df_detalle_raw, desc_detalle)

        with c_i:
            st.markdown("""<div class="flip-card-container">
<div class="flip-card">
    <div class="flip-card-inner">
        <div class="flip-card-front">
            <span style="font-size:2.4rem;">🗄️</span>
            <div class="flip-title">inventario.csv</div>
            <div class="flip-subtitle">Catálogo y Kardex</div>
        </div>
        <div class="flip-card-back">
            <span class="flip-badge">Almacén & SKU</span>
            <div class="flip-desc">Catálogo maestro y Kardex de almacén con stocks actuales, stock mínimo, costos y precios.</div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)
            if st.button("👁️ Ver Tabla (`inventario`)", key="btn_modal_inv", use_container_width=True):
                show_dataset_modal("inventario.csv", df_inv_raw, desc_inv)

    with col_der:
        st.markdown("#### 📊 Resumen Dimensional")
        st.markdown(f"""<div class="summary-shape-card">
<div>
    <div class="summary-title">Estructura del Repositorio</div>
    <p style="font-size:0.84rem; color:#5D5F5F; margin-bottom:1rem;">
        Consolidado de dimensionalidad de las fuentes de datos originales capturadas en el sistema informático.
    </p>
    <div class="summary-item">
        <span class="summary-ds-name">🧾 ventas.csv</span>
        <span class="summary-ds-meta">{df_ventas_raw.shape[0]} filas × {df_ventas_raw.shape[1]} cols</span>
    </div>
    <div class="summary-item">
        <span class="summary-ds-name">📦 detalle_ventas.csv</span>
        <span class="summary-ds-meta">{df_detalle_raw.shape[0]} filas × {df_detalle_raw.shape[1]} cols</span>
    </div>
    <div class="summary-item">
        <span class="summary-ds-name">🗄️ inventario.csv</span>
        <span class="summary-ds-meta">{df_inv_raw.shape[0]} filas × {df_inv_raw.shape[1]} cols</span>
    </div>
</div>
<div style="margin-top:1.2rem; padding-top:0.8rem; border-top:1px dashed #cfc4c5; font-size:0.75rem; color:#777777;">
    ✅ Codificación original: UTF-8 / ISO-8859-1 con separador de punto y coma (<code>;</code>).
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── 2. Sección de Resumen de Calidad de Datos (Dinámico por Dataset) ──
    st.markdown("### Resumen de Calidad de Datos")
    
    ds_sel = st.radio(
        "🔍 Seleccionar Dataset para Evaluación de Calidad:",
        ["🧾 ventas.csv", "📦 detalle_ventas.csv", "🗄️ inventario.csv"],
        horizontal=True,
        key="ds_quality_selector"
    )
    
    if "detalle" in ds_sel:
        target_df = df_detalle_raw
        ds_label = "Detalle Ventas"
    elif "ventas" in ds_sel:
        target_df = df_ventas_raw
        ds_label = "Ventas"
    else:
        target_df = df_inv_raw
        ds_label = "Inventario"

    clean_cols = [c for c in target_df.columns if not str(c).startswith("Unnamed")]
    target_clean = target_df[clean_cols] if clean_cols else target_df

    total_rec = len(target_clean)
    nulos_rec = int(target_clean.isnull().sum().sum())
    validos_rec = len(target_clean.dropna())
    cobertura_pct = (validos_rec / total_rec * 100.0) if total_rec > 0 else 100.0

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div class="kpi-quality-card">
        <span class="kpi-quality-title">Total Registros ({ds_label})</span>
        <span class="kpi-quality-val">{total_rec}</span>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="kpi-quality-card">
        <span class="kpi-quality-title">Registros Válidos</span>
        <span class="kpi-quality-val">{validos_rec}</span>
    </div>
    """, unsafe_allow_html=True)
    
    c3.markdown(f"""
    <div class="kpi-quality-card">
        <span class="kpi-quality-title">Nulos en Origen</span>
        <span class="kpi-quality-val">{nulos_rec}</span>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="kpi-quality-card">
        <span class="kpi-quality-title">Cobertura</span>
        <span class="kpi-quality-val">{cobertura_pct:.1f} %</span>
    </div>
    """, unsafe_allow_html=True)

    if ds_label == "Inventario":
        st.markdown(
            "<div style='text-align: center; color: #ef4444; font-size: 0.85rem; margin-top: 1rem; font-weight: 500;'>"
            "⚠️ <b>Nota:</b> El alto número de nulos se debe al campo <b>Stock_Minimo</b> "
            "en los productos de fotocopiadora (Fenómeno 3)."
            "</div>", unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── 3. Sección de Auditoría por Campo ──
    st.markdown("### Auditoría por Campo")
    st.caption("Consolidación forense de calidad de datos, tipología e imputaciones aplicadas en las tablas transaccionales.")
    
    audit_consolidated = pd.DataFrame({
        "Tabla Fuente": [
            "ventas.csv", "ventas.csv", "ventas.csv", "ventas.csv", "ventas.csv",
            "detalle_ventas.csv", "detalle_ventas.csv", "detalle_ventas.csv",
            "inventario.csv", "inventario.csv", "inventario.csv", "inventario.csv"
        ],
        "Campo": [
            "ID", "Fecha", "Total", "Metodo_Pago", "ID_Cliente",
            "ID_Venta", "Cantidad / Subtotal", "Descripcion",
            "ID / Descripcion", "Stock_Minimo", "Stock_Actual", "Costo / Precio_Venta"
        ],
        "Tipo": [
            "str", "datetime -> ISO", "float", "str", "int -> str",
            "str", "float", "str",
            "str", "int", "int", "float"
        ],
        "Nulos": [
            0, 3, 0, 2, 8,
            0, 0, 0,
            0, 314, 0, 0
        ],
        "% Completitud": [
            "100.0%", "99.6%", "100.0%", "99.8%", "99.1%",
            "100.0%", "100.0%", "100.0%",
            "100.0%", "23.98%", "100.0%", "100.0%"
        ],
        "Estado": [
            "✅ OK", "⚠️ Imputado (Mediana mes)", "✅ OK", "⚠️ Imputado (Moda EFECTIVO)", "⚠️ Imputado (CLI-0000)",
            "✅ OK", "✅ OK (Clip > 0)", "⚠️ Imputado (SIN DESCRIPCION)",
            "✅ OK", "⚠️ Imputado por Regla Departamental", "⚠️ Ajustado (Truncado < 0 a 0)", "✅ OK"
        ]
    })
    
    st.dataframe(audit_consolidated, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 2: FENÓMENO 1 – AUDITORÍA TEMPORAL Y SESGO HORARIO
# ═══════════════════════════════════════════════════════════════
def _render_tab_2(df_ventas_raw, df_inv_raw):
    st.markdown("### Fenómeno 1: Auditoría Temporal y Auditoría de Sesgo Horario")
    st.markdown("""
    En proyectos transaccionales reales, los timestamps y horas capturados pueden reflejar la actividad del digitador administrativo y no la hora en que el consumidor compró en vitrina.
    Sometemos a examen forense las columnas temporales de `ventas.csv` e `inventario.csv`.
    """)

    # Procesamiento analítico fiel al Cuaderno 1A (CELL 5)
    df_ventas_temp = df_ventas_raw.dropna(subset=['ID', 'Fecha']).copy() if 'Fecha' in df_ventas_raw.columns else pd.DataFrame()
    if not df_ventas_temp.empty:
        df_ventas_temp['dt'] = pd.to_datetime(df_ventas_temp['Fecha'], format='mixed', errors='coerce')
        df_ventas_temp['Dia_Semana'] = df_ventas_temp['dt'].dt.day_name()
        df_ventas_temp['Tiene_Hora'] = df_ventas_temp['Fecha'].astype(str).str.contains(':')
        df_con_hora = df_ventas_temp[df_ventas_temp['Tiene_Hora']].copy()
        if not df_con_hora.empty:
            df_con_hora['Hora'] = df_con_hora['dt'].dt.hour
    else:
        df_con_hora = pd.DataFrame()

    df_inv_temp = df_inv_raw.copy()
    if 'ID' in df_inv_temp.columns:
        df_inv_temp = df_inv_temp.dropna(subset=['ID']).copy()
        
    col_fecha_inv = None
    for candidate in ['Fecha Ingreso', 'Fecha_Ingreso', 'Fecha', 'FechaIngreso']:
        if candidate in df_inv_temp.columns:
            col_fecha_inv = candidate
            break

    if col_fecha_inv:
        df_inv_temp['dt_ingreso'] = pd.to_datetime(df_inv_temp[col_fecha_inv], format='mixed', errors='coerce')
        df_inv_temp['Dia_Ingreso'] = df_inv_temp['dt_ingreso'].dt.day_name()
    
    col_f1_plot, col_f1_txt = st.columns([1.1, 0.9], gap="large")

    with col_f1_plot:
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5))
        
        # Gráfico 1: Concentración por Día de la Semana en Ingresos a Almacén
        if col_fecha_inv and 'Dia_Ingreso' in df_inv_temp.columns and df_inv_temp['Dia_Ingreso'].dropna().count() > 0:
            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sns.countplot(data=df_inv_temp, x='Dia_Ingreso', order=dias_orden, ax=axes[0], color='#000000', edgecolor='#5D5F5F')
            axes[0].set_title('Ingreso de Kardex por Día de la Semana\n(Concentración masiva en Fines de Semana y Lunes)', fontweight='bold')
            axes[0].set_xlabel('Día de la Semana')
            axes[0].set_ylabel('Número de Registros')
            axes[0].tick_params(axis='x', rotation=25)
        else:
            axes[0].text(0.5, 0.5, 'Sin columna de Fecha Ingreso', ha='center', va='center', color='#5D5F5F')
        
        # Gráfico 2: Distribución Horaria de Ventas Digitadas
        if not df_con_hora.empty and 'Hora' in df_con_hora.columns:
            sns.histplot(data=df_con_hora, x='Hora', bins=24, kde=True, ax=axes[1], color='#5D5F5F', edgecolor='#000000')
            axes[1].set_title('Distribución Horaria con Timestamp\n(Concentración en ventanas administrativas de digitación por lotes)', fontweight='bold')
            axes[1].set_xlabel('Hora del Día (0 - 23 hs)')
            axes[1].set_ylabel('Frecuencia de Comprobantes')
            
        _apply_chart_style(fig, axes)
        _safe_render(fig)

    with col_f1_txt:
        st.markdown("#### 📑 Evidencia y Justificación de Negocio")
        
        # Diseño en tarjetas de flujo con Flip Animation (Paso 1 -> Paso 2 -> Paso 3)
        st.markdown("""<div style="display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.5rem;">
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #ef4444; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #ef4444; text-transform: uppercase; letter-spacing: 0.05em;">PASO 1 ── SESGO OPERATIVO (DIGITING BIAS)</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Concentración Anómala en Fines de Semana</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver la evidencia</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #ef4444; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(239, 68, 68, 0.25); color: #fca5a5; margin-bottom: 0.4rem;">Evidencia Forense</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    El <strong>Domingo (201 registros - 47.6%)</strong> y el <strong>Lunes (121 registros - 28.7%)</strong> acumulan más del <strong>76% de la actividad</strong>. De martes a viernes es &lt; 3%.
                </p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; font-size: 1rem; color: #94a3b8; margin: -0.2rem 0;">⬇️</div>
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #f59e0b; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #f59e0b; text-transform: uppercase; letter-spacing: 0.05em;">PASO 2 ── CONCLUSIÓN OPERATIVA DE NEGOCIO</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Carga de Mostrador vs. Digitación Diferida por Lotes</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver la justificación</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #f59e0b; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(245, 158, 11, 0.25); color: #fde68a; margin-bottom: 0.4rem;">Análisis de Causa</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    SmartBazar atiende al público escolar de Lunes a Viernes. El personal acumula notas de compra y facturas para digitarlas al sistema en lote durante el fin de semana o lunes.
                </p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; font-size: 1rem; color: #94a3b8; margin: -0.2rem 0;">⬇️</div>
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #10b981; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #10b981; text-transform: uppercase; letter-spacing: 0.05em;">PASO 3 ── DECISIÓN Y ACCIÓN DE SANEAMIENTO</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Normalización ISO 8601 sin Componente Horario</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver la acción</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #10b981; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(16, 185, 129, 0.25); color: #a7f3d0; margin-bottom: 0.4rem;">Regla de Limpieza</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    El timestamp no representa la compra real. <strong>Acción:</strong> Normalizar <code>Fecha</code> al estándar <strong>ISO 8601 (YYYY-MM-DD)</strong> eliminando la hora engañosa.
                </p>
            </div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

        if col_fecha_inv and 'Dia_Ingreso' in df_inv_temp.columns and df_inv_temp['Dia_Ingreso'].dropna().count() > 0:
            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            conteo_dias = df_inv_temp['Dia_Ingreso'].value_counts()
            proporc = (conteo_dias / len(df_inv_temp) * 100).round(2)
            tabla_dias = pd.DataFrame({'Registros': conteo_dias, 'Porcentaje (%)': proporc})
            st.markdown("**Tabla Empírica: Frecuencia de Registro en Almacén**")
            st.dataframe(tabla_dias, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 3: FENÓMENO 2 – DESFASE DE KARDEX Y STOCKS NEGATIVOS
# ═══════════════════════════════════════════════════════════════
def _render_tab_3(df_inv_raw):
    st.markdown("### Fenómeno 2: Desfase Contable en Kardex y Stocks Negativos")
    st.markdown("""
    En teoría de inventarios (*Supply Chain Management*), la existencia de mercadería es estrictamente no negativa ($Stock \\ge 0$). La aparición de valores menores a cero constituye una patología contable que requiere diagnóstico.
    """)

    df_inv_clean = df_inv_raw.loc[:, ~df_inv_raw.columns.str.contains('^Unnamed')].copy()
    df_inv_clean = df_inv_clean.dropna(subset=['ID']).copy() if 'ID' in df_inv_clean.columns else df_inv_clean
    if 'Stock_Actual' in df_inv_clean.columns:
        df_inv_clean['Stock_Actual_num'] = pd.to_numeric(df_inv_clean['Stock_Actual'], errors='coerce').fillna(0)
        df_negativos = df_inv_clean[df_inv_clean['Stock_Actual_num'] < 0].sort_values(by='Stock_Actual_num', ascending=True)
        top_10_neg = df_negativos.head(10)
    else:
        df_negativos = pd.DataFrame()
        top_10_neg = pd.DataFrame()

    col_f2_plot, col_f2_txt = st.columns([1.1, 0.9], gap="large")

    with col_f2_plot:
        if not top_10_neg.empty:
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            sns.barplot(data=top_10_neg, x='Stock_Actual_num', y='Descripcion', color='#000000', edgecolor='#5D5F5F', ax=ax)
            _apply_chart_style(
                fig, ax,
                title='Top 10 Artículos con Stock Negativo en Base de Datos\n(Desfase Asincrónico de Kardex)',
                xlabel='Unidades Negativas Registradas en Sistema',
                ylabel='Descripción del SKU'
            )
            ax.axvline(0, color='#5D5F5F', linestyle='--')
            _safe_render(fig)
        else:
            st.info("No se detectaron stocks negativos o la columna no está disponible.")

    with col_f2_txt:
        st.markdown(f"#### 🔎 Diagnóstico de Causa Raíz (`{len(df_negativos)} SKU detectados`)")
        
        st.markdown("""<div style="display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.5rem;">
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #ef4444; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #ef4444; text-transform: uppercase; letter-spacing: 0.05em;">PUNTO 1 ── ARTÍCULOS AFECTADOS</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Déficit Concentrado en Útiles de Alta Rotación</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver productos</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #ef4444; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(239, 68, 68, 0.25); color: #fca5a5; margin-bottom: 0.4rem;">Mayor Impacto</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    Los ítems con mayor déficit pertenecen al departamento <strong>UTILES</strong>: <strong>HOJA DE COLORES (-184 un.)</strong> y <strong>HOJA BOND A4 ATLAS (-173 un.)</strong>.
                </p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; font-size: 1rem; color: #94a3b8; margin: -0.2rem 0;">⬇️</div>
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #f59e0b; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #f59e0b; text-transform: uppercase; letter-spacing: 0.05em;">PUNTO 2 ── CAUSA RAÍZ (TIMING DISCREPANCY)</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Desfase Mostrador vs. Procesamiento Kardex</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver la explicación</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #f59e0b; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(245, 158, 11, 0.25); color: #fde68a; margin-bottom: 0.4rem;">Explicación Operativa</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    Ventas registran salidas de stock antes de procesar el ingreso en Kardex. Al descontar sobre saldo 0, el balance se proyecta en negativo.
                </p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; font-size: 1rem; color: #94a3b8; margin: -0.2rem 0;">⬇️</div>
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #10b981; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #10b981; text-transform: uppercase; letter-spacing: 0.05em;">PUNTO 3 ── ACCIÓN DE SANEAMIENTO</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Auditoría y Truncamiento Físico a Cero</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver la acción</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #10b981; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(16, 185, 129, 0.25); color: #a7f3d0; margin-bottom: 0.4rem;">Regla Aplicada</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    Flag <code>Alerta_Kardex_Negativo = True</code> y truncamiento físico a <strong>0</strong> (<code>.clip(lower=0)</code>), reflejando agotamiento real del lote.
                </p>
            </div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

        if not top_10_neg.empty:
            st.markdown("**Top 5 Artículos con Mayor Desfase de Kardex**")
            st.dataframe(top_10_neg[['ID', 'Descripcion', 'Departamento', 'Stock_Actual']].head(5), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 4: FENÓMENO 3 – IMPUTACIÓN EN STOCK_MINIMO
# ═══════════════════════════════════════════════════════════════
def _render_tab_4(df_inv_raw):
    st.markdown("### Fenómeno 3: Auditoría y Reglas de Negocio para Imputación en `Stock_Minimo`")
    st.markdown("""
    El `Stock_Minimo` representa el Punto de Reorden (*Reorder Point - ROP*), crítico para generar alertas automáticas. Auditemos el nivel de omisión y analicemos estadísticamente las reglas de negocio descubiertas por departamento.
    """)

    df_inv_clean = df_inv_raw.loc[:, ~df_inv_raw.columns.str.contains('^Unnamed')].copy()
    if 'Stock_Minimo' in df_inv_clean.columns and 'Departamento' in df_inv_clean.columns:
        df_inv_clean['Stock_Minimo_num'] = pd.to_numeric(df_inv_clean['Stock_Minimo'], errors='coerce')
        total_items = len(df_inv_clean)
        nulos_stk = df_inv_clean['Stock_Minimo_num'].isna().sum()
        tasa_nulos = (nulos_stk / total_items) * 100 if total_items > 0 else 0
    else:
        total_items = len(df_inv_clean)
        nulos_stk = 0
        tasa_nulos = 0

    col_f3_plot, col_f3_txt = st.columns([1.1, 0.9], gap="large")

    with col_f3_plot:
        if 'Stock_Minimo_num' in df_inv_clean.columns and 'Departamento' in df_inv_clean.columns:
            df_presentes = df_inv_clean.dropna(subset=['Stock_Minimo_num'])
            if not df_presentes.empty:
                fig, ax = plt.subplots(figsize=(9.5, 5))
                sns.countplot(data=df_presentes, x='Stock_Minimo_num', hue='Departamento', palette=['#000000', '#5D5F5F', '#94a3b8'], ax=ax)
                _apply_chart_style(
                    fig, ax,
                    title='Distribución Empírica de Stock_Minimo Registrado por Departamento\n(Evidencia de moda y mediana en 5 para Útiles)',
                    xlabel='Valor de Stock Mínimo Capturado',
                    ylabel='Cantidad de SKU'
                )
                _safe_render(fig)
        else:
            st.info("No se dispone de datos de Stock Mínimo para graficar.")

    with col_f3_txt:
        st.markdown(f"#### 📐 Regla de Negocio (`{tasa_nulos:.1f}%` de omisión en POS)")
        
        st.markdown("""<div style="display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.5rem;">
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #3b82f6; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.05em;">REGLA 1 ── DEPARTAMENTO UTILES</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Punto de Reorden Empírico (ROP = 5)</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver el criterio</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #3b82f6; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(59, 130, 246, 0.25); color: #93c5fd; margin-bottom: 0.4rem;">Papelería & Librería</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    El <strong>83% de útiles</strong> con dato capturado tiene exactamente 5 unidades. Moda y mediana = <strong>5</strong>. Umbral empírico para evitar quiebres en vitrina escolar.
                </p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; font-size: 1rem; color: #94a3b8; margin: -0.2rem 0;">⬇️</div>
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #f59e0b; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #f59e0b; text-transform: uppercase; letter-spacing: 0.05em;">REGLA 2 ── DEPARTAMENTO FOTOCOPIADORA</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Resguardo Técnico por Servicios (ROP = 2)</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver la justificación</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #f59e0b; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(245, 158, 11, 0.25); color: #fde68a; margin-bottom: 0.4rem;">Servicios Reprográficos</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    El <strong>100% de ítems de fotocopiadora es Nulo</strong>. Al ser servicios por clic de máquina, se justifica un resguardo técnico de <strong>2</strong> (reserva de insumos/tóner).
                </p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; font-size: 1rem; color: #94a3b8; margin: -0.2rem 0;">⬇️</div>
<div class="flip-card-container" style="height: 140px; margin-bottom: 0;">
    <div class="flip-card" style="height: 140px;">
        <div class="flip-card-inner">
            <div class="flip-card-front" style="border-left: 5px solid #10b981; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem;">
                <span style="font-family: 'Inter', sans-serif; font-size: 0.68rem; font-weight: 800; color: #10b981; text-transform: uppercase; letter-spacing: 0.05em;">REGLA 3 ── FÓRMULA DE IMPUTACIÓN</span>
                <div style="font-family: 'Inter', sans-serif; font-size: 1.02rem; font-weight: 800; color: #000000; margin-top: 0.4rem;">Imputación Condicional Algorítmica</div>
                <div style="font-family: 'Inter', sans-serif; font-size: 0.72rem; color: #94a3b8; margin-top: 0.6rem; display: flex; align-items: center; gap: 4px;">
                    <span>🔄 Pasa el cursor para ver la fórmula</span>
                </div>
            </div>
            <div class="flip-card-back" style="border-left: 5px solid #10b981; text-align: left; align-items: flex-start; justify-content: center; padding: 1.1rem 1.3rem; background: #0f172a;">
                <span class="flip-badge" style="background: rgba(16, 185, 129, 0.25); color: #a7f3d0; margin-bottom: 0.4rem;">Fórmula Aplicada</span>
                <p class="flip-desc" style="font-size: 0.8rem; color: #e2e8f0; margin: 0; line-height: 1.4;">
                    Fórmula condicional ejecutada: <code>np.where(Departamento == 'UTILES', 5, 2)</code>. Imputación completa del 100% de nulos.
                </p>
            </div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

        if 'Stock_Minimo_num' in df_inv_clean.columns and 'Departamento' in df_inv_clean.columns:
            tabla_stk = pd.crosstab(df_inv_clean['Departamento'], df_inv_clean['Stock_Minimo_num'].fillna(-999), margins=True)
            tabla_stk = tabla_stk.rename(columns={-999.0: 'NULO / AUSENTE'})
            st.markdown("**Tabla Empírica Pre-Imputación (Observed vs Nulos)**")
            st.dataframe(tabla_stk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 5: INGENIERÍA DE CARACTERÍSTICAS (PANEL 1B)
# ═══════════════════════════════════════════════════════════════
def _render_tab_5(df_ventas_cl, df_detalle_cl, df_inv_cl):
    st.markdown("### Ingeniería de Características & Análisis Univariado/Bivariado (Panel 1B)")
    st.caption("Caracterización estadística formal univariada, pivotaje de canastas por ticket (`ID_Venta`) e investigación bivariada multivariada.")

    # ── 1. Resumen Estadístico Univariado ──
    st.markdown("#### 1. Resumen Estadístico Univariado de Variables Continuas")
    
    def _get_univariate_stats(df, col_name):
        if col_name not in df.columns:
            return None
        s = pd.to_numeric(df[col_name], errors='coerce').dropna()
        if s.empty:
            return None
        media = s.mean()
        mediana = s.median()
        moda = s.mode()[0] if not s.mode().empty else np.nan
        stdev = s.std()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        cv = (stdev / media * 100) if media != 0 else np.nan
        skew = stats.skew(s)
        kurt = stats.kurtosis(s)
        return {
            'Variable': col_name,
            'Media': media,
            'Mediana': mediana,
            'Moda': moda,
            'Desv. Est.': stdev,
            'IQR': iqr,
            'CV (%)': cv,
            'Asimetría (Skew)': skew,
            'Curtosis (Kurt)': kurt
        }

    stats_list = []
    for var in ['Total']:
        res = _get_univariate_stats(df_ventas_cl, var)
        if res: stats_list.append(res)
    for var in ['Cantidad', 'Precio_Unitario', 'Subtotal']:
        res = _get_univariate_stats(df_detalle_cl, var)
        if res: stats_list.append(res)
    for var in ['Precio_Venta', 'Stock_Actual']:
        res = _get_univariate_stats(df_inv_cl, var)
        if res: stats_list.append(res)

    if stats_list:
        st.dataframe(pd.DataFrame(stats_list), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── 2. Gráficos Duales Univariados (Boxplot + KDE) ──
    st.markdown("#### 2. Inspección Visual Dual Univariada (`Boxplot + KDE`)")
    col_u1, col_u2, col_u3 = st.columns(3, gap="medium")

    with col_u1:
        st.markdown("##### 📌 Total de Venta por Ticket\n`ventas.csv` (Variable: `Total`)")
        if 'Total' in df_ventas_cl.columns:
            s_tot = pd.to_numeric(df_ventas_cl['Total'], errors='coerce').dropna()
            fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(6, 4.8), gridspec_kw={"height_ratios": [0.25, 0.75]}, sharex=True)
            sns.boxplot(x=s_tot, ax=ax_box, color='#000000', fliersize=3)
            ax_box.axvline(s_tot.mean(), color='#5D5F5F', linestyle='--', label='Media')
            ax_box.axvline(s_tot.median(), color='#000000', linestyle='-', label='Mediana')
            sns.histplot(x=s_tot, ax=ax_hist, kde=True, color='#5D5F5F', bins=35)
            _apply_chart_style(fig, [ax_box, ax_hist], title='Total por Ticket (S/.)', xlabel='Monto (S/.)', ylabel='Frecuencia')
            _safe_render(fig)

    with col_u2:
        st.markdown("##### 📌 Subtotal por Línea en Detalle\n`detalle_ventas.csv` (Variable: `Subtotal`)")
        if 'Subtotal' in df_detalle_cl.columns:
            s_sub = pd.to_numeric(df_detalle_cl['Subtotal'], errors='coerce').dropna()
            fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(6, 4.8), gridspec_kw={"height_ratios": [0.25, 0.75]}, sharex=True)
            sns.boxplot(x=s_sub, ax=ax_box, color='#000000', fliersize=3)
            ax_box.axvline(s_sub.mean(), color='#5D5F5F', linestyle='--', label='Media')
            ax_box.axvline(s_sub.median(), color='#000000', linestyle='-', label='Mediana')
            sns.histplot(x=s_sub, ax=ax_hist, kde=True, color='#5D5F5F', bins=35)
            _apply_chart_style(fig, [ax_box, ax_hist], title='Subtotal por Línea (S/.)', xlabel='Subtotal (S/.)', ylabel='Frecuencia')
            _safe_render(fig)

    with col_u3:
        st.markdown("##### 📌 Precio de Venta en Catálogo\n`inventario.csv` (Variable: `Precio_Venta`)")
        if 'Precio_Venta' in df_inv_cl.columns:
            s_pv = pd.to_numeric(df_inv_cl['Precio_Venta'], errors='coerce').dropna()
            if not s_pv.empty:
                fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(6, 4.8), gridspec_kw={"height_ratios": [0.25, 0.75]}, sharex=True)
                sns.boxplot(x=s_pv, ax=ax_box, color='#000000', fliersize=3)
                ax_box.axvline(s_pv.mean(), color='#5D5F5F', linestyle='--', label='Media')
                ax_box.axvline(s_pv.median(), color='#000000', linestyle='-', label='Mediana')
                sns.histplot(x=s_pv, ax=ax_hist, kde=True, color='#5D5F5F', bins=25)
                _apply_chart_style(fig, [ax_box, ax_hist], title='Precio de Venta (S/.)', xlabel='Precio (S/.)', ylabel='Frecuencia')
                _safe_render(fig)

    st.divider()

    # ── 3. Agregación e Ingeniería de Características por Ticket (`ID_Venta`) ──
    st.markdown("#### 3. Matriz de Características Agregadas a Nivel Ticket (`ID_Venta`)")
    st.caption("Pivotaje del modelo relacional 1:N hacia un vector transaccional único para alimentar los modelos analíticos de agrupamiento.")

    if 'ID_Venta' in df_detalle_cl.columns:
        df_det_feat = df_detalle_cl.copy()
        df_det_feat['Departamento'] = df_det_feat['Departamento'].astype(str).str.strip().str.upper() if 'Departamento' in df_det_feat.columns else 'UTILES'
        for col in ['Cantidad', 'Subtotal']:
            df_det_feat[col] = pd.to_numeric(df_det_feat[col], errors='coerce').fillna(0)

        df_det_feat['Subtotal_Utiles'] = np.where(df_det_feat['Departamento'] == 'UTILES', df_det_feat['Subtotal'], 0.0)
        df_det_feat['Subtotal_Fotocopiadora'] = np.where(df_det_feat['Departamento'] == 'FOTOCOPIADORA', df_det_feat['Subtotal'], 0.0)

        df_ticket_features = df_det_feat.groupby('ID_Venta').agg(
            Cantidad_Items_Total=('Cantidad', 'sum'),
            Variedad_Items_Ticket=('ID_Producto', 'nunique') if 'ID_Producto' in df_det_feat.columns else ('Subtotal', 'count'),
            Subtotal_Total=('Subtotal', 'sum'),
            Gasto_Utiles=('Subtotal_Utiles', 'sum'),
            Gasto_Fotocopiadora=('Subtotal_Fotocopiadora', 'sum')
        ).reset_index()

        df_ticket_features['Precio_Promedio_Item'] = np.where(
            df_ticket_features['Cantidad_Items_Total'] > 0,
            df_ticket_features['Subtotal_Total'] / df_ticket_features['Cantidad_Items_Total'],
            0.0
        ).round(4)

        df_ticket_features['Ratio_Utiles'] = np.where(
            df_ticket_features['Subtotal_Total'] > 0,
            (df_ticket_features['Gasto_Utiles'] / df_ticket_features['Subtotal_Total']).round(4),
            0.0
        )

        conds = [
            df_ticket_features['Gasto_Utiles'] >= df_ticket_features['Gasto_Fotocopiadora'],
            df_ticket_features['Gasto_Fotocopiadora'] > df_ticket_features['Gasto_Utiles']
        ]
        df_ticket_features['Departamento_Dominante'] = np.select(conds, ['UTILES', 'FOTOCOPIADORA'], default='MIXTO')

        st.dataframe(df_ticket_features[['ID_Venta', 'Cantidad_Items_Total', 'Variedad_Items_Ticket', 'Subtotal_Total', 'Precio_Promedio_Item', 'Departamento_Dominante', 'Ratio_Utiles']].head(8), use_container_width=True, hide_index=True)
        st.caption(f"✅ Dimensionalidad de matriz transaccional única: {df_ticket_features.shape[0]} canastas × {df_ticket_features.shape[1]} variables derivadas.")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # ── 4. Análisis Bivariado sobre Características del Ticket ──
        st.markdown("#### 4. Exploración Bivariada: Variedad e Intensidad de Compra")
        col_b1, col_b2 = st.columns(2, gap="large")

        with col_b1:
            st.markdown("1. **Hipótesis Bivariada 1 (`Variedad` vs `Subtotal`):** ¿El incremento en el ticket promedio de un estudiante de SmartBazar obedece a una canasta variada (comprar muchos útiles diferentes de la lista escolar) o a una compra concentrada en volumen intensivo (ej. tirajes masivos de fotocopias o resmas por parte de un solo docente)? Lo verificaremos con un gráfico de dispersión con línea de regresión de tendencia (`scatterplot + regplot`).")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(data=df_ticket_features, x='Variedad_Items_Ticket', y='Subtotal_Total', hue='Departamento_Dominante', palette=['#000000', '#5D5F5F'], alpha=0.8, s=60, ax=ax)
            sns.regplot(data=df_ticket_features, x='Variedad_Items_Ticket', y='Subtotal_Total', scatter=False, ax=ax, color='#000000', line_kws={'linestyle': '--'})
            _apply_chart_style(fig, ax, title='Variedad de SKU vs. Subtotal Pagado', xlabel='Variedad de Ítems ($Nunique$)', ylabel='Subtotal del Ticket (S/.)')
            _safe_render(fig)

        with col_b2:
            st.markdown("2. **Hipótesis Bivariada 2 (`Departamento Dominante` vs `Subtotal`):** ¿Qué categoría genera carritos de mayor valor monetario global y mayor dispersión? Lo verificaremos mediante un diagrama comparativo de caja (boxplot).")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.boxplot(data=df_ticket_features, x='Departamento_Dominante', y='Subtotal_Total', palette=['#000000', '#5D5F5F'], ax=ax, fliersize=4)
            _apply_chart_style(fig, ax, title='Distribución del Gasto por Departamento Dominante', xlabel='Departamento Dominante del Ticket', ylabel='Subtotal del Ticket (S/.)')
            _safe_render(fig)
    else:
        st.info("No se pudo construir la matriz por ticket debido a falta de columnas requeridas.")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT DEL PANEL EDA
# ═══════════════════════════════════════════════════════════════
def show_panel():
    """Punto de entrada oficial de Panel 1A & 1B en el Dashboard Streamlit."""
    _inject_custom_css()
    np.random.seed(42)

    st.header("Análisis Exploratorio de Datos (EDA)")
    st.caption("Auditoría de calidad, limpieza de datos y exploración estadística de las transacciones de SmartBazar.")

    # Carga de datos crudos y limpios fieles a los cuadernos
    df_v_raw, df_d_raw, df_i_raw = _load_raw_datasets()
    df_v_cl, df_d_cl, df_i_cl = _load_clean_datasets()

    # Pestañas exactas en orden estricto (NO REORDENAR)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Resumen de Auditoria y Limpieza",
        "Fenómeno 1: Auditoría Temporal y Auditoría de Sesgo Horario",
        "Fenómeno 2: Desfase Contable en Kardex y Stocks Negativos",
        "Fenómeno 3: Auditoría y Reglas de Negocio para Imputación en Stock_Minimo",
        "Ingenieria de Caracteristicas"
    ])

    with tab1:
        _render_tab_1(df_v_raw, df_d_raw, df_i_raw)

    with tab2:
        _render_tab_2(df_v_raw, df_i_raw)

    with tab3:
        _render_tab_3(df_i_raw)

    with tab4:
        _render_tab_4(df_i_raw)

    with tab5:
        _render_tab_5(df_v_cl, df_d_cl, df_i_cl)
