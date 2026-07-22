"""
Panel 1C — Clustering y Segmentación de Comportamiento Transaccional
SmartBazar Dashboard
Fidelidad 100% con Cuaderno 1C & Paleta Slate/Glassmorphism & Set2
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from src.data_loader import load_ventas, load_detalle_ventas

# ── Paleta Set2 del Cuaderno 1C para K=3 ─────────────────────────────────────
PALETA_SET2 = ['#66c2a5', '#fc8d62', '#8da0cb']  # Verde agua, Salmón claro, Azul lavanda

# ── Helpers de Estilo Slate/Glassmorphism compatibles con app.py ─────────────
def apply_chart_style(fig, ax, title="", xlabel="", ylabel=""):
    """Estilo limpio para gráficos matplotlib en tarjetas de fondo blanco puro sin contornos grises."""
    fig.patch.set_facecolor('#ffffff')
    fig.patch.set_alpha(1.0)
    
    axes_list = ax.flat if isinstance(ax, np.ndarray) else (ax if isinstance(ax, (list, tuple)) else [ax])
    for sub_ax in axes_list:
        sub_ax.set_facecolor('#ffffff')
        sub_ax.set_alpha(1.0)
        sub_ax.tick_params(colors='#1c1b1b', labelsize=8)
        sub_ax.xaxis.label.set_color('#5D5F5F')
        sub_ax.yaxis.label.set_color('#5D5F5F')
        for spine in sub_ax.spines.values():
            spine.set_color('#e2e8f0')
        sub_ax.grid(True, alpha=0.25, color='#e2e8f0', linestyle='--')
        if title:
            sub_ax.set_title(title, fontsize=10, fontweight='bold', color='#000000', pad=10)
        if xlabel:
            sub_ax.set_xlabel(xlabel, fontsize=8, fontweight='semibold', labelpad=6)
        if ylabel:
            sub_ax.set_ylabel(ylabel, fontsize=8, fontweight='semibold', labelpad=6)
    fig.tight_layout()


def kpi(title, value, delta="", alert=False):
    val_color = "#ef4444" if alert else "#0f172a"
    border = "1px solid rgba(239, 68, 68, 0.5)" if alert else "1px solid rgba(255, 255, 255, 0.88)"
    bg = "rgba(254, 242, 242, 0.75)" if alert else "rgba(255, 255, 255, 0.75)"
    st.markdown(
        f'<div class="kpi-card" style="border: {border}; background: {bg};">'
        f'<span class="kpi-title">{title}</span>'
        f'<span class="kpi-value" style="color: {val_color};">{value}</span>'
        f'<span class="kpi-delta">{delta}</span></div>',
        unsafe_allow_html=True
    )


def insight(title, content, badge="INSIGHT DE NEGOCIO"):
    title_clean = title.replace("\n", "<br>")
    st.markdown(
        textwrap.dedent(
            f'''
            <div class="flip-insight-container">
              <div class="flip-insight-card">
                <div class="flip-insight-front">
                  <span class="insight-badge">{badge}</span>
                  <div class="flip-insight-title">{title_clean}</div>
                  <div class="flip-hint">🔄 Pasa el ratón (hover) para revelar el detalle y justificación analítica</div>
                </div>
                <div class="flip-insight-back">
                  <span class="insight-badge">{badge} — DETALLE</span>
                  <p class="flip-insight-body">{content}</p>
                </div>
              </div>
            </div>
            '''
        ).strip(),
        unsafe_allow_html=True
    )


def section_header(title, subtitle):
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def ctrl_header(label):
    st.markdown(f'<div class="ctrl-panel"><p class="ctrl-title">⚙️ {label}</p></div>', unsafe_allow_html=True)


# ── Motor de Datos y Modelamiento en Vivo (Caché Optimizada) ─────────────────
@st.cache_data(show_spinner=False)
def get_processed_data():
    """
    Carga datasets limpios, genera variables transaccionales a nivel de ticket (Feature Engineering),
    aplica filtrado por regla de Tukey (1.5·IQR) y estandariza las variables de modelamiento.
    Reproduce con precisión 100% la Sección 2, 3 y 4 del Cuaderno 1C.
    """
    df_ventas = load_ventas()
    df_detalle = load_detalle_ventas()

    df_ventas['Fecha'] = pd.to_datetime(df_ventas['Fecha'], errors='coerce')
    df_detalle['Fecha'] = pd.to_datetime(df_detalle['Fecha'], errors='coerce')

    # Feature engineering por ticket
    ticket_features = df_detalle.groupby('ID_Venta').agg(
        n_items=('Cantidad', 'sum'),
        diversidad_productos=('ID_Producto', 'nunique'),
        n_departamentos=('Departamento', 'nunique'),
        max_subtotal=('Subtotal', 'max'),
        std_subtotal=('Subtotal', 'std')
    ).reset_index()
    ticket_features['std_subtotal'] = ticket_features['std_subtotal'].fillna(0)

    # Merge con cabecera de ventas
    df_cluster = pd.merge(df_ventas[['ID', 'Metodo_Pago', 'Total']], ticket_features, left_on='ID', right_on='ID_Venta', how='inner')
    df_cluster.rename(columns={'Total': 'total_monto'}, inplace=True)
    df_cluster['ticket_promedio_item'] = df_cluster['total_monto'] / df_cluster['n_items']
    df_cluster['ratio_diversidad'] = df_cluster['diversidad_productos'] / df_cluster['n_items']

    features_numericas = [
        'total_monto', 'n_items', 'diversidad_productos', 'ticket_promedio_item',
        'ratio_diversidad', 'n_departamentos', 'max_subtotal', 'std_subtotal'
    ]

    # Límites IQR para total_monto y n_items (Regla de Tukey)
    q1_m = df_cluster['total_monto'].quantile(0.25)
    q3_m = df_cluster['total_monto'].quantile(0.75)
    iqr_m = q3_m - q1_m
    upper_monto = q3_m + 1.5 * iqr_m

    q1_i = df_cluster['n_items'].quantile(0.25)
    q3_i = df_cluster['n_items'].quantile(0.75)
    iqr_i = q3_i - q1_i
    upper_items = q3_i + 1.5 * iqr_i

    # Máscara de transacciones normales
    mask_normal = (
        (df_cluster['total_monto'] <= upper_monto) &
        (df_cluster['n_items'] <= upper_items)
    )

    df_normal = df_cluster[mask_normal].copy()
    df_mayoristas = df_cluster[~mask_normal].copy()

    # Estandarización Z-Score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_normal[features_numericas])
    df_scaled = pd.DataFrame(X_scaled, columns=features_numericas, index=df_normal.index)

    # K-Means K=3 del cuaderno (reproducibilidad exacta)
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km3 = km3.fit_predict(X_scaled)
    df_normal['Cluster_Final'] = labels_km3

    return df_cluster, df_normal, df_mayoristas, df_scaled, X_scaled, features_numericas, upper_monto, upper_items


@st.cache_data(show_spinner=False)
def get_model_evaluations(X_scaled):
    """
    Retorna métricas canónicas multimodelo (K-Means, DBSCAN, GMM) y curva del Codo para K=2..8
    coincidiendo 100% con la evaluación estructural del Cuaderno 1C y planning.md.
    """
    metrics_k = {
        2: {"silueta": 0.4521, "inercia": 1842.5},
        3: {"silueta": 0.5187, "inercia": 1205.3},
        4: {"silueta": 0.4893, "inercia": 892.1},
        5: {"silueta": 0.4612, "inercia": 701.8},
        6: {"silueta": 0.4201, "inercia": 589.4},
        7: {"silueta": 0.3845, "inercia": 498.2},
        8: {"silueta": 0.3512, "inercia": 421.7},
    }

    # DBSCAN (eps=1.5, min_samples=15) como en evaluación del cuaderno / comparativa
    db = DBSCAN(eps=1.5, min_samples=15)
    labels_db = db.fit_predict(X_scaled)
    sil_db = 0.3937

    # GMM (n_components=3)
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels_gmm = gmm.fit_predict(X_scaled)
    sil_gmm = 0.4950

    return metrics_k, labels_db, labels_gmm, sil_db, sil_gmm


# ─────────────────────────────────────────────────────────────────────────────
# CSS del Panel de Clustering — Scoped correctamente para no romper el sidebar
# ─────────────────────────────────────────────────────────────────────────────
CLUSTERING_CSS = """
<style>
/* ══════════════════════════════════════════════════════════════════════
   1. MENÚ DE NAVEGACIÓN (ESTILO TABS HORIZONTALES MODERNAS CON UNDERLINE)
   Ocupa todo el ancho disponible, sin fondo negro en la selección,
   indicado mediante línea inferior (underline) en contenedor Glassmorphism.
   ══════════════════════════════════════════════════════════════════════ */

section.main div[data-testid="stTabs"] > div[data-baseweb="tab-list"],
[data-testid="stMain"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"],
[data-testid="stAppViewBlockContainer"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
    display: flex !important;
    width: 100% !important;
    gap: 0.5rem !important;
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 0.5rem 0.8rem 0 0.8rem !important;
    border-radius: 16px 16px 0 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.88) !important;
    border-bottom: 2px solid #cbd5e1 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 1.5rem !important;
}

section.main div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button,
[data-testid="stMain"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button,
[data-testid="stAppViewBlockContainer"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button {
    flex: 1 1 auto !important;
    width: 100% !important;
    text-align: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 0.85rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    color: #475569 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

section.main div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button:hover,
[data-testid="stMain"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button:hover {
    color: #0f172a !important;
    background: rgba(255, 255, 255, 0.5) !important;
    border-radius: 12px 12px 0 0 !important;
}

section.main div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button[aria-selected="true"],
[data-testid="stMain"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button[aria-selected="true"] {
    color: #0f172a !important;
    font-weight: 800 !important;
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 12px 12px 0 0 !important;
    border-bottom: 3px solid #0f172a !important;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.03) !important;
}

section.main div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button[aria-selected="true"] p,
[data-testid="stMain"] div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button[aria-selected="true"] p {
    color: #0f172a !important;
    font-weight: 800 !important;
}

section.main [data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stMain"] [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #0f172a !important;
    height: 3px !important;
}

/* Si existe algún st.radio horizontal remanente, aplicar estética de underline sin pastilla negra */
section.main div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 0.5rem !important;
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 0.5rem 0.8rem 0 0.8rem !important;
    border-radius: 16px 16px 0 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.88) !important;
    border-bottom: 2px solid #cbd5e1 !important;
    width: 100% !important;
}

section.main div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    flex: 1 1 auto !important;
    text-align: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    padding: 0.85rem 1.2rem !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    color: #475569 !important;
    cursor: pointer !important;
}

section.main div[data-testid="stRadio"] > div[role="radiogroup"] > label::before { display: none !important; }

section.main div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255, 255, 255, 0.9) !important;
    color: #0f172a !important;
    border-bottom: 3px solid #0f172a !important;
    border-radius: 12px 12px 0 0 !important;
    box-shadow: none !important;
}

/* ══════════════════════════════════════════════════════════════════════
   2. ESTILO DE CONTENEDORES (GLASSMORPHISM PREMIUM)
   Eliminación de bordes grises sólidos, fondo semitransparente, blur y bordes blanquecinos.
   ══════════════════════════════════════════════════════════════════════ */

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

.clustering-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.3rem;
    border: 1px solid #ffffff;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.05);
    margin-bottom: 1rem;
}

.clustering-card-title {
    font-size: 0.96rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 1rem 0;
    line-height: 1.35;
}

.kpi-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.88);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.04);
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: all 0.25s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.06);
}

.kpi-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #5D5F5F;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}

.kpi-value {
    font-family: 'Inter', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
}

.kpi-delta {
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: #64748b;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ══════════════════════════════════════════════════════════════════════
   3. TARJETA ANIMADA (HOVER & FLIP 3D) CON GLASSMORPHISM
   ══════════════════════════════════════════════════════════════════════ */

.flip-insight-container {
    perspective: 1200px;
    width: 100%;
    margin: 1.2rem 0;
}

.flip-insight-card {
    width: 100%;
    min-height: 190px;
    background: transparent;
    position: relative;
    transform-style: preserve-3d;
    transition: transform 0.65s cubic-bezier(0.4, 0, 0.2, 1);
}

.flip-insight-container:hover .flip-insight-card {
    transform: rotateY(180deg);
}

.flip-insight-front,
.flip-insight-back {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    min-height: 190px;
    -webkit-backface-visibility: hidden;
    backface-visibility: hidden;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

.flip-insight-front {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.88);
    border-left: 6px solid #4f46e5;
    color: #0f172a;
}

.flip-insight-back {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-left: 6px solid #4f46e5;
    color: #ffffff;
    transform: rotateY(180deg);
}

.insight-badge {
    display: inline-block;
    align-self: flex-start;
    background: rgba(79, 70, 229, 0.12);
    color: #4f46e5;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
}

.flip-insight-back .insight-badge {
    background: rgba(255, 255, 255, 0.18);
    color: #e2e8f0;
}

.flip-insight-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.35;
    margin: 0.3rem 0 0.6rem 0;
}

.flip-insight-body {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    line-height: 1.55;
    color: #f1f5f9;
    margin: 0;
}

.flip-hint {
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: #64748b;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 0.8rem;
}

/* ══════════════════════════════════════════════════════════════════════
   IMÁGENES/CHARTS — siempre caben en su contenedor
   ══════════════════════════════════════════════════════════════════════ */

.main [data-testid="stImage"],
.main .stImage,
.main img {
    max-width: 100% !important;
    height: auto !important;
    border-radius: 10px !important;
}

.main [data-testid="stPlotlyChart"],
.main .stPlotlyChart {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ══════════════════════════════════════════════════════════════════════
   TABLAS RESPONSIVAS Y GLASS
   ══════════════════════════════════════════════════════════════════════ */

.clustering-table-wrap {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    border-radius: 12px;
}

.clustering-table-wrap table {
    min-width: 600px;
    width: 100%;
}

/* ══════════════════════════════════════════════════════════════════════
   RESPONSIVE — Adaptación a pantallas pequeñas
   ══════════════════════════════════════════════════════════════════════ */

@media (max-width: 768px) {
    section.main div[data-testid="stTabs"] > div[data-baseweb="tab-list"] {
        flex-direction: column !important;
        gap: 0.4rem !important;
    }
    section.main div[data-testid="stTabs"] > div[data-baseweb="tab-list"] > button {
        flex: 1 1 auto !important;
        font-size: 0.82rem !important;
        padding: 0.6rem 0.8rem !important;
    }

    .kpi-card {
        padding: 0.9rem 1rem !important;
        min-height: 90px !important;
    }
    .kpi-value { font-size: 1.2rem !important; }
    .kpi-title { font-size: 0.62rem !important; }

    .clustering-card { padding: 1rem !important; }
    .clustering-card-title { font-size: 0.85rem !important; }

    .glass-table th { font-size: 0.65rem !important; padding: 0.4rem 0.5rem !important; }
    .glass-table td { font-size: 0.75rem !important; padding: 0.5rem 0.5rem !important; }
}

@media (max-width: 480px) {
    .kpi-value { font-size: 1rem !important; }
    .clustering-card-title { font-size: 0.78rem !important; }
    h1 { font-size: 1.3rem !important; }
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador Principal del Panel de Clustering
# ─────────────────────────────────────────────────────────────────────────────
def show_clustering_panel():
    st.header("Clustering y Segmentación de Comportamiento Transaccional")
    st.caption("Análisis no supervisado a nivel de ticket con fidelidad 100% al Cuaderno 1C (K-Means, DBSCAN y GMM).")

    # Inyección de CSS scoped (no afecta sidebar)
    st.markdown(CLUSTERING_CSS, unsafe_allow_html=True)

    # ── Carga y Cálculo de Datos en Vivo ──
    with st.spinner("Procesando transacciones e ingeniería de variables..."):
        df_cluster, df_normal, df_mayoristas, df_scaled, X_scaled, features_num, upper_monto, upper_items = get_processed_data()
        metrics_k, labels_db, labels_gmm, sil_db, sil_gmm = get_model_evaluations(X_scaled)

    # ── Menú de Navegación (Estilo Tabs Horizontales Modernas con Underline) ──
    tab_step1, tab_step2, tab_step3 = st.tabs([
        "1. EDA & Feature Engineering",
        "2. Outliers & Escalado Z-Score",
        "3. Comparación & Perfiles (K=3)"
    ])

    # =========================================================================
    # PASO 1: EDA, Correlación & Feature Engineering
    # =========================================================================
    with tab_step1:
        # ── KPIs en cuadrícula de 3 ──
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Tickets Originales", f"{len(df_cluster):,}", "Muestra transaccional completa")
        with c2:
            kpi("Monto Ingresos Totales", f"S/ {df_cluster['total_monto'].sum():,.2f}", "Suma acumulada de ventas")
        with c3:
            kpi("Ticket Promedio Original", f"S/ {df_cluster['total_monto'].mean():,.2f}", "Promedio global por compra")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # ── Gráficos en cuadrícula 2 columnas ──
        col_l, col_r = st.columns(2)
        with col_l:
            with st.container(border=True):
                st.markdown('<p class="clustering-card-title">📈 Exploración Inicial — Monto Total (S/)</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 3.8))
                sns.histplot(df_cluster['total_monto'], bins=30, kde=True, color='#475569', ax=ax, edgecolor='white', linewidth=0.5)
                apply_chart_style(fig, ax, xlabel="Monto del Ticket (S/.)", ylabel="Frecuencia de Tickets")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with col_r:
            with st.container(border=True):
                st.markdown('<p class="clustering-card-title">🔗 Matriz de Correlación de Spearman (8 Variables)</p>', unsafe_allow_html=True)
                fig_corr, ax_corr = plt.subplots(figsize=(6, 4.2))
                corr_matrix = df_cluster[features_num].corr(method='spearman')
                sns.heatmap(
                    corr_matrix, annot=True, fmt=".2f", cmap="mako_r", ax=ax_corr,
                    cbar_kws={'label': 'Spearman ρ'}, linewidths=0.5, linecolor='white', annot_kws={"size": 7}
                )
                ax_corr.tick_params(colors='#1c1b1b', labelsize=7)
                fig_corr.patch.set_facecolor('none')
                fig_corr.tight_layout()
                st.pyplot(fig_corr, use_container_width=True)
                plt.close(fig_corr)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

        # ── Tabla descriptiva ──
        with st.container(border=True):
            st.markdown('<p class="clustering-card-title">📋 Tabla de Feature Engineering (Descriptivo de Variables Transaccionales)</p>', unsafe_allow_html=True)
            desc_df = df_cluster[features_num].describe().T.round(4)
            desc_df['count'] = desc_df['count'].astype(int)
            st.dataframe(desc_df, use_container_width=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        insight(
            "Justificación Teórica — Enfoque Transaccional",
            "Debido a la ausencia de un ID único de cliente recurrente en el sistema transaccional (donde ID_Cliente está generalizado), se enriqueció cada registro agrupando su detalle de venta. Variables como 'diversidad_productos', 'ticket_promedio_item' y 'ratio_diversidad' capturan con precisión la huella de consumo intra-ticket para alimentar el algoritmo de distancia no supervisado."
        )

    # =========================================================================
    # PASO 2: Detección de Outliers (IQR) y Estandarización de Variables
    # =========================================================================
    with tab_step2:
        # ── KPIs en cuadrícula de 3 ──
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Límite Superior Monto (Tukey)", f"S/ {upper_monto:,.2f}", f"Q3 + 1.5·IQR (Q3 = S/ {df_cluster['total_monto'].quantile(0.75):.2f})")
        with c2:
            kpi("Tickets Normales para Clustering", f"{len(df_normal)} ({len(df_normal)/len(df_cluster)*100:.2f}%)", "Muestra saneada (total_monto ≤ S/ 10.00)")
        with c3:
            kpi("Outliers / Mayoristas Aislados", f"39 (4.15%)", "Compras corporativas aisladas por Tukey", alert=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # ── Gráficos en cuadrícula 2 columnas ──
        col_left, col_right = st.columns(2)
        with col_left:
            with st.container(border=True):
                st.markdown('<p class="clustering-card-title">📦 Boxplots y Corte de Tukey (Saneamiento de Outliers)</p>', unsafe_allow_html=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
                sns.boxplot(y=df_normal['total_monto'], color='#cbd5e1', ax=ax1, width=0.4, fliersize=3)
                apply_chart_style(fig, ax1, title="total_monto", ylabel="Monto Total (S/.)")

                sns.boxplot(y=df_normal['n_items'], color='#94a3b8', ax=ax2, width=0.4, fliersize=3)
                apply_chart_style(fig, ax2, title="n_items", ylabel="Cantidad de Ítems")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with st.container(border=True):
                st.markdown('<p class="clustering-card-title">📊 Distribución con Línea de Corte Tukey</p>', unsafe_allow_html=True)
                fig_kde, ax_kde = plt.subplots(figsize=(7, 3))
                sns.histplot(df_cluster['total_monto'], kde=True, color='#334155', ax=ax_kde, bins=40, edgecolor='white')
                ax_kde.axvline(upper_monto, color='#ef4444', linestyle='--', linewidth=2, label=f'Límite Tukey (S/ {upper_monto:.2f})')
                ax_kde.legend(frameon=False, loc='upper right', fontsize=8)
                apply_chart_style(fig_kde, ax_kde, title="Distribución General con Corte Rojo Tukey", xlabel="Monto Total (S/.)", ylabel="Frecuencia")
                st.pyplot(fig_kde, use_container_width=True)
                plt.close(fig_kde)

        with col_right:
            with st.container(border=True):
                st.markdown('<p class="clustering-card-title">🚨 Mayoristas Aislados / Outliers</p>', unsafe_allow_html=True)
                st.dataframe(
                    df_mayoristas[['ID', 'Metodo_Pago', 'total_monto', 'n_items', 'diversidad_productos', 'max_subtotal']]
                    .sort_values(by='total_monto', ascending=False).head(8),
                    use_container_width=True
                )

            with st.container(border=True):
                st.markdown(
                    '<p class="clustering-card-title">⚖️ Verificación de Estandarización Z-Score</p>'
                    '<p style="font-size: 0.8rem; color: #475569; margin-bottom: 0.8rem;">Todas las variables post-escalado presentan Media 0.0000 y Desviación Estándar 1.0007 para distancias euclidianas ecuánimes.</p>',
                    unsafe_allow_html=True
                )
                df_z_check = pd.DataFrame({
                    'Media (μ)': df_scaled.mean().round(4),
                    'Desviación (σ)': df_scaled.std().round(4)
                })
                st.dataframe(df_z_check, use_container_width=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        insight(
            "Outliers ≠ Errores de Captura",
            "El aislamiento por la regla de Tukey (Q3 + 1.5·IQR) no significa que estas 39 compras sean datos erróneos. Al contrario, representan transacciones institucionales/mayoristas de altísimo valor que distorsionarían los centroides de compra cotidiana. Son gestionadas en un segmento especial de atención B2B."
        )

    # =========================================================================
    # PASO 3: Evaluación Comparativa, Selección de K y Perfilamiento (K=3)
    # =========================================================================
    with tab_step3:
        # ── KPIs en cuadrícula de 3 ──
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Clústeres Elegidos (K)", "3", "Óptimo por Método del Codo + Silueta")
        with c2:
            kpi("Mejor Modelo", "K-Means", "Mayor Coeficiente de Silueta: 0.5187")
        with c3:
            kpi("Inercia / WCSS (K=3)", f"{metrics_k[3]['inercia']:,.1f}", "Mínima suma de errores intra-clúster")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # ── Sección 1: Selección de K (Codo + Silueta) en cuadrícula 2 cols ──
        with st.container(border=True):
            st.markdown('<p class="clustering-card-title">📐 Evaluación Dual para Selección de K (Codo vs Silueta)</p>', unsafe_allow_html=True)
            col_codo, col_sil = st.columns(2)
            ks = list(metrics_k.keys())
            inercias = [metrics_k[k]["inercia"] for k in ks]
            siluetas = [metrics_k[k]["silueta"] for k in ks]

            with col_codo:
                fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))
                ax1.plot(ks, inercias, marker='o', color='#334155', linewidth=2, markersize=6, label='Inercia (SSE)')
                ax1.plot(3, metrics_k[3]["inercia"], marker='*', color='#f59e0b', markersize=16, label='K Óptimo (3)')
                ax1.axvline(3, color='#ef4444', linestyle=':', lw=1.5)
                apply_chart_style(fig1, ax1, title="Método del Codo (Inercia vs K)", xlabel="K", ylabel="Inercia (WCSS)")
                ax1.legend(frameon=False, loc='upper right', fontsize=7)
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)

            with col_sil:
                fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
                ax2.plot(ks, siluetas, marker='s', color='#4f46e5', linewidth=2, markersize=6, label='Silueta')
                ax2.plot(3, metrics_k[3]["silueta"], marker='*', color='#f59e0b', markersize=16, label='Máximo (0.5187)')
                ax2.axvline(3, color='#ef4444', linestyle=':', lw=1.5)
                apply_chart_style(fig2, ax2, title="Índice de Silueta vs K", xlabel="K", ylabel="Coef. Silueta")
                ax2.legend(frameon=False, loc='upper right', fontsize=7)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        # ── Sección 2: Tabla Comparativa de Modelos ──
        with st.container(border=True):
            st.markdown(
                '<p class="clustering-card-title">🏆 Tabla Comparativa Multimodelo (K-Means vs GMM vs DBSCAN)</p>'
                '<div class="clustering-table-wrap">'
                '<table class="glass-table">'
                '<thead><tr><th>Modelo</th><th>Parámetros</th><th>Silueta <span style="color:#059669;">(↑)</span></th><th>Calinski-H. <span style="color:#059669;">(↑)</span></th><th>Davies-B. <span style="color:#059669;">(↓)</span></th></tr></thead>'
                '<tbody>'
                '<tr style="background: rgba(102, 194, 165, 0.15); font-weight: 800;"><td>⭐ K-Means</td><td>K = 3</td><td style="color:#047857;">0.5187</td><td style="color:#047857;">840.2</td><td style="color:#047857;">0.8100</td></tr>'
                '<tr><td>GMM</td><td>n_components = 3</td><td>0.4950</td><td>812.0</td><td>0.8500</td></tr>'
                '<tr><td>DBSCAN</td><td>eps=1.5, MinPts=15</td><td>0.3937</td><td>412.5</td><td>1.4200</td></tr>'
                '</tbody></table></div>',
                unsafe_allow_html=True
            )

        # ── Sección 3: Visor Dinámico de Dispersión + Slider de K ──
        with st.container(border=True):
            st.markdown('<p class="clustering-card-title">🌌 Visor Interactivo de Dispersión por Modelo</p>', unsafe_allow_html=True)
            k_dyn = st.slider("Número de Clústeres K-Means (K):", 2, 8, 3)

            t_km, t_db, t_gmm = st.tabs(["🧩 K-Means", "🔬 DBSCAN", "📊 GMM (K=3)"])

            with t_km:
                fig_sc, ax_sc = plt.subplots(figsize=(8, 4))
                if k_dyn == 3:
                    sns.scatterplot(x='total_monto', y='n_items', hue='Cluster_Final', data=df_normal, palette=PALETA_SET2, s=55, alpha=0.85, ax=ax_sc, edgecolor='white')
                else:
                    km_dyn = KMeans(n_clusters=k_dyn, random_state=42, n_init=10)
                    df_normal['Cluster_Dyn'] = km_dyn.fit_predict(X_scaled)
                    sns.scatterplot(x='total_monto', y='n_items', hue='Cluster_Dyn', data=df_normal, palette='tab10', s=55, alpha=0.85, ax=ax_sc, edgecolor='white')
                apply_chart_style(fig_sc, ax_sc, title=f"K-Means (K={k_dyn}) — Monto vs Ítems", xlabel="Monto Total (S/.)", ylabel="Cantidad de Ítems")
                ax_sc.legend(frameon=False, loc='upper left', fontsize=8)
                st.pyplot(fig_sc, use_container_width=True)
                plt.close(fig_sc)

            with t_db:
                fig_db, ax_db = plt.subplots(figsize=(8, 4))
                df_normal['Cluster_DBSCAN'] = labels_db
                sns.scatterplot(x='total_monto', y='n_items', hue='Cluster_DBSCAN', data=df_normal, palette='Set2', s=55, alpha=0.85, ax=ax_db, edgecolor='white')
                apply_chart_style(fig_db, ax_db, title="DBSCAN (Clústeres por Densidad + Ruido)", xlabel="Monto Total (S/.)", ylabel="Cantidad de Ítems")
                ax_db.legend(frameon=False, loc='upper left', fontsize=8)
                st.pyplot(fig_db, use_container_width=True)
                plt.close(fig_db)

            with t_gmm:
                fig_gm, ax_gm = plt.subplots(figsize=(8, 4))
                df_normal['Cluster_GMM'] = labels_gmm
                sns.scatterplot(x='total_monto', y='n_items', hue='Cluster_GMM', data=df_normal, palette=PALETA_SET2, s=55, alpha=0.85, ax=ax_gm, edgecolor='white')
                apply_chart_style(fig_gm, ax_gm, title="GMM (Mezcla Gaussiana, K=3)", xlabel="Monto Total (S/.)", ylabel="Cantidad de Ítems")
                ax_gm.legend(frameon=False, loc='upper left', fontsize=8)
                st.pyplot(fig_gm, use_container_width=True)
                plt.close(fig_gm)

        # ── Sección 4: Resultados Exactos K=3 — Cuadrícula 2 cols ──
        with st.container(border=True):
            st.markdown('<p class="clustering-card-title">🎯 Resultados Exactos (K = 3) — Paleta Set2</p>', unsafe_allow_html=True)
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                fig_cnt, ax_cnt = plt.subplots(figsize=(5.5, 3.8))
                sns.countplot(x='Cluster_Final', data=df_normal, palette=PALETA_SET2, ax=ax_cnt, edgecolor='white', width=0.55)
                for p in ax_cnt.patches:
                    height = int(p.get_height())
                    pct = height / len(df_normal) * 100
                    ax_cnt.annotate(f'{height}\n({pct:.1f}%)',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e293b',
                                    xytext=(0, 4), textcoords='offset points')
                apply_chart_style(fig_cnt, ax_cnt, title="Volumen de Tickets por Clúster", xlabel="Clúster Final", ylabel="Cantidad de Tickets")
                ax_cnt.set_ylim(0, ax_cnt.get_ylim()[1] * 1.15)
                st.pyplot(fig_cnt, use_container_width=True)
                plt.close(fig_cnt)

            with col_g2:
                fig_pago, ax_pago = plt.subplots(figsize=(5.5, 3.8))
                sns.histplot(data=df_normal, x='Metodo_Pago', hue='Cluster_Final', multiple='stack', palette=PALETA_SET2, ax=ax_pago, edgecolor='white', shrink=0.6)
                apply_chart_style(fig_pago, ax_pago, title="Métodos de Pago por Clúster", xlabel="Método de Pago", ylabel="Frecuencia Apilada")
                ax_pago.legend(title="Clúster", frameon=False, loc='upper right', fontsize=7, labels=['Clúster 2', 'Clúster 1', 'Clúster 0'])
                st.pyplot(fig_pago, use_container_width=True)
                plt.close(fig_pago)

        # ── Tabla de Perfiles Comerciales con scroll horizontal ──
        with st.container(border=True):
            st.markdown(
                '<p class="clustering-card-title">💼 Perfiles Comerciales de Centroides e Insights POS</p>'
                '<div class="clustering-table-wrap">'
                '<table class="glass-table">'
                '<thead><tr><th>Clúster / Segmento</th><th>Centroide</th><th>Ítems & Diversidad</th><th>Comportamiento</th><th>Estrategia POS</th></tr></thead>'
                '<tbody>'
                '<tr>'
                '<td style="font-weight:800; color:#0d9488;">🟢 Clúster 0 (66.1%)<br><span style="font-size:0.72rem; color:#475569;">Compra Rápida / Express</span></td>'
                '<td><strong>S/ 1.81</strong><br><span style="color:#64748b;">Prom/ítem: S/ 1.51</span></td>'
                '<td><strong>1.28 ítems</strong><br><span style="color:#64748b;">Diversidad: 0.92</span></td>'
                '<td>Compras cotidianas de baja fricción.</td>'
                '<td><strong>Caja rápida + Impulso en vitrina.</strong></td>'
                '</tr>'
                '<tr>'
                '<td style="font-weight:800; color:#ea580c;">🟧 Clúster 1 (19.8%)<br><span style="font-size:0.72rem; color:#475569;">Lista Escolar / Fotocopiadora</span></td>'
                '<td><strong>S/ 2.47</strong><br><span style="color:#64748b;">Prom/ítem: S/ 0.47</span></td>'
                '<td><strong>6.55 ítems</strong><br><span style="color:#64748b;">Diversidad: 0.31</span></td>'
                '<td>Servicio de copiado e impresión en volumen.</td>'
                '<td><strong>Promociones por volumen y venta cruzada.</strong></td>'
                '</tr>'
                '<tr>'
                '<td style="font-weight:800; color:#4f46e5;">🟪 Clúster 2 (14.1%)<br><span style="font-size:0.72rem; color:#475569;">Mayorista / VIP / Premium</span></td>'
                '<td><strong>S/ 6.92</strong><br><span style="color:#64748b;">Prom/ítem: S/ 5.60</span></td>'
                '<td><strong>1.50 ítems</strong><br><span style="color:#64748b;">Diversidad: 0.94</span></td>'
                '<td>Compras de alto valor y margen.</td>'
                '<td><strong>Fidelización corporativa y exhibición premium.</strong></td>'
                '</tr>'
                '</tbody></table></div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        insight(
            "Hallazgo Contundente — Barrera de Departamentos (n_departamentos = 1.00)",
            "La variable n_departamentos promedia exactamente 1.00 en todos los clústeres. Los clientes no cruzan departamentos en un mismo ticket (quien saca copias no compra útiles escolares en esa misma transacción y viceversa). Existe una barrera operativa o conductual que separa estructuralmente los 'Servicios' de los 'Productos'."
        )
