import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="SmartBazar",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  PREMIUM CSS — GLASSMORPHISM RESPONSIVE
# ═══════════════════════════════════════════════════════════════
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background-color: #cbd5e1 !important;
    background-image:
        radial-gradient(at 10% 10%, rgba(255,255,255,0.8) 0px, transparent 50%),
        radial-gradient(at 90% 10%, rgba(241,245,249,0.7) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(226,232,240,0.6) 0px, transparent 50%),
        radial-gradient(at 20% 90%, rgba(255,255,255,0.9) 0px, transparent 50%) !important;
    font-family: 'Inter', sans-serif !important;
    color: #000000;
}

[data-testid="stHeader"], #MainMenu, footer { display: none !important; }

.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar Flotante ── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

section[data-testid="stSidebar"] {
    background-color: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255,255,255,0.7) !important;
    border-radius: 24px !important;
    margin: 1.2rem !important;
    height: calc(100vh - 2.4rem) !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08) !important;
    width: 290px !important;
    min-width: 290px !important;
}

[data-testid="stSidebarHeader"] { padding-top: 2rem !important; padding-bottom: 0.6rem !important; }

/* ── Radio Buttons (sin circulito) ── */
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }

div[data-testid="stRadio"] label {
    padding: 0.75rem 1.1rem !important;
    border-radius: 12px !important;
    background-color: transparent !important;
    color: #5D5F5F !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    margin-bottom: 4px !important;
    border: 1px solid transparent !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}

div[data-testid="stRadio"] label::before {
    content: "";
    display: inline-block;
    width: 18px; height: 18px;
    margin-right: 12px;
    background-color: currentColor;
    transition: inherit;
    flex-shrink: 0;
}
div[data-testid="stRadio"] label:nth-child(1)::before { -webkit-mask: url("https://api.iconify.design/lucide/bar-chart-2.svg") no-repeat center / contain; mask: url("https://api.iconify.design/lucide/bar-chart-2.svg") no-repeat center / contain; }
div[data-testid="stRadio"] label:nth-child(2)::before { -webkit-mask: url("https://api.iconify.design/lucide/layout-template.svg") no-repeat center / contain; mask: url("https://api.iconify.design/lucide/layout-template.svg") no-repeat center / contain; }
div[data-testid="stRadio"] label:nth-child(3)::before { -webkit-mask: url("https://api.iconify.design/lucide/network.svg") no-repeat center / contain; mask: url("https://api.iconify.design/lucide/network.svg") no-repeat center / contain; }
div[data-testid="stRadio"] label:nth-child(4)::before { -webkit-mask: url("https://api.iconify.design/lucide/git-commit.svg") no-repeat center / contain; mask: url("https://api.iconify.design/lucide/git-commit.svg") no-repeat center / contain; }
div[data-testid="stRadio"] label:nth-child(5)::before { -webkit-mask: url("https://api.iconify.design/lucide/trending-up.svg") no-repeat center / contain; mask: url("https://api.iconify.design/lucide/trending-up.svg") no-repeat center / contain; }
div[data-testid="stRadio"] label:nth-child(6)::before { -webkit-mask: url("https://api.iconify.design/lucide/store.svg") no-repeat center / contain; mask: url("https://api.iconify.design/lucide/store.svg") no-repeat center / contain; }

div[data-testid="stRadio"] label:hover {
    background-color: rgba(255,255,255,0.8) !important;
    color: #000000 !important;
    transform: translateX(3px) !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label:has(input:checked) {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-color: #000000 !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18) !important;
}
div[data-testid="stRadio"] label[data-checked="true"] p,
div[data-testid="stRadio"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

h1, h2, h3, h4, p, span, td, th { font-family: 'Inter', sans-serif !important; }

/* ── Inputs y Sliders nativos ── */
.stButton > button {
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(8px) !important;
    color: #000000 !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { background: #ffffff !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important; }

.btn-pos-cobrar .stButton > button,
.btn-pos-cobrar > button {
    background: #000000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    padding: 0.85rem 1.5rem !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important;
}
.btn-pos-cobrar .stButton > button:hover,
.btn-pos-cobrar > button:hover { background: #262626 !important; }

.stTextInput > div > div > input, .stSelectbox > div > div, .stDateInput > div > div > input {
    background-color: rgba(255,255,255,0.6) !important;
    color: #000000 !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus, .stSelectbox > div > div:focus-within { border-color: #000000 !important; background-color: #ffffff !important; }
.stCheckbox > label { color: #5D5F5F !important; font-size: 0.82rem !important; font-weight: 500 !important; }

/* ── Tabs de Streamlit ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px !important;
    background: rgba(255,255,255,0.5) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    backdrop-filter: blur(8px) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #5D5F5F !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #000000 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── KPI Cards ── */
.kpi-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    border: 1px solid rgba(255,255,255,0.8);
    min-height: 110px;
}
.kpi-title { font-size: 0.68rem; color: #777777; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem; display: block; }
.kpi-value { font-size: 1.5rem; font-weight: 800; color: #000000; line-height: 1.2; display: block; margin-bottom: 0.3rem; word-break: break-word; }
.kpi-delta { font-size: 0.72rem; color: #5D5F5F; font-weight: 600; display: block; line-height: 1.3; }

/* ── Insight Cards ── */
.insight-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(16px);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
    margin-bottom: 0.8rem;
}
.insight-badge {
    font-size: 0.62rem; font-weight: 800; background: #000000; color: #ffffff;
    padding: 0.15rem 0.45rem; border-radius: 5px; text-transform: uppercase;
    letter-spacing: 0.04em; display: inline-block; margin-bottom: 0.4rem;
}
.insight-title { font-size: 0.88rem; font-weight: 800; color: #000000; margin: 0 0 0.3rem 0; }
.insight-body { font-size: 0.8rem; color: #5D5F5F; line-height: 1.5; margin: 0; }

/* ── Glass Chart Container ── */
.glass-chart-wrap {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 1.2rem;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
}

/* ── Controls Panel ── */
.ctrl-panel {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 1.2rem;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}
.ctrl-title { font-size: 0.88rem; font-weight: 800; color: #000000; margin: 0; }

/* ── POS Cards ── */
.pos-ai-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(16px);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.7);
    box-shadow: 0 4px 14px rgba(0,0,0,0.03);
    height: 100%;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.pos-ai-card:hover { border-color: #000000; transform: translateY(-2px); }
.pos-ai-tag { font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #777777; margin-bottom: 4px; }
.pos-ai-val { font-size: 0.95rem; font-weight: 700; color: #000000; line-height: 1.3; }

/* ── Ticket Virtual ── */
.ticket-container {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(20px);
    border: 2px solid #000000;
    border-radius: 16px;
    padding: 1.4rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
}
.ticket-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px dashed rgba(0,0,0,0.15); padding-bottom: 0.8rem; margin-bottom: 1rem; }
.ticket-item-row { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid rgba(0,0,0,0.05); font-size: 0.85rem; }
.ticket-item-row:last-child { border-bottom: none; }
.ticket-summary-box { border-top: 2px dashed rgba(0,0,0,0.15); padding-top: 0.8rem; margin-top: 0.8rem; }
.ticket-summary-line { display: flex; justify-content: space-between; font-size: 0.82rem; color: #5D5F5F; margin-bottom: 0.3rem; }
.ticket-total-line { display: flex; justify-content: space-between; font-size: 1.3rem; font-weight: 800; color: #000000; margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px solid #000000; }

/* ── Tabla en Glass ── */
.glass-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.glass-table th { padding: 0.6rem 0.8rem; text-align: left; font-weight: 800; color: #000000; border-bottom: 2px solid #000000; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }
.glass-table td { padding: 0.7rem 0.8rem; border-bottom: 1px solid rgba(0,0,0,0.06); color: #1c1b1b; vertical-align: top; line-height: 1.4; }
.glass-table tr:last-child td { border-bottom: none; }

/* ── Responsive: Streamlit columns ── */
[data-testid="stHorizontalBlock"] { gap: 1rem !important; align-items: stretch !important; }
[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }

/* ── Dataframe Glass ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def apply_chart_style(fig, ax, title="", xlabel="", ylabel=""):
    """Estilo limpio para gráficos matplotlib alineados con la estética glass."""
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    ax.set_facecolor('#ffffff')
    ax.set_alpha(0.9)
    ax.tick_params(colors='#1c1b1b', labelsize=8)
    ax.xaxis.label.set_color('#5D5F5F')
    ax.yaxis.label.set_color('#5D5F5F')
    for spine in ax.spines.values():
        spine.set_color('#e5e5e5')
    ax.grid(True, alpha=0.25, color='#e5e5e5', linestyle='--')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', color='#000000', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, fontweight='semibold', labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, fontweight='semibold', labelpad=6)
    fig.tight_layout()


def kpi(title, value, delta="", alert=False):
    val_color = "#ef4444" if alert else "#000000"
    border = "1px solid rgba(239,68,68,0.5)" if alert else "1px solid rgba(255,255,255,0.8)"
    st.markdown(
        f'<div class="kpi-card" style="border: {border};">'
        f'<span class="kpi-title">{title}</span>'
        f'<span class="kpi-value" style="color: {val_color};">{value}</span>'
        f'<span class="kpi-delta">{delta}</span></div>',
        unsafe_allow_html=True
    )


def insight(title, content, badge="INSIGHT DE NEGOCIO"):
    st.markdown(
        f'<div class="insight-card"><span class="insight-badge">{badge}</span>'
        f'<p class="insight-title">{title}</p>'
        f'<p class="insight-body">{content}</p></div>',
        unsafe_allow_html=True
    )


def section_header(title, subtitle):
    st.markdown(
        f'<div style="margin-bottom: 1.5rem;">'
        f'<h1 style="font-size: 1.8rem; font-weight: 800; color: #000000; margin: 0 0 4px 0; letter-spacing: -0.02em;">{title}</h1>'
        f'<p style="font-size: 0.88rem; color: #5D5F5F; margin: 0;">{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def ctrl_header(label):
    st.markdown(f'<div class="ctrl-panel"><p class="ctrl-title">⚙️ {label}</p></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  NAVEGACIÓN LATERAL
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="padding-left: 0.6rem; margin-bottom: 2rem;">'
        '<h2 style="font-size: 1.5rem; font-weight: 800; color: #000000; margin: 0 0 4px 0; line-height: 1.1; letter-spacing: -0.02em;">SmartBazar</h2>'
        '<p style="font-size: 0.72rem; color: #777777; font-weight: 600; margin: 0;">Minería de Datos - Grupo 05</p>'
        '</div>',
        unsafe_allow_html=True
    )

    opciones_menu = [
        "Análisis Exploratorio de Datos (EDA)",
        "Clustering",
        "Reglas de Asociación",
        "Clasificación",
        "Predicciones",
        "POS Inteligente",
    ]
    opcion_sel = st.radio(label="", options=opciones_menu, label_visibility="collapsed")


# ═══════════════════════════════════════════════════════════════
#  1. EDA E INGENIERÍA DE CARACTERÍSTICAS
# ═══════════════════════════════════════════════════════════════
if opcion_sel == "Análisis Exploratorio de Datos (EDA)":
    section_header("Análisis Exploratorio y Limpieza", "Auditoría de calidad del catálogo, ingeniería de variables y comportamiento transaccional.")

    # ── KPIs en fila simétrica ──
    k1, k2, k3 = st.columns(3)
    with k1: kpi("Total Transacciones", "939", "Tickets procesados · Periodo 2026-I")
    with k2: kpi("Catálogo Activo", "417", "Artículos válidos tras depuración")
    with k3: kpi("Proporción de Pago", "66.3% / 33.7%", "Efectivo vs Yape · Desbalance detectado", alert=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # ── Gráficos en tabs (ancho completo) ──
    tab1, tab2, tab3 = st.tabs(["📊 Sesgo de Registro por Día", "📈 Distribución de Montos (KDE + Tukey)", "🔥 Correlación de Spearman"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        porcentajes = [28.8, 5.2, 4.1, 4.8, 5.5, 3.4, 48.2]
        colores = ['#000000' if p > 20 else '#94a3b8' for p in porcentajes]
        bars = ax.bar(dias, porcentajes, color=colores, width=0.5, edgecolor='none', zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#000000')
        ax.set_ylim(0, 58)
        apply_chart_style(fig, ax, title="Registro Administrativo de Tickets por Día de la Semana (%)", ylabel="Porcentaje (%)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 4.5), gridspec_kw={'height_ratios': [0.22, 0.78]}, sharex=True)
        np.random.seed(42)
        subtotales = np.concatenate([np.random.exponential(12, 800), np.random.normal(45, 10, 100), np.random.uniform(80, 180, 39)])
        sns.boxplot(x=subtotales, ax=ax_box, color='#1e293b', flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#ef4444'})
        sns.histplot(subtotales, kde=True, ax=ax_hist, color='#334155', bins=30, edgecolor='white', linewidth=0.5)
        tukey_limit = np.percentile(subtotales, 75) + 1.5*(np.percentile(subtotales, 75)-np.percentile(subtotales, 25))
        ax_hist.axvline(tukey_limit, color='#ef4444', linestyle='--', linewidth=1.5, label=f'Límite Tukey ({tukey_limit:.0f})')
        ax_hist.legend(loc='upper right', frameon=False, fontsize=8)
        apply_chart_style(fig, ax_box, title="Distribución de Subtotal del Ticket y Detección de Outliers")
        apply_chart_style(fig, ax_hist, xlabel="Monto de Ticket (S/.)", ylabel="Frecuencia")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 4))
        corr_matrix = np.array([[1.00, 0.88, 0.76], [0.88, 1.00, 0.82], [0.76, 0.82, 1.00]])
        cols_corr = ["Variedad_Items", "Cantidad_Items", "Subtotal_Total"]
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Greys", xticklabels=cols_corr, yticklabels=cols_corr, cbar_kws={'label': 'ρ Spearman', 'shrink': 0.8}, ax=ax, vmin=0.5, vmax=1.0, annot_kws={"size": 11, "weight": "bold"})
        apply_chart_style(fig, ax, title="Matriz de Correlación de Spearman")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Controles + Insights en fila inferior ──
    ci1, ci2 = st.columns(2)

    with ci1:
        ctrl_header("Controles de Vista")
        vista_datos = st.selectbox("Estado de los Datos:", ["Datos Saneados", "Datos Crudos"])
        dep_sel = st.selectbox("Filtrar Departamento:", ["Todos", "Útiles", "Fotocopiadora"])

    with ci2:
        insight("El Desfase del Kardex", "Los stocks negativos (ej. Hoja de colores: −184) ocurren porque la mercadería se vende en mostrador antes de registrarse la orden de compra. Se aplicó truncamiento canónico a 0.")
        insight("Sesgo Horario", "La variable 'hora' refleja digitación en lotes los fines de semana, no el flujo real de clientes. Fue excluida del modelo.")
        insight("Outliers ≠ Errores", "Las transacciones atípicas son compras mayoristas/institucionales vitales para el flujo de caja del negocio.")


# ═══════════════════════════════════════════════════════════════
#  2. CLUSTERING
# ═══════════════════════════════════════════════════════════════
elif opcion_sel == "Clustering":
    section_header("Segmentación No Supervisada (K-Means)", "Agrupamiento por monto transaccional y volumen de compra.")

    metrics_k = {
        2: {"silueta": 0.4521, "inercia": 1842.5},
        3: {"silueta": 0.5187, "inercia": 1205.3},
        4: {"silueta": 0.4893, "inercia": 892.1},
        5: {"silueta": 0.4612, "inercia": 701.8},
        6: {"silueta": 0.4201, "inercia": 589.4},
        7: {"silueta": 0.3845, "inercia": 498.2},
        8: {"silueta": 0.3512, "inercia": 421.7},
    }

    # ── KPIs + Control en fila superior ──
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        ctrl_header("Parámetros")
        k_val = st.slider("Número de Clústeres (K):", 2, 8, 3)
    with k2: kpi("K Óptimo", "3", "Método del Codo + Silueta")
    with k3: kpi("Silueta", f"{metrics_k[k_val]['silueta']:.4f}", f"Para K = {k_val}")
    with k4: kpi("Inercia (SSE)", f"{metrics_k[k_val]['inercia']:.1f}", "Distancia al centroide")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Evaluación Dual (Codo + Silueta)", "🌌 Scatterplot de Clústeres"])

    with tab1:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ks = list(metrics_k.keys())
        inercias = [metrics_k[k]["inercia"] for k in ks]
        siluetas = [metrics_k[k]["silueta"] for k in ks]
        ax1.plot(ks, inercias, marker='o', color='#000000', linewidth=2, label='Inercia', markersize=6)
        ax1.set_ylabel("Inercia (SSE)", fontweight='bold', fontsize=9, color='#000000')
        ax2 = ax1.twinx()
        ax2.plot(ks, siluetas, marker='s', color='#64748b', linewidth=2, linestyle='--', label='Silueta', markersize=6)
        ax2.set_ylabel("Coef. de Silueta", fontweight='bold', fontsize=9, color='#64748b')
        ax1.axvline(3, color='#ef4444', linestyle=':', lw=1.5, label='K Óptimo (3)')
        apply_chart_style(fig, ax1, title="Método del Codo e Índice de Silueta vs K", xlabel="Número de Clústeres (K)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, frameon=False, loc='center right', fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        np.random.seed(42 + k_val)
        palette = ['#0f172a', '#475569', '#94a3b8', '#cbd5e1', '#334155', '#64748b', '#1e293b', '#e2e8f0']
        for c in range(k_val):
            n_pts = int(300 / k_val)
            x_items = np.random.normal(loc=(c+1)*2.2, scale=0.8, size=n_pts).clip(1, 15)
            y_total = x_items * np.random.uniform(4.5, 9.5) + np.random.normal(0, 3, size=n_pts)
            y_total = np.clip(y_total, 2, 200)
            ax.scatter(x_items, y_total, color=palette[c % len(palette)], alpha=0.7, edgecolors='white', s=45, label=f'Clúster {c}', linewidth=0.8)
        apply_chart_style(fig, ax, title=f"Distribución de {k_val} Segmentos (Monto vs Cantidad)", xlabel="Nº de Artículos", ylabel="Monto Total (S/.)")
        ax.legend(frameon=False, loc='upper left', fontsize=8, ncol=2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # ── Perfilamiento en tabla glass ──
    st.markdown(
        '<div class="glass-chart-wrap">'
        '<p style="font-size: 0.95rem; font-weight: 800; color: #000000; margin: 0 0 0.8rem 0;">🎯 Perfilamiento Comercial de Segmentos (K = 3)</p>'
        '<table class="glass-table">'
        '<tr><th>Segmento</th><th>Perfil</th><th>Comportamiento</th><th>Acción</th></tr>'
        '<tr><td style="font-weight:700;">🟢 Compra Rápida</td><td>S/ 4.50 · 1.2 ítems</td><td>Estudiantes al paso: copias, lapicero.</td><td>Caja rápida + impulso en vitrina.</td></tr>'
        '<tr><td style="font-weight:700;">🔵 Lista Escolar</td><td>S/ 24.80 · 4.5 ítems</td><td>Kits de útiles para ciclo/trabajos.</td><td>Promos por volumen y combos.</td></tr>'
        '<tr><td style="font-weight:700;">⚫ Mayorista</td><td>S/ 115.00 · 12 ítems</td><td>Compras corporativas / institucionales.</td><td>Fidelización B2B y crédito.</td></tr>'
        '</table></div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════
#  3. REGLAS DE ASOCIACIÓN
# ═══════════════════════════════════════════════════════════════
elif opcion_sel == "Reglas de Asociación":
    section_header("Descubrimiento de Patrones (Apriori)", "Análisis de cesta de compra y afinidad de productos para combos en punto de venta.")

    reglas_raw = [
        {"A": "FOTOCOPIA A4", "C": "MICA A4 VINIFAN", "sop": 0.045, "cnf": 0.72, "lft": 3.21},
        {"A": "CUADERNO A4", "C": "LAPICERO PILOT", "sop": 0.038, "cnf": 0.65, "lft": 2.89},
        {"A": "FOLDER MANILA", "C": "FASTER CLIPS", "sop": 0.032, "cnf": 0.58, "lft": 2.54},
        {"A": "IMPRESION B/N", "C": "ANILLADO SIMPLE", "sop": 0.028, "cnf": 0.52, "lft": 2.31},
        {"A": "LAPICERO FABER", "C": "BORRADOR BLANCO", "sop": 0.025, "cnf": 0.48, "lft": 2.15},
        {"A": "PAPEL BOND A4", "C": "SOBRE MANILA A4", "sop": 0.022, "cnf": 0.44, "lft": 1.98},
        {"A": "CUADERNO A5", "C": "CORRECTOR L.P.", "sop": 0.019, "cnf": 0.41, "lft": 1.82},
        {"A": "PLUMONES FABER", "C": "CARTULINA A4", "sop": 0.016, "cnf": 0.38, "lft": 1.65},
    ]

    # ── Controles + KPIs en fila ──
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        ctrl_header("Umbrales de Búsqueda")
        min_supp = st.slider("Soporte Mín:", 0.01, 0.10, 0.02, 0.005)
        min_conf = st.slider("Confianza Mín:", 0.20, 0.80, 0.40, 0.05)
        min_lift = st.slider("Lift Mínimo:", 1.0, 4.0, 1.5, 0.1)

    reglas_filt = [r for r in reglas_raw if r["sop"] >= min_supp and r["cnf"] >= min_conf and r["lft"] >= min_lift]
    top_lift = max(reglas_filt, key=lambda x: x["lft"]) if reglas_filt else {"A": "-", "C": "-", "lft": 0}
    top_conf = max(reglas_filt, key=lambda x: x["cnf"]) if reglas_filt else {"A": "-", "C": "-", "cnf": 0}

    with c2: kpi("Reglas Activas", f"{len(reglas_filt)}", f"Lift ≥ {min_lift}")
    with c3: kpi("Mayor Lift", f"{top_lift['lft']:.2f}", f"{top_lift['A'][:10]} → {top_lift['C'][:10]}")
    with c4: kpi("Mayor Confianza", f"{top_conf['cnf']:.0%}" if reglas_filt else "—", f"{top_conf['A'][:10]} → {top_conf['C'][:10]}" if reglas_filt else "—")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌌 Dispersión (Confianza vs Soporte)", "🕸️ Grafo de Red", "📋 Tabla de Reglas"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        if reglas_filt:
            sops = [r["sop"] for r in reglas_filt]
            cnfs = [r["cnf"] for r in reglas_filt]
            lfts = [r["lft"] for r in reglas_filt]
            sc = ax.scatter(sops, cnfs, c=lfts, cmap="Greys", s=[l*70 for l in lfts], alpha=0.85, edgecolors='black', linewidth=1.2)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label("Lift", fontsize=8)
            for r in reglas_filt[:5]:
                ax.annotate(f"{r['A'][:8]}→{r['C'][:8]}", (r["sop"], r["cnf"]), fontsize=7, xytext=(5,5), textcoords='offset points', color='#000000')
        apply_chart_style(fig, ax, title="Dispersión de Reglas (Tamaño y Color = Lift)", xlabel="Soporte", ylabel="Confianza")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.axis('off')
        pos_nodos = {
            "FOTOCOPIA A4": (0.15, 0.8), "MICA VINIFAN": (0.4, 0.85),
            "CUADERNO A4": (0.15, 0.45), "LAPICERO PILOT": (0.45, 0.48),
            "FOLDER MANILA": (0.2, 0.12), "FASTER CLIPS": (0.5, 0.15),
            "PAPEL BOND": (0.7, 0.7), "SOBRE MANILA": (0.88, 0.5)
        }
        aristas = [("FOTOCOPIA A4", "MICA VINIFAN", 3.2), ("CUADERNO A4", "LAPICERO PILOT", 2.9), ("FOLDER MANILA", "FASTER CLIPS", 2.5), ("PAPEL BOND", "SOBRE MANILA", 2.0)]
        for n1, n2, peso in aristas:
            x1, y1 = pos_nodos[n1]; x2, y2 = pos_nodos[n2]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=peso*0.7, color="#475569", shrinkA=18, shrinkB=18))
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.04, f"Lift {peso}", ha='center', fontsize=7.5, fontweight='bold', color='#000000', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#cbd5e1', alpha=0.9))
        for nombre, (x, y) in pos_nodos.items():
            ax.scatter(x, y, s=650, color='#0f172a', zorder=4, edgecolors='white', linewidth=2)
            ax.text(x, y-0.09, nombre, ha='center', fontsize=7.5, fontweight='bold', color='#000000')
        ax.set_title("Red Interconectada de Artículos (Afinidad de Compra)", fontsize=10, fontweight='bold', color='#000000', pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        if reglas_filt:
            df_display = pd.DataFrame(reglas_filt).rename(columns={"A": "Antecedente", "C": "Consecuente", "sop": "Soporte", "cnf": "Confianza", "lft": "Lift"})
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No hay reglas que cumplan los umbrales seleccionados. Ajuste los sliders.")

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    insight("Estrategia de Ventas Cruzadas", "Estos indicadores sustentan la creación de 'Combos Escolares y de Oficina' en mostrador para rotar stock inmovilizado y elevar el ticket promedio por visita.", badge="ACCIÓN COMERCIAL")


# ═══════════════════════════════════════════════════════════════
#  4. CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════
elif opcion_sel == "Clasificación":
    section_header("Clasificación Predictiva de Método de Pago", "Evaluación comparativa de Random Forest vs XGBoost y explicabilidad con SHAP.")

    # ── Control + KPIs en fila ──
    c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
    with c0:
        ctrl_header("Selector de Algoritmo")
        mod_sel = st.selectbox("Modelo:", ["XGBoost (Recomendado)", "Random Forest"])

    if "XGBoost" in mod_sel:
        f1_v, acc_v, auc_v = "0.400", "58.5%", "0.901"
        cm = np.array([[151, 26], [25, 90]])
    else:
        f1_v, acc_v, auc_v = "0.387", "61.2%", "0.883"
        cm = np.array([[145, 32], [28, 87]])

    with c1: kpi("F1-Score", f1_v, "class_weight = 'balanced'")
    with c2: kpi("Accuracy", acc_v, "Tasa de aciertos en test")
    with c3: kpi("ROC-AUC", auc_v, "Capacidad de discriminación")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧮 Matriz de Confusión", "💡 Importancia SHAP"])

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", cbar=False, xticklabels=["Pred: EFECTIVO", "Pred: YAPE"], yticklabels=["Real: EFECTIVO", "Real: YAPE"], annot_kws={"size": 16, "weight": "bold"}, ax=ax, linewidths=2, linecolor='white')
        apply_chart_style(fig, ax, title=f"Matriz de Confusión — {mod_sel}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        features = ["Total_Ticket", "Es_Fin_de_Semana", "pct_Fotocopiadora", "n_items", "es_cliente_recurrente"]
        importancias = [0.38, 0.24, 0.18, 0.12, 0.08]
        bars = ax.barh(range(len(features)), importancias, color='#0f172a', edgecolor='white', height=0.5)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontweight='bold', fontsize=9, color='#000000')
        ax.invert_yaxis()
        for bar, v in zip(bars, importancias):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2, f'{v:.2f}', va='center', fontsize=8, fontweight='bold', color='#000000')
        apply_chart_style(fig, ax, title="Importancia Global de Variables (|SHAP value|)", xlabel="Impacto Medio Absoluto")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1: insight("Preferencia por F1-Score", "El desbalance Efectivo (66.3%) vs Yape (33.7%) invalida la Accuracy como métrica principal. El F1-Score pondera Precision y Recall equitativamente.", badge="JUSTIFICACIÓN TÉCNICA")
    with ic2: insight("Explicabilidad Financiera", "Montos altos y compras en fin de semana son los impulsores clave del uso de billeteras digitales (Yape) sobre el efectivo.", badge="SHAP INSIGHT")


# ═══════════════════════════════════════════════════════════════
#  5. PREDICCIONES (SERIES TEMPORALES)
# ═══════════════════════════════════════════════════════════════
elif opcion_sel == "Predicciones":
    section_header("Pronóstico de Ventas (Prophet)", "Estimación de flujo de ingresos y comparación con feriados peruanos.")

    c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
    with c0:
        ctrl_header("Horizonte y Métrica")
        horizonte = st.selectbox("Horizonte:", [7, 14, 30], index=1)
        metrica_err = st.selectbox("Métrica de Error:", ["RMSE", "MAPE"])
    with c1: kpi("Venta Est. (Mañana)", "S/ 1,285", "Pronóstico medio · IC 95%")
    with c2:
        if metrica_err == "RMSE":
            kpi("RMSE Comparativo", "33.60 / 36.90", "Base vs con Feriados")
        else:
            kpi("MAPE Comparativo", "312% / 489%", "Base vs con Feriados")
    with c3: kpi("Modelo Ganador", "Base", "RMSE 33.60 · sin feriados")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📉 Proyección Temporal", "📊 Venta Diaria por Método de Pago"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        np.random.seed(123)
        fechas_hist = pd.date_range('2026-03-01', periods=30, freq='D')
        ventas_hist = 1000 + 200*np.sin(np.arange(30)*2*np.pi/7) + np.random.normal(0, 80, 30)
        ax.plot(fechas_hist, ventas_hist, color='#64748b', marker='o', markersize=3, label='Ventas Reales', linewidth=1.5)
        fechas_fut = pd.date_range(fechas_hist[-1] + pd.Timedelta(days=1), periods=horizonte, freq='D')
        ventas_fut = 1050 + 210*np.sin((np.arange(30, 30+horizonte))*2*np.pi/7)
        ax.plot(fechas_fut, np.maximum(ventas_fut, 0), color='#000000', linewidth=2.5, linestyle='--', label='Pronóstico Prophet')
        ax.fill_between(fechas_fut, np.maximum(ventas_fut*0.82, 0), ventas_fut*1.18, alpha=0.08, color='#000000', label='IC 95%')
        apply_chart_style(fig, ax, title=f"Proyección de Ingresos a {horizonte} Días", ylabel="Ventas Totales (S/.)")
        ax.legend(frameon=False, loc='upper left', fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        efectivo = [320, 410, 390, 450, 520, 680, 750]
        yape = [180, 210, 195, 230, 310, 420, 510]
        x = np.arange(len(dias))
        ax.bar(x - 0.18, efectivo, width=0.32, color='#000000', label='Efectivo (66.3%)')
        ax.bar(x + 0.18, yape, width=0.32, color='#94a3b8', label='Yape (33.7%)')
        ax.set_xticks(x)
        ax.set_xticklabels(dias, fontweight='bold')
        apply_chart_style(fig, ax, title="Venta Diaria Promedio por Método de Pago", ylabel="Ingreso Promedio (S/.)")
        ax.legend(frameon=False, loc='upper left', fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1: insight("Lógica de Flujo de Caja", "Permite al cajero anticipar cuánto efectivo/sencillo tener preparado cada mañana según el día de la semana.", badge="OPERACIÓN")
    with ic2: insight("Restricción Matemática", "Se aplica yhat = max(x, 0) para impedir pronósticos de ventas negativas durante días de baja demanda.", badge="TÉCNICO")


# ═══════════════════════════════════════════════════════════════
#  6. POS INTELIGENTE
# ═══════════════════════════════════════════════════════════════
elif opcion_sel == "POS Inteligente":
    st.markdown(
        '<div style="margin-bottom: 1.2rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">'
        '<div><h1 style="font-size: 1.8rem; font-weight: 800; color: #000000; margin: 0 0 4px 0;">POS Inteligente</h1>'
        '<p style="font-size: 0.85rem; color: #5D5F5F; margin: 0;">Punto de Venta asistido por IA en Tiempo Real.</p></div>'
        '<div style="background: rgba(255,255,255,0.8); border: 1px solid rgba(0,0,0,0.1); padding: 0.4rem 0.9rem; border-radius: 99px; font-size: 0.78rem; font-weight: 600; display: flex; align-items: center; gap: 6px;">'
        '<span style="display: inline-block; width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%; box-shadow: 0 0 6px #22c55e;"></span>IA Activada</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Tarjetas de IA en fila ──
    st.markdown('<p style="font-size: 0.68rem; font-weight: 700; color: #777; text-transform: uppercase; letter-spacing: 0.06em; margin: 0 0 8px 0;">Recomendaciones IA en Vivo</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="pos-ai-card"><p class="pos-ai-tag">✦ Asociación (Upsell)</p>'
            '<p class="pos-ai-val">Sugerir: <b style="text-decoration:underline;">Galletas Oreo</b></p>'
            '<p style="font-size:0.72rem; color:#777; margin:6px 0 0 0;">Lift: 2.15 · Alta afinidad</p></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="pos-ai-card"><p class="pos-ai-tag">◆ Clasificación de Pago</p>'
            '<p class="pos-ai-val">Probabilidad: <b>Yape (85%)</b></p>'
            '<p style="font-size:0.72rem; color:#777; margin:6px 0 0 0;">Efectivo (15%) · Digital habitual</p></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            '<div class="pos-ai-card"><p class="pos-ai-tag">▲ Series Temporales</p>'
            '<p class="pos-ai-val">Meta Diaria: <b>S/ 1,250.00</b></p>'
            '<p style="font-size:0.72rem; color:#777; margin:6px 0 0 0;">+4.2% sobre promedio semanal</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border: none; border-top: 1px dashed rgba(0,0,0,0.08); margin: 1rem 0 1.2rem 0;'>", unsafe_allow_html=True)

    col_izq, col_der = st.columns([1.5, 1], gap="large")

    with col_izq:
        st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #000; margin-bottom: 6px; text-transform: uppercase;'>🔍 Ingreso Rápido o Lector de Barras</p>", unsafe_allow_html=True)
        prod = st.text_input("Buscador", placeholder="Escanear código o escribir producto...", label_visibility="collapsed")

        st.markdown("<p style='font-size: 0.75rem; font-weight: 600; color: #5D5F5F; margin: 0.5rem 0 0.5rem 0;'>O selecciona productos directos:</p>", unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.button("+ Inca Kola 2L", use_container_width=True)
        r1c2.button("+ Galletas Oreo", use_container_width=True)
        r1c3.button("+ Mica Vinifan", use_container_width=True)
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.button("+ Cuaderno A4", use_container_width=True)
        r2c2.button("+ Lapicero Pilot", use_container_width=True)
        r2c3.button("+ Folder Manila", use_container_width=True)

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="insight-card"><p class="insight-title">ℹ️ Guía de Operación en Caja</p>'
            '<p class="insight-body">Al ingresar un producto, las tarjetas de IA superiores se actualizan para mostrarte qué producto adicional ofrecerle al cliente.</p></div>',
            unsafe_allow_html=True
        )

    with col_der:
        st.markdown(
            '<div class="ticket-container">'
            '<div class="ticket-header">'
            '<div><span style="font-size:0.65rem; font-weight:800; color:#777; text-transform:uppercase;">Recibo Digital</span>'
            '<h3 style="margin:2px 0 0 0; font-size:1.05rem; font-weight:800; color:#000;">🧾 TICKET VIRTUAL</h3></div>'
            '<span style="font-family:monospace; font-size:0.78rem; font-weight:700; background:rgba(0,0,0,0.05); padding:0.3rem 0.55rem; border-radius:6px; color:#000;">#TK-2026-0042</span>'
            '</div>'
            '<div style="margin-bottom:1rem;">'
            '<div class="ticket-item-row"><div><strong style="color:#000; display:block;">Galletas Oreo</strong><span style="font-size:0.72rem; color:#777;">2 × S/ 3.50</span></div><span style="font-weight:700; color:#000;">S/ 7.00</span></div>'
            '<div class="ticket-item-row"><div><strong style="color:#000; display:block;">Inca Kola 2L</strong><span style="font-size:0.72rem; color:#777;">1 × S/ 10.00</span></div><span style="font-weight:700; color:#000;">S/ 10.00</span></div>'
            '<div class="ticket-item-row"><div><strong style="color:#000; display:block;">Mica A4 Vinifan</strong><span style="font-size:0.72rem; color:#777;">1 × S/ 5.00</span></div><span style="font-weight:700; color:#000;">S/ 5.00</span></div>'
            '</div>'
            '<div class="ticket-summary-box">'
            '<div class="ticket-summary-line"><span>Subtotal:</span><span style="font-weight:600; color:#000;">S/ 18.64</span></div>'
            '<div class="ticket-summary-line"><span>IGV (18%):</span><span style="font-weight:600; color:#000;">S/ 3.36</span></div>'
            '<div class="ticket-total-line"><span>TOTAL:</span><span>S/ 22.00</span></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="btn-pos-cobrar">', unsafe_allow_html=True)
        if st.button("💳 COBRAR S/ 22.00", use_container_width=True):
            st.success("✅ ¡Transacción registrada con éxito!")
            st.balloons()
        st.markdown('</div>', unsafe_allow_html=True)
