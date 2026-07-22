import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
from src.category_rules import (
    PENDING_CATEGORY,
    mine_category_rules,
    prepare_category_data,
)
from src.data_loader import load_detalle_ventas, load_inventario

st.set_page_config(
    page_title="SmartBazar — Dashboard Data Mining",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  2. IMPORTACIÓN DE MÓDULOS DESDE CAPA src/
# ═══════════════════════════════════════════════════════════════
from src.panel_eda import show_panel as show_eda
from src.panel_clustering import show_clustering_panel as show_clustering
from src.panel_asociacion import show_panel as show_asociacion
from src.panel_predictivo import show_panel as show_predictivo
from src.panel_forecast import show_panel as show_forecast
from src.panel_crud import show_panel as show_pos

# ═══════════════════════════════════════════════════════════════
#  3. CSS PREMIUM (RESTAURACIÓN DE ÍCONOS NATIVOS Y MATERIAL SYMBOLS)
# ═══════════════════════════════════════════════════════════════
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* Reset & Base Canvas */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background-color: #f1f5f9 !important;
    background-image:
        radial-gradient(at 10% 10%, rgba(255,255,255,0.8) 0px, transparent 50%),
        radial-gradient(at 90% 10%, rgba(241,245,249,0.7) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(226,232,240,0.6) 0px, transparent 50%) !important;
    font-family: 'Inter', sans-serif;
    color: #000000;
}

#MainMenu, footer { display: none !important; }

/* ── SIDEBAR (IMAGEN 2) ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 15px rgba(0,0,0,0.03) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
}

/* Ocultar barra de navegación por defecto si existiera */
[data-testid="stSidebarNav"] { display: none !important; }

/* Encabezado Principal */
.sidebar-header-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #000000 !important;
    letter-spacing: -0.03em !important;
    margin: 0 0 2px 0 !important;
    line-height: 1.1 !important;
}

.sidebar-header-sub {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #5D5F5F !important;
    margin: 0 0 1.2rem 0 !important;
}

/* Etiquetas de División por Fases */
.sidebar-phase-label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 800 !important;
    color: #777777 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 0.8rem 0 0.4rem 0.2rem !important;
}

/* ── CORRECCIÓN DE ERRORES EN RADIO BUTTONS DEL SIDEBAR ── */

/* 1. ELIMINAR OPCIÓN FANTASMA: Ocultar la etiqueta principal del widget stRadio en Sidebar */
[data-testid="stSidebar"] div[data-testid="stRadio"] > label:first-child {
    display: none !important;
}

/* 2. ELIMINAR CÍRCULO SELECTOR: Ocultar el dot/radio circle por completo en Sidebar */
[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] input {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
}

/* Contenedor del grupo de opciones en Sidebar */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.4rem !important;
    width: 100% !important;
}

/* Formato Base de las Opciones (Pill Shape de la Imagen 2) */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 0.68rem 1.1rem !important;
    border-radius: 999px !important; /* Total Pill Shape */
    background-color: transparent !important;
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: 1px solid transparent !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    width: 100% !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background-color: #f1f5f9 !important;
    color: #000000 !important;
    transform: translateX(3px) !important;
}

/* ESTADO ACTIVO: Píldora Negra Sólida con Texto Blanco (Imagen 2) */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22) !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p,
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) span {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* SVG ÍCONOS VECTORIALES PARA OPCIONES DEL MENÚ */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 12px;
    background-color: currentColor;
    flex-shrink: 0;
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: center;
    mask-position: center;
    -webkit-mask-size: contain;
    mask-size: contain;
}

/* Ícono 1: EDA / Bar Chart */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(1)::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Crect x='7' y='10' width='3' height='8' rx='1'/%3E%3Crect x='13' y='6' width='3' height='12' rx='1'/%3E%3Crect x='19' y='13' width='3' height='5' rx='1'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3v18h18'/%3E%3Crect x='7' y='10' width='3' height='8' rx='1'/%3E%3Crect x='13' y='6' width='3' height='12' rx='1'/%3E%3Crect x='19' y='13' width='3' height='5' rx='1'/%3E%3C/svg%3E");
}

/* Ícono 2: Clustering / Nodes */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(2)::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='5' r='3'/%3E%3Ccircle cx='5' cy='19' r='3'/%3E%3Ccircle cx='19' cy='19' r='3'/%3E%3Cline x1='8.5' y1='16.5' x2='10.5' y2='7.5'/%3E%3Cline x1='15.5' y1='16.5' x2='13.5' y2='7.5'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='5' r='3'/%3E%3Ccircle cx='5' cy='19' r='3'/%3E%3Ccircle cx='19' cy='19' r='3'/%3E%3Cline x1='8.5' y1='16.5' x2='10.5' y2='7.5'/%3E%3Cline x1='15.5' y1='16.5' x2='13.5' y2='7.5'/%3E%3C/svg%3E");
}

/* Ícono 3: Reglas de Asociación / Links */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(3)::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='18' cy='18' r='3'/%3E%3Ccircle cx='6' cy='6' r='3'/%3E%3Cpath d='M13 6h3a2 2 0 0 1 2 2v7'/%3E%3Cline x1='6' y1='9' x2='6' y2='21'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='18' cy='18' r='3'/%3E%3Ccircle cx='6' cy='6' r='3'/%3E%3Cpath d='M13 6h3a2 2 0 0 1 2 2v7'/%3E%3Cline x1='6' y1='9' x2='6' y2='21'/%3E%3C/svg%3E");
}

/* Separador y Cabecera Fase 2 antes del Ítem 4 (Clasificación) */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(4) {
    margin-top: 1.8rem !important;
    position: relative !important;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(4)::after {
    content: "FASE 2: MODELOS PREDICTIVOS";
    position: absolute;
    top: -1.3rem;
    left: 0.4rem;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 800 !important;
    color: #777777 !important;
    letter-spacing: 0.08em !important;
}

/* Ícono 4: Clasificación / Shapes */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(4)::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8.3 10a.7.7 0 0 1-.6-1.1l3.4-5.3a.7.7 0 0 1 1.2 0l3.4 5.3a.7.7 0 0 1-.6 1.1z'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Ccircle cx='17.5' cy='17.5' r='3.5'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M8.3 10a.7.7 0 0 1-.6-1.1l3.4-5.3a.7.7 0 0 1 1.2 0l3.4 5.3a.7.7 0 0 1-.6 1.1z'/%3E%3Crect x='3' y='14' width='7' height='7' rx='1'/%3E%3Ccircle cx='17.5' cy='17.5' r='3.5'/%3E%3C/svg%3E");
}

/* Ícono 5: Predicciones / Trending Line */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(5)::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='23 6 13.5 15.5 8.5 10.5 1 18'/%3E%3Cpolyline points='17 6 23 6 23 12'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='23 6 13.5 15.5 8.5 10.5 1 18'/%3E%3Cpolyline points='17 6 23 6 23 12'/%3E%3C/svg%3E");
}

/* Separador y Cabecera Fase 3 antes del Ítem 6 (POS Inteligente) */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(6) {
    margin-top: 1.8rem !important;
    position: relative !important;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(6)::after {
    content: "FASE 3: OPERACIÓN";
    position: absolute;
    top: -1.3rem;
    left: 0.4rem;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 800 !important;
    color: #777777 !important;
    letter-spacing: 0.08em !important;
}

/* Ícono 6: POS Inteligente / Store */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(6)::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='7' width='20' height='14' rx='2'/%3E%3Cpath d='M6 7V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v3'/%3E%3Cline x1='12' y1='12' x2='12' y2='16'/%3E%3Cline x1='10' y1='14' x2='14' y2='14'/%3E%3C/svg%3E");
    mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='2' y='7' width='20' height='14' rx='2'/%3E%3Cpath d='M6 7V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v3'/%3E%3Cline x1='12' y1='12' x2='12' y2='16'/%3E%3Cline x1='10' y1='14' x2='14' y2='14'/%3E%3C/svg%3E");
}

/* Layout del contenido principal */
.main .block-container {
    padding: 1.8rem 2.2rem 3rem 2.2rem !important;
    max-width: 100% !important;
}

/* ── APLICAR INTER A TEXTO PERO PRESERVAR ÍCONOS NATIVOS DE STREAMLIT ── */
h1, h2, h3, h4, p, td, th, label {
    font-family: 'Inter', sans-serif;
}

[data-testid="stIcon"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="stHeader"] *,
[data-baseweb="icon"],
[data-baseweb="icon"] *,
[data-testid="stExpander"] summary *,
[class*="material-symbols"],
[class*="material-icons"],
i {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}
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


@st.cache_data(show_spinner="Preparando categorías y reglas reales...")
def load_association_data():
    """Carga los CSV limpios y conserva la trazabilidad de la categoría analítica."""
    return prepare_category_data(load_detalle_ventas(), load_inventario())


def render_association_panel():
    st.markdown(
        """
        <style>
        /* ── KPI Cards (Local) ── */
        .kpi-card {
            background: rgba(255,255,255,0.65) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-radius: 14px !important;
            padding: 1.2rem 1.3rem !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
            border: 1px solid rgba(255,255,255,0.8) !important;
            min-height: 110px !important;
        }
        .kpi-title { font-size: 0.68rem !important; color: #777777 !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; margin-bottom: 0.6rem !important; display: block !important; }
        .kpi-value { font-size: 1.5rem !important; font-weight: 800 !important; color: #000000 !important; line-height: 1.2 !important; display: block !important; margin-bottom: 0.3rem !important; word-break: break-word !important; }
        .kpi-delta { font-size: 0.72rem !important; color: #5D5F5F !important; font-weight: 600 !important; display: block !important; line-height: 1.3 !important; }

        /* ── Insight Cards (Local) ── */
        .insight-card {
            background: rgba(255,255,255,0.7) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-radius: 14px !important;
            padding: 1rem 1.2rem !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.03) !important;
            margin-bottom: 0.8rem !important;
        }
        .insight-badge {
            font-size: 0.62rem !important; font-weight: 800 !important; background: #000000 !important; color: #ffffff !important;
            padding: 0.15rem 0.45rem !important; border-radius: 5px !important; text-transform: uppercase !important;
            letter-spacing: 0.04em !important; display: inline-block !important; margin-bottom: 0.4rem !important;
        }
        .insight-title { font-size: 0.88rem !important; font-weight: 800 !important; color: #000000 !important; margin: 0 0 0.3rem 0 !important; }
        .insight-body { font-size: 0.8rem !important; color: #5D5F5F !important; line-height: 1.5 !important; margin: 0 !important; }

        /* ── Controls Panel (Local) ── */
        .ctrl-panel {
            background: rgba(255,255,255,0.65) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-radius: 16px !important;
            padding: 1.2rem !important;
            border: 1px solid rgba(255,255,255,0.8) !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
            margin-bottom: 1rem !important;
        }
        .ctrl-title { font-size: 0.88rem !important; font-weight: 800 !important; color: #000000 !important; margin: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )
    section_header("Descubrimiento de Patrones (Apriori)", "Reglas reales por categoría analítica, con auditoría y trazabilidad del catálogo.")
    ventas_cat, auditoria = load_association_data()
    c1, c2, c3, c4 = st.columns([1.25, 1, 1, 1])
    with c1:
        ctrl_header("Umbrales de Búsqueda")
        min_supp = st.slider("Soporte Mínimo", 0.005, 0.10, 0.005, 0.005, key="asoc_support")
        min_conf = st.slider("Confianza Mínima", 0.05, 1.00, 0.40, 0.05, key="asoc_confidence")
        min_lift = st.slider("Lift Mínimo", 1.0, 10.0, 1.0, 0.1, key="asoc_lift")

    reglas = mine_category_rules(ventas_cat, min_supp)
    reglas_filt = reglas[(reglas["Confianza"] >= min_conf) & (reglas["Lift"] >= min_lift)].copy()
    top_lift = reglas_filt.iloc[0] if not reglas_filt.empty else None
    top_conf = reglas_filt.sort_values("Confianza", ascending=False).iloc[0] if not reglas_filt.empty else None
    pendientes = int((ventas_cat["Categoria_Analitica"] == PENDING_CATEGORY).sum())
    with c2: kpi("Reglas Activas", str(len(reglas_filt)), f"Lift ≥ {min_lift:.1f}")
    with c3: kpi("Mayor Lift", f"{top_lift['Lift']:.2f}" if top_lift is not None else "—", top_lift["Antecedente"][:22] if top_lift is not None else "Sin reglas")
    with c4: kpi("Cobertura", f"{1 - pendientes / len(ventas_cat):.1%}", f"{pendientes} líneas pendientes", alert=pendientes > 0)

    tab1, tab2, tab3, tab4 = st.tabs(["Dispersión", "Grafo de Red", "Tabla de Reglas", "Auditoría de Categorías"])
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        if not reglas_filt.empty:
            sc = ax.scatter(reglas_filt["Soporte"], reglas_filt["Confianza"], c=reglas_filt["Lift"], cmap="Greys", s=reglas_filt["Lift"] * 70, alpha=0.85, edgecolors="black", linewidth=1)
            plt.colorbar(sc, ax=ax, shrink=0.8, label="Lift")
            for _, rule in reglas_filt.head(6).iterrows():
                ax.annotate(f"{rule['Antecedente'][:12]} → {rule['Consecuente'][:12]}", (rule["Soporte"], rule["Confianza"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        apply_chart_style(fig, ax, title="Reglas reales: confianza vs soporte", xlabel="Soporte", ylabel="Confianza")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with tab2:
        if reglas_filt.empty:
            st.info("No hay reglas que cumplan los umbrales seleccionados.")
        else:
            graph = nx.DiGraph()
            for _, rule in reglas_filt.head(15).iterrows():
                graph.add_edge(rule["Antecedente"], rule["Consecuente"], weight=rule["Lift"])
            fig, ax = plt.subplots(figsize=(10, 4.5))
            position = nx.spring_layout(graph, seed=42)
            nx.draw_networkx_nodes(graph, position, node_color="#0f172a", node_size=1050, ax=ax)
            nx.draw_networkx_labels(graph, position, font_size=7, font_color="white", ax=ax)
            nx.draw_networkx_edges(graph, position, width=[graph[a][b]["weight"] for a, b in graph.edges], edge_color="#64748b", arrows=True, arrowsize=14, ax=ax)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with tab3:
        if reglas_filt.empty:
            st.info("No hay reglas que cumplan los umbrales seleccionados. Ajusta los controles.")
        else:
            st.dataframe(reglas_filt.style.format({"Soporte": "{:.2%}", "Confianza": "{:.2%}", "Lift": "{:.2f}"}), use_container_width=True, hide_index=True)
    with tab4:
        st.caption("La categoría derivada por descripción tiene prioridad; la categoría de inventario normalizada se usa como respaldo.")
        st.dataframe(auditoria.sort_values(["Accion", "Productos"], ascending=[True, False]), use_container_width=True, hide_index=True)
        pendientes_df = ventas_cat.loc[ventas_cat["Categoria_Analitica"] == PENDING_CATEGORY, ["ID_Producto", "Descripcion", "Categoria"]].drop_duplicates()
        if not pendientes_df.empty:
            st.warning("Productos que requieren revisión manual")
            st.dataframe(pendientes_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  4. SIDEBAR REFACTORIZADO (FIEL A IMAGEN DE REFERENCIA 2)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 0.8rem;">
            <h1 class="sidebar-header-title">SmartBazar</h1>
            <p class="sidebar-header-sub">Data Mining Dashboard</p>
        </div>
        <p class="sidebar-phase-label">FASE 1: ANÁLISIS DE DATOS</p>
        """,
        unsafe_allow_html=True
    )

    opciones_menu = [
        "EDA",
        "Clustering",
        "Reglas de Asociación",
        "Clasificación",
        "Predicciones",
        "POS Inteligente",
    ]

    opcion_sel = st.radio(
        label="NavegacionSmartBazar",
        options=opciones_menu,
        label_visibility="collapsed"
    )

# ═══════════════════════════════════════════════════════════════
#  5. ENRUTAMIENTO MODULAR DE PÁGINAS
# ═══════════════════════════════════════════════════════════════
if opcion_sel == "EDA":
    show_eda()
elif opcion_sel == "Clustering":
    show_clustering()
elif opcion_sel == "Reglas de Asociación":
    render_association_panel()


# ═══════════════════════════════════════════════════════════════
#  4. CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════
elif opcion_sel == "Clasificación":
    show_predictivo()


# ═══════════════════════════════════════════════════════════════
#  5. PREDICCIONES (SERIES TEMPORALES)
# ═══════════════════════════════════════════════════════════════
elif opcion_sel == "Predicciones":
    show_forecast()
elif opcion_sel == "POS Inteligente":
    show_pos()
