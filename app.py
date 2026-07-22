import streamlit as st

# ═══════════════════════════════════════════════════════════════
#  1. CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════
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
    show_asociacion()
elif opcion_sel == "Clasificación":
    show_predictivo()
elif opcion_sel == "Predicciones":
    show_forecast()
elif opcion_sel == "POS Inteligente":
    show_pos()
