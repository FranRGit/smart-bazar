import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="SmartBazar - Dashboard Inteligente",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Apply font family */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
        color: #38bdf8;
    }
    
    /* Clean Main View Background */
    .main {
        background-color: #f8fafc;
    }
    
    /* Custom cards styled like glassmorphism */
    div.stMetric {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div.stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.08);
    }
    
    /* Stylized buttons */
    div.stButton > button {
        background-color: #0284c7;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #0369a1;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(1px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Sidebar Header
    st.sidebar.title("🏪 SmartBazar")
    st.sidebar.write("Inteligencia de Negocios para Caja")
    st.sidebar.markdown("---")
    
    # Sidebar Navigation Menu
    st.sidebar.subheader("Seleccione un Panel:")
    option = st.sidebar.radio(
        label="Navegación",
        options=[
            "📊 Panel 1A: EDA y Clustering",
            "🛒 Panel 1B: Reglas de Asociación",
            "🔮 Panel 2: Predicción de Pagos",
            "📈 Panel 3: Pronóstico de Ingresos",
            "🏪 Panel 4: Punto de Venta (CRUD)"
        ],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Integrantes:**
        * Grupo de Minería de Datos
        
        **Docente:**
        * Dr. José Alfredo Herrera Quispe
        
        *UNMSM - FISI (2026-I)*
        """
    )
    
    # Render selected panel
    if option == "📊 Panel 1A: EDA y Clustering":
        from src import panel_eda_clustering
        panel_eda_clustering.show_panel()
        
    elif option == "🛒 Panel 1B: Reglas de Asociación":
        from src import panel_asociacion
        panel_asociacion.show_panel()
        
    elif option == "🔮 Panel 2: Predicción de Pagos":
        from src import panel_predictivo
        panel_predictivo.show_panel()
        
    elif option == "📈 Panel 3: Pronóstico de Ingresos":
        from src import panel_forecast
        panel_forecast.show_panel()
        
    elif option == "🏪 Panel 4: Punto de Venta (CRUD)":
        from src import panel_crud
        panel_crud.show_panel()

if __name__ == "__main__":
    main()
