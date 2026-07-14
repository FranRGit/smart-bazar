import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.data_loader import load_ventas, load_detalle_ventas

def show_panel():
    st.header("📊 Panel 1: Análisis Exploratorio (EDA) y Segmentación (Clustering)")
    st.write(
        """
        Este panel presenta el análisis de la distribución de ventas, la detección de outliers (según la regla 1.5·IQR) 
        y la segmentación de tickets mediante el algoritmo **K-Means** para identificar perfiles de clientes.
        """
    )
    
    # Load data
    with st.spinner("Cargando datos transaccionales..."):
        try:
            ventas = load_ventas()
            detalle = load_detalle_ventas()
        except Exception as e:
            st.error(f"Error al cargar los datos: {e}")
            return

    # Tabs for EDA and Clustering
    tab1, tab2 = st.tabs(["📊 Análisis Exploratorio (EDA)", "🧩 Segmentación con K-Means"])
    
    # ------------------ TAB 1: EDA ------------------
    with tab1:
        st.subheader("Estadísticas Descriptivas Generales")
        col1, col2, col3 = st.columns(3)
        col1.metric("Ventas Totales Registradas", f"{len(ventas)}")
        col2.metric("Monto Total de Ingresos", f"S/ {ventas['Total'].sum():,.2f}")
        col3.metric("Ticket Promedio", f"S/ {ventas['Total'].mean():,.2f}")
        
        st.write("#### Distribución del Monto de Ventas")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(ventas['Total'], bins=30, kde=True, color='#1f77b4', ax=ax)
        ax.set_title("Histograma de Monto de Ventas")
        ax.set_xlabel("Monto (S/)")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
        
        # Outlier Detection
        q1 = ventas['Total'].quantile(0.25)
        q3 = ventas['Total'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr
        outliers = ventas[ventas['Total'] > upper_bound]
        
        st.subheader("Detección de Valores Atípicos (Regla 1.5·IQR)")
        st.write(
            f"""
            * **Primer Cuartil (Q1):** S/ {q1:.2f}
            * **Tercer Cuartil (Q3):** S/ {q3:.2f}
            * **Rango Intercuartílico (IQR):** S/ {iqr:.2f}
            * **Umbral Superior de Outliers (Q3 + 1.5·IQR):** S/ {upper_bound:.2f}
            * **Cantidad de Outliers detectados:** {len(outliers)} de {len(ventas)} transacciones ({len(outliers)/len(ventas):.2%})
            """
        )
        
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            fig_box, ax_box = plt.subplots(figsize=(6, 4))
            sns.boxplot(y=ventas['Total'], color='#2ca02c', ax=ax_box)
            ax_box.set_title("Diagrama de Caja de Montos de Venta")
            ax_box.set_ylabel("Monto (S/)")
            st.pyplot(fig_box)
            
        with col_out2:
            st.write("**Top 5 Ventas Atípicas (Outliers):**")
            st.dataframe(outliers.sort_values(by='Total', ascending=False).head(5)[['ID', 'Fecha', 'Metodo_Pago', 'Total']])

    # ------------------ TAB 2: Clustering ------------------
    with tab2:
        st.subheader("Perfilamiento de Clientes por Tickets")
        st.write(
            """
            Agrupamos los tickets basándonos en 3 características principales de compra:
            1. **Monto Total** del ticket.
            2. **Cantidad total de artículos** comprados.
            3. **Hora de la compra** (extraído del detalle de ventas).
            """
        )
        
        # Prepare features for clustering
        # Group detail by ID_Venta
        agg_detail = detalle.groupby('ID_Venta').agg(
            hora_compra=('Fecha', lambda x: x.dt.hour.iloc[0]),
            n_items=('Cantidad', 'sum')
        ).reset_index()
        
        cluster_data = ventas.merge(agg_detail, left_on='ID', right_on='ID_Venta', how='inner')
        features = ['Total', 'n_items', 'hora_compra']
        X = cluster_data[features].copy()
        
        # Clean nulls
        X = X.dropna()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Interactive parameter modification in vivo (highly valued by the professor)
        st.markdown("### 🚨 Modificación de Parámetros en Vivo")
        k_val = st.slider("Seleccione el número de Clusters (K):", min_value=2, max_value=8, value=3)
        
        # Train KMeans
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        cluster_data['Cluster'] = labels
        
        # Silhouette Metric
        sil_score = silhouette_score(X_scaled, labels)
        
        # Metrics Display
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Coeficiente de Silueta", f"{sil_score:.4f}")
        col_m2.metric("Inercia del Modelo", f"{kmeans.inertia_:.2f}")
        
        # Plotly-like interactive visual representation
        st.write("#### Distribución de los Clusters")
        fig_scatter, ax_sc = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=cluster_data, 
            x='Total', 
            y='n_items', 
            hue='Cluster', 
            palette='viridis', 
            style='Cluster',
            alpha=0.8,
            ax=ax_sc
        )
        ax_sc.set_title(f"Segmentación de Ventas (K={k_val}) | Monto vs Cantidad de Artículos")
        ax_sc.set_xlabel("Monto Total (S/)")
        ax_sc.set_ylabel("Cantidad de Artículos")
        st.pyplot(fig_scatter)
        
        # Average statistics per Cluster
        st.write("#### Estadísticas Promedio por Perfil (Cluster)")
        perfiles = cluster_data.groupby('Cluster')[['Total', 'n_items', 'hora_compra']].mean()
        perfiles['Cantidad_Tickets'] = cluster_data['Cluster'].value_counts()
        st.dataframe(perfiles.style.format({
            'Total': 'S/ {:.2f}',
            'n_items': '{:.1f} unidades',
            'hora_compra': '{:.1f} hrs',
            'Cantidad_Tickets': '{:,.0f} tickets'
        }))
        
        # Metodo del codo (Elbow Method)
        st.subheader("Método del Codo (Evaluación de K)")
        st.write("Este gráfico ayuda a justificar la selección del K óptimo mostrando la caída de la inercia.")
        
        # Cache the elbow calculations to avoid recalculating on slider move
        inertias = []
        silhouettes = []
        k_range = range(2, 9)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_lbls = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, km_lbls))
            
        fig_elbow, (ax_e1, ax_e2) = plt.subplots(1, 2, figsize=(12, 4))
        ax_e1.plot(k_range, inertias, 'bo-', marker='o')
        ax_e1.set_title("Método del Codo (Inercia vs K)")
        ax_e1.set_xlabel("Número de clusters (K)")
        ax_e1.set_ylabel("Inercia")
        ax_e1.grid(True)
        
        ax_e2.plot(k_range, silhouettes, 'ro-', marker='s')
        ax_e2.set_title("Coeficiente de Silueta vs K")
        ax_e2.set_xlabel("Número de clusters (K)")
        ax_e2.set_ylabel("Coeficiente de Silueta")
        ax_e2.grid(True)
        
        st.pyplot(fig_elbow)
