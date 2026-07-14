import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from src.data_loader import load_detalle_ventas

def show_panel():
    st.header("🛒 Panel 1 (Adicional): Fábrica de Combos y Reglas de Asociación")
    st.write(
        """
        Este panel utiliza el algoritmo **Apriori** sobre el detalle de ventas para encontrar qué productos se compran 
        juntos con frecuencia. Esto permite sugerir combos promocionales fundamentados en datos reales del negocio.
        """
    )
    
    with st.spinner("Cargando detalles de ventas..."):
        try:
            detalle = load_detalle_ventas()
        except Exception as e:
            st.error(f"Error al cargar los datos: {e}")
            return
            
    # Cleaning description column
    detalle = detalle.dropna(subset=['Descripcion'])
    # Remove some generic descriptions like "FOTOCOPIA" if they are too dominant or keep them
    # Let's keep them, but let's offer a checkbox to exclude common photocopy items to see other product combos
    exclude_copias = st.checkbox("Excluir Fotocopias e Impresiones del análisis (para ver combos de útiles)")
    
    if exclude_copias:
        detalle = detalle[~detalle['Descripcion'].str.contains('FOTOCOPIA|IMPRESION|IMPRESIÓN', case=False, na=False)]
        
    # Prepare transactional data (basket)
    # Group by ticket and item description
    basket = (detalle.groupby(['ID_Venta', 'Descripcion'])['Cantidad']
              .sum().unstack().reset_index().fillna(0)
              .set_index('ID_Venta'))
              
    # Convert to 1s and 0s
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0)
    
    st.write(f"**Matriz de Transacciones:** {basket_sets.shape[0]} tickets y {basket_sets.shape[1]} productos distintos.")
    
    # Sliders for parameters modification in vivo (very useful for the exam!)
    st.markdown("### 🚨 Ajuste de Reglas en Vivo")
    col1, col2, col3 = st.columns(3)
    min_supp = col1.slider("Soporte Mínimo (Frecuencia relativa del combo):", min_value=0.005, max_value=0.10, value=0.015, step=0.005, format="%.3f")
    min_conf = col2.slider("Confianza Mínima (Probabilidad condicional):", min_value=0.05, max_value=1.0, value=0.20, step=0.05)
    min_lift = col3.slider("Lift Mínimo (Factor de asociación aleatoria):", min_value=0.5, max_value=10.0, value=1.2, step=0.1)
    
    # Calculate Apriori
    try:
        frequent_itemsets = apriori(basket_sets, min_support=min_supp, use_colnames=True)
        if len(frequent_itemsets) == 0:
            st.warning("No se encontraron conjuntos de productos frecuentes con el soporte mínimo seleccionado. Prueba reduciendo el Soporte Mínimo.")
            return
            
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        
        # Filter by confidence
        rules = rules[rules['confidence'] >= min_conf]
        
    except Exception as e:
        st.error(f"Error al calcular reglas de asociación: {e}")
        return
        
    if len(rules) == 0:
        st.warning("No se encontraron reglas de asociación con los umbrales seleccionados. Prueba reduciendo el Soporte, la Confianza o el Lift.")
        return
        
    st.success(f"¡Se han encontrado **{len(rules)} reglas de asociación** válidas!")
    
    # Format rules for presentation
    # Convert antecedents and consequents from frozenset to list/string
    rules_df = rules.copy()
    rules_df['antecedents_str'] = rules_df['antecedents'].apply(lambda x: ", ".join(list(x)))
    rules_df['consequents_str'] = rules_df['consequents'].apply(lambda x: ", ".join(list(x)))
    
    # Show table
    st.subheader("Tabla de Reglas Generadas")
    display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage', 'conviction']
    formatted_table = rules_df[display_cols].rename(columns={
        'antecedents_str': 'Si compran (Antecedente)',
        'consequents_str': 'También compran (Consecuente)',
        'support': 'Soporte',
        'confidence': 'Confianza',
        'lift': 'Lift',
        'leverage': 'Leverage',
        'conviction': 'Convicción'
    })
    
    st.dataframe(formatted_table.style.format({
        'Soporte': '{:.2%}',
        'Confianza': '{:.2%}',
        'Lift': '{:.2f}',
        'Leverage': '{:.4f}',
        'Convicción': '{:.2f}'
    }))
    
    # Recommendations of Combos (Actionable insights)
    st.subheader("💡 Combos de Venta Recomendados")
    st.write("Basado en el análisis de Lift y Confianza, sugerimos los siguientes combos para el POS:")
    
    top_rules = rules_df.sort_values(by='lift', ascending=False).head(5)
    for idx, row in top_rules.iterrows():
        st.info(
            f"🎁 **Combo sugerido:** Al vender **{row['antecedents_str']}**, recomendar proactivamente **{row['consequents_str']}**.\n\n"
            f"* **Soporte:** {row['support']:.2%} de los tickets de venta incluyen ambos productos.\n"
            f"* **Confianza:** Si un cliente compra el antecedente, tiene un **{row['confidence']:.1%}** de probabilidad de llevar el consecuente.\n"
            f"* **Fuerza (Lift):** La probabilidad de compra conjunta es **{row['lift']:.2f} veces mayor** que si se compraran de forma independiente."
        )
        
    # Visualizing Rules
    st.subheader("Visualización de Reglas (Soporte vs Confianza)")
    fig_rules, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        rules_df['support'], 
        rules_df['confidence'], 
        c=rules_df['lift'], 
        cmap='plasma', 
        s=rules_df['lift'] * 50, 
        alpha=0.75
    )
    fig_rules.colorbar(scatter, label='Lift')
    ax.set_title("Mapa de Reglas de Asociación")
    ax.set_xlabel("Soporte (Frecuencia)")
    ax.set_ylabel("Confianza (Probabilidad)")
    ax.grid(True)
    st.pyplot(fig_rules)
