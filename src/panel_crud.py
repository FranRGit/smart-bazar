import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from src.data_loader import load_ventas, load_detalle_ventas
from src.panel_predictivo import preparar_datos_modelo
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize database
def init_db():
    conn = sqlite3.connect("db_consultas.sqlite")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_venta TEXT UNIQUE,
            id_cliente INTEGER,
            total REAL,
            hora INTEGER,
            departamento TEXT,
            metodo_pago_predicho TEXT,
            recomendacion_combo TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def run_query(query, params=(), commit=False):
    conn = sqlite3.connect("db_consultas.sqlite")
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if commit:
            conn.commit()
            result = cursor.lastrowid
        else:
            result = cursor.fetchall()
            # Get columns
            colnames = [d[0] for d in cursor.description]
            result = (result, colnames)
    except Exception as e:
        conn.close()
        raise e
    conn.close()
    return result

# Cache model training for POS quick predictions
@st.cache_resource
def get_pos_predictor():
    try:
        ventas = load_ventas()
        detalle = load_detalle_ventas()
        X_train, X_test, y_train, y_test, le_depto, features_cols = preparar_datos_modelo(ventas, detalle, test_size=0.2)
        
        # Train a robust Random Forest for quick POS predictions
        model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        return model, le_depto
    except Exception as e:
        return None, None

def show_panel():
    st.header("🛒 Panel 4: Punto de Venta Inteligente (CRUD Caja)")
    st.write(
        """
        Este panel simula una caja registradora moderna (POS). Al ingresar una nueva venta, el sistema:
        1. **Predice el método de pago** (YAPE o EFECTIVO) en tiempo real.
        2. **Sugiere un producto adicional** (combo) según reglas de asociación.
        3. **Guarda la transacción** en la base de datos (CRUD completo en SQLite).
        """
    )
    
    # Initialize DB
    init_db()
    
    # Get predictor
    model, le_depto = get_pos_predictor()
    
    # Tabs for CRUD
    tab_register, tab_view = st.tabs(["🆕 Registrar Venta (POS)", "📋 Historial de Transacciones (CRUD)"])
    
    # ------------------ TAB 1: REGISTRAR VENTA ------------------
    with tab_register:
        st.subheader("Ingreso de Nueva Transacción")
        
        col1, col2 = st.columns(2)
        
        with col1:
            id_venta_input = st.text_input("Código de Venta (ej. VTA-0000855):", placeholder="VTA-XXXXXXX")
            id_cliente_input = st.number_input("ID Cliente:", min_value=1, value=1, step=1)
            total_input = st.number_input("Total de la Compra (S/):", min_value=0.1, value=5.0, step=0.1)
            
        with col2:
            hora_now = datetime.now().hour
            hora_input = st.slider("Hora de Compra:", min_value=8, max_value=22, value=hora_now if 8 <= hora_now <= 22 else 12)
            depto_list = list(le_depto.classes_) if le_depto else ["UTILES", "FOTOCOPIADORA"]
            depto_input = st.selectbox("Departamento Principal de Compra:", options=depto_list)
            
        # Automatic predictions & recommendations
        pred_label = "EFECTIVO"
        recomendacion = "Sin recomendaciones específicas."
        
        # Calculate dynamic predictions on input values
        if model and le_depto:
            # Prepare single input row
            depto_enc = le_depto.transform([depto_input])[0]
            dia_idx = datetime.now().weekday()
            es_fds = 1 if dia_idx in [5, 6] else 0
            
            # Simple rule to estimate pct_fotocopiadora for POS
            pct_foto = 1.0 if depto_input == "FOTOCOPIADORA" else 0.0
            
            input_row = pd.DataFrame([{
                'Total': total_input,
                'hora_compra': hora_input,
                'n_items': 1,  # baseline
                'n_productos_distintos': 1,
                'departamento_principal_enc': depto_enc,
                'pct_fotocopiadora': pct_foto,
                'dia_semana': dia_idx,
                'es_fin_de_semana': es_fds
            }])
            
            # Make prediction
            pred_code = model.predict(input_row)[0]
            pred_label = "YAPE" if pred_code == 1 else "EFECTIVO"
            
            # Simple association rule simulator for POS based on catalog
            if depto_input == "FOTOCOPIADORA":
                recomendacion = "Ofrecer proactivamente: MICA A4 VINIFAN (Lift: 2.10)"
            elif "CUADERNO" in depto_input or depto_input == "UTILES":
                if total_input > 10.0:
                    recomendacion = "Ofrecer proactivamente: CAJA DE COLORES FABERCASTELL (Lift: 1.80)"
                else:
                    recomendacion = "Ofrecer proactivamente: LAPICERO SHARPIE (Lift: 1.45)"
        
        # Display Prediction Cards
        st.write("---")
        st.write("#### Asistente Inteligente de Caja:")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.info(f"📱 **Previsión de Caja:** El cliente probablemente pagará con: **{pred_label}**")
        with col_c2:
            st.success(f"💡 **Recomendación Combo:** {recomendacion}")
            
        if st.button("💾 Registrar y Guardar Venta"):
            if not id_venta_input.strip():
                st.error("El Código de Venta no puede estar vacío.")
            else:
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    run_query(
                        """
                        INSERT INTO consultas (id_venta, id_cliente, total, hora, departamento, metodo_pago_predicho, recomendacion_combo, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (id_venta_input.strip().upper(), id_cliente_input, total_input, hora_input, depto_input, pred_label, recomendacion, timestamp_str),
                        commit=True
                    )
                    st.success(f"Venta **{id_venta_input.upper()}** guardada exitosamente en la base de datos.")
                except sqlite3.IntegrityError:
                    st.error(f"Error: Ya existe una venta registrada con el código **{id_venta_input.upper()}**.")
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

    # ------------------ TAB 2: HISTORIAL (CRUD) ------------------
    with tab_view:
        st.subheader("Transacciones Guardadas en SQLite")
        
        # Load data
        try:
            db_data, colnames = run_query("SELECT id, id_venta, id_cliente, total, hora, departamento, metodo_pago_predicho, recomendacion_combo, timestamp FROM consultas ORDER BY id DESC")
            df_db = pd.DataFrame(db_data, columns=colnames)
        except Exception as e:
            st.error(f"Error al leer base de datos: {e}")
            df_db = pd.DataFrame()
            
        if df_db.empty:
            st.write("No hay transacciones registradas en el historial.")
        else:
            st.dataframe(df_db.rename(columns={
                'id': 'ID Reg',
                'id_venta': 'Código Venta',
                'id_cliente': 'ID Cliente',
                'total': 'Monto (S/)',
                'hora': 'Hora',
                'departamento': 'Dpto Principal',
                'metodo_pago_predicho': 'Método Pago (Pred.)',
                'recomendacion_combo': 'Combo Sugerido',
                'timestamp': 'Fecha y Hora Reg'
            }))
            
            # Actions: Edit and Delete
            st.write("---")
            st.subheader("Acciones de Caja: Editar / Eliminar")
            
            col_act1, col_act2 = st.columns(2)
            
            # Delete form
            with col_act1:
                st.write("**🗑️ Eliminar Venta**")
                venta_delete = st.selectbox("Seleccione Código de Venta a eliminar:", options=df_db['id_venta'].tolist(), key='delete_box')
                if st.button("❌ Confirmar Eliminación"):
                    try:
                        run_query("DELETE FROM consultas WHERE id_venta = ?", (venta_delete,), commit=True)
                        st.success(f"Venta **{venta_delete}** eliminada correctamente.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al eliminar: {e}")
                        
            # Edit form
            with col_act2:
                st.write("**✏️ Editar Venta**")
                venta_edit = st.selectbox("Seleccione Código de Venta a editar:", options=df_db['id_venta'].tolist(), key='edit_box')
                
                # Fetch selected row
                row_to_edit = df_db[df_db['id_venta'] == venta_edit].iloc[0]
                
                with st.form("edit_form"):
                    st.write(f"Editando venta: **{venta_edit}**")
                    new_cliente = st.number_input("ID Cliente:", min_value=1, value=int(row_to_edit['id_cliente']))
                    new_total = st.number_input("Total de la Compra (S/):", min_value=0.1, value=float(row_to_edit['total']))
                    new_hora = st.slider("Hora de Compra:", min_value=8, max_value=22, value=int(row_to_edit['hora']))
                    new_depto = st.selectbox("Departamento Principal:", options=depto_list, index=depto_list.index(row_to_edit['departamento']) if row_to_edit['departamento'] in depto_list else 0)
                    new_pago = st.selectbox("Método de Pago Predicho:", options=["EFECTIVO", "YAPE"], index=0 if row_to_edit['metodo_pago_predicho'] == "EFECTIVO" else 1)
                    
                    submit_edit = st.form_submit_button("💾 Guardar Cambios")
                    
                    if submit_edit:
                        try:
                            # Update timestamp too
                            new_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            run_query(
                                """
                                UPDATE consultas 
                                SET id_cliente = ?, total = ?, hora = ?, departamento = ?, metodo_pago_predicho = ?, timestamp = ?
                                WHERE id_venta = ?
                                """,
                                (new_cliente, new_total, new_hora, new_depto, new_pago, new_ts, venta_edit),
                                commit=True
                            )
                            st.success(f"Venta **{venta_edit}** actualizada correctamente.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al actualizar: {e}")
