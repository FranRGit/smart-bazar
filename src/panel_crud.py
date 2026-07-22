"""
Panel 6 – Terminal POS (Punto de Venta Inteligente)
Replicación fiel de la interfaz 'Smart Bazar - POS (Modo Claro)' de Stitch
Asistente IA Integrado + CRUD completo en SQLite (Sin librerías pesadas de ML)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import textwrap
from datetime import datetime
from src.panel_predictivo import cargar_modelo


# ── Database helpers ──────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect('db_consultas.sqlite')
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
    conn = sqlite3.connect('db_consultas.sqlite')
    cursor = conn.cursor()
    try:
        cursor.execute(query, params)
        if commit:
            conn.commit()
            result = cursor.lastrowid
        else:
            result = cursor.fetchall()
            colnames = [d[0] for d in cursor.description]
            result = (result, colnames)
    except Exception as e:
        conn.close()
        raise e
    conn.close()
    return result


# ── Mock predictors (Rule-based, NO ML) ──────────────────────────────────────

def mock_predict_payment(total, hora, departamento):
    """Predicción determinista del método de pago basada en reglas de negocio."""
    if departamento == 'FOTOCOPIADORA' or total < 8.0:
        return 'EFECTIVO', 0.73
    elif total > 30.0 or hora >= 18:
        return 'YAPE', 0.84
    else:
        return 'YAPE', 0.68


def predict_payment_real(datos_modelo, total, hora, departamento):
    """Predicción en tiempo real usando el modelo XGBoost/RF real."""
    if datos_modelo is None:
        # Fallback en caso de que no cargue el modelo
        return "EFECTIVO", 0.50
        
    modelo = datos_modelo["modelo"]
    le_depto = datos_modelo["label_encoder_departamento"]
    columnas = datos_modelo["columnas"]
    
    # 1. Preprocesar las variables de entrada del POS
    # Obtener el día actual (para simular el día de la semana)
    dia_idx = datetime.now().weekday() # 0 = Lunes, 6 = Domingo
    
    # Codificar departamento
    try:
        depto_enc = le_depto.transform([departamento])[0]
    except Exception:
        depto_enc = 0 # Fallback
        
    pct_foto = 1.0 if departamento == "FOTOCOPIADORA" else 0.0
    
    # 2. Crear fila estructurada exactamente como la espera el modelo
    fila = pd.DataFrame([{
        "Total": total,
        "n_items": 2, # Valor promedio aproximado para POS
        "n_productos_distintos": 1, 
        "departamento_principal_enc": depto_enc,
        "pct_fotocopiadora": pct_foto,
        "dia_semana": dia_idx,
        "es_fin_de_semana": int(dia_idx in [5, 6]),
    }])[columnas]
    
    # 3. Realizar inferencia real
    pred = modelo.predict(fila)[0]
    prob = modelo.predict_proba(fila)[0, 1]
    
    metodo = "YAPE" if pred == 1 else "EFECTIVO"
    # La confianza es la probabilidad asociada a la clase ganadora
    confianza = prob if pred == 1 else (1 - prob)
    
    return metodo, confianza


def mock_recommend_combo(departamento, total):
    """Recomendación de combo basada en catálogo de reglas de asociación Apriori."""
    combos = {
        'FOTOCOPIADORA': 'MICA A4 VINIFAN (Lift: 3.21)',
        'UTILES': 'LAPICERO PILOT (Lift: 2.89)' if total <= 15 else 'CAJA COLORES FABER CASTELL (Lift: 1.80)',
        'GOLOSINAS': 'GALLETAS OREO (Lift: 1.95)',
        'BEBIDAS': 'GALLETAS SODA (Lift: 1.72)',
        'SERVICIOS': 'FOLDER MANILA A4 (Lift: 2.54)',
    }
    return combos.get(departamento, 'MICA A4 VINIFAN (Lift: 2.10)')


# ── Catálogo de productos rápidos para POS ───────────────────────────────────
PRODUCT_CATALOG = {
    'Inca Kola 2L': {'depto': 'BEBIDAS', 'precio': 10.00},
    'Cuaderno A4 College': {'depto': 'UTILES', 'precio': 8.50},
    'Mica A4 Vinifan (10 und)': {'depto': 'FOTOCOPIADORA', 'precio': 5.00},
    'Lapicero Pilot G2': {'depto': 'UTILES', 'precio': 6.00},
    'Impresión B/N (20 copias)': {'depto': 'FOTOCOPIADORA', 'precio': 4.00},
    'Galletas Oreo / Soda': {'depto': 'GOLOSINAS', 'precio': 3.50},
    'Folder Manila A4 (Paquete)': {'depto': 'SERVICIOS', 'precio': 7.00},
    'Entrada Libre / Monto Personalizado': {'depto': 'UTILES', 'precio': 12.50},
}


# ── Panel principal ──────────────────────────────────────────────────────────

def show_panel():
    init_db()

    # Cargar modelo de clasificación real para predicción de pago
    try:
        datos_modelo = cargar_modelo()
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo real ({e}). Usando simulador de contingencia.")
        datos_modelo = None

    # ── Pestañas (Stitch Tabs) ──────────────────────────────────────────────
    tab_pos, tab_crud = st.tabs(
        ['🏪 Terminal POS (Modo Claro)', '📋 Historial de Transacciones (CRUD)']
    )

    # ════════════════════════════════════════════════════════════════════════
    #  TAB 1: TERMINAL POS (Fiel a interfaz Stitch)
    # ════════════════════════════════════════════════════════════════════════
    with tab_pos:
        # Header Section de POS
        st.markdown(
            textwrap.dedent(
                """
                <div style="background: #f7f3f2; border: 1px solid rgba(0,0,0,0.06); border-radius: 18px; padding: 1rem 1.6rem; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <h3 style="margin:0; font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #000000;">Punto de Venta Inteligente</h3>
                    <p style="margin: 2px 0 0 0; font-size: 0.82rem; color: #747878;">Terminal Activa — Caja 01 · Asistente Apriori & K-Means Online</p>
                  </div>
                  <div style="display: flex; gap: 10px; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 6px; background: #ffffff; padding: 0.4rem 1rem; border-radius: 9999px; border: 1px solid rgba(0,0,0,0.06);">
                      <span style="font-size: 0.9rem;">👤</span>
                      <span style="font-size: 0.82rem; font-weight: 600; color: #1c1b1b;">Cajero 01</span>
                    </div>
                  </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

        # Top Section: 60/40 Split (Span 7 vs Span 5)
        col_left, col_right = st.columns([7, 5])

        with col_left:
            # Bento Card: Large Search / Product Selector
            st.markdown(
                textwrap.dedent(
                    """
                    <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.06); border-radius: 20px; padding: 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.02);">
                      <p style="font-size: 0.80rem; font-weight: 700; color: #747878; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 0.8rem 0;">
                        🔍 Escanear Código o Seleccionar Producto del Catálogo
                      </p>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

            producto_sel = st.selectbox(
                "Producto rápido / Escaneo de Código de Barras:",
                options=list(PRODUCT_CATALOG.keys()),
                label_visibility="collapsed"
            )

            item_info = PRODUCT_CATALOG[producto_sel]
            depto_default = item_info['depto']
            precio_default = item_info['precio']

            # Formulario de parámetros de transacción en vivo
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1:
                id_venta_input = st.text_input(
                    "Código Venta:",
                    value=f"VTA-{np.random.randint(100000, 999999)}",
                )
            with c_p2:
                id_cliente_input = st.number_input(
                    "ID Cliente / Tarjeta:", min_value=1, value=np.random.randint(1, 150), step=1
                )
            with c_p3:
                total_input = st.number_input(
                    "Monto Compra (S/):", min_value=0.1, value=float(precio_default), step=0.5
                )

            c_p4, c_p5 = st.columns(2)
            with c_p4:
                depto_list = ['UTILES', 'FOTOCOPIADORA', 'GOLOSINAS', 'BEBIDAS', 'SERVICIOS']
                depto_idx = depto_list.index(depto_default) if depto_default in depto_list else 0
                depto_input = st.selectbox(
                    "Departamento:", options=depto_list, index=depto_idx
                )
            with c_p5:
                hora_now = datetime.now().hour
                hora_input = st.slider(
                    "Hora del Día (8 - 22h):",
                    min_value=8,
                    max_value=22,
                    value=hora_now if 8 <= hora_now <= 22 else 14,
                )

            # Predicciones inteligentes reales en tiempo real
            pred_pago, pred_conf = predict_payment_real(datos_modelo, total_input, hora_input, depto_input)
            sugerencia_combo = mock_recommend_combo(depto_input, total_input)

            # Recent Scans & Quick Actions (Grid 2 cols)
            st.write("")
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div style="background: #f7f3f2; border: 1px solid rgba(0,0,0,0.05); border-radius: 16px; padding: 1.1rem; display: flex; align-items: center; gap: 14px;">
                          <div style="width: 44px; height: 44px; border-radius: 12px; background: #ebe7e6; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">🕒</div>
                          <div>
                            <p style="font-size: 0.72rem; color: #747878; font-weight: 700; text-transform: uppercase; margin: 0;">Último Escaneo</p>
                            <p style="font-size: 1rem; font-weight: 700; color: #000000; margin: 2px 0 0 0;">{producto_sel}</p>
                          </div>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )
            with col_q2:
                st.markdown(
                    textwrap.dedent(
                        """
                        <div style="background: #fff5f5; border: 1px solid #ffdad6; border-radius: 16px; padding: 1.1rem; display: flex; align-items: center; gap: 14px;">
                          <div style="width: 44px; height: 44px; border-radius: 12px; background: #ffdad6; display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">✖</div>
                          <div>
                            <p style="font-size: 0.72rem; color: #ba1a1a; font-weight: 700; text-transform: uppercase; margin: 0;">Acción Rápida</p>
                            <p style="font-size: 1rem; font-weight: 700; color: #ba1a1a; margin: 2px 0 0 0;">Anular / Limpiar Item</p>
                          </div>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )

            # Asistente IA Integrado (Motor de Recomendación en Vivo)
            st.write("")
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div style="background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 20px; padding: 1.4rem; box-shadow: 0 4px 20px rgba(0,0,0,0.02);">
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 0.82rem; font-weight: 700; color: #000000; text-transform: uppercase; letter-spacing: 0.05em;">🧠 Asistente IA en Vivo (Reglas Apriori)</span>
                        <span style="background: #dcfce7; color: #15803d; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.7rem; border-radius: 9999px;">Confianza: {pred_conf:.0%}</span>
                      </div>
                      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div style="background: #f7f3f2; padding: 1rem; border-radius: 14px;">
                          <p style="font-size: 0.72rem; color: #747878; font-weight: 600; text-transform: uppercase; margin: 0;">Método de Pago Previsto</p>
                          <p style="font-size: 1.15rem; font-weight: 800; color: #8b5cf6; margin: 4px 0 0 0;">📱 {pred_pago}</p>
                        </div>
                        <div style="background: #f7f3f2; padding: 1rem; border-radius: 14px;">
                          <p style="font-size: 0.72rem; color: #747878; font-weight: 600; text-transform: uppercase; margin: 0;">Combo Sugerido para Upsell</p>
                          <p style="font-size: 0.95rem; font-weight: 700; color: #10b981; margin: 4px 0 0 0;">💡 {sugerencia_combo}</p>
                        </div>
                      </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        with col_right:
            # Bento Card Dark (Ticket Virtual #TK-2023-8902 - Fiel a Stitch)
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div style="background: #1c1b1b; color: #ffffff; border-radius: 24px; padding: 2rem 1.8rem; box-shadow: 0 16px 36px rgba(0,0,0,0.22); min-height: 520px; display: flex; flex-direction: column; justify-content: space-between;">
                      <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 1.2rem; margin-bottom: 1.5rem;">
                          <h3 style="margin: 0; color: #ffffff !important; font-family: 'Outfit', sans-serif; font-size: 1.35rem; display: flex; align-items: center; gap: 8px;">
                            <span>🧾</span> Ticket Virtual
                          </h3>
                          <span style="font-size: 0.82rem; color: rgba(255,255,255,0.5); font-family: monospace;">#{id_venta_input.strip()}</span>
                        </div>

                        <!-- Lista de Items -->
                        <div style="display: flex; flex-direction: column; gap: 16px; margin-bottom: 2rem;">
                          <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                              <p style="font-size: 1.05rem; font-weight: 600; color: #ffffff; margin: 0 0 3px 0;">{producto_sel}</p>
                              <p style="font-size: 0.80rem; color: rgba(255,255,255,0.6); margin: 0;">Depto: {depto_input} · 1 x S/ {total_input:.2f}</p>
                            </div>
                            <span style="font-size: 1.05rem; font-weight: 600; color: #ffffff;">S/ {total_input:.2f}</span>
                          </div>
                          
                          <div style="display: flex; justify-content: space-between; align-items: flex-start; opacity: 0.75;">
                            <div>
                              <p style="font-size: 0.95rem; font-weight: 500; color: #10b981; margin: 0 0 3px 0;">+ Sugerencia Apriori</p>
                              <p style="font-size: 0.78rem; color: rgba(255,255,255,0.6); margin: 0;">{sugerencia_combo.split(' (')[0]}</p>
                            </div>
                            <span style="font-size: 0.90rem; color: #10b981;">Upsell Sugerido</span>
                          </div>
                        </div>
                      </div>

                      <!-- Totales y Botón de Pago -->
                      <div>
                        <div style="border-top: 1px dashed rgba(255,255,255,0.18); padding-top: 1.2rem; display: flex; flex-direction: column; gap: 8px; margin-bottom: 1.6rem;">
                          <div style="display: flex; justify-content: space-between; font-size: 0.88rem; color: rgba(255,255,255,0.7);">
                            <span>Subtotal</span>
                            <span>S/ {(total_input / 1.18):.2f}</span>
                          </div>
                          <div style="display: flex; justify-content: space-between; font-size: 0.88rem; color: rgba(255,255,255,0.7);">
                            <span>IGV (18%)</span>
                            <span>S/ {(total_input - (total_input / 1.18)):.2f}</span>
                          </div>
                          <div style="display: flex; justify-content: space-between; font-size: 1.35rem; font-weight: 800; color: #ffffff; margin-top: 4px;">
                            <span>TOTAL A PAGAR</span>
                            <span style="color: #06b6d4;">S/ {total_input:.2f}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

            st.write("")
            cobrar_btn = st.button("💳 Confirmar Pago y Guardar Transacción", type="primary", use_container_width=True)
            if cobrar_btn:
                if not id_venta_input.strip():
                    st.error("Por favor ingresa o genera un Código de Venta válido.")
                else:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        run_query(
                            """
                            INSERT INTO consultas
                                (id_venta, id_cliente, total, hora, departamento,
                                 metodo_pago_predicho, recomendacion_combo, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                id_venta_input.strip().upper(),
                                id_cliente_input,
                                total_input,
                                hora_input,
                                depto_input,
                                pred_pago,
                                sugerencia_combo,
                                timestamp_str,
                            ),
                            commit=True,
                        )
                        st.balloons()
                        st.success(
                            f"✅ **Venta {id_venta_input.upper()}** por **S/ {total_input:.2f}** procesada con éxito y almacenada en SQLite (`db_consultas.sqlite`)."
                        )
                    except sqlite3.IntegrityError:
                        st.error(
                            f"❌ Error: El código de venta **{id_venta_input.upper()}** ya existe en la base de datos."
                        )
                    except Exception as e:
                        st.error(f"Error al guardar: {e}")

    # ════════════════════════════════════════════════════════════════════════
    #  TAB 2: HISTORIAL DE TRANSACCIONES (CRUD Completo)
    # ════════════════════════════════════════════════════════════════════════
    with tab_crud:
        st.subheader("📋 Registro de Transacciones Almacenadas en SQLite")
        st.write("Visualiza, edita o elimina las ventas registradas en el sistema en tiempo real.")

        try:
            db_data, colnames = run_query(
                "SELECT id, id_venta, id_cliente, total, hora, departamento, "
                "metodo_pago_predicho, recomendacion_combo, timestamp "
                "FROM consultas ORDER BY id DESC"
            )
            df_db = pd.DataFrame(db_data, columns=colnames)
        except Exception as e:
            st.error(f"Error al leer base de datos: {e}")
            df_db = pd.DataFrame()

        if df_db.empty:
            st.info(
                "📭 No hay transacciones registradas en el historial todavía. "
                "Registra una venta en la pestaña **Terminal POS** para comenzar."
            )
        else:
            # Estadísticas rápidas del historial
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Total Ventas", f"{len(df_db)} transacciones")
            col_s2.metric("Ingreso Acumulado", f"S/ {df_db['total'].sum():,.2f}")
            col_s3.metric("Ticket Promedio", f"S/ {df_db['total'].mean():,.2f}")
            pago_yape_pct = (df_db['metodo_pago_predicho'] == 'YAPE').mean() * 100
            col_s4.metric("Adopción Yape", f"{pago_yape_pct:.1f}%")

            st.write("---")
            st.dataframe(
                df_db.rename(
                    columns={
                        'id': 'ID Reg',
                        'id_venta': 'Código Venta',
                        'id_cliente': 'ID Cliente',
                        'total': 'Monto (S/)',
                        'hora': 'Hora',
                        'departamento': 'Dpto Principal',
                        'metodo_pago_predicho': 'Método Pago (Pred.)',
                        'recomendacion_combo': 'Combo Sugerido Apriori',
                        'timestamp': 'Fecha y Hora Reg',
                    }
                ),
                use_container_width=True,
            )

            # Acciones de edición y eliminación (CRUD)
            st.write("---")
            st.subheader("⚙️ Acciones de Gestión CRUD: Editar / Eliminar")

            col_act1, col_act2 = st.columns(2)

            with col_act1:
                st.markdown(
                    textwrap.dedent(
                        """
                        <div style="background: #fff5f5; border: 1px solid #ffdad6; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem;">
                          <h4 style="margin:0; color: #ba1a1a;">🗑️ Eliminar Registro de Venta</h4>
                          <p style="font-size: 0.82rem; color: #747878; margin: 4px 0 0 0;">Esta acción es irreversible y removerá la venta de la base de datos.</p>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )
                venta_delete = st.selectbox(
                    "Seleccione Código de Venta a eliminar:",
                    options=df_db['id_venta'].tolist(),
                    key='delete_box',
                )
                if st.button("❌ Confirmar Eliminación"):
                    try:
                        run_query(
                            "DELETE FROM consultas WHERE id_venta = ?",
                            (venta_delete,),
                            commit=True,
                        )
                        st.success(f"✅ Venta **{venta_delete}** eliminada correctamente de SQLite.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al eliminar: {e}")

            with col_act2:
                st.markdown(
                    textwrap.dedent(
                        """
                        <div style="background: #fdf8f8; border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem;">
                          <h4 style="margin:0; color: #000000;">✏️ Modificar / Editar Registro</h4>
                          <p style="font-size: 0.82rem; color: #747878; margin: 4px 0 0 0;">Actualiza el monto, cliente o departamento de una venta existente.</p>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )
                venta_edit = st.selectbox(
                    "Seleccione Código de Venta a modificar:",
                    options=df_db['id_venta'].tolist(),
                    key='edit_box',
                )

                row_to_edit = df_db[df_db['id_venta'] == venta_edit].iloc[0]

                with st.form("edit_form"):
                    st.write(f"Modificando datos para venta: **{venta_edit}**")
                    new_cliente = st.number_input(
                        "ID Cliente:", min_value=1, value=int(row_to_edit['id_cliente'])
                    )
                    new_total = st.number_input(
                        "Total Compra (S/):", min_value=0.1, value=float(row_to_edit['total'])
                    )
                    new_hora = st.slider(
                        "Hora del Día:", min_value=8, max_value=22, value=int(row_to_edit['hora'])
                    )
                    depto_list = ['UTILES', 'FOTOCOPIADORA', 'GOLOSINAS', 'BEBIDAS', 'SERVICIOS']
                    new_depto = st.selectbox(
                        "Departamento:",
                        options=depto_list,
                        index=depto_list.index(row_to_edit['departamento']) if row_to_edit['departamento'] in depto_list else 0
                    )
                    new_pago = st.selectbox(
                        "Método Pago Predicho:",
                        options=["EFECTIVO", "YAPE"],
                        index=0 if row_to_edit['metodo_pago_predicho'] == "EFECTIVO" else 1
                    )

                    if st.form_submit_button("💾 Guardar Cambios en DB"):
                        try:
                            new_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            run_query(
                                """
                                UPDATE consultas
                                SET id_cliente = ?, total = ?, hora = ?,
                                    departamento = ?, metodo_pago_predicho = ?,
                                    timestamp = ?
                                WHERE id_venta = ?
                                """,
                                (new_cliente, new_total, new_hora, new_depto, new_pago, new_ts, venta_edit),
                                commit=True,
                            )
                            st.success(f"✅ Registro **{venta_edit}** actualizado exitosamente.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al actualizar: {e}")
