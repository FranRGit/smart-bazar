"""
Panel de Clasificación — Predicción de Método de Pago (YAPE vs EFECTIVO)
Persona 2 del proyecto SmartBazar.

Este módulo es autocontenido: no depende de variables globales de app.py.
Se apoya en 3 archivos generados por el notebook Panel_2_Prediccion_Metodo_Pago:
  - models/modelo_metodo_pago.json          (metadatos + encoder de departamento)
  - models/modelo_metodo_pago_booster.json  (booster de XGBoost en formato nativo,
                                              solo existe si el modelo ganador es XGBoost)
  - models/resultados_panel4.json           (métricas, matrices de confusión, importancias)
"""

import os
import json
import base64
import hashlib
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────
# Rutas — ancladas a la ubicación real de este archivo, no al directorio
# desde el que se ejecute streamlit run (evita bugs de ruta relativa)
# ─────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../smart-bazar-main/src
_BASE_DIR = os.path.dirname(_THIS_DIR)                           # .../smart-bazar-main
_CARPETA_MODELOS = os.path.join(_BASE_DIR, "models")

_RUTA_MODELO_META = os.path.join(_CARPETA_MODELOS, "modelo_metodo_pago.json")
_RUTA_RESULTADOS = os.path.join(_CARPETA_MODELOS, "resultados_panel4.json")

_COLUMNAS_ESPERADAS = [
    "Total", "n_items", "n_productos_distintos",
    "departamento_principal_enc", "pct_fotocopiadora",
    "dia_semana", "es_fin_de_semana",
]

_DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ─────────────────────────────────────────────────────────────────────
# Estilo visual — mismas clases CSS que el resto del dashboard (glass)
# Se duplican aquí como funciones mínimas para que este módulo sea
# independiente de app.py (evita imports circulares).
# ─────────────────────────────────────────────────────────────────────
def _apply_chart_style(fig, ax, title="", xlabel="", ylabel=""):
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    ax.set_facecolor('#ffffff')
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


def _kpi(title, value, delta="", alert=False):
    val_color = "#ef4444" if alert else "#000000"
    border = "1px solid rgba(239,68,68,0.5)" if alert else "1px solid rgba(255,255,255,0.8)"
    st.markdown(
        f'<div class="kpi-card" style="border: {border};">'
        f'<span class="kpi-title">{title}</span>'
        f'<span class="kpi-value" style="color: {val_color};">{value}</span>'
        f'<span class="kpi-delta">{delta}</span></div>',
        unsafe_allow_html=True
    )


def _insight(title, content, badge="INSIGHT DE NEGOCIO"):
    st.markdown(
        f'<div class="insight-card"><span class="insight-badge">{badge}</span>'
        f'<p class="insight-title">{title}</p>'
        f'<p class="insight-body">{content}</p></div>',
        unsafe_allow_html=True
    )


def _ctrl_header(label):
    st.markdown(f'<div class="ctrl-panel"><p class="ctrl-title">⚙️ {label}</p></div>', unsafe_allow_html=True)


def _section_header(title, subtitle):
    st.markdown(
        f'<div style="margin-bottom: 1.5rem;">'
        f'<h1 style="font-size: 1.8rem; font-weight: 800; color: #000000; margin: 0 0 4px 0; letter-spacing: -0.02em;">{title}</h1>'
        f'<p style="font-size: 0.88rem; color: #5D5F5F; margin: 0;">{subtitle}</p></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────
# Carga de artefactos — con verificación de integridad y errores claros
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def _cargar_modelo():
    """
    Carga el modelo ganador (XGBoost o Random Forest) según lo que indique
    modelo_metodo_pago.json. Para XGBoost usa el formato NATIVO (load_model),
    nunca pickle, para evitar el error 'input stream corrupted' por
    incompatibilidad de versiones entre Colab y el entorno local.
    """
    if not os.path.exists(_RUTA_MODELO_META):
        raise FileNotFoundError(f"No existe '{_RUTA_MODELO_META}'.")

    with open(_RUTA_MODELO_META, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    le_depto = LabelEncoder()
    le_depto.classes_ = np.array(metadata["departamentos_clases"])

    if metadata["tipo_modelo"] == "XGBoost":
        import xgboost as xgb

        ruta_booster = os.path.join(_CARPETA_MODELOS, metadata["archivo_booster"])
        if not os.path.exists(ruta_booster):
            raise FileNotFoundError(f"No existe el booster: '{ruta_booster}'.")

        modelo = xgb.XGBClassifier()
        modelo.load_model(ruta_booster)  # carga nativa: nunca pasa por pickle

        version_actual = xgb.__version__
        version_entrenamiento = metadata.get("xgboost_version")
        if version_actual != version_entrenamiento:
            st.warning(
                f"⚠️ El modelo se entrenó con xgboost {version_entrenamiento}, "
                f"este entorno tiene {version_actual}. El formato nativo debería "
                f"seguir funcionando; si notas algo raro, iguala versiones con: "
                f"`pip install xgboost=={version_entrenamiento}`"
            )
    else:
        payload_binario = base64.b64decode(metadata["payload_base64"])
        checksum_calculado = hashlib.sha256(payload_binario).hexdigest()
        if checksum_calculado != metadata["checksum_sha256"]:
            raise ValueError(
                "El modelo Random Forest llegó corrupto (el checksum no coincide "
                "con el generado en Colab). Vuelve a descargarlo desde Drive."
            )
        modelo = pickle.loads(payload_binario)

    return {
        "modelo": modelo,
        "label_encoder_departamento": le_depto,
        "columnas": metadata["columnas"],
        "tipo_modelo": metadata["tipo_modelo"],
    }


@st.cache_data
def _cargar_resultados():
    """Carga métricas, matrices de confusión e importancias, con verificación de checksum."""
    if not os.path.exists(_RUTA_RESULTADOS):
        raise FileNotFoundError(f"No existe '{_RUTA_RESULTADOS}'.")

    with open(_RUTA_RESULTADOS, "r", encoding="utf-8") as f:
        contenido = json.load(f)

    if "checksum_sha256" in contenido:
        checksum_guardado = contenido["checksum_sha256"]
        sin_checksum = {k: v for k, v in contenido.items() if k != "checksum_sha256"}
        checksum_calculado = hashlib.sha256(
            json.dumps(sin_checksum, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if checksum_calculado != checksum_guardado:
            st.warning(
                "⚠️ 'resultados_panel4.json' podría estar corrupto o fue editado "
                "manualmente (el checksum no coincide). Los números mostrados podrían no ser confiables."
            )

    return contenido


# ─────────────────────────────────────────────────────────────────────
# Función principal — llamada desde app.py
# ─────────────────────────────────────────────────────────────────────
def render():
    _section_header(
        "Clasificación Predictiva de Método de Pago",
        "Evaluación comparativa de Random Forest vs XGBoost y explicabilidad con SHAP."
    )

    try:
        datos_modelo = _cargar_modelo()
        resultados = _cargar_resultados()
    except FileNotFoundError as e:
        st.error(
            f"⚠️ Archivo no encontrado: {e}\n\n"
            f"Verifica que estos 3 archivos estén en `{_CARPETA_MODELOS}`:\n"
            f"- modelo_metodo_pago.json\n- modelo_metodo_pago_booster.json (si el modelo es XGBoost)\n"
            f"- resultados_panel4.json"
        )
        return
    except ValueError as e:
        st.error(f"⚠️ {e}")
        return

    if datos_modelo["columnas"] != _COLUMNAS_ESPERADAS:
        st.warning(
            f"⚠️ Las columnas del modelo cargado no coinciden con las esperadas.\n\n"
            f"Esperadas: {_COLUMNAS_ESPERADAS}\n\nEncontradas: {datos_modelo['columnas']}\n\n"
            f"El modelo probablemente quedó desactualizado — vuelve a correr el notebook."
        )

    # ── Selector + KPIs ─────────────────────────────────────────────
    c0, c1, c2, c3 = st.columns([1.2, 1, 1, 1])
    with c0:
        _ctrl_header("Selector de Algoritmo")
        mod_sel = st.selectbox("Modelo:", ["XGBoost", "Random Forest"])

    metricas_modelo = resultados["metricas"][mod_sel]
    cm = np.array(resultados["matrices_confusion"][mod_sel])

    with c1: _kpi("F1-Score", f"{metricas_modelo['f1']:.3f}", "class_weight = 'balanced'")
    with c2: _kpi("Accuracy", f"{metricas_modelo['accuracy']:.1%}", "Tasa de aciertos en test")
    with c3: _kpi("ROC-AUC", f"{metricas_modelo['roc_auc']:.3f}", "Capacidad de discriminación")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧮 Matriz de Confusión", "💡 Importancia SHAP", "🧪 Inferencia en Vivo"])

    # ── Tab 1: Matriz de confusión ───────────────────────────────────
    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greys", cbar=False,
            xticklabels=["Pred: EFECTIVO", "Pred: YAPE"],
            yticklabels=["Real: EFECTIVO", "Real: YAPE"],
            annot_kws={"size": 16, "weight": "bold"}, ax=ax, linewidths=2, linecolor='white'
        )
        _apply_chart_style(fig, ax, title=f"Matriz de Confusión — {mod_sel}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Tab 2: Importancia de variables ──────────────────────────────
    with tab2:
        importancias_modelo = resultados["importancias"][mod_sel]
        items = sorted(importancias_modelo.items(), key=lambda x: x[1])
        features = [k for k, v in items]
        importancias = [v for k, v in items]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.barh(range(len(features)), importancias, color='#0f172a', edgecolor='white', height=0.5)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontweight='bold', fontsize=9, color='#000000')
        for bar, v in zip(bars, importancias):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2, f'{v:.3f}',
                     va='center', fontsize=8, fontweight='bold', color='#000000')
        _apply_chart_style(fig, ax, title=f"Importancia Global de Variables — {mod_sel}", xlabel="Impacto en el modelo")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Tab 3: Inferencia en vivo (usa SIEMPRE el modelo realmente cargado) ──
    with tab3:
        st.markdown("Simula una compra para predecir el método de pago con el modelo real.")

        if datos_modelo["tipo_modelo"] != mod_sel:
            st.info(
                f"ℹ️ El modelo cargado en memoria es **{datos_modelo['tipo_modelo']}** "
                f"(el que se guardó como ganador). La inferencia en vivo usa ese modelo, "
                f"independientemente de cuál elijas arriba para ver métricas/gráficos."
            )

        modelo = datos_modelo["modelo"]
        le_depto = datos_modelo["label_encoder_departamento"]
        columnas = datos_modelo["columnas"]

        colf1, colf2 = st.columns(2)
        with colf1:
            total_inf = st.number_input("Total (S/)", min_value=0.1, max_value=250.0, value=15.0, step=0.5)
            n_items_inf = st.number_input("Nº Items", min_value=1, max_value=100, value=2, step=1)
            n_prod_inf = st.number_input("Nº Productos Distintos", min_value=1, max_value=50, value=1, step=1)
        with colf2:
            depto_inf = st.selectbox("Departamento", resultados["departamentos_validos"])
            pct_foto_inf = 1.0 if depto_inf == "FOTOCOPIADORA" else 0.0
            dia_inf = st.selectbox("Día de la Semana", _DIAS_SEMANA)

        if st.button("🚀 Predecir Método de Pago"):
            dia_idx = _DIAS_SEMANA.index(dia_inf)
            fila = pd.DataFrame([{
                "Total": total_inf,
                "n_items": n_items_inf,
                "n_productos_distintos": n_prod_inf,
                "departamento_principal_enc": le_depto.transform([depto_inf])[0],
                "pct_fotocopiadora": pct_foto_inf,
                "dia_semana": dia_idx,
                "es_fin_de_semana": int(dia_idx in [5, 6]),
            }])[columnas]

            pred = modelo.predict(fila)[0]
            prob = modelo.predict_proba(fila)[0, 1]
            if pred == 1:
                st.success(f"📱 Método Predicho: **YAPE** — Probabilidad: {prob:.0%}")
            else:
                st.info(f"💵 Método Predicho: **EFECTIVO** — Probabilidad de YAPE: {prob:.0%}")

    # ── Insights ──────────────────────────────────────────────────
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        _insight(
            "Preferencia por F1-Score",
            "El desbalance Efectivo (66.3%) vs Yape (33.7%) invalida la Accuracy como métrica "
            "principal. El F1-Score pondera Precision y Recall equitativamente.",
            badge="JUSTIFICACIÓN TÉCNICA"
        )
    with ic2:
        _insight(
            "Lectura honesta del desempeño",
            f"El modelo {resultados['modelo_recomendado']} es el recomendado (mejor F1), pero el "
            f"desempeño es moderado: monto, producto y día se relacionan con el pago de forma "
            f"real pero no muy fuerte.",
            badge="SHAP INSIGHT"
        )