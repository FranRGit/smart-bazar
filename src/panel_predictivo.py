import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────
# Utilidades de estilo para gráficos con tema oscuro
# ─────────────────────────────────────────────────────────────────────
def _apply_dark_style(fig, ax, title=""):
    """Aplica el tema claro premium a una figura matplotlib."""
    fig.patch.set_facecolor('#fdf8f8')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#1c1b1b')
    ax.xaxis.label.set_color('#1c1b1b')
    ax.yaxis.label.set_color('#1c1b1b')
    if title:
        ax.set_title(title, color='#000000', fontsize=13, fontweight='bold',
                     pad=12)
    for spine in ax.spines.values():
        spine.set_color('#cfc4c5')
    ax.grid(True, alpha=0.3, color='#cfc4c5')


def _safe_close(fig):
    """Renderiza la figura en Streamlit y la cierra correctamente."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Datos mock – métricas de modelos
# ─────────────────────────────────────────────────────────────────────
_MODEL_METRICS = pd.DataFrame(
    {
        "Modelo": ["Random Forest", "XGBoost"],
        "Accuracy": [0.8245, 0.8456],
        "Precision": [0.7912, 0.8134],
        "Recall": [0.7654, 0.7891],
        "F1-Score": [0.7781, 0.8011],
        "ROC-AUC": [0.8834, 0.9012],
    }
)

_CM_RF = np.array([[145, 32],
                   [28, 87]])

_CM_XGB = np.array([[151, 26],
                    [25, 90]])

_CM_LABELS = ["EFECTIVO", "YAPE"]

# Mock SHAP – valores locales por defecto
_LOCAL_SHAP_DEFAULTS = {
    "Total": 0.15,
    "hora_compra": 0.08,
    "n_items": -0.04,
    "departamento": 0.12,
    "pct_foto": -0.09,
    "dia_semana": 0.03,
    "es_fds": 0.02,
    "n_prod_distintos": -0.01,
}

# Mock SHAP – importancia global (mean |SHAP|)
_GLOBAL_SHAP = {
    "Total": 0.182,
    "hora_compra": 0.145,
    "pct_fotocopiadora": 0.123,
    "departamento": 0.098,
    "n_items": 0.076,
    "dia_semana": 0.054,
    "es_fin_de_semana": 0.043,
    "n_productos_distintos": 0.031,
}


# ─────────────────────────────────────────────────────────────────────
# Panel principal
# ─────────────────────────────────────────────────────────────────────
def show_panel():
    """Panel 4 – Predicción de Pagos (YAPE vs EFECTIVO)."""

    st.header("🔮 Panel 4: Predicción de Pagos")
    st.markdown(
        """
        Este panel predice el **Método de Pago** (YAPE vs EFECTIVO) de una
        compra basado en el perfil del ticket.  Compara dos modelos
        (**Random Forest** y **XGBoost**) e implementa la explicabilidad
        del modelo con **SHAP** (valores pre-calculados).
        """
    )

    # ── Pestañas ────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📊 Comparación de Modelos",
         "🧪 Inferencia en Vivo",
         "🔍 Explicabilidad (SHAP)"]
    )

    # ================================================================
    # TAB 1 – Comparación de Modelos
    # ================================================================
    with tab1:
        _render_tab_comparacion()

    # ================================================================
    # TAB 2 – Inferencia en Vivo
    # ================================================================
    with tab2:
        _render_tab_inferencia()

    # ================================================================
    # TAB 3 – Explicabilidad (SHAP)
    # ================================================================
    with tab3:
        _render_tab_explicabilidad()


# ─────────────────────────────────────────────────────────────────────
# TAB 1: Comparación de Modelos
# ─────────────────────────────────────────────────────────────────────
def _render_tab_comparacion():
    st.subheader("Métricas de Rendimiento en el Conjunto de Test")
    st.markdown(
        "La siguiente tabla muestra las métricas de evaluación de ambos "
        "modelos sobre el **20 %** de datos reservados para prueba."
    )

    # Dataframe estilizado – resaltar máximos en verde
    df_show = _MODEL_METRICS.set_index("Modelo")

    def _highlight_max(s):
        is_max = s == s.max()
        return [
            "background-color: #1b5e20; color: #a5d6a7; font-weight: bold"
            if v else "" for v in is_max
        ]

    styled = (
        df_show.style
        .format("{:.4f}")
        .apply(_highlight_max, axis=0)
    )
    st.dataframe(styled, use_container_width=True)

    st.success("🏆 El mejor modelo según F1-Score es: **XGBoost**")

    # ── Matrices de confusión ───────────────────────────────────────
    st.subheader("Matrices de Confusión")
    col_rf, col_xgb = st.columns(2)

    with col_rf:
        st.markdown("**Random Forest**")
        fig_rf, ax_rf = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            _CM_RF, annot=True, fmt="d", cmap="Blues",
            xticklabels=_CM_LABELS, yticklabels=_CM_LABELS,
            linewidths=0.8, linecolor='#333333',
            cbar_kws={"shrink": 0.75}, ax=ax_rf,
        )
        ax_rf.set_xlabel("Predicho", color='#e5e2e1')
        ax_rf.set_ylabel("Real", color='#e5e2e1')
        _apply_dark_style(fig_rf, ax_rf, title="Matriz de Confusión — RF")
        ax_rf.tick_params(colors='#c4c7c8')
        _safe_close(fig_rf)

    with col_xgb:
        st.markdown("**XGBoost**")
        fig_xgb, ax_xgb = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            _CM_XGB, annot=True, fmt="d", cmap="Oranges",
            xticklabels=_CM_LABELS, yticklabels=_CM_LABELS,
            linewidths=0.8, linecolor='#333333',
            cbar_kws={"shrink": 0.75}, ax=ax_xgb,
        )
        ax_xgb.set_xlabel("Predicho", color='#e5e2e1')
        ax_xgb.set_ylabel("Real", color='#e5e2e1')
        _apply_dark_style(fig_xgb, ax_xgb, title="Matriz de Confusión — XGB")
        ax_xgb.tick_params(colors='#c4c7c8')
        _safe_close(fig_xgb)

    st.markdown(
        """
        > **Interpretación del Error:** Un falso positivo (predecir YAPE
        > cuando paga en Efectivo) reduce el sencillo físico sin alerta
        > previa.  Maximizar la **precisión de YAPE** es fundamental.
        """
    )


# ─────────────────────────────────────────────────────────────────────
# TAB 2: Inferencia en Vivo
# ─────────────────────────────────────────────────────────────────────
def _render_tab_inferencia():
    st.subheader("Simulación de Compra (Predicción en Vivo)")
    st.markdown(
        "Modifique los valores para predecir si este cliente pagará "
        "con **YAPE** o **EFECTIVO**."
    )

    # ── Formulario ──────────────────────────────────────────────────
    with st.form("form_inferencia"):
        col_left, col_right = st.columns(2)

        with col_left:
            total = st.number_input(
                "Total (S/)", min_value=0.1, max_value=500.0,
                value=15.0, step=0.5,
            )
            hora = st.slider("Hora de Compra", 8, 22, 12)
            n_items = st.number_input(
                "Nº Items", min_value=1, max_value=100, value=2, step=1,
            )
            n_prod = st.number_input(
                "Nº Productos Distintos", min_value=1, max_value=50,
                value=1, step=1,
            )

        with col_right:
            departamento = st.selectbox(
                "Departamento",
                ["UTILES", "FOTOCOPIADORA", "GOLOSINAS",
                 "BEBIDAS", "SERVICIOS"],
            )
            pct_foto = st.slider(
                "% Fotocopiadora", 0.0, 1.0, 0.0, step=0.05,
            )
            dia_semana = st.selectbox(
                "Día de la Semana",
                ["Lunes", "Martes", "Miércoles", "Jueves",
                 "Viernes", "Sábado", "Domingo"],
            )

        submitted = st.form_submit_button("🚀 Predecir Método de Pago")

    # ── Lógica determinista mock ────────────────────────────────────
    if submitted:
        if departamento == "FOTOCOPIADORA" or total < 8:
            pred_label = "EFECTIVO"
            prob = 0.72
        elif total > 30 or hora >= 18:
            pred_label = "YAPE"
            prob = 0.81
        else:
            pred_label = "YAPE"
            prob = 0.63

        st.markdown("### Resultado de la Predicción")
        if pred_label == "YAPE":
            st.success(
                f"📱 **Método Predicho: YAPE** — "
                f"Probabilidad: {prob:.0%}"
            )
        else:
            st.info(
                f"💵 **Método Predicho: EFECTIVO** — "
                f"Probabilidad: {prob:.0%}"
            )

        # ── Gráfico SHAP local (mock) ──────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Explicabilidad Local — ¿Por qué esta predicción?")

        shap_vals = dict(_LOCAL_SHAP_DEFAULTS)  # copia

        # Pequeñas perturbaciones deterministas según inputs
        if total > 30:
            shap_vals["Total"] = 0.24
        if departamento == "FOTOCOPIADORA":
            shap_vals["departamento"] = -0.18
            shap_vals["pct_foto"] = 0.16
        if hora >= 18:
            shap_vals["hora_compra"] = 0.14

        df_shap = (
            pd.DataFrame(
                list(shap_vals.items()),
                columns=["Característica", "SHAP"],
            )
            .sort_values("SHAP", key=abs, ascending=True)
        )

        colors = [
            "#ff0d57" if v > 0 else "#1e88e5"
            for v in df_shap["SHAP"]
        ]

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh(
            df_shap["Característica"], df_shap["SHAP"],
            color=colors, edgecolor='none', height=0.6,
        )
        ax.axvline(0, color='#888888', linewidth=0.8)
        ax.set_xlabel("Contribución SHAP")
        _apply_dark_style(
            fig, ax,
            title="Explicación Local (SHAP)\n"
                  "Rojo → YAPE  ·  Azul → EFECTIVO",
        )
        _safe_close(fig)

        st.markdown(
            """
            - 🔴 **Rojo (positivo):** empuja la predicción hacia **YAPE**.
            - 🔵 **Azul (negativo):** empuja la predicción hacia **EFECTIVO**.
            """
        )


# ─────────────────────────────────────────────────────────────────────
# TAB 3: Explicabilidad (SHAP)
# ─────────────────────────────────────────────────────────────────────
def _render_tab_explicabilidad():
    st.subheader("Importancia Global de Variables (SHAP)")
    st.markdown(
        """
        El siguiente gráfico muestra el **impacto promedio global**
        (mean |SHAP value|) de cada variable en el modelo XGBoost
        seleccionado.  Ayuda a entender qué factores determinan el
        método de pago a nivel general del negocio.
        """
    )

    # ── 1. Barras de importancia global ─────────────────────────────
    df_global = (
        pd.DataFrame(
            list(_GLOBAL_SHAP.items()),
            columns=["Característica", "mean_|SHAP|"],
        )
        .sort_values("mean_|SHAP|", ascending=True)
    )

    n = len(df_global)
    # Gradiente de colores: de plata (baja) a esmeralda (alta)
    palette = [
        plt.cm.GnBu(0.25 + 0.65 * i / (n - 1)) for i in range(n)
    ]

    fig_g, ax_g = plt.subplots(figsize=(9, 5))
    ax_g.barh(
        df_global["Característica"],
        df_global["mean_|SHAP|"],
        color=palette, edgecolor='none', height=0.6,
    )
    ax_g.set_xlabel("mean |SHAP value|")
    _apply_dark_style(
        fig_g, ax_g,
        title="Importancia Global de Variables — XGBoost",
    )
    _safe_close(fig_g)

    # ── 2. Beeswarm simulado (scatter con jitter) ──────────────────
    st.subheader("Distribución Detallada (Beeswarm Simulado)")
    st.markdown(
        "Cada punto representa una observación.  El color indica el "
        "**valor de la variable** (alto → rojo, bajo → azul).  La "
        "posición horizontal es el impacto SHAP."
    )

    features = list(reversed(df_global["Característica"].tolist()))
    rng = np.random.default_rng(42)
    n_points = 120  # puntos por feature

    fig_b, ax_b = plt.subplots(figsize=(9, 5.5))

    for idx, feat in enumerate(features):
        base_importance = _GLOBAL_SHAP[feat]
        # Valores SHAP simulados centrados en 0, dispersión proporcional
        shap_vals = rng.normal(0, base_importance * 1.8, n_points)
        # Valor de la variable simulado (0-1) correlacionado con SHAP
        feat_vals = np.clip(
            0.5 + shap_vals / (2 * base_importance + 1e-6)
            + rng.normal(0, 0.15, n_points),
            0, 1,
        )
        # Jitter vertical
        y_jitter = idx + rng.uniform(-0.3, 0.3, n_points)

        sc = ax_b.scatter(
            shap_vals, y_jitter,
            c=feat_vals, cmap="coolwarm", s=12, alpha=0.65,
            edgecolors='none', vmin=0, vmax=1,
        )

    ax_b.set_yticks(range(len(features)))
    ax_b.set_yticklabels(features)
    ax_b.axvline(0, color='#888888', linewidth=0.8)
    ax_b.set_xlabel("Valor SHAP (impacto en la predicción)")

    # Barra de color
    cbar = fig_b.colorbar(sc, ax=ax_b, pad=0.02, aspect=30)
    cbar.set_label("Valor de la variable", color='#c4c7c8')
    cbar.ax.yaxis.set_tick_params(color='#c4c7c8')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#c4c7c8')
    cbar.outline.set_edgecolor('#444748')

    _apply_dark_style(
        fig_b, ax_b,
        title="Beeswarm Plot (simulado) — XGBoost",
    )
    _safe_close(fig_b)

    st.markdown(
        """
        > **Lectura del gráfico:** Si los puntos rojos (valor alto de la
        > variable) se concentran a la **derecha**, esa variable alta
        > empuja hacia YAPE.  Si se concentran a la **izquierda**, empuja
        > hacia EFECTIVO.
        """
    )
