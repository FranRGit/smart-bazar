"""
Panel 1 – Análisis Exploratorio de Datos (EDA)
Auditoría, limpieza y exploración visual de las ventas de SmartBazar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _apply_chart_style(fig, ax, *, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Aplica el estilo claro de Stitch a cualquier figura matplotlib."""
    fig.patch.set_facecolor("#fdf8f8")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#1c1b1b")
    ax.xaxis.label.set_color("#1c1b1b")
    ax.yaxis.label.set_color("#1c1b1b")
    for spine in ax.spines.values():
        spine.set_color("#cfc4c5")
    ax.grid(True, alpha=0.3, color="#cfc4c5")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#000000")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.tight_layout()


def _safe_render(fig):
    """Renderiza la figura en Streamlit y la cierra para liberar memoria."""
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


# ──────────────────────────────────────────────
# Mock data generators (deterministic)
# ──────────────────────────────────────────────

def _generate_sales_amounts() -> np.ndarray:
    """Genera montos de venta sintéticos con forma exponencial + normal + uniforme."""
    return np.concatenate([
        np.random.exponential(10, 600),
        np.random.normal(25, 8, 200),
        np.random.uniform(50, 120, 54),
    ])


def _build_audit_table() -> pd.DataFrame:
    """Tabla de auditoría de campos."""
    return pd.DataFrame({
        "Campo": ["ID", "Fecha", "Total", "Metodo_Pago", "ID_Cliente"],
        "Tipo": ["str", "datetime", "float", "str", "int"],
        "Nulos": [0, 3, 0, 2, 8],
        "% Completitud": ["100%", "99.6%", "100%", "99.8%", "99.1%"],
        "Estado": ["✅ OK", "⚠️ Imputado", "✅ OK", "⚠️ Imputado", "⚠️ Imputado"],
    })


def _build_outliers_table() -> pd.DataFrame:
    """Top 5 outliers de ventas."""
    return pd.DataFrame({
        "ID": ["V-0847", "V-0623", "V-0419", "V-0751", "V-0538"],
        "Fecha": [
            "2025-12-28", "2025-11-15", "2025-10-03",
            "2025-12-10", "2025-11-22",
        ],
        "Metodo_Pago": ["EFECTIVO", "YAPE", "EFECTIVO", "YAPE", "EFECTIVO"],
        "Total": [118.74, 115.20, 112.95, 109.88, 107.42],
    })


# ──────────────────────────────────────────────
# Tab renderers
# ──────────────────────────────────────────────

def _render_tab_auditoria():
    """Tab 1 – Auditoría y Limpieza de Datos."""

    # ── KPIs ────────────────────────────────
    st.markdown("#### 📋 Resumen de Calidad de Datos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Registros", "854")
    c2.metric("Registros Válidos", "841")
    c3.metric("Nulos Corregidos", "13")
    c4.metric("Cobertura", "98.5 %")

    st.divider()

    # ── Tabla de auditoría ──────────────────
    st.markdown("#### 🗂️ Auditoría por Campo")
    audit_df = _build_audit_table()
    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── Detalle de correcciones ─────────────
    with st.expander("🔧 Ver Detalle de Correcciones"):
        st.markdown(
            """
- **Fecha** → 3 registros con fecha nula fueron imputados con la mediana del mes correspondiente.
- **Metodo_Pago** → 2 registros vacíos se asignaron como *EFECTIVO* (moda del dataset).
- **ID_Cliente** → 8 registros sin identificador de cliente se completaron con un código genérico `CLI-0000`.
- Se eliminaron 0 filas duplicadas; no se detectaron registros repetidos.
- Todos los montos de **Total** pasaron la validación de rango (> 0 y < 500).
            """
        )


def _render_tab_exploratorio(sales: np.ndarray):
    """Tab 2 – Análisis Exploratorio."""

    # ── KPIs ────────────────────────────────
    st.markdown("#### 📈 Estadísticos de Venta")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ticket Promedio", "S/ 12.47")
    c2.metric("Mediana", "S/ 8.50")
    c3.metric("Desviación Estándar", "S/ 15.32")

    st.divider()

    # ── Histograma de montos ────────────────
    st.markdown("#### 📊 Distribución de Montos de Venta")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        sales,
        bins=40,
        kde=True,
        color="#8b5cf6",
        edgecolor="#cfc4c5",
        linewidth=0.5,
        ax=ax,
    )
    # KDE line color
    if ax.lines:
        ax.lines[0].set_color("#06b6d4")
    _apply_chart_style(
        fig, ax,
        title="Distribución de Montos de Venta (S/)",
        xlabel="Monto (S/)",
        ylabel="Frecuencia",
    )
    _safe_render(fig)

    st.divider()

    # ── Boxplot + Outliers ──────────────────
    st.markdown("#### 🔎 Detección de Outliers")
    col_box, col_tbl = st.columns([1, 1])

    with col_box:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bp = ax2.boxplot(
            sales,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#8b5cf6", edgecolor="#06b6d4", linewidth=1.2),
            whiskerprops=dict(color="#06b6d4", linewidth=1.2),
            capprops=dict(color="#06b6d4", linewidth=1.2),
            medianprops=dict(color="#10b981", linewidth=2),
            flierprops=dict(
                marker="o",
                markerfacecolor="#f59e0b",
                markeredgecolor="#f59e0b",
                markersize=4,
                alpha=0.6,
            ),
        )
        _apply_chart_style(
            fig2, ax2,
            title="Boxplot de Montos",
            ylabel="Monto (S/)",
        )
        ax2.set_xticks([])
        _safe_render(fig2)

    with col_tbl:
        st.markdown("**Top 5 Outliers**")
        outliers_df = _build_outliers_table()
        st.dataframe(outliers_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Ventas por Método de Pago ───────────
    st.markdown("#### 💳 Ventas por Método de Pago")
    pago_df = pd.DataFrame({
        "Metodo_Pago": ["EFECTIVO", "YAPE"],
        "Cantidad": [512, 329],
    }).sort_values("Cantidad")

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    colors_pago = ["#06b6d4", "#8b5cf6"]
    ax3.barh(
        pago_df["Metodo_Pago"],
        pago_df["Cantidad"],
        color=colors_pago,
        edgecolor="#cfc4c5",
        height=0.5,
    )
    for i, val in enumerate(pago_df["Cantidad"]):
        ax3.text(val + 8, i, str(val), va="center", color="#1c1b1b", fontsize=11)
    _apply_chart_style(
        fig3, ax3,
        title="Cantidad de Ventas por Método de Pago",
        xlabel="Cantidad de Transacciones",
    )
    _safe_render(fig3)

    st.divider()

    # ── Ventas por Día de la Semana ─────────
    st.markdown("#### 📅 Ventas por Día de la Semana")
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    ventas_dia = [110, 98, 125, 102, 140, 168, 111]

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    bar_colors = [
        "#8b5cf6", "#8b5cf6", "#8b5cf6", "#8b5cf6",
        "#06b6d4", "#10b981", "#8b5cf6",
    ]
    bars = ax4.bar(
        dias,
        ventas_dia,
        color=bar_colors,
        edgecolor="#cfc4c5",
        width=0.6,
    )
    for bar, val in zip(bars, ventas_dia):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(val),
            ha="center",
            color="#1c1b1b",
            fontsize=10,
        )
    _apply_chart_style(
        fig4, ax4,
        title="Transacciones por Día de la Semana",
        ylabel="Cantidad",
    )
    ax4.set_xticks(range(len(dias)))
    ax4.set_xticklabels(dias, rotation=30, ha="right")
    fig4.tight_layout()
    _safe_render(fig4)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def show_panel():
    """Punto de entrada del panel EDA – invocado desde el dashboard principal."""

    np.random.seed(42)

    st.header("🔬 Análisis Exploratorio de Datos (EDA)")
    st.caption(
        "Auditoría de calidad, limpieza de datos y exploración estadística "
        "de las transacciones de SmartBazar."
    )

    tab_audit, tab_eda = st.tabs(["🔍 Auditoría y Limpieza", "📊 Análisis Exploratorio"])

    sales = _generate_sales_amounts()

    with tab_audit:
        _render_tab_auditoria()

    with tab_eda:
        _render_tab_exploratorio(sales)
