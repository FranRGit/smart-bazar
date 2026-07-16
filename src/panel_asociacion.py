"""
Panel 3 – Reglas de Asociación
SmartBazar Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ── Mock data ────────────────────────────────────────────────────────────────
def _build_rules_df() -> pd.DataFrame:
    """Return a hardcoded DataFrame of 8 association rules."""
    data = {
        "Si compran (Antecedente)": [
            "FOTOCOPIA A4",
            "CUADERNO A4 COLLEGE",
            "FOLDER MANILA A4",
            "IMPRESION B/N",
            "LAPICERO FABER CASTELL",
            "PAPEL BOND A4",
            "CUADERNO A5",
            "PLUMONES FABER CASTELL",
        ],
        "También compran (Consecuente)": [
            "MICA A4 VINIFAN",
            "LAPICERO PILOT",
            "FASTER CLIPS",
            "ANILLADO SIMPLE",
            "BORRADOR BLANCO",
            "SOBRE MANILA A4",
            "CORRECTOR LIQUID PAPER",
            "CARTULINA A4",
        ],
        "Soporte": [0.045, 0.038, 0.032, 0.028, 0.025, 0.022, 0.019, 0.016],
        "Confianza": [0.72, 0.65, 0.58, 0.52, 0.48, 0.44, 0.41, 0.38],
        "Lift": [3.21, 2.89, 2.54, 2.31, 2.15, 1.98, 1.82, 1.65],
        "Leverage": [0.031, 0.025, 0.019, 0.016, 0.013, 0.011, 0.009, 0.007],
        "Convicción": [2.85, 2.31, 1.98, 1.72, 1.61, 1.52, 1.43, 1.35],
    }
    return pd.DataFrame(data)


# ── Chart constants ──────────────────────────────────────────────────────────
_BG_OUTER = "#fdf8f8"
_BG_INNER = "#ffffff"
_TEXT = "#1c1b1b"
_TITLE = "#000000"
_TICK = "#1c1b1b"
_SPINE = "#cfc4c5"
_GRID = "#cfc4c5"


def _style_ax(ax: plt.Axes) -> None:
    """Apply the light-theme styling to an axes object."""
    ax.set_facecolor(_BG_INNER)
    ax.tick_params(colors=_TICK)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TITLE)
    for spine in ax.spines.values():
        spine.set_color(_SPINE)
    ax.grid(True, alpha=0.3, color=_GRID)


# ── Chart builders ───────────────────────────────────────────────────────────
def _scatter_support_confidence(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot: Soporte vs Confianza, size ∝ Lift."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(_BG_OUTER)
    _style_ax(ax)

    scatter = ax.scatter(
        df["Soporte"],
        df["Confianza"],
        s=df["Lift"] * 50,
        c=df["Lift"],
        cmap="magma",
        edgecolors="#ffffff",
        linewidths=0.6,
        alpha=0.90,
        zorder=3,
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Lift", color=_TEXT)
    cbar.ax.yaxis.set_tick_params(color=_TICK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TICK)
    cbar.outline.set_edgecolor(_SPINE)

    # Annotate each point
    for _, row in df.iterrows():
        ax.annotate(
            row["Si compran (Antecedente)"],
            (row["Soporte"], row["Confianza"]),
            fontsize=7,
            color="#b0b0b0",
            textcoords="offset points",
            xytext=(6, 6),
        )

    ax.set_xlabel("Soporte", fontsize=11)
    ax.set_ylabel("Confianza", fontsize=11)
    ax.set_title(
        "Soporte vs Confianza (tamaño = Lift)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    return fig


def _bar_top_lift(df: pd.DataFrame, n: int = 5) -> plt.Figure:
    """Horizontal bar chart of the top-n rules by Lift."""
    top = df.nlargest(n, "Lift").iloc[::-1]  # reverse for bottom-to-top order
    labels = (
        top["Si compran (Antecedente)"] + " → " + top["También compran (Consecuente)"]
    )

    palette = ["#A78BFA", "#818CF8", "#34D399", "#FBBF24", "#F472B6"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(_BG_OUTER)
    _style_ax(ax)

    bars = ax.barh(
        labels,
        top["Lift"],
        color=palette,
        edgecolor="#2a2a2a",
        height=0.55,
        zorder=3,
    )

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha="left",
            color=_TEXT,
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Lift", fontsize=11)
    ax.set_title("Top 5 Reglas por Lift", fontsize=13, fontweight="bold")
    ax.set_xlim(0, top["Lift"].max() * 1.15)
    fig.tight_layout()
    return fig


# ── Main panel entry point ───────────────────────────────────────────────────
def show_panel() -> None:
    """Render Panel 3 – Reglas de Asociación."""

    st.header("📊 Panel 3: Reglas de Asociación")

    tab_reglas, tab_combos = st.tabs(
        ["🛒 Reglas de Asociación", "💡 Combos Sugeridos"]
    )

    rules_df = _build_rules_df()

    # ── Tab 1: Reglas de Asociación ──────────────────────────────────────
    with tab_reglas:
        st.markdown(
            """
            **Metodología Apriori**

            El algoritmo *Apriori* identifica conjuntos de productos que se compran
            frecuentemente juntos. A partir de las transacciones históricas se
            calculan tres métricas clave:

            | Métrica | Descripción |
            |---------|-------------|
            | **Soporte** | Proporción de transacciones que contienen la combinación. |
            | **Confianza** | Probabilidad de comprar el consecuente dado el antecedente. |
            | **Lift** | Ratio de co-ocurrencia respecto a la independencia estadística (lift > 1 indica asociación positiva). |

            Ajuste los umbrales con los controles inferiores para filtrar las reglas
            más relevantes.
            """
        )

        st.divider()

        # Sliders
        col1, col2, col3 = st.columns(3)
        with col1:
            min_support = st.slider(
                "Soporte Mínimo",
                min_value=0.005,
                max_value=0.10,
                value=0.015,
                step=0.005,
                format="%.3f",
            )
        with col2:
            min_confidence = st.slider(
                "Confianza Mínima",
                min_value=0.05,
                max_value=1.0,
                value=0.20,
                step=0.05,
            )
        with col3:
            min_lift = st.slider(
                "Lift Mínimo",
                min_value=0.5,
                max_value=10.0,
                value=1.2,
                step=0.1,
            )

        # Filter rules based on slider values
        filtered_df = rules_df[
            (rules_df["Soporte"] >= min_support)
            & (rules_df["Confianza"] >= min_confidence)
            & (rules_df["Lift"] >= min_lift)
        ].reset_index(drop=True)

        n_rules = len(filtered_df)
        st.success(f"Se han encontrado **{n_rules} reglas de asociación** válidas!")

        # Styled dataframe
        st.dataframe(
            filtered_df.style.format(
                {
                    "Soporte": "{:.3f}",
                    "Confianza": "{:.2f}",
                    "Lift": "{:.2f}",
                    "Leverage": "{:.3f}",
                    "Convicción": "{:.2f}",
                }
            )
            .background_gradient(subset=["Lift"], cmap="YlGn")
            .background_gradient(subset=["Confianza"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("🔍 Dispersión: Soporte vs Confianza")
        if not filtered_df.empty:
            fig_scatter = _scatter_support_confidence(filtered_df)
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)
        else:
            st.warning("No hay reglas que cumplan los filtros seleccionados.")

    # ── Tab 2: Combos Sugeridos ──────────────────────────────────────────
    with tab_combos:
        st.subheader("🎁 Combos Recomendados para Promociones")
        st.markdown(
            "Estas combinaciones de productos presentan la mayor asociación "
            "estadística. Considérelas para armar **paquetes promocionales**, "
            "**descuentos cruzados** o **ubicación estratégica** en tienda."
        )

        st.divider()

        top5 = rules_df.nlargest(5, "Lift")

        for _, row in top5.iterrows():
            soporte_pct = row["Soporte"] * 100
            confianza_pct = row["Confianza"] * 100
            st.info(
                f"🎁 **Combo: {row['Si compran (Antecedente)']} + "
                f"{row['También compran (Consecuente)']}**\n\n"
                f"- 📦 Soporte: **{soporte_pct:.1f}%** de las transacciones\n"
                f"- 🎯 Confianza: **{confianza_pct:.0f}%** de probabilidad\n"
                f"- ⚡ Lift: **{row['Lift']:.2f}x** más probable que al azar"
            )

        st.divider()
        st.subheader("📊 Top 5 Reglas por Lift")

        fig_bar = _bar_top_lift(rules_df, n=5)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
