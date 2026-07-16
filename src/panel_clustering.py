"""
Panel 2 – Segmentación con K-Means
SmartBazar Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Seed global para reproducibilidad ───────────────────────────────────────
np.random.seed(42)

# ── Paleta de colores para clusters ─────────────────────────────────────────
CLUSTER_PALETTE = [
    "#8b5cf6",  # violeta
    "#06b6d4",  # cian
    "#10b981",  # esmeralda
    "#f59e0b",  # ámbar
    "#ef4444",  # rojo
    "#ec4899",  # rosa
    "#6366f1",  # índigo
    "#14b8a6",  # teal
]

# ── Métricas mock por valor de K ────────────────────────────────────────────
MOCK_METRICS: dict[int, dict] = {
    2: {"silueta": 0.4521, "inercia": 1842.5},
    3: {"silueta": 0.5187, "inercia": 1205.3},
    4: {"silueta": 0.4893, "inercia": 892.1},
    5: {"silueta": 0.4612, "inercia": 701.8},
    6: {"silueta": 0.4201, "inercia": 589.4},
    7: {"silueta": 0.3845, "inercia": 498.2},
    8: {"silueta": 0.3512, "inercia": 421.7},
}

# ── Centros base (se interpolan para K != 3) ────────────────────────────────
_BASE_CENTERS: list[tuple[float, float]] = [
    (12, 3),
    (45, 8),
    (85, 15),
    (25, 12),
    (60, 5),
    (95, 20),
    (35, 18),
    (70, 10),
]

# ── Horas promedio mock por cluster (hasta 8 clusters) ──────────────────────
_MOCK_HOURS = [10.5, 14.2, 18.7, 12.1, 16.3, 9.8, 20.1, 11.4]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _apply_chart_style(fig: plt.Figure, ax: plt.Axes) -> None:
    """Aplica el estilo claro estándar del dashboard a una figura."""
    fig.patch.set_facecolor("#fdf8f8")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#1c1b1b")
    ax.xaxis.label.set_color("#1c1b1b")
    ax.yaxis.label.set_color("#1c1b1b")
    if ax.title:
        ax.title.set_color("#000000")
    for spine in ax.spines.values():
        spine.set_color("#cfc4c5")
    ax.grid(True, alpha=0.3, color="#cfc4c5")


def _generate_cluster_data(k: int, n_per_cluster: int = 60) -> pd.DataFrame:
    """Genera datos mock de clusters con centros diferenciados."""
    rng = np.random.RandomState(42)
    frames: list[pd.DataFrame] = []

    centers = _BASE_CENTERS[:k]
    for idx, (cx, cy) in enumerate(centers):
        x = rng.normal(loc=cx, scale=6, size=n_per_cluster)
        y = rng.normal(loc=cy, scale=2, size=n_per_cluster)
        df = pd.DataFrame(
            {
                "Monto Total (S/)": np.clip(x, 1, None),
                "Cantidad de Artículos": np.clip(y, 1, None).astype(int),
                "Cluster": idx,
            }
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _build_stats_table(data: pd.DataFrame, k: int) -> pd.DataFrame:
    """Construye la tabla de estadísticas por cluster."""
    rows: list[dict] = []
    for c in range(k):
        subset = data[data["Cluster"] == c]
        rows.append(
            {
                "Cluster": f"Cluster {c}",
                "Monto Promedio (S/)": round(subset["Monto Total (S/)"].mean(), 2),
                "Artículos Promedio": round(
                    subset["Cantidad de Artículos"].mean(), 1
                ),
                "Hora Promedio": f"{_MOCK_HOURS[c]:.1f}",
                "Tickets": len(subset),
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Scatter de clusters
# ─────────────────────────────────────────────────────────────────────────────
def _plot_clusters(data: pd.DataFrame, k: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_chart_style(fig, ax)

    for c in range(k):
        subset = data[data["Cluster"] == c]
        ax.scatter(
            subset["Monto Total (S/)"],
            subset["Cantidad de Artículos"],
            c=CLUSTER_PALETTE[c],
            label=f"Cluster {c}",
            alpha=0.72,
            edgecolors="white",
            linewidths=0.3,
            s=48,
        )

    ax.set_xlabel("Monto Total (S/)", fontsize=11)
    ax.set_ylabel("Cantidad de Artículos", fontsize=11)
    ax.set_title("Segmentación de Clientes por K-Means", fontsize=13, fontweight="bold")
    ax.legend(
        facecolor="#1a1a1a",
        edgecolor="#444748",
        labelcolor="#e5e2e1",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos de evaluación
# ─────────────────────────────────────────────────────────────────────────────
def _plot_elbow(metrics: dict) -> plt.Figure:
    ks = sorted(metrics.keys())
    inercias = [metrics[k]["inercia"] for k in ks]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    _apply_chart_style(fig, ax)

    ax.plot(ks, inercias, marker="o", color="#06b6d4", linewidth=2, markersize=7)
    # Marcar K óptimo = 3
    opt_idx = ks.index(3)
    ax.plot(
        3,
        inercias[opt_idx],
        marker="*",
        color="#f59e0b",
        markersize=18,
        zorder=5,
    )
    ax.annotate(
        "K óptimo = 3",
        xy=(3, inercias[opt_idx]),
        xytext=(4.3, inercias[opt_idx] + 180),
        fontsize=9,
        color="#f59e0b",
        arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2),
    )

    ax.set_xlabel("Número de Clusters (K)", fontsize=10)
    ax.set_ylabel("Inercia", fontsize=10)
    ax.set_title("Método del Codo", fontsize=12, fontweight="bold")
    ax.set_xticks(ks)
    fig.tight_layout()
    return fig


def _plot_silhouette(metrics: dict) -> plt.Figure:
    ks = sorted(metrics.keys())
    siluetas = [metrics[k]["silueta"] for k in ks]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    _apply_chart_style(fig, ax)

    ax.plot(ks, siluetas, marker="o", color="#8b5cf6", linewidth=2, markersize=7)
    # Marcar K óptimo = 3
    opt_idx = ks.index(3)
    ax.plot(
        3,
        siluetas[opt_idx],
        marker="*",
        color="#f59e0b",
        markersize=18,
        zorder=5,
    )
    ax.annotate(
        "K óptimo = 3",
        xy=(3, siluetas[opt_idx]),
        xytext=(4.3, siluetas[opt_idx] + 0.02),
        fontsize=9,
        color="#f59e0b",
        arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.2),
    )

    ax.set_xlabel("Número de Clusters (K)", fontsize=10)
    ax.set_ylabel("Coeficiente de Silueta", fontsize=10)
    ax.set_title("Puntuación de Silueta", fontsize=12, fontweight="bold")
    ax.set_xticks(ks)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Panel principal
# ─────────────────────────────────────────────────────────────────────────────
def show_panel() -> None:
    """Renderiza el Panel 2 – Segmentación con K-Means."""

    tab_seg, tab_eval = st.tabs(
        ["🧩 Segmentación K-Means", "📐 Evaluación del Modelo"]
    )

    # ── Tab 1: Segmentación ─────────────────────────────────────────────────
    with tab_seg:
        st.markdown(
            """
            ### 🧩 Segmentación de Clientes con K-Means

            Este módulo aplica el algoritmo **K-Means** para segmentar a los clientes
            de SmartBazar en grupos homogéneos basándose en su **monto total de compra**
            y la **cantidad de artículos** adquiridos por ticket. La segmentación
            permite identificar perfiles de consumo y diseñar estrategias
            comerciales diferenciadas para cada grupo.
            """
        )

        st.divider()

        k = st.slider(
            "Seleccione el número de Clusters (K):",
            min_value=2,
            max_value=8,
            value=3,
            step=1,
        )

        # ── Métricas ────────────────────────────────────────────────────────
        current = MOCK_METRICS[k]
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Coeficiente de Silueta",
                value=f"{current['silueta']:.4f}",
                delta="Óptimo" if k == 3 else None,
            )
        with col_m2:
            st.metric(
                label="Inercia del Modelo",
                value=f"{current['inercia']:,.1f}",
            )

        st.markdown("")  # espaciador

        # ── Scatter plot ────────────────────────────────────────────────────
        cluster_data = _generate_cluster_data(k)
        fig_scatter = _plot_clusters(cluster_data, k)
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)

        st.markdown("")  # espaciador

        # ── Tabla de estadísticas ───────────────────────────────────────────
        st.markdown("#### 📊 Estadísticas por Cluster")
        stats_df = _build_stats_table(cluster_data, k)
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
        )

    # ── Tab 2: Evaluación del Modelo ────────────────────────────────────────
    with tab_eval:
        st.markdown(
            """
            ### 📐 Evaluación del Modelo K-Means

            Se evalúa el rendimiento del modelo a través de dos métricas
            complementarias para determinar el **número óptimo de clusters**.
            """
        )

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            fig_elbow = _plot_elbow(MOCK_METRICS)
            st.pyplot(fig_elbow)
            plt.close(fig_elbow)

        with col_right:
            fig_sil = _plot_silhouette(MOCK_METRICS)
            st.pyplot(fig_sil)
            plt.close(fig_sil)

        st.markdown("")  # espaciador

        with st.expander("📖 Interpretación de los Resultados", expanded=False):
            st.markdown(
                """
                **Método del Codo (Elbow Method):**
                La inercia mide la suma de las distancias al cuadrado de cada punto
                a su centroide asignado. A medida que **K** aumenta, la inercia
                disminuye. El «codo» de la curva — donde la reducción marginal
                se vuelve menor — indica el K óptimo. En nuestro caso, el codo
                se observa claramente en **K = 3**.

                **Coeficiente de Silueta:**
                Valores cercanos a **1** indican clusters bien separados y
                cohesionados, mientras que valores cercanos a **0** sugieren
                solapamiento. El máximo se alcanza en **K = 3** con un valor
                de **0.5187**, confirmando que tres segmentos representan la
                mejor partición para estos datos.

                **Conclusión:**
                Con **K = 3** se obtiene el balance óptimo entre cohesión
                intra-cluster y separación inter-cluster, identificando tres
                perfiles de clientes:
                - **Cluster 0** – Compras de bajo monto y pocos artículos.
                - **Cluster 1** – Compras de monto medio con volumen moderado.
                - **Cluster 2** – Compras premium de alto monto y muchos artículos.
                """
            )
