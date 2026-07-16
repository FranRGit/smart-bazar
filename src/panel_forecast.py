import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers: dark-theme chart styling
# ---------------------------------------------------------------------------

def _apply_dark_style(fig, ax):
    """Apply the SmartBazar light theme to a matplotlib figure/axes pair."""
    fig.patch.set_facecolor('#fdf8f8')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#1c1b1b')
    ax.xaxis.label.set_color('#1c1b1b')
    ax.yaxis.label.set_color('#1c1b1b')
    ax.title.set_color('#000000')
    for spine in ax.spines.values():
        spine.set_color('#cfc4c5')
    ax.grid(True, alpha=0.3, color='#cfc4c5')


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------

def _generate_historical_series():
    """Return a DataFrame with 90 days of mock daily revenue."""
    np.random.seed(42)
    dates_hist = pd.date_range('2025-03-01', periods=90, freq='D')
    base = (
        80
        + 15 * np.sin(np.arange(90) * 2 * np.pi / 7)
        + np.random.normal(0, 12, 90)
    )
    base = np.clip(base, 0, None)
    return pd.DataFrame({'ds': dates_hist, 'y': base})


def _generate_forecast(df_hist, horizon):
    """Generate a mock forecast continuing the historical pattern."""
    np.random.seed(123)
    last_date = df_hist['ds'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                 periods=horizon, freq='D')

    start_idx = len(df_hist)
    t = np.arange(start_idx, start_idx + horizon)

    # Continue the sinusoidal weekly pattern with a slight upward drift
    yhat = (
        85
        + 0.15 * t
        + 15 * np.sin(t * 2 * np.pi / 7)
        + np.random.normal(0, 6, horizon)
    )
    yhat = np.clip(yhat, 0, None)

    # Confidence bands widen as we move further into the future
    spread = np.linspace(8, 18, horizon)
    yhat_lower = np.clip(yhat - spread, 0, None)
    yhat_upper = yhat + spread

    return pd.DataFrame({
        'Fecha': future_dates,
        'Ingreso Estimado (S/)': np.round(yhat, 2),
        'Límite Mínimo (S/)': np.round(yhat_lower, 2),
        'Límite Máximo (S/)': np.round(yhat_upper, 2),
    })


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def show_panel():
    """Panel 5 – Pronóstico de Ingresos."""

    st.header("📈 Panel 5: Pronóstico de Ingresos Diarios (Series Temporales)")
    st.write(
        """
        Este panel proyecta los ingresos diarios para los próximos días
        utilizando un modelo de **series temporales** y lo compara con una
        **Media Móvil de 7 días** (baseline).  Los datos mostrados son
        simulados con fines demostrativos.
        """
    )

    # ── Sidebar controls ─────────────────────────────────────────────────
    st.sidebar.markdown("### 🔮 Ajustes de Pronóstico")
    forecast_horizon = st.sidebar.slider(
        "Horizonte de Predicción (Días):",
        min_value=3,
        max_value=30,
        value=7,
    )

    # ── Data generation ──────────────────────────────────────────────────
    df_hist = _generate_historical_series()
    df_forecast = _generate_forecast(df_hist, forecast_horizon)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["🔮 Pronóstico Futuro", "📊 Evaluación en Test", "🧩 Componentes del Modelo"]
    )

    # ==================================================================
    # TAB 1 – Pronóstico Futuro
    # ==================================================================
    with tab1:
        st.subheader(f"Pronóstico de Ingresos para los Próximos {forecast_horizon} Días")

        # ── Forecast table ────────────────────────────────────────────
        st.write("#### 📋 Tabla de Proyecciones de Ingresos")
        st.dataframe(
            df_forecast.style.format({
                'Ingreso Estimado (S/)': 'S/ {:.2f}',
                'Límite Mínimo (S/)': 'S/ {:.2f}',
                'Límite Máximo (S/)': 'S/ {:.2f}',
            }),
            use_container_width=True,
        )

        # ── Forecast chart ────────────────────────────────────────────
        st.write("#### 📈 Gráfico del Pronóstico Completo")
        fig_f, ax = plt.subplots(figsize=(11, 5))
        _apply_dark_style(fig_f, ax)

        # Historical line
        ax.plot(
            df_hist['ds'], df_hist['y'],
            label='Histórico Real',
            color='#ffffff',
            linewidth=1.2,
            alpha=0.85,
        )

        # Forecast line
        ax.plot(
            df_forecast['Fecha'],
            df_forecast['Ingreso Estimado (S/)'],
            label='Pronóstico',
            color='#e056fd',
            linewidth=2.5,
            marker='o',
            markersize=5,
        )

        # Confidence band
        ax.fill_between(
            df_forecast['Fecha'],
            df_forecast['Límite Mínimo (S/)'],
            df_forecast['Límite Máximo (S/)'],
            color='#e056fd',
            alpha=0.15,
            label='Intervalo de Confianza',
        )

        ax.set_title("Pronóstico Futuro de Flujo de Caja", fontsize=14, pad=12)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ingresos Diarios (S/)")
        ax.legend(facecolor='#1a1a1a', edgecolor='#444748', labelcolor='#e5e2e1')
        fig_f.autofmt_xdate()
        fig_f.tight_layout()
        st.pyplot(fig_f)
        plt.close(fig_f)

    # ==================================================================
    # TAB 2 – Evaluación en Test
    # ==================================================================
    with tab2:
        st.subheader("Evaluación de Modelos: Prophet vs. Media Móvil")
        st.write("Evaluados sobre los últimos **30 días** del historial.")

        # ── Metrics ──────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prophet RMSE", "S/ 18.45")
            st.metric("Prophet MAPE", "23.1 %")
        with col2:
            st.metric("Media Móvil RMSE", "S/ 24.67")
            st.metric("Media Móvil MAPE", "31.8 %")

        # ── Comparison chart ─────────────────────────────────────────
        st.write("#### 📊 Gráfico Comparativo en Ventana de Test")

        test_df = df_hist.tail(30).copy().reset_index(drop=True)
        np.random.seed(99)
        prophet_pred = test_df['y'] + np.random.normal(0, 10, 30)
        prophet_pred = np.clip(prophet_pred, 0, None)
        ma_pred = np.full(30, test_df['y'].mean())

        fig_t, ax_t = plt.subplots(figsize=(11, 5))
        _apply_dark_style(fig_t, ax_t)

        ax_t.plot(
            test_df['ds'], test_df['y'],
            label='Real',
            color='#74b9ff',
            marker='o',
            markersize=4,
            linewidth=1.3,
        )
        ax_t.plot(
            test_df['ds'], prophet_pred,
            label='Prophet',
            color='#ff7675',
            linestyle='--',
            linewidth=1.5,
        )
        ax_t.plot(
            test_df['ds'], ma_pred,
            label='Media Móvil (7)',
            color='#55efc4',
            linestyle=':',
            linewidth=2,
        )

        ax_t.set_title("Valores Reales vs. Predicciones (Conjunto de Prueba)",
                        fontsize=14, pad=12)
        ax_t.set_xlabel("Fecha")
        ax_t.set_ylabel("Ingresos Totales (S/)")
        ax_t.legend(facecolor='#1a1a1a', edgecolor='#444748', labelcolor='#e5e2e1')
        fig_t.autofmt_xdate()
        fig_t.tight_layout()
        st.pyplot(fig_t)
        plt.close(fig_t)

        with st.expander("ℹ️ ¿Qué significan estas métricas?"):
            st.markdown(
                """
                * **RMSE (Root Mean Squared Error):** Magnitud de los errores
                  en las mismas unidades que la variable (Soles). Menor es mejor.
                * **MAPE (Mean Absolute Percentage Error):** Error expresado
                  en porcentaje. Un MAPE menor indica mayor precisión.
                """
            )

    # ==================================================================
    # TAB 3 – Componentes del Modelo
    # ==================================================================
    with tab3:
        st.subheader("Componentes del Modelo de Series Temporales")
        st.write(
            "El modelo descompone la serie temporal en una **tendencia global** "
            "y una **estacionalidad semanal**."
        )

        # ── Trend component ──────────────────────────────────────────
        st.write("#### 📈 Componente de Tendencia")
        days = np.arange(90)
        trend = np.linspace(70, 95, 90)

        fig_tr, ax_tr = plt.subplots(figsize=(11, 4))
        _apply_dark_style(fig_tr, ax_tr)

        ax_tr.plot(
            df_hist['ds'], trend,
            color='#ffeaa7',
            linewidth=2.2,
        )
        ax_tr.set_title("Tendencia Global (90 días)", fontsize=14, pad=12)
        ax_tr.set_xlabel("Fecha")
        ax_tr.set_ylabel("Nivel de Tendencia (S/)")
        fig_tr.autofmt_xdate()
        fig_tr.tight_layout()
        st.pyplot(fig_tr)
        plt.close(fig_tr)

        # ── Weekly seasonality component ─────────────────────────────
        st.write("#### 🔄 Componente de Estacionalidad Semanal")
        day_labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        # Higher on Mon–Wed, dipping on weekends
        seasonality_vals = [12.5, 10.8, 9.2, 4.3, 1.5, -8.0, -11.5]

        fig_s, ax_s = plt.subplots(figsize=(8, 4))
        _apply_dark_style(fig_s, ax_s)

        bar_colors = ['#a29bfe' if v >= 0 else '#636e72' for v in seasonality_vals]
        ax_s.bar(day_labels, seasonality_vals, color=bar_colors, edgecolor='#444748',
                 linewidth=0.6, width=0.6)
        ax_s.axhline(0, color='#636e72', linewidth=0.8, linestyle='--')
        ax_s.set_title("Efecto Estacional por Día de la Semana", fontsize=14, pad=12)
        ax_s.set_xlabel("Día de la Semana")
        ax_s.set_ylabel("Efecto sobre Ingresos (S/)")
        fig_s.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)
