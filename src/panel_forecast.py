import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from src.data_loader import load_ventas

# Helper function to preprocess sales time series
def preparar_serie_temporal(ventas):
    # Normalize dates
    df = ventas.copy()
    # Normalize to YYYY-MM-DD
    df['Fecha_Diaria'] = df['Fecha'].dt.normalize()
    
    # Group by date
    ingreso_diario = df.groupby('Fecha_Diaria')['Total'].sum().reset_index()
    ingreso_diario.rename(columns={'Total': 'y'}, inplace=True)
    ingreso_diario.set_index('Fecha_Diaria', inplace=True)
    
    # Fill missing dates with 0
    if not ingreso_diario.empty:
        min_date = ingreso_diario.index.min()
        max_date = ingreso_diario.index.max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        ingreso_diario_completo = ingreso_diario.reindex(full_date_range, fill_value=0)
        ingreso_diario_completo.index.name = 'ds'
        return ingreso_diario_completo.reset_index()
    return pd.DataFrame(columns=['ds', 'y'])

def show_panel():
    st.header("📈 Panel 3: Pronóstico de Ingresos Diarios (Series Temporales)")
    st.write(
        """
        Este panel predice los ingresos diarios para los próximos días utilizando el modelo **Prophet** de Meta 
        y lo compara con una **Media Móvil de 7 días** (baseline).
        """
    )
    
    with st.spinner("Cargando y preparando serie temporal de ingresos..."):
        try:
            ventas = load_ventas()
            df_ts = preparar_serie_temporal(ventas)
        except Exception as e:
            st.error(f"Error al procesar la serie temporal: {e}")
            return
            
    if df_ts.empty:
        st.warning("No hay suficientes datos para realizar el pronóstico.")
        return
        
    # Sidebar parameter tuning for Prophet (excellent for live exams)
    st.sidebar.markdown("### 🚨 Ajustes de Series Temporales")
    test_days = st.sidebar.slider("Días para ventana de prueba (Test):", min_value=7, max_value=60, value=30)
    forecast_horizon = st.sidebar.slider("Horizonte de Predicción Futura (Días):", min_value=3, max_value=30, value=7)
    seasonality_mode = st.sidebar.selectbox("Modo de Estacionalidad:", options=["additive", "multiplicative"])
    
    st.write(f"**Rango de Fechas de la Data:** Desde {df_ts['ds'].min().strftime('%Y-%m-%d')} hasta {df_ts['ds'].max().strftime('%Y-%m-%d')} ({len(df_ts)} días totales).")
    
    # Train/Test Split
    train_df = df_ts[:-test_days].copy()
    test_df = df_ts[-test_days:].copy()
    
    # Train Prophet Model
    with st.spinner("Entrenando Prophet y Media Móvil..."):
        # Prophet model
        model = Prophet(
            growth='linear',
            seasonality_mode=seasonality_mode,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(train_df)
        
        # Test predictions
        future_eval = model.make_future_dataframe(periods=test_days, freq='D', include_history=False)
        forecast_eval = model.predict(future_eval)
        
        # Post-process for non-negative values (Business requirement)
        forecast_eval['yhat'] = forecast_eval['yhat'].apply(lambda x: max(x, 0))
        
        # Evaluation Metrics
        performance_df = pd.merge(test_df, forecast_eval[['ds', 'yhat']], on='ds', how='left')
        performance_df = performance_df.rename(columns={'y': 'y_true'})
        performance_df.dropna(subset=['yhat'], inplace=True)
        
        rmse_prophet = np.sqrt(mean_squared_error(performance_df['y_true'], performance_df['yhat']))
        # Avoid division by zero by replacing zero values with 1 for metric calculation
        mape_prophet = mean_absolute_percentage_error(performance_df['y_true'].replace(0, 1), performance_df['yhat'])
        
        # Baseline: Moving Average (MA 7 days)
        rolling_mean_train = train_df['y'].rolling(window=7).mean()
        ma_prediction_val = rolling_mean_train.iloc[-1] if not rolling_mean_train.dropna().empty else train_df['y'].mean()
        
        ma_test_predictions = [ma_prediction_val] * len(test_df)
        rmse_ma = np.sqrt(mean_squared_error(test_df['y'], ma_test_predictions))
        mape_ma = mean_absolute_percentage_error(test_df['y'].replace(0, 1), ma_test_predictions)
        
        # Future Forecast with Full Data
        model_full = Prophet(
            growth='linear',
            seasonality_mode=seasonality_mode,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model_full.fit(df_ts)
        future_dates = model_full.make_future_dataframe(periods=forecast_horizon, freq='D')
        forecast_full = model_full.predict(future_dates)
        
        # Post-process future values
        forecast_full['yhat'] = forecast_full['yhat'].apply(lambda x: max(x, 0))
        forecast_full['yhat_lower'] = forecast_full['yhat_lower'].apply(lambda x: max(x, 0))
        forecast_full['yhat_upper'] = forecast_full['yhat_upper'].apply(lambda x: max(x, 0))

    # Tabs for Forecast and Metrics
    tab1, tab2, tab3 = st.tabs(["🔮 Pronóstico Futuro", "📊 Evaluación en Test", "🧩 Componentes del Modelo"])
    
    # ------------------ TAB 1: FORECAST FUTURO ------------------
    with tab1:
        st.subheader(f"Pronóstico de Ventas para los Próximos {forecast_horizon} Días")
        
        # Display future table
        futuras_pred = forecast_full.tail(forecast_horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        futuras_pred.rename(columns={
            'ds': 'Fecha',
            'yhat': 'Ingreso Estimado (S/)',
            'yhat_lower': 'Límite Mínimo (S/)',
            'yhat_upper': 'Límite Máximo (S/)'
        }, inplace=True)
        
        st.write("#### Tabla de Proyecciones de Ingresos")
        st.dataframe(futuras_pred.style.format({
            'Ingreso Estimado (S/)': 'S/ {:.2f}',
            'Límite Mínimo (S/)': 'S/ {:.2f}',
            'Límite Máximo (S/)': 'S/ {:.2f}'
        }))
        
        # Plot Future Forecast
        st.write("#### Gráfico del Pronóstico Completo")
        fig_f, ax = plt.subplots(figsize=(10, 4.5))
        # Historical
        ax.plot(df_ts['ds'], df_ts['y'], label='Histórico Real', color='black', alpha=0.6)
        # Forecasted future
        ax.plot(futuras_pred['Fecha'], futuras_pred['Ingreso Estimado (S/)'], label='Pronóstico Prophet', color='#e056fd', linewidth=2.5, marker='o')
        ax.fill_between(
            futuras_pred['Fecha'], 
            futuras_pred['Límite Mínimo (S/)'], 
            futuras_pred['Límite Máximo (S/)'], 
            color='#e056fd', 
            alpha=0.15, 
            label='Intervalo de Confianza'
        )
        ax.set_title("Pronóstico Futuro de Flujo de Caja")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ingresos Diarios (S/)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_f)

    # ------------------ TAB 2: EVALUACION TEST ------------------
    with tab2:
        st.subheader("Evaluación de Modelos: Prophet vs. Media Móvil")
        st.write(f"Evaluados sobre los últimos **{test_days} días** del historial.")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.metric("Prophet RMSE", f"S/ {rmse_prophet:.2f}")
            st.metric("Prophet MAPE", f"{mape_prophet:.2%}")
        with col_e2:
            st.metric("Media Móvil RMSE", f"S/ {rmse_ma:.2f}")
            st.metric("Media Móvil MAPE", f"{mape_ma:.2%}")
            
        st.write("#### Gráfico Comparativo en Ventana de Test")
        fig_test, ax_t = plt.subplots(figsize=(10, 4.5))
        ax_t.plot(performance_df['ds'], performance_df['y_true'], label='Valores Reales', color='blue', marker='o')
        ax_t.plot(performance_df['ds'], performance_df['yhat'], label='Predicciones Prophet', color='red', linestyle='--', marker='x')
        ax_t.plot(test_df['ds'], ma_test_predictions, label='Predicciones Media Móvil', color='green', linestyle=':', marker='^')
        ax_t.set_title("Valores Reales vs. Predicciones (Conjunto de Prueba)")
        ax_t.set_xlabel("Fecha")
        ax_t.set_ylabel("Ingresos Totales (S/)")
        ax_t.legend()
        ax_t.grid(True)
        st.pyplot(fig_test)
        
        st.write(
            """
            * **RMSE (Root Mean Squared Error):** Mide la magnitud de los errores en las mismas unidades que la variable (Soles).
            * **MAPE (Mean Absolute Percentage Error):** Expresa el error en porcentaje. Un MAPE menor indica mayor precisión.
            """
        )

    # ------------------ TAB 3: COMPONENTES ------------------
    with tab3:
        st.subheader("Componentes del Modelo Prophet")
        st.write("Prophet descompone la serie temporal en tendencia global y estacionalidad semanal.")
        
        try:
            fig_components = model_full.plot_components(forecast_full)
            st.pyplot(fig_components)
        except Exception as e:
            st.warning(f"No se pudieron graficar los componentes: {e}")
