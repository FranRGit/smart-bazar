from pathlib import Path
from dateutil import parser
import pickle
import holidays

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - fallback for lean environments
    class _JoblibFallback:
        @staticmethod
        def load(path):
            with open(path, "rb") as handle:
                return pickle.load(handle)

    joblib = _JoblibFallback()

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .data_loader import load_ventas
except Exception:  # pragma: no cover - fallback for direct execution
    from data_loader import load_ventas


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_prophet_model.joblib"
MA_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ma_model_params.joblib"


def _inject_styles() -> None:
    """
    Inyecta estilos CSS personalizados en la aplicación Streamlit para mejorar la apariencia del panel de pronóstico.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .forecast-root, .forecast-root * {
            box-sizing: border-box;
            font-family: 'Inter', sans-serif;
        }

        .forecast-root {
            background: linear-gradient(180deg, #ffffff 0%, #f3f4f6 100%);
            color: #111111;
        }

        .forecast-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .forecast-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0;
            color: #0f172a;
            letter-spacing: -0.03em;
        }

        .forecast-subtitle {
            margin: 0.35rem 0 0 0;
            color: #6b7280;
            font-size: 0.92rem;
        }

        .sidebar-card, .panel-card, .metric-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
        }

        .sidebar-card {
            padding: 1.1rem;
            min-height: 680px;
        }

        .panel-card {
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.2rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            padding: 1rem 0.95rem;
            min-height: 124px;
            margin-bottom: 1.2rem;
        }

        /* ── Flip Card Styles ── */
        .flip-card {
            background-color: transparent;
            width: 100%;
            height: 135px;
            perspective: 1000px;
            margin-bottom: 1.2rem;
        }

        .flip-card-inner {
            position: relative;
            width: 100%;
            height: 100%;
            transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            transform-style: preserve-3d;
        }

        .flip-card:hover .flip-card-inner {
            transform: rotateY(180deg);
        }

        .flip-card-front, .flip-card-back {
            position: absolute;
            width: 100%;
            height: 100%;
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
            border-radius: 18px;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-sizing: border-box;
        }

        .flip-card-front {
            background: rgba(255, 255, 255, 0.95);
            color: #111827;
            padding: 1rem 1.1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: left;
        }

        .flip-card-back {
            background: #0f172a;
            color: #ffffff;
            transform: rotateY(180deg);
            padding: 1rem 1.15rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            text-align: left;
        }

        .back-title {
            font-size: 0.64rem;
            font-weight: 800;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .back-body {
            font-size: 0.72rem;
            line-height: 1.35;
            color: #f1f5f9;
        }
        
        .back-body b {
            color: #f8fafc;
        }

        .metric-kicker {
            display: block;
            color: #6b7280;
            font-size: 0.64rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.55rem;
        }

        .metric-value {
            display: block;
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.05;
            color: #111827;
            margin-bottom: 0.4rem;
        }

        .metric-note {
            display: block;
            font-size: 0.72rem;
            line-height: 1.35;
            color: #667085;
            font-weight: 600;
        }

        .metric-positive {
            color: #111827;
        }

        .metric-accent {
            color: #374151;
        }

        .forecast-shell {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 1rem;
            align-items: start;
        }

        .sidebar-label {
            font-size: 0.92rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.1rem;
        }

        .sidebar-subtitle {
            font-size: 0.78rem;
            color: #6b7280;
            margin-bottom: 1rem;
        }

        .sidebar-item {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.7rem 0.8rem;
            border-radius: 12px;
            margin-bottom: 0.35rem;
            font-size: 0.82rem;
            font-weight: 700;
            color: #374151;
            background: transparent;
        }

        .sidebar-item.active {
            background: #000000;
            color: #ffffff;
        }

        .sidebar-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(15,23,42,0.12), transparent);
            margin: 1rem 0;
        }

        .chart-shell {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.2rem;
        }

        .section-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.25rem;
        }

        .section-caption {
            font-size: 0.78rem;
            color: #6b7280;
            margin-bottom: 0.85rem;
        }

        .footer-note {
            font-size: 0.75rem;
            color: #6b7280;
        }

        div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            color: #4b5563 !important;
        }

        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-testid="stSlider"] [data-baseweb="slider"] {
            border-radius: 12px !important;
        }

        /* Estilo para el slider negro */
        div[data-testid="stSlider"] [role="slider"] {
            background-color: #000000 !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 0 0 2px #000000 !important;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            filter: grayscale(100%) brightness(30%) contrast(150%) !important;
        }

        /* Quitar texto blanco de los extremos de los sliders */
        div[data-testid="stSlider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _currency(value: float) -> str:
    """
    Formatea un valor numérico como moneda peruana (S/), con separadores de miles y sin decimales.
    """
    return f"S/ {value:,.0f}".replace(",", ".")


def _currency_2(value: float) -> str:
    """
    Formatea un valor numérico como moneda peruana (S/), con separadores de miles y dos decimales.
    """
    return f"S/ {value:,.2f}".replace(",", ".")


def _apply_chart_style(fig, ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """
    Aplica un estilo visual consistente a los gráficos de Matplotlib.
    """
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#64748b", labelsize=8)
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    for spine in ax.spines.values():
        spine.set_color("#e5e7eb")
    ax.grid(True, alpha=0.35, color="#e5e7eb", linestyle="--", linewidth=0.7)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color="#111827", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, fontweight="semibold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, fontweight="semibold")
    fig.tight_layout()


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """
    Calcula las métricas RMSE y MAPE de manera segura, manejando la posible división por cero en el cálculo de MAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return rmse, mape

def parsear_fechas_cronologicas(fechas):
    """
    Toma una lista de fechas en formato string y devuelve una lista de fechas en formato datetime.
    """
    fechas_limpias = []
    fecha_anterior = pd.Timestamp.min

    for fecha_str in fechas:
        if pd.isna(fecha_str):
            fechas_limpias.append(pd.NaT)
            continue

        try:
            d1 = parser.parse(str(fecha_str), dayfirst=True)
            d2 = parser.parse(str(fecha_str), dayfirst=False)

            if d1 == d2:
                fecha_elegida = d1
            elif d1 >= fecha_anterior and d2 < fecha_anterior:
                fecha_elegida = d1
            elif d2 >= fecha_anterior and d1 < fecha_anterior:
                fecha_elegida = d2
            else:
                fecha_elegida = d1 if (d1 - fecha_anterior) < (d2 - fecha_anterior) else d2

            fechas_limpias.append(fecha_elegida)
            fecha_anterior = fecha_elegida
        except Exception:
            fechas_limpias.append(pd.NaT)

    return fechas_limpias

@st.cache_data(show_spinner=False)
def _load_sales_data() -> pd.DataFrame:
    """
    Carga y prepara los datos de ventas desde el archivo ventas.csv.
    Devuelve un DataFrame con las columnas 'Fecha', 'Total' y 'Fecha_Diaria'.
    """
    df = load_ventas(limpio=True)
    df = df.copy()
    if "Fecha" not in df.columns or "Total" not in df.columns:
        raise ValueError("El archivo ventas.csv debe incluir las columnas 'Fecha' y 'Total'.")
    df["Fecha"] = pd.to_datetime(parsear_fechas_cronologicas(df["Fecha"]))
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Total"])
    cutoff_date = pd.Timestamp.now().normalize()
    df = df[df["Fecha"].dt.normalize() <= cutoff_date].copy()
    df["Fecha_Diaria"] = df["Fecha"].dt.normalize()
    return df


def _build_school_dates(years: list[int]) -> pd.DatetimeIndex:
    """
    Construye un índice de fechas que representan los periodos escolares para los años especificados.
    """
    school_ranges = []
    for year in years:
        school_ranges.append(pd.date_range(f"{year}-03-01", periods=30, freq="D"))
        school_ranges.append(pd.date_range(f"{year}-07-20", periods=15, freq="D"))
    if not school_ranges:
        return pd.DatetimeIndex([])
    combined = school_ranges[0]
    for date_range in school_ranges[1:]:
        combined = combined.union(date_range)
    return combined


def _build_regressors(df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """
    Construye un DataFrame de regresores para el modelo de pronóstico. 
    Donde:
    - 'ds' es la fecha.
    - 'is_nat' indica si la fecha es un feriado nacional.
    - 'is_school' indica si la fecha cae dentro de un periodo escolar.
    """
    try:
        pe_holidays = holidays.country_holidays("PE", years=years)
        hols_df = pd.DataFrame({"ds": pd.to_datetime(list(pe_holidays.keys())), "is_nat": 1})
    except Exception:
        hols_df = pd.DataFrame(columns=["ds", "is_nat"])

    school_dates = _build_school_dates(years)
    school_df = pd.DataFrame({"ds": school_dates, "is_school": 1})

    reg_df = pd.DataFrame({"ds": pd.to_datetime(df["ds"])}).sort_values("ds")

    if not hols_df.empty:
        hols_weekly = (
            hols_df.set_index("ds")
            .resample("W-SUN")
            .max()
            .fillna(0)
            .reset_index()
        )
        reg_df = reg_df.merge(hols_weekly, on="ds", how="left")
    else:
        reg_df["is_nat"] = 0

    if not school_df.empty:
        school_weekly = (
            school_df.set_index("ds")
            .resample("W-SUN")
            .max()
            .fillna(0)
            .reset_index()
        )
        reg_df = reg_df.merge(school_weekly, on="ds", how="left")
    else:
        reg_df["is_school"] = 0

    reg_df = reg_df.fillna(0)
    if "is_nat" not in reg_df.columns:
        reg_df["is_nat"] = 0
    if "is_school" not in reg_df.columns:
        reg_df["is_school"] = 0

    # ==========================================================================
    # === EVALUACIÓN DOCENTE: INYECCIÓN DE REGRESORES EXÓGENOS ===
    # Teoría:
    #   - Prophet es un modelo aditivo. Los regresores exógenos (feriados 'is_nat',
    #     periodo escolar 'is_school') se suman a la ecuación de tendencia y
    #     estacionalidad. Permiten absorber picos o caídas recurrentes que no
    #     se explican solo con el tiempo.
    #
    # Código en Vivo (Crear y anexar 'is_quincena' para salarios los 15 y 30):
    #   # 1. Crear regresor (dentro de esta función o antes de agrupar):
    #   # reg_df['is_quincena'] = reg_df['ds'].dt.day.isin([15, 30]).astype(int)
    #   #
    #   # 2. Agregar al modelo final en el notebook/código de entrenamiento:
    #   # final_model.add_regressor('is_quincena')
    #   # final_model.fit(df_full_p)
    # ==========================================================================

    return reg_df[["ds", "is_nat", "is_school"]]


@st.cache_data(show_spinner=False)
def _prepare_weekly_data() -> pd.DataFrame:
    """
    Prepara los datos de ventas para el modelo de pronóstico, agregando las ventas diarias y semanales,
    y construyendo los regresores necesarios.
    """
    df_raw = _load_sales_data()
    ingreso_diario = df_raw.groupby("Fecha_Diaria")["Total"].sum().sort_index().to_frame()
    full_range = pd.date_range(start=ingreso_diario.index.min(), end=ingreso_diario.index.max(), freq="D")
    ingreso_diario = ingreso_diario.reindex(full_range, fill_value=0)
    ingreso_diario.index.name = "Fecha_Diaria"

    weekly_df = ingreso_diario.resample("W-SUN").sum().reset_index()
    weekly_df.columns = ["ds", "y"]
    weekly_df = weekly_df.sort_values("ds").reset_index(drop=True)

    start_year = int(weekly_df["ds"].dt.year.min())
    end_year = int(weekly_df["ds"].dt.year.max()) + 2
    weekly_df = weekly_df.merge(_build_regressors(weekly_df[["ds"]], list(range(start_year, end_year + 1))), on="ds", how="left")
    weekly_df[["is_nat", "is_school"]] = weekly_df[["is_nat", "is_school"]].fillna(0).astype(int)

    # ==============================================================================
    # === EVALUACIÓN DOCENTE: ANÁLISIS DE ESTACIONARIEDAD ===
    # Teoría:
    #   - El Test Aumentado de Dickey-Fuller (ADF) evalúa la hipótesis nula (H0) de
    #     que la serie temporal tiene una raíz unitaria (es no estacionaria).
    #     Si el p-valor es < 0.05, rechazamos H0 y la serie se considera estacionaria.
    #   - Si la serie no es estacionaria, se aplica "diferenciación" (restar el valor
    #     actual menos el anterior: y_t - y_t-1) para estabilizar la media.
    #
    # Código en Vivo (Descomentar para probar en vivo con el docente):
    #   # from statsmodels.tsa.stattools import adfuller
    #   # result = adfuller(weekly_df['y'])
    #   # print(f"Estadístico ADF: {result[0]}")
    #   # print(f"p-valor: {result[1]}")
    #   # if result[1] < 0.05:
    #   #     print("La serie es Estacionaria (Rechaza H0)")
    #   # else:
    #   #     print("La serie NO es Estacionaria (Requiere diferenciar: weekly_df['y'].diff().dropna())")
    # ==============================================================================

    return weekly_df


@st.cache_resource(show_spinner=False)
def _load_model():
    """
    Carga el modelo Prophet desde un archivo joblib.
    """
    # ==============================================================================
    # === EVALUACIÓN DOCENTE: FLEXIBILIDAD DE TENDENCIA (OVERFITTING) ===
    # Teoría:
    #   - 'changepoint_prior_scale' regula la rigidez de la tendencia en Prophet.
    #     Valor por defecto: 0.05.
    #     Si se eleva (ej. a 0.5 o 1.0), el modelo se vuelve extremadamente flexible,
    #     siguiendo el ruido de los datos (overfitting), debilitando la estacionalidad.
    #     Si se baja (ej. a 0.001), la tendencia se vuelve casi una línea recta (underfitting).
    #
    # Código en Vivo (Forzar sobreajuste al inicializar m_prophet):
    #   # from prophet import Prophet
    #   # m_overfit = Prophet(changepoint_prior_scale=0.9, growth='linear')
    #   # m_overfit.fit(df_train)
    #   # m_overfit.plot_components(m_overfit.predict(df_train))
    # ==============================================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    try:
        return joblib.load(MODEL_PATH)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "El modelo serializado depende de 'prophet'. Instala esa libreria en el entorno "
            "antes de ejecutar el panel."
        ) from exc


@st.cache_resource(show_spinner=False)
def _load_ma_params() -> dict:
    """
    Carga los parámetros del modelo de media móvil desde un archivo joblib.
    """
    if not MA_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo de media móvil en {MA_MODEL_PATH}")
    try:
        return joblib.load(MA_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError("No se pudo cargar el modelo de media móvil exportado.") from exc


def _fit_forecast_frame(model, weekly_df: pd.DataFrame, horizon_weeks: int):
    """
    Ajusta el marco de pronóstico utilizando el modelo Prophet y los datos semanales.
    Devuelve dos DataFrames: uno con el pronóstico histórico y otro con el pronóstico futuro.
    """
    years = list(range(int(weekly_df["ds"].dt.year.min()), int(weekly_df["ds"].dt.year.max()) + 3))
    history_features = _build_regressors(weekly_df[["ds"]], years)
    history_features = history_features.merge(weekly_df[["ds", "y"]], on="ds", how="left")

    forecast_hist = model.predict(history_features[["ds", "is_nat", "is_school"]])
    forecast_hist = forecast_hist[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # ==========================================================================
    # === EVALUACIÓN DOCENTE: REVERSIÓN LOGARÍTMICA (SESGO DE EXPONENCIACIÓN) ===
    # Teoría:
    #   - Al entrenar con log(1+y) y predecir yhat, aplicar np.expm1(yhat) introduce
    #     un sesgo sistemático hacia abajo debido a la desigualdad de Jensen:
    #     E[exp(X)] != exp(E[X]).
    #   - Corrección (Smearing de Duan): Se multiplica por un factor empírico, o
    #     se suma la mitad de la varianza del error residual (sigma^2 / 2) antes de
    #     exponenciar: yhat_corrected = exp(yhat + 0.5 * resid_variance) - 1.
    #
    # Código en Vivo (Corrección empírica de Smearing de Duan):
    #   # 1. Calcular la varianza de los residuos en escala logarítmica:
    #   # y_log_real = np.log1p(weekly_df['y'])
    #   # resid_variance = np.var(y_log_real - forecast_hist['yhat'])
    #   # 2. Aplicar corrección antes de expm1:
    #   # forecast_hist["yhat_original"] = np.expm1(forecast_hist["yhat"] + 0.5 * resid_variance).clip(lower=0)
    # ==========================================================================

    forecast_hist["yhat_original"] = np.expm1(forecast_hist["yhat"]).clip(lower=0)
    forecast_hist["yhat_lower_original"] = np.expm1(forecast_hist["yhat_lower"]).clip(lower=0)
    forecast_hist["yhat_upper_original"] = np.expm1(forecast_hist["yhat_upper"]).clip(lower=0)

    future = model.make_future_dataframe(periods=horizon_weeks, freq="W-SUN")
    future = future.drop_duplicates("ds").sort_values("ds").reset_index(drop=True)
    future = future.merge(_build_regressors(future[["ds"]], years), on="ds", how="left").fillna(0)
    forecast_future = model.predict(future[["ds", "is_nat", "is_school"]])
    forecast_future = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_future["yhat_original"] = np.expm1(forecast_future["yhat"]).clip(lower=0)
    forecast_future["yhat_lower_original"] = np.expm1(forecast_future["yhat_lower"]).clip(lower=0)
    forecast_future["yhat_upper_original"] = np.expm1(forecast_future["yhat_upper"]).clip(lower=0)
    return forecast_hist, forecast_future


def _fit_moving_average_frame(weekly_df: pd.DataFrame, horizon_weeks: int, window: int, last_mean: float):
    """
    Ajusta el marco de pronóstico utilizando un modelo de media móvil.
    Devuelve dos DataFrames: uno con el pronóstico histórico y otro con el pronóstico futuro.
    """
    history = weekly_df[["ds", "y"]].copy()
    history["ma_pred"] = history["y"].rolling(window=window, min_periods=1).mean().shift(1)
    history["ma_pred"] = history["ma_pred"].fillna(last_mean)

    # ==============================================================================
    # === EVALUACIÓN DOCENTE: COMPARACIÓN CON BASELINE NAÏVE ===
    # Teoría:
    #   - Una Media Móvil (3 semanas) suaviza la tendencia reciente pero tiene lag.
    #   - Un baseline "Naïve" simple predice que el valor de esta semana es igual
    #     al de la semana pasada (shift(1)). Sirve como el benchmark mínimo.
    #
    # Código en Vivo (Modelo Naïve Estacional/Simple y cálculo de su MAPE):
    #   # from sklearn.metrics import mean_absolute_percentage_error
    #   # naive_pred = weekly_df['y'].shift(1)
    #   # val_y = weekly_df['y'].iloc[1:]
    #   # val_pred = naive_pred.iloc[1:]
    #   # mape_naive = mean_absolute_percentage_error(val_y, val_pred) * 100
    #   # print(f"MAPE Naïve (shift 1): {mape_naive:.2f}%")
    # ==============================================================================

    future_dates = pd.date_range(weekly_df["ds"].max() + pd.Timedelta(days=7), periods=horizon_weeks, freq="W-SUN")
    future = pd.DataFrame({
        "ds": future_dates,
        "ma_pred": np.full(horizon_weeks, float(last_mean)),
    })
    return history, future


def _render_metric_card(
    title: str,
    value: str,
    note: str,
    back_error_calc: str,
    back_estimation: str,
    accent: bool = False,
    positive: bool = False
) -> None:
    """
    Renderiza una tarjeta de métrica con efecto flip en 3D en la interfaz de usuario de Streamlit.
    """
    value_class = "metric-value"
    note_class = "metric-note"
    value = value.replace("\n", "<br>")
    note = note.replace("\n", "<br>")
    if accent:
        value_class += " metric-accent"
    if positive:
        value_class += " metric-positive"
        note_class += " metric-positive"
        
    st.markdown(
        f"""
        <div class='flip-card'>
            <div class='flip-card-inner'>
                <div class='flip-card-front'>
                    <span class='metric-kicker'>{title}</span>
                    <span class='{value_class}'>{value}</span>
                    <span class='{note_class}'>{note}</span>
                </div>
                <div class='flip-card-back'>
                    <div class='back-title'>Métrica y Estimación</div>
                    <div class='back-body'>
                        <b>Cálculo:</b> {back_error_calc}<br>
                        <div style='height: 0.35rem;'></div>
                        <b>Estimación:</b> {back_estimation}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_backtest_frame(history_df: pd.DataFrame, prophet_hist: pd.DataFrame, ma_history: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Construye un DataFrame de backtesting que contiene las métricas de error (MAPE y RMSE) para los modelos Prophet y Media Móvil.
    """
    merged = history_df[["ds", "y"]].merge(prophet_hist[["ds", "yhat_original"]], on="ds", how="left")
    merged = merged.merge(ma_history[["ds", "ma_pred"]], on="ds", how="left")
    recent = merged.tail(window).copy().reset_index(drop=True)
    denom = np.where(recent["y"] == 0, 1.0, recent["y"])
    recent["prophet_mape"] = np.abs((recent["y"] - recent["yhat_original"]) / denom) * 100.0
    recent["ma_mape"] = np.abs((recent["y"] - recent["ma_pred"]) / denom) * 100.0
    recent["prophet_rmse"] = np.sqrt((recent["y"] - recent["yhat_original"]) ** 2)
    recent["ma_rmse"] = np.sqrt((recent["y"] - recent["ma_pred"]) ** 2)
    recent["semana"] = [f"Sem {i + 1}" for i in range(len(recent))]
    return recent


def show_panel():
    """
    Panel 5 - Pronostico de Ingresos con Prophet.
    """

    _inject_styles()

    # Cargar el modelo Prophet y los datos semanales
    try:
        model = _load_model()
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo Prophet: {exc}")
        return

    # Preparar los datos semanales desde ventas.csv
    try:
        weekly_df = _prepare_weekly_data()
    except Exception as exc:
        st.error(f"No se pudo preparar la serie semanal desde ventas.csv: {exc}")
        return

    # Cargar los parámetros del modelo de media móvil
    try:
        ma_params = _load_ma_params()
    except Exception as exc:
        st.error(f"No se pudo cargar el modelo de media móvil: {exc}")
        return

    # Renderizar la interfaz de usuario del panel de pronóstico
    # Renderizar la interfaz de usuario del panel de pronóstico
    # Título principal del panel (este queda fijo arriba de los tabs)
    st.markdown(
        "<div class='forecast-header'>"
        "<div>"
        "<h1 class='forecast-title'>Panel de Series Temporales</h1>"
        "<p class='forecast-subtitle'>Gestion avanzada de ventas y pronostico analitico con Prophet.</p>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "🔮 Pronóstico y Proyecciones",
        "📊 Evaluación de Modelos",
        "💰 Resumen de Ventas"
    ])

    # ── TAB 1: Pronóstico y Proyecciones ──────────────────────────────────
    with tab1:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Ajustes de pronóstico</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-caption'>Ajusta los parámetros para volver a ejecutar el modelo de series temporales.</div>", unsafe_allow_html=True)

        horizon_options = [4, 8, 12, 16, 24, 52]
        horizon_weeks = st.select_slider(
            "Horizonte de prediccion",
            options=horizon_options,
            value=4,
            format_func=lambda value: f"Proximas {value} semanas",
            key="horizon_weeks"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # CÓMPUTOS DEL MODELO (se ejecutan en Tab 1 y quedan disponibles para el resto de pestañas)
        forecast_hist, forecast_future = _fit_forecast_frame(model, weekly_df, horizon_weeks)
        ma_history, ma_future = _fit_moving_average_frame(
            weekly_df,
            horizon_weeks,
            int(ma_params.get("window", 3)),
            float(ma_params.get("last_mean", weekly_df["y"].tail(int(ma_params.get("window", 3))).mean())),
        )
        future_rows = forecast_future[forecast_future["ds"] > weekly_df["ds"].max()].copy().reset_index(drop=True)
        if future_rows.empty:
            future_rows = forecast_future.tail(1).copy().reset_index(drop=True)
        
        backtest_window = min(4, len(weekly_df))
        error_df = _build_backtest_frame(weekly_df, forecast_hist, ma_history, window=backtest_window)

        last_hist = weekly_df.iloc[-1]
        next_forecast = future_rows.iloc[0]
        total_history = float(weekly_df["y"].sum())
        avg_weekly = float(weekly_df["y"].mean())

        recent_eval = weekly_df[["ds", "y"]].merge(forecast_hist[["ds", "yhat_original"]], on="ds", how="left")
        recent_eval["ma_pred"] = recent_eval["y"].rolling(window=3, min_periods=1).mean().shift(1)
        recent_eval["ma_pred"] = recent_eval["ma_pred"].fillna(recent_eval["y"].expanding().mean())
        recent_slice = recent_eval.tail(backtest_window)

        prophet_rmse, prophet_mape = _safe_metrics(recent_slice["y"].to_numpy(), recent_slice["yhat_original"].to_numpy())
        ma_rmse, ma_mape = _safe_metrics(recent_slice["y"].to_numpy(), recent_slice["ma_pred"].to_numpy())
        
        if prophet_mape < ma_mape:
            winner = "Prophet"
            winner_note = "Menor error porcentual (MAPE)"
        elif ma_mape < prophet_mape:
            winner = "Media Movil"
            winner_note = f"Menor MAPE. Ventana {int(ma_params.get('window', 3))} y último promedio"
        else:
            winner = "Empate"
            winner_note = "Ambos modelos muestran el mismo MAPE"



        # Gráfico: Evolución de ventas semanales
        st.markdown(
            "<div class='chart-shell'>"
            "<div class='section-title'>Evolucion de Ventas Semanales</div>"
            "<div class='section-caption'>Historial real, pronostico Prophet, intervalo de confianza y media movil. Mueve el cursor para ver detalles.</div>",
            unsafe_allow_html=True,
        )
        
        fig_main = go.Figure()
        
        # 1. Ventas Reales
        fig_main.add_trace(go.Scatter(
            x=weekly_df["ds"],
            y=weekly_df["y"],
            mode="lines+markers",
            name="Ventas Reales",
            line=dict(color="#000000", width=2.2),
            marker=dict(size=4, color="#000000"),
            hovertemplate="Ventas Reales: S/ %{y:,.2f}<extra></extra>"
        ))
        
        # 2. Prophet Ajuste
        fig_main.add_trace(go.Scatter(
            x=forecast_hist["ds"],
            y=forecast_hist["yhat_original"],
            mode="lines",
            name="Prophet Ajuste",
            line=dict(color="#737373", width=1.8, dash="dot"),
            hovertemplate="Prophet Ajuste: S/ %{y:,.2f}<extra></extra>"
        ))
        
        # 3. Prophet Futuro
        prophet_future_plot = forecast_future[forecast_future["ds"] >= weekly_df["ds"].max()]
        fig_main.add_trace(go.Scatter(
            x=prophet_future_plot["ds"],
            y=prophet_future_plot["yhat_original"],
            mode="lines+markers",
            name="Prophet Futuro",
            line=dict(color="#1f2937", width=2.2, dash="dash"),
            marker=dict(size=5, color="#1f2937"),
            hovertemplate="Prophet Futuro: S/ %{y:,.2f}<extra></extra>"
        ))
        
        # 4. Intervalo de Confianza (fill)
        fig_main.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_upper_original"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        fig_main.add_trace(go.Scatter(
            x=forecast_future["ds"],
            y=forecast_future["yhat_lower_original"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(15, 23, 42, 0.08)",
            line=dict(width=0),
            name="Intervalo de Confianza",
            hoverinfo="skip"
        ))
        
        # 5. Media Móvil
        fig_main.add_trace(go.Scatter(
            x=ma_history["ds"],
            y=ma_history["ma_pred"],
            mode="lines",
            name="Media Movil",
            line=dict(color="#a3a3a3", width=1.5),
            hovertemplate="Media Movil: S/ %{y:,.2f}<extra></extra>"
        ))
        
        fig_main.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=10, b=10),
            height=340,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=9)
            ),
            xaxis=dict(
                gridcolor="#f1f5f9",
                tickfont=dict(size=9, color="#64748b")
            ),
            yaxis=dict(
                gridcolor="#f1f5f9",
                tickfont=dict(size=9, color="#64748b"),
                tickprefix="S/ "
            )
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Predicciones futuras
        st.markdown(
            "<div class='panel-card'>"
            "<div class='section-title'>Predicciones Futuras Proyectadas</div>"
            "<div class='section-caption'>Detalle del pronóstico semana a semana según el horizonte seleccionado.</div>",
            unsafe_allow_html=True,
        )
        future_table = future_rows[["ds", "yhat_original", "yhat_lower_original", "yhat_upper_original"]].copy()
        future_table = future_table.head(horizon_weeks).copy()
        future_table = future_table.rename(columns={
            "ds": "Fecha",
            "yhat_original": "Pronostico Prophet",
            "yhat_lower_original": "Pronostico minimo",
            "yhat_upper_original": "Pronostico maximo",
        })
        future_table["Fecha"] = future_table["Fecha"].dt.strftime("%d/%m/%Y")
        st.dataframe(
            future_table.style.format({
                "Pronostico Prophet": "S/ {:,.2f}",
                "Pronostico minimo": "S/ {:,.2f}",
                "Pronostico maximo": "S/ {:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Breve descripción del panel en la parte inferior
        st.markdown(
            """
            <div class='panel-card' style='background: #f8fafc; border-left: 4px solid #0f172a;'>
                <div class='section-title'>💡 Acerca del Panel de Pronóstico</div>
                <p style='font-size: 0.82rem; color: #475569; line-height: 1.5; margin: 0;'>
                    Este panel implementa un modelo de series temporales avanzado basado en <b>Prophet</b> para pronosticar los ingresos semanales de la tienda, 
                    incorporando variables exógenas como días festivos nacionales y calendarios de campañas escolares. Adicionalmente, evalúa y compara el desempeño 
                    del modelo predictivo contra una <b>Media Móvil (Baseline)</b> para proporcionar una referencia simple de tendencia histórica.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── TAB 2: Evaluación de Modelos ─────────────────────────────────────
    with tab2:
        # 1. Horizonte de predicción
        st.markdown(
            f"""
            <div class='panel-card' style='background: #fdf8f8; border: 1px solid rgba(15,23,42,0.08);'>
                <div class='section-title'>⚙️ Configuración del Horizonte de Predicción</div>
                <div style='font-size: 1.15rem; font-weight: 800; color: #111827; margin-top: 0.4rem;'>
                    Horizonte seleccionado: <span style='color: #8b5cf6;'>Próximas {horizon_weeks} semanas</span>
                </div>
                <div style='font-size: 0.9rem; color: #6b7280; margin-top: 0.2rem;'>
                    Fecha de arranque de proyección: <b>{next_forecast['ds'].strftime('%d/%m/%Y')}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gráfica comparativa de las últimas 4 semanas (Prophet vs. Media Móvil vs. Real)
        st.markdown(
            "<div class='chart-shell'>"
            "<div class='section-title'>Comparativa de Modelos: Ventana de Prueba</div>"
            "<div class='section-caption'>Desempeño real vs. predicciones de Prophet y Media Móvil en las últimas 4 semanas de prueba. Mueve el cursor para comparar.</div>",
            unsafe_allow_html=True,
        )

        fig_comp = go.Figure()
        
        slice_4w = recent_eval.tail(4).copy()
        
        # Real
        fig_comp.add_trace(go.Scatter(
            x=slice_4w["ds"],
            y=slice_4w["y"],
            mode="lines+markers",
            name="Real",
            line=dict(color="#000000", width=2.2),
            marker=dict(size=7, symbol="circle", color="#000000"),
            hovertemplate="Real: S/ %{y:,.2f}<extra></extra>"
        ))
        
        # Prophet
        fig_comp.add_trace(go.Scatter(
            x=slice_4w["ds"],
            y=slice_4w["yhat_original"],
            mode="lines+markers",
            name="Prophet",
            line=dict(color="#10b981", width=2, dash="dash"),
            marker=dict(size=7, symbol="square", color="#10b981"),
            hovertemplate="Prophet: S/ %{y:,.2f}<extra></extra>"
        ))
        
        # Media Móvil
        fig_comp.add_trace(go.Scatter(
            x=slice_4w["ds"],
            y=slice_4w["ma_pred"],
            mode="lines+markers",
            name="Media Móvil",
            line=dict(color="#3b82f6", width=2, dash="dash"),
            marker=dict(size=7, symbol="triangle-up", color="#3b82f6"),
            hovertemplate="Media Móvil: S/ %{y:,.2f}<extra></extra>"
        ))
        
        fig_comp.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=9)
            ),
            xaxis=dict(
                gridcolor="#f1f5f9",
                tickfont=dict(size=9, color="#64748b"),
                tickvals=slice_4w["ds"],
                ticktext=[d.strftime('%Y-%m-%d') for d in slice_4w["ds"]]
            ),
            yaxis=dict(
                gridcolor="#f1f5f9",
                tickfont=dict(size=9, color="#64748b"),
                tickprefix="S/ "
            )
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 2. Tarjetas métricas de errores y modelo ganador
        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            _render_metric_card(
                "Error Prophet",
                f"RMSE: {prophet_rmse:,.1f}\nMAPE: {prophet_mape:.1f}%".replace(",", "."),
                "Calculado en ventana de prueba (test)",
                "Diferencia cuadrática (RMSE) y error porcentual absoluto medio (MAPE) de Prophet vs. Real.",
                "Modelado bayesiano por Prophet ajustando estacionalidades, feriados peruanos y calendario escolar."
            )
        with col_met2:
            _render_metric_card(
                "Error Media Móvil",
                f"RMSE: {ma_rmse:,.1f}\nMAPE: {ma_mape:.1f}%".replace(",", "."),
                f"Baseline (Ventana {int(ma_params.get('window', 3))})",
                "Desviación del promedio móvil de 3 semanas históricas respecto a las ventas reales.",
                "Predicción simple extrapolada usando la media aritmética de la ventana de tiempo del modelo base."
            )
        with col_met3:
            _render_metric_card(
                "Ganador Dinámico",
                winner,
                winner_note,
                "Comparación del error porcentual absoluto medio (MAPE) de ambos modelos en la ventana de prueba.",
                "Selección automática del algoritmo con el menor MAPE de backtesting.",
                accent=True,
                positive=True
            )

        # 3. Gráficos de barra comparativos de RMSE y MAPE
        st.markdown(
            "<div class='panel-card'>"
            "<div class='section-title'>Comparativa de Errores por Semana (Backtesting)</div>"
            "<div class='section-caption'>Análisis del error de pronóstico semanal promedio (MAPE) y la desviación cuadrática (RMSE).</div>",
            unsafe_allow_html=True,
        )

        err_left, err_right = st.columns(2)
        with err_left:
            fig_err_mape, ax_err_mape = plt.subplots(figsize=(4.3, 4.2), dpi=100)
            x = np.arange(len(error_df))
            width = 0.33
            ax_err_mape.bar(x - width / 2, error_df["prophet_mape"], width=width, color="#000000", label="Prophet")
            ax_err_mape.bar(x + width / 2, error_df["ma_mape"], width=width, color="#d4d4d8", label="Media Movil")
            ax_err_mape.set_xticks(x)
            ax_err_mape.set_xticklabels(error_df["semana"], fontsize=7)
            ax_err_mape.legend(frameon=False, fontsize=7, loc="upper left")
            for idx, value in enumerate(error_df["prophet_mape"]):
                ax_err_mape.text(idx - width / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom", fontsize=7, color="#000000", fontweight="bold")
            for idx, value in enumerate(error_df["ma_mape"]):
                ax_err_mape.text(idx + width / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom", fontsize=7, color="#52525b", fontweight="bold")
            _apply_chart_style(fig_err_mape, ax_err_mape, ylabel="MAPE (%)", xlabel="Semana")
            st.pyplot(fig_err_mape, use_container_width=True)
            plt.close(fig_err_mape)

        with err_right:
            fig_err_rmse, ax_err_rmse = plt.subplots(figsize=(4.3, 4.2), dpi=100)
            ax_err_rmse.bar(x - width / 2, error_df["prophet_rmse"], width=width, color="#000000", label="Prophet")
            ax_err_rmse.bar(x + width / 2, error_df["ma_rmse"], width=width, color="#d4d4d8", label="Media Movil")
            ax_err_rmse.set_xticks(x)
            ax_err_rmse.set_xticklabels(error_df["semana"], fontsize=7)
            ax_err_rmse.legend(frameon=False, fontsize=7, loc="upper left")
            for idx, value in enumerate(error_df["prophet_rmse"]):
                ax_err_rmse.text(idx - width / 2, value + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color="#000000", fontweight="bold")
            for idx, value in enumerate(error_df["ma_rmse"]):
                ax_err_rmse.text(idx + width / 2, value + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color="#52525b", fontweight="bold")
            _apply_chart_style(fig_err_rmse, ax_err_rmse, ylabel="RMSE", xlabel="Semana")
            st.pyplot(fig_err_rmse, use_container_width=True)
            plt.close(fig_err_rmse)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: Resumen de Ventas ──────────────────────────────────────────
    with tab3:
        st.markdown(
            "<div class='panel-card'>"
            "<div class='section-title'>Resumen de Desempeño de Ventas</div>"
            "<div class='section-caption'>Indicadores clave de ventas históricas, promedio de facturación y proyección inmediata.</div>",
            unsafe_allow_html=True,
        )

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            _render_metric_card(
                "Ventas Históricas", 
                _currency(total_history), 
                f"Última semana: {_currency_2(last_hist['y'])}", 
                "Sumatoria agregada de todos los ingresos de ventas registrados en ventas.csv.",
                "Acumulado real histórico neto de transacciones en caja procesadas por el sistema.",
                positive=True
            )
        with col_res2:
            _render_metric_card(
                "Ventas Promedio", 
                _currency(avg_weekly), 
                f"Media semanal global ({len(weekly_df)} semanas)", 
                "Media aritmética global de ingresos dividida por el total de semanas analizadas.",
                "Línea base representativa del ingreso típico semanal esperado en condiciones normales.",
                accent=True
            )
        with col_res3:
            _render_metric_card(
                "Pronóstico Próxima Semana", 
                _currency_2(float(next_forecast['yhat_original'])), 
                f"Intervalo: {_currency_2(float(next_forecast['yhat_lower_original']))} - {_currency_2(float(next_forecast['yhat_upper_original']))}", 
                "Límites superior e inferior que abarcan el 80% del intervalo de confianza predictivo.",
                "Proyección central de Prophet para la siguiente semana incluyendo estacionalidades locales.",
                accent=True,
                positive=True
            )

        # Información complementaria en el tab de resumen
        st.markdown(
            f"""
            <div class='panel-card' style='margin-top: 1rem;'>
                <div class='section-title'>📊 Detalles de Datos de Ventas</div>
                <ul style='font-size: 0.82rem; color: #475569; line-height: 1.6; padding-left: 1.2rem; margin: 0.5rem 0 0 0;'>
                    <li><b>Total de registros:</b> {len(weekly_df)} semanas de datos de ventas reales cargadas desde <code>ventas.csv</code>.</li>
                    <li><b>Última semana registrada:</b> {last_hist['ds'].strftime('%d/%m/%Y')} con un ingreso real de <b>{_currency_2(last_hist['y'])}</b>.</li>
                    <li><b>Modelo Prophet:</b> Inicializado y parametrizado desde el archivo de persistencia <code>{MODEL_PATH.name}</code>.</li>
                    <li><b>Modelo Baseline (Media Móvil):</b> Utiliza parámetros cargados de <code>{MA_MODEL_PATH.name}</code>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================================================================
# === EVALUACIÓN DOCENTE: TRANSICIÓN A REDES NEURONALES COMPLEJAS (CNN-LSTM) ===
# Teoría:
#   - Prophet es un modelo aditivo ad-hoc bayesiano que extrae componentes locales
#     (tendencia lineal/logística, estacionalidades Fourier y feriados) por separado.
#   - CNN-LSTM es una arquitectura híbrida de Deep Learning:
#       - CNN (1D): Extrae características locales espaciales o patrones de corto plazo en secuencias.
#       - LSTM: Aprende relaciones recursivas de largo plazo sobre esas características extraídas.
#   - Estructuración de datos: A diferencia de Prophet (tabular ds/y), requiere dar formato
#     a las entradas como un tensor tridimensional 3D: [samples, timesteps, features].
#
# Código en Vivo (Estructura Keras para CNN-LSTM):
#   # import tensorflow as tf
#   # from tensorflow.keras.models import Sequential
#   # from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, TimeDistributed
#   #
#   # # Ejemplo de dimensiones del tensor
#   # n_samples = 100    # Ventanas deslizantes de entrenamiento
#   # n_timesteps = 4    # Timesteps de secuencia temporal (lags: t-4, t-3, t-2, t-1)
#   # n_features = 6     # Variables exógenas e históricas por timestep
#   #
#   # # Entrada: tensor 3D de dimensiones (n_samples, n_timesteps, n_features)
#   #
#   # model = Sequential([
#   #     # CNN 1D para convolucionar sobre las características temporales locales:
#   #     Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(n_timesteps, n_features)),
#   #     # LSTM para modelar la secuencia a lo largo del tiempo:
#   #     LSTM(64, activation='relu', return_sequences=False),
#   #     # Capa densa de salida para la predicción de ventas t+1:
#   #     Dense(1)
#   # ])
#   # model.compile(optimizer='adam', loss='mse')
#   # model.summary()
# ==============================================================================
