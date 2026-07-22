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
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
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
            padding: 1rem 1.1rem 0.7rem 1.1rem;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
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
    return weekly_df


@st.cache_resource(show_spinner=False)
def _load_model():
    """
    Carga el modelo Prophet desde un archivo joblib.
    """
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

    future_dates = pd.date_range(weekly_df["ds"].max() + pd.Timedelta(days=7), periods=horizon_weeks, freq="W-SUN")
    future = pd.DataFrame({
        "ds": future_dates,
        "ma_pred": np.full(horizon_weeks, float(last_mean)),
    })
    return history, future


def _render_metric_card(title: str, value: str, note: str, accent: bool = False, positive: bool = False) -> None:
    """
    Renderiza una tarjeta de métrica en la interfaz de usuario de Streamlit.
    """
    classes = "metric-card"
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
        f"<div class='{classes}'><span class='metric-kicker'>{title}</span>"
        f"<span class='{value_class}'>{value}</span>"
        f"<span class='{note_class}'>{note}</span></div>",
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
    left_col, right_col = st.columns([0.24, 0.76], gap="large")

    # Renderizar la barra lateral con información y controles
    with left_col:
        st.markdown(
            "<div class='sidebar-card'>"
            "<div class='sidebar-label'>Libreria Analitica</div>"
            "<div class='sidebar-subtitle'>Panel de pronostico semanal</div>"
            "<div class='sidebar-item active'>Resumen Temporal</div>"
            "<div class='sidebar-item'>Rango de fechas</div>"
            "<div class='sidebar-item'>Campanas escolares</div>"
            "<div class='sidebar-item'>Festivos</div>"
            "<div class='sidebar-item'>Modelos predictivos</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-card'>"
            "<div class='section-title'>Ajustes de pronostico</div>"
            "<div class='section-caption'>Controla la ventana futura del modelo.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        horizon_options = [4, 8, 12, 16, 24, 52]
        horizon_weeks = st.select_slider(
            "Horizonte de prediccion",
            options=horizon_options,
            value=4,
            format_func=lambda value: f"Proximas {value} semanas",
        )
        focus_year = st.select_slider(
            "Vista de analisis",
            options=["Resumen temporal", "Ultimas 8 semanas", "Ultimas 12 semanas"],
            value="Resumen temporal",
        )
        run_model = st.button("Ejecutar modelo", use_container_width=True)

    # Renderizar la sección principal con los resultados del pronóstico
    with right_col:
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
        recent_window = 8 if focus_year == "Ultimas 8 semanas" else 12 if focus_year == "Ultimas 12 semanas" else min(4, len(weekly_df))
        backtest_window = min(max(recent_window, 4), max(4, len(weekly_df) // 4 if len(weekly_df) >= 8 else len(weekly_df)))
        error_df = _build_backtest_frame(weekly_df, forecast_hist, ma_history, window=backtest_window)

        if not run_model:
            st.markdown(
                "<div class='panel-card'>Presiona <b>Ejecutar modelo</b> para refrescar el pronostico con el horizonte seleccionado.</div>",
                unsafe_allow_html=True,
            )

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
        prophet_score = prophet_rmse + prophet_mape
        ma_score = ma_rmse + ma_mape
        if prophet_score < ma_score:
            winner = "Prophet"
            winner_note = "Menor error combinado en RMSE + MAPE"
        elif ma_score < prophet_score:
            winner = "Media Movil"
            winner_note = f"Ventana {int(ma_params.get('window', 3))} y ultimo promedio exportado"
        else:
            winner = "Empate"
            winner_note = "Ambos modelos muestran el mismo error combinado"

        st.markdown(
            "<div class='forecast-header'>"
            "<div>"
            "<h1 class='forecast-title'>Panel de Series Temporales</h1>"
            "<p class='forecast-subtitle'>Gestion avanzada de ventas y pronostico analitico con Prophet.</p>"
            "</div>"
            f"<div class='panel-card' style='min-width: 340px; margin: 0;'>"
            f"<div class='section-title'>Horizonte de Prediccion</div>"
            f"<div class='section-caption'>Próximas {horizon_weeks} semanas</div>"
            f"<div style='font-size: 1.55rem; font-weight: 800; color: #111827;'>Semana de arranque: {next_forecast['ds'].strftime('%d/%m/%Y')}</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Renderizar las tarjetas de métricas con los resultados del pronóstico y las evaluaciones de los modelos
        metric_cols = st.columns(6, gap="small")
        with metric_cols[0]:
            _render_metric_card("Ventas Historicas", _currency(total_history), f"Ultima semana: {_currency_2(last_hist['y'])}", positive=True)
        with metric_cols[1]:
            _render_metric_card("Ventas Promedio", _currency(avg_weekly), f"Benchmark MA: ventana {int(ma_params.get('window', 3))}", accent=True)
        with metric_cols[2]:
            _render_metric_card("Pronostico Proxima Semana", _currency_2(float(next_forecast['yhat_original'])), f"Intervalo: {_currency_2(float(next_forecast['yhat_lower_original']))} - {_currency_2(float(next_forecast['yhat_upper_original']))}", accent=True)
        with metric_cols[3]:
            _render_metric_card("Prophet", f"RMSE {prophet_rmse:,.1f}\nMAPE {prophet_mape:.1f}%".replace(",", "."), "Backtesting reciente", positive=True)
        with metric_cols[4]:
            _render_metric_card("Media Movil", f"RMSE {ma_rmse:,.1f}\nMAPE {ma_mape:.1f}%".replace(",", "."), f"Modelo exportado: window {int(ma_params.get('window', 3))}", positive=True)
        with metric_cols[5]:
            _render_metric_card("Ganador dinamico", winner, winner_note, accent=True, positive=True)

        st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='chart-shell'>"
            "<div class='section-title'>Evolucion de Ventas Semanales</div>"
            "<div class='section-caption'>Historial real, pronostico Prophet, intervalo de confianza y media movil exportada.</div>",
            unsafe_allow_html=True,
        )

        # Renderizar el gráfico principal con las ventas reales, el pronóstico de Prophet, el intervalo de confianza y la media móvil
        fig_main, ax_main = plt.subplots(figsize=(11.6, 4.2))
        _apply_chart_style(fig_main, ax_main)
        ax_main.plot(weekly_df["ds"], weekly_df["y"], color="#000000", linewidth=2.0, label="Ventas Reales")
        ax_main.plot(forecast_hist["ds"], forecast_hist["yhat_original"], color="#737373", linewidth=1.8, linestyle=":", label="Prophet Ajuste")
        prophet_future_plot = forecast_future[forecast_future["ds"] >= weekly_df["ds"].max()]
        ax_main.plot(prophet_future_plot["ds"], prophet_future_plot["yhat_original"], color="#1f2937", linewidth=2.0, linestyle="--", label="Prophet Futuro")
        ax_main.fill_between(
            forecast_future["ds"],
            forecast_future["yhat_lower_original"],
            forecast_future["yhat_upper_original"],
            color="#000000",
            alpha=0.08,
            label="Intervalo de Confianza",
        )


        # Calcular la media móvil de las ventas reales para el gráfico
        ma_line = weekly_df["y"].rolling(window=3, min_periods=1).mean()
        ax_main.plot(ma_history["ds"], ma_history["ma_pred"], color="#a3a3a3", linewidth=1.2, alpha=0.8, label="Media Movil")
        ax_main.plot(ma_future["ds"], ma_future["ma_pred"], color="#a3a3a3", linewidth=1.2, alpha=0.4, linestyle="--")
        ax_main.scatter([next_forecast["ds"]], [next_forecast["yhat_original"]], color="#111827", s=28, zorder=5)
        ax_main.annotate(
            f"Hoy S/ {next_forecast['yhat_original']:,.0f}".replace(",", "."),
            xy=(next_forecast["ds"], next_forecast["yhat_original"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="#111827",
        )
        ax_main.set_xlabel("Fecha")
        ax_main.set_ylabel("Ventas (S/)")
        ax_main.legend(frameon=False, ncol=5, fontsize=7, loc="upper left")
        fig_main.autofmt_xdate()
        fig_main.tight_layout()
        st.pyplot(fig_main, use_container_width=True)
        plt.close(fig_main)
        st.markdown("</div>", unsafe_allow_html=True)

        bottom_left, bottom_right = st.columns([1.05, 0.95], gap="large")

        # Renderizar la sección de predicciones futuras con detalles del horizonte seleccionado
        with bottom_left:
            st.markdown(
                "<div class='panel-card'><div class='section-title'>Predicciones Futuras</div><div class='section-caption'>Detalle del horizonte seleccionado.</div>",
                unsafe_allow_html=True,
            )
            future_table = future_rows[["ds", "yhat_original", "yhat_lower_original", "yhat_upper_original"]].copy()
            future_table = future_table.merge(ma_future, on="ds", how="left")
            future_table = future_table.head(horizon_weeks).copy()
            future_table = future_table.rename(columns={
                "ds": "Fecha",
                "yhat_original": "Pronostico Prophet",
                "yhat_lower_original": "Limite Inf",
                "yhat_upper_original": "Limite Sup",
                "ma_pred": "Media Movil",
            })
            future_table["Fecha"] = future_table["Fecha"].dt.strftime("%d/%m/%Y")
            st.dataframe(
                future_table.style.format({
                    "Pronostico Prophet": "S/ {:,.2f}",
                    "Limite Inf": "S/ {:,.2f}",
                    "Limite Sup": "S/ {:,.2f}",
                    "Media Movil": "S/ {:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Renderizar la sección de errores semanales (MAPE y RMSE) para ambos modelos
        with bottom_right:
            st.markdown(
                "<div class='panel-card'><div class='section-title'>Error Semanal (MAPE y RMSE)</div><div class='section-caption'>Ambos errores se muestran simultaneamente sin selector.</div>",
                unsafe_allow_html=True,
            )
            err_left, err_right = st.columns(2)

            with err_left:
                fig_err_mape, ax_err_mape = plt.subplots(figsize=(4.3, 4.2))
                _apply_chart_style(fig_err_mape, ax_err_mape)
                x = np.arange(len(error_df))
                width = 0.33
                ax_err_mape.bar(x - width / 2, error_df["prophet_mape"], width=width, color="#000000", label="Prophet")
                ax_err_mape.bar(x + width / 2, error_df["ma_mape"], width=width, color="#d4d4d8", label="Media Movil")
                ax_err_mape.set_xticks(x)
                ax_err_mape.set_xticklabels(error_df["semana"], fontsize=7)
                ax_err_mape.set_ylabel("MAPE (%)")
                ax_err_mape.set_xlabel("Semana")
                ax_err_mape.legend(frameon=False, fontsize=7, loc="upper left")
                for idx, value in enumerate(error_df["prophet_mape"]):
                    ax_err_mape.text(idx - width / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom", fontsize=7, color="#000000", fontweight="bold")
                for idx, value in enumerate(error_df["ma_mape"]):
                    ax_err_mape.text(idx + width / 2, value + 0.3, f"{value:.1f}%", ha="center", va="bottom", fontsize=7, color="#52525b", fontweight="bold")
                st.pyplot(fig_err_mape, use_container_width=True)
                plt.close(fig_err_mape)

            with err_right:
                fig_err_rmse, ax_err_rmse = plt.subplots(figsize=(4.3, 4.2))
                _apply_chart_style(fig_err_rmse, ax_err_rmse)
                ax_err_rmse.bar(x - width / 2, error_df["prophet_rmse"], width=width, color="#000000", label="Prophet")
                ax_err_rmse.bar(x + width / 2, error_df["ma_rmse"], width=width, color="#d4d4d8", label="Media Movil")
                ax_err_rmse.set_xticks(x)
                ax_err_rmse.set_xticklabels(error_df["semana"], fontsize=7)
                ax_err_rmse.set_ylabel("RMSE")
                ax_err_rmse.set_xlabel("Semana")
                ax_err_rmse.legend(frameon=False, fontsize=7, loc="upper left")
                for idx, value in enumerate(error_df["prophet_rmse"]):
                    ax_err_rmse.text(idx - width / 2, value + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color="#000000", fontweight="bold")
                for idx, value in enumerate(error_df["ma_rmse"]):
                    ax_err_rmse.text(idx + width / 2, value + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color="#52525b", fontweight="bold")
                st.pyplot(fig_err_rmse, use_container_width=True)
                plt.close(fig_err_rmse)

            st.markdown("</div>", unsafe_allow_html=True)


        st.markdown(
            f"<div class='panel-card'>Actualizado con {len(weekly_df)} semanas historicas cargadas desde ventas.csv. "
            f"El modelo Prophet se ejecuta desde {MODEL_PATH.name} y la media movil usa {MA_MODEL_PATH.name}.</div>",
            unsafe_allow_html=True,
        )
