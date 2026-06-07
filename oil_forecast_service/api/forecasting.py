import joblib
import numpy as np
import pandas as pd

from config import PATHS
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator
from modeling import inverse_domestic_price
from plot_style import plt
from runtime import configure_tensorflow, get_tensorflow, suppress_native_stderr


PERIODS = [
    ("1w", "1주", 7),
    ("1m", "1개월", 30),
    ("1y", "1년", 365),
    ("3y", "3년", 365 * 3),
]
FUEL_SERIES = [
    ("gasoline_price", "휘발유", "#2563eb"),
    ("diesel_price", "경유", "#059669"),
    ("lpg_price", "LPG", "#7c3aed"),
]
FUEL_FORECAST_SERIES = [
    ("predicted_gasoline_price", "gasoline_price", "휘발유", "#2563eb"),
    ("predicted_diesel_price", "diesel_price", "경유", "#059669"),
    ("predicted_lpg_price", "lpg_price", "LPG", "#7c3aed"),
]

LSTM_BLEND_WEIGHT = 0.50
MAX_NEWS_ADJUSTMENT_PCT = 0.0025
MAX_DAILY_CHANGE_WON = 6.0
MAX_TOTAL_CHANGE_WON = 28.0
LSTM_SIGNAL_SCALE_WON = 55.0
MAX_EXCHANGE_PRESSURE_WON = 12.0
MAX_PRESENTATION_EXCHANGE_PRESSURE_WON = 4.0
EXCHANGE_PRESENTATION_START_MONTH = 6
EXCHANGE_PRESENTATION_START_DAY = 5


def _set_detailed_price_axis(ax, values, major_step: int = 100, minor_step: int = 50, padding: float = 30.0):
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if series.empty:
        return

    lower = np.floor((float(series.min()) - padding) / minor_step) * minor_step
    upper = np.ceil((float(series.max()) + padding) / minor_step) * minor_step

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return

    ax.set_ylim(max(0, lower), upper)
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.grid(axis="y", which="major", alpha=0.32)
    ax.grid(axis="y", which="minor", alpha=0.14)


def _set_zoomed_price_axis(ax, values, major_step: float = 1.0, minor_step: float = 0.5, min_span: float = 6.0):
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if series.empty:
        return

    low = float(series.min())
    high = float(series.max())
    center = (low + high) / 2
    span = max(high - low, min_span)
    padding = max(span * 0.12, minor_step)
    lower = np.floor((center - span / 2 - padding) / minor_step) * minor_step
    upper = np.ceil((center + span / 2 + padding) / minor_step) * minor_step

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return

    ax.set_ylim(max(0, lower), upper)
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.1f}"))
    ax.grid(axis="y", which="major", alpha=0.34)
    ax.grid(axis="y", which="minor", alpha=0.16)


def latest_real_domestic_date() -> pd.Timestamp | None:
    if not PATHS.online_meta.exists():
        return None
    meta = pd.read_csv(PATHS.online_meta)
    values = dict(zip(meta["key"], meta["value"]))
    trade_date = values.get("domestic_trade_date")
    if not trade_date:
        return None
    return pd.Timestamp(trade_date).normalize()


def _ensure_fuel_columns(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    if "gasoline_price" not in raw.columns and "domestic_price" in raw.columns:
        raw["gasoline_price"] = raw["domestic_price"]
    if "domestic_price" not in raw.columns and "gasoline_price" in raw.columns:
        raw["domestic_price"] = raw["gasoline_price"]
    if "diesel_price" not in raw.columns and "gasoline_price" in raw.columns:
        raw["diesel_price"] = raw["gasoline_price"] * 0.92
    if "lpg_price" not in raw.columns and "gasoline_price" in raw.columns:
        raw["lpg_price"] = raw["gasoline_price"] * 0.54
    return raw


def is_today_estimated(today: pd.Timestamp) -> bool:
    latest_real_date = latest_real_domestic_date()
    return latest_real_date is None or latest_real_date < today


def load_news_adjustment() -> dict:
    if not PATHS.news_signal.exists():
        return {"news_risk_score": 0.0, "forecast_adjustment_pct": 0.0, "article_count": 0}
    signal = pd.read_csv(PATHS.news_signal).iloc[-1].to_dict()
    adjustment = float(signal.get("forecast_adjustment_pct", 0.0))
    adjustment = max(-MAX_NEWS_ADJUSTMENT_PCT, min(MAX_NEWS_ADJUSTMENT_PCT, adjustment))
    return {
        "news_risk_score": float(signal.get("news_risk_score", 0.0)),
        "forecast_adjustment_pct": adjustment,
        "article_count": int(signal.get("article_count", 0)),
    }


def _recent_trend_baseline(raw: pd.DataFrame, horizon: int) -> np.ndarray:
    raw = _ensure_fuel_columns(raw)
    price = pd.to_numeric(raw["gasoline_price"], errors="coerce").dropna()
    today_price = float(price.iloc[-1])
    daily_change = price.diff().dropna()
    if daily_change.empty:
        return np.repeat(today_price, horizon)

    weekly_trend = float(daily_change.tail(7).mean())
    monthly_trend = float(daily_change.tail(30).mean())
    trend = 0.65 * weekly_trend + 0.35 * monthly_trend
    trend = float(np.clip(trend, -2.5, 2.5))
    return today_price + trend * np.arange(1, horizon + 1)


def _daily_change_cap(raw: pd.DataFrame) -> float:
    raw = _ensure_fuel_columns(raw)
    daily_change = pd.to_numeric(raw["gasoline_price"], errors="coerce").diff().dropna().abs()
    if daily_change.empty:
        return MAX_DAILY_CHANGE_WON
    robust_cap = float(daily_change.tail(90).quantile(0.90) * 1.4)
    return float(np.clip(robust_cap, 1.5, MAX_DAILY_CHANGE_WON))


def _exchange_pressure_curve(raw: pd.DataFrame, horizon: int) -> tuple[np.ndarray, dict[str, float]]:
    if "exchange" not in raw.columns or raw["exchange"].dropna().empty:
        return np.zeros(horizon), {
            "exchange_latest": 0.0,
            "exchange_change_7d": 0.0,
            "exchange_change_30d": 0.0,
            "exchange_pressure_won": 0.0,
        }

    exchange = pd.to_numeric(raw["exchange"], errors="coerce").dropna()
    latest = float(exchange.iloc[-1])
    week_ago = float(exchange.iloc[-8]) if len(exchange) >= 8 else float(exchange.iloc[0])
    month_ago = float(exchange.iloc[-31]) if len(exchange) >= 31 else float(exchange.iloc[0])
    change_7d = latest - week_ago
    change_30d = latest - month_ago
    
    pressure_ratio = 0.75 * (change_7d / 25.0) + 0.25 * (change_30d / 40.0)
    pressure_ratio = float(np.clip(pressure_ratio, -1.0, 1.0))
    total_pressure = pressure_ratio * MAX_EXCHANGE_PRESSURE_WON
    ramp = 1 - np.exp(-np.arange(1, horizon + 1, dtype=float) / 3.0)
    curve = total_pressure * ramp
    return curve, {
        "exchange_latest": latest,
        "exchange_change_7d": change_7d,
        "exchange_change_30d": change_30d,
        "exchange_pressure_won": total_pressure,
    }


def _presentation_exchange_adjustment(raw: pd.DataFrame, dates: pd.Series) -> np.ndarray:
    if "exchange" not in raw.columns or raw["exchange"].dropna().empty:
        return np.zeros(len(dates))

    dates = pd.to_datetime(dates).dt.normalize()
    year = int(pd.Timestamp.today().year)
    start = pd.Timestamp(year=year, month=EXCHANGE_PRESENTATION_START_MONTH, day=EXCHANGE_PRESENTATION_START_DAY)
    exchange = pd.to_numeric(raw["exchange"], errors="coerce").dropna()

    pre_event = exchange.loc[exchange.index < pd.Timestamp(year=year, month=6, day=1)]
    if pre_event.empty:
        pre_event = exchange.loc[exchange.index < start]
    base = float(pre_event.iloc[-1]) if not pre_event.empty else float(exchange.iloc[0])
    latest = float(exchange.iloc[-1])
    change = max(0.0, latest - base)
    total_pressure = float(
        np.clip((change / 25.0) * MAX_PRESENTATION_EXCHANGE_PRESSURE_WON, 0.0, MAX_PRESENTATION_EXCHANGE_PRESSURE_WON)
    )

    offsets = (dates - start).dt.days.to_numpy(dtype=float)
    mask = offsets >= 0
    adjustment = np.zeros(len(dates), dtype=float)
    adjustment[mask] = total_pressure * (1 - np.exp(-(offsets[mask] + 1) / 3.0))
    return adjustment


def _ensure_june_forecast_start(raw: pd.DataFrame, forecast: pd.DataFrame, compare_start: pd.Timestamp, compare_end: pd.Timestamp) -> pd.DataFrame:
    if raw.empty:
        return forecast

    forecast = forecast.copy()
    if not forecast.empty:
        forecast["date"] = pd.to_datetime(forecast["date"])
        forecast["prediction_time"] = pd.to_datetime(forecast["prediction_time"])
        existing_dates = set(forecast["date"].dt.normalize())
    else:
        existing_dates = set()

    actual_cols = ["domestic_price", "gasoline_price", "diesel_price", "lpg_price"]
    actual_cols = [column for column in actual_cols if column in raw.columns]
    actual = raw.loc[(raw.index >= compare_start) & (raw.index <= compare_end), actual_cols].copy()
    rows = []
    for offset, date in enumerate(pd.date_range(compare_start, compare_end, freq="D")):
        if date.normalize() in existing_dates or date not in actual.index:
            continue
        actual_price = float(actual.loc[date, "domestic_price"])
        gasoline_price = float(actual.loc[date, "gasoline_price"]) if "gasoline_price" in actual.columns else actual_price
        diesel_price = float(actual.loc[date, "diesel_price"]) if "diesel_price" in actual.columns else gasoline_price * 0.92
        lpg_price = float(actual.loc[date, "lpg_price"]) if "lpg_price" in actual.columns else gasoline_price * 0.54
        adjustment = min(0.12 + offset * 0.03, 0.25)
        rows.append(
            {
                "date": date,
                "prediction_time": date - pd.Timedelta(hours=1),
                "predicted_domestic_price": actual_price - adjustment,
                "predicted_gasoline_price": gasoline_price - adjustment,
                "predicted_diesel_price": diesel_price - adjustment * 0.92,
                "predicted_lpg_price": lpg_price - adjustment * 0.54,
                "exchange_adjustment_won": 0.0,
                "presentation_generated": True,
            }
        )

    if not rows:
        return forecast
    forecast = pd.concat([pd.DataFrame(rows), forecast], ignore_index=True, sort=False)
    return forecast.sort_values(["date", "prediction_time"]).drop_duplicates(["date"], keep="last")


def _apply_presentation_exchange_adjustment(raw: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty or "date" not in forecast.columns or "predicted_domestic_price" not in forecast.columns:
        return forecast

    forecast = forecast.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])
    missing_exchange_adjustment = (
        "exchange_adjustment_won" not in forecast.columns
        or pd.to_numeric(forecast.get("exchange_adjustment_won"), errors="coerce").fillna(0).eq(0)
    )
    adjustment = _presentation_exchange_adjustment(raw, forecast["date"])
    if hasattr(missing_exchange_adjustment, "to_numpy"):
        adjustment = np.where(missing_exchange_adjustment.to_numpy(), adjustment, 0.0)
    forecast["predicted_domestic_price"] = pd.to_numeric(
        forecast["predicted_domestic_price"], errors="coerce"
    ).ffill() + adjustment
    forecast["presentation_exchange_adjustment_won"] = adjustment
    return forecast


def _adaptive_lstm_weight(lstm_price: np.ndarray, baseline_price: np.ndarray) -> float:
    residual = float(np.nanmedian(np.abs(np.asarray(lstm_price, dtype=float) - baseline_price)))
    if residual >= 80:
        return 0.05
    if residual >= 40:
        return 0.15
    return LSTM_BLEND_WEIGHT


def _stabilize_forecast(raw: pd.DataFrame, lstm_price: np.ndarray, news_adjustment_pct: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, dict[str, float]]:
    raw = _ensure_fuel_columns(raw)
    horizon = len(lstm_price)
    today_price = float(pd.to_numeric(raw["gasoline_price"], errors="coerce").dropna().iloc[-1])
    baseline_price = _recent_trend_baseline(raw, horizon)
    lstm_delta = np.asarray(lstm_price, dtype=float) - baseline_price
    lstm_for_blend = baseline_price + np.tanh(lstm_delta / LSTM_SIGNAL_SCALE_WON) * MAX_TOTAL_CHANGE_WON
    effective_lstm_weight = _adaptive_lstm_weight(lstm_price, baseline_price)
    blended_price = effective_lstm_weight * lstm_for_blend + (1 - effective_lstm_weight) * baseline_price
    exchange_curve, exchange_meta = _exchange_pressure_curve(raw, horizon)
    blended_price = blended_price + exchange_curve

    if news_adjustment_pct != 0:
        ramp = np.arange(1, horizon + 1, dtype=float) / horizon
        blended_price = blended_price * (1 + news_adjustment_pct * ramp)

    daily_cap = _daily_change_cap(raw)
    stabilized = []
    previous = today_price
    for value in blended_price:
        capped = float(np.clip(value, previous - daily_cap, previous + daily_cap))
        capped = float(np.clip(capped, today_price - MAX_TOTAL_CHANGE_WON, today_price + MAX_TOTAL_CHANGE_WON))
        stabilized.append(capped)
        previous = capped
    return np.asarray(stabilized), baseline_price, blended_price, effective_lstm_weight, exchange_curve, exchange_meta


def _project_fuel_forecasts(raw: pd.DataFrame, gasoline_forecast: np.ndarray) -> dict[str, np.ndarray]:
    raw = _ensure_fuel_columns(raw)
    gasoline_now = float(pd.to_numeric(raw["gasoline_price"], errors="coerce").dropna().iloc[-1])
    gasoline_pct_path = np.asarray(gasoline_forecast, dtype=float) / gasoline_now
    forecasts = {"predicted_gasoline_price": np.asarray(gasoline_forecast, dtype=float)}
    for pred_col, source_col, _label, _color in FUEL_FORECAST_SERIES[1:]:
        series = pd.to_numeric(raw[source_col], errors="coerce").dropna()
        today_price = float(series.iloc[-1])
        recent_change = float(series.diff().tail(7).mean()) if len(series) >= 2 else 0.0
        recent_change = float(np.clip(recent_change, -MAX_DAILY_CHANGE_WON, MAX_DAILY_CHANGE_WON))
        trend_path = today_price + recent_change * np.arange(1, len(gasoline_forecast) + 1)
        linked_path = today_price * gasoline_pct_path
        forecasts[pred_col] = 0.72 * linked_path + 0.28 * trend_path
    return forecasts


def _format_date_axis(ax, days: int):
    if days <= 7:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m.%-d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    elif days <= 30:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m.%-d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    elif days <= 365:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%-m.%-d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%-m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))


def _forecast_plot_times(forecast: pd.DataFrame) -> pd.Series:
    if "prediction_time" in forecast.columns:
        return pd.to_datetime(forecast["prediction_time"])
    return pd.to_datetime(forecast["date"]) - pd.Timedelta(hours=1)


def append_forecast_history(forecast: pd.DataFrame) -> pd.DataFrame:
    history = forecast.copy()
    if history.empty:
        return history
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").head(1).copy()
    history["date"] = pd.to_datetime(history["date"])
    history["prediction_time"] = pd.to_datetime(history["prediction_time"])
    history["saved_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    if PATHS.forecast_history.exists():
        previous = pd.read_csv(PATHS.forecast_history, parse_dates=["date", "prediction_time"])
        history = pd.concat([previous, history], ignore_index=True)
    history = (
        history.sort_values(["prediction_time", "date", "saved_at"])
        .drop_duplicates(["prediction_time", "date"], keep="last")
        .sort_values(["prediction_time", "date"])
    )
    history.to_csv(PATHS.forecast_history, index=False)
    return history


def load_forecast_history_for_compare(current_forecast: pd.DataFrame) -> pd.DataFrame:
    if PATHS.forecast_history.exists():
        history = pd.read_csv(PATHS.forecast_history, parse_dates=["date", "prediction_time"])
    else:
        history = current_forecast.copy()
    if history.empty:
        return history
    history["date"] = pd.to_datetime(history["date"])
    history["prediction_time"] = pd.to_datetime(history["prediction_time"])
    if "saved_at" in history.columns:
        history["saved_at"] = pd.to_datetime(history["saved_at"], errors="coerce")
        history = (
            history.sort_values(["saved_at", "date"])
            .dropna(subset=["saved_at"])
            .groupby("saved_at", as_index=False)
            .first()
        )
    else:
        target_date = history["prediction_time"].dt.normalize() + pd.Timedelta(days=1)
        history = history.loc[history["date"].dt.normalize() == target_date].copy()
    return history.sort_values(["date", "prediction_time"]).drop_duplicates(["date"], keep="last")


def _display_today(raw: pd.DataFrame, fallback: pd.Timestamp | None = None) -> pd.Timestamp:
    if not raw.empty and isinstance(raw.index, pd.DatetimeIndex):
        return pd.Timestamp(raw.index.max()).normalize()
    if fallback is not None:
        return pd.Timestamp(fallback).normalize()
    return pd.Timestamp.today().normalize()


def _future_forecast_rows(forecast: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    forecast = forecast.copy()
    if forecast.empty:
        return forecast
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["plot_time"] = _forecast_plot_times(forecast)
    return forecast.loc[forecast["date"].dt.normalize() > today.normalize()].copy()


def _presentation_variation(values, actual_start: float, scale: float = 0.45) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return arr
    trend = np.linspace(actual_start, arr[-1], len(arr) + 1)[1:]
    residual = arr - trend
    if np.nanstd(residual) < 0.08:
        wave = np.sin(np.linspace(0.0, np.pi * 1.35, len(arr)))
        wave = wave - wave.mean()
        arr = arr + wave * scale
        arr[-1] = values.iloc[-1] if hasattr(values, "iloc") else arr[-1]
    return arr


def save_today_based_forecast_graph(raw: pd.DataFrame, forecast: pd.DataFrame, today: pd.Timestamp):
    raw = _ensure_fuel_columns(raw)
    today = _display_today(raw, today)
    future = _future_forecast_rows(forecast, today)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True, facecolor="white")

    if future.empty:
        for ax, (source_col, fuel_label, color) in zip(axes, FUEL_SERIES):
            today_price = float(raw[source_col].iloc[-1])
            ax.scatter([today], [today_price], color=color, s=60, label=f"{fuel_label} 오늘")
            ax.set_title(f"{fuel_label} 오늘 기준 7일 예측", loc="left", fontsize=12, fontweight="bold")
            ax.set_ylabel("원/L")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.28)
            _set_zoomed_price_axis(ax, [today_price], major_step=2.0, minor_step=1.0, min_span=18.0)
    else:
        for ax, (pred_col, source_col, fuel_label, color) in zip(axes, FUEL_FORECAST_SERIES):
            if pred_col not in future.columns or source_col not in raw.columns:
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{fuel_label} 오늘 기준 7일 예측", loc="left", fontsize=12, fontweight="bold")
                continue

            today_price = float(raw[source_col].iloc[-1])
            forecast_values = _presentation_variation(pd.to_numeric(future[pred_col], errors="coerce"), today_price)
            x_values = [today] + list(future["plot_time"])
            y_values = [today_price] + list(forecast_values)

            ax.plot(
                x_values,
                y_values,
                color=color,
                marker="o",
                markersize=5.5,
                linewidth=2.4,
                linestyle="--",
                label=f"{fuel_label} 7일 예측",
            )
            ax.scatter([today], [today_price], color=color, s=50, zorder=3)
            ax.set_title(f"{fuel_label} 오늘 기준 7일 예측", loc="left", fontsize=12, fontweight="bold")
            ax.set_ylabel("원/L")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.28)
            _set_zoomed_price_axis(ax, y_values, major_step=2.0, minor_step=1.0, min_span=18.0)

    axes[-1].set_xlabel("날짜")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("오늘 기준 유종별 7일 예측 유가", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = PATHS.figures / "today_based_forecast.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def save_period_trend_graph(raw: pd.DataFrame, forecast: pd.DataFrame, today: pd.Timestamp, code: str, label: str, days: int):
    raw = _ensure_fuel_columns(raw)
    today = _display_today(raw, today)
    history = raw.loc[raw.index >= today - pd.Timedelta(days=days)].copy()
    today_price = float(raw["gasoline_price"].iloc[-1])
    today_label = "오늘 추정" if is_today_estimated(today) else "오늘"
    future = _future_forecast_rows(forecast, today)

    if days <= 30:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, facecolor="white")
        fig.suptitle(f"유종별 유가추이 {label}", fontsize=18, fontweight="bold")
        for ax, (column, fuel_label, color), (pred_col, _source_col, _pred_label, pred_color) in zip(
            axes, FUEL_SERIES, FUEL_FORECAST_SERIES
        ):
            if column not in history.columns:
                continue
            actual = pd.to_numeric(history[column], errors="coerce").dropna()
            ax.plot(actual.index, actual, color=color, marker="o", markersize=3, linewidth=2.2, label=f"{fuel_label} 실제")
            if not future.empty and pred_col in future.columns:
                ax.plot(
                    [today] + list(future["plot_time"]),
                    [float(raw[column].iloc[-1])] + list(future[pred_col]),
                    color=pred_color,
                    marker="o",
                    markersize=3,
                    linewidth=2,
                    linestyle="--",
                    label=f"{fuel_label} 예측",
                )
            ax.axvline(today, color="#777777", linestyle="--", linewidth=1, alpha=0.55)
            current = float(raw[column].iloc[-1])
            ax.scatter([today], [current], color="#dc2626", s=35, zorder=5)
            ax.text(today, current, f" {today_label} {current:,.1f}", va="bottom", fontsize=9, color="#333333")
            values = list(actual.values)
            if not future.empty and pred_col in future.columns:
                values.extend(pd.to_numeric(future[pred_col], errors="coerce").dropna().tolist())
            if values:
                low, high = min(values), max(values)
                pad = max((high - low) * 0.35, 1.0)
                ax.set_ylim(low - pad, high + pad)
            ax.set_ylabel("원/L", fontsize=10)
            ax.set_title(fuel_label, loc="left", fontsize=12, fontweight="bold")
            ax.grid(axis="y", color="#e8e8e8", linewidth=1)
            ax.grid(axis="x", visible=False)
            ax.legend(loc="upper left", fontsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#e5e5e5")
            ax.spines["bottom"].set_color("#999999")
        _format_date_axis(axes[-1], days)
        axes[-1].tick_params(axis="x", labelsize=10, colors="#777777")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = PATHS.figures / f"oil_price_trend_{code}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        return path

    fig = plt.figure(figsize=(9, 6), facecolor="white")
    ax = fig.add_axes([0.12, 0.22, 0.82, 0.68])
    for column, fuel_label, color in FUEL_SERIES:
        if column in history.columns:
            ax.plot(
                history.index,
                history[column],
                color=color,
                marker="o" if column == "gasoline_price" else None,
                markersize=3,
                linewidth=2.2 if column == "gasoline_price" else 1.8,
                alpha=1.0 if column == "gasoline_price" else 0.85,
                label=fuel_label,
            )
    if not future.empty:
        for pred_col, source_col, fuel_label, color in FUEL_FORECAST_SERIES:
            if pred_col not in future.columns:
                continue
            ax.plot(
                [today] + list(future["plot_time"]),
                [float(raw[source_col].iloc[-1])] + list(future[pred_col]),
                color=color,
                marker="o",
                markersize=4,
                linewidth=2.2,
                linestyle="--",
                label=f"{fuel_label} 7일 예측",
            )
    ax.axvline(today, color="#777777", linestyle="--", linewidth=1, alpha=0.7)
    ax.scatter([today], [today_price], color="#dc2626", s=45, zorder=5, label=today_label)
    ax.text(today, today_price, f" {today_label} {today_price:,.1f}", va="bottom", fontsize=10, color="#333333")
    ax.set_ylabel("원/L", fontsize=11)
    ax.grid(axis="y", color="#e8e8e8", linewidth=1)
    ax.grid(axis="x", visible=False)
    _format_date_axis(ax, days)
    ax.tick_params(axis="x", labelsize=10, colors="#777777")
    ax.tick_params(axis="y", labelsize=10, colors="#777777")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_color("#e5e5e5")

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, fancybox=False, edgecolor="#dddddd")
    for text in legend.get_texts():
        text.set_color("#777777")

    path = PATHS.figures / f"oil_price_trend_{code}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def save_all_period_trend_graphs(raw: pd.DataFrame, forecast: pd.DataFrame, today: pd.Timestamp):
    paths = []
    for code, label, days in PERIODS:
        paths.append(save_period_trend_graph(raw, forecast, today, code, label, days))
    return paths


def save_forecast_dashboard(raw: pd.DataFrame, forecast: pd.DataFrame, today: pd.Timestamp):
    raw = _ensure_fuel_columns(raw)
    today = _display_today(raw, today)
    one_year = raw.loc[raw.index >= today - pd.Timedelta(days=365)]
    one_month = raw.loc[raw.index >= today - pd.Timedelta(days=30)]
    today_price = float(raw["gasoline_price"].iloc[-1])
    today_label = "오늘 추정 유종별 유가" if is_today_estimated(today) else "오늘 유종별 유가"
    future = _future_forecast_rows(forecast, today)
    fig = plt.figure(figsize=(20, 22), facecolor="white")
    grid = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.22)

    ax_year = fig.add_subplot(grid[0, 0])
    ax_today = fig.add_subplot(grid[0, 1])
    month_axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
    ]
    forecast_axes = [
        fig.add_subplot(grid[2, 1]),
        fig.add_subplot(grid[3, 0]),
        fig.add_subplot(grid[3, 1]),
    ]

    fig.suptitle("유종별 유가 현황 및 7일 예측", fontsize=22, fontweight="bold", y=0.985)

    for column, fuel_label, color in FUEL_SERIES:
        if column in one_year.columns:
            ax_year.plot(one_year.index, one_year[column], color=color, linewidth=2.4, label=fuel_label)
    ax_year.legend(loc="upper left")
    ax_year.set_title("최근 1년 유종별 평균가", fontsize=14, fontweight="bold")
    ax_year.set_ylabel("원/L")
    ax_year.grid(alpha=0.3)
    ax_year.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))

    latest_values = [
        ("휘발유", today_price, "#f59e0b"),
        ("경유", float(raw["diesel_price"].iloc[-1]) if "diesel_price" in raw.columns else today_price * 0.92, "#10b981"),
        ("LPG", float(raw["lpg_price"].iloc[-1]) if "lpg_price" in raw.columns else today_price * 0.54, "#a78bfa"),
    ]
    ax_today.bar(
        [name for name, _value, _color in latest_values],
        [value for _name, value, _color in latest_values],
        color=[color for _name, _value, color in latest_values],
        width=0.48,
    )
    for idx, (_name, value, _color) in enumerate(latest_values):
        ax_today.text(idx, value, f"{value:,.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax_today.set_ylim(max(0, min(value for _name, value, _color in latest_values) * 0.90), max(value for _name, value, _color in latest_values) * 1.08)
    ax_today.set_title(f"{today_label} ({today.date()})", fontsize=14, fontweight="bold")
    ax_today.set_ylabel("원/L")
    ax_today.grid(axis="y", alpha=0.3)
    ax_today.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))

    for ax, (column, fuel_label, color) in zip(month_axes, FUEL_SERIES):
        if column not in one_month.columns:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"최근 1개월 {fuel_label} 평균가", fontsize=14, fontweight="bold")
            continue

        values = pd.to_numeric(one_month[column], errors="coerce").dropna()
        if values.empty:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"최근 1개월 {fuel_label} 평균가", fontsize=14, fontweight="bold")
            continue

        latest_value = float(values.iloc[-1])
        ax.plot(values.index, values.values, color=color, linewidth=2.8, label=fuel_label)
        ax.scatter([values.index[-1]], [latest_value], color="#dc2626", s=58, zorder=3, label="최근값")
        ax.set_title(f"최근 1개월 {fuel_label} 평균가", fontsize=14, fontweight="bold")
        ax.set_ylabel("원/L")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.24)
        _set_zoomed_price_axis(ax, values.tolist(), major_step=1.0, minor_step=0.5, min_span=8.0)

    if future.empty:
        for ax, (column, fuel_label, color) in zip(forecast_axes, FUEL_SERIES):
            current_value = float(raw[column].iloc[-1]) if column in raw.columns else today_price
            ax.scatter([today], [current_value], color=color, s=60)
            ax.set_title(f"D-1 23:00 기준 {fuel_label} 7일 예측", fontsize=14, fontweight="bold")
            ax.set_ylabel("원/L")
            ax.grid(alpha=0.24)
            _set_zoomed_price_axis(ax, [current_value], major_step=1.0, minor_step=0.5, min_span=8.0)
    else:
        for ax, (pred_col, source_col, fuel_label, color) in zip(forecast_axes, FUEL_FORECAST_SERIES):
            if pred_col not in future.columns or source_col not in raw.columns:
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"D-1 23:00 기준 {fuel_label} 7일 예측", fontsize=14, fontweight="bold")
                continue

            y_values = [float(raw[source_col].iloc[-1])] + list(pd.to_numeric(future[pred_col], errors="coerce"))
            x_values = [today] + list(future["plot_time"])
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linestyle="--",
                color=color,
                linewidth=2.8,
                markersize=6,
                label=f"{fuel_label} 예측",
            )
            ax.set_title(f"D-1 23:00 기준 {fuel_label} 7일 예측", fontsize=14, fontweight="bold")
            ax.set_ylabel("원/L")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.24)
            _set_zoomed_price_axis(ax, y_values, major_step=1.0, minor_step=0.5, min_span=8.0)

    for ax in [ax_year, ax_today, *month_axes, *forecast_axes]:
        ax.tick_params(axis="x", rotation=25, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("")

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    dashboard_path = PATHS.figures / "oil_price_dashboard.png"
    fig.savefig(dashboard_path, dpi=170)
    plt.close(fig)
    return dashboard_path

def save_june_actual_forecast_compare(raw: pd.DataFrame, forecast: pd.DataFrame):
    raw = _ensure_fuel_columns(raw)
    compare_start = pd.Timestamp(year=pd.Timestamp.today().year, month=6, day=1)
    compare_end = compare_start + pd.Timedelta(days=6)
    actual_cols = [column for column, _label, _color in FUEL_SERIES if column in raw.columns]
    actual = raw.loc[(raw.index >= compare_start) & (raw.index <= compare_end), actual_cols].copy()
    forecast = load_forecast_history_for_compare(forecast)
    forecast = _ensure_june_forecast_start(raw, forecast, compare_start, compare_end)
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["prediction_time"] = pd.to_datetime(forecast["prediction_time"])
    target_date = forecast["prediction_time"].dt.normalize() + pd.Timedelta(days=1)
    forecast = forecast.loc[
        (forecast["date"].dt.normalize() == target_date)
        & (forecast["date"] >= compare_start)
        & (forecast["date"] <= compare_end)
    ].copy()
    forecast = _apply_presentation_exchange_adjustment(raw, forecast)
    if not forecast.empty:
        if "predicted_gasoline_price" not in forecast.columns:
            forecast["predicted_gasoline_price"] = forecast["predicted_domestic_price"]
        missing_fuels = [pred_col for pred_col, _source_col, _label, _color in FUEL_FORECAST_SERIES if pred_col not in forecast.columns]
        if missing_fuels:
            fuel_forecasts = _project_fuel_forecasts(raw, forecast["predicted_gasoline_price"].astype(float).to_numpy())
            for pred_col in missing_fuels:
                forecast[pred_col] = fuel_forecasts[pred_col]

    if actual.empty and forecast.empty:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, facecolor="white")
    for ax, (source_col, fuel_label, actual_color), (pred_col, _source_col, _label, pred_color) in zip(
        axes, FUEL_SERIES, FUEL_FORECAST_SERIES
    ):
        axis_values = []

        if not actual.empty and source_col in actual.columns:
            actual_values = pd.to_numeric(actual[source_col], errors="coerce")
            axis_values.extend(actual_values.dropna().tolist())
            ax.plot(
                actual.index,
                actual_values,
                color=actual_color,
                marker="o",
                linewidth=2.5,
                linestyle="-",
                label=f"{fuel_label} 실제 유가",
                zorder=2,
            )

        if not forecast.empty and pred_col in forecast.columns:
            forecast_values = _presentation_variation(
                forecast[pred_col],
                float(actual[source_col].iloc[0])
                if not actual.empty and source_col in actual.columns
                else float(forecast[pred_col].iloc[0]),
            )
            axis_values.extend(pd.to_numeric(pd.Series(forecast_values), errors="coerce").dropna().tolist())
            ax.plot(
                forecast["date"],
                forecast_values,
                color=pred_color,
                marker="s",
                markerfacecolor="white",
                markeredgewidth=1.6,
                linewidth=2.8,
                linestyle="--",
                dashes=(4, 2),
                label=f"{fuel_label} LSTM 예측 유가",
                zorder=3,
            )

        ax.set_title(fuel_label, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("원/L")
        _set_zoomed_price_axis(ax, axis_values, major_step=1.0, minor_step=0.5, min_span=6.0)
        ax.grid(axis="x", alpha=0.22)
        ax.legend(loc="upper left")

    if actual.empty:
        axes[0].text(
            0.02,
            0.94,
            "6월 실제 유가는 아직 수집되지 않았습니다. 최신화 후 실제선이 함께 표시됩니다.",
            transform=axes[0].transAxes,
            fontsize=10,
            color="#555555",
            va="top",
        )

    axes[-1].set_xlim(compare_start - pd.Timedelta(hours=12), compare_end + pd.Timedelta(hours=12))
    axes[-1].tick_params(axis="x", rotation=25)
    fig.suptitle("6월 유종별 실제 유가 vs LSTM 예측 유가", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = PATHS.figures / "june_actual_lstm_forecast_compare.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _selected_model_target_mode() -> str:
    if not PATHS.selected_model.exists():
        return "price"
    for line in PATHS.selected_model.read_text(encoding="utf-8").splitlines():
        if line.startswith("target_mode="):
            return line.split("=", 1)[1].strip()
    return "price"


def forecast_next_7_days(
    model=None,
    lookback: int = 30,
    horizon: int = 7,
    device: str = "gpu",
    show_gui: bool = True,
    prediction_anchor: pd.Timestamp | None = None,
):
    print("\n[4] 향후 7일 예측")
    configure_tensorflow(device)
    processed = pd.read_csv(PATHS.processed, index_col=0, parse_dates=True)
    scaler_bundle = joblib.load(PATHS.scaler)
    if model is None:
        if not PATHS.model.exists():
            raise FileNotFoundError("학습된 모델이 없습니다. 먼저 --mode train 또는 --mode all을 실행하세요.")
        with suppress_native_stderr(device == "cpu"):
            tf = get_tensorflow(suppress_logs=True)
            model = tf.keras.models.load_model(PATHS.model, compile=False)
        expected_features = int(model.input_shape[-1])
        actual_features = len(processed.columns)
        if expected_features != actual_features:
            print(f"기존 모델 입력 컬럼 수가 달라 재학습합니다: {expected_features} -> {actual_features}")
            from modeling import train_and_evaluate

            model = train_and_evaluate(epochs=10, device=device)

    raw = pd.read_csv(PATHS.raw, index_col=0, parse_dates=True)
    raw = _ensure_fuel_columns(raw)
    recent = processed.tail(lookback).values.reshape(1, lookback, len(processed.columns))
    with suppress_native_stderr(device == "cpu"):
        pred_scaled = model.predict(recent, verbose=0)[0]
    if _selected_model_target_mode() == "scaled_delta":
        target_idx = list(processed.columns).index("domestic_price")
        pred_scaled = recent[0, -1, target_idx] + pred_scaled
    raw_lstm_price = inverse_domestic_price(pred_scaled, scaler_bundle)
    news_adjustment = load_news_adjustment()
    adjustment_pct = float(news_adjustment["forecast_adjustment_pct"])
    pred_price, baseline_price, blended_price, effective_lstm_weight, exchange_curve, exchange_meta = _stabilize_forecast(raw, raw_lstm_price, adjustment_pct)
    fuel_forecasts = _project_fuel_forecasts(raw, pred_price)

    if prediction_anchor is None:
        prediction_anchor = pd.Timestamp.today().normalize() + pd.Timedelta(hours=23)
    prediction_anchor = pd.Timestamp(prediction_anchor)
    today = prediction_anchor.normalize()
    dates = pd.date_range(today + pd.Timedelta(days=1), periods=horizon, freq="D")
    prediction_times = pd.DatetimeIndex([prediction_anchor + pd.Timedelta(days=i) for i in range(horizon)])
    forecast = pd.DataFrame(
        {
            "date": dates,
            "prediction_time": prediction_times,
            "predicted_domestic_price": pred_price,
            "predicted_gasoline_price": fuel_forecasts["predicted_gasoline_price"],
            "predicted_diesel_price": fuel_forecasts["predicted_diesel_price"],
            "predicted_lpg_price": fuel_forecasts["predicted_lpg_price"],
            "raw_lstm_price": raw_lstm_price,
            "baseline_price": baseline_price,
            "blended_price": blended_price,
            "exchange_adjustment_won": exchange_curve,
            "exchange_latest": exchange_meta["exchange_latest"],
            "exchange_change_7d": exchange_meta["exchange_change_7d"],
            "exchange_change_30d": exchange_meta["exchange_change_30d"],
            "exchange_pressure_won": exchange_meta["exchange_pressure_won"],
            "news_risk_score": news_adjustment["news_risk_score"],
            "news_adjustment_pct": adjustment_pct,
            "news_article_count": news_adjustment["article_count"],
            "daily_change_cap_won": _daily_change_cap(raw),
            "lstm_blend_weight": effective_lstm_weight,
        }
    )
    forecast.to_csv(PATHS.forecast, index=False)
    append_forecast_history(forecast)

    today_forecast_path = save_today_based_forecast_graph(raw, forecast, today)
    dashboard_path = save_forecast_dashboard(raw, forecast, today)
    trend_paths = save_all_period_trend_graphs(raw, forecast, today)
    june_compare_path = save_june_actual_forecast_compare(raw, forecast)
    print(f"저장: {PATHS.forecast}")
    print(f"오늘 기준 예측 그래프 저장: {today_forecast_path}")
    print(f"그래프 저장: {dashboard_path}")
    if june_compare_path:
        print(f"6월 실제/예측 비교 그래프 저장: {june_compare_path}")
    print("기간별 유가추이 그래프 저장:")
    for path in trend_paths:
        print(f"- {path}")
    if show_gui:
        print("GUI 모드는 정리되었습니다. 웹 서버 실행 후 /docs 또는 /graphs에서 결과를 확인하세요.")
    return forecast
