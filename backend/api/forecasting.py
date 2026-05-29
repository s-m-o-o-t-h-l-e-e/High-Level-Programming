import joblib
import numpy as np
import pandas as pd

from config import PATHS
import matplotlib.dates as mdates
from modeling import inverse_domestic_price
from plot_style import plt
from runtime import configure_tensorflow, get_tensorflow, suppress_native_stderr


PERIODS = [
    ("1w", "1주", 7),
    ("1m", "1개월", 30),
    ("1y", "1년", 365),
    ("3y", "3년", 365 * 3),
]

LSTM_BLEND_WEIGHT = 0.35
MAX_NEWS_ADJUSTMENT_PCT = 0.0025
MAX_DAILY_CHANGE_WON = 4.0
MAX_TOTAL_CHANGE_WON = 18.0
LSTM_SIGNAL_SCALE_WON = 80.0


def latest_real_domestic_date() -> pd.Timestamp | None:
    if not PATHS.online_meta.exists():
        return None
    meta = pd.read_csv(PATHS.online_meta)
    values = dict(zip(meta["key"], meta["value"]))
    trade_date = values.get("domestic_trade_date")
    if not trade_date:
        return None
    return pd.Timestamp(trade_date).normalize()


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
    price = pd.to_numeric(raw["domestic_price"], errors="coerce").dropna()
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
    daily_change = pd.to_numeric(raw["domestic_price"], errors="coerce").diff().dropna().abs()
    if daily_change.empty:
        return MAX_DAILY_CHANGE_WON
    robust_cap = float(daily_change.tail(90).quantile(0.90) * 1.4)
    return float(np.clip(robust_cap, 1.5, MAX_DAILY_CHANGE_WON))


def _adaptive_lstm_weight(lstm_price: np.ndarray, baseline_price: np.ndarray) -> float:
    residual = float(np.nanmedian(np.abs(np.asarray(lstm_price, dtype=float) - baseline_price)))
    if residual >= 80:
        return 0.05
    if residual >= 40:
        return 0.15
    return LSTM_BLEND_WEIGHT


def _stabilize_forecast(raw: pd.DataFrame, lstm_price: np.ndarray, news_adjustment_pct: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    horizon = len(lstm_price)
    today_price = float(pd.to_numeric(raw["domestic_price"], errors="coerce").dropna().iloc[-1])
    baseline_price = _recent_trend_baseline(raw, horizon)
    lstm_delta = np.asarray(lstm_price, dtype=float) - baseline_price
    lstm_for_blend = baseline_price + np.tanh(lstm_delta / LSTM_SIGNAL_SCALE_WON) * MAX_TOTAL_CHANGE_WON
    effective_lstm_weight = _adaptive_lstm_weight(lstm_price, baseline_price)
    blended_price = effective_lstm_weight * lstm_for_blend + (1 - effective_lstm_weight) * baseline_price

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
    return np.asarray(stabilized), baseline_price, blended_price, effective_lstm_weight


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


def save_period_trend_graph(raw: pd.DataFrame, forecast: pd.DataFrame, today: pd.Timestamp, code: str, label: str, days: int):
    history = raw.loc[raw.index >= today - pd.Timedelta(days=days)].copy()
    today_price = float(raw["domestic_price"].iloc[-1])
    today_label = "오늘 추정" if is_today_estimated(today) else "오늘"
    forecast_dates = pd.to_datetime(forecast["date"])
    forecast_prices = forecast["predicted_domestic_price"]

    fig = plt.figure(figsize=(9, 6), facecolor="white")
    ax = fig.add_axes([0.12, 0.22, 0.82, 0.68])
    ax.plot(history.index, history["domestic_price"], color="#5f8ffb", marker="o", markersize=3, linewidth=2.2, label="국내 유가")
    ax.plot([today] + list(forecast_dates), [today_price] + list(forecast_prices), color="#f2a900", marker="o", markersize=4, linewidth=2.4, label="7일 예측")
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
    one_year = raw.loc[raw.index >= today - pd.Timedelta(days=365)]
    one_month = raw.loc[raw.index >= today - pd.Timedelta(days=30)]
    today_price = float(raw["domestic_price"].iloc[-1])
    today_label = "오늘 추정 유가" if is_today_estimated(today) else "오늘 유가"
    forecast_dates = pd.to_datetime(forecast["date"])
    forecast_prices = forecast["predicted_domestic_price"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("유가 현황 및 7일 예측", fontsize=18)

    axes[0, 0].plot(one_year.index, one_year["domestic_price"], color="#2563eb", linewidth=2)
    axes[0, 0].set_title("최근 1년 유가")
    axes[0, 0].set_ylabel("원/L")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(one_month.index, one_month["domestic_price"], color="#16a34a", linewidth=2)
    axes[0, 1].scatter([one_month.index[-1]], [today_price], color="#dc2626", s=60, zorder=3)
    axes[0, 1].set_title("최근 1개월 유가")
    axes[0, 1].set_ylabel("원/L")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].bar([today_label], [today_price], color="#f59e0b", width=0.45)
    axes[1, 0].text(0, today_price, f"{today_price:,.1f} 원/L", ha="center", va="bottom", fontsize=14)
    axes[1, 0].set_ylim(max(0, today_price * 0.92), today_price * 1.08)
    axes[1, 0].set_title(f"{today_label} ({today.date()})")
    axes[1, 0].set_ylabel("원/L")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].plot([today] + list(forecast_dates), [today_price] + list(forecast_prices), "o-", color="#dc2626", linewidth=2)
    axes[1, 1].set_title("오늘 기준 7일 예측 유가")
    axes[1, 1].set_ylabel("원/L")
    axes[1, 1].grid(alpha=0.3)

    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    dashboard_path = PATHS.figures / "oil_price_dashboard.png"
    fig.savefig(dashboard_path, dpi=170)
    plt.close(fig)
    return dashboard_path


def forecast_next_7_days(model=None, lookback: int = 30, horizon: int = 7, device: str = "gpu", show_gui: bool = True):
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
    recent = processed.tail(lookback).values.reshape(1, lookback, len(processed.columns))
    with suppress_native_stderr(device == "cpu"):
        pred_scaled = model.predict(recent, verbose=0)[0]
    raw_lstm_price = inverse_domestic_price(pred_scaled, scaler_bundle)
    news_adjustment = load_news_adjustment()
    adjustment_pct = float(news_adjustment["forecast_adjustment_pct"])
    pred_price, baseline_price, blended_price, effective_lstm_weight = _stabilize_forecast(raw, raw_lstm_price, adjustment_pct)

    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(today + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame(
        {
            "date": dates,
            "predicted_domestic_price": pred_price,
            "raw_lstm_price": raw_lstm_price,
            "baseline_price": baseline_price,
            "blended_price": blended_price,
            "news_risk_score": news_adjustment["news_risk_score"],
            "news_adjustment_pct": adjustment_pct,
            "news_article_count": news_adjustment["article_count"],
            "daily_change_cap_won": _daily_change_cap(raw),
            "lstm_blend_weight": effective_lstm_weight,
        }
    )
    forecast.to_csv(PATHS.forecast, index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(raw.index[-90:], raw["domestic_price"].tail(90), label="최근 국내 유가")
    plt.plot([today] + list(dates), [raw["domestic_price"].iloc[-1]] + list(pred_price), "r--o", label="오늘 기준 7일 예측")
    plt.title("향후 7일 유가 예측")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS.figures / "seven_day_forecast.png", dpi=160)
    plt.close()
    dashboard_path = save_forecast_dashboard(raw, forecast, today)
    trend_paths = save_all_period_trend_graphs(raw, forecast, today)
    print(f"저장: {PATHS.forecast}")
    print(f"그래프 저장: {dashboard_path}")
    print("기간별 유가추이 그래프 저장:")
    for path in trend_paths:
        print(f"- {path}")
    if show_gui:
        print("GUI 모드는 정리되었습니다. 웹 서버 실행 후 /docs 또는 /graphs에서 결과를 확인하세요.")
    return forecast
