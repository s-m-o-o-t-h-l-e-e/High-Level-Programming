import joblib
import matplotlib.dates as mdates
import pandas as pd

from config import PATHS
from modeling import inverse_domestic_price
from plot_style import plt
from runtime import configure_tensorflow, get_tensorflow, suppress_native_stderr


PERIODS = [
    ("1w", "1주", 7),
    ("1m", "1개월", 30),
    ("1y", "1년", 365),
    ("3y", "3년", 365 * 3),
]


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
    return {
        "news_risk_score": float(signal.get("news_risk_score", 0.0)),
        "forecast_adjustment_pct": float(signal.get("forecast_adjustment_pct", 0.0)),
        "article_count": int(signal.get("article_count", 0)),
    }


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

    fig = plt.figure(figsize=(9, 8), facecolor="white")
    fig.text(0.08, 0.93, "유가추이", fontsize=28, fontweight="bold", color="#333333", ha="left", va="center")
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.86, 0.86], color="black", linewidth=2, transform=fig.transFigure))

    tab_ax = fig.add_axes([0.08, 0.74, 0.84, 0.08])
    tab_ax.axis("off")
    tab_width = 1 / len(PERIODS)
    for idx, (period_code, period_label, _period_days) in enumerate(PERIODS):
        x0 = idx * tab_width
        selected = period_code == code
        face = "#6b6b6b" if selected else "white"
        text_color = "white" if selected else "#666666"
        rect = plt.Rectangle((x0, 0.05), tab_width, 0.82, facecolor=face, edgecolor="#666666", linewidth=1.2)
        tab_ax.add_patch(rect)
        tab_ax.text(x0 + tab_width / 2, 0.46, period_label, ha="center", va="center", fontsize=17, color=text_color, fontweight="bold")

    ax = fig.add_axes([0.16, 0.2, 0.74, 0.46])
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

    recent = processed.tail(lookback).values.reshape(1, lookback, len(processed.columns))
    with suppress_native_stderr(device == "cpu"):
        pred_scaled = model.predict(recent, verbose=0)[0]
    pred_price = inverse_domestic_price(pred_scaled, scaler_bundle)
    news_adjustment = load_news_adjustment()
    adjustment_pct = float(news_adjustment["forecast_adjustment_pct"])
    if adjustment_pct != 0:
        ramp = pd.Series(range(1, horizon + 1), dtype=float) / horizon
        pred_price = pred_price * (1 + adjustment_pct * ramp.values)

    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(today + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame(
        {
            "date": dates,
            "predicted_domestic_price": pred_price,
            "news_risk_score": news_adjustment["news_risk_score"],
            "news_adjustment_pct": adjustment_pct,
            "news_article_count": news_adjustment["article_count"],
        }
    )
    forecast.to_csv(PATHS.forecast, index=False)

    raw = pd.read_csv(PATHS.raw, index_col=0, parse_dates=True)
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
        try:
            from gui_app import launch_analysis_gui, launch_indicator_gui, launch_oil_price_gui

            launch_oil_price_gui(raw, forecast, today)
            launch_indicator_gui(raw, today)
            launch_analysis_gui(raw)
        except Exception as exc:
            print(f"GUI 실행 실패: {exc}")
    return forecast
