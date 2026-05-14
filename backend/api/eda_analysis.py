from datetime import timedelta

import pandas as pd

from config import EVENTS, OUT_DIR, PATHS
from plot_style import plt


def lag_correlation(df: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    rows = []
    returns = df["domestic_price"].pct_change()
    for lag in range(max_lag + 1):
        rows.append(
            {
                "lag_days": lag,
                "wti_to_domestic_corr": df["wti"].pct_change().shift(lag).corr(returns),
                "brent_to_domestic_corr": df["brent"].pct_change().shift(lag).corr(returns),
                "risk_to_domestic_corr": df["risk_index"].shift(lag).corr(returns.abs()),
            }
        )
    return pd.DataFrame(rows)


def event_window_analysis(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    rows = []
    for date_text, event_name, _weight in EVENTS:
        date = pd.Timestamp(date_text)
        before = df.loc[date - timedelta(days=window): date - timedelta(days=1)]
        after = df.loc[date: date + timedelta(days=window)]
        if before.empty or after.empty:
            continue
        base = before["domestic_price"].iloc[-1]
        peak = after["domestic_price"].max()
        trough = after["domestic_price"].min()
        rows.append(
            {
                "event": event_name,
                "date": date.date(),
                "before_mean": before["domestic_price"].mean(),
                "after_mean": after["domestic_price"].mean(),
                "peak_change_pct": (peak / base - 1) * 100,
                "trough_change_pct": (trough / base - 1) * 100,
                "duration_days": len(after),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(PATHS.event_windows, index=False)
    return result


def run_eda(df: pd.DataFrame):
    print("\n[2] EDA/가설 검증")
    summary = df.describe().T
    summary["missing"] = df.isna().sum()
    summary.to_csv(PATHS.summary)

    lag_df = lag_correlation(df)
    lag_df.to_csv(OUT_DIR / "lag_correlation.csv", index=False)
    event_df = event_window_analysis(df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes[0, 0].plot(df.index, df["domestic_price"], label="국내 유가")
    axes[0, 0].set_title("국내 유가 시계열")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df.index, df["wti"], label="WTI", color="tab:orange")
    axes[0, 1].plot(df.index, df["brent"], label="Brent", color="tab:green", alpha=0.8)
    axes[0, 1].set_title("국제 유가 WTI/Brent")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(df.index, df["risk_index"], label="리스크 지수", color="tab:red")
    axes[1, 0].set_title("지정학적 리스크 지수")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(lag_df["lag_days"], lag_df["wti_to_domestic_corr"], label="WTI lag")
    axes[1, 1].plot(lag_df["lag_days"], lag_df["brent_to_domestic_corr"], label="Brent lag")
    axes[1, 1].plot(lag_df["lag_days"], lag_df["risk_to_domestic_corr"], label="Risk lag")
    axes[1, 1].set_title("시차 상관 분석")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PATHS.figures / "eda_overview.png", dpi=160)
    plt.close(fig)

    if not event_df.empty:
        event_df.plot(x="event", y=["peak_change_pct", "trough_change_pct"], kind="bar", figsize=(12, 6))
        plt.title("사건 전후 유가 변동 폭")
        plt.ylabel("변동률(%)")
        plt.tight_layout()
        plt.savefig(PATHS.figures / "event_window_changes.png", dpi=160)
        plt.close()

    analysis = df.copy()
    analysis["domestic_return"] = analysis["domestic_price"].pct_change() * 100
    analysis["wti_return"] = analysis["wti"].pct_change() * 100
    analysis["brent_return"] = analysis["brent"].pct_change() * 100
    analysis = analysis.dropna()

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        analysis["wti_return"],
        analysis["domestic_return"],
        c=analysis["exchange"],
        cmap="viridis",
        s=28,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.3,
    )
    plt.axhline(0, color="#777777", linestyle="--", linewidth=1)
    plt.axvline(0, color="#777777", linestyle="--", linewidth=1)
    plt.title("WTI 변동률과 국내 유가 변동률 산점도")
    plt.xlabel("WTI 일간 변동률(%)")
    plt.ylabel("국내 유가 일간 변동률(%)")
    plt.colorbar(scatter, label="원/달러 환율")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PATHS.figures / "scatter_wti_domestic.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.hist(analysis["domestic_return"], bins=34, alpha=0.68, label="국내 유가")
    plt.hist(analysis["wti_return"], bins=34, alpha=0.48, label="WTI")
    plt.hist(analysis["brent_return"], bins=34, alpha=0.48, label="Brent")
    plt.axvline(0, color="#777777", linestyle="--", linewidth=1)
    plt.title("일간 변동률 히스토그램")
    plt.xlabel("일간 변동률(%)")
    plt.ylabel("빈도")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PATHS.figures / "histogram_daily_returns.png", dpi=160)
    plt.close()

    labels = {
        "domestic_price": "국내 유가",
        "wti": "WTI",
        "brent": "Brent",
        "exchange": "원/달러",
        "risk_index": "고정 리스크",
        "news_risk_index": "뉴스 리스크",
        "volatility_7d": "7일 변동성",
    }
    columns = [column for column in labels if column in df.columns]
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("주요 변수 상관관계 히트맵")
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels([labels[column] for column in columns], rotation=35, ha="right")
    ax.set_yticklabels([labels[column] for column in columns])
    for row in range(len(columns)):
        for col in range(len(columns)):
            value = corr.iloc[row, col]
            color = "white" if abs(value) >= 0.55 else "#333333"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, label="상관계수")
    fig.tight_layout()
    fig.savefig(PATHS.figures / "correlation_heatmap.png", dpi=160)
    plt.close(fig)
