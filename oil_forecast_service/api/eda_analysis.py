from datetime import timedelta

import numpy as np
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


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _save_distribution_boxplot(df: pd.DataFrame) -> None:
    columns = [
        ("domestic_price", "국내 유가", "원/L"),
        ("wti", "WTI", "달러/배럴"),
        ("brent", "Brent", "달러/배럴"),
        ("exchange", "원/달러", "원"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (column, label, unit) in zip(axes.ravel(), columns):
        series = _numeric_series(df, column)
        if series.empty:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.boxplot(
            series,
            patch_artist=True,
            widths=0.45,
            boxprops={"facecolor": "#9996e2", "alpha": 0.55, "edgecolor": "#4f46a8"},
            medianprops={"color": "#111827", "linewidth": 2},
            whiskerprops={"color": "#4b5563"},
            capprops={"color": "#4b5563"},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#f59e0b", "markeredgecolor": "#f59e0b", "alpha": 0.55},
        )
        ax.set_title(f"{label} 분포")
        ax.set_ylabel(unit)
        ax.set_xticks([1])
        ax.set_xticklabels([label])
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("주요 가격/지표 분포 박스플롯", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PATHS.figures / "boxplot_price_distribution.png", dpi=160)
    plt.close(fig)


def _save_recent_change_bar(df: pd.DataFrame) -> None:
    price = _numeric_series(df, "domestic_price")
    if price.empty:
        return
    changes = price.diff().dropna().tail(14)
    if changes.empty:
        return

    colors = ["#dc2626" if value > 0 else "#2563eb" if value < 0 else "#6b7280" for value in changes]
    labels = [index.strftime("%m.%d") if hasattr(index, "strftime") else str(index) for index in changes.index]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, changes.values, color=colors, alpha=0.86)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("최근 14일 국내 유가 일간 변화 막대그래프")
    ax.set_xlabel("날짜")
    ax.set_ylabel("전일 대비 변화(원/L)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS.figures / "bar_recent_changes.png", dpi=160)
    plt.close(fig)


def _kde_curve(values: pd.Series, points: int = 240) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) < 3:
        return None, None
    std = array.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return None, None
    bandwidth = 1.06 * std * (len(array) ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return None, None
    padding = std * 0.6
    grid = np.linspace(array.min() - padding, array.max() + padding, points)
    scaled = (grid[:, None] - array[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1) / (len(array) * bandwidth * np.sqrt(2 * np.pi))
    return grid, density


def _save_return_kde(analysis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = False
    series_info = [
        ("domestic_return", "국내 유가", "#5b82f1"),
        ("wti_return", "WTI", "#f59e0b"),
        ("brent_return", "Brent", "#16a34a"),
    ]
    for column, label, color in series_info:
        grid, density = _kde_curve(_numeric_series(analysis, column))
        if grid is None or density is None:
            continue
        ax.plot(grid, density, color=color, linewidth=2.2, label=label)
        ax.fill_between(grid, density, color=color, alpha=0.12)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title("일간 변동률 밀도 그래프")
    ax.set_xlabel("일간 변동률(%)")
    ax.set_ylabel("밀도")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(PATHS.figures / "kde_return_distribution.png", dpi=160)
    plt.close(fig)


def _save_market_violin(df: pd.DataFrame) -> None:
    columns = [
        ("domestic_price", "국내 유가"),
        ("wti", "WTI"),
        ("brent", "Brent"),
        ("exchange", "원/달러"),
        ("news_risk_index", "뉴스 리스크"),
    ]
    data = []
    labels = []
    for column, label in columns:
        series = _numeric_series(df, column)
        if len(series) < 3:
            continue
        std = series.std()
        if not np.isfinite(std) or std == 0:
            continue
        data.append(((series - series.mean()) / std).to_numpy())
        labels.append(label)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 6.8))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#9996e2")
        body.set_edgecolor("#4f46a8")
        body.set_alpha(0.5)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#111827")
            parts[key].set_linewidth(1.4)
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title("주요 시장 지표 표준화 분포 바이올린 플롯")
    ax.set_ylabel("표준화 값(z-score)")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS.figures / "violin_market_distribution.png", dpi=160)
    plt.close(fig)


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

    _save_distribution_boxplot(df)
    _save_recent_change_bar(df)
    _save_return_kde(analysis)
    _save_market_violin(df)

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
