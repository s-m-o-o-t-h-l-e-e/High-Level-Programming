from datetime import timedelta

import numpy as np
import pandas as pd

from config import EVENTS, OUT_DIR, PATHS
from plot_style import plt

FUEL_COLUMNS = [
    ("gasoline_price", "휘발유", "#2563eb"),
    ("diesel_price", "경유", "#16a34a"),
    ("lpg_price", "LPG", "#7c3aed"),
]


def ensure_fuel_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "gasoline_price" not in df.columns and "domestic_price" in df.columns:
        df["gasoline_price"] = df["domestic_price"]

    if "domestic_price" not in df.columns and "gasoline_price" in df.columns:
        df["domestic_price"] = df["gasoline_price"]

    if "diesel_price" not in df.columns and "gasoline_price" in df.columns:
        df["diesel_price"] = df["gasoline_price"] * 0.92

    if "lpg_price" not in df.columns and "gasoline_price" in df.columns:
        df["lpg_price"] = df["gasoline_price"] * 0.54

    return df


def lag_correlation(df: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    rows = []

    gasoline_returns = df["gasoline_price"].pct_change()

    for lag in range(max_lag + 1):
        row = {"lag_days": lag}

        row["wti_to_gasoline_corr"] = df["wti"].pct_change().shift(lag).corr(gasoline_returns)
        row["brent_to_gasoline_corr"] = df["brent"].pct_change().shift(lag).corr(gasoline_returns)
        row["risk_to_gasoline_corr"] = df["risk_index"].shift(lag).corr(gasoline_returns.abs())

        for column, _name, _color in FUEL_COLUMNS[1:]:
            returns = df[column].pct_change()

            row[f"wti_to_{column}_corr"] = df["wti"].pct_change().shift(lag).corr(returns)
            row[f"brent_to_{column}_corr"] = df["brent"].pct_change().shift(lag).corr(returns)
            row[f"risk_to_{column}_corr"] = df["risk_index"].shift(lag).corr(returns.abs())

        rows.append(row)

    return pd.DataFrame(rows)


def make_best_lag_summary(lag_df: pd.DataFrame) -> pd.DataFrame:
    target_columns = [
        ("wti_to_gasoline_corr", "WTI→휘발유", "#2563eb"),
        ("brent_to_gasoline_corr", "Brent→휘발유", "#1e40af"),
        ("wti_to_diesel_price_corr", "WTI→경유", "#16a34a"),
        ("brent_to_diesel_price_corr", "Brent→경유", "#166534"),
        ("wti_to_lpg_price_corr", "WTI→LPG", "#7c3aed"),
        ("brent_to_lpg_price_corr", "Brent→LPG", "#581c87"),
    ]

    rows = []

    for column, label, color in target_columns:
        if column not in lag_df.columns:
            continue

        series = pd.to_numeric(lag_df[column], errors="coerce").dropna()

        if series.empty:
            continue

        best_idx = series.abs().idxmax()

        rows.append(
            {
                "relation": label,
                "best_lag": int(lag_df.loc[best_idx, "lag_days"]),
                "max_corr": float(lag_df.loc[best_idx, column]),
                "abs_corr": abs(float(lag_df.loc[best_idx, column])),
                "color": color,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values("abs_corr", ascending=True)

    return result


def event_window_analysis(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    rows = []

    for date_text, event_name, _weight in EVENTS:
        date = pd.Timestamp(date_text)

        before = df.loc[date - timedelta(days=window): date - timedelta(days=1)]
        after = df.loc[date: date + timedelta(days=window)]

        if before.empty or after.empty:
            continue

        for column, fuel_label, _color in FUEL_COLUMNS:
            base = before[column].iloc[-1]
            peak = after[column].max()
            trough = after[column].min()
            peak_change_pct = (peak / base - 1) * 100
            trough_change_pct = (trough / base - 1) * 100

            rows.append(
                {
                    "event": event_name,
                    "fuel_type": fuel_label,
                    "date": date.date(),
                    "before_mean": before[column].mean(),
                    "after_mean": after[column].mean(),
                    "peak_change_pct": peak_change_pct,
                    "trough_change_pct": trough_change_pct,
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


def _trim_for_plot(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 20:
        return values
    lower = values.quantile(lower_q)
    upper = values.quantile(upper_q)
    return values.clip(lower, upper)


def _save_distribution_boxplot(df: pd.DataFrame) -> None:
    columns = [
        ("gasoline_price", "휘발유", "원/L"),
        ("diesel_price", "경유", "원/L"),
        ("lpg_price", "LPG", "원/L"),
        ("wti", "WTI", "달러/배럴"),
        ("brent", "Brent", "달러/배럴"),
        ("exchange", "원/달러", "원"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

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
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "#f59e0b",
                "markeredgecolor": "#f59e0b",
                "alpha": 0.55,
            },
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
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    plotted = False

    for ax, (column, label, _line_color) in zip(axes, FUEL_COLUMNS):
        price = _numeric_series(df, column)
        changes = price.diff().dropna().tail(14)

        if changes.empty:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
            continue

        plotted = True

        colors = [
            "#dc2626" if value > 0 else "#2563eb" if value < 0 else "#6b7280"
            for value in changes
        ]

        labels = [
            index.strftime("%m.%d") if hasattr(index, "strftime") else str(index)
            for index in changes.index
        ]

        ax.bar(labels, changes.values, color=colors, alpha=0.86)
        ax.axhline(0, color="#111827", linewidth=1)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylabel("원/L")
        ax.grid(axis="y", alpha=0.25)

    if not plotted:
        plt.close(fig)
        return

    axes[-1].set_xlabel("날짜")
    fig.suptitle("최근 14일 유종별 일간 변화 막대그래프", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PATHS.figures / "bar_recent_changes.png", dpi=160)
    plt.close(fig)


def _kde_curve(values: pd.Series, points: int = 240):
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
    density = np.exp(-0.5 * scaled**2).sum(axis=1) / (
        len(array) * bandwidth * np.sqrt(2 * np.pi)
    )

    return grid, density


def _save_return_kde(analysis: pd.DataFrame) -> None:
    series_info = [
        ("gasoline_return", "휘발유", "#2563eb"),
        ("diesel_return", "경유", "#16a34a"),
        ("lpg_return", "LPG", "#7c3aed"),
        ("wti_return", "WTI", "#f97316"),
        ("brent_return", "Brent", "#111827"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(11, 11.5), sharex=False)
    plotted = False

    for ax, (column, label, color) in zip(axes, series_info):
        values = _trim_for_plot(_numeric_series(analysis, column))
        grid, density = _kde_curve(values)

        if grid is None or density is None:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, loc="left", fontweight="bold")
            continue

        ax.plot(grid, density, color=color, linewidth=2.2, label=label)
        ax.fill_between(grid, density, color=color, alpha=0.16)
        ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylabel("밀도")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.22)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    axes[-1].set_xlabel("일간 변동률(%)")
    fig.suptitle("유종별/국제유가 일간 변동률 밀도 그래프", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PATHS.figures / "kde_return_distribution.png", dpi=160)
    plt.close(fig)


def _save_market_violin(df: pd.DataFrame) -> None:
    columns = [
        ("gasoline_price", "휘발유"),
        ("diesel_price", "경유"),
        ("lpg_price", "LPG"),
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


def _draw_best_lag_line(ax, best_lag_df: pd.DataFrame) -> None:
    if best_lag_df.empty:
        ax.text(0.5, 0.5, "시차 상관 데이터 없음", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ordered = best_lag_df.sort_values("best_lag").reset_index(drop=True)
    x = np.arange(len(ordered))
    ax.plot(x, ordered["best_lag"], color="#9996e2", linewidth=2.4, marker="o", markersize=7)
    for idx, row in ordered.iterrows():
        ax.scatter(idx, row["best_lag"], color=row["color"], s=70, zorder=3)
        ax.text(idx, row["best_lag"] + 0.7, f"r={row['max_corr']:.3f}", ha="center", fontsize=8.5, color="#111827")

    ax.set_title("국제유가 → 국내유가 최대 상관 시차")
    ax.set_xlabel("관계")
    ax.set_ylabel("최대 상관 시차(일)")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["relation"], rotation=25, ha="right")
    ax.set_ylim(0, max(float(ordered["best_lag"].max()) + 4, 5))
    ax.grid(axis="y", alpha=0.3)


def run_eda(df: pd.DataFrame):
    print("\n[2] EDA/가설 검증")

    df = ensure_fuel_columns(df)

    summary = df.describe().T
    summary["missing"] = df.isna().sum()
    summary.to_csv(PATHS.summary, index_label="metric")

    lag_df = lag_correlation(df)
    lag_df.to_csv(OUT_DIR / "lag_correlation.csv", index=False)

    best_lag_df = make_best_lag_summary(lag_df)
    best_lag_df.to_csv(OUT_DIR / "lag_correlation_summary.csv", index=False)

    event_df = event_window_analysis(df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for column, label, color in FUEL_COLUMNS:
        axes[0, 0].plot(df.index, df[column], label=label, color=color, linewidth=1.8)

    axes[0, 0].set_title("유종별 국내 유가 시계열")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(df.index, df["wti"], label="WTI", color="#f97316", linewidth=1.8)
    axes[0, 1].plot(df.index, df["brent"], label="Brent", color="#111827", linewidth=1.8, alpha=0.85)
    axes[0, 1].set_title("국제 유가 WTI/Brent")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(df.index, df["risk_index"], label="리스크 지수", color="#dc2626", linewidth=1.8)
    axes[1, 0].set_title("지정학적 리스크 지수")
    axes[1, 0].grid(alpha=0.3)

    _draw_best_lag_line(axes[1, 1], best_lag_df)

    fig.tight_layout()
    fig.savefig(PATHS.figures / "eda_overview.png", dpi=160)
    plt.close(fig)

    if not event_df.empty:
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

        for ax, (_column, fuel_label, color) in zip(axes, FUEL_COLUMNS):
            subset = event_df.loc[event_df["fuel_type"] == fuel_label]

            if subset.empty:
                continue

            x = np.arange(len(subset))
            peak_values = subset["peak_change_pct"]
            trough_values = subset["trough_change_pct"]

            peak_bars = ax.bar(
                x - 0.18,
                peak_values,
                width=0.36,
                color=color,
                alpha=0.82,
                label="최고 변화율",
            )

            trough_bars = ax.bar(
                x + 0.18,
                trough_values,
                width=0.36,
                color="#6b7280",
                alpha=0.62,
                label="최저 변화율",
            )

            ax.axhline(0, color="#111827", linewidth=1)
            ax.set_title(fuel_label, loc="left", fontweight="bold")
            ax.set_ylabel("변동률(%)")
            ax.set_xticks(x)
            ax.set_xticklabels(subset["event"], rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="best")

            for bars, values in ((peak_bars, peak_values), (trough_bars, trough_values)):
                for bar, value in zip(bars, values):
                    value = float(value)
                    if not np.isfinite(value):
                        continue

                    va = "bottom" if value >= 0 else "top"
                    offset = 0.25 if value >= 0 else -0.25
                    label = f"{value:.3f}%" if abs(value) < 0.1 else f"{value:.1f}%"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + offset,
                        label,
                        ha="center",
                        va=va,
                        fontsize=8,
                        color="#374151",
                    )

        fig.suptitle("사건 전후 유종별 유가 변동 폭", fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(PATHS.figures / "event_window_changes.png", dpi=160)
        plt.close(fig)

    analysis = df.copy()

    analysis["gasoline_return"] = analysis["gasoline_price"].pct_change() * 100
    analysis["diesel_return"] = analysis["diesel_price"].pct_change() * 100
    analysis["lpg_return"] = analysis["lpg_price"].pct_change() * 100
    analysis["domestic_return"] = analysis["gasoline_return"]
    analysis["wti_return"] = analysis["wti"].pct_change() * 100
    analysis["brent_return"] = analysis["brent"].pct_change() * 100
    analysis = analysis.dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, constrained_layout=True)
    scatter = None

    for ax, (return_col, label, _color) in zip(
        axes,
        [
            ("gasoline_return", "휘발유", "#2563eb"),
            ("diesel_return", "경유", "#16a34a"),
            ("lpg_return", "LPG", "#7c3aed"),
        ],
    ):
        scatter = ax.scatter(
            analysis["wti_return"],
            analysis[return_col],
            c=analysis["exchange"],
            cmap="viridis",
            s=24,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.3,
        )

        ax.axhline(0, color="#777777", linestyle="--", linewidth=1)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(f"WTI vs {label}")
        ax.set_xlabel("WTI 일간 변동률(%)")
        ax.set_ylabel(f"{label} 일간 변동률(%)")
        ax.grid(alpha=0.25)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="원/달러 환율", shrink=0.88, pad=0.02)

    fig.suptitle("WTI 변동률과 유종별 국내 유가 변동률 산점도", fontsize=16, fontweight="bold")
    fig.savefig(PATHS.figures / "scatter_wti_domestic.png", dpi=160)
    plt.close(fig)

    hist_info = [
        ("gasoline_return", "휘발유", "#2563eb"),
        ("diesel_return", "경유", "#16a34a"),
        ("lpg_return", "LPG", "#7c3aed"),
        ("wti_return", "WTI", "#f97316"),
        ("brent_return", "Brent", "#111827"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(11, 11.5), sharex=False)
    for ax, (column, label, color) in zip(axes, hist_info):
        values = _trim_for_plot(_numeric_series(analysis, column))
        if values.empty:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label, loc="left", fontweight="bold")
            continue
        ax.hist(values, bins=34, alpha=0.82, label=label, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylabel("빈도")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("일간 변동률(%)")
    fig.suptitle("유종별/국제유가 일간 변동률 히스토그램", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PATHS.figures / "histogram_daily_returns.png", dpi=160)
    plt.close(fig)

    _save_distribution_boxplot(df)
    _save_recent_change_bar(df)
    _save_return_kde(analysis)
    _save_market_violin(df)

    labels = {
        "gasoline_price": "휘발유",
        "diesel_price": "경유",
        "lpg_price": "LPG",
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
