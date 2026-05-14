import tkinter as tk
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / "matplotlib_cache"))
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / "outputs" / "matplotlib_cache")

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


PERIODS = [
    ("1w", "1주", 7),
    ("1m", "1개월", 30),
    ("1y", "1년", 365),
    ("3y", "3년", 365 * 3),
]


def _line_style(days: int):
    if days <= 7:
        return {"linewidth": 2.2, "markersize": 3.8, "markevery": 1}
    if days <= 30:
        return {"linewidth": 2.0, "markersize": 3.0, "markevery": 2}
    if days <= 365:
        return {"linewidth": 1.6, "markersize": 0, "markevery": None}
    return {"linewidth": 1.25, "markersize": 0, "markevery": None}


def _moving_average(series: pd.Series, days: int) -> pd.Series:
    if days <= 7:
        return series.dropna()
    if days <= 30:
        return series.rolling("1D", min_periods=1).mean()
    if days <= 365:
        return series.rolling("3D", min_periods=1).mean()
    return series.rolling("7D", min_periods=1).mean()


def _break_long_time_gaps(series: pd.Series, max_gap: str = "3h") -> pd.Series:
    series = series.dropna().copy()
    if len(series) < 2:
        return series

    gap = series.index.to_series().diff()
    break_points = gap > pd.Timedelta(max_gap)
    if not break_points.any():
        return series

    result = series.copy()
    result.loc[break_points] = pd.NA
    return result


def _download_intraday_oil_prices() -> pd.DataFrame:
    try:
        import yfinance as yf

        data = yf.download(
            ["CL=F", "BZ=F"],
            period="7d",
            interval="1h",
            progress=False,
            auto_adjust=False,
            threads=True,
        )
    except Exception as exc:
        print(f"1주 국제유가 시간봉 다운로드 실패, 일별 데이터 사용: {exc}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return pd.DataFrame()
        close = data["Close"].copy()
    elif "Close" in data.columns:
        close = data[["Close"]].rename(columns={"Close": "wti"})
    else:
        return pd.DataFrame()

    close = close.rename(columns={"CL=F": "wti", "BZ=F": "brent"})
    columns = [column for column in ["wti", "brent"] if column in close.columns]
    if not columns:
        return pd.DataFrame()

    close = close[columns].dropna(how="all").ffill()
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert("Asia/Seoul").tz_localize(None)
    return close


def _plot_series(ax, index, values, color: str, label: str, style: dict):
    marker = "o" if style["markersize"] else None
    ax.plot(
        index,
        values,
        color=color,
        marker=marker,
        markersize=style["markersize"],
        markevery=style["markevery"],
        linewidth=style["linewidth"],
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
        label=label,
    )


def _format_axis(ax, days: int):
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


def _format_intraday_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["domestic_return"] = frame["domestic_price"].pct_change() * 100
    frame["wti_return"] = frame["wti"].pct_change() * 100
    frame["brent_return"] = frame["brent"].pct_change() * 100
    frame["exchange_return"] = frame["exchange"].pct_change() * 100
    return frame.replace([float("inf"), float("-inf")], pd.NA).dropna()


class OilPriceTrendApp:
    def __init__(self, raw: pd.DataFrame, forecast: pd.DataFrame, today: pd.Timestamp):
        self.raw = raw
        self.forecast = forecast.copy()
        self.forecast["date"] = pd.to_datetime(self.forecast["date"])
        self.today = today
        self.today_price = float(raw["domestic_price"].iloc[-1])
        self.active_period = "1w"
        self.buttons = {}

        self.root = tk.Tk()
        self.root.title("유가추이")
        self.root.geometry("980x860")
        self.root.configure(bg="white")

        self._build_layout()
        self.draw_chart("1w")

    def _build_layout(self):
        main = tk.Frame(self.root, bg="white", padx=48, pady=36)
        main.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(main, text="유가추이", font=("AppleGothic", 38, "bold"), bg="white", fg="#333333", anchor="w")
        title.pack(fill=tk.X)

        rule = tk.Frame(main, bg="#111111", height=3)
        rule.pack(fill=tk.X, pady=(34, 42))

        tabs = tk.Frame(main, bg="white", highlightbackground="#666666", highlightthickness=1)
        tabs.pack(fill=tk.X, pady=(0, 36))

        for code, label, _days in PERIODS:
            button = tk.Button(
                tabs,
                text=label,
                command=lambda period=code: self.draw_chart(period),
                bd=1,
                relief=tk.SOLID,
                font=("AppleGothic", 24, "bold"),
                cursor="hand2",
                height=1,
            )
            button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.buttons[code] = button

        self.figure = Figure(figsize=(8.6, 5.2), dpi=100, facecolor="white")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        legend = tk.Frame(main, bg="white", highlightbackground="#dddddd", highlightthickness=1, padx=20, pady=8)
        legend.pack(pady=(18, 0))
        self._legend_item(legend, "#5f8ffb", "국내 유가").pack(side=tk.LEFT, padx=18)
        self._legend_item(legend, "#f2a000", "7일 예측").pack(side=tk.LEFT, padx=18)
        self._legend_item(legend, "#e52525", "오늘", dot_only=True).pack(side=tk.LEFT, padx=18)

    def _legend_item(self, parent, color: str, text: str, dot_only: bool = False):
        item = tk.Frame(parent, bg="white")
        sample = tk.Canvas(item, width=42, height=18, bg="white", highlightthickness=0)
        if dot_only:
            sample.create_oval(14, 4, 27, 17, fill=color, outline=color)
        else:
            sample.create_line(2, 9, 40, 9, fill=color, width=4)
            sample.create_oval(31, 4, 42, 15, fill=color, outline=color)
        sample.pack(side=tk.LEFT)
        tk.Label(item, text=text, bg="white", fg="#777777", font=("AppleGothic", 17, "bold")).pack(side=tk.LEFT, padx=(8, 0))
        return item

    def _set_active_button(self, period: str):
        for code, button in self.buttons.items():
            if code == period:
                button.configure(bg="#6b6b6b", fg="white", activebackground="#6b6b6b", activeforeground="white")
            else:
                button.configure(bg="white", fg="#666666", activebackground="#eeeeee", activeforeground="#333333")

    def draw_chart(self, period: str):
        self.active_period = period
        self._set_active_button(period)
        days = dict((code, days) for code, _label, days in PERIODS)[period]
        history = self.raw.loc[self.raw.index >= self.today - pd.Timedelta(days=days)].copy()
        forecast_dates = self.forecast["date"]
        forecast_prices = self.forecast["predicted_domestic_price"]
        style = _line_style(days)

        self.ax.clear()
        domestic_ma = _moving_average(history["domestic_price"], days)
        _plot_series(self.ax, domestic_ma.index, domestic_ma, "#5f8ffb", "국내 유가", style)
        self.ax.plot(
            [self.today] + list(forecast_dates),
            [self.today_price] + list(forecast_prices),
            color="#f2a000",
            marker="o",
            markersize=4.2,
            linewidth=2.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
        self.ax.axvline(self.today, color="#999999", linestyle="--", linewidth=1.6)
        self.ax.scatter([self.today], [self.today_price], color="#e52525", s=75, zorder=5)
        self.ax.text(self.today, self.today_price, f" 오늘 {self.today_price:,.1f}", va="bottom", fontsize=12, color="#333333")

        self.ax.set_ylabel("원/L", fontsize=14, fontweight="bold")
        self.ax.grid(axis="y", color="#e8e8e8", linewidth=1)
        self.ax.minorticks_on()
        self.ax.grid(axis="y", which="minor", color="#f3f3f3", linewidth=0.6)
        self.ax.grid(axis="x", visible=False)
        _format_axis(self.ax, days)
        self.ax.tick_params(axis="x", labelsize=12, colors="#777777")
        self.ax.tick_params(axis="y", labelsize=12, colors="#777777")

        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)
        self.ax.spines["bottom"].set_linewidth(4)
        self.ax.spines["bottom"].set_color("#666666")
        self.ax.spines["left"].set_color("#e5e5e5")

        self.figure.tight_layout()
        self.canvas.draw()

    def run(self):
        self.root.mainloop()


def launch_oil_price_gui(raw: pd.DataFrame, forecast: pd.DataFrame, today: pd.Timestamp):
    app = OilPriceTrendApp(raw, forecast, today)
    app.run()


class IndicatorTrendApp:
    def __init__(self, raw: pd.DataFrame, today: pd.Timestamp):
        self.raw = raw
        self.today = today
        self.active_period = "1m"
        self.active_indicator = "oil"
        self.intraday_oil = _download_intraday_oil_prices()
        self.period_buttons = {}
        self.indicator_buttons = {}

        self.root = tk.Tk()
        self.root.title("보조지표 추이")
        self.root.geometry("1080x860")
        self.root.configure(bg="white")

        self._build_layout()
        self.draw_chart()

    def _build_layout(self):
        main = tk.Frame(self.root, bg="white", padx=48, pady=36)
        main.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(main, text="보조지표 추이", font=("AppleGothic", 36, "bold"), bg="white", fg="#333333", anchor="w")
        title.pack(fill=tk.X)

        rule = tk.Frame(main, bg="#111111", height=3)
        rule.pack(fill=tk.X, pady=(30, 32))

        indicator_tabs = tk.Frame(main, bg="white", highlightbackground="#666666", highlightthickness=1)
        indicator_tabs.pack(fill=tk.X, pady=(0, 18))
        for code, label in [
            ("oil", "WTI/Brent"),
            ("exchange", "원/달러"),
            ("risk", "뉴스/리스크"),
        ]:
            button = tk.Button(
                indicator_tabs,
                text=label,
                command=lambda value=code: self.set_indicator(value),
                bd=1,
                relief=tk.SOLID,
                font=("AppleGothic", 20, "bold"),
                cursor="hand2",
            )
            button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.indicator_buttons[code] = button

        period_tabs = tk.Frame(main, bg="white", highlightbackground="#666666", highlightthickness=1)
        period_tabs.pack(fill=tk.X, pady=(0, 30))
        for code, label, _days in PERIODS:
            button = tk.Button(
                period_tabs,
                text=label,
                command=lambda value=code: self.set_period(value),
                bd=1,
                relief=tk.SOLID,
                font=("AppleGothic", 18, "bold"),
                cursor="hand2",
            )
            button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.period_buttons[code] = button

        self.figure = Figure(figsize=(9.5, 5.6), dpi=100, facecolor="white")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _set_active_buttons(self):
        for code, button in self.indicator_buttons.items():
            selected = code == self.active_indicator
            button.configure(
                bg="#6b6b6b" if selected else "white",
                fg="white" if selected else "#666666",
                activebackground="#6b6b6b" if selected else "#eeeeee",
                activeforeground="white" if selected else "#333333",
            )
        for code, button in self.period_buttons.items():
            selected = code == self.active_period
            button.configure(
                bg="#6b6b6b" if selected else "white",
                fg="white" if selected else "#666666",
                activebackground="#6b6b6b" if selected else "#eeeeee",
                activeforeground="white" if selected else "#333333",
            )

    def set_period(self, period: str):
        self.active_period = period
        self.draw_chart()

    def set_indicator(self, indicator: str):
        self.active_indicator = indicator
        self.draw_chart()

    def draw_chart(self):
        self._set_active_buttons()
        days = dict((code, days) for code, _label, days in PERIODS)[self.active_period]
        frame = self.raw.loc[self.raw.index >= self.today - pd.Timedelta(days=days)].copy()
        style = _line_style(days)
        self.ax.clear()

        if self.active_indicator == "oil":
            use_intraday = days <= 7 and not self.intraday_oil.empty
            oil_frame = self.intraday_oil if use_intraday else frame
            if use_intraday:
                latest = oil_frame.index.max()
                oil_frame = oil_frame.loc[oil_frame.index >= latest - pd.Timedelta(days=7)].copy()
                style = {"linewidth": 1.8, "markersize": 0, "markevery": None}

            wti_ma = _moving_average(oil_frame["wti"], days) if "wti" in oil_frame else pd.Series(dtype=float)
            brent_ma = _moving_average(oil_frame["brent"], days) if "brent" in oil_frame else pd.Series(dtype=float)
            if use_intraday:
                wti_ma = _break_long_time_gaps(wti_ma)
                brent_ma = _break_long_time_gaps(brent_ma)
            _plot_series(self.ax, wti_ma.index, wti_ma, "#f97316", "WTI", style)
            _plot_series(self.ax, brent_ma.index, brent_ma, "#16a34a", "Brent", style)
            self.ax.set_ylabel("달러/배럴", fontsize=13, fontweight="bold")
            self.ax.set_title("국제 유가 WTI/Brent", fontsize=16, fontweight="bold")
        elif self.active_indicator == "exchange":
            exchange_ma = _moving_average(frame["exchange"], days)
            _plot_series(self.ax, exchange_ma.index, exchange_ma, "#2563eb", "원/달러 환율", style)
            self.ax.set_ylabel("원/달러", fontsize=13, fontweight="bold")
            self.ax.set_title("원/달러 환율", fontsize=16, fontweight="bold")
        else:
            risk_ma = _moving_average(frame["risk_index"], days)
            news_ma = _moving_average(frame["news_risk_index"], days)
            _plot_series(self.ax, risk_ma.index, risk_ma, "#dc2626", "고정 이벤트 리스크", style)
            _plot_series(self.ax, news_ma.index, news_ma, "#7c3aed", "뉴스 리스크", style)
            self.ax.set_ylabel("리스크 지수", fontsize=13, fontweight="bold")
            self.ax.set_title("뉴스/지정학 리스크", fontsize=16, fontweight="bold")

        self.ax.grid(axis="y", color="#e8e8e8", linewidth=1)
        self.ax.minorticks_on()
        self.ax.grid(axis="y", which="minor", color="#f3f3f3", linewidth=0.6)
        self.ax.grid(axis="x", visible=False)
        if self.active_indicator == "oil" and days <= 7 and not self.intraday_oil.empty:
            _format_intraday_axis(self.ax)
        else:
            _format_axis(self.ax, days)
        self.ax.tick_params(axis="x", labelsize=11, colors="#777777")
        self.ax.tick_params(axis="y", labelsize=11, colors="#777777")
        self.ax.legend(loc="upper left", frameon=True, edgecolor="#dddddd")

        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)
        self.ax.spines["bottom"].set_linewidth(4)
        self.ax.spines["bottom"].set_color("#666666")
        self.ax.spines["left"].set_color("#e5e5e5")

        self.figure.tight_layout()
        self.canvas.draw()

    def run(self):
        self.root.mainloop()


def launch_indicator_gui(raw: pd.DataFrame, today: pd.Timestamp):
    app = IndicatorTrendApp(raw, today)
    app.run()


class AnalysisGraphApp:
    def __init__(self, raw: pd.DataFrame):
        self.raw = raw
        self.analysis = _analysis_frame(raw)
        self.active_graph = "scatter"
        self.buttons = {}

        self.root = tk.Tk()
        self.root.title("분석 그래프")
        self.root.geometry("1080x860")
        self.root.configure(bg="white")

        self._build_layout()
        self.draw_chart()

    def _build_layout(self):
        main = tk.Frame(self.root, bg="white", padx=48, pady=36)
        main.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(main, text="분석 그래프", font=("AppleGothic", 36, "bold"), bg="white", fg="#333333", anchor="w")
        title.pack(fill=tk.X)

        rule = tk.Frame(main, bg="#111111", height=3)
        rule.pack(fill=tk.X, pady=(30, 32))

        tabs = tk.Frame(main, bg="white", highlightbackground="#666666", highlightthickness=1)
        tabs.pack(fill=tk.X, pady=(0, 30))
        for code, label in [
            ("scatter", "산점도"),
            ("histogram", "히스토그램"),
            ("heatmap", "히트맵"),
        ]:
            button = tk.Button(
                tabs,
                text=label,
                command=lambda value=code: self.set_graph(value),
                bd=1,
                relief=tk.SOLID,
                font=("AppleGothic", 20, "bold"),
                cursor="hand2",
            )
            button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.buttons[code] = button

        self.figure = Figure(figsize=(9.5, 5.8), dpi=100, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _set_active_button(self):
        for code, button in self.buttons.items():
            selected = code == self.active_graph
            button.configure(
                bg="#6b6b6b" if selected else "white",
                fg="white" if selected else "#666666",
                activebackground="#6b6b6b" if selected else "#eeeeee",
                activeforeground="white" if selected else "#333333",
            )

    def set_graph(self, graph: str):
        self.active_graph = graph
        self.draw_chart()

    def draw_chart(self):
        self._set_active_button()
        self.figure.clear()

        if self.active_graph == "scatter":
            self._draw_scatter()
        elif self.active_graph == "histogram":
            self._draw_histogram()
        else:
            self._draw_heatmap()

        self.figure.tight_layout()
        self.canvas.draw()

    def _draw_scatter(self):
        ax = self.figure.add_subplot(111)
        frame = self.analysis.dropna(subset=["wti_return", "domestic_return", "exchange"])
        scatter = ax.scatter(
            frame["wti_return"],
            frame["domestic_return"],
            c=frame["exchange"],
            cmap="viridis",
            s=34,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
        ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
        ax.set_title("WTI 변동률과 국내 유가 변동률", fontsize=16, fontweight="bold")
        ax.set_xlabel("WTI 일간 변동률(%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("국내 유가 일간 변동률(%)", fontsize=12, fontweight="bold")
        ax.grid(color="#e8e8e8", linewidth=1)
        colorbar = self.figure.colorbar(scatter, ax=ax)
        colorbar.set_label("원/달러 환율", fontsize=11, fontweight="bold")

    def _draw_histogram(self):
        ax = self.figure.add_subplot(111)
        frame = self.analysis
        bins = 34
        ax.hist(frame["domestic_return"].dropna(), bins=bins, alpha=0.68, color="#5f8ffb", label="국내 유가")
        ax.hist(frame["wti_return"].dropna(), bins=bins, alpha=0.48, color="#f97316", label="WTI")
        ax.hist(frame["brent_return"].dropna(), bins=bins, alpha=0.48, color="#16a34a", label="Brent")
        ax.axvline(0, color="#777777", linewidth=1.2, linestyle="--")
        ax.set_title("일간 변동률 분포", fontsize=16, fontweight="bold")
        ax.set_xlabel("일간 변동률(%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("빈도", fontsize=12, fontweight="bold")
        ax.grid(axis="y", color="#e8e8e8", linewidth=1)
        ax.legend(frameon=True, edgecolor="#dddddd")

    def _draw_heatmap(self):
        ax = self.figure.add_subplot(111)
        labels = {
            "domestic_price": "국내 유가",
            "wti": "WTI",
            "brent": "Brent",
            "exchange": "원/달러",
            "risk_index": "고정 리스크",
            "news_risk_index": "뉴스 리스크",
            "volatility_7d": "7일 변동성",
        }
        columns = [column for column in labels if column in self.raw.columns]
        corr = self.raw[columns].corr()
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_title("주요 변수 상관관계 히트맵", fontsize=16, fontweight="bold")
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels([labels[column] for column in columns], rotation=35, ha="right")
        ax.set_yticklabels([labels[column] for column in columns])

        for row in range(len(columns)):
            for col in range(len(columns)):
                value = corr.iloc[row, col]
                color = "white" if abs(value) >= 0.55 else "#333333"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

        colorbar = self.figure.colorbar(image, ax=ax)
        colorbar.set_label("상관계수", fontsize=11, fontweight="bold")

    def run(self):
        self.root.mainloop()


def launch_analysis_gui(raw: pd.DataFrame):
    app = AnalysisGraphApp(raw)
    app.run()
