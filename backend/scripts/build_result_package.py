from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUT_DIR / "figures"
RESULT_DIR = Path.home() / "Desktop" / "result"
RESULT_CHARTS = RESULT_DIR / "charts"


def setup_font() -> str:
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", font_path))
                plt.rcParams["font.family"] = "AppleGothic"
                plt.rcParams["axes.unicode_minus"] = False
                return "KoreanFont"
            except Exception:
                continue
    return "Helvetica"


FONT_NAME = setup_font()


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    raw = pd.read_csv(OUT_DIR / "raw_oil_project.csv", index_col=0, parse_dates=True)
    forecast = pd.read_csv(OUT_DIR / "seven_day_forecast.csv", parse_dates=["date"])
    metrics = read_key_value_csv(OUT_DIR / "model_metrics.csv")
    meta = read_key_value_csv(OUT_DIR / "online_dataset_meta.csv")
    return raw, forecast, metrics, meta


def read_key_value_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if {"key", "value"}.issubset(df.columns):
        return dict(zip(df["key"], df["value"]))
    if len(df) == 1:
        return df.iloc[0].to_dict()
    return {}


def clear_result_dir() -> None:
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    RESULT_CHARTS.mkdir(parents=True, exist_ok=True)


def save_price_forecast(raw: pd.DataFrame, forecast: pd.DataFrame) -> Path:
    path = RESULT_CHARTS / "01_price_forecast.png"
    recent = raw.tail(90)
    today = raw.index[-1]
    today_price = float(raw["domestic_price"].iloc[-1])
    plt.figure(figsize=(12, 6.5), facecolor="white")
    plt.plot(recent.index, recent["domestic_price"], color="#4f83ff", linewidth=2.4, label="국내 유가")
    plt.plot(
        [today] + list(forecast["date"]),
        [today_price] + list(forecast["predicted_domestic_price"]),
        color="#f2a000",
        marker="o",
        linewidth=2.4,
        label="7일 예측",
    )
    plt.axvline(today, color="#999999", linestyle="--", linewidth=1.2)
    plt.scatter([today], [today_price], color="#dc2626", s=70, zorder=5, label="오늘")
    plt.title("최근 국내 유가와 7일 예측", fontsize=17, fontweight="bold")
    plt.ylabel("원/L")
    plt.grid(axis="y", alpha=0.28)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m.%d"))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_indicator_trends(raw: pd.DataFrame) -> Path:
    path = RESULT_CHARTS / "02_indicator_trends.png"
    recent = raw.tail(365)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, facecolor="white")
    axes[0].plot(recent.index, recent["wti"], label="WTI", color="#f97316", linewidth=1.8)
    axes[0].plot(recent.index, recent["brent"], label="Brent", color="#16a34a", linewidth=1.8)
    axes[0].set_title("국제 유가 WTI/Brent", fontweight="bold")
    axes[0].legend()
    axes[1].plot(recent.index, recent["exchange"], color="#2563eb", linewidth=1.8)
    axes[1].set_title("원/달러 환율", fontweight="bold")
    axes[2].plot(recent.index, recent["news_risk_index"], color="#7c3aed", linewidth=1.8, label="뉴스 리스크")
    axes[2].plot(recent.index, recent["risk_index"], color="#dc2626", linewidth=1.5, alpha=0.7, label="이벤트 리스크")
    axes[2].set_title("뉴스/지정학 리스크", fontweight="bold")
    axes[2].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_heatmap(raw: pd.DataFrame) -> Path:
    path = RESULT_CHARTS / "03_correlation_heatmap.png"
    labels = {
        "domestic_price": "국내 유가",
        "wti": "WTI",
        "brent": "Brent",
        "exchange": "환율",
        "risk_index": "이벤트",
        "news_risk_index": "뉴스",
        "volatility_7d": "변동성",
    }
    cols = [col for col in labels if col in raw.columns]
    corr = raw[cols].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor="white")
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([labels[col] for col in cols], rotation=35, ha="right")
    ax.set_yticklabels([labels[col] for col in cols])
    for row in range(len(cols)):
        for col in range(len(cols)):
            value = corr.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.55 else "#222")
    ax.set_title("주요 변수 상관관계 히트맵", fontsize=16, fontweight="bold")
    fig.colorbar(image, ax=ax, label="상관계수")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_return_distribution(raw: pd.DataFrame) -> Path:
    path = RESULT_CHARTS / "04_return_distribution.png"
    returns = pd.DataFrame(
        {
            "국내 유가": raw["domestic_price"].pct_change() * 100,
            "WTI": raw["wti"].pct_change() * 100,
            "Brent": raw["brent"].pct_change() * 100,
        }
    ).dropna()
    plt.figure(figsize=(11, 6.5), facecolor="white")
    plt.hist(returns["국내 유가"], bins=35, alpha=0.7, label="국내 유가", color="#4f83ff")
    plt.hist(returns["WTI"], bins=35, alpha=0.45, label="WTI", color="#f97316")
    plt.hist(returns["Brent"], bins=35, alpha=0.45, label="Brent", color="#16a34a")
    plt.axvline(0, color="#555555", linestyle="--", linewidth=1)
    plt.title("일간 변동률 분포", fontsize=16, fontweight="bold")
    plt.xlabel("일간 변동률(%)")
    plt.ylabel("빈도")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_news_forecast_impact(raw: pd.DataFrame, forecast: pd.DataFrame) -> Path:
    path = RESULT_CHARTS / "05_news_forecast_impact.png"
    today_price = float(raw["domestic_price"].iloc[-1])
    dates = pd.to_datetime(forecast["date"])
    predicted = forecast["predicted_domestic_price"].astype(float)
    adjustment = forecast.get("news_adjustment_pct", pd.Series([0] * len(forecast))).astype(float)
    baseline = predicted / (1 + adjustment.clip(lower=-0.2, upper=0.2))
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
    ax.plot([raw.index[-1]] + list(dates), [today_price] + list(predicted), marker="o", color="#f2a000", linewidth=2.4, label="뉴스 반영 예측")
    ax.plot([raw.index[-1]] + list(dates), [today_price] + list(baseline), marker="o", color="#94a3b8", linewidth=2.0, linestyle="--", label="뉴스 보정 전 추정")
    ax.set_title("뉴스 리스크 보정이 7일 예측에 미치는 영향", fontsize=16, fontweight="bold")
    ax.set_ylabel("원/L")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%d"))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def copy_existing_figures() -> None:
    existing_dir = RESULT_DIR / "all_project_figures"
    existing_dir.mkdir(exist_ok=True)
    for figure in sorted(FIGURES_DIR.glob("*.png")):
        shutil.copy2(figure, existing_dir / figure.name)


def build_analysis_json(raw: pd.DataFrame, forecast: pd.DataFrame, metrics: dict, meta: dict, chart_paths: list[Path]) -> Path:
    today_price = float(raw["domestic_price"].iloc[-1])
    final_forecast = float(forecast["predicted_domestic_price"].iloc[-1])
    returns = raw["domestic_price"].pct_change().dropna()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": "국내 유가 예측 및 뉴스 리스크 분석",
        "summary": {
            "rows": int(len(raw)),
            "latest_date": str(raw.index[-1].date()),
            "today_domestic_price": round(today_price, 2),
            "seven_day_forecast": round(final_forecast, 2),
            "forecast_change": round(final_forecast - today_price, 2),
            "forecast_change_pct": round((final_forecast / today_price - 1) * 100, 3),
            "wti_latest": round(float(raw["wti"].iloc[-1]), 2),
            "brent_latest": round(float(raw["brent"].iloc[-1]), 2),
            "exchange_latest": round(float(raw["exchange"].iloc[-1]), 2),
            "domestic_return_std_pct": round(float(returns.std() * 100), 4),
        },
        "model_metrics": metrics,
        "data_sources": meta,
        "charts": [{"name": path.name, "path": f"charts/{path.name}"} for path in chart_paths],
        "interpretation": [
            "국내 유가는 국제 유가와 환율, 뉴스 리스크를 함께 반영하여 예측했다.",
            "WTI/Brent는 국내 유가의 원가 방향성을 설명하는 보조 지표로 사용했다.",
            "뉴스 리스크는 이란/미국/호르무즈/원유 관련 속보를 기반으로 예측값에 보정 항목으로 반영했다.",
            "상관관계와 변동률 분포를 통해 단순 시계열 그래프 외의 분석 근거를 추가했다.",
        ],
    }
    path = RESULT_DIR / f"oil_price_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_snapshot_json(raw: pd.DataFrame, forecast: pd.DataFrame) -> Path:
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_record": {
            "date": str(raw.index[-1].date()),
            "domestic_price": round(float(raw["domestic_price"].iloc[-1]), 2),
            "wti": round(float(raw["wti"].iloc[-1]), 2),
            "brent": round(float(raw["brent"].iloc[-1]), 2),
            "exchange": round(float(raw["exchange"].iloc[-1]), 2),
            "risk_index": round(float(raw["risk_index"].iloc[-1]), 4),
            "news_risk_index": round(float(raw["news_risk_index"].iloc[-1]), 4),
        },
        "forecast": [
            {
                "date": str(pd.Timestamp(row["date"]).date()),
                "predicted_domestic_price": round(float(row["predicted_domestic_price"]), 2),
                "news_adjustment_pct": round(float(row.get("news_adjustment_pct", 0)), 4),
                "news_article_count": int(row.get("news_article_count", 0)),
            }
            for row in forecast.to_dict(orient="records")
        ],
    }
    path = RESULT_DIR / "oil_price_forecast_snapshot.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def make_table(data: list[list[str]]) -> Table:
    table = Table(data, colWidths=[52 * mm, 108 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def build_pdf(raw: pd.DataFrame, forecast: pd.DataFrame, metrics: dict, meta: dict, chart_paths: list[Path]) -> Path:
    path = RESULT_DIR / "국내 유가 예측 모델 분석 보고서.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="KTitle", parent=styles["Title"], fontName=FONT_NAME, fontSize=22, leading=30, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="KHeading", parent=styles["Heading2"], fontName=FONT_NAME, fontSize=15, leading=20, spaceBefore=10, spaceAfter=8))
    styles.add(ParagraphStyle(name="KBody", parent=styles["BodyText"], fontName=FONT_NAME, fontSize=10.5, leading=16))

    today_price = float(raw["domestic_price"].iloc[-1])
    final_forecast = float(forecast["predicted_domestic_price"].iloc[-1])
    story = [
        Paragraph("국내 유가 예측 모델 분석 보고서", styles["KTitle"]),
        Spacer(1, 8),
        Paragraph(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["KBody"]),
        Spacer(1, 12),
        Paragraph("1. 분석 개요", styles["KHeading"]),
        Paragraph(
            "본 보고서는 국내 유가, 국제 유가(WTI/Brent), 원/달러 환율, 뉴스/지정학 리스크 데이터를 이용하여 "
            "국내 유가 흐름과 향후 7일 예측 결과를 정리한 산출물이다.",
            styles["KBody"],
        ),
        Spacer(1, 10),
        make_table(
            [
                ["항목", "값"],
                ["분석 데이터 행 수", f"{len(raw):,}건"],
                ["최신 기준일", str(raw.index[-1].date())],
                ["오늘 국내 유가", f"{today_price:,.2f} 원/L"],
                ["7일 뒤 예측 유가", f"{final_forecast:,.2f} 원/L"],
                ["예상 변화", f"{final_forecast - today_price:,.2f} 원/L"],
                ["WTI / Brent", f"{raw['wti'].iloc[-1]:.2f} / {raw['brent'].iloc[-1]:.2f} 달러/배럴"],
                ["원/달러 환율", f"{raw['exchange'].iloc[-1]:,.2f} 원"],
            ]
        ),
        Spacer(1, 12),
        Paragraph("2. 데이터 출처 및 수집 방식", styles["KHeading"]),
        Paragraph(
            f"국내 유가는 {meta.get('domestic_source_name', 'OPINET')} 기반 데이터를 사용했고, "
            f"국제 유가는 {meta.get('international_oil_source', 'WTI/Brent 데이터')}를 사용했다. "
            f"환율 데이터는 {meta.get('exchange_source', 'FRED DEXKOUS')}를 활용했다.",
            styles["KBody"],
        ),
        Spacer(1, 12),
        Paragraph("3. 모델 및 예측 결과", styles["KHeading"]),
        Paragraph(
            "LSTM 기반 시계열 모델을 사용하여 최근 시점의 입력 특성으로 향후 7일 국내 유가를 예측했다. "
            "뉴스 리스크가 감지되는 경우 예측값에 단계적으로 보정률을 반영했다.",
            styles["KBody"],
        ),
        Spacer(1, 8),
        make_table([["지표", "값"]] + [[str(k), str(v)] for k, v in metrics.items()]),
        PageBreak(),
        Paragraph("4. 시각화 분석", styles["KHeading"]),
    ]

    for chart in chart_paths:
        story.append(Paragraph(chart.stem.replace("_", " "), styles["KHeading"]))
        story.append(Image(str(chart), width=165 * mm, height=92 * mm))
        story.append(Spacer(1, 10))

    story.extend(
        [
            PageBreak(),
            Paragraph("5. 결론", styles["KHeading"]),
            Paragraph(
                "국내 유가는 국제 유가와 환율의 영향을 받되, 뉴스 리스크와 지정학적 이벤트에 따라 단기 변동성이 확대될 수 있다. "
                "따라서 단순 가격 추세뿐 아니라 WTI/Brent, 환율, 뉴스 리스크, 변동성 지표를 함께 보는 방식이 예측 설명력을 높인다.",
                styles["KBody"],
            ),
        ]
    )
    doc.build(story)
    return path


def build_hwpx_placeholder(pdf_path: Path) -> Path:
    path = RESULT_DIR / "국내 유가 예측 모델 분석 보고서.hwpx"
    mimetype = "application/hwp+zip"
    content = """
    <document>
      <title>국내 유가 예측 모델 분석 보고서</title>
      <note>정식 PDF 보고서는 같은 폴더의 국내 유가 예측 모델 분석 보고서.pdf 파일을 확인하세요.</note>
      <source>backend 분석 결과 기반 자동 생성</source>
    </document>
    """.strip()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", mimetype)
        archive.writestr("Contents/section0.xml", content)
        archive.writestr("META-INF/container.xml", f"<container><rootfile>{pdf_path.name}</rootfile></container>")
    return path


def main() -> None:
    clear_result_dir()
    raw, forecast, metrics, meta = read_data()
    chart_paths = [
        save_price_forecast(raw, forecast),
        save_indicator_trends(raw),
        save_heatmap(raw),
        save_return_distribution(raw),
        save_news_forecast_impact(raw, forecast),
    ]
    copy_existing_figures()
    analysis_json = build_analysis_json(raw, forecast, metrics, meta, chart_paths)
    snapshot_json = build_snapshot_json(raw, forecast)
    pdf_path = build_pdf(raw, forecast, metrics, meta, chart_paths)
    hwpx_path = build_hwpx_placeholder(pdf_path)
    print(f"created: {RESULT_DIR}")
    print(f"- {analysis_json.name}")
    print(f"- {snapshot_json.name}")
    print(f"- {pdf_path.name}")
    print(f"- {hwpx_path.name}")


if __name__ == "__main__":
    main()
