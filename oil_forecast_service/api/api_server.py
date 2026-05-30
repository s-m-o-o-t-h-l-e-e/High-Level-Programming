from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

API_DIR = Path(__file__).resolve().parent
BACKEND_DIR = API_DIR.parent
WEB_DIR = BACKEND_DIR / "web"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from config import PATHS, ensure_dirs
from data_pipeline import collect_and_preprocess
from eda_analysis import run_eda
from forecasting import forecast_next_7_days, load_news_adjustment


class AgentRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    question: str
    answer: str
    facts: dict[str, Any]


class RefreshResponse(BaseModel):
    status: str
    message: str


ensure_dirs()
WEB_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Oil Price Forecast Analysis API",
    description="국내 유가, WTI/Brent, 환율, 뉴스 리스크 기반 유가 분석/예측 서버",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.mount("/figures", StaticFiles(directory=str(PATHS.figures)), name="figures")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
_AUTO_REFRESH_DONE_FOR: str | None = None
REFRESH_COOLDOWN_SECONDS = 10 * 60


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _run_full_refresh() -> None:
    df = collect_and_preprocess()
    run_eda(df)
    forecast_next_7_days(device="auto", show_gui=False)
    pd.DataFrame(
        [
            {
                "refreshed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "data_date": pd.Timestamp.today().date().isoformat(),
            }
        ]
    ).to_csv(PATHS.refresh_state, index=False)


def _refresh_age_seconds() -> float | None:
    if not PATHS.refresh_state.exists():
        return None
    state = _read_csv(PATHS.refresh_state)
    if state.empty or "refreshed_at" not in state.columns:
        return None
    refreshed_at = pd.to_datetime(state["refreshed_at"].iloc[-1], errors="coerce")
    if pd.isna(refreshed_at):
        return None
    if refreshed_at.tzinfo is not None:
        refreshed_at = refreshed_at.tz_convert(None)
    return max(0.0, (pd.Timestamp.now() - refreshed_at).total_seconds())


def _human_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}초"
    return f"{int(seconds // 60)}분 {int(seconds % 60)}초"


def _outputs_are_current() -> bool:
    today = pd.Timestamp.today().normalize()
    if not PATHS.raw.exists() or not PATHS.forecast.exists():
        return False

    raw = _read_csv(PATHS.raw, index_col=0, parse_dates=True)
    if raw.empty or raw.index.max().normalize() < today:
        return False

    forecast = _read_csv(PATHS.forecast, parse_dates=["date"])
    if forecast.empty:
        return False
    first_forecast_date = pd.Timestamp(forecast["date"].min()).normalize()
    return first_forecast_date > today


def _ensure_current_outputs() -> None:
    global _AUTO_REFRESH_DONE_FOR
    today_key = pd.Timestamp.today().date().isoformat()
    if _AUTO_REFRESH_DONE_FOR == today_key or _outputs_are_current():
        return
    _run_full_refresh()
    _AUTO_REFRESH_DONE_FOR = today_key


def _figure_files() -> list[dict[str, str]]:
    titles = {
        "oil_price_dashboard.png": "유가 현황 및 7일 예측",
        "today_based_forecast.png": "오늘 기준 7일 예측 유가 그래프",
        "oil_price_trend_1w.png": "유가추이 1주",
        "oil_price_trend_1m.png": "유가추이 1개월",
        "oil_price_trend_1y.png": "유가추이 1년",
        "oil_price_trend_3y.png": "유가추이 3년",
        "eda_overview.png": "EDA 전체 분석",
        "event_window_changes.png": "사건 전후 유가 변동",
        "scatter_wti_domestic.png": "WTI-국내 유가 산점도",
        "histogram_daily_returns.png": "일간 변동률 히스토그램",
        "boxplot_price_distribution.png": "가격 분포 박스플롯",
        "bar_recent_changes.png": "최근 일간 변화 막대그래프",
        "kde_return_distribution.png": "일간 변동률 밀도 그래프",
        "violin_market_distribution.png": "시장 지표 바이올린 플롯",
        "correlation_heatmap.png": "상관관계 히트맵",
    }
    order = {
        "oil_price_trend_1w.png": 10,
        "oil_price_trend_1m.png": 20,
        "oil_price_trend_1y.png": 30,
        "oil_price_trend_3y.png": 40,
        "today_based_forecast.png": 50,
        "oil_price_dashboard.png": 60,
        "histogram_daily_returns.png": 70,
        "bar_recent_changes.png": 80,
        "scatter_wti_domestic.png": 90,
        "boxplot_price_distribution.png": 100,
        "kde_return_distribution.png": 110,
        "violin_market_distribution.png": 120,
        "eda_overview.png": 130,
        "event_window_changes.png": 140,
        "correlation_heatmap.png": 150,
    }

    files = []
    figure_paths = sorted(PATHS.figures.glob("*.png"), key=lambda path: (order.get(path.name, 999), path.name))
    for path in figure_paths:
        version = int(path.stat().st_mtime)
        files.append(
            {
                "filename": path.name,
                "title": titles.get(path.name, path.stem.replace("_", " ")),
                "url": f"/figures/{path.name}?v={version}",
                "detail_url": f"/graphs/{path.name}",
            }
        )
    return files


def _forecast_reason(
    item: dict[str, Any],
    latest: dict[str, Any],
    today_price: float | None,
    previous_price: float | None,
    news_headlines: list[str] | None = None,
) -> str:
    predicted_price = float(item.get("predicted_domestic_price", 0))
    today_diff = predicted_price - today_price if today_price is not None else 0
    day_diff = predicted_price - previous_price if previous_price is not None else today_diff
    news_adjustment = float(item.get("news_adjustment_pct", 0))
    news_count = int(item.get("news_article_count", 0))
    wti = latest.get("wti")
    brent = latest.get("brent")
    exchange = latest.get("exchange")

    if abs(today_diff) < 0.05:
        movement = "오늘과 거의 비슷한 보합권으로 예상됩니다."
    elif today_diff > 0 and day_diff >= 0:
        movement = "오늘보다 높고 전일 예측보다도 올라 상승 흐름으로 예상됩니다."
    elif today_diff > 0 and day_diff < 0:
        movement = "오늘보다는 높지만 전일 예측보다 낮아 상승폭이 줄어드는 조정 구간으로 보입니다."
    elif today_diff < 0 and day_diff <= 0:
        movement = "오늘보다 낮고 전일 예측보다도 내려 하락 흐름으로 예상됩니다."
    else:
        movement = "오늘보다는 낮지만 전일 예측보다 올라 단기 반등 구간으로 보입니다."

    news_text = ""
    if news_count > 0 and abs(news_adjustment) > 0:
        news_direction = "상승 압력" if news_adjustment > 0 else "하락 압력"
        news_text = f"최근 뉴스 {news_count}건의 지정학적 리스크 신호는 단기 유가에 {news_direction}으로 반영됐습니다."
    elif news_count > 0:
        news_text = f"최근 뉴스 {news_count}건은 뚜렷한 방향성보다 관망 신호에 가깝게 반영됐습니다."

    headline_text = ""
    if news_headlines:
        headline_text = "주요 뉴스 근거는 " + ", ".join(f"'{headline}'" for headline in news_headlines[:2]) + "입니다."
    event_detail_text = _news_event_detail(news_headlines or [], news_adjustment)

    exchange_text = ""
    exchange_level = float(exchange) if exchange else None
    if exchange_level is not None:
        if exchange_level >= 1450:
            exchange_text = (
                f"원/달러 환율이 {exchange_level:,.2f}원으로 높은 구간이라 원유 수입 비용 부담이 커져 "
                "국내 유가 하방을 제한할 수 있습니다."
            )
        elif exchange_level <= 1300:
            exchange_text = (
                f"원/달러 환율이 {exchange_level:,.2f}원으로 비교적 낮아 수입 비용 부담이 완화되어 "
                "국내 유가 상승 압력을 줄일 수 있습니다."
            )
        else:
            exchange_text = (
                f"원/달러 환율 {exchange_level:,.2f}원은 수입 비용 변수로 반영됐지만, "
                "단기 방향성은 뉴스 리스크와 국제유가 흐름의 영향이 더 큽니다."
            )

    oil_text = ""
    if wti and brent:
        spread = float(brent) - float(wti)
        oil_text = f"WTI {float(wti):.2f}, Brent {float(brent):.2f}달러 수준의 국제유가 흐름도 함께 반영했습니다."
        if spread >= 4:
            oil_text += " Brent가 WTI보다 높아 글로벌 원유 수급 부담이 남아 있는 구간입니다."
        elif spread <= 1:
            oil_text += " 두 지표 간 격차가 작아 국제유가 방향성은 비교적 중립적으로 해석됩니다."

    balance_text = ""
    if news_adjustment < 0 and exchange_level is not None and exchange_level >= 1450:
        balance_text = (
            "따라서 뉴스 리스크는 가격을 낮추는 쪽으로 작용하지만, 높은 환율이 수입 비용을 밀어 올려 "
            "하락폭은 제한되는 구조입니다."
        )
    elif news_adjustment > 0 and exchange_level is not None and exchange_level >= 1450:
        balance_text = "뉴스 리스크와 높은 환율이 같은 방향으로 작용해 국내 유가 상승 가능성을 키우는 구조입니다."
    elif news_adjustment > 0 and exchange_level is not None and exchange_level <= 1300:
        balance_text = "뉴스 리스크는 상승 요인이지만 낮은 환율이 수입 비용 부담을 줄여 상승폭을 일부 완화합니다."
    elif news_adjustment < 0 and exchange_level is not None and exchange_level <= 1300:
        balance_text = "뉴스 리스크와 낮은 환율이 모두 부담 완화 쪽으로 작용해 하락 가능성이 더 크게 반영됐습니다."

    hypothesis_text = ""
    if news_adjustment < 0 and exchange_level is not None and exchange_level >= 1450:
        hypothesis_text = (
            "현재 가설은 '지정학적 뉴스가 완화 신호를 보이더라도 고환율 때문에 국내 휘발유 가격은 "
            "급락하지 않고 완만하게 조정된다'입니다."
        )
    elif news_adjustment > 0 and exchange_level is not None and exchange_level >= 1450:
        hypothesis_text = (
            "현재 가설은 '지정학적 긴장과 고환율이 동시에 작용하면 국제유가 상승분이 국내 가격에 "
            "더 빠르게 전가될 수 있다'입니다."
        )
    elif news_adjustment > 0:
        hypothesis_text = "현재 가설은 '뉴스 리스크가 단기 불확실성을 키워 국내 유가에 제한적인 상승 압력을 만든다'입니다."
    elif news_adjustment < 0:
        hypothesis_text = "현재 가설은 '뉴스 리스크 완화가 국제유가 부담을 낮추며 국내 유가도 완만히 내려갈 수 있다'입니다."
    elif exchange_level is not None and exchange_level >= 1450:
        hypothesis_text = "현재 가설은 '뉴스 방향성이 약해도 높은 환율이 수입 비용을 높여 국내 유가 하락을 제한한다'입니다."
    else:
        hypothesis_text = "현재 가설은 '뚜렷한 외부 충격이 없으면 국내 유가는 최근 평균 흐름을 따라 완만하게 움직인다'입니다."

    return " ".join(
        text
        for text in [movement, hypothesis_text, news_text, headline_text, event_detail_text, exchange_text, oil_text, balance_text]
        if text
    )


def _news_event_detail(headlines: list[str], adjustment_pct: float) -> str:
    if not headlines:
        return ""

    text = " ".join(headlines).lower()
    timing = _news_timing_text(text)
    escalation_terms = {
        "war": "전쟁 가능성",
        "attack": "공격",
        "attacked": "공격",
        "blockade": "봉쇄",
        "sanctions": "제재",
        "disruption": "수송 차질",
        "hormuz": "호르무즈 해협 리스크",
        "tanker": "유조선 운송 차질",
        "strait": "해협 통항 리스크",
    }
    deescalation_terms = {
        "ceasefire": "휴전",
        "deal": "합의",
        "talks": "협상",
        "negotiation": "협상",
        "negotiations": "협상",
        "approval": "최종 승인 대기",
        "lifted": "봉쇄 해제",
        "open": "해협 개방",
        "reopen": "재개방",
    }
    uncertainty_terms = {
        "rejects": "합의 부인",
        "final decision": "최종 결정 대기",
        "tentative": "잠정 합의",
        "claims": "주장 엇갈림",
    }

    escalation_hits = sorted({label for word, label in escalation_terms.items() if word in text})
    deescalation_hits = sorted({label for word, label in deescalation_terms.items() if word in text})
    uncertainty_hits = sorted({label for word, label in uncertainty_terms.items() if word in text})

    timing_prefix = f"{timing} " if timing else ""
    if escalation_hits and not deescalation_hits:
        return (
            f"{timing_prefix}뉴스 내용상 {', '.join(escalation_hits[:3])}이 확인되어 원유 수송 또는 공급 차질 가능성이 커진 것으로 해석했습니다. "
            "이 경우 국제유가 상승 압력이 국내 유가에 시차를 두고 반영될 수 있습니다."
        )
    if deescalation_hits and not escalation_hits:
        detail = (
            f"{timing_prefix}뉴스 내용상 {', '.join(deescalation_hits[:3])} 신호가 확인되어 전쟁/봉쇄 리스크가 완화되는 쪽으로 해석했습니다. "
            "이 경우 국제유가 하락 압력이 생기지만, 국내 가격은 환율 때문에 천천히 반영될 수 있습니다."
        )
        if uncertainty_hits:
            detail += f" 다만 {', '.join(uncertainty_hits[:2])} 표현도 있어 완전한 하락 전환보다는 제한적 조정으로 봤습니다."
        return detail
    if escalation_hits and deescalation_hits:
        return (
            f"{timing_prefix}뉴스 안에 {', '.join(escalation_hits[:2])}와 {', '.join(deescalation_hits[:2])}가 함께 나타나 방향성이 엇갈립니다. "
            "따라서 급등락보다는 환율과 국제유가 수준을 함께 고려한 제한적 변동으로 반영했습니다."
        )
    if uncertainty_hits:
        return f"{timing_prefix}뉴스 제목에 {', '.join(uncertainty_hits[:3])} 신호가 있어 협상 결과가 확정되기 전까지는 관망 요인으로 반영했습니다."
    if adjustment_pct > 0:
        return "뉴스 내용은 공급 차질 가능성을 키우는 방향으로 해석되어 단기 상승 요인으로 반영했습니다."
    if adjustment_pct < 0:
        return "뉴스 내용은 지정학적 긴장 완화 가능성을 보여 단기 하락 요인으로 반영했습니다."
    return "뉴스 내용에서 강한 상승/하락 사건 신호가 뚜렷하지 않아 중립 요인으로 반영했습니다."


def _news_timing_text(text: str) -> str:
    digit_match = re.search(r"(?:in|within)\s+(\d+)\s+days?", text)
    if digit_match:
        return f"기사에서 약 {digit_match.group(1)}일 뒤 사건 가능성이 언급되어"

    word_days = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
    }
    for word, day in word_days.items():
        if re.search(rf"(?:in|within)\s+{word}\s+days?", text):
            return f"기사에서 약 {day}일 뒤 사건 가능성이 언급되어"
    korean_match = re.search(r"(\d+)\s*일\s*(?:뒤|후)", text)
    if korean_match:
        return f"기사에서 약 {korean_match.group(1)}일 뒤 사건 가능성이 언급되어"
    if "next week" in text or "다음 주" in text:
        return "기사에서 다음 주 사건 가능성이 언급되어"
    return ""


def _news_headlines_for_reason(adjustment_pct: float, limit: int = 2) -> list[str]:
    articles = _read_csv(PATHS.news_articles)
    if articles.empty or "title" not in articles.columns:
        return []

    scored = articles.copy()
    if "net_score" not in scored.columns:
        scored["net_score"] = 0
    scored["net_score"] = pd.to_numeric(scored["net_score"], errors="coerce").fillna(0)
    scored["title"] = scored["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    scored = scored[scored["title"].ne("")]
    if scored.empty:
        return []

    if adjustment_pct < 0:
        selected = scored.sort_values(["net_score", "title"], ascending=[True, True]).head(limit)
    elif adjustment_pct > 0:
        selected = scored.sort_values(["net_score", "title"], ascending=[False, True]).head(limit)
    else:
        selected = scored.reindex(scored["net_score"].abs().sort_values(ascending=False).index).head(limit)

    cleaned = []
    for title in selected["title"].tolist():
        title = title.replace(" - Google News", "").strip()
        if len(title) > 95:
            title = title[:92].rstrip() + "..."
        cleaned.append(title)
    return cleaned


def _latest_snapshot() -> dict[str, Any]:
    _ensure_current_outputs()
    raw = _read_csv(PATHS.raw, index_col=0, parse_dates=True)
    forecast = _read_csv(PATHS.forecast, parse_dates=["date"])
    meta = _read_csv(PATHS.online_meta)
    audit = _read_csv(PATHS.source_audit)
    news = load_news_adjustment()
    news_headlines = _news_headlines_for_reason(float(news.get("forecast_adjustment_pct", 0.0)))
    meta_values = dict(zip(meta.get("key", []), meta.get("value", []))) if not meta.empty else {}

    latest: dict[str, Any] = {}
    if not raw.empty:
        row = raw.iloc[-1]
        updated_at = meta_values.get("downloaded_at") or str(pd.Timestamp(raw.index[-1]).date())
        latest = {
            "date": str(raw.index[-1].date()),
            "updated_at": updated_at,
            "domestic_price": round(float(row.get("domestic_price", 0)), 2),
            "wti": round(float(row.get("wti", 0)), 2),
            "brent": round(float(row.get("brent", 0)), 2),
            "exchange": round(float(row.get("exchange", 0)), 2),
            "risk_index": round(float(row.get("risk_index", 0)), 4),
            "news_risk_index": round(float(row.get("news_risk_index", 0)), 4),
        }

    forecast_rows = []
    if not forecast.empty:
        previous_price = latest.get("domestic_price") if latest else None
        for item in forecast.to_dict(orient="records"):
            predicted = round(float(item["predicted_domestic_price"]), 2)
            today_price = latest.get("domestic_price") if latest else None
            normalized_item = {
                "predicted_domestic_price": predicted,
                "news_adjustment_pct": round(float(item.get("news_adjustment_pct", 0)), 4),
                "news_article_count": int(item.get("news_article_count", 0)),
                "raw_lstm_price": round(float(item["raw_lstm_price"]), 2) if "raw_lstm_price" in item else None,
                "baseline_price": round(float(item["baseline_price"]), 2) if "baseline_price" in item else None,
                "daily_change_cap_won": round(float(item["daily_change_cap_won"]), 2) if "daily_change_cap_won" in item else None,
                "lstm_blend_weight": round(float(item["lstm_blend_weight"]), 4) if "lstm_blend_weight" in item else None,
            }
            forecast_rows.append(
                {
                    "date": str(pd.Timestamp(item["date"]).date()),
                    "predicted_domestic_price": predicted,
                    "raw_lstm_price": normalized_item["raw_lstm_price"],
                    "baseline_price": normalized_item["baseline_price"],
                    "daily_change_cap_won": normalized_item["daily_change_cap_won"],
                    "lstm_blend_weight": normalized_item["lstm_blend_weight"],
                    "news_risk_score": round(float(item.get("news_risk_score", 0)), 4),
                    "news_adjustment_pct": normalized_item["news_adjustment_pct"],
                    "news_article_count": normalized_item["news_article_count"],
                    "reason": _forecast_reason(normalized_item, latest, today_price, previous_price, news_headlines),
                }
            )
            previous_price = predicted

    return {
        "latest": latest,
        "forecast": forecast_rows,
        "news": news,
        "news_headlines": news_headlines,
        "meta": meta_values,
        "sources": dict(zip(audit.get("key", []), audit.get("value", []))) if not audit.empty else {},
    }


def _build_agent_answer(question: str) -> tuple[str, dict[str, Any]]:
    snapshot = _latest_snapshot()
    latest = snapshot["latest"]
    forecast = snapshot["forecast"]
    news = snapshot["news"]

    if not latest:
        return "아직 분석 데이터가 없습니다. 먼저 /refresh 또는 forecast 모드를 실행해서 데이터를 생성해야 합니다.", snapshot

    today_price = latest["domestic_price"]
    last_forecast = forecast[-1]["predicted_domestic_price"] if forecast else today_price
    diff = last_forecast - today_price
    direction = "상승" if diff > 0 else "하락" if diff < 0 else "보합"
    question_lower = question.lower()

    if any(keyword in question_lower for keyword in ["news", "뉴스", "전쟁", "이란", "미국", "리스크", "호르무즈"]):
        answer = (
            f"뉴스 리스크 점수는 {news['news_risk_score']:.3f}이고, "
            f"예측 보정률은 {news['forecast_adjustment_pct'] * 100:.2f}%입니다. "
            f"현재 데이터 기준으로 7일 뒤 국내 유가는 {last_forecast:,.1f}원/L 수준으로 예측되어 "
            f"오늘 {today_price:,.1f}원/L 대비 {abs(diff):,.1f}원/L {direction} 압력이 있습니다."
        )
    elif any(keyword in question_lower for keyword in ["wti", "brent", "국제", "원유"]):
        answer = (
            f"최근 국제 유가는 WTI {latest['wti']:.2f}달러/배럴, Brent {latest['brent']:.2f}달러/배럴입니다. "
            "국내 유가는 국제 유가와 환율 영향을 며칠 시차를 두고 반영할 수 있으므로, "
            f"현재 7일 예측은 {direction} 방향으로 보고 있습니다."
        )
    elif any(keyword in question_lower for keyword in ["환율", "달러", "exchange"]):
        answer = (
            f"최근 원/달러 환율은 {latest['exchange']:,.2f}원입니다. "
            "환율 상승은 원유 수입 비용을 키워 국내 유가 상승 압력으로 작용할 수 있습니다."
        )
    else:
        answer = (
            f"오늘 국내 유가는 {today_price:,.1f}원/L이고, "
            f"7일 뒤 예측값은 {last_forecast:,.1f}원/L입니다. "
            f"현재 모델은 약 {abs(diff):,.1f}원/L {direction} 흐름으로 해석합니다."
        )

    return answer, snapshot


def _web_file(filename: str) -> FileResponse:
    return FileResponse(
        WEB_DIR / filename,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


def _render_web_template(template_name: str, **values: str) -> str:
    text = (WEB_DIR / template_name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


@app.get("/", response_class=HTMLResponse, tags=["homepage"])
def homepage() -> FileResponse:
    return _web_file("dashboard.html")


@app.get("/home", response_class=HTMLResponse, include_in_schema=False)
def home_alias() -> FileResponse:
    return _web_file("dashboard.html")


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def browser_icon_probe() -> Response:
    return Response(status_code=204)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/summary", tags=["analysis"])
def summary() -> dict[str, Any]:
    return _latest_snapshot()


@app.get("/forecast", tags=["analysis"])
def forecast() -> list[dict[str, Any]]:
    return _latest_snapshot()["forecast"]


@app.get("/graphs/list", tags=["graphs"])
def graph_list() -> list[dict[str, str]]:
    return _figure_files()


@app.get("/graphs", response_class=HTMLResponse, tags=["graphs"])
def graph_gallery() -> FileResponse:
    return _web_file("graph_gallery.html")


@app.get("/graphs/{filename}", response_class=HTMLResponse, tags=["graphs"])
def graph_detail(filename: str) -> HTMLResponse:
    safe_name = Path(filename).name
    figures = {figure["filename"]: figure for figure in _figure_files()}
    figure = figures.get(safe_name)
    if figure is None:
        return HTMLResponse("<h1>그래프를 찾을 수 없습니다</h1><p><a href='/graphs'>전체 그래프로 돌아가기</a></p>", status_code=404)
    return HTMLResponse(
        _render_web_template(
            "graph_detail.html",
            title=figure["title"],
            url=figure["url"],
            filename=figure["filename"],
        )
    )


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def simple_docs() -> FileResponse:
    return _web_file("api_docs.html")


@app.post("/refresh", response_model=RefreshResponse, tags=["analysis"])
def refresh_data(force: bool = False) -> RefreshResponse:
    global _AUTO_REFRESH_DONE_FOR
    age = _refresh_age_seconds()
    if not force and age is not None and age < REFRESH_COOLDOWN_SECONDS and _outputs_are_current():
        return RefreshResponse(
            status="cached",
            message=(
                f"{_human_age(age)} 전에 이미 최신 데이터로 갱신했습니다. "
                "예측값이 매번 흔들리지 않도록 기존 결과를 유지합니다."
            ),
        )
    _run_full_refresh()
    _AUTO_REFRESH_DONE_FOR = pd.Timestamp.today().date().isoformat()
    return RefreshResponse(status="ok", message="최신 데이터 수집, EDA, 7일 예측을 완료했습니다.")


@app.post("/agent", response_model=AgentResponse, tags=["agent"])
def agent(request: AgentRequest) -> AgentResponse:
    answer, facts = _build_agent_answer(request.question)
    return AgentResponse(question=request.question, answer=answer, facts=facts)
