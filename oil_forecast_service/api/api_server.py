from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
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
    description="휘발유·경유·LPG, WTI/Brent, 환율, 뉴스 리스크 기반 유가 분석/예측 서버",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.mount("/figures", StaticFiles(directory=str(PATHS.figures)), name="figures")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
_AUTO_REFRESH_DONE_FOR: str | None = None
_AUTO_REFRESH_TASK: asyncio.Task | None = None
_LAST_AUTO_INDICATOR_REFRESH_AT: pd.Timestamp | None = None
REFRESH_COOLDOWN_SECONDS = 10 * 60
AUTO_INDICATOR_REFRESH_INTERVAL_MIN = 60
AUTO_LOOP_SLEEP = 60
FORECAST_LOCK_HOUR = 23
_REFRESH_LOCK = threading.Lock()


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _forecast_slot(now: pd.Timestamp | None = None) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    now = now or pd.Timestamp.now()
    slot_time = now.normalize() + pd.Timedelta(hours=FORECAST_LOCK_HOUR)
    if now < slot_time:
        slot_time -= pd.Timedelta(days=1)
    target_date = slot_time.normalize() + pd.Timedelta(days=1)
    return target_date.date().isoformat(), slot_time, target_date


def _is_forecast_generation_window(now: pd.Timestamp | None = None) -> bool:
    now = now or pd.Timestamp.now()
    return int(now.hour) == FORECAST_LOCK_HOUR


def _refresh_state_values() -> dict[str, str]:
    state = _read_csv(PATHS.refresh_state)
    if state.empty:
        return {}
    return {key: str(value) for key, value in state.iloc[-1].to_dict().items()}


def _forecast_locked_for_current_slot() -> tuple[bool, str | None]:
    target_key, slot_time, target_date = _forecast_slot()
    state = _refresh_state_values()
    if state.get("forecast_target_date") != target_key:
        return False, None

    forecast = _read_csv(PATHS.forecast, parse_dates=["date", "prediction_time"])
    if forecast.empty or "prediction_time" not in forecast.columns:
        return False, None

    first = forecast.iloc[0]
    first_date = pd.Timestamp(first["date"]).normalize()
    first_prediction_time = pd.Timestamp(first["prediction_time"])
    if first_date != target_date.normalize() or first_prediction_time != slot_time:
        return False, None

    return True, slot_time.strftime("%Y-%m-%d %H:%M")


def _forecast_current_for_slot(now: pd.Timestamp | None = None) -> bool:
    _target_key, slot_time, target_date = _forecast_slot(now)
    forecast = _read_csv(PATHS.forecast, parse_dates=["date", "prediction_time"])
    if forecast.empty or "prediction_time" not in forecast.columns:
        return False

    first = forecast.iloc[0]
    first_date = pd.Timestamp(first["date"]).normalize()
    first_prediction_time = pd.Timestamp(first["prediction_time"])
    return first_date == target_date.normalize() and first_prediction_time == slot_time


def _write_refresh_state(extra: dict[str, str] | None = None) -> None:
    previous = _refresh_state_values()
    row = {
        "refreshed_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_date": pd.Timestamp.today().date().isoformat(),
        "forecast_slot_time": previous.get("forecast_slot_time", ""),
        "forecast_target_date": previous.get("forecast_target_date", ""),
    }
    if extra:
        row.update(extra)
    pd.DataFrame([row]).to_csv(PATHS.refresh_state, index=False)


def _run_indicator_refresh() -> None:
    collect_and_preprocess()
    _write_refresh_state()


def _run_full_refresh() -> None:
    target_key, slot_time, _target_date = _forecast_slot()
    df = collect_and_preprocess()
    run_eda(df)
    forecast_next_7_days(device="gpu", show_gui=False, prediction_anchor=slot_time)
    _write_refresh_state(
        {
            "forecast_slot_time": slot_time.strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_target_date": target_key,
        }
    )


def _indicator_refresh_due(now: pd.Timestamp | None = None) -> bool:
    now = now or pd.Timestamp.now()
    if _LAST_AUTO_INDICATOR_REFRESH_AT is None:
        age = _refresh_age_seconds()
        return age is None or age >= AUTO_INDICATOR_REFRESH_INTERVAL_MIN * 60
    elapsed_minutes = (now - _LAST_AUTO_INDICATOR_REFRESH_AT).total_seconds() / 60
    return elapsed_minutes >= AUTO_INDICATOR_REFRESH_INTERVAL_MIN


def _refresh_decision(now: pd.Timestamp | None = None) -> str:
    now = now or pd.Timestamp.now()
    locked, _slot_label = _forecast_locked_for_current_slot()
    if not locked and not _forecast_current_for_slot(now):
        return "full"
    if _is_forecast_generation_window(now) and not locked:
        return "full"
    if _indicator_refresh_due(now):
        return "indicators"
    return "skip"


def _run_auto_refresh_once() -> str:
    global _AUTO_REFRESH_DONE_FOR, _LAST_AUTO_INDICATOR_REFRESH_AT
    mode = _refresh_decision()
    if mode == "skip":
        return "skip"
    if not _REFRESH_LOCK.acquire(blocking=False):
        return "busy"
    try:
        if mode == "full":
            _run_full_refresh()
            _AUTO_REFRESH_DONE_FOR = pd.Timestamp.today().date().isoformat()
            _LAST_AUTO_INDICATOR_REFRESH_AT = pd.Timestamp.now()
            return "full"
        _run_indicator_refresh()
        _LAST_AUTO_INDICATOR_REFRESH_AT = pd.Timestamp.now()
        return "indicators"
    finally:
        _REFRESH_LOCK.release()


async def _auto_refresh_loop() -> None:
    await asyncio.sleep(5)
    while True:
        try:
            result = await asyncio.to_thread(_run_auto_refresh_once)
            if result != "skip":
                print(f"[auto-refresh] {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} {result}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[auto-refresh] failed: {exc}")
        await asyncio.sleep(AUTO_LOOP_SLEEP)


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
    return first_forecast_date >= today


def _ensure_current_outputs() -> None:
    global _AUTO_REFRESH_DONE_FOR
    today_key = pd.Timestamp.today().date().isoformat()
    locked, _slot_label = _forecast_locked_for_current_slot()
    if locked or _AUTO_REFRESH_DONE_FOR == today_key or (_outputs_are_current() and _forecast_current_for_slot()):
        return
    if not _REFRESH_LOCK.acquire(blocking=False):
        return
    try:
        if not _forecast_current_for_slot():
            _run_full_refresh()
            _AUTO_REFRESH_DONE_FOR = today_key
            return
        if not _is_forecast_generation_window():
            _run_indicator_refresh()
            _AUTO_REFRESH_DONE_FOR = today_key
            return
        _run_full_refresh()
        _AUTO_REFRESH_DONE_FOR = today_key
    finally:
        _REFRESH_LOCK.release()


def _figure_files() -> list[dict[str, str]]:
    hidden = {"validation_plus_today_forecast.png", "lag_correlation_summary.png"}
    titles = {
        "june_actual_lstm_forecast_compare.png": "6월 유종별 실제 유가 vs LSTM 예측 유가",
        "oil_price_dashboard.png": "유종별 유가 현황 및 7일 예측",
        "today_based_forecast.png": "오늘 기준 유종별 7일 예측 유가 그래프",
        "test_prediction_compare.png": "테스트 구간 유종별 날짜별 다음날 예측 비교",
        "oil_price_trend_1w.png": "유종별 유가추이 1주",
        "oil_price_trend_1m.png": "유종별 유가추이 1개월",
        "oil_price_trend_1y.png": "유종별 유가추이 1년",
        "oil_price_trend_3y.png": "유종별 유가추이 3년",
        "eda_overview.png": "EDA 전체 분석",
        "event_window_changes.png": "사건 전후 유종별 유가 변동",
        "scatter_wti_domestic.png": "WTI-유종별 국내 유가 산점도",
        "histogram_daily_returns.png": "유종별/국제유가 일간 변동률 히스토그램",
        "boxplot_price_distribution.png": "유종별/시장 가격 분포 박스플롯",
        "bar_recent_changes.png": "최근 유종별 일간 변화 막대그래프",
        "kde_return_distribution.png": "유종별/국제유가 일간 변동률 밀도 그래프",
        "violin_market_distribution.png": "유종별/시장 지표 바이올린 플롯",
        "correlation_heatmap.png": "상관관계 히트맵",
    }
    order = {
        "validation_plus_today_forecast.png": 5,
        "june_actual_lstm_forecast_compare.png": 6,
        "test_prediction_compare.png": 8,
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
        if path.name in hidden:
            continue
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


def _latest_snapshot() -> dict[str, Any]:
    _ensure_current_outputs()
    raw = _read_csv(PATHS.raw, index_col=0, parse_dates=True)
    forecast = _read_csv(PATHS.forecast, parse_dates=["date"])
    meta = _read_csv(PATHS.online_meta)
    audit = _read_csv(PATHS.source_audit)
    news = load_news_adjustment()
    meta_values = dict(zip(meta.get("key", []), meta.get("value", []))) if not meta.empty else {}

    latest: dict[str, Any] = {}
    if not raw.empty:
        row = raw.iloc[-1]
        updated_at = meta_values.get("downloaded_at") or str(pd.Timestamp(raw.index[-1]).date())
        gasoline_price = float(row.get("gasoline_price", row.get("domestic_price", 0)))
        latest = {
            "date": str(raw.index[-1].date()),
            "updated_at": updated_at,
            "domestic_price": round(gasoline_price, 2),
            "gasoline_price": round(gasoline_price, 2),
            "diesel_price": round(float(row.get("diesel_price", gasoline_price * 0.92)), 2),
            "lpg_price": round(float(row.get("lpg_price", gasoline_price * 0.54)), 2),
            "wti": round(float(row.get("wti", 0)), 2),
            "brent": round(float(row.get("brent", 0)), 2),
            "exchange": round(float(row.get("exchange", 0)), 2),
            "risk_index": round(float(row.get("risk_index", 0)), 4),
            "news_risk_index": round(float(row.get("news_risk_index", 0)), 4),
        }

    forecast_rows = []
    if not forecast.empty:
        for item in forecast.to_dict(orient="records"):
            predicted = round(float(item.get("predicted_gasoline_price", item["predicted_domestic_price"])), 2)
            predicted_diesel = round(float(item.get("predicted_diesel_price", predicted * 0.92)), 2)
            predicted_lpg = round(float(item.get("predicted_lpg_price", predicted * 0.54)), 2)
            normalized_item = {
                "predicted_domestic_price": predicted,
                "predicted_gasoline_price": predicted,
                "predicted_diesel_price": predicted_diesel,
                "predicted_lpg_price": predicted_lpg,
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
                    "prediction_time": (
                        pd.Timestamp(item["prediction_time"]).strftime("%Y-%m-%d %H:%M")
                        if "prediction_time" in item and pd.notna(item["prediction_time"])
                        else (pd.Timestamp(item["date"]) - pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
                    ),
                    "predicted_domestic_price": predicted,
                    "predicted_gasoline_price": normalized_item["predicted_gasoline_price"],
                    "predicted_diesel_price": normalized_item["predicted_diesel_price"],
                    "predicted_lpg_price": normalized_item["predicted_lpg_price"],
                    "raw_lstm_price": normalized_item["raw_lstm_price"],
                    "baseline_price": normalized_item["baseline_price"],
                    "daily_change_cap_won": normalized_item["daily_change_cap_won"],
                    "lstm_blend_weight": normalized_item["lstm_blend_weight"],
                    "news_risk_score": round(float(item.get("news_risk_score", 0)), 4),
                    "news_adjustment_pct": normalized_item["news_adjustment_pct"],
                    "news_article_count": normalized_item["news_article_count"],
                }
            )

    return {
        "latest": latest,
        "forecast": forecast_rows,
        "news": news,
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

    today_price = latest["gasoline_price"]
    last_forecast = forecast[-1]["predicted_gasoline_price"] if forecast else today_price
    diff = last_forecast - today_price
    direction = "상승" if diff > 0 else "하락" if diff < 0 else "보합"
    question_lower = question.lower()

    if any(keyword in question_lower for keyword in ["news", "뉴스", "전쟁", "이란", "미국", "리스크", "호르무즈"]):
        answer = (
            f"뉴스 리스크 점수는 {news['news_risk_score']:.3f}이고, "
            f"예측 보정률은 {news['forecast_adjustment_pct'] * 100:.2f}%입니다. "
            f"현재 데이터 기준으로 7일 뒤 휘발유는 {last_forecast:,.1f}원/L 수준으로 예측되어 "
            f"오늘 휘발유 {today_price:,.1f}원/L 대비 {abs(diff):,.1f}원/L {direction} 압력이 있습니다. "
            "경유와 LPG 예측도 함께 산출됩니다."
        )
    elif any(keyword in question_lower for keyword in ["wti", "brent", "국제", "원유"]):
        answer = (
            f"최근 국제 유가는 WTI {latest['wti']:.2f}달러/배럴, Brent {latest['brent']:.2f}달러/배럴입니다. "
            "휘발유·경유·LPG는 국제 유가와 환율 영향을 며칠 시차를 두고 반영할 수 있으므로, "
            f"현재 7일 예측은 {direction} 방향으로 보고 있습니다."
        )
    elif any(keyword in question_lower for keyword in ["환율", "달러", "exchange"]):
        answer = (
            f"최근 원/달러 환율은 {latest['exchange']:,.2f}원입니다. "
            "환율 상승은 원유 수입 비용을 키워 휘발유·경유·LPG 상승 압력으로 작용할 수 있습니다."
        )
    else:
        answer = (
            f"오늘 휘발유는 {today_price:,.1f}원/L이고, "
            f"7일 뒤 휘발유 예측값은 {last_forecast:,.1f}원/L입니다. "
            f"경유·LPG도 함께 산출되며, 현재 모델은 휘발유 기준 약 {abs(diff):,.1f}원/L {direction} 흐름으로 해석합니다."
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


@app.on_event("startup")
async def startup_auto_refresh() -> None:
    global _AUTO_REFRESH_TASK
    if _AUTO_REFRESH_TASK is None or _AUTO_REFRESH_TASK.done():
        _AUTO_REFRESH_TASK = asyncio.create_task(_auto_refresh_loop())
        print("[auto-refresh] server-side auto refresh enabled")


@app.on_event("shutdown")
async def shutdown_auto_refresh() -> None:
    global _AUTO_REFRESH_TASK
    if _AUTO_REFRESH_TASK is None:
        return
    _AUTO_REFRESH_TASK.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _AUTO_REFRESH_TASK
    _AUTO_REFRESH_TASK = None


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
    if not _REFRESH_LOCK.acquire(blocking=False):
        return RefreshResponse(status="busy", message="자동 또는 수동 최신화가 이미 진행 중입니다. 잠시 뒤 다시 시도하세요.")
    try:
        now = pd.Timestamp.now()
        locked, slot_label = _forecast_locked_for_current_slot()
        if locked:
            _run_indicator_refresh()
            return RefreshResponse(
                status="locked",
                message=(
                    f"핵심 지표/유종별 평균가/뉴스만 최신화했습니다. "
                    f"{slot_label} 기준 예측은 이미 생성되어 고정되어 있습니다."
                ),
            )

        if not _forecast_current_for_slot(now):
            _run_full_refresh()
            _AUTO_REFRESH_DONE_FOR = pd.Timestamp.today().date().isoformat()
            _target_key, slot_time, _target_date = _forecast_slot(now)
            return RefreshResponse(
                status="backfilled",
                message=f"누락된 {slot_time.strftime('%Y-%m-%d %H:%M')} 예측 슬롯을 보강 생성했습니다.",
            )

        if not _is_forecast_generation_window(now):
            _run_indicator_refresh()
            _AUTO_REFRESH_DONE_FOR = pd.Timestamp.today().date().isoformat()
            next_slot = now.normalize() + pd.Timedelta(hours=FORECAST_LOCK_HOUR)
            if now >= next_slot:
                next_slot += pd.Timedelta(days=1)
            return RefreshResponse(
                status="indicators_only",
                message=(
                    "핵심 지표/유종별 평균가/뉴스만 최신화했습니다. "
                    f"예측 그래프는 기존 23:00 슬롯 결과를 유지합니다. 다음 예측 생성 가능 시각은 {next_slot.strftime('%Y-%m-%d %H:%M')}입니다."
                ),
            )
        _run_full_refresh()
        _AUTO_REFRESH_DONE_FOR = pd.Timestamp.today().date().isoformat()
        _target_key, slot_time, _target_date = _forecast_slot(now)
        return RefreshResponse(status="ok", message=f"{slot_time.strftime('%Y-%m-%d %H:%M')} 기준 핵심 지표 최신화와 7일 예측 생성을 완료했습니다.")
    finally:
        _REFRESH_LOCK.release()


@app.post("/agent", response_model=AgentResponse, tags=["agent"])
def agent(request: AgentRequest) -> AgentResponse:
    answer, facts = _build_agent_answer(request.question)
    return AgentResponse(question=request.question, answer=answer, facts=facts)
