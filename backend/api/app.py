from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

API_DIR = Path(__file__).resolve().parent
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

app = FastAPI(
    title="Oil Price Forecast Analysis API",
    description="국내 유가, WTI/Brent, 환율, 뉴스 리스크 기반 유가 분석/예측 서버",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.mount("/figures", StaticFiles(directory=str(PATHS.figures)), name="figures")
_AUTO_REFRESH_DONE_FOR: str | None = None


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _run_full_refresh() -> None:
    df = collect_and_preprocess()
    run_eda(df)
    forecast_next_7_days(device="auto", show_gui=False)


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
    if first_forecast_date <= today:
        return False

    return True


def _ensure_current_outputs() -> None:
    global _AUTO_REFRESH_DONE_FOR
    today_key = pd.Timestamp.today().date().isoformat()
    if _AUTO_REFRESH_DONE_FOR == today_key or _outputs_are_current():
        return
    _run_full_refresh()
    _AUTO_REFRESH_DONE_FOR = today_key


def _figure_files() -> list[dict[str, str]]:
    files = []
    titles = {
        "oil_price_dashboard.png": "유가 현황 및 7일 예측",
        "seven_day_forecast.png": "향후 7일 유가 예측",
        "oil_price_trend_1w.png": "유가추이 1주",
        "oil_price_trend_1m.png": "유가추이 1개월",
        "oil_price_trend_1y.png": "유가추이 1년",
        "oil_price_trend_3y.png": "유가추이 3년",
        "eda_overview.png": "EDA 전체 분석",
        "event_window_changes.png": "사건 전후 유가 변동",
        "scatter_wti_domestic.png": "WTI-국내 유가 산점도",
        "histogram_daily_returns.png": "일간 변동률 히스토그램",
        "correlation_heatmap.png": "상관관계 히트맵",
    }
    order = {
        "oil_price_trend_1w.png": 10,
        "oil_price_trend_1m.png": 20,
        "oil_price_trend_1y.png": 30,
        "oil_price_trend_3y.png": 40,
        "seven_day_forecast.png": 50,
        "oil_price_dashboard.png": 60,
        "histogram_daily_returns.png": 70,
        "scatter_wti_domestic.png": 80,
        "eda_overview.png": 90,
        "event_window_changes.png": 100,
        "correlation_heatmap.png": 110,
    }
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


def _graph_gallery_html() -> str:
    figures = _figure_files()
    cards = "\n".join(
        f"""
        <article class="graph-row">
          <a class="thumb" href="{figure["detail_url"]}"><img src="{figure["url"]}" alt="{figure["title"]}" /></a>
          <div>
            <span class="eyebrow">FIGURE</span>
            <h2>{figure["title"]}</h2>
            <p>{figure["filename"]}</p>
          </div>
          <nav>
            <a href="{figure["detail_url"]}">열기</a>
            <a href="{figure["url"]}">PNG</a>
          </nav>
        </article>
        """
        for figure in figures
    )
    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>전체 분석 그래프</title>
      <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; background: #f4f1ea; color: #17202a; }}
        a {{ color: inherit; text-decoration: none; font-weight: 850; }}
        header {{ padding: 28px clamp(18px, 4vw, 54px); background: #17202a; color: #fff; display: flex; align-items: end; justify-content: space-between; gap: 18px; }}
        h1 {{ margin: 0; font-size: clamp(30px, 5vw, 58px); letter-spacing: 0; }}
        header nav {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        header nav a {{ border: 1px solid rgba(255,255,255,.28); padding: 9px 12px; border-radius: 4px; }}
        main {{ max-width: 1260px; margin: 0 auto; padding: 28px clamp(18px, 4vw, 54px) 54px; }}
        .gallery {{ display: grid; gap: 12px; }}
        .graph-row {{ min-height: 132px; display: grid; grid-template-columns: 210px 1fr auto; gap: 18px; align-items: center; padding: 12px; background: #fff; border: 1px solid #2c3440; box-shadow: 6px 6px 0 #d9d2c4; }}
        .thumb img {{ width: 210px; aspect-ratio: 16 / 9; object-fit: cover; display: block; background: #fff; border: 1px solid #d6d9df; }}
        .eyebrow {{ display: block; color: #b85c00; font-size: 11px; font-weight: 950; letter-spacing: .08em; margin-bottom: 8px; }}
        .graph-row h2 {{ margin: 0; font-size: 21px; }}
        .graph-row p {{ margin: 7px 0 0; color: #68717d; font-weight: 720; }}
        .graph-row nav {{ display: flex; gap: 8px; }}
        .graph-row nav a {{ min-width: 58px; text-align: center; border: 1px solid #17202a; padding: 9px 12px; border-radius: 4px; background: #f8fafc; }}
        @media (max-width: 760px) {{ header {{ align-items: flex-start; flex-direction: column; }} .graph-row {{ grid-template-columns: 1fr; }} .thumb img {{ width: 100%; }} .graph-row nav {{ justify-content: flex-start; }} }}
      </style>
    </head>
    <body>
      <header>
        <h1>Graph Library</h1>
        <nav><a href="/">홈</a><a href="/docs">API Docs</a><a href="/graphs/list">그래프 JSON</a></nav>
      </header>
      <main><section class="gallery">{cards or "<p>아직 생성된 그래프가 없습니다. /refresh를 먼저 실행하세요.</p>"}</section></main>
    </body>
    </html>
    """


def _forecast_reason(
    item: dict[str, Any],
    latest: dict[str, Any],
    today_price: float | None,
    previous_price: float | None,
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
        movement = "오늘과 거의 비슷한 보합권으로 예상됩니다"
    elif today_diff > 0 and day_diff >= 0:
        movement = "오늘보다 높고 전일 예측보다도 올라 상승 흐름으로 예상됩니다"
    elif today_diff > 0 and day_diff < 0:
        movement = "오늘보다는 높지만 전일 예측보다 낮아 상승폭이 줄어드는 조정 구간으로 보입니다"
    elif today_diff < 0 and day_diff <= 0:
        movement = "오늘보다 낮고 전일 예측보다도 내려 하락 흐름으로 예상됩니다"
    else:
        movement = "오늘보다는 낮지만 전일 예측보다 올라 단기 반등 구간으로 보입니다"

    risk_text = ""
    if news_count > 0 and abs(news_adjustment) > 0:
        risk_text = (
            f"뉴스 {news_count}건은 예측값을 올리는 방향으로 {abs(news_adjustment) * 100:.2f}% 보정했습니다."
            if news_adjustment > 0
            else f"뉴스 {news_count}건은 예측값을 낮추는 방향으로 {abs(news_adjustment) * 100:.2f}% 보정했습니다."
        )

    market_parts = []
    if wti and brent:
        market_parts.append(f"WTI {wti:.2f}, Brent {brent:.2f}")
    if exchange:
        market_parts.append(f"원/달러 {exchange:,.2f}원")
    market_text = f"입력 지표는 {', '.join(market_parts)}입니다." if market_parts else ""

    return " ".join(text for text in [movement, risk_text, market_text] if text)


def _latest_snapshot() -> dict[str, Any]:
    _ensure_current_outputs()
    raw = _read_csv(PATHS.raw, index_col=0, parse_dates=True)
    forecast = _read_csv(PATHS.forecast, parse_dates=["date"])
    meta = _read_csv(PATHS.online_meta)
    audit = _read_csv(PATHS.source_audit)
    news = load_news_adjustment()

    latest: dict[str, Any] = {}
    if not raw.empty:
        row = raw.iloc[-1]
        latest = {
            "date": str(raw.index[-1].date()),
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
            }
            forecast_rows.append(
                {
                    "date": str(pd.Timestamp(item["date"]).date()),
                    "predicted_domestic_price": predicted,
                    "news_risk_score": round(float(item.get("news_risk_score", 0)), 4),
                    "news_adjustment_pct": normalized_item["news_adjustment_pct"],
                    "news_article_count": normalized_item["news_article_count"],
                    "reason": _forecast_reason(normalized_item, latest, today_price, previous_price),
                }
            )
            previous_price = predicted

    return {
        "latest": latest,
        "forecast": forecast_rows,
        "news": news,
        "meta": dict(zip(meta.get("key", []), meta.get("value", []))) if not meta.empty else {},
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


@app.get("/", response_class=HTMLResponse, tags=["homepage"])
def homepage() -> str:
    return _HOME_HTML


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
def graph_gallery() -> str:
    return _graph_gallery_html()


@app.get("/graphs/{filename}", response_class=HTMLResponse, tags=["graphs"])
def graph_detail(filename: str) -> str:
    safe_name = Path(filename).name
    figures = {figure["filename"]: figure for figure in _figure_files()}
    figure = figures.get(safe_name)
    if figure is None:
        return "<h1>그래프를 찾을 수 없습니다</h1><p><a href='/graphs'>전체 그래프로 돌아가기</a></p>"
    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{figure["title"]}</title>
      <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; background: #f7f8fa; color: #222; }}
        header {{ padding: 28px 42px 20px; background: #fff; border-bottom: 1px solid #e5e7eb; }}
        h1 {{ margin: 0; font-size: 32px; }}
        main {{ padding: 28px 42px 48px; }}
        img {{ width: 100%; max-width: 1320px; display: block; margin: 0 auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }}
        a {{ color: #111827; font-weight: 700; }}
      </style>
    </head>
    <body>
      <header><h1>{figure["title"]}</h1><p><a href="/graphs">전체 그래프</a> · <a href="{figure["url"]}">원본 PNG</a> · <a href="/docs">API Docs</a></p></header>
      <main><img src="{figure["url"]}" alt="{figure["title"]}" /></main>
    </body>
    </html>
    """


_DOCS_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>유가 예측 API 실행</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --ink: #162033;
      --muted: #64748b;
      --line: #dbe2ea;
      --panel: #ffffff;
      --blue: #9996e2;
      --green: #07855f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Segoe UI", sans-serif;
    }
    a { color: inherit; text-decoration: none; }
    button, textarea { font: inherit; }
    .wrap {
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 30px clamp(16px, 4vw, 34px) 54px;
    }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(34px, 6vw, 62px);
      line-height: 1.02;
      letter-spacing: 0;
    }
    .lead {
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.65;
      word-break: keep-all;
    }
    .top-links {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .link,
    .run {
      min-height: 42px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-weight: 900;
      cursor: pointer;
    }
    .run {
      border-color: var(--blue);
      background: var(--blue);
      color: #fff;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .card {
      min-height: 182px;
      display: grid;
      align-content: space-between;
      gap: 14px;
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(15, 23, 42, .07);
    }
    .card h2 {
      margin: 0;
      font-size: 22px;
    }
    .card p {
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.55;
      font-weight: 750;
      word-break: keep-all;
    }
    .method {
      display: inline-flex;
      width: fit-content;
      min-height: 28px;
      align-items: center;
      padding: 0 9px;
      border-radius: 999px;
      background: #f0effc;
      color: var(--blue);
      font-size: 12px;
      font-weight: 950;
    }
    .method.post {
      background: #e7f7ef;
      color: var(--green);
    }
    textarea {
      width: 100%;
      min-height: 104px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      line-height: 1.6;
      outline: none;
    }
    .result {
      margin-top: 16px;
      background: #111827;
      color: #f8fafc;
      border-radius: 8px;
      border: 1px solid #0f172a;
      overflow: hidden;
    }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,.12);
      font-weight: 900;
    }
    pre {
      margin: 0;
      min-height: 260px;
      max-height: 520px;
      overflow: auto;
      padding: 15px;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.55;
      font-size: 13px;
    }
    @media (max-width: 800px) {
      header, .grid { grid-template-columns: 1fr; }
      .top-links { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <header>
      <div>
        <h1>유가 분석 실행 센터</h1>
        <p class="lead">국내 유가, 국제 유가, 환율, 뉴스 리스크 기반 분석 기능을 한 화면에서 실행합니다.</p>
      </div>
      <nav class="top-links">
        <a class="link" href="/">홈</a>
        <a class="link" href="/graphs">그래프</a>
        <a class="link" href="/openapi.json">OpenAPI JSON</a>
      </nav>
    </header>

    <section class="grid">
      <article class="card">
        <div>
          <span class="method">GET /summary</span>
          <h2>오늘 유가 요약</h2>
          <p>오늘 국내 유가, WTI, Brent, 환율, 뉴스 리스크를 한 번에 확인합니다.</p>
        </div>
        <button class="run" onclick="runRequest('GET', '/summary')">요약 보기</button>
      </article>

      <article class="card">
        <div>
          <span class="method">GET /forecast</span>
          <h2>7일 예측</h2>
          <p>오늘 날짜 기준 다음 7일간 예측 유가 표를 불러옵니다.</p>
        </div>
        <button class="run" onclick="runRequest('GET', '/forecast')">예측 보기</button>
      </article>

      <article class="card">
        <div>
          <span class="method">GET /graphs/list</span>
          <h2>그래프 목록</h2>
          <p>웹에서 볼 수 있는 전체 그래프 파일과 바로가기 주소를 확인합니다.</p>
        </div>
        <button class="run" onclick="runRequest('GET', '/graphs/list')">목록 보기</button>
      </article>

      <article class="card">
        <div>
          <span class="method post">POST /refresh</span>
          <h2>최신 데이터 갱신</h2>
          <p>인터넷에서 최신 데이터를 다시 받고 EDA/예측/그래프를 재생성합니다.</p>
        </div>
        <button class="run" onclick="refreshData()">최신화 실행</button>
      </article>
    </section>

    <section class="result">
      <div class="result-head">
        <span id="resultTitle">실행 결과</span>
        <span id="resultStatus">대기 중</span>
      </div>
      <pre id="output">위 버튼 중 하나를 누르면 결과가 여기에 표시됩니다.</pre>
    </section>
  </main>

  <script>
    function showResult(title, status, data) {
      document.getElementById('resultTitle').textContent = title;
      document.getElementById('resultStatus').textContent = status;
      document.getElementById('output').textContent =
        typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    }
    async function runRequest(method, path, body = null) {
      showResult(`${method} ${path}`, '실행 중...', '');
      try {
        const options = { method, headers: {} };
        if (body) {
          options.headers['Content-Type'] = 'application/json';
          options.body = JSON.stringify(body);
        }
        const response = await fetch(path, options);
        const data = await response.json();
        showResult(`${method} ${path}`, `${response.status} ${response.statusText}`, data);
      } catch (error) {
        showResult(`${method} ${path}`, '오류', String(error));
      }
    }
    async function refreshData() {
      if (!confirm('최신 데이터 수집과 그래프 재생성을 실행할까요? 시간이 걸릴 수 있습니다.')) return;
      await runRequest('POST', '/refresh');
    }
  </script>
</body>
</html>
"""


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def simple_docs() -> str:
    return _DOCS_HTML


@app.post("/refresh", response_model=RefreshResponse, tags=["analysis"])
def refresh_data() -> RefreshResponse:
    global _AUTO_REFRESH_DONE_FOR
    _run_full_refresh()
    _AUTO_REFRESH_DONE_FOR = pd.Timestamp.today().date().isoformat()
    return RefreshResponse(status="ok", message="최신 데이터 수집, EDA, 7일 예측을 완료했습니다.")


@app.post("/agent", response_model=AgentResponse, tags=["agent"])
def agent(request: AgentRequest) -> AgentResponse:
    answer, facts = _build_agent_answer(request.question)
    return AgentResponse(question=request.question, answer=answer, facts=facts)


_HOME_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Oil Price Market Terminal</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #f4f1ea;
      --ink: #17202a;
      --muted: #68717d;
      --line: #2c3440;
      --hair: #d9d2c4;
      --white: #fffdf8;
      --blue: #9996e2;
      --red: #d91e18;
      --green: #087f5b;
      --gold: #b85c00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(90deg, rgba(23,32,42,.035) 1px, transparent 1px),
        linear-gradient(rgba(23,32,42,.035) 1px, transparent 1px),
        var(--paper);
      background-size: 34px 34px;
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Segoe UI", sans-serif;
    }
    a { color: inherit; text-decoration: none; }
    button, textarea {
      font: inherit;
      color: inherit;
    }
    button { cursor: pointer; }
    .terminal {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }
    .command-bar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: center;
      padding: 14px clamp(16px, 4vw, 52px);
      background: #17202a;
      color: #fff;
      border-bottom: 4px solid #b85c00;
    }
    .brand {
      display: flex;
      align-items: baseline;
      gap: 14px;
      min-width: 0;
    }
    .brand strong {
      font-size: clamp(22px, 3vw, 38px);
      line-height: 1;
      letter-spacing: 0;
      white-space: nowrap;
    }
    .brand span {
      color: #cbd5e1;
      font-size: 12px;
      font-weight: 850;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .command-links {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: wrap;
    }
    .terminal-button {
      min-height: 38px;
      padding: 0 13px;
      border: 1px solid currentColor;
      border-radius: 4px;
      background: transparent;
      color: currentColor;
      font-weight: 900;
    }
    .terminal-button.fill {
      background: #fff;
      color: #17202a;
      border-color: #fff;
    }
    .ticker-tape {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      border-bottom: 1px solid var(--line);
      background: var(--white);
      overflow-x: auto;
    }
    .ticker-item {
      min-height: 78px;
      padding: 14px 18px;
      border-right: 1px solid var(--hair);
    }
    .ticker-item:last-child { border-right: 0; }
    .ticker-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 950;
      letter-spacing: .04em;
    }
    .ticker-value {
      margin-top: 8px;
      font-size: 24px;
      font-weight: 950;
      letter-spacing: 0;
    }
    .workspace {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: 26px clamp(16px, 4vw, 52px) 58px;
    }
    .tabbar {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      border: 2px solid var(--line);
      background: var(--white);
      box-shadow: 8px 8px 0 var(--hair);
    }
    .tabbar button {
      min-height: 54px;
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      font-size: 17px;
      font-weight: 950;
    }
    .tabbar button:last-child { border-right: 0; }
    .tabbar button.active {
      background: var(--ink);
      color: #fff;
    }
    .screen { display: none; margin-top: 26px; }
    .screen.active { display: block; }
    .board {
      background: var(--white);
      border: 2px solid var(--line);
      box-shadow: 8px 8px 0 var(--hair);
    }
    .board-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 15px 18px;
      border-bottom: 2px solid var(--line);
      background: #fbf8ef;
    }
    .board-head h2 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }
    .board-head small {
      color: var(--muted);
      font-weight: 850;
    }
    .overview-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(360px, .92fr);
      gap: 18px;
    }
    .quote-main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      gap: 22px;
      padding: 24px;
    }
    .quote-main h1 {
      margin: 0;
      font-size: clamp(44px, 8vw, 104px);
      line-height: .92;
      letter-spacing: 0;
    }
    .quote-text {
      margin: 16px 0 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
      max-width: 780px;
    }
    .price-ticket {
      align-self: stretch;
      display: grid;
      align-content: center;
      justify-items: end;
      padding: 18px;
      border-left: 2px solid var(--line);
    }
    .price-ticket span {
      color: var(--muted);
      font-size: 13px;
      font-weight: 950;
    }
    .price-ticket strong {
      margin-top: 8px;
      font-size: 38px;
      letter-spacing: 0;
      text-align: right;
    }
    .price-ticket em {
      margin-top: 9px;
      font-style: normal;
      font-weight: 950;
    }
    .up { color: var(--red); }
    .down { color: var(--blue); }
    .flat { color: var(--green); }
    .matrix {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      border-top: 2px solid var(--line);
    }
    .metric {
      min-height: 116px;
      padding: 18px;
      border-right: 1px solid var(--hair);
      background: #fff;
    }
    .metric:last-child { border-right: 0; }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 950;
    }
    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 24px;
      letter-spacing: 0;
    }
    .metric small {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-weight: 750;
    }
    .forecast-list {
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .forecast-line {
      display: grid;
      grid-template-columns: 94px 1fr 92px;
      gap: 12px;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid var(--hair);
    }
    .forecast-line:last-child { border-bottom: 0; }
    .forecast-line time {
      font-weight: 900;
      color: var(--muted);
    }
    .rail {
      height: 11px;
      position: relative;
      background: #ece6d9;
      border: 1px solid #cabfaa;
    }
    .rail i {
      position: absolute;
      top: -5px;
      width: 19px;
      height: 19px;
      border: 2px solid var(--ink);
      border-radius: 50%;
      background: var(--gold);
      transform: translateX(-50%);
    }
    .forecast-line strong {
      text-align: right;
      font-size: 15px;
    }
    .chart-grid {
      display: grid;
      grid-template-columns: 290px minmax(0, 1fr);
      gap: 18px;
    }
    .graph-menu {
      display: grid;
      align-content: start;
      gap: 8px;
    }
    .graph-menu button {
      width: 100%;
      padding: 12px;
      border: 1px solid var(--line);
      background: var(--white);
      text-align: left;
      font-weight: 900;
      border-radius: 4px;
    }
    .graph-menu button.active {
      background: var(--ink);
      color: #fff;
    }
    .graph-stage {
      min-height: 560px;
      padding: 18px;
    }
    .graph-stage img {
      width: 100%;
      display: block;
      background: #fff;
      border: 1px solid var(--hair);
    }
    .graph-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    .data-table th,
    .data-table td {
      padding: 13px 14px;
      border-bottom: 1px solid var(--hair);
      text-align: left;
    }
    .data-table th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 950;
      background: #fbf8ef;
    }
    .risk-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(340px, .8fr);
      gap: 18px;
    }
    .risk-score {
      padding: 28px;
      min-height: 260px;
      display: grid;
      align-content: center;
    }
    .risk-score span {
      color: var(--muted);
      font-weight: 950;
    }
    .risk-score strong {
      display: block;
      margin-top: 8px;
      font-size: clamp(60px, 10vw, 132px);
      line-height: .9;
      letter-spacing: 0;
    }
    .risk-score p {
      margin: 16px 0 0;
      color: var(--muted);
      line-height: 1.65;
      font-weight: 750;
    }
    .source-list {
      padding: 16px 18px;
      display: grid;
      gap: 12px;
    }
    .source-item {
      display: grid;
      grid-template-columns: 130px 1fr;
      gap: 14px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--hair);
      word-break: break-word;
    }
    .source-item:last-child { border-bottom: 0; padding-bottom: 0; }
    .source-item span {
      color: var(--muted);
      font-weight: 950;
    }
    .source-item strong {
      font-weight: 850;
    }
    .agent-grid {
      display: grid;
      grid-template-columns: minmax(0, .9fr) minmax(0, 1.1fr);
      gap: 18px;
    }
    textarea {
      width: 100%;
      min-height: 220px;
      resize: vertical;
      border: 0;
      border-bottom: 2px solid var(--line);
      background: #fff;
      padding: 18px;
      line-height: 1.6;
      font-size: 16px;
      outline: none;
    }
    .answer {
      min-height: 320px;
      margin: 0;
      padding: 20px;
      white-space: pre-wrap;
      word-break: keep-all;
      background: #17202a;
      color: #f8fafc;
      line-height: 1.75;
      font-size: 15px;
    }
    .empty {
      padding: 28px;
      color: var(--muted);
      font-weight: 900;
    }
    @media (max-width: 1020px) {
      .command-bar,
      .overview-grid,
      .chart-grid,
      .risk-grid,
      .agent-grid {
        grid-template-columns: 1fr;
      }
      .command-links { justify-content: flex-start; }
      .ticker-tape { grid-template-columns: repeat(5, 180px); }
      .quote-main { grid-template-columns: 1fr; }
      .price-ticket {
        justify-items: start;
        border-left: 0;
        border-top: 2px solid var(--line);
      }
      .price-ticket strong { text-align: left; }
      .matrix { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(2) { border-right: 0; }
      .metric:nth-child(3), .metric:nth-child(4) { border-top: 1px solid var(--hair); }
    }
    @media (max-width: 660px) {
      .brand { align-items: flex-start; flex-direction: column; gap: 6px; }
      .tabbar { grid-template-columns: 1fr; }
      .tabbar button { border-right: 0; border-bottom: 1px solid var(--line); }
      .tabbar button:last-child { border-bottom: 0; }
      .matrix { grid-template-columns: 1fr; }
      .metric { border-right: 0; border-top: 1px solid var(--hair); }
      .metric:first-child { border-top: 0; }
      .forecast-line { grid-template-columns: 1fr; }
      .forecast-line strong { text-align: left; }
      .source-item { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="terminal">
    <header class="command-bar">
      <div class="brand">
        <strong>Oil Market Terminal</strong>
        <span>High-Level Programming</span>
      </div>
      <nav class="command-links">
        <a class="terminal-button" href="/docs">API Docs</a>
        <a class="terminal-button" href="/graphs">Graph Library</a>
        <a class="terminal-button" href="/summary">JSON</a>
        <button class="terminal-button fill" onclick="refreshData()">Refresh</button>
      </nav>
    </header>

    <section class="ticker-tape" id="tickerTape"></section>

    <main class="workspace">
      <nav class="tabbar">
        <button class="active" data-screen="overview">Overview</button>
        <button data-screen="forecast">Forecast</button>
        <button data-screen="graphs">Graphs</button>
        <button data-screen="risk">Risk</button>
        <button data-screen="agent">Agent</button>
      </nav>

      <section class="screen active" id="overview">
        <div class="overview-grid">
          <article class="board">
            <div class="quote-main">
              <div>
                <h1>국내 유가<br />실시간 분석</h1>
                <p class="quote-text" id="overviewText">최신 데이터를 불러오는 중입니다.</p>
              </div>
              <aside class="price-ticket">
                <span>오늘 전국 평균</span>
                <strong id="todayPrice">-</strong>
                <em id="todayDelta">예측 계산 중</em>
              </aside>
            </div>
            <div class="matrix" id="metricMatrix"></div>
          </article>

          <article class="board">
            <div class="board-head">
              <h2>7-Day Rail</h2>
              <small id="railDate">기준일 확인 중</small>
            </div>
            <div class="forecast-list" id="forecastRail"></div>
          </article>
        </div>
      </section>

      <section class="screen" id="forecast">
        <article class="board">
          <div class="board-head">
            <div>
              <h2>Forecast Sheet</h2>
              <small>오늘 가격 대비 7일 예측 변화</small>
            </div>
            <a class="terminal-button" href="/forecast">Forecast JSON</a>
          </div>
          <table class="data-table">
            <thead>
              <tr><th>날짜</th><th>예측 유가</th><th>오늘 대비</th><th>뉴스 보정률</th><th>기사 수</th></tr>
            </thead>
            <tbody id="forecastRows"></tbody>
          </table>
        </article>
      </section>

      <section class="screen" id="graphs">
        <div class="chart-grid">
          <aside class="graph-menu" id="graphMenu"></aside>
          <article class="board">
            <div class="board-head">
              <div>
                <h2 id="selectedGraphTitle">Graph Viewer</h2>
                <small id="selectedGraphFile">그래프 로딩 중</small>
              </div>
              <a class="terminal-button" id="selectedGraphRaw" href="/graphs">PNG</a>
            </div>
            <div class="graph-stage" id="graphStage"></div>
          </article>
        </div>
      </section>

      <section class="screen" id="risk">
        <div class="risk-grid">
          <article class="board risk-score">
            <span>NEWS RISK SCORE</span>
            <strong id="riskScore">-</strong>
            <p id="riskText">뉴스 리스크와 예측 보정률을 불러오는 중입니다.</p>
          </article>
          <article class="board">
            <div class="board-head">
              <h2>Data Sources</h2>
              <small>수집 경로</small>
            </div>
            <div class="source-list" id="sourceList"></div>
          </article>
        </div>
      </section>

      <section class="screen" id="agent">
        <div class="agent-grid">
          <article class="board">
            <div class="board-head">
              <h2>Ask Agent</h2>
              <small>현재 분석 결과 기반</small>
            </div>
            <textarea id="question">전쟁 뉴스 리스크가 유가 예측에 어떤 영향을 줘?</textarea>
            <div style="padding: 16px 18px;">
              <button class="terminal-button fill" style="background:#17202a;color:#fff;border-color:#17202a;" onclick="askAgent()">Run Agent</button>
            </div>
          </article>
          <article class="board">
            <div class="board-head">
              <h2>Agent Output</h2>
              <small>요약 답변</small>
            </div>
            <pre class="answer" id="agentAnswer">질문을 입력하고 Run Agent를 누르세요.</pre>
          </article>
        </div>
      </section>
    </main>
  </div>

  <script>
    let snapshot = null;
    let figures = [];
    let selectedFigure = null;

    function fmt(value, digits = 1) {
      const number = Number(value);
      if (!Number.isFinite(number)) return '-';
      return number.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }
    function signClass(value) {
      if (value > 0) return 'up';
      if (value < 0) return 'down';
      return 'flat';
    }
    function setScreen(name) {
      document.querySelectorAll('.tabbar button').forEach(button => {
        button.classList.toggle('active', button.dataset.screen === name);
      });
      document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.toggle('active', screen.id === name);
      });
    }
    document.querySelectorAll('.tabbar button').forEach(button => {
      button.addEventListener('click', () => setScreen(button.dataset.screen));
    });
    function renderTicker(data) {
      const latest = data.latest || {};
      const news = data.news || {};
      const items = [
        ['DOMESTIC', latest.domestic_price ? `${fmt(latest.domestic_price, 1)} 원/L` : '-'],
        ['WTI', latest.wti ? `${fmt(latest.wti, 2)} $/bbl` : '-'],
        ['BRENT', latest.brent ? `${fmt(latest.brent, 2)} $/bbl` : '-'],
        ['USD/KRW', latest.exchange ? `${fmt(latest.exchange, 2)} 원` : '-'],
        ['NEWS RISK', news.news_risk_score !== undefined ? Number(news.news_risk_score).toFixed(3) : '-']
      ];
      document.getElementById('tickerTape').innerHTML = items.map(([label, value]) => `
        <div class="ticker-item">
          <div class="ticker-label">${label}</div>
          <div class="ticker-value">${value}</div>
        </div>
      `).join('');
    }
    function renderOverview(data) {
      const latest = data.latest || {};
      const forecast = data.forecast || [];
      const news = data.news || {};
      const today = Number(latest.domestic_price);
      const last = forecast.length ? Number(forecast[forecast.length - 1].predicted_domestic_price) : today;
      const diff = last - today;
      const diffText = Number.isFinite(diff)
        ? `7일 뒤 ${fmt(Math.abs(diff), 1)} 원/L ${diff >= 0 ? '상승' : '하락'}`
        : '예측 데이터 없음';
      document.getElementById('todayPrice').textContent = today ? `${fmt(today, 1)} 원/L` : '-';
      document.getElementById('todayDelta').className = signClass(diff);
      document.getElementById('todayDelta').textContent = diffText;
      document.getElementById('overviewText').textContent = latest.date
        ? `${latest.date} 기준 최신 데이터입니다. 국제 유가, 환율, 뉴스 리스크를 함께 반영해 향후 7일 국내 유가 흐름을 계산합니다.`
        : '생성된 분석 데이터가 없습니다. Refresh를 실행하면 최신 데이터 수집과 예측을 다시 수행합니다.';
      document.getElementById('railDate').textContent = latest.date ? `기준일 ${latest.date}` : '데이터 없음';
      const metrics = [
        ['WTI', latest.wti ? `${fmt(latest.wti, 2)} $/bbl` : '-', '국제 유가'],
        ['Brent', latest.brent ? `${fmt(latest.brent, 2)} $/bbl` : '-', '국제 유가'],
        ['환율', latest.exchange ? `${fmt(latest.exchange, 2)} 원` : '-', '원/달러'],
        ['뉴스', news.article_count !== undefined ? `${news.article_count}건` : '-', `보정 ${(Number(news.forecast_adjustment_pct || 0) * 100).toFixed(2)}%`]
      ];
      document.getElementById('metricMatrix').innerHTML = metrics.map(([label, value, note]) => `
        <div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>
      `).join('');
    }
    function renderForecast(data) {
      const latest = data.latest || {};
      const forecast = data.forecast || [];
      const today = Number(latest.domestic_price);
      const values = forecast.map(row => Number(row.predicted_domestic_price)).filter(Number.isFinite);
      const min = Math.min(today || 0, ...values);
      const max = Math.max(today || 0, ...values);
      const span = Math.max(1, max - min);
      document.getElementById('forecastRail').innerHTML = forecast.length ? forecast.map(row => {
        const value = Number(row.predicted_domestic_price);
        const pos = Math.max(0, Math.min(100, ((value - min) / span) * 100));
        return `<div class="forecast-line">
          <time>${row.date}</time>
          <div class="rail"><i style="left:${pos}%"></i></div>
          <strong>${fmt(value, 1)}</strong>
        </div>`;
      }).join('') : '<div class="empty">예측 데이터가 없습니다.</div>';
      document.getElementById('forecastRows').innerHTML = forecast.length ? forecast.map(row => {
        const value = Number(row.predicted_domestic_price);
        const diff = value - today;
        return `<tr>
          <td>${row.date}</td>
          <td><strong>${fmt(value, 1)} 원/L</strong></td>
          <td class="${signClass(diff)}">${diff >= 0 ? '+' : '-'}${fmt(Math.abs(diff), 1)} 원/L</td>
          <td>${(Number(row.news_adjustment_pct || 0) * 100).toFixed(2)}%</td>
          <td>${row.news_article_count || 0}</td>
        </tr>`;
      }).join('') : '<tr><td colspan="5">예측 데이터가 없습니다.</td></tr>';
    }
    function renderRisk(data) {
      const news = data.news || {};
      const sources = data.sources || {};
      const meta = data.meta || {};
      document.getElementById('riskScore').textContent = news.news_risk_score !== undefined
        ? Number(news.news_risk_score).toFixed(3)
        : '-';
      document.getElementById('riskText').textContent =
        `뉴스 기사 ${news.article_count || 0}건, 예측 보정률 ${(Number(news.forecast_adjustment_pct || 0) * 100).toFixed(2)}% 기준으로 지정학적 이벤트 영향을 반영했습니다.`;
      const rows = [
        ['국내 유가', sources.domestic_source_name || 'OPINET'],
        ['국제 유가', sources.market_source_name || 'Alpha Vantage / FRED'],
        ['뉴스', sources.news_source_name || 'Google News / GDELT'],
        ['수집 시각', meta.collected_at || meta.last_updated || '-']
      ];
      document.getElementById('sourceList').innerHTML = rows.map(([key, value]) => `
        <div class="source-item"><span>${key}</span><strong>${value}</strong></div>
      `).join('');
    }
    function selectFigure(index) {
      selectedFigure = figures[index] || figures[0] || null;
      if (!selectedFigure) {
        document.getElementById('graphStage').innerHTML = '<div class="empty">표시할 그래프가 없습니다.</div>';
        return;
      }
      document.querySelectorAll('.graph-menu button').forEach((button, buttonIndex) => {
        button.classList.toggle('active', buttonIndex === figures.indexOf(selectedFigure));
      });
      document.getElementById('selectedGraphTitle').textContent = selectedFigure.title;
      document.getElementById('selectedGraphFile').textContent = selectedFigure.filename;
      document.getElementById('selectedGraphRaw').href = selectedFigure.url;
      document.getElementById('graphStage').innerHTML = `
        <img src="${selectedFigure.url}" alt="${selectedFigure.title}" />
        <div class="graph-actions">
          <a class="terminal-button" href="${selectedFigure.detail_url}">크게 보기</a>
          <a class="terminal-button" href="${selectedFigure.url}">원본 PNG</a>
        </div>
      `;
    }
    function renderFigures() {
      document.getElementById('graphMenu').innerHTML = figures.length ? figures.map((figure, index) => `
        <button data-index="${index}">${figure.title}</button>
      `).join('') : '<div class="empty">생성된 그래프가 없습니다.</div>';
      document.querySelectorAll('.graph-menu button').forEach(button => {
        button.addEventListener('click', () => selectFigure(Number(button.dataset.index)));
      });
      const dashboardIndex = Math.max(0, figures.findIndex(figure => figure.filename === 'oil_price_dashboard.png'));
      selectFigure(dashboardIndex);
    }
    async function loadSummary() {
      const res = await fetch('/summary');
      const data = await res.json();
      snapshot = data;
      renderTicker(data);
      renderOverview(data);
      renderForecast(data);
      renderRisk(data);
    }
    async function loadFigures() {
      const res = await fetch('/graphs/list');
      figures = await res.json();
      renderFigures();
    }
    async function askAgent() {
      const question = document.getElementById('question').value;
      const answer = document.getElementById('agentAnswer');
      answer.textContent = '분석 중...';
      const res = await fetch('/agent', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question})
      });
      const data = await res.json();
      answer.textContent = data.answer;
    }
    async function refreshData() {
      if (!confirm('최신 데이터 수집과 예측을 다시 실행할까요?')) return;
      const res = await fetch('/refresh', { method: 'POST' });
      const data = await res.json();
      alert(data.message || '완료');
      await loadSummary();
      await loadFigures();
    }
    loadSummary();
    loadFigures();
  </script>
</body>
</html>
"""

_HOME_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>유가 예측 서비스</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f9fc;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d7dee8;
      --panel: #ffffff;
      --blue: #9996e2;
      --blue-soft: #f0effc;
      --green: #087f5b;
      --green-soft: #e7f7ef;
      --red: #d92d20;
      --amber: #b7791f;
      --shadow: 0 18px 45px rgba(15, 23, 42, .08);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Segoe UI", sans-serif;
    }
    a { color: inherit; text-decoration: none; }
    button, textarea { font: inherit; }
    button { cursor: pointer; }
    .page {
      width: min(1240px, 100%);
      margin: 0 auto;
      padding: 24px clamp(16px, 4vw, 34px) 58px;
    }
    .hero {
      min-height: 320px;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, .9fr);
      gap: 20px;
      align-items: stretch;
    }
    .hero-main,
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .hero-main {
      padding: clamp(24px, 4vw, 42px);
      display: grid;
      align-content: space-between;
      gap: 28px;
    }
    .eyebrow {
      color: var(--blue);
      font-size: 13px;
      font-weight: 900;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    h1 {
      margin: 10px 0 0;
      font-size: clamp(36px, 6vw, 72px);
      line-height: 1.02;
      letter-spacing: 0;
    }
    .hero-copy {
      margin: 16px 0 0;
      max-width: 680px;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.7;
      word-break: keep-all;
    }
    .quick-actions {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .action {
      min-height: 74px;
      display: grid;
      align-content: center;
      gap: 4px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      text-align: center;
      font-weight: 900;
    }
    .action span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .action.primary {
      background: var(--blue);
      color: #fff;
      border-color: var(--blue);
    }
    .action.primary span { color: rgba(255,255,255,.78); }
    .today-box {
      display: grid;
      grid-template-rows: auto 1fr auto;
      padding: 24px;
    }
    .today-label {
      color: var(--muted);
      font-size: 14px;
      font-weight: 900;
    }
    .today-price {
      align-self: center;
      font-size: clamp(44px, 7vw, 78px);
      line-height: 1;
      font-weight: 950;
      letter-spacing: 0;
    }
    .today-meta {
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-weight: 800;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      min-height: 32px;
      padding: 0 10px;
      border-radius: 999px;
      background: var(--blue-soft);
      color: var(--blue);
      font-weight: 900;
      font-size: 13px;
    }
    .section {
      margin-top: 22px;
    }
    .section-head {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 12px;
    }
    .section-head h2 {
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }
    .section-head p {
      margin: 5px 0 0;
      color: var(--muted);
      font-weight: 750;
    }
    .link-button,
    .solid-button {
      min-height: 40px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0 13px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      font-weight: 900;
      white-space: nowrap;
    }
    .solid-button {
      border-color: var(--blue);
      background: var(--blue);
      color: #fff;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      min-height: 112px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 900;
    }
    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 27px;
      letter-spacing: 0;
    }
    .metric small {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-weight: 750;
    }
    .forecast-layout {
      display: grid;
      grid-template-columns: minmax(0, .95fr) minmax(520px, 1.05fr);
      gap: 16px;
    }
    .graph-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(340px, .75fr);
      gap: 16px;
    }
    .panel-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .panel-head h3 {
      margin: 0;
      font-size: 20px;
    }
    .chart-frame {
      padding: 16px;
    }
    .chart-frame img {
      width: 100%;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }
    th {
      color: var(--muted);
      background: #f8fafc;
      font-size: 12px;
      font-weight: 950;
    }
    .up { color: var(--red); font-weight: 950; }
    .down { color: var(--blue); font-weight: 950; }
    .flat { color: var(--green); font-weight: 950; }
    .reason-cell {
      min-width: 260px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      word-break: keep-all;
    }
    .graph-list {
      display: grid;
      gap: 8px;
      padding: 14px;
      max-height: 640px;
      overflow: auto;
    }
    .graph-list button {
      width: 100%;
      min-height: 46px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      text-align: left;
      padding: 10px 12px;
      font-weight: 900;
      color: var(--ink);
    }
    .graph-list button.active {
      border-color: var(--blue);
      background: var(--blue-soft);
      color: var(--blue);
    }
    .status-line {
      margin-top: 12px;
      padding: 12px 14px;
      border: 1px solid #bee3d4;
      background: var(--green-soft);
      color: var(--green);
      border-radius: 8px;
      font-weight: 850;
      word-break: keep-all;
    }
    @media (max-width: 980px) {
      .hero,
      .forecast-layout,
      .graph-layout {
        grid-template-columns: 1fr;
      }
      .quick-actions,
      .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 620px) {
      .quick-actions,
      .metrics {
        grid-template-columns: 1fr;
      }
      .section-head {
        align-items: flex-start;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-main">
        <div>
          <div class="eyebrow">Oil Price Forecast</div>
          <h1>국내 유가 예측</h1>
          <p class="hero-copy">
            오늘 전국 휘발유 평균가, 국제 유가, 환율, 뉴스 리스크를 한 화면에서 확인합니다.
            아래 버튼만 누르면 필요한 분석으로 바로 이동합니다.
          </p>
          <div class="status-line" id="freshStatus">최신 데이터 확인 중...</div>
        </div>
        <nav class="quick-actions">
          <a class="action primary" href="#today">오늘 유가<span>현재 가격</span></a>
          <a class="action" href="#forecast">7일 예측<span>미래 가격</span></a>
          <a class="action" href="#graphs">그래프 보기<span>전체 차트</span></a>
          <button class="action" onclick="refreshData()">최신화<span>데이터 재수집</span></button>
        </nav>
      </div>

      <aside class="panel today-box" id="today">
        <div>
          <div class="today-label">오늘 전국 휘발유 평균가</div>
          <span class="badge" id="baseDate">기준일 확인 중</span>
        </div>
        <div class="today-price" id="todayPrice">-</div>
        <div class="today-meta">
          <span id="forecastDelta">7일 예측 계산 중</span>
          <span id="sourceName">데이터 출처 확인 중</span>
        </div>
      </aside>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>핵심 지표</h2>
          <p>국내 유가 예측에 들어가는 주요 입력값입니다.</p>
        </div>
        <a class="link-button" href="/summary">JSON 보기</a>
      </div>
      <div class="metrics" id="metricGrid"></div>
    </section>

    <section class="section" id="forecast">
      <div class="section-head">
        <div>
          <h2>7일 예측</h2>
          <p>오늘 가격에서 다음 7일 동안 어떻게 움직이는지 보여줍니다.</p>
        </div>
        <a class="link-button" href="/forecast">예측 데이터</a>
      </div>
      <div class="forecast-layout">
        <article class="panel">
          <div class="panel-head">
            <h3>예측 그래프</h3>
            <a class="link-button" id="dashboardPng" href="/figures/oil_price_dashboard.png">PNG</a>
          </div>
          <div class="chart-frame" id="mainChart"></div>
        </article>
        <article class="panel">
          <div class="panel-head"><h3>예측표</h3></div>
          <table>
            <thead><tr><th>날짜</th><th>예측</th><th>변화</th><th>변화 이유</th></tr></thead>
            <tbody id="forecastRows"></tbody>
          </table>
        </article>
      </div>
    </section>

    <section class="section" id="graphs">
      <div class="section-head">
        <div>
          <h2>그래프</h2>
          <p>보고서에 들어가는 그래프를 바로 선택해서 확인합니다.</p>
        </div>
        <a class="link-button" href="/graphs">전체 그래프 페이지</a>
      </div>
      <div class="graph-layout">
        <article class="panel">
          <div class="panel-head">
            <h3 id="selectedGraphTitle">그래프 선택</h3>
            <a class="link-button" id="selectedGraphRaw" href="/graphs">원본</a>
          </div>
          <div class="chart-frame" id="graphStage"></div>
        </article>
        <aside class="panel">
          <div class="panel-head"><h3>그래프 목록</h3></div>
          <div class="graph-list" id="graphList"></div>
        </aside>
      </div>
    </section>

  </main>

  <script>
    let figures = [];
    let selectedFigureIndex = 0;

    function fmt(value, digits = 1) {
      const number = Number(value);
      if (!Number.isFinite(number)) return '-';
      return number.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }
    function changeClass(value) {
      if (value > 0) return 'up';
      if (value < 0) return 'down';
      return 'flat';
    }
    function withCacheBust(url) {
      const sep = url.includes('?') ? '&' : '?';
      return `${url}${sep}client=${Date.now()}`;
    }
    function renderSummary(data) {
      const latest = data.latest || {};
      const forecast = data.forecast || [];
      const news = data.news || {};
      const sources = data.sources || {};
      const meta = data.meta || {};
      const todayPrice = Number(latest.domestic_price);
      const lastForecast = forecast.length ? Number(forecast[forecast.length - 1].predicted_domestic_price) : todayPrice;
      const diff = lastForecast - todayPrice;
      const isCurrent = latest.date === new Date().toISOString().slice(0, 10);

      document.getElementById('freshStatus').textContent = isCurrent
        ? `최신 데이터입니다. 현재 기준일: ${latest.date}`
        : `데이터 기준일: ${latest.date || '-'} / 서버가 최신화를 시도합니다.`;
      document.getElementById('baseDate').textContent = latest.date ? `${latest.date} 기준` : '데이터 없음';
      document.getElementById('todayPrice').textContent = todayPrice ? `${fmt(todayPrice, 1)} 원/L` : '-';
      document.getElementById('forecastDelta').className = changeClass(diff);
      document.getElementById('forecastDelta').textContent = Number.isFinite(diff)
        ? `7일 뒤 ${fmt(Math.abs(diff), 1)} 원/L ${diff >= 0 ? '상승' : '하락'} 전망`
        : '예측 데이터 없음';
      document.getElementById('sourceName').textContent = sources.domestic_source_name || 'OPINET';

      const metrics = [
        ['WTI', latest.wti ? `${fmt(latest.wti, 2)} $/bbl` : '-', '국제 유가'],
        ['Brent', latest.brent ? `${fmt(latest.brent, 2)} $/bbl` : '-', '국제 유가'],
        ['원/달러', latest.exchange ? `${fmt(latest.exchange, 2)} 원` : '-', '환율'],
        ['뉴스 리스크', news.news_risk_score !== undefined ? Number(news.news_risk_score).toFixed(3) : '-', `${news.article_count || 0}개 기사`]
      ];
      document.getElementById('metricGrid').innerHTML = metrics.map(([label, value, note]) => `
        <div class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>
      `).join('');
      document.getElementById('forecastRows').innerHTML = forecast.length ? forecast.map(row => {
        const value = Number(row.predicted_domestic_price);
        const rowDiff = value - todayPrice;
        return `<tr>
          <td>${row.date}</td>
          <td><strong>${fmt(value, 1)}</strong></td>
          <td class="${changeClass(rowDiff)}">${rowDiff >= 0 ? '+' : '-'}${fmt(Math.abs(rowDiff), 1)}</td>
          <td class="reason-cell">${row.reason || '최근 유가와 외부 지표 흐름을 반영한 예측입니다.'}</td>
        </tr>`;
      }).join('') : '<tr><td colspan="4">예측 데이터가 없습니다.</td></tr>';

      if (figures.length) renderMainChart();
    }
    function renderMainChart() {
      const dashboard = figures.find(figure => figure.filename === 'oil_price_dashboard.png') || figures[0];
      if (!dashboard) return;
      document.getElementById('dashboardPng').href = dashboard.url;
      document.getElementById('mainChart').innerHTML = `<img src="${withCacheBust(dashboard.url)}" alt="${dashboard.title}" />`;
    }
    function renderFigures() {
      const list = document.getElementById('graphList');
      list.innerHTML = figures.length ? figures.map((figure, index) => `
        <button class="${index === selectedFigureIndex ? 'active' : ''}" data-index="${index}">${figure.title}</button>
      `).join('') : '<div style="padding:14px;color:#64748b;font-weight:850;">그래프가 없습니다.</div>';
      list.querySelectorAll('button').forEach(button => {
        button.addEventListener('click', () => selectFigure(Number(button.dataset.index)));
      });
      selectFigure(selectedFigureIndex);
      renderMainChart();
    }
    function selectFigure(index) {
      if (!figures.length) return;
      selectedFigureIndex = Math.max(0, Math.min(index, figures.length - 1));
      const figure = figures[selectedFigureIndex];
      document.getElementById('selectedGraphTitle').textContent = figure.title;
      document.getElementById('selectedGraphRaw').href = figure.url;
      document.getElementById('graphStage').innerHTML = `<img src="${withCacheBust(figure.url)}" alt="${figure.title}" />`;
      document.querySelectorAll('#graphList button').forEach((button, idx) => {
        button.classList.toggle('active', idx === selectedFigureIndex);
      });
    }
    async function loadSummary() {
      const response = await fetch(`/summary?client=${Date.now()}`);
      const data = await response.json();
      renderSummary(data);
    }
    async function loadFigures() {
      const response = await fetch(`/graphs/list?client=${Date.now()}`);
      figures = await response.json();
      const dashboardIndex = figures.findIndex(figure => figure.filename === 'oil_price_dashboard.png');
      selectedFigureIndex = dashboardIndex >= 0 ? dashboardIndex : 0;
      renderFigures();
    }
    async function refreshData() {
      const status = document.getElementById('freshStatus');
      status.textContent = '최신 데이터 수집, 그래프 재생성 중입니다...';
      const response = await fetch('/refresh', { method: 'POST' });
      const data = await response.json();
      status.textContent = data.message || '최신화 완료';
      await loadSummary();
      await loadFigures();
    }
    loadSummary().then(loadFigures);
  </script>
</body>
</html>
"""
