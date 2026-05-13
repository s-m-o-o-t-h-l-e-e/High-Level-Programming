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
)
app.mount("/figures", StaticFiles(directory=str(PATHS.figures)), name="figures")


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


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
    for path in sorted(PATHS.figures.glob("*.png")):
        files.append(
            {
                "filename": path.name,
                "title": titles.get(path.name, path.stem.replace("_", " ")),
                "url": f"/figures/{path.name}",
                "detail_url": f"/graphs/{path.name}",
            }
        )
    return files


def _graph_gallery_html() -> str:
    figures = _figure_files()
    cards = "\n".join(
        f"""
        <article class="graph-card">
          <h2>{figure["title"]}</h2>
          <a href="{figure["detail_url"]}"><img src="{figure["url"]}" alt="{figure["title"]}" /></a>
          <p><a href="{figure["detail_url"]}">크게 보기</a> · <a href="{figure["url"]}">원본 PNG</a></p>
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
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; background: #f7f8fa; color: #222; }}
        header {{ padding: 34px 48px 24px; background: #fff; border-bottom: 1px solid #e5e7eb; }}
        h1 {{ margin: 0; font-size: 36px; }}
        nav {{ margin-top: 16px; display: flex; gap: 10px; }}
        a {{ color: #111827; font-weight: 700; }}
        main {{ max-width: 1280px; margin: 0 auto; padding: 28px 48px 48px; }}
        .gallery {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
        .graph-card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }}
        .graph-card h2 {{ margin: 0 0 12px; font-size: 20px; }}
        .graph-card img {{ width: 100%; display: block; border: 1px solid #e5e7eb; border-radius: 6px; background: #fff; }}
        @media (max-width: 900px) {{ header, main {{ padding-left: 22px; padding-right: 22px; }} .gallery {{ grid-template-columns: 1fr; }} }}
      </style>
    </head>
    <body>
      <header>
        <h1>전체 분석 그래프</h1>
        <nav><a href="/">홈</a><a href="/docs">API Docs</a><a href="/graphs/list">그래프 JSON</a></nav>
      </header>
      <main><section class="gallery">{cards or "<p>아직 생성된 그래프가 없습니다. /refresh를 먼저 실행하세요.</p>"}</section></main>
    </body>
    </html>
    """


def _latest_snapshot() -> dict[str, Any]:
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
        for item in forecast.to_dict(orient="records"):
            forecast_rows.append(
                {
                    "date": str(pd.Timestamp(item["date"]).date()),
                    "predicted_domestic_price": round(float(item["predicted_domestic_price"]), 2),
                    "news_risk_score": round(float(item.get("news_risk_score", 0)), 4),
                    "news_adjustment_pct": round(float(item.get("news_adjustment_pct", 0)), 4),
                    "news_article_count": int(item.get("news_article_count", 0)),
                }
            )

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


@app.post("/refresh", response_model=RefreshResponse, tags=["analysis"])
def refresh_data() -> RefreshResponse:
    df = collect_and_preprocess()
    run_eda(df)
    forecast_next_7_days(device="auto", show_gui=False)
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
  <title>유가 예측 분석</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; color: #222; background: #f7f8fa; }
    header { padding: 34px 48px 24px; background: #fff; border-bottom: 1px solid #e5e7eb; }
    h1 { margin: 0; font-size: 38px; letter-spacing: 0; }
    nav { margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap; }
    nav a, button { border: 1px solid #333; background: #fff; color: #222; padding: 10px 16px; font-size: 15px; font-weight: 700; text-decoration: none; cursor: pointer; }
    main { padding: 28px 48px 48px; max-width: 1280px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; }
    .label { color: #6b7280; font-size: 13px; font-weight: 700; }
    .value { margin-top: 8px; font-size: 26px; font-weight: 800; }
    .section { margin-top: 22px; }
    .charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .charts img { width: 100%; display: block; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }
    textarea { width: 100%; min-height: 82px; box-sizing: border-box; border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; font-size: 15px; resize: vertical; }
    pre { white-space: pre-wrap; word-break: keep-all; background: #111827; color: #f9fafb; border-radius: 8px; padding: 16px; min-height: 72px; }
    table { width: 100%; border-collapse: collapse; background: #fff; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 10px; text-align: left; }
    th { color: #6b7280; font-size: 13px; }
    @media (max-width: 900px) { .grid, .charts { grid-template-columns: 1fr; } header, main { padding-left: 22px; padding-right: 22px; } }
  </style>
</head>
<body>
  <header>
    <h1>유가 예측 분석</h1>
    <nav>
      <a href="/docs">API Docs</a>
      <a href="/graphs">전체 그래프</a>
      <a href="/summary">Summary JSON</a>
      <button onclick="refreshData()">최신 데이터 갱신</button>
    </nav>
  </header>
  <main>
    <section class="grid" id="cards"></section>

    <section class="section card">
      <div class="label">7일 예측</div>
      <table>
        <thead><tr><th>날짜</th><th>예측 유가</th><th>뉴스 기사 수</th><th>뉴스 보정률</th></tr></thead>
        <tbody id="forecastRows"></tbody>
      </table>
    </section>

    <section class="section charts" id="figureGallery">
    </section>

    <section class="section card">
      <div class="label">분석 Agent</div>
      <textarea id="question">전쟁 뉴스 리스크가 유가 예측에 어떤 영향을 줘?</textarea>
      <div style="margin-top:10px"><button onclick="askAgent()">질문하기</button></div>
      <pre id="agentAnswer">질문을 입력하고 버튼을 누르세요.</pre>
    </section>
  </main>
  <script>
    async function loadSummary() {
      const res = await fetch('/summary');
      const data = await res.json();
      const latest = data.latest || {};
      const cards = [
        ['기준일', latest.date || '-'],
        ['국내 유가', latest.domestic_price ? latest.domestic_price.toLocaleString() + ' 원/L' : '-'],
        ['WTI', latest.wti ? latest.wti + ' $/bbl' : '-'],
        ['Brent', latest.brent ? latest.brent + ' $/bbl' : '-'],
        ['원/달러', latest.exchange ? latest.exchange.toLocaleString() + ' 원' : '-'],
        ['뉴스 리스크', data.news ? Number(data.news.news_risk_score).toFixed(3) : '-'],
        ['예측 보정률', data.news ? (Number(data.news.forecast_adjustment_pct) * 100).toFixed(2) + '%' : '-'],
        ['뉴스 기사 수', data.news ? data.news.article_count : '-']
      ];
      document.getElementById('cards').innerHTML = cards.map(([label, value]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
      document.getElementById('forecastRows').innerHTML = (data.forecast || []).map(row => `<tr><td>${row.date}</td><td>${row.predicted_domestic_price.toLocaleString()} 원/L</td><td>${row.news_article_count}</td><td>${(row.news_adjustment_pct * 100).toFixed(2)}%</td></tr>`).join('');
    }
    async function loadFigures() {
      const res = await fetch('/graphs/list');
      const figures = await res.json();
      document.getElementById('figureGallery').innerHTML = figures.map(figure => `
        <article class="card">
          <div class="label">${figure.title}</div>
          <a href="${figure.detail_url}"><img src="${figure.url}" alt="${figure.title}" style="margin-top:10px" /></a>
        </article>
      `).join('');
    }
    async function askAgent() {
      const question = document.getElementById('question').value;
      const res = await fetch('/agent', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question}) });
      const data = await res.json();
      document.getElementById('agentAnswer').textContent = data.answer;
    }
    async function refreshData() {
      if (!confirm('최신 데이터 수집과 예측을 다시 실행할까요? 시간이 걸릴 수 있습니다.')) return;
      const res = await fetch('/refresh', { method: 'POST' });
      const data = await res.json();
      alert(data.message || '완료');
      await loadSummary();
    }
    loadSummary();
    loadFigures();
  </script>
</body>
</html>
"""
