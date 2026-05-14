import json
import math
import os
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from config import PATHS


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

NEWS_QUERY = '(Iran OR Hormuz OR Tehran) (oil OR crude OR tanker OR blockade OR Strait)'

ESCALATION_WORDS = {
    "attack": 1.0,
    "attacks": 1.0,
    "blockade": 1.3,
    "closed": 1.2,
    "closure": 1.2,
    "deadlock": 1.0,
    "disrupted": 1.0,
    "disruption": 1.0,
    "escalation": 1.0,
    "missile": 1.0,
    "no progress": 1.2,
    "not happy": 0.9,
    "sanctions": 0.7,
    "shipping constraints": 1.0,
    "standoff": 1.0,
    "strait": 0.4,
    "war": 0.8,
    "전쟁": 0.8,
    "공격": 1.0,
    "봉쇄": 1.3,
    "호르무즈": 0.5,
    "제재": 0.7,
    "공급 차질": 1.0,
}

DEESCALATION_WORDS = {
    "ceasefire": 1.3,
    "deal": 0.9,
    "diplomacy": 0.7,
    "peace": 1.0,
    "proposal": 0.5,
    "reopen": 1.1,
    "reopening": 1.1,
    "sanctions relief": 0.9,
    "talks": 0.4,
    "휴전": 1.3,
    "평화": 1.0,
    "협상": 0.5,
    "재개": 0.8,
}


def _fetch_gdelt_articles(max_records: int = 75, timespan: str = "3d") -> list[dict]:
    params = {
        "query": NEWS_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "datedesc",
        "timespan": timespan,
    }
    request = Request(f"{GDELT_DOC_URL}?{urlencode(params)}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    return payload.get("articles", [])


def _fetch_google_news_articles() -> list[dict]:
    params = {
        "q": f"{NEWS_QUERY} when:3d",
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    request = Request(f"{GOOGLE_NEWS_RSS_URL}?{urlencode(params)}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=25) as response:
        root = ET.fromstring(response.read())
    rows = []
    for item in root.findall(".//item"):
        rows.append(
            {
                "title": item.findtext("title", default=""),
                "seendate": item.findtext("pubDate", default=""),
                "domain": "news.google.com",
                "sourcecountry": "",
                "url": item.findtext("link", default=""),
                "source": "Google News RSS",
            }
        )
    return rows


def _load_aihub_news() -> list[dict]:
    path = os.getenv("AIHUB_NEWS_CSV_PATH")
    if not path:
        return []
    df = pd.read_csv(path)
    title_col = next((col for col in df.columns if str(col).lower() in {"title", "제목", "headline"}), None)
    date_col = next((col for col in df.columns if str(col).lower() in {"date", "일자", "published_at", "pubdate"}), None)
    url_col = next((col for col in df.columns if str(col).lower() in {"url", "link"}), None)
    if title_col is None:
        raise ValueError("AIHUB_NEWS_CSV_PATH 파일에는 title/제목/headline 컬럼이 필요합니다.")
    rows = []
    for _, row in df.tail(500).iterrows():
        title = str(row[title_col])
        lowered = title.lower()
        if not any(word in lowered for word in ["iran", "hormuz", "tehran", "oil", "crude", "이란", "호르무즈", "원유", "유가"]):
            continue
        rows.append(
            {
                "title": title,
                "seendate": str(row[date_col]) if date_col else "",
                "domain": "AI Hub",
                "sourcecountry": "KR",
                "url": str(row[url_col]) if url_col else "",
                "source": "AI Hub news CSV",
            }
        )
    return rows


def _keyword_score(title: str, keywords: dict[str, float]) -> float:
    lowered = title.lower()
    return sum(weight for word, weight in keywords.items() if word in lowered)


def score_news_articles(articles: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for article in articles:
        title = article.get("title", "")
        escalation = _keyword_score(title, ESCALATION_WORDS)
        deescalation = _keyword_score(title, DEESCALATION_WORDS)
        rows.append(
            {
                "seen_date": article.get("seendate"),
                "title": title,
                "domain": article.get("domain"),
                "source_country": article.get("sourcecountry"),
                "source": article.get("source", "GDELT" if article.get("domain") else ""),
                "url": article.get("url"),
                "escalation_score": escalation,
                "deescalation_score": deescalation,
                "net_score": escalation - deescalation,
            }
        )

    article_df = pd.DataFrame(rows)
    if article_df.empty:
        signal = {
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "article_count": 0,
            "escalation_total": 0.0,
            "deescalation_total": 0.0,
            "net_total": 0.0,
            "news_risk_score": 0.0,
            "forecast_adjustment_pct": 0.0,
        }
        return article_df, signal

    escalation_total = float(article_df["escalation_score"].sum())
    deescalation_total = float(article_df["deescalation_score"].sum())
    volume_boost = math.log1p(len(article_df)) / math.log1p(75)
    net_total = escalation_total - deescalation_total
    news_risk_score = max(-1.0, min(1.0, (net_total / 12.0) * volume_boost))
    forecast_adjustment_pct = news_risk_score * 0.035

    signal = {
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "article_count": int(len(article_df)),
        "escalation_total": round(escalation_total, 4),
        "deescalation_total": round(deescalation_total, 4),
        "net_total": round(net_total, 4),
        "news_risk_score": round(news_risk_score, 4),
        "forecast_adjustment_pct": round(forecast_adjustment_pct, 6),
    }
    return article_df, signal


def download_news_signal() -> dict:
    print("뉴스/이벤트 분석: AI Hub 뉴스 + Google News 크롤링")
    articles = []
    aihub_articles = _load_aihub_news()
    google_articles = _fetch_google_news_articles()
    articles.extend(aihub_articles)
    articles.extend(google_articles)
    if not articles:
        articles = _fetch_gdelt_articles()
    article_df, signal = score_news_articles(articles)
    signal["aihub_article_count"] = len(aihub_articles)
    signal["google_news_article_count"] = len(google_articles)
    article_df.to_csv(PATHS.news_articles, index=False)
    pd.DataFrame([signal]).to_csv(PATHS.news_signal, index=False)
    return signal


def load_or_download_news_signal() -> dict:
    try:
        return download_news_signal()
    except Exception as exc:
        if PATHS.news_signal.exists():
            print(f"뉴스 다운로드 실패, 마지막 뉴스 신호 사용: {exc}")
            return pd.read_csv(PATHS.news_signal).iloc[-1].to_dict()
        print(f"뉴스 다운로드 실패, 뉴스 보정 없이 진행: {exc}")
        return {
            "article_count": 0,
            "news_risk_score": 0.0,
            "forecast_adjustment_pct": 0.0,
        }
