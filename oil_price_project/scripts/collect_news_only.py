from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import time
import xml.etree.ElementTree as ET

import pandas as pd


# Project folder location changed to Desktop.
BASE_DIR = Path(r"C:\Users\ksh62\OneDrive\Desktop\oil_price_project")
DATA_DIR = BASE_DIR / "data"
RAW_NEWS_DIR = DATA_DIR / "raw" / "news"
RELEVANT_DIR = RAW_NEWS_DIR / "relevant"
IRRELEVANT_DIR = RAW_NEWS_DIR / "irrelevant"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = DATA_DIR / "logs"

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
MAX_ARTICLES_PER_QUERY = 100
SLEEP_SECONDS = 1.5


EVENTS = [
    {
        "event_name": "global_financial_crisis",
        "start": "2008-06-01",
        "end": "2009-06-30",
        "queries": [
            "financial crisis oil price",
            "global financial crisis crude oil",
            "oil price collapse 2008",
        ],
    },
    {
        "event_name": "covid19_pandemic",
        "start": "2020-01-01",
        "end": "2022-12-31",
        "queries": [
            "COVID oil price",
            "pandemic oil demand",
            "coronavirus crude oil",
            "COVID gasoline price",
        ],
    },
    {
        "event_name": "russia_ukraine_war",
        "start": "2021-11-01",
        "end": "2022-12-31",
        "queries": [
            "Russia Ukraine war oil price",
            "Ukraine war crude oil",
            "Russia sanctions oil",
            "Europe energy crisis oil",
        ],
    },
    {
        "event_name": "israel_hamas_middle_east_conflict",
        "start": "2023-07-01",
        "end": "2024-06-30",
        "queries": [
            "Israel Hamas oil price",
            "Middle East conflict oil",
            "Iran oil price",
            "Strait of Hormuz oil",
        ],
    },
    {
        "event_name": "recent_middle_east_oil_risk",
        "start": "2024-01-01",
        "end": datetime.today().strftime("%Y-%m-%d"),
        "queries": [
            "Middle East oil price",
            "Iran oil price",
            "Strait of Hormuz oil",
            "OPEC oil price",
            "Korea gasoline price",
        ],
    },
]


RELEVANT_KEYWORDS = [
    "oil",
    "crude",
    "wti",
    "brent",
    "opec",
    "gasoline",
    "fuel",
    "energy",
    "exchange rate",
    "korea",
    "war",
    "conflict",
    "russia",
    "ukraine",
    "middle east",
    "israel",
    "hamas",
    "iran",
    "hormuz",
    "covid",
    "pandemic",
    "financial crisis",
    "sanctions",
    "supply",
    "demand",
    "원유",
    "유가",
    "휘발유",
    "국제유가",
    "환율",
    "중동",
    "전쟁",
    "분쟁",
    "코로나",
    "감산",
]

IRRELEVANT_KEYWORDS = [
    "cooking oil",
    "olive oil",
    "essential oil",
    "hair oil",
    "skin oil",
    "beauty",
    "recipe",
    "sports",
    "movie",
    "celebrity",
]

RISK_WORDS = {
    "war": 3,
    "conflict": 3,
    "invasion": 3,
    "sanctions": 2,
    "opec cut": 3,
    "supply disruption": 3,
    "pandemic": 2,
    "demand collapse": 3,
    "middle east": 2,
    "iran": 2,
    "strait of hormuz": 3,
    "financial crisis": 2,
    "전쟁": 3,
    "분쟁": 3,
    "코로나": 2,
    "감산": 3,
    "공급 차질": 3,
    "환율 급등": 2,
    "국제유가 급등": 3,
}


def make_dirs():
    for path in [RAW_NEWS_DIR, RELEVANT_DIR, IRRELEVANT_DIR, PROCESSED_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{now}] {message}"
    print(text)
    with open(LOG_DIR / "news_collect_log.txt", "a", encoding="utf-8") as file:
        file.write(text + "\n")


def parse_pub_date(pub_date):
    try:
        return parsedate_to_datetime(pub_date).strftime("%Y-%m-%d")
    except Exception:
        return ""


def fetch_google_news(query, start_date, end_date):
    search_query = f"{query} after:{start_date} before:{end_date}"
    params = {
        "q": search_query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    url = GOOGLE_NEWS_RSS_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request, timeout=45) as response:
        xml_text = response.read().decode("utf-8", errors="replace")

    root = ET.fromstring(xml_text)
    items = root.findall(".//item")
    rows = []

    for item in items[:MAX_ARTICLES_PER_QUERY]:
        source_node = item.find("source")
        source_name = source_node.text if source_node is not None else "Google News"

        rows.append(
            {
                "date": parse_pub_date(item.findtext("pubDate", "")),
                "title": item.findtext("title", ""),
                "url": item.findtext("link", ""),
                "source_name": source_name,
                "language": "en",
                "keyword": query,
            }
        )

    return rows


def classify_relevance(title, keyword):
    text = f"{title} {keyword}".lower()

    for word in IRRELEVANT_KEYWORDS:
        if word in text:
            return "irrelevant"

    for word in RELEVANT_KEYWORDS:
        if word in text:
            return "relevant"

    return "irrelevant"


def calculate_risk_score(title, keyword):
    text = f"{title} {keyword}".lower()
    score = 0

    for word, point in RISK_WORDS.items():
        if word.lower() in text:
            score += point

    return score


def collect_news():
    all_rows = []

    for event in EVENTS:
        event_name = event["event_name"]
        start_date = event["start"]
        end_date = event["end"]
        log(f"event start: {event_name} ({start_date} ~ {end_date})")

        for query in event["queries"]:
            try:
                rows = fetch_google_news(query, start_date, end_date)
                log(f"query success: {event_name} / {query} / {len(rows)} articles")

                for row in rows:
                    row["event_name"] = event_name
                    row["relevance_label"] = classify_relevance(row["title"], row["keyword"])
                    row["risk_score"] = calculate_risk_score(row["title"], row["keyword"])
                    all_rows.append(row)

            except Exception as error:
                log(f"query fail: {event_name} / {query} / {error}")

            time.sleep(SLEEP_SECONDS)

    columns = [
        "date",
        "title",
        "url",
        "source_name",
        "language",
        "keyword",
        "event_name",
        "relevance_label",
        "risk_score",
    ]

    news = pd.DataFrame(all_rows, columns=columns)

    if len(news) > 0:
        news = news.drop_duplicates(subset=["url"])
        news = news.sort_values(["date", "event_name", "keyword"])

    return news


def safe_to_csv(dataframe, path):
    try:
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        dataframe.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        log(f"file was locked, saved with timestamp instead: {fallback_path}")
        return fallback_path


def clear_old_news_csv_files():
    for folder in [RAW_NEWS_DIR, RELEVANT_DIR, IRRELEVANT_DIR]:
        for path in folder.glob("*.csv"):
            try:
                path.unlink()
            except PermissionError:
                log(f"old file was locked, could not delete: {path}")


def save_news_files(news):
    clear_old_news_csv_files()

    safe_to_csv(news, RAW_NEWS_DIR / "news_collected_all.csv")
    safe_to_csv(news, RAW_NEWS_DIR / "news_classified_all.csv")

    relevant = news[news["relevance_label"] == "relevant"].copy()
    irrelevant = news[news["relevance_label"] == "irrelevant"].copy()

    safe_to_csv(relevant, RELEVANT_DIR / "news_relevant_all.csv")
    safe_to_csv(irrelevant, IRRELEVANT_DIR / "news_irrelevant_all.csv")

    for event_name, event_data in relevant.groupby("event_name"):
        safe_to_csv(event_data, RELEVANT_DIR / f"{event_name}_relevant.csv")

    for event_name, event_data in irrelevant.groupby("event_name"):
        safe_to_csv(event_data, IRRELEVANT_DIR / f"{event_name}_irrelevant.csv")

    log(f"saved total news rows: {len(news)}")
    log(f"saved relevant news rows: {len(relevant)}")
    log(f"saved irrelevant news rows: {len(irrelevant)}")


def save_news_risk_index(news):
    if len(news) == 0:
        risk_index = pd.DataFrame(
            columns=[
                "date",
                "article_count",
                "relevant_article_count",
                "average_risk_score",
                "max_risk_score",
            ]
        )
    else:
        news = news.copy()
        news["date"] = pd.to_datetime(news["date"], errors="coerce")
        news = news.dropna(subset=["date"])
        news["is_relevant"] = news["relevance_label"].eq("relevant").astype(int)

        risk_index = (
            news.groupby("date")
            .agg(
                article_count=("title", "count"),
                relevant_article_count=("is_relevant", "sum"),
                average_risk_score=("risk_score", "mean"),
                max_risk_score=("risk_score", "max"),
            )
            .reset_index()
        )
        risk_index["date"] = risk_index["date"].dt.strftime("%Y-%m-%d")

    safe_to_csv(risk_index, PROCESSED_DIR / "news_risk_index.csv")
    log(f"saved risk index rows: {len(risk_index)}")

    return risk_index


def main():
    make_dirs()
    log("news collection only pipeline start")
    log("source: Google News RSS historical search")

    news = collect_news()
    save_news_files(news)
    save_news_risk_index(news)

    log("news collection only pipeline finished")


if __name__ == "__main__":
    main()
