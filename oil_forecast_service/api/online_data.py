import json
import os
import re
import time
from io import BytesIO
from io import StringIO
from datetime import datetime
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np

from config import PATHS


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
INVESTING_WTI_URL = "https://www.investing.com/commodities/crude-oil-historical-data"
INVESTING_BRENT_URL = "https://www.investing.com/commodities/brent-oil-historical-data"
OPINET_AVG_URL = "https://www.opinet.co.kr/api/avgAllPrice.do"
OPINET_HISTORY_CSV_URL = "https://www.opinet.co.kr/user/dopospdrg/dopOsPdrgCsv.do"
TODAY_OIL_URL = "https://oil.achveons.com/api/today"
TODAY = pd.Timestamp.today().normalize()
EXCHANGE_MIN_KRW = 500
EXCHANGE_MAX_KRW = 2500
FUEL_PRODUCTS = {
    "gasoline": {"code": "B027", "name": "휘발유", "column": "gasoline_price"},
    "diesel": {"code": "D047", "name": "경유", "column": "diesel_price"},
    "lpg": {"code": "K015", "name": "LPG", "column": "lpg_price"},
}
FUEL_FALLBACK_RATIOS = {"diesel_price": 0.92, "lpg_price": 0.54}


def _load_local_env():
    env_path = PATHS.online_raw.parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()


def _download_fred_series(series_id: str, column_name: str) -> pd.DataFrame:
    url = FRED_URL.format(series_id=series_id)
    df = pd.read_csv(url)
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise ValueError(f"FRED 응답 형식이 예상과 다릅니다: {series_id}")

    df = df.rename(columns={"observation_date": "Date", series_id: column_name})
    df["Date"] = pd.to_datetime(df["Date"])
    df[column_name] = pd.to_numeric(df[column_name].replace(".", pd.NA), errors="coerce")
    return df.set_index("Date")[[column_name]].dropna()


def _download_yahoo_chart(symbol: str, output_col: str, range_text: str = "max", interval: str = "1d") -> pd.DataFrame:
    params = urlencode(
        {
            "range": range_text,
            "interval": interval,
            "includePrePost": "false",
            "events": "history",
        }
    )
    url = f"{YAHOO_CHART_URL.format(symbol=quote(symbol, safe=''))}?{params}"
    payload = json.loads(_fetch_url_text(url))
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo Finance {symbol} 오류: {error}")
    result = (chart.get("result") or [None])[0]
    if not result or "timestamp" not in result:
        raise ValueError(f"Yahoo Finance {symbol} 응답에 timestamp가 없습니다.")

    timestamps = result["timestamp"]
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote_data.get("close")
    if not closes:
        raise ValueError(f"Yahoo Finance {symbol} 응답에 close 가격이 없습니다.")

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True)
            .tz_convert("Asia/Seoul")
            .tz_localize(None)
            .normalize(),
            output_col: closes,
        }
    )
    df[output_col] = pd.to_numeric(df[output_col], errors="coerce")
    df = df.dropna().groupby("Date").last().sort_index()
    df.index.name = "Date"
    return df[[output_col]]


def _download_yahoo_latest(symbol: str, output_col: str) -> pd.DataFrame:
    params = urlencode({"range": "5d", "interval": "1h", "includePrePost": "false"})
    url = f"{YAHOO_CHART_URL.format(symbol=quote(symbol, safe=''))}?{params}"
    payload = json.loads(_fetch_url_text(url))
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo Finance latest {symbol} 오류: {error}")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise ValueError(f"Yahoo Finance latest {symbol} 응답이 비어 있습니다.")

    meta = result.get("meta", {})
    price = meta.get("regularMarketPrice")
    market_time = meta.get("regularMarketTime")
    if price is None:
        quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
        closes = pd.Series(quote_data.get("close", []), dtype="float64").dropna()
        if closes.empty:
            raise ValueError(f"Yahoo Finance latest {symbol} 가격이 없습니다.")
        price = float(closes.iloc[-1])
        timestamps = result.get("timestamp", [])
        market_time = timestamps[-1] if timestamps else None

    date = (
        pd.to_datetime(market_time, unit="s", utc=True).tz_convert("Asia/Seoul").tz_localize(None).normalize()
        if market_time
        else TODAY
    )
    return pd.DataFrame({output_col: [float(price)]}, index=pd.DatetimeIndex([date], name="Date"))


def _clean_exchange_rates(exchange: pd.DataFrame) -> pd.DataFrame:
    exchange = exchange.copy()
    invalid = exchange["exchange"].lt(EXCHANGE_MIN_KRW) | exchange["exchange"].gt(EXCHANGE_MAX_KRW)
    exchange.loc[invalid, "exchange"] = pd.NA
    exchange["exchange"] = exchange["exchange"].ffill().bfill()
    return exchange.dropna()


def _normalize_price_frame(df: pd.DataFrame, date_col: str, price_col: str, output_col: str) -> pd.DataFrame:
    result = df[[date_col, price_col]].copy()
    result = result.rename(columns={date_col: "Date", price_col: output_col})
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    result[output_col] = (
        result[output_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    result = result.dropna().set_index("Date").sort_index()
    result.index.name = "Date"
    return result[[output_col]]


def _pick_column(columns, names):
    lowered = {str(col).strip().lower(): col for col in columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    for col in columns:
        col_text = str(col).strip().lower()
        if any(name.lower() in col_text for name in names):
            return col
    return None


def _load_kaggle_oil_csv() -> pd.DataFrame | None:
    path = os.getenv("KAGGLE_OIL_CSV_PATH")
    if not path:
        return None
    source = pd.read_csv(path)
    date_col = _pick_column(source.columns, ["date", "Date"])
    wti_col = _pick_column(source.columns, ["wti", "WTI", "WTI Close", "Crude Oil WTI"])
    brent_col = _pick_column(source.columns, ["brent", "Brent", "Brent Close", "Brent Oil"])
    if date_col is None or wti_col is None or brent_col is None:
        raise ValueError("KAGGLE_OIL_CSV_PATH 파일에는 Date, WTI, Brent 컬럼이 필요합니다.")
    wti = _normalize_price_frame(source, date_col, wti_col, "wti")
    brent = _normalize_price_frame(source, date_col, brent_col, "brent")
    return wti.join(brent, how="outer")


def _download_alpha_vantage_commodity(function_name: str, output_col: str) -> pd.DataFrame:
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not apikey:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY가 없습니다.")
    query = urlencode(
        {
            "function": function_name,
            "interval": "daily",
            "datatype": "csv",
            "apikey": apikey,
        }
    )
    url = f"{ALPHA_VANTAGE_URL}?{query}"
    last_response = ""
    for attempt in range(3):
        text = _fetch_url_text(url)
        last_response = text
        if "Thank you for using Alpha Vantage" in text or "rate limit" in text.lower():
            wait_seconds = 20 * (attempt + 1)
            print(f"Alpha Vantage {function_name} 속도 제한 감지, {wait_seconds}초 후 재시도")
            time.sleep(wait_seconds)
            continue
        df = pd.read_csv(StringIO(text))
        if "timestamp" in df.columns and "value" in df.columns:
            return _normalize_price_frame(df, "timestamp", "value", output_col)
        raise ValueError(f"Alpha Vantage {function_name} 응답 형식이 예상과 다릅니다: {text[:120]}")
    raise RuntimeError(f"Alpha Vantage {function_name} 속도 제한으로 실패했습니다: {last_response[:120]}")


def _download_alpha_vantage_oil() -> pd.DataFrame:
    wti = _download_alpha_vantage_commodity("WTI", "wti")
    brent = _download_alpha_vantage_commodity("BRENT", "brent")
    return wti.join(brent, how="outer")


def _download_yahoo_oil_history() -> pd.DataFrame:
    wti = _download_yahoo_chart("CL=F", "wti")
    brent = _download_yahoo_chart("BZ=F", "brent")
    return wti.join(brent, how="outer")


def _download_yahoo_oil_latest() -> pd.DataFrame:
    wti = _download_yahoo_latest("CL=F", "wti")
    brent = _download_yahoo_latest("BZ=F", "brent")
    return wti.join(brent, how="outer")


def download_exchange_rates() -> tuple[pd.DataFrame, str]:
    source_notes = []
    exchange = pd.DataFrame()

    try:
        yahoo = _download_yahoo_chart("KRW=X", "exchange")
        exchange = yahoo
        source_notes.append("Yahoo Finance KRW=X daily")
    except Exception as exc:
        source_notes.append(f"Yahoo Finance KRW=X history unavailable: {exc}")

    try:
        fred = _download_fred_series("DEXKOUS", "exchange")
        exchange = exchange.combine_first(fred) if not exchange.empty else fred
        source_notes.append("FRED DEXKOUS")
    except Exception as exc:
        source_notes.append(f"FRED DEXKOUS unavailable: {exc}")

    try:
        yahoo_latest = _download_yahoo_latest("KRW=X", "exchange")
        exchange = exchange.combine_first(yahoo_latest) if not exchange.empty else yahoo_latest
        exchange.update(yahoo_latest)
        source_notes.append("Yahoo Finance KRW=X latest")
    except Exception as exc:
        source_notes.append(f"Yahoo Finance KRW=X latest unavailable: {exc}")

    if exchange.empty:
        raise RuntimeError("원/달러 환율 데이터를 가져오지 못했습니다.")
    return _clean_exchange_rates(exchange.sort_index()), " + ".join(source_notes)


def _read_investing_table(url: str, output_col: str) -> pd.DataFrame:
    text = _fetch_url_text(url)
    tables = pd.read_html(StringIO(text))
    for table in tables:
        date_col = _pick_column(table.columns, ["Date"])
        price_col = _pick_column(table.columns, ["Price", "Close", "Last"])
        if date_col is not None and price_col is not None:
            return _normalize_price_frame(table, date_col, price_col, output_col)
    raise ValueError(f"Investing.com 테이블에서 {output_col} 가격 컬럼을 찾지 못했습니다.")


def download_international_oil_prices() -> tuple[pd.DataFrame, str]:
    source_notes = []
    oil = pd.DataFrame()

    try:
        yahoo = _download_yahoo_oil_history()
        oil = yahoo
        source_notes.append("Yahoo Finance CL=F/BZ=F daily")
    except Exception as exc:
        source_notes.append(f"Yahoo Finance oil history unavailable: {exc}")

    try:
        alpha = _download_alpha_vantage_oil()
        oil = oil.combine_first(alpha) if not oil.empty else alpha
        source_notes.append("Alpha Vantage API WTI/BRENT")
    except Exception as exc:
        source_notes.append(f"Alpha Vantage unavailable: {exc}")

    kaggle = _load_kaggle_oil_csv()
    if kaggle is not None:
        oil = oil.combine_first(kaggle) if not oil.empty else kaggle
        source_notes.append(f"Kaggle CSV: {os.getenv('KAGGLE_OIL_CSV_PATH')}")

    try:
        wti_recent = _read_investing_table(INVESTING_WTI_URL, "wti")
        brent_recent = _read_investing_table(INVESTING_BRENT_URL, "brent")
        investing = wti_recent.join(brent_recent, how="outer")
        oil = oil.combine_first(investing) if not oil.empty else investing
        source_notes.append("Investing.com historical tables")
    except Exception as exc:
        source_notes.append(f"Investing.com unavailable: {exc}")

    try:
        yahoo_latest = _download_yahoo_oil_latest()
        oil = oil.combine_first(yahoo_latest) if not oil.empty else yahoo_latest
        oil.update(yahoo_latest)
        source_notes.append("Yahoo Finance CL=F/BZ=F latest")
    except Exception as exc:
        source_notes.append(f"Yahoo Finance oil latest unavailable: {exc}")

    if oil.empty or not {"wti", "brent"}.issubset(oil.columns):
        raise RuntimeError(
            "국제 유가 데이터가 부족합니다. ALPHA_VANTAGE_API_KEY를 설정하거나 "
            "KAGGLE_OIL_CSV_PATH에 Date/WTI/Brent 컬럼이 있는 Kaggle CSV를 지정하세요. "
            "Investing.com은 공식 API가 아니라 접속 차단/403이 발생할 수 있습니다."
        )
    return oil.sort_index(), " + ".join(source_notes)


def _fetch_url_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", "replace")


def _post_url_bytes(url: str, data: dict) -> bytes:
    body = urlencode(data).encode()
    request = Request(
        url,
        data=body,
        headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def _fuel_column_from_item(item: dict) -> str | None:
    prod_code = str(item.get("PRODCD", "")).strip()
    prod_name = str(item.get("PRODNM", "")).strip()
    for product in FUEL_PRODUCTS.values():
        if prod_code == product["code"] or prod_name == product["name"]:
            return product["column"]
    return None


def _with_fuel_fallbacks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "gasoline_price" not in frame.columns and "domestic_price" in frame.columns:
        frame["gasoline_price"] = frame["domestic_price"]
    if "gasoline_price" not in frame.columns:
        return frame
    frame["gasoline_price"] = pd.to_numeric(frame["gasoline_price"], errors="coerce")
    for column, ratio in FUEL_FALLBACK_RATIOS.items():
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame["gasoline_price"] * ratio)
    frame["domestic_price"] = frame["gasoline_price"]
    return frame


def _today_prices_from_opinet_api() -> dict | None:
    certkey = os.getenv("OPINET_AVG_API_KEY") or os.getenv("OPINET_API_KEY")
    if not certkey:
        return None

    payload = None
    for key_name in ("certkey", "code"):
        query = urlencode({"out": "json", key_name: certkey})
        text = _fetch_url_text(f"{OPINET_AVG_URL}?{query}")
        payload = json.loads(text)
        oils = payload.get("RESULT", {}).get("OIL", [])
        if oils:
            break
    oils = payload.get("RESULT", {}).get("OIL", [])
    if isinstance(oils, dict):
        oils = [oils]

    result = {
        "date": None,
        "prices": {},
        "source": OPINET_AVG_URL,
        "source_name": "OPINET official avgAllPrice API",
    }
    for item in oils:
        column = _fuel_column_from_item(item)
        if not column:
            continue
        result["date"] = pd.to_datetime(str(item["TRADE_DT"]), format="%Y%m%d")
        result["prices"][column] = float(str(item["PRICE"]).replace(",", ""))
    if "gasoline_price" not in result["prices"]:
        raise ValueError("OPINET API 응답에서 휘발유(B027) 평균가격을 찾지 못했습니다.")
    return result


def _today_prices_from_public_opinet_page() -> dict:
    text = _fetch_url_text(TODAY_OIL_URL)
    meta_match = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*기준\s*전국\s*휘발유\s*평균\s*([\d,]+)원", text)
    title_match = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*전국\s*주유소\s*휘발유\s*평균가\s*([\d,]+)원", text)
    match = meta_match or title_match
    if not match:
        raise ValueError("오늘 유종별 유가 기준 페이지에서 전국 휘발유 평균가를 찾지 못했습니다.")

    year, month, day, price = match.groups()
    return {
        "date": pd.Timestamp(year=int(year), month=int(month), day=int(day)),
        "prices": {"gasoline_price": float(price.replace(",", ""))},
        "source": TODAY_OIL_URL,
        "source_name": "Opinet-linked public today oil price page",
    }


def download_today_domestic_prices() -> dict:
    result = _today_prices_from_opinet_api() or _today_prices_from_public_opinet_page()
    prices = pd.DataFrame([result["prices"]])
    prices = _with_fuel_fallbacks(prices).iloc[0].to_dict()
    result["prices"] = {
        key: float(value)
        for key, value in prices.items()
        if key in {"gasoline_price", "domestic_price", "diesel_price", "lpg_price"}
    }
    return result


def _align_today_fuel_prices_with_history(domestic: pd.DataFrame, today_prices: dict) -> dict:
    prices = dict(today_prices)
    gasoline = float(prices.get("gasoline_price", prices.get("domestic_price", np.nan)))
    if not np.isfinite(gasoline) or gasoline <= 0 or domestic.empty:
        return prices

    history = _with_fuel_fallbacks(domestic.copy()).sort_index().tail(30)
    for column, fallback_ratio in FUEL_FALLBACK_RATIOS.items():
        value = float(prices.get(column, np.nan))
        ratio_history = (history[column] / history["gasoline_price"]).replace([np.inf, -np.inf], np.nan).dropna()
        expected_ratio = float(ratio_history.median()) if not ratio_history.empty else fallback_ratio
        current_ratio = value / gasoline if np.isfinite(value) and gasoline else np.nan
        expected_value = gasoline * expected_ratio
        absolute_gap = abs(value - expected_value) if np.isfinite(value) else np.inf
        max_gap = max(8.0, gasoline * 0.004)
        if not np.isfinite(current_ratio) or abs(current_ratio / expected_ratio - 1) > 0.06 or absolute_gap > max_gap:
            prices[column] = gasoline * expected_ratio

    prices["domestic_price"] = gasoline
    return prices


def download_opinet_daily_history(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    params = {
        "all_chk_cnt": "5",
        "INIF_FLAG": "N",
        "chk_cnt": "1",
        "h_maxYY": str(TODAY.year - 1),
        "h_maxQQ": f"{TODAY.year}1",
        "h_maxMM": f"{TODAY.year}{TODAY.month - 1:02d}",
        "h_maxDD": (TODAY - pd.Timedelta(days=1)).strftime("%Y%m%d"),
        "h_maxWW": f"{TODAY.year}{TODAY.month:02d}1",
        "TERM": "D",
        "STA_Y": str(start_date.year),
        "STA_M": f"{start_date.month:02d}",
        "STA_D": f"{start_date.day:02d}",
        "END_Y": str(end_date.year),
        "END_M": f"{end_date.month:02d}",
        "END_D": f"{end_date.day:02d}",
        "OIL_CD_B027": "Y",
        "OIL_CD_D047": "Y",
        "OIL_CD_K015": "Y",
        "equal": "Y",
    }
    body = _post_url_bytes(OPINET_HISTORY_CSV_URL, params)
    history = pd.read_csv(BytesIO(body), encoding="cp949")
    history = history.rename(columns={history.columns[0]: "Date"})
    rename_map = {}
    fallback_columns = [column for column in history.columns if column != "Date"]
    for product in FUEL_PRODUCTS.values():
        matched = None
        for column in history.columns:
            if product["name"] in str(column) or product["code"] in str(column):
                matched = column
                break
        if matched is None and product["column"] == "gasoline_price" and fallback_columns:
            matched = fallback_columns.pop(0)
        if matched is not None:
            rename_map[matched] = product["column"]
    history = history.rename(columns=rename_map)
    history["Date"] = history["Date"].astype(str).str.extract(r"(\d{4})년(\d{2})월(\d{2})일").agg("-".join, axis=1)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    if "gasoline_price" not in history.columns:
        candidate_columns = [column for column in history.columns if column != "Date"]
        if not candidate_columns:
            raise ValueError("OPINET 이력 CSV에서 유가 컬럼을 찾지 못했습니다.")
        history["gasoline_price"] = history[candidate_columns[0]]
    history = history.drop(columns=[column for column in ("diesel_price", "lpg_price") if column in history.columns])
    fuel_columns = [product["column"] for product in FUEL_PRODUCTS.values() if product["column"] in history.columns]
    for column in fuel_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = _with_fuel_fallbacks(history)
    history = history.dropna(subset=["Date", "gasoline_price"]).set_index("Date").sort_index()
    history.index.name = "Date"
    return history[["gasoline_price", "diesel_price", "lpg_price", "domestic_price"]]


def download_latest_dataset() -> pd.DataFrame:
    oil_prices, oil_source = download_international_oil_prices()
    exchange, exchange_source = download_exchange_rates()
    today_domestic = download_today_domestic_prices()

    start_date = pd.Timestamp("2008-04-15")
    history_end = TODAY - pd.Timedelta(days=1)
    domestic = download_opinet_daily_history(start_date, history_end)
    today_domestic["prices"] = _align_today_fuel_prices_with_history(domestic, today_domestic["prices"])
    if today_domestic["date"] >= domestic.index.max():
        for column, price in today_domestic["prices"].items():
            domestic.loc[today_domestic["date"], column] = price
    actual_domestic_date = domestic.index.max()
    actual_gasoline_price = float(domestic.loc[actual_domestic_date, "gasoline_price"])
    actual_domestic_price = actual_gasoline_price
    actual_diesel_price = float(domestic.loc[actual_domestic_date, "diesel_price"])
    actual_lpg_price = float(domestic.loc[actual_domestic_date, "lpg_price"])

    df = domestic.join(oil_prices, how="left").join(exchange, how="left").sort_index()
    daily_index = pd.date_range(df.index.min(), TODAY, freq="D")
    df = df.reindex(daily_index)
    df.index.name = "Date"
    df = _with_fuel_fallbacks(df)
    df[["gasoline_price", "diesel_price", "lpg_price", "domestic_price"]] = df[
        ["gasoline_price", "diesel_price", "lpg_price", "domestic_price"]
    ].ffill()
    df[["wti", "brent", "exchange"]] = df[["wti", "brent", "exchange"]].ffill().bfill()

    df.to_csv(PATHS.online_raw)

    meta = pd.DataFrame(
        [
            {"key": "downloaded_at", "value": datetime.now().isoformat(timespec="seconds")},
            {"key": "calendar_extended_to", "value": TODAY.date().isoformat()},
            {"key": "domestic_trade_date", "value": actual_domestic_date.date().isoformat()},
            {"key": "gasoline_price", "value": str(actual_gasoline_price)},
            {"key": "domestic_price", "value": str(actual_domestic_price)},
            {"key": "diesel_price", "value": str(actual_diesel_price)},
            {"key": "lpg_price", "value": str(actual_lpg_price)},
            {"key": "domestic_source", "value": today_domestic["source"]},
            {"key": "domestic_source_name", "value": today_domestic["source_name"]},
            {"key": "domestic_history_source", "value": "OPINET Open API current + OPINET official CSV history"},
            {"key": "international_oil_source", "value": oil_source},
            {"key": "exchange_source", "value": exchange_source},
            {"key": "rows", "value": str(len(df))},
        ]
    )
    meta.to_csv(PATHS.online_meta, index=False)
    meta.to_csv(PATHS.source_audit, index=False)
    return df


def load_latest_online_dataset() -> pd.DataFrame:
    try:
        print("온라인 데이터 다운로드: WTI/Brent(Yahoo 실시간 + Alpha/Kaggle) + Opinet + 원달러 환율(Yahoo/FRED)")
        return download_latest_dataset()
    except (URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"온라인 데이터 다운로드에 실패했습니다: {exc}") from exc
