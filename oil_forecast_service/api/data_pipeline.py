import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import EVENTS, PATHS
from news_signal import load_or_download_news_signal
from online_data import load_latest_online_dataset


def load_base_data() -> pd.DataFrame:
    df = load_latest_online_dataset()
    df = df.sort_index().ffill().bfill()
    if "gasoline_price" not in df.columns and "domestic_price" in df.columns:
        df["gasoline_price"] = df["domestic_price"]
    if "domestic_price" not in df.columns and "gasoline_price" in df.columns:
        df["domestic_price"] = df["gasoline_price"]
    required = ["wti", "brent", "exchange", "domestic_price", "gasoline_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    if "diesel_price" not in df.columns:
        df["diesel_price"] = df["gasoline_price"] * 0.92
    if "lpg_price" not in df.columns:
        df["lpg_price"] = df["gasoline_price"] * 0.54
    return df[required + ["diesel_price", "lpg_price"]]



def _score_1_to_5(value: float | int | None, default: float = 3.0) -> float:
    if value is None:
        value = default
    value = float(value)
    value = min(5.0, max(1.0, value))
    return (value - 1.0) / 4.0


def calculate_event_weight(event_name: str, meta: dict | None, news_risk_score: float | None = None) -> tuple[float, dict[str, float]]:
    meta = meta or {}
    severity_score = _score_1_to_5(meta.get("severity"))
    oil_supply_score = _score_1_to_5(meta.get("oil_supply"))
    region_score = _score_1_to_5(meta.get("region"))
    duration_score = _score_1_to_5(meta.get("duration"))

    base_score = (
        0.35 * severity_score
        + 0.30 * oil_supply_score
        + 0.20 * region_score
        + 0.15 * duration_score
    )

    keyword_bonus = 0.0
    lowered = str(event_name).lower()
    oil_keywords = ["호르무즈", "봉쇄", "중동", "전쟁", "러시아", "우크라이나", "iran", "hormuz", "oil", "war"]
    matched_keyword_count = sum(1 for keyword in oil_keywords if keyword in lowered)
    keyword_bonus = min(0.08, matched_keyword_count * 0.02)

    calculated = min(1.0, base_score + keyword_bonus)

    news_component = None
    if news_risk_score is not None:
        news_component = float(min(1.0, max(0.0, news_risk_score)))
        calculated = 0.75 * calculated + 0.25 * news_component

    details = {
        "severity_score": severity_score,
        "oil_supply_score": oil_supply_score,
        "region_score": region_score,
        "duration_score": duration_score,
        "base_score": base_score,
        "keyword_bonus": keyword_bonus,
        "news_component": np.nan if news_component is None else news_component,
        "calculated_weight": calculated,
    }
    return float(min(1.0, max(0.0, calculated))), details


def build_risk_index(index: pd.DatetimeIndex, news_risk_score: float | None = None) -> pd.Series:
    risk = pd.Series(0.0, index=index, name="risk_index")
    rows = []
    latest_date = index.max().normalize() if len(index) else pd.Timestamp.today().normalize()

    for date_text, event_name, meta in EVENTS:
        center = pd.Timestamp(date_text)
        is_recent_event = abs((latest_date - center.normalize()).days) <= 45
        event_news_score = news_risk_score if is_recent_event else None
        weight, detail = calculate_event_weight(event_name, meta, event_news_score)

        days = (index - center).days.astype(float)
        event_curve = weight * np.exp(-0.5 * (days / 14.0) ** 2)
        risk = np.maximum(risk, event_curve)

        rows.append(
            {
                "date": center.date().isoformat(),
                "event": event_name,
                "severity": (meta or {}).get("severity"),
                "oil_supply": (meta or {}).get("oil_supply"),
                "region": (meta or {}).get("region"),
                "duration": (meta or {}).get("duration"),
                **detail,
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(PATHS.event_risk_scores, index=False)

    return pd.Series(risk, index=index, name="risk_index").clip(0, 1)


def collect_and_preprocess() -> pd.DataFrame:
    print("\n[1] 데이터 수집/전처리")
    df = load_base_data()
    news_signal = load_or_download_news_signal()
    df["risk_index"] = build_risk_index(df.index, news_risk_score=float(news_signal.get("news_risk_score", 0.0)))
    df["news_risk_index"] = 0.0
    df.loc[df.index >= df.index.max() - pd.Timedelta(days=14), "news_risk_index"] = float(
        news_signal.get("news_risk_score", 0.0)
    )
    df["domestic_return"] = df["domestic_price"].pct_change().fillna(0)
    df["volatility_7d"] = df["domestic_return"].rolling(7, min_periods=1).std().fillna(0)
    df["domestic_ma7"] = df["domestic_price"].rolling(7, min_periods=1).mean()
    df["domestic_ma30"] = df["domestic_price"].rolling(30, min_periods=1).mean()
    df = df.ffill().bfill()
    df.to_csv(PATHS.raw)

    scaler = MinMaxScaler()
    feature_cols = [
        "wti",
        "brent",
        "exchange",
        "domestic_price",
        "gasoline_price",
        "diesel_price",
        "lpg_price",
        "risk_index",
        "news_risk_index",
        "volatility_7d",
    ]
    scaled = scaler.fit_transform(df[feature_cols])
    processed = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    processed.to_csv(PATHS.processed)
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, PATHS.scaler)
    print(f"저장: {PATHS.raw.name}, {PATHS.processed.name}")
    return df
