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
    required = ["wti", "brent", "exchange", "domestic_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    return df[required]


def build_risk_index(index: pd.DatetimeIndex) -> pd.Series:
    risk = pd.Series(0.0, index=index, name="risk_index")
    for date_text, _name, weight in EVENTS:
        center = pd.Timestamp(date_text)
        days = (index - center).days.astype(float)
        event_curve = weight * np.exp(-0.5 * (days / 14.0) ** 2)
        risk = np.maximum(risk, event_curve)
    return pd.Series(risk, index=index, name="risk_index").clip(0, 1)


def collect_and_preprocess() -> pd.DataFrame:
    print("\n[1] 데이터 수집/전처리")
    df = load_base_data()
    news_signal = load_or_download_news_signal()
    df["risk_index"] = build_risk_index(df.index)
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
    feature_cols = ["wti", "brent", "exchange", "domestic_price", "risk_index", "news_risk_index", "volatility_7d"]
    scaled = scaler.fit_transform(df[feature_cols])
    processed = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    processed.to_csv(PATHS.processed)
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, PATHS.scaler)
    print(f"저장: {PATHS.raw.name}, {PATHS.processed.name}")
    return df
