import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_METAL_LOGGING", "0")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "matplotlib_cache"))


@dataclass
class Paths:
    online_raw: Path = OUT_DIR / "online_oil_dataset.csv"
    online_meta: Path = OUT_DIR / "online_dataset_meta.csv"
    news_signal: Path = OUT_DIR / "news_signal.csv"
    news_articles: Path = OUT_DIR / "news_articles.csv"
    source_audit: Path = OUT_DIR / "data_source_audit.csv"
    raw: Path = OUT_DIR / "raw_oil_project.csv"
    processed: Path = OUT_DIR / "processed_oil_project.csv"
    scaler: Path = MODEL_DIR / "oil_project_scaler.pkl"
    model: Path = MODEL_DIR / "oil_project_lstm.keras"
    summary: Path = OUT_DIR / "analysis_summary.csv"
    forecast: Path = OUT_DIR / "seven_day_forecast.csv"
    forecast_history: Path = OUT_DIR / "forecast_history.csv"
    event_windows: Path = OUT_DIR / "event_window_summary.csv"
    event_risk_scores: Path = OUT_DIR / "event_risk_scores.csv"
    metrics: Path = OUT_DIR / "model_metrics.csv"
    selected_model: Path = OUT_DIR / "selected_model.txt"
    figures: Path = OUT_DIR / "figures"
    refresh_state: Path = OUT_DIR / "refresh_state.csv"


PATHS = Paths()

EVENTS = [
    (
        "2008-09-15",
        "글로벌 금융위기",
        {"severity": 5, "oil_supply": 2, "region": 3, "duration": 5},
    ),
    (
        "2020-03-11",
        "코로나19 팬데믹",
        {"severity": 5, "oil_supply": 3, "region": 5, "duration": 5},
    ),
    (
        "2022-02-24",
        "러시아-우크라이나 전쟁",
        {"severity": 5, "oil_supply": 5, "region": 4, "duration": 5},
    ),
    (
        "2023-10-07",
        "중동 분쟁",
        {"severity": 4, "oil_supply": 4, "region": 5, "duration": 4},
    ),
    (
        "2026-06-05",
        "이란-미국 긴장 및 호르무즈 봉쇄 리스크",
        {"severity": 5, "oil_supply": 5, "region": 5, "duration": 4},
    ),
]


def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PATHS.figures.mkdir(exist_ok=True)
    (OUT_DIR / "matplotlib_cache").mkdir(exist_ok=True)
