import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_METAL_LOGGING", "0")
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
    event_windows: Path = OUT_DIR / "event_window_summary.csv"
    metrics: Path = OUT_DIR / "model_metrics.csv"
    figures: Path = OUT_DIR / "figures"


PATHS = Paths()

EVENTS = [
    ("2008-09-15", "글로벌 금융위기", 0.90),
    ("2020-03-11", "코로나19 팬데믹", 1.00),
    ("2022-02-24", "러시아-우크라이나 전쟁", 1.00),
    ("2023-10-07", "중동 분쟁", 0.80),
]


def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    PATHS.figures.mkdir(exist_ok=True)
    (OUT_DIR / "matplotlib_cache").mkdir(exist_ok=True)
