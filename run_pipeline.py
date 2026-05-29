import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent / "oil_forecast_service" / "api"
sys.path.insert(0, str(API_DIR))

from cli import main


if __name__ == "__main__":
    main()
