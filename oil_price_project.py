import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent / "backend" / "api"
sys.path.insert(0, str(API_DIR))

from oil_price_project import main


if __name__ == "__main__":
    main()
