import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / "matplotlib_cache"))
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / "outputs" / "matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

__all__ = ["plt"]
