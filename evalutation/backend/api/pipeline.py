import pandas as pd
from config import PATHS, ensure_dirs
from data_pipeline import collect_and_preprocess
from eda_analysis import run_eda

def run_all(epochs: int, device: str = "gpu", show_gui: bool = True):
    from forecasting import forecast_next_7_days
    from modeling import train_and_evaluate

    ensure_dirs()
    df = collect_and_preprocess()
    run_eda(df)
    model = train_and_evaluate(epochs=epochs, device=device)
    forecast_next_7_days(model=model, device=device, show_gui=show_gui)
    print(f"\n완료: {PATHS.online_raw.parent} 폴더에 CSV/그래프, {PATHS.model.parent} 폴더에 모델을 저장했습니다.")

def run_mode(mode: str, epochs: int, device: str = "gpu", show_gui: bool = True):
    ensure_dirs()
    if mode == "all":
        run_all(epochs, device=device, show_gui=show_gui)
    elif mode == "preprocess":
        collect_and_preprocess()
    elif mode == "eda":
        df = collect_and_preprocess()
        run_eda(df)
    elif mode == "train":
        from modeling import train_and_evaluate

        collect_and_preprocess()
        train_and_evaluate(epochs=epochs, device=device)
    elif mode == "forecast":
        from forecasting import forecast_next_7_days

        collect_and_preprocess()
        forecast_next_7_days(device=device, show_gui=show_gui)
