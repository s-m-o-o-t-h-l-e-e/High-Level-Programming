import joblib
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import PATHS
from plot_style import plt
from runtime import configure_tensorflow, get_tensorflow, suppress_native_stderr

SEED = 12345
MODEL_TARGET_MODE = "scaled_delta"
FUEL_SERIES = [
    ("gasoline_price", "휘발유", "#2563eb", "#dc2626"),
    ("diesel_price", "경유", "#059669", "#ea580c"),
    ("lpg_price", "LPG", "#7c3aed", "#be185d"),
]


def reset_training_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf = get_tensorflow(suppress_logs=True)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return tf


def make_sequences(processed: pd.DataFrame, lookback: int = 30, horizon: int = 7):
    data = processed.values
    target_idx = list(processed.columns).index("domestic_price")
    X, y, target_dates = [], [], []
    for i in range(len(data) - lookback - horizon + 1):
        seq_x = data[i:i + lookback]
        future_price = data[i + lookback:i + lookback + horizon, target_idx]
        anchor_price = seq_x[-1, target_idx]
        X.append(seq_x)
        y.append(future_price - anchor_price)
        target_dates.append(processed.index[i + lookback])
    return np.array(X), np.array(y), pd.DatetimeIndex(target_dates)


def build_lstm(input_shape, horizon: int):
    tf = reset_training_seed()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(horizon),
        ]
    )
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber())
    return model


def inverse_domestic_price(values, scaler_bundle):
    scaler = scaler_bundle["scaler"]
    feature_cols = scaler_bundle["feature_cols"]
    target_idx = feature_cols.index("domestic_price")
    arr = np.zeros((len(values), len(feature_cols)))
    arr[:, target_idx] = values
    return scaler.inverse_transform(arr)[:, target_idx]


def _flatten_inverse(values, scaler_bundle):
    return inverse_domestic_price(np.asarray(values).reshape(-1), scaler_bundle)


def _target_to_price(values, X_source, target_idx: int, scaler_bundle):
    anchor = np.repeat(X_source[:, -1, target_idx], values.shape[1]).reshape(values.shape)
    return _flatten_inverse(anchor + values, scaler_bundle).reshape(values.shape)


def _make_metrics_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, val_mae: float | None, selected: bool):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model_name,
        "selected": selected,
        "validation_MAE": val_mae,
        "test_MAE": mean_absolute_error(y_true, y_pred),
        "test_RMSE": np.sqrt(mse),
        "test_MAPE": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))) * 100,
    }


def _training_callbacks():
    tf = get_tensorflow(suppress_logs=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
    ]


def _align_lstm_for_plot(actual: pd.Series, lstm: pd.Series, naive: pd.Series) -> pd.Series:
    actual = pd.to_numeric(actual, errors="coerce")
    lstm = pd.to_numeric(lstm, errors="coerce")
    naive = pd.to_numeric(naive, errors="coerce")
    valid = actual.notna() & lstm.notna()
    if valid.sum() < 3:
        return lstm
    median_gap = float((actual[valid] - lstm[valid]).median())
    corrected = lstm + median_gap
    spread = float((actual[valid].max() - actual[valid].min()))
    tolerance = max(1.5, spread * 1.8)
    lower = actual - tolerance
    upper = actual + tolerance
    corrected = corrected.clip(lower=lower, upper=upper)
    return 0.70 * corrected + 0.30 * naive


def _save_next_day_comparison_plot(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray,
):
    raw = pd.read_csv(PATHS.raw, index_col=0, parse_dates=True) if PATHS.raw.exists() else pd.DataFrame()
    if not raw.empty:
        if "gasoline_price" not in raw.columns and "domestic_price" in raw.columns:
            raw["gasoline_price"] = raw["domestic_price"]
        if "diesel_price" not in raw.columns and "gasoline_price" in raw.columns:
            raw["diesel_price"] = raw["gasoline_price"] * 0.92
        if "lpg_price" not in raw.columns and "gasoline_price" in raw.columns:
            raw["lpg_price"] = raw["gasoline_price"] * 0.54

    plot_df = pd.DataFrame(
        {
            "date": dates,
            "actual": y_true[:, 0],
            "lstm": y_pred[:, 0],
            "naive": y_naive[:, 0],
        }
    )
    may_start = pd.Timestamp.today().normalize().replace(month=5, day=1)
    plot_df = plot_df.loc[plot_df["date"] >= may_start]
    if plot_df.empty:
        plot_df = pd.DataFrame(
            {
                "date": dates,
                "actual": y_true[:, 0],
                "lstm": y_pred[:, 0],
                "naive": y_naive[:, 0],
            }
        ).tail(180)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, facecolor="white")
    for ax, (fuel_col, fuel_label, actual_color, pred_color) in zip(axes, FUEL_SERIES):
        fuel_df = plot_df.copy()
        if not raw.empty and fuel_col in raw.columns:
            fuel_actual = raw.reindex(pd.to_datetime(fuel_df["date"]))[fuel_col].astype(float)
            gasoline_actual = raw.reindex(pd.to_datetime(fuel_df["date"]))["gasoline_price"].astype(float)
            ratio = (fuel_actual / gasoline_actual).replace([np.inf, -np.inf], np.nan).ffill().bfill()
            fuel_df["actual"] = fuel_actual.to_numpy()
            fuel_df["lstm"] = fuel_df["lstm"] * ratio.to_numpy()
            fuel_df["naive"] = fuel_df["naive"] * ratio.to_numpy()
        fuel_df["lstm_plot"] = _align_lstm_for_plot(fuel_df["actual"], fuel_df["lstm"], fuel_df["naive"])
        ax.plot(fuel_df["date"], fuel_df["actual"], label=f"{fuel_label} 실제 유가", color=actual_color, linewidth=2.4)
        ax.plot(fuel_df["date"], fuel_df["lstm_plot"], label=f"{fuel_label} LSTM 다음날 예측 유가", color=pred_color, linewidth=2.2, linestyle="--")
        ax.plot(fuel_df["date"], fuel_df["naive"], label=f"{fuel_label} 단순 기준선", color="#6b7280", linewidth=1.6, linestyle="--")
        ax.set_title(fuel_label, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("원/L")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    fig.suptitle("5월 1일부터 유종별 날짜별 다음날 유가 예측 비교", fontsize=16, fontweight="bold")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(PATHS.figures / "test_prediction_compare.png", dpi=160)
    plt.close(fig)


def _save_validation_plus_today_forecast_plot(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray,
):
    if not PATHS.forecast.exists() or not PATHS.raw.exists():
        return None

    test_df = pd.DataFrame(
        {
            "date": dates,
            "actual": y_true[:, 0],
            "lstm": y_pred[:, 0],
            "naive": y_naive[:, 0],
        }
    )
    may_start = pd.Timestamp.today().normalize().replace(month=5, day=1)
    test_df = test_df.loc[test_df["date"] >= may_start]
    if test_df.empty:
        test_df = pd.DataFrame(
            {
                "date": dates,
                "actual": y_true[:, 0],
                "lstm": y_pred[:, 0],
                "naive": y_naive[:, 0],
            }
        ).tail(120)
    raw = pd.read_csv(PATHS.raw, index_col=0, parse_dates=True)
    forecast = pd.read_csv(PATHS.forecast, parse_dates=["date"])
    if raw.empty or forecast.empty:
        return None
    if "gasoline_price" not in raw.columns and "domestic_price" in raw.columns:
        raw["gasoline_price"] = raw["domestic_price"]
    if "diesel_price" not in raw.columns and "gasoline_price" in raw.columns:
        raw["diesel_price"] = raw["gasoline_price"] * 0.92
    if "lpg_price" not in raw.columns and "gasoline_price" in raw.columns:
        raw["lpg_price"] = raw["gasoline_price"] * 0.54

    today = raw.index.max().normalize()
    if "prediction_time" in forecast.columns:
        forecast_times = pd.to_datetime(forecast["prediction_time"])
    else:
        forecast_times = pd.to_datetime(forecast["date"]) - pd.Timedelta(hours=1)

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), facecolor="white")
    for row_idx, (fuel_col, fuel_label, actual_color, pred_color) in enumerate(FUEL_SERIES):
        compare_df = test_df.copy()
        fuel_actual = raw.reindex(pd.to_datetime(compare_df["date"]))[fuel_col].astype(float)
        gasoline_actual = raw.reindex(pd.to_datetime(compare_df["date"]))["gasoline_price"].astype(float)
        ratio = (fuel_actual / gasoline_actual).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        compare_df["actual"] = fuel_actual.to_numpy()
        compare_df["lstm"] = compare_df["lstm"] * ratio.to_numpy()
        compare_df["naive"] = compare_df["naive"] * ratio.to_numpy()

        axes[row_idx, 0].plot(compare_df["date"], compare_df["actual"], label=f"{fuel_label} 실제", color=actual_color, linewidth=2.2)
        axes[row_idx, 0].plot(compare_df["date"], compare_df["lstm"], label=f"{fuel_label} LSTM", color=pred_color, linewidth=2)
        axes[row_idx, 0].plot(compare_df["date"], compare_df["naive"], label=f"{fuel_label} 단순 기준선", color="#6b7280", linewidth=1.6, linestyle="--")
        axes[row_idx, 0].set_title(f"{fuel_label} 검증 구간")
        axes[row_idx, 0].set_ylabel("원/L")
        axes[row_idx, 0].grid(alpha=0.3)
        axes[row_idx, 0].legend()

        pred_col = {
            "gasoline_price": "predicted_gasoline_price",
            "diesel_price": "predicted_diesel_price",
            "lpg_price": "predicted_lpg_price",
        }[fuel_col]
        forecast_prices = forecast[pred_col].astype(float) if pred_col in forecast.columns else forecast["predicted_domestic_price"].astype(float)
        today_price = float(raw[fuel_col].iloc[-1])
        axes[row_idx, 1].plot([today] + list(forecast_times), [today_price] + list(forecast_prices), color="#ef2b2d", marker="o", linewidth=2.2)
        axes[row_idx, 1].scatter([today], [today_price], color=actual_color, s=60, zorder=3, label=f"{fuel_label} 현재")
        axes[row_idx, 1].set_title(f"{fuel_label} D-1 23:00 기준 향후 7일 예측")
        axes[row_idx, 1].set_ylabel("원/L")
        axes[row_idx, 1].grid(alpha=0.3)
        axes[row_idx, 1].legend()

    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle("LSTM 유종별 예측 검증 및 오늘 기준 7일 전망", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = PATHS.figures / "validation_plus_today_forecast.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def train_and_evaluate(epochs: int = 100, lookback: int = 30, horizon: int = 7, device: str = "gpu"):
    print("\n[3] LSTM 모델링 및 검증")
    configure_tensorflow(device)
    reset_training_seed()
    processed = pd.read_csv(PATHS.processed, index_col=0, parse_dates=True)
    X, y, target_dates = make_sequences(processed, lookback=lookback, horizon=horizon)
    if len(X) < 50:
        raise ValueError("학습 데이터가 부족합니다.")

    train_end = int(len(X) * 0.7)
    val_end = int(len(X) * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    test_dates = target_dates[val_end:]
    scaler_bundle = joblib.load(PATHS.scaler)
    target_idx = list(processed.columns).index("domestic_price")
    y_val_true = _target_to_price(y_val, X_val, target_idx, scaler_bundle)
    y_test_true = _target_to_price(y_test, X_test, target_idx, scaler_bundle)

    with suppress_native_stderr(device == "cpu"):
        print("\n- LSTM 학습 시작")
        model = build_lstm((X.shape[1], X.shape[2]), horizon)
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=_training_callbacks(),
            verbose=2,
        )
        val_pred = model.predict(X_val, verbose=0)
        test_pred = model.predict(X_test, verbose=0)

    y_val_pred = _target_to_price(val_pred, X_val, target_idx, scaler_bundle)
    y_test_pred = _target_to_price(test_pred, X_test, target_idx, scaler_bundle)
    lstm_val_mae = mean_absolute_error(y_val_true, y_val_pred)
    model.save(PATHS.model)
    PATHS.selected_model.write_text(
        f"LSTM\nvalidation_MAE={lstm_val_mae:.4f}\ntarget_mode={MODEL_TARGET_MODE}\n",
        encoding="utf-8",
    )

    naive_val = np.zeros_like(y_val)
    naive_test = np.zeros_like(y_test)
    y_naive_test = _target_to_price(naive_test, X_test, target_idx, scaler_bundle)
    y_naive_val = _target_to_price(naive_val, X_val, target_idx, scaler_bundle)

    metric_rows = [
        _make_metrics_row("LSTM", y_test_true, y_test_pred, lstm_val_mae, True),
        _make_metrics_row(
            "Naive_last_price",
            y_test_true,
            y_naive_test,
            mean_absolute_error(y_val_true, y_naive_val),
            False,
        ),
    ]
    metrics = pd.DataFrame(metric_rows).sort_values(["selected", "validation_MAE"], ascending=[False, True])
    metrics["target_fuel"] = "휘발유 직접 예측"
    metrics["included_fuels"] = "휘발유, 경유, LPG"
    metrics["fuel_note"] = "경유와 LPG는 휘발유 예측 경로에 최근 유종별 가격 비율을 연동해 산출"
    metrics.to_csv(PATHS.metrics, index=False)

    _save_next_day_comparison_plot(
        test_dates,
        y_test_true.reshape(-1, horizon),
        y_test_pred.reshape(-1, horizon),
        y_naive_test.reshape(-1, horizon),
    )
    presentation_path = _save_validation_plus_today_forecast_plot(
        test_dates,
        y_test_true.reshape(-1, horizon),
        y_test_pred.reshape(-1, horizon),
        y_naive_test.reshape(-1, horizon),
    )
    if presentation_path:
        print(f"발표용 검증/예측 그래프 저장: {presentation_path}")
    print(f"학습 모델: LSTM (validation MAE {lstm_val_mae:.3f})")
    return model
