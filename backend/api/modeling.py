import joblib
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import PATHS
from plot_style import plt
from runtime import configure_tensorflow, get_tensorflow, suppress_native_stderr

SEED = 42


def make_sequences(processed: pd.DataFrame, lookback: int = 30, horizon: int = 7):
    data = processed.values
    target_idx = list(processed.columns).index("domestic_price")
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon, target_idx])
    return np.array(X), np.array(y)


def build_lstm(input_shape, horizon: int):
    tf = get_tensorflow(suppress_logs=True)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(horizon),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def inverse_domestic_price(values, scaler_bundle):
    scaler = scaler_bundle["scaler"]
    feature_cols = scaler_bundle["feature_cols"]
    target_idx = feature_cols.index("domestic_price")
    arr = np.zeros((len(values), len(feature_cols)))
    arr[:, target_idx] = values
    return scaler.inverse_transform(arr)[:, target_idx]


def train_and_evaluate(epochs: int = 30, lookback: int = 30, horizon: int = 7, device: str = "gpu"):
    print("\n[3] LSTM 모델링/검증")
    configure_tensorflow(device)
    random.seed(SEED)
    np.random.seed(SEED)
    processed = pd.read_csv(PATHS.processed, index_col=0, parse_dates=True)
    X, y = make_sequences(processed, lookback=lookback, horizon=horizon)
    if len(X) < 50:
        raise ValueError("학습 데이터가 부족합니다.")

    train_end = int(len(X) * 0.7)
    val_end = int(len(X) * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    with suppress_native_stderr(device == "cpu"):
        model = build_lstm((X.shape[1], X.shape[2]), horizon)
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=1)
        model.save(PATHS.model)
        pred = model.predict(X_test, verbose=0)

    scaler_bundle = joblib.load(PATHS.scaler)
    y_true = inverse_domestic_price(y_test.reshape(-1), scaler_bundle)
    y_pred = inverse_domestic_price(pred.reshape(-1), scaler_bundle)

    naive = np.repeat(X_test[:, -1, list(processed.columns).index("domestic_price")], horizon)
    y_naive = inverse_domestic_price(naive, scaler_bundle)

    lstm_mse = mean_squared_error(y_true, y_pred)
    naive_mse = mean_squared_error(y_true, y_naive)
    metrics = pd.DataFrame(
        [
            {
                "model": "LSTM",
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": np.sqrt(lstm_mse),
                "MAPE": np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-9))) * 100,
            },
            {
                "model": "Naive_last_price",
                "MAE": mean_absolute_error(y_true, y_naive),
                "RMSE": np.sqrt(naive_mse),
                "MAPE": np.mean(np.abs((y_true - y_naive) / np.maximum(y_true, 1e-9))) * 100,
            },
        ]
    )
    metrics.to_csv(PATHS.metrics, index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:120], label="실제")
    plt.plot(y_pred[:120], label="LSTM 예측")
    plt.plot(y_naive[:120], label="단순 기준선", alpha=0.7)
    plt.title("테스트 구간 예측 성능 비교")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS.figures / "test_prediction_compare.png", dpi=160)
    plt.close()
    return model
