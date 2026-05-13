import contextlib
import os
import platform
from pathlib import Path


_TF = None
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / "matplotlib_cache"))
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / "outputs" / "matplotlib_cache")


def get_tensorflow(suppress_logs: bool = False):
    global _TF
    if _TF is not None:
        return _TF
    with suppress_native_stderr(suppress_logs):
        try:
            import tensorflow as tf
        except Exception:
            tf = None
    _TF = tf
    return _TF


def configure_tensorflow(device: str = "auto"):
    tf = get_tensorflow(suppress_logs=device == "cpu")
    if tf is None:
        raise RuntimeError("TensorFlow가 설치되어 있지 않아 LSTM 학습/예측을 실행할 수 없습니다.")

    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        print("실행 장치: CPU (MacBook/CUDA 미사용)")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        message = "사용 가능한 TensorFlow Metal GPU가 없습니다."
        if device == "gpu":
            raise RuntimeError(
                f"{message} 현재 환경에서는 GPU 실행을 할 수 없습니다. "
                "tensorflow-metal/OS/칩셋 지원 상태를 확인해야 합니다."
            )
        print(f"실행 장치: CPU ({message})")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    backend = "Apple Metal GPU" if platform.system() == "Darwin" else "GPU"
    print(f"실행 장치: {backend} ({len(gpus)}개 감지)")


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool):
    if not enabled:
        yield
        return

    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)
