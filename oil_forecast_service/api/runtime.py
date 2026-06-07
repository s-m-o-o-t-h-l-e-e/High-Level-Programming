import contextlib
import os
import platform
from pathlib import Path


_TF = None
_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)
(_OUTPUT_DIR / "matplotlib_cache").mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_OUTPUT_DIR / "matplotlib_cache")


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
            python_hint = "python3 -m pip install --upgrade tensorflow-macos tensorflow-metal"
            system_hint = f"{platform.system()} {platform.machine()} / Python {platform.python_version()} / TensorFlow {tf.__version__}"
            raise RuntimeError(
                f"{message} CPU로 대체 실행하지 않습니다.\n"
                f"현재 환경: {system_hint}\n"
                "Mac GPU 학습은 Apple Metal을 TensorFlow가 GPU 장치로 인식해야만 가능합니다.\n"
                f"로컬 설치 확인 예: {python_hint}"
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
