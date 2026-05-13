import argparse
from pipeline import run_mode

def main():
    parser = argparse.ArgumentParser(description="지정학적 이벤트/거시지표 기반 유가 분석 및 LSTM 예측")
    parser.add_argument("--mode", choices=["all", "preprocess", "eda", "train", "forecast"], default="forecast")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="gpu",
        help="기본값 gpu는 Apple Metal GPU를 강제합니다. GPU 미감지 시 CPU로 실행하지 않고 바로 에러를 냅니다.",
    )
    parser.add_argument("--no-gui", action="store_true", help="GUI 창을 띄우지 않고 CSV/그래프 파일만 저장합니다.")
    args = parser.parse_args()
    run_mode(args.mode, args.epochs, device=args.device, show_gui=not args.no_gui)

if __name__ == "__main__":
    main()
