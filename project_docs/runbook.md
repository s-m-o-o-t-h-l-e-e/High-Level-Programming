# 실행 가이드

## 1. 서버 실행

```bash
cd "term project"
python3 -m pip install -r oil_forecast_service/requirements.txt
python3 run_server.py
```

브라우저에서 접속:

```text
http://127.0.0.1:8001
```

## 2. API 문서 확인

```text
http://127.0.0.1:8001/docs
```

## 3. 전체 그래프 확인

```text
http://127.0.0.1:8001/graphs
```

## 4. 최신 데이터 갱신

홈페이지의 `최신 데이터 갱신` 버튼을 누르거나 `/docs`에서 `POST /refresh`를 실행합니다.

주의: 네트워크 요청과 모델 예측이 포함되어 시간이 걸릴 수 있습니다.
23시대에는 다음날 예측을 생성하고, 그 외 시간에는 핵심 지표와 뉴스 중심으로 갱신됩니다.

## 5. 파이프라인 직접 실행

웹 서버 없이 산출물만 다시 만들 때 사용합니다.

```bash
cd "term project"
python3 run_pipeline.py --mode forecast --device gpu --no-gui
```

전체 재생성은 다음과 같이 실행합니다.

```bash
python3 run_pipeline.py --mode all --epochs 30 --device gpu --no-gui
```

## 문제 해결

### Address already in use

이미 8001번 포트에서 서버가 실행 중입니다.

```bash
lsof -ti :8001
kill <PID>
```

또는 기존 서버를 그대로 사용해도 됩니다.

다른 포트로 실행하려면 다음처럼 실행합니다.

```bash
python3 -m uvicorn oil_forecast_service.api.api_server:app --host 127.0.0.1 --port 8002
```

### 그래프가 안 보일 때

아래 경로에 PNG 파일이 있는지 확인합니다.

```text
oil_forecast_service/outputs/figures
```

없으면 `/refresh`를 실행하거나 기존 산출물을 다시 복사합니다.

### Mac GPU가 안 잡힐 때

이 프로젝트는 MacBook 환경에서 CUDA가 아니라 Apple Metal 기반 GPU 사용을 전제로 합니다.
`--device gpu` 실행 시 GPU가 감지되지 않으면 오류가 발생하므로 `tensorflow-macos`, `tensorflow-metal` 설치 상태를 확인합니다.
