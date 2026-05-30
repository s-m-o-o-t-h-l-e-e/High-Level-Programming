# High-Level Programming Oil Price Forecast

국내 휘발유 가격을 대상으로 WTI, Brent, 원/달러 환율, 뉴스 리스크를 함께 반영해
향후 7일 유가를 예측하고 웹 대시보드에서 검증할 수 있는 분석 프로젝트입니다.

## Project Goal

단순히 LSTM 예측값만 보여주는 것이 아니라, 현재 시장 상황을 바탕으로 다음 가설을 세우고
예측 결과를 해석합니다.

> 지정학적 뉴스 리스크가 완화되면 국제유가에는 하락 압력이 발생할 수 있지만,
> 원/달러 환율이 높은 구간에서는 원유 수입 비용 부담이 커져 국내 휘발유 가격의
> 하락폭은 제한될 것이다.

## Structure

```text
run_server.py                 FastAPI 로컬 서버 실행 파일
run_pipeline.py               CLI 방식 분석/예측 실행 진입점
oil_forecast_service/         FastAPI 로컬 서버와 유가 예측 파이프라인
oil_forecast_service/api/     데이터 수집, 전처리, EDA, 모델링, 예측 API
oil_forecast_service/web/     웹 화면 HTML, CSS, JavaScript
oil_forecast_service/models/  학습된 LSTM 모델과 스케일러
oil_forecast_service/outputs/ 최신 CSV, 예측표, 그래프 산출물
project_docs/                 서비스 정의, 데이터 출처, 실행 가이드
```

## Main Features

- OPINET 기반 국내 휘발유 평균가격 수집
- WTI, Brent, 원/달러 환율, 뉴스 리스크 결합
- LSTM 기반 향후 7일 국내 유가 예측
- 과도한 예측 급등락을 줄이는 안정화 로직
- 실제 뉴스 제목, 환율, 국제유가를 엮은 변화 이유 자동 생성
- FastAPI 기반 로컬 홈 대시보드
- `/docs` 커스텀 API 실행 문서
- 전체 그래프 갤러리와 JSON API 제공
- 최신화 직후 예측값이 계속 흔들리지 않도록 10분 캐시 적용

## Quick Start

```bash
python3 run_server.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:8000
```

포트가 이미 사용 중이면 원하는 포트로 직접 실행합니다.

```bash
uvicorn oil_forecast_service.api.api_server:app --host 127.0.0.1 --port 8001
```

이 경우 브라우저 주소는 다음과 같습니다.

```text
http://127.0.0.1:8001
```

## Web Pages

| Path | 화면 |
| --- | --- |
| `/` | 홈 대시보드: 오늘 유가, 핵심 지표, 7일 예측, 그래프 미리보기 |
| `/home` | 홈 대시보드 별칭 |
| `/docs` | API를 버튼으로 실행하는 커스텀 실행 문서 |
| `/graphs` | 전체 분석 그래프 갤러리 |
| `/graphs/{filename}` | 개별 그래프 상세 보기 |

## Main API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/summary` | 최신 유가, 예측, 뉴스 리스크 요약 |
| GET | `/forecast` | 향후 7일 예측표 |
| GET | `/graphs/list` | 그래프 목록 JSON |
| POST | `/refresh` | 최신 데이터 수집, EDA, 예측 재실행 |
| POST | `/agent` | 질문 기반 간단 분석 응답 |
| GET | `/health` | 서버 상태 확인 |

## CLI Usage

웹 없이 CSV/그래프를 직접 생성하고 싶으면 아래처럼 실행합니다.

```bash
python3 run_pipeline.py --mode forecast --device auto --no-gui
```

전체 파이프라인을 다시 돌릴 때는 다음 명령을 사용합니다.

```bash
python3 run_pipeline.py --mode all --epochs 30 --device auto --no-gui
```

사용 가능한 모드:

| Mode | 역할 |
| --- | --- |
| `preprocess` | 온라인 데이터 수집 및 전처리 |
| `eda` | EDA/그래프 생성 |
| `train` | LSTM 모델 학습 |
| `forecast` | 최신 데이터 수집 후 7일 예측 |
| `all` | 전처리, EDA, 학습, 예측 전체 실행 |

## Key Outputs

| File | 내용 |
| --- | --- |
| `oil_forecast_service/outputs/raw_oil_project.csv` | 원본 분석 데이터 |
| `oil_forecast_service/outputs/processed_oil_project.csv` | 모델 입력용 전처리 데이터 |
| `oil_forecast_service/outputs/seven_day_forecast.csv` | 향후 7일 예측표 |
| `oil_forecast_service/outputs/news_articles.csv` | 뉴스 수집 결과 |
| `oil_forecast_service/outputs/news_signal.csv` | 뉴스 리스크 점수 |
| `oil_forecast_service/outputs/model_metrics.csv` | 모델 평가 결과 |
| `oil_forecast_service/models/oil_project_lstm.keras` | 학습된 LSTM 모델 |
| `oil_forecast_service/models/oil_project_scaler.pkl` | 스케일러 |

## Endpoint Summary

| Method | Path | Output |
| --- | --- | --- |
| GET | `/summary` | 최신 유가, 예측, 뉴스 리스크 요약 |
| GET | `/forecast` | 향후 7일 예측표 |
| GET | `/graphs/list` | 그래프 목록 JSON |
| POST | `/refresh` | 최신 데이터 수집, EDA, 예측 재실행 |
| POST | `/agent` | 질문과 요약 근거 |

## Forecast Logic

1. 최신 국내 유가, WTI, Brent, 환율, 뉴스 데이터를 수집합니다.
2. LSTM 모델이 7일 예측값을 생성합니다.
3. LSTM 원시 예측이 최근 추세에서 크게 벗어나면 반영 비중을 자동으로 낮춥니다.
4. 최근 국내 유가 추세 기준선을 함께 반영합니다.
5. 뉴스 리스크 보정률과 일간 변화폭 제한을 적용해 단기 예측을 안정화합니다.
6. 환율과 뉴스 리스크를 엮어 변화 이유를 생성합니다.

## Graph Outputs

`oil_forecast_service/outputs/figures`에서 아래 그래프를 확인할 수 있습니다.

- 유가추이 1주
- 유가추이 1개월
- 유가추이 1년
- 유가추이 3년
- 오늘 기준 7일 예측 유가 그래프
- 유가 현황 및 7일 예측
- 일간 변동률 히스토그램
- 최근 일간 변화 막대그래프
- WTI-국내 유가 산점도
- 가격 분포 박스플롯
- 일간 변동률 밀도 그래프
- 시장 지표 바이올린 플롯
- EDA 전체 분석
- 사건 전후 유가 변동
- 상관관계 히트맵

## Data Sources

- 국내 유가: OPINET Open API 및 OPINET 기반 전국 평균가격 데이터
- 국제 유가: WTI, Brent
- 거시 지표: 원/달러 환율
- 뉴스/이벤트: Google News RSS, GDELT, AI Hub CSV 연동 구조

자세한 내용은 [데이터 출처 문서](project_docs/data-sources.md)를 참고하세요.

## Documents

- [서비스 정의서](project_docs/service-definition.md)
- [데이터 출처 및 수집 방식](project_docs/data-sources.md)
- [실행 가이드](project_docs/runbook.md)
