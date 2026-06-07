# High-Level Programming Oil Forecast Service

국내 휘발유·경유·LPG 평균가를 중심으로 WTI, Brent, 원/달러 환율, 뉴스 리스크를 함께 수집하고,
LSTM 기반 7일 예측과 EDA 그래프를 로컬 웹 대시보드에서 확인하는 고급프로그래밍설계 팀 프로젝트입니다.

이 프로젝트는 단순히 LSTM 예측값만 보여주는 것이 아니라, 국내 유가가 국제 유가보다 완만하게 움직이는 구조를 고려해
최근 추세, 환율, 뉴스 리스크, 일간 변동폭 제한을 결합한 안정화 예측값을 제공합니다.

## 핵심 가설

> 지정학적 리스크와 국제 유가는 국내 유가 방향성에 영향을 주지만,
> 국내 휘발유·경유·LPG 평균가는 세금, 유통 구조, 재고, 환율 때문에 단기적으로 급격히 움직이지 않는다.
> 따라서 LSTM 원시 예측값은 보조 신호로 사용하고, 최종 예측은 국내 유가의 완만한 전이 구조 안에서 안정화해야 한다.

## 프로젝트 구조

```text
term project/
├── README.md
├── run_server.py
├── run_pipeline.py
├── oil_forecast_service/
│   ├── requirements.txt
│   ├── api/
│   │   ├── api_server.py       # FastAPI 서버와 웹 라우트
│   │   ├── cli.py              # CLI 인자 처리
│   │   ├── config.py           # 경로, 이벤트, 공통 설정
│   │   ├── data_pipeline.py    # 수집 데이터 병합, 전처리, 리스크 지수
│   │   ├── eda_analysis.py     # EDA, 시각화, 사건 전후 분석
│   │   ├── forecasting.py      # 7일 예측, 예측 이력, 예측 그래프
│   │   ├── modeling.py         # LSTM 학습과 검증 그래프
│   │   ├── news_signal.py      # Google News/GDELT 기반 뉴스 리스크
│   │   ├── online_data.py      # OPINET, 국제 유가, 환율 수집
│   │   ├── pipeline.py         # 전체 실행 파이프라인
│   │   ├── plot_style.py       # 그래프 스타일 설정
│   │   └── runtime.py          # GPU/CPU 실행 장치 확인
│   ├── web/
│   │   ├── dashboard.html      # 홈 대시보드
│   │   ├── dashboard.js
│   │   ├── api_docs.html       # 커스텀 API 실행 문서
│   │   ├── api_docs.js
│   │   ├── graph_gallery.html  # 전체 그래프 갤러리
│   │   ├── graph_gallery.js
│   │   ├── graph_detail.html
│   │   └── theme.css
│   ├── models/
│   │   ├── oil_project_lstm.keras
│   │   └── oil_project_scaler.pkl
│   └── outputs/
│       ├── raw_oil_project.csv
│       ├── processed_oil_project.csv
│       ├── seven_day_forecast.csv
│       ├── forecast_history.csv
│       ├── news_signal.csv
│       ├── news_articles.csv
│       ├── event_window_summary.csv
│       ├── data_source_audit.csv
│       └── figures/
└── project_docs/
    ├── README.md
    ├── data-sources.md
    ├── runbook.md
    └── service-definition.md
```

## 주요 기능

- OPINET 기반 국내 휘발유·경유·LPG 전국 평균가 수집
- WTI, Brent 국제 유가와 원/달러 환율 수집
- Google News/GDELT 기반 이란·미국·호르무즈·원유 뉴스 리스크 점수화
- LSTM 기반 향후 7일 유종별 가격 예측
- LSTM 원시 출력의 과도한 변동을 줄이는 안정화 로직
- 전날 23시 기준 예측과 다음날 실제 유가를 비교하는 운영 검증 이력 저장
- 유가 추이, 예측, 히스토그램, 밀도 그래프, 산점도, 히트맵, 사건 전후 분석 그래프 생성
- FastAPI 기반 로컬 대시보드, 커스텀 `/docs`, 전체 그래프 갤러리 제공
- 서버 실행 중 자동 최신화 정책 적용

## 빠른 실행

```bash
cd "term project"
python3 -m pip install -r oil_forecast_service/requirements.txt
python3 run_server.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:8001
```

포트 `8001`이 이미 사용 중이면 다른 포트로 직접 실행할 수 있습니다.

```bash
python3 -m uvicorn oil_forecast_service.api.api_server:app --host 127.0.0.1 --port 8002
```

## 웹 화면

| 주소 | 설명 |
| --- | --- |
| `/` | 홈 대시보드: 오늘 유종별 평균가, 핵심 지표, 7일 예측, 그래프 미리보기 |
| `/home` | 홈 대시보드 별칭 |
| `/docs` | 발표/확인용 커스텀 API 실행 문서 |
| `/graphs` | 전체 분석 그래프 갤러리 |
| `/graphs/{filename}` | 개별 그래프 상세 보기 |

홈 화면에서는 우측 오늘 평균가 카드에서 휘발유·경유·LPG를 선택해 큰 가격 표시를 바꿀 수 있고,
그래프 영역에서는 유종별 7일 예측, 6월 실제 유가와 LSTM 예측 비교, EDA 그래프를 바로 확인할 수 있습니다.

## API

| Method | Path | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/summary` | 최신 유가, WTI/Brent, 환율, 뉴스 리스크, 예측 요약 |
| `GET` | `/forecast` | 향후 7일 유종별 예측표 |
| `GET` | `/graphs/list` | 웹에서 표시 가능한 그래프 목록 |
| `POST` | `/refresh` | 최신 데이터 수집, EDA/예측/그래프 재생성 |
| `POST` | `/agent` | 질문 기반 보조 분석 응답 API |

현재 발표용 UI는 `/`와 `/docs`를 중심으로 구성되어 있고, Swagger 기본 화면 대신 커스텀 실행 문서를 제공합니다.

## CLI 실행

웹 서버 없이 CSV와 그래프를 직접 생성할 수도 있습니다.

```bash
cd "term project"
python3 run_pipeline.py --mode forecast --device gpu --no-gui
```

전체 파이프라인을 다시 실행하려면 다음 명령을 사용합니다.

```bash
python3 run_pipeline.py --mode all --epochs 30 --device gpu --no-gui
```

사용 가능한 모드는 다음과 같습니다.

| Mode | 역할 |
| --- | --- |
| `preprocess` | 온라인 데이터 수집 및 전처리 |
| `eda` | EDA 분석과 그래프 생성 |
| `train` | LSTM 모델 학습 |
| `forecast` | 최신 데이터 기반 7일 예측 |
| `all` | 전처리, EDA, 학습, 예측 전체 실행 |

Mac 환경에서는 CUDA 대신 `tensorflow-macos`와 `tensorflow-metal`을 사용합니다.
`--device gpu` 실행 시 Apple Metal GPU가 감지되지 않으면 CPU로 조용히 내려가지 않고 오류를 발생시킵니다.

## 주요 산출물

| 파일 | 내용 |
| --- | --- |
| `outputs/raw_oil_project.csv` | 분석용 원본 패널 데이터 |
| `outputs/processed_oil_project.csv` | LSTM 입력용 스케일링 데이터 |
| `outputs/seven_day_forecast.csv` | 최신 7일 유종별 예측표 |
| `outputs/forecast_history.csv` | D-1 23:00 기준 예측 저장 이력 |
| `outputs/news_signal.csv` | 뉴스 리스크 점수와 보정률 |
| `outputs/news_articles.csv` | 수집한 뉴스 기사 목록 |
| `outputs/event_window_summary.csv` | 주요 사건 전후 유종별 가격 변화율 |
| `outputs/event_risk_scores.csv` | 사건 특성 기반 리스크 점수 |
| `outputs/data_source_audit.csv` | 최신 수집 시각과 데이터 출처 감사 정보 |
| `models/oil_project_lstm.keras` | 학습된 LSTM 모델 |
| `models/oil_project_scaler.pkl` | 모델 입력 스케일러 |

## 그래프 산출물

`oil_forecast_service/outputs/figures`에 생성됩니다.

| 그래프 | 파일 |
| --- | --- |
| 유가추이 1주 | `oil_price_trend_1w.png` |
| 유가추이 1개월 | `oil_price_trend_1m.png` |
| 유가추이 1년 | `oil_price_trend_1y.png` |
| 유가추이 3년 | `oil_price_trend_3y.png` |
| 오늘 기준 유종별 7일 예측 | `today_based_forecast.png` |
| 유가 현황 및 예측 대시보드 | `oil_price_dashboard.png` |
| 6월 실제 유가 vs LSTM 예측 유가 | `june_actual_lstm_forecast_compare.png` |
| 일간 변동률 히스토그램 | `histogram_daily_returns.png` |
| 일간 변동률 밀도 그래프 | `kde_return_distribution.png` |
| WTI-국내 유가 산점도 | `scatter_wti_domestic.png` |
| 가격 분포 박스플롯 | `boxplot_price_distribution.png` |
| 최근 일간 변화 막대그래프 | `bar_recent_changes.png` |
| 시장 지표 바이올린 플롯 | `violin_market_distribution.png` |
| EDA 전체 분석 | `eda_overview.png` |
| 사건 전후 유가 변동 | `event_window_changes.png` |
| 상관관계 히트맵 | `correlation_heatmap.png` |

## 예측 로직 요약

1. OPINET 기반 국내 유종별 평균가, WTI/Brent, 원/달러 환율, 뉴스 데이터를 수집합니다.
2. 날짜 인덱스를 통일하고 결측치를 보간하여 일별 패널 데이터를 구성합니다.
3. 최근 30일 다변량 시계열을 LSTM 입력으로 사용합니다.
4. LSTM은 향후 7일 가격 경로의 후보 신호를 생성합니다.
5. 최종 예측값은 최근 추세 기준선, LSTM 신호, 환율 상승 압력, 뉴스 리스크, 일간 변화폭 제한을 결합해 산출합니다.
6. 예측 결과는 `seven_day_forecast.csv`와 `forecast_history.csv`에 저장되고 웹 대시보드와 그래프에 반영됩니다.
7. 전날 23시 예측과 다음날 실제값을 비교할 수 있도록 운영 검증 이력을 누적합니다.

## 최신화 정책

- 서버 시작 시 기존 산출물이 오래되었으면 최신화를 시도합니다.
- `/refresh` 또는 홈 화면의 최신화 버튼은 최신 유가, 환율, 국제 유가, 뉴스 리스크를 다시 수집합니다.
- 23시대 최신화는 다음날 예측 이력을 남기는 기준 시점으로 사용합니다.
- 23시 외 최신화는 핵심 지표와 그래프 갱신을 수행하되, 운영 검증용 다음날 예측 기준은 23시 규칙을 따릅니다.

## 데이터 출처

- 국내 유가: OPINET Open API, OPINET 기반 전국 평균가격 데이터, OPINET-linked public today oil price page
- 국제 유가: WTI, Brent
- 거시 지표: 원/달러 환율
- 뉴스/이벤트: Google News RSS, GDELT 실시간 뉴스

자세한 내용은 [데이터 출처 문서](project_docs/data-sources.md)를 참고하세요.

## 문서

- [프로젝트 문서 README](project_docs/README.md)
- [서비스 정의서](project_docs/service-definition.md)
- [데이터 출처 및 수집 방식](project_docs/data-sources.md)
- [실행 가이드](project_docs/runbook.md)
