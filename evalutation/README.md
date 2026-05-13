# Oil Price Forecast Lab

국내 유가, 국제 유가(WTI/Brent), 원/달러 환율, 뉴스 리스크를 함께 사용해
국내 휘발유 가격을 분석하고 향후 7일을 예측하는 로컬 분석 서버 프로젝트입니다.

ChungMaru 저장소의 구성 방식을 참고해 `backend`, `evaluation`, `docs`, `scripts`, `shared` 중심으로 정리했습니다.

## 현재 구조

```text
backend/                  FastAPI 로컬 서버와 유가 예측 파이프라인
backend/api/              데이터 수집, 전처리, EDA, 모델링, 예측 API
backend/models/           학습된 LSTM 모델과 스케일러
backend/outputs/          서버 실행 시 생성되는 최신 CSV/그래프
evaluation/oil-price/     발표/보고서용 평가 결과와 시각화 산출물
docs/                     서비스 정의, 데이터 출처, 실행 가이드
scripts/                  결과 패키지 생성 등 보조 스크립트
shared/contracts/         API 응답 예시와 데이터 계약 문서
```

## 핵심 기능

- OPINET 기반 국내 유가 수집
- WTI/Brent, 원/달러 환율, 뉴스 리스크 결합
- LSTM 기반 향후 7일 국내 유가 예측
- FastAPI 홈페이지와 `/docs` 제공
- 전체 그래프 갤러리 제공
- 간단한 분석 Agent API 제공
- 발표용 차트/JSON/PDF 결과 패키지 생성

## 빠른 실행

```bash
cd "/Users/seungwoolee/Desktop/High-level programming/evalutation"
python3 backend/server.py
```

실행 후 브라우저에서 접속합니다.

```text
http://127.0.0.1:8000
```

API 문서는 아래에서 확인합니다.

```text
http://127.0.0.1:8000/docs
```

## 주요 API

| Method | Path | 설명 |
| --- | --- | --- |
| GET | `/` | 웹 분석 대시보드 |
| GET | `/summary` | 최신 유가/예측/뉴스 리스크 요약 |
| GET | `/forecast` | 7일 예측 결과 |
| GET | `/graphs` | 전체 그래프 갤러리 |
| GET | `/graphs/list` | 그래프 목록 JSON |
| POST | `/agent` | 질문 기반 분석 답변 |
| POST | `/refresh` | 최신 데이터 수집 + EDA + 예측 재실행 |

## 그래프 산출물

`evaluation/oil-price/figures`와 `backend/outputs/figures`에 아래 그래프가 정리되어 있습니다.

- 유가 현황 및 7일 예측
- 향후 7일 유가 예측
- 유가추이 1주/1개월/1년/3년
- EDA 전체 분석
- 사건 전후 유가 변동
- WTI-국내 유가 산점도
- 일간 변동률 히스토그램
- 상관관계 히트맵

## 결과 패키지 생성

바탕화면의 `result` 폴더에 보고서형 산출물을 다시 만들 수 있습니다.

```bash
cd "/Users/seungwoolee/Desktop/High-level programming/evalutation"
python3 scripts/build_result_package.py
```

생성 결과:

```text
~/Desktop/result/
├── charts/
├── all_project_figures/
├── oil_price_analysis_*.json
├── oil_price_forecast_snapshot.json
├── 국내 유가 예측 모델 분석 보고서.pdf
└── 국내 유가 예측 모델 분석 보고서.hwpx
```

## 참고 문서

- [서비스 정의서](docs/service-definition.md)
- [데이터 출처 및 수집 방식](docs/data-sources.md)
- [실행 가이드](docs/runbook.md)
- [API 계약 문서](shared/contracts/README.md)
