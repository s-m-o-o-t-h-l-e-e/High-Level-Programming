# High-Level Programming Oil Price Forecast

국내 휘발유 가격을 대상으로 WTI, Brent, 원/달러 환율, 뉴스 리스크를 함께 반영해
향후 7일 유가를 예측하고 웹 대시보드에서 검증할 수 있는 발표용 분석 프로젝트입니다.

## Project Goal

단순히 LSTM 예측값만 보여주는 것이 아니라, 현재 시장 상황을 바탕으로 다음 가설을 세우고
예측 결과를 해석합니다.

> 지정학적 뉴스 리스크가 완화되면 국제유가에는 하락 압력이 발생할 수 있지만,
> 원/달러 환율이 높은 구간에서는 원유 수입 비용 부담이 커져 국내 휘발유 가격의
> 하락폭은 제한될 것이다.

따라서 예측표의 변화 이유는 LSTM 내부값 설명에 그치지 않고, 뉴스 리스크, 환율,
국제유가 흐름을 묶어 발표용 검증 문장으로 제공합니다.

## Current Release

`v0.0.2`는 발표 검증용 개선 버전입니다.

- LSTM 원시 예측이 과도하게 튀는 문제를 완화
- 최근 추세 기준선과 LSTM 예측을 혼합하는 안정화 로직 추가
- 뉴스 리스크 보정률을 단기 예측에 맞게 축소
- 예측표의 변화 이유를 환율과 뉴스 리스크 중심으로 재작성
- 박스플롯, 막대그래프, KDE 밀도 그래프, 바이올린 플롯 추가
- `/docs` 커스텀 UI와 그래프 목록에서 추가 시각화 확인 가능

## Structure

```text
backend/                  FastAPI 로컬 서버와 유가 예측 파이프라인
backend/api/              데이터 수집, 전처리, EDA, 모델링, 예측 API
backend/models/           학습된 LSTM 모델과 스케일러
backend/outputs/          최신 CSV, 예측표, 그래프 산출물
docs/                     서비스 정의, 데이터 출처, 실행 가이드
evaluation/oil-price/     발표/보고서용 평가 결과
scripts/                  결과 패키지 생성 스크립트
shared/contracts/         API 응답 예시와 데이터 계약 문서
```

## Main Features

- OPINET 기반 국내 휘발유 평균가격 수집
- WTI, Brent, 원/달러 환율, 뉴스 리스크 결합
- LSTM 기반 향후 7일 국내 유가 예측
- 과도한 예측 급등락을 줄이는 안정화 로직
- 현재 상황 가설 기반 변화 이유 자동 생성
- FastAPI 기반 로컬 웹 대시보드
- `/docs` 커스텀 실행 UI
- 전체 그래프 갤러리와 JSON API 제공

## Quick Start

```bash
cd High-Level-Programming
python3 server.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:8001
```

커스텀 실행 UI는 아래 주소에서 확인합니다.

```text
http://127.0.0.1:8001/docs
```

포트가 이미 사용 중이면 다음처럼 다른 포트로 실행합니다.

```bash
uvicorn backend.api.app:app --host 127.0.0.1 --port 8002
```

## Main API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | 웹 분석 대시보드 |
| GET | `/summary` | 최신 유가, 예측, 뉴스 리스크 요약 |
| GET | `/forecast` | 향후 7일 예측표 |
| GET | `/graphs` | 전체 그래프 갤러리 |
| GET | `/graphs/list` | 그래프 목록 JSON |
| POST | `/refresh` | 최신 데이터 수집, EDA, 예측 재실행 |

## Forecast Logic

1. 최신 국내 유가, WTI, Brent, 환율, 뉴스 데이터를 수집합니다.
2. LSTM 모델이 7일 예측값을 생성합니다.
3. LSTM 원시 예측이 최근 추세에서 크게 벗어나면 반영 비중을 자동으로 낮춥니다.
4. 최근 국내 유가 추세 기준선을 함께 반영합니다.
5. 뉴스 리스크 보정률과 일간 변화폭 제한을 적용해 단기 예측을 안정화합니다.
6. 환율과 뉴스 리스크를 엮어 변화 이유를 생성합니다.

## Graph Outputs

`backend/outputs/figures`에서 아래 그래프를 확인할 수 있습니다.

- 유가추이 1주
- 유가추이 1개월
- 유가추이 1년
- 유가추이 3년
- 향후 7일 유가 예측
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

자세한 내용은 [데이터 출처 문서](docs/data-sources.md)를 참고하세요.

## Result Package

발표/제출용 결과 패키지는 다음 명령으로 다시 만들 수 있습니다.

```bash
python3 scripts/build_result_package.py
```

생성 위치:

```text
~/Desktop/result/
```

## Documents

- [서비스 정의서](docs/service-definition.md)
- [데이터 출처 및 수집 방식](docs/data-sources.md)
- [실행 가이드](docs/runbook.md)
- [API 계약 문서](shared/contracts/README.md)
