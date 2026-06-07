# 데이터 출처 및 수집 방식

## 국내 유가

- 주 출처: OPINET 전국 평균가격 API 및 OPINET 기반 전국 평균가격 이력
- 최신 당일 보강: OPINET-linked public today oil price page
- 사용 목적: 국내 휘발유·경유·LPG 전국 평균 가격의 기준값 및 예측 대상
- 저장 위치: `oil_forecast_service/outputs/raw_oil_project.csv`

## 국제 유가

- 주 출처: Yahoo Finance futures ticker
  - WTI: `CL=F`
  - Brent: `BZ=F`
- 보조 출처: Alpha Vantage WTI/BRENT API, Investing.com historical table 시도
- 접근 제한 또는 API 키 문제 발생 시 주 경로 데이터와 기존 산출물로 자동 보강

## 환율

- 주 출처: Yahoo Finance `KRW=X`
- 보조 출처: FRED `DEXKOUS`
- 의미: 원/달러 환율이 국내 유가에 미치는 수입 비용 압력 반영

## 뉴스/이벤트

- 주 출처: Google News RSS
- 보조 출처: GDELT Doc API
- 키워드 예시:
  - Iran
  - United States
  - Hormuz
  - oil
  - crude
  - 이란
  - 미국
  - 호르무즈
  - 원유

## 산출 파일

```text
oil_forecast_service/outputs/
├── online_oil_dataset.csv
├── raw_oil_project.csv
├── processed_oil_project.csv
├── seven_day_forecast.csv
├── forecast_history.csv
├── news_signal.csv
├── news_articles.csv
├── event_window_summary.csv
├── event_risk_scores.csv
├── data_source_audit.csv
└── figures/
```

## 주의 사항

- 국제 유가와 국내 유가는 단위와 시장 시간이 다릅니다.
- 국내 유가는 휘발유·경유·LPG의 일별 전국 평균 데이터 중심입니다.
- WTI/Brent의 1주 그래프는 시간봉을 쓰며, 주말/휴장 공백은 선을 끊어 표시합니다.
- 뉴스 리스크는 정량화된 보조 신호이며 실제 가격을 보장하지 않습니다.
- LSTM 원시 예측값은 최종값이 아니라 최근 추세 기준선과 안정화 로직에 결합되는 보조 신호입니다.
