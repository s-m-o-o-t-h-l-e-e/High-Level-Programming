# Oil Price Evaluation

유가 예측 프로젝트의 분석 결과와 발표용 그래프를 모아둔 폴더입니다.

## 구성

```text
figures/       보고서와 웹 대시보드에서 사용하는 PNG 그래프
results/       CSV 기반 분석 결과와 예측 결과
README.md      평가 산출물 설명
```

## 주요 그래프

- `oil_price_dashboard.png`: 유가 현황 및 7일 예측
- `seven_day_forecast.png`: 향후 7일 예측 선 그래프
- `oil_price_trend_1w.png`: 1주 유가 추이
- `oil_price_trend_1m.png`: 1개월 유가 추이
- `oil_price_trend_1y.png`: 1년 유가 추이
- `oil_price_trend_3y.png`: 3년 유가 추이
- `scatter_wti_domestic.png`: WTI 변동률과 국내 유가 변동률 산점도
- `histogram_daily_returns.png`: 일간 변동률 분포
- `correlation_heatmap.png`: 주요 변수 상관관계
- `eda_overview.png`: 전체 EDA 요약
- `event_window_changes.png`: 사건 전후 유가 변동

## 주요 결과 CSV

- `seven_day_forecast.csv`: 7일 예측 결과
- `analysis_summary.csv`: 기술통계 요약
- `model_metrics.csv`: 모델 평가 지표
- `lag_correlation.csv`: 시차 상관 분석
- `news_signal.csv`: 뉴스 리스크 신호
- `data_source_audit.csv`: 데이터 출처 감사 로그
