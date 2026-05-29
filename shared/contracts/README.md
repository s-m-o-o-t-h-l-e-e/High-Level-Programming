# API 계약 문서

FastAPI 서버가 제공하는 주요 응답 구조를 정리합니다.

## Summary

```http
GET /summary
```

응답 필드:

- `latest`: 최신 국내 유가, WTI, Brent, 환율, 리스크 지표
- `forecast`: 향후 7일 예측 목록
- `news`: 뉴스 리스크 점수와 예측 보정률
- `meta`: 데이터 수집 메타데이터
- `sources`: 데이터 출처 감사 정보

## Forecast

```http
GET /forecast
```

응답은 7일 예측 배열입니다.

```json
[
  {
    "date": "2026-05-03",
    "predicted_domestic_price": 2027.13,
    "news_risk_score": 1.0,
    "news_adjustment_pct": 0.035,
    "news_article_count": 100
  }
]
```

## Graphs

```http
GET /graphs/list
```

응답:

```json
[
  {
    "filename": "correlation_heatmap.png",
    "title": "상관관계 히트맵",
    "url": "/figures/correlation_heatmap.png",
    "detail_url": "/graphs/correlation_heatmap.png"
  }
]
```
