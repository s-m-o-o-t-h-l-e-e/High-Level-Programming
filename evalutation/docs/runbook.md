# 실행 가이드

## 1. 서버 실행

```bash
cd "/Users/seungwoolee/Desktop/High-level programming/evalutation"
python3 backend/server.py
```

브라우저에서 접속:

```text
http://127.0.0.1:8000
```

## 2. API 문서 확인

```text
http://127.0.0.1:8000/docs
```

## 3. 전체 그래프 확인

```text
http://127.0.0.1:8000/graphs
```

## 4. 최신 데이터 갱신

홈페이지의 `최신 데이터 갱신` 버튼을 누르거나 `/docs`에서 `POST /refresh`를 실행합니다.

주의: 네트워크 요청과 모델 예측이 포함되어 시간이 걸릴 수 있습니다.

## 5. Agent 질문 예시

`POST /agent`에 아래처럼 질문할 수 있습니다.

```json
{
  "question": "전쟁 뉴스 리스크가 유가 예측에 어떤 영향을 줘?"
}
```

질문 예시:

- 오늘 국내 유가와 7일 예측을 알려줘
- WTI와 Brent가 국내 유가에 어떤 영향을 줘?
- 원/달러 환율이 유가에 미치는 영향은?
- 뉴스 리스크가 높은 이유가 뭐야?

## 6. 결과 패키지 생성

```bash
cd "/Users/seungwoolee/Desktop/High-level programming/evalutation"
python3 scripts/build_result_package.py
```

결과는 바탕화면에 생성됩니다.

```text
~/Desktop/result
```

## 문제 해결

### Address already in use

이미 8000번 포트에서 서버가 실행 중입니다.

```bash
lsof -ti :8000
kill <PID>
```

또는 기존 서버를 그대로 사용해도 됩니다.

### 그래프가 안 보일 때

아래 경로에 PNG 파일이 있는지 확인합니다.

```text
backend/outputs/figures
```

없으면 `/refresh`를 실행하거나 기존 산출물을 다시 복사합니다.
