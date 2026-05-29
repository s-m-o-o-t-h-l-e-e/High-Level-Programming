# 실행 가이드

## 1. 서버 실행

```bash
cd High-Level-Programming
python3 server.py
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

## 5. 결과 패키지 생성

```bash
cd High-Level-Programming
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
