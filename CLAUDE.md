\# CLAUDE.md 생성

@"

\# ETF\_mixer 프로젝트



\## 서버

\- AWS: /home/ubuntu/ETF\_mixer

\- 서비스 재시작: sudo systemctl restart etf\_mixer.service

\- 브랜치: main (작업별 feature 브랜치 생성)



\## 데이터

\- DB: /home/ubuntu/ETF\_mixer/data/prices.sqlite

\- 캐시: backend/data/cache/

\- ETF 626개, 주봉 52주 기준 (weekly\_window=52, months=24)



\## 현재 완료 상태 (2026-03-04)

\- STEP2 완료: QP 3버킷(0-3%/3-6%/6-9%), 5종목, base/delta

\- UI: backend/static/index2.html (strategy=qp)

\- API: /api/portfolios?strategy=qp



\## 다음 작업

\- STEP3: /api/custom\_portfolio\_eval, /api/benchmark\_prices

\- STEP4: UI 개편

\- STEP5: 실시간 가격 반영 (미완성 주봉 처리)



\## 핵심 설계 규칙

\- RISK\_BUCKETS: 0-3%, 3-6%, 6-9% (3개)

\- 버킷별 gamma: 0-3%=8.0, 3-6%=4.0, 6-9%=2.0

\- delta score = S\_now + 0.5\*(delta\_S / max(risk\_pct, 2.0))

\- min\_weight=0.10, upper\_bound=0.40, 5종목 고정

\- 동일 카테고리 최대 2개

\- 벤치마크: KOSPI=069500, KOSDAQ=229200, SNP500=360750



\## 알려진 이슈

\- 포트 수익률(일봉 복리) vs 개별종목 수익률(주봉 log) 방식 불일치

&nbsp; → STEP4에서 주봉 기준으로 통일 예정

\- 6-9% 버킷 3종목 (fallback, 구조적 한계)

\- 0-3% 위험 3.08% (버킷 상한 3% 미세 초과, 허용)



\## 작업 환경 규칙

\- 코드 수정: 로컬 Codex → git push → AWS pull

\- AWS 직접 수정: 긴급 시에만, 반드시 git push로 동기화

\- 검증: AWS curl/python 조회

"@ | Out-File -FilePath CLAUDE.md -Encoding UTF8



git add CLAUDE.md

git commit -m "docs: add CLAUDE.md"

git push origin main

