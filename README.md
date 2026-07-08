# 유타

원유가격 기반 일본행 항공권 발권 타이밍 참고 서비스입니다.

## 핵심 안내

- 이 앱은 실제 항공권 가격을 예측하지 않습니다.
- 이 앱은 실제 항공사 유류할증료 고시표를 직접 사용하지 않습니다.
- 원유가격은 유류할증료 변동을 참고하기 위한 간접 지표입니다.
- 환율, 항공사 정책, 거리 구간, 발권일 기준 고시금액은 반영하지 않습니다.
- MVP 범위는 일본행 항공권을 고민하는 사용자를 위한 참고 서비스입니다.

## 데이터 파일

- 원본 파일: `data/opinet_raw.csv`
- 정규화 파일: `data/opinet_full.csv`
- 앱 입력 파일: `data/opinet_full.csv`
- 환율 파일: `data/usd_krw.csv`

## 정규화 스크립트

원본 CSV를 다시 정규화하려면 아래 명령을 실행합니다.

```bash
python normalize_opinet.py
```

정규화 스크립트는 다음 작업을 수행합니다.

- CSV 인코딩을 UTF-8 기준으로 정규화합니다.
- 날짜를 `YYYY-MM-DD` 형식으로 변환합니다.
- `0.00` 값은 결측치로 처리합니다.
- `data/opinet_full.csv`를 생성합니다.
- 검증 결과로 `meta.firstDate`, `meta.lastDate`, 전체 행 수, 유효한 날짜 개수, 누락/깨진 날짜 개수, 중복 날짜 개수를 출력합니다.

## 실행

로컬 실행:

```bash
python app.py
```

정규화 확인:

```bash
python normalize_opinet.py
python -m py_compile app.py normalize_opinet.py
```

NPM 기반 검증이 필요하면 다음 명령도 사용할 수 있습니다.

```bash
npm install
npm run build
npm run lint
```

## Vercel 배포

- `app.py`의 Flask 앱을 기준으로 배포합니다.
- `vercel.json`은 Python 함수 설정을 사용합니다.
- `data/opinet_full.csv`가 리포지토리에 포함되어 있어야 앱이 안정적으로 동작합니다.

