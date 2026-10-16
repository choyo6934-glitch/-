# MANIFEST

- 다운로드 완료 시각 (UTC): 2026-10-15T23:54:28Z
- 재생성 순서:
  1. `python3 scripts/fetch_daily_data.py --out-dir data_daily`
  2. `python3 scripts/build_binance_um_dataset.py --symbols ADAUSDT BNBUSDT BTCUSDT DOGEUSDT ETHUSDT LINKUSDT QQQUSDT SOLUSDT SPYUSDT XRPUSDT --start 2020-01 --end 2026-09 --interval 1h --out-dir data`
  3. `python3 scripts/write_manifest.py`

## 실패 목록

- Binance `https://fapi.binance.com/fapi/v1/exchangeInfo`: HTTP 451 (접속 지역 제한으로 거부됨. 우회하지 않았다)

## 1) Yahoo Finance (yfinance) 일봉

`period="max", interval="1d", auto_adjust=False, actions=True`. 열: `date,open,high,low,close,adj_close,volume,dividends,stock_splits`. `date`는 거래소 현지 기준 날짜다 (시간대 열 참고).

- 요청한 열에 없는 yfinance 열은 제외했다: `Capital Gains`
- 다운로드 시각(2026-10-15T23:54:28Z)에 거래 중이던 시장(FX `=X`, 선물 `=F` 등)은 마지막 행이 미완성 봉일 수 있다.
- '빈 값 행' = open/high/low/close/adj_close/volume 중 빈 값이 있는 행. 휴장일 달력과 대조하지 않았으므로 날짜 누락은 '최대 간격(일)'으로만 표시한다.

| 파일 | 행 수 | 첫 날짜 | 마지막 날짜 | 중복 날짜 | 빈 값 행 | 최대 간격(일) | 시간대 |
|---|---:|---|---|---:|---:|---|---|
| `yahoo/QQQ.csv` | 6,944 | 1999-03-10 | 2026-10-15 | 0 | 0 | 7 (2001-09-10→2001-09-17) | America/New_York |
| `yahoo/SPY.csv` | 8,486 | 1993-01-29 | 2026-10-15 | 0 | 0 | 7 (2001-09-10→2001-09-17) | America/New_York |
| `yahoo/QLD.csv` | 5,112 | 2006-06-21 | 2026-10-15 | 0 | 0 | 5 (2006-12-29→2007-01-03) | America/New_York |
| `yahoo/TQQQ.csv` | 4,195 | 2010-02-11 | 2026-10-15 | 0 | 0 | 5 (2012-10-26→2012-10-31) | America/New_York |
| `yahoo/SSO.csv` | 5,112 | 2006-06-21 | 2026-10-15 | 0 | 0 | 5 (2006-12-29→2007-01-03) | America/New_York |
| `yahoo/UPRO.csv` | 4,354 | 2009-06-25 | 2026-10-15 | 0 | 0 | 5 (2012-10-26→2012-10-31) | America/New_York |
| `yahoo/GLD.csv` | 5,511 | 2004-11-18 | 2026-10-15 | 0 | 0 | 5 (2006-12-29→2007-01-03) | America/New_York |
| `yahoo/TLT.csv` | 6,093 | 2002-07-30 | 2026-10-15 | 0 | 0 | 5 (2006-12-29→2007-01-03) | America/New_York |
| `yahoo/IEF.csv` | 6,093 | 2002-07-30 | 2026-10-15 | 0 | 0 | 5 (2006-12-29→2007-01-03) | America/New_York |
| `yahoo/USO.csv` | 5,162 | 2006-04-10 | 2026-10-15 | 0 | 0 | 5 (2006-12-29→2007-01-03) | America/New_York |
| `yahoo/FXE.csv` | 5,243 | 2005-12-12 | 2026-10-15 | 0 | 0 | 5 (2006-12-29→2007-01-03) | America/New_York |
| `yahoo/^NDX.csv` | 10,340 | 1985-10-01 | 2026-10-15 | 0 | 0 | 7 (2001-09-10→2001-09-17) | America/New_York |
| `yahoo/^GSPC.csv` | 24,815 | 1927-12-30 | 2026-10-15 | 0 | 0 | 12 (1933-03-03→1933-03-15) | America/New_York |
| `yahoo/069500.KS.csv` | 4,291 | 2007-01-29 | 2026-10-15 | 0 | 5 | 261 (2008-07-30→2009-04-17) | Asia/Seoul |
| `yahoo/122630.KS.csv` | 4,064 | 2010-02-22 | 2026-10-15 | 0 | 0 | 11 (2017-09-29→2017-10-10) | Asia/Seoul |
| `yahoo/133690.KS.csv` | 3,901 | 2010-10-18 | 2026-10-15 | 0 | 1 | 11 (2017-09-29→2017-10-10) | Asia/Seoul |
| `yahoo/360750.KS.csv` | 1,492 | 2020-08-07 | 2026-10-15 | 0 | 1 | 8 (2025-10-02→2025-10-10) | Asia/Seoul |
| `yahoo/NQ=F.csv` | 6,584 | 2000-09-18 | 2026-10-15 | 0 | 0 | 6 (2001-09-11→2001-09-17) | America/New_York |
| `yahoo/ES=F.csv` | 6,584 | 2000-09-18 | 2026-10-15 | 0 | 0 | 6 (2001-09-11→2001-09-17) | America/New_York |
| `yahoo/GC=F.csv` | 6,557 | 2000-08-30 | 2026-10-15 | 0 | 0 | 5 (2000-11-22→2000-11-27) | America/New_York |
| `yahoo/CL=F.csv` | 6,565 | 2000-08-23 | 2026-10-15 | 0 | 0 | 5 (2000-11-22→2000-11-27) | America/New_York |
| `yahoo/ZN=F.csv` | 6,546 | 2000-09-21 | 2026-10-15 | 0 | 0 | 12 (2008-06-19→2008-07-01) | America/New_York |
| `yahoo/6E=F.csv` | 6,586 | 2000-09-12 | 2026-10-15 | 0 | 0 | 5 (2012-10-26→2012-10-31) | America/New_York |
| `yahoo/BTC=F.csv` | 2,221 | 2017-12-18 | 2026-10-15 | 0 | 0 | 4 (2017-12-22→2017-12-26) | America/New_York |
| `yahoo/KRW=X.csv` | 5,934 | 2003-12-01 | 2026-10-16 | 0 | 0 | 26 (2008-07-31→2008-08-26) | Europe/London |
| `yahoo/EURUSD=X.csv` | 5,936 | 2003-12-01 | 2026-10-16 | 0 | 0 | 18 (2008-08-08→2008-08-26) | Europe/London |
| `yahoo/JPY=X.csv` | 7,770 | 1996-10-30 | 2026-10-16 | 0 | 0 | 18 (2008-08-08→2008-08-26) | Europe/London |

## 2) FRED

`https://fred.stlouisfed.org/graph/fredgraph.csv?id=ID` 응답을 그대로 저장했다. '빈 값' = 값이 비어 있거나 `.`인 행 (FRED의 휴일·미발표 표시).

| 파일 | 행 수 | 첫 날짜 | 마지막 날짜 | 중복 날짜 | 빈 값 | 최대 간격(일) |
|---|---:|---|---|---:|---:|---|
| `fred/DTB3.csv` | 18,988 | 1954-01-04 | 2026-10-14 | 0 | 801 | 3 (1954-01-08→1954-01-11) |
| `fred/DEXKOUS.csv` | 11,870 | 1981-04-13 | 2026-10-09 | 0 | 498 | 3 (1981-04-17→1981-04-20) |
| `fred/DEXUSEU.csv` | 7,245 | 1999-01-04 | 2026-10-09 | 0 | 280 | 3 (1999-01-08→1999-01-11) |
| `fred/DEXJPUS.csv` | 14,550 | 1971-01-04 | 2026-10-09 | 0 | 570 | 3 (1971-01-08→1971-01-11) |
| `fred/FEDFUNDS.csv` | 867 | 1954-07-01 | 2026-09-01 | 0 | 0 | 31 (1954-07-01→1954-08-01) |

## 3) Binance USDⓈ-M 1h 캔들 · 펀딩비

`scripts/build_binance_um_dataset.py`로 생성 (2020-01 ~ 2026-09 요청, 상장 전 월은 아카이브에 없어 건너뜀). 상세 점검과 원본 중복 제거 수는 `data/README.md`에 있다.

- 누락 봉 = 첫 봉부터 2026-09-30T23:00Z까지의 1h 슬롯 중 없는 수.
- 체결 0건 봉 = 행은 있지만 `count = 0` (거래소 점검 등으로 생긴 자리채움 봉).

| 파일 | 행 수 | 첫 시각 | 마지막 시각 | 중복 | 누락 봉 | 체결 0건 봉 |
|---|---:|---|---|---:|---:|---:|
| `data/ADAUSDT_1h.csv` | 58,432 | 2020-01-31T08:00Z | 2026-09-30T23:00Z | 0 | 0 | 1 |
| `data/BNBUSDT_1h.csv` | 58,192 | 2020-02-10T08:00Z | 2026-09-30T23:00Z | 0 | 0 | 1 |
| `data/BTCUSDT_1h.csv` | 59,160 | 2020-01-01T00:00Z | 2026-09-30T23:00Z | 0 | 0 | 1 |
| `data/DOGEUSDT_1h.csv` | 54,567 | 2020-07-10T09:00Z | 2026-09-30T23:00Z | 0 | 0 | 1 |
| `data/ETHUSDT_1h.csv` | 59,160 | 2020-01-01T00:00Z | 2026-09-30T23:00Z | 0 | 0 | 1 |
| `data/LINKUSDT_1h.csv` | 58,768 | 2020-01-17T08:00Z | 2026-09-30T23:00Z | 0 | 0 | 1 |
| `data/QQQUSDT_1h.csv` | 4,259 | 2026-04-06T13:00Z | 2026-09-30T23:00Z | 0 | 0 | 0 |
| `data/SOLUSDT_1h.csv` | 52,865 | 2020-09-14T07:00Z | 2026-09-30T23:00Z | 0 | 120 | 1 |
| `data/SPYUSDT_1h.csv` | 4,259 | 2026-04-06T13:00Z | 2026-09-30T23:00Z | 0 | 0 | 0 |
| `data/XRPUSDT_1h.csv` | 58,912 | 2020-01-06T08:00Z | 2026-09-30T23:00Z | 0 | 120 | 1 |

- 누락 회차 = 같은 주기가 이어지는 구간에서 정산 간격 / `funding_interval_hours` - 1의 합.
- 같은 정시 2건 = `calc_time`이 다른 두 기록이 같은 `funding_time`으로 반올림된 경우 (둘 다 남김).

| 파일 | 행 수 | 첫 시각 | 마지막 시각 | 중복 (calc_time) | 같은 정시 2건 | 누락 회차 | 주기 변경 | 펀딩 주기(h) |
|---|---:|---|---|---:|---:|---:|---:|---|
| `data/ADAUSDT_funding.csv` | 7,339 | 2020-01-19T16:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 0 | 8 |
| `data/BNBUSDT_funding.csv` | 7,274 | 2020-02-10T08:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 0 | 8 |
| `data/BTCUSDT_funding.csv` | 7,395 | 2020-01-01T00:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 0 | 8 |
| `data/DOGEUSDT_funding.csv` | 6,821 | 2020-07-10T08:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 0 | 8 |
| `data/ETHUSDT_funding.csv` | 7,395 | 2020-01-01T00:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 0 | 8 |
| `data/LINKUSDT_funding.csv` | 7,346 | 2020-01-17T08:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 0 | 8 |
| `data/QQQUSDT_funding.csv` | 556 | 2026-04-03T08:00Z | 2026-09-30T16:00Z | 0 | 2 | 0 | 4 | 1, 8 |
| `data/SOLUSDT_funding.csv` | 6,700 | 2020-09-13T16:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 3 | 2, 4, 8 |
| `data/SPYUSDT_funding.csv` | 556 | 2026-04-03T08:00Z | 2026-09-30T16:00Z | 0 | 2 | 0 | 4 | 1, 8 |
| `data/XRPUSDT_funding.csv` | 7,379 | 2020-01-06T08:00Z | 2026-09-30T16:00Z | 0 | 0 | 0 | 0 | 8 |

## 4) Binance 주식·지수 연동 심볼

`https://fapi.binance.com/fapi/v1/exchangeInfo` 호출이 실패해 (HTTP 451) 전체 목록과 `onboardDate`를 얻지 못했다. 대신 요청에 예시로 적힌 심볼만 공개 아카이브(data.binance.vision) 목록에서 확인했다. 따라서 아래가 주식·지수 연동 심볼 전체는 아닐 수 있다.

| 심볼 | onboardDate | 확인 경로 | 아카이브 월 | 첫 1h 봉 | 첫 펀딩 | 파일 |
|---|---|---|---|---|---|---|
| QQQUSDT | 확인 불가 | archive listing | 2026-04 ~ 2026-09 | 2026-04-06T13:00Z | 2026-04-03T08:00Z | `data/QQQUSDT_1h.csv`, `data/QQQUSDT_funding.csv` |
| SPYUSDT | 확인 불가 | archive listing | 2026-04 ~ 2026-09 | 2026-04-06T13:00Z | 2026-04-03T08:00Z | `data/SPYUSDT_1h.csv`, `data/SPYUSDT_funding.csv` |
