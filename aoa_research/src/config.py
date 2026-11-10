"""경로와 공통 상수. 원자료 위치는 환경변수로 바꿀 수 있다."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # aoa_research/
REPO = ROOT.parent                                    # 저장소 루트
RESULTS = ROOT / "results"
CACHE = Path(os.environ.get("AOA_CACHE", "/home/user/aoa_cache"))

# 원자료(재배포하지 않음): 공개 압축 파일을 푼 폴더
AOA_RAW = Path(os.environ.get("AOA_RAW", "/home/user/aoa_raw/data"))
# 가격 데이터
BINANCE_UM = Path(os.environ.get("BINANCE_UM", str(REPO / "data")))                 # 선물 1h + 펀딩
BINANCE_SPOT = Path(os.environ.get("BINANCE_SPOT", "/home/user/mktdata/spot"))      # 현물 1h
DAILY = Path(os.environ.get("DAILY_DIR", "/home/user/mktdata/daily"))               # 주식 일봉
BITSTAMP_1M = Path(os.environ.get("BITSTAMP_1M",
                   "/home/user/alessiostomeo/btc-cycle-data/bitstamp/1min"))

SAT = 1e8
ZERO_UUID = "00000000-0000-0000-0000-000000000000"

# 2단계 분할 (PREREG 2절)
SPLITS = {
    "y2018": ("2018-03-05", "2019-01-01"),
    "train": ("2019-01-01", "2021-01-01"),
    "valid": ("2021-01-01", "2021-07-01"),
    "sealed": ("2021-07-01", "2022-01-01"),
}
# 4단계 구간 (PREREG 3절)
BT_WINDOWS_COIN = {
    "dev": ("2020-02-01", "2021-07-01"),
    "sealed": ("2021-07-01", "2022-01-01"),
    "oos": ("2022-01-01", "2026-10-01"),
}
BT_WINDOWS_STOCK = {
    "pre2022": ("2010-01-01", "2022-01-01"),
    "oos": ("2022-01-01", "2026-10-01"),
}
SEED = 20261110

for p in (RESULTS, CACHE):
    p.mkdir(parents=True, exist_ok=True)
