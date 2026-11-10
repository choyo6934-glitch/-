"""원자료 CSV를 읽어 캐시(parquet)로 저장한다.

- 체결: 4개 파일을 이어 붙이고 파일 순서(row_order)를 보존한다.
- 지갑: 빈 행을 버리고, 깨진 timestamp 대신 date 열을 쓴다.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from config import AOA_RAW, CACHE

EXEC_COLS = {
    "execid": "string", "orderid": "string", "symbol": "string", "side": "string",
    "lastqty": "float64", "lastpx": "float64", "lastliquidityind": "string",
    "orderqty": "float64", "price": "float64", "displayqty": "float64", "stoppx": "float64",
    "pegpricetype": "string", "currency": "string", "settlcurrency": "string",
    "exectype": "string", "ordtype": "string", "timeinforce": "string", "execinst": "string",
    "ordstatus": "string", "triggered": "string", "leavesqty": "float64", "cumqty": "float64",
    "avgpx": "float64", "commission": "float64", "text": "string", "trdmatchid": "string",
    "execcost": "float64", "execcomm": "float64", "homenotional": "float64",
    "foreignnotional": "float64", "transacttime": "string", "timestamp": "string",
}


def load_executions(refresh: bool = False) -> pd.DataFrame:
    out = CACHE / "executions.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    files = sorted(glob.glob(str(AOA_RAW / "aoa-execution-*.csv")))
    assert len(files) == 4, files
    parts = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=list(EXEC_COLS), dtype=EXEC_COLS, encoding="utf-8-sig")
        d["file_no"] = i
        d["file_row"] = np.arange(len(d))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["t"] = pd.to_datetime(df["transacttime"], format="mixed", utc=True)
    df["ts"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["row_order"] = np.arange(len(df))
    # execcost/execcomm는 사토시 정수
    for c in ("execcost", "execcomm"):
        assert (df[c].dropna() == df[c].dropna().round()).all(), c
    df.to_parquet(out, index=False)
    return df


def load_wallet() -> pd.DataFrame:
    f = glob.glob(str(AOA_RAW / "aoa-wallet-*.csv"))[0]
    w = pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    w = w[w["date"].str.strip() != ""].copy()
    w["file_row"] = np.arange(len(w))
    w["date"] = pd.to_datetime(w["date"]).dt.tz_localize("UTC")
    w["amount"] = w["amount"].astype(float)
    w["walletbalance"] = pd.to_numeric(w["walletbalance"], errors="coerce")
    return w.reset_index(drop=True)


if __name__ == "__main__":
    df = load_executions(refresh=True)
    print(df.shape, df["t"].min(), df["t"].max())
    w = load_wallet()
    print(w.shape)
    print(w.groupby(["transacttype", "transactstatus"]).size())
