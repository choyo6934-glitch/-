"""가격 데이터 로더. 봉은 시작 시각(open_time, UTC)으로 색인한다: [T, T+주기)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BINANCE_UM, BINANCE_SPOT, BITSTAMP_1M, CACHE, DAILY


def bitstamp_1m(years=range(2017, 2023)) -> pd.DataFrame:
    out = CACHE / f"bitstamp_1m_{min(years)}_{max(years)}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    parts = []
    for y in years:
        d = pd.read_csv(BITSTAMP_1M / f"BTCUSD_bitstamp_1min_{y}.csv.gz")
        d["t"] = pd.to_datetime(d["datetime_utc"], utc=True)
        parts.append(d.drop(columns=["datetime_utc"]))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("t").sort_values("t").set_index("t")
    df.to_parquet(out)
    return df


def resample_ohlcv(m: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = m.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    # 빈 봉은 직전 종가로 채운다(거래 없음)
    o["close"] = o["close"].ffill()
    for c in ("open", "high", "low"):
        o[c] = o[c].fillna(o["close"])
    o["volume"] = o["volume"].fillna(0)
    return o


def bitstamp_bars(rule: str = "1h") -> pd.DataFrame:
    out = CACHE / f"bitstamp_{rule}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    o = resample_ohlcv(bitstamp_1m(), rule)
    o.to_parquet(out)
    return o


def binance_um_1h(symbol: str) -> pd.DataFrame:
    d = pd.read_csv(BINANCE_UM / f"{symbol}_1h.csv")
    d["t"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.set_index("t")[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    return d


def binance_funding(symbol: str) -> pd.Series:
    f = pd.read_csv(BINANCE_UM / f"{symbol}_funding.csv")
    f["t"] = pd.to_datetime(f["funding_time"], utc=True)
    return f.set_index("t")["last_funding_rate"].astype(float)


def binance_spot_1h(symbol: str) -> pd.DataFrame:
    d = pd.read_csv(BINANCE_SPOT / f"{symbol}_1h.csv")
    d["t"] = pd.to_datetime(d["open_time"], utc=True)
    d = d.set_index("t")[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    # 현물 아카이브에 빠진 시간이 있으면 직전 종가로 채운 봉을 넣는다(거래 없음)
    full = pd.date_range(d.index[0], d.index[-1], freq="1h")
    if len(full) != len(d):
        d = d.reindex(full)
        d["close"] = d["close"].ffill()
        for c in ("open", "high", "low"):
            d[c] = d[c].fillna(d["close"])
        d[["volume", "quote_volume"]] = d[["volume", "quote_volume"]].fillna(0)
        d.index.name = "t"
    return d


def daily_stock(ticker: str) -> pd.DataFrame:
    """Yahoo 일봉. adj_close/close 비율로 OHLC 전체를 보정(배당·분할 반영). 빈 값·거래량 0 행은 버린다."""
    d = pd.read_csv(DAILY / f"{ticker}.csv")
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    d = d.dropna(subset=["open", "high", "low", "close", "adj_close"])
    d = d[(d["volume"] > 0) & (d["open"] > 0)]
    k = d["adj_close"] / d["close"]
    out = pd.DataFrame({c: d[c] * k for c in ("open", "high", "low", "close")})
    out["volume"] = d["volume"]
    out["raw_close"] = d["close"]
    # 고가·저가가 시가·종가를 감싸도록 정리(데이터 오류 방지)
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    return out


if __name__ == "__main__":
    h = bitstamp_bars("1h")
    q = bitstamp_bars("15min")
    print(h.index.min(), h.index.max(), len(h), h.close.isna().sum())
    print(q.index.min(), q.index.max(), len(q))
    for s in ("BTCUSDT", "ETHUSDT"):
        print(s, binance_um_1h(s).shape, binance_spot_1h(s).shape, binance_funding(s).shape)
    for t in ("QQQ", "SPY", "069500.KS", "005930.KS"):
        d = daily_stock(t)
        print(t, d.index.min().date(), d.index.max().date(), len(d))
