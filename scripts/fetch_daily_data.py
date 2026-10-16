#!/usr/bin/env python3
"""Download daily market data from Yahoo Finance (yfinance) and FRED, and look up
stock/index-linked Binance USDⓈ-M perpetuals.

Failures are skipped and recorded in data_daily/fetch_status.json, which
scripts/write_manifest.py turns into data_daily/MANIFEST.md.

Usage:
    pip install yfinance
    python3 scripts/fetch_daily_data.py --out-dir data_daily
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

YAHOO_TICKERS = (
    "QQQ SPY QLD TQQQ SSO UPRO GLD TLT IEF USO FXE ^NDX ^GSPC "
    "069500.KS 122630.KS 133690.KS 360750.KS NQ=F ES=F GC=F CL=F ZN=F 6E=F BTC=F "
    "KRW=X EURUSD=X JPY=X"
).split()
YAHOO_COLS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
    "Dividends": "dividends", "Stock Splits": "stock_splits",
}
FRED_IDS = ["DTB3", "DEXKOUS", "DEXUSEU", "DEXJPUS", "FEDFUNDS"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
# Public archive bucket listing; used only when exchangeInfo is unreachable.
ARCHIVE_LIST_URL = ("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
                    "?prefix=data/futures/um/monthly/klines/{}/1h/")
STOCK_EXAMPLES = ["QQQUSDT", "SPYUSDT"]
TRADFI_HINTS = ("tradfi", "stock", "equity", "index", "etf")


def http_get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "daily-data-fetch/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def error_text(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code} {e.reason}".strip()
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- Yahoo

def fetch_yahoo(ticker: str, out_dir: Path, retries: int = 3) -> dict:
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError

    yf.config.debug.hide_exceptions = False  # raise instead of printing and returning empty
    for attempt in range(retries + 1):
        try:
            df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False,
                                           actions=True)
            break
        except YFRateLimitError as e:
            if attempt == retries:
                return {"status": "failed", "error": error_text(e)}
            time.sleep(10 * 2 ** attempt)
        except Exception as e:  # missing ticker, no prices, network error
            return {"status": "failed", "error": error_text(e)}
    if df.empty:
        return {"status": "failed", "error": "empty response"}
    absent = [c for c in YAHOO_COLS if c not in df.columns]
    if absent:
        return {"status": "failed", "error": f"missing columns {absent}"}

    out = df[list(YAHOO_COLS)].rename(columns=YAHOO_COLS)
    out.insert(0, "date", df.index.strftime("%Y-%m-%d"))  # exchange-local trading date
    path = out_dir / "yahoo" / f"{ticker}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return {"status": "ok", "path": path.as_posix(), "timezone": str(df.index.tz),
            "dropped_columns": [c for c in df.columns if c not in YAHOO_COLS]}


# ---------------------------------------------------------------- FRED

def fetch_fred(series_id: str, out_dir: Path) -> dict:
    try:
        data = http_get(FRED_URL.format(series_id))
    except Exception as e:
        return {"status": "failed", "error": error_text(e)}
    text = data.decode("utf-8-sig")
    header = text.splitlines()[0].strip().split(",") if text else []
    if len(header) != 2 or header[1] != series_id:
        return {"status": "failed", "error": f"unexpected response header {header[:3]}"}
    path = out_dir / "fred" / f"{series_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"status": "ok", "path": path.as_posix()}


# ---------------------------------------------------------------- Binance

def archive_months(symbol: str) -> list[str]:
    xml = http_get(ARCHIVE_LIST_URL.format(symbol)).decode()
    return sorted(re.findall(rf"{symbol}-1h-(\d{{4}}-\d{{2}})\.zip</Key>", xml))


def find_stock_perps() -> dict:
    """Stock/index-linked USDⓈ-M symbols with their onboardDate (public endpoint, no auth)."""
    result = {"exchange_info": {"url": EXCHANGE_INFO_URL}, "symbols": []}
    try:
        info = json.loads(http_get(EXCHANGE_INFO_URL, retries=1))
    except Exception as e:
        result["exchange_info"].update(status="failed", error=error_text(e))
        info = None

    if info is not None:
        result["exchange_info"]["status"] = "ok"
        for s in info.get("symbols", []):
            tags = " ".join([s.get("contractType", ""), s.get("underlyingType", ""),
                             *s.get("underlyingSubType", [])]).lower()
            if s["symbol"] in STOCK_EXAMPLES or any(h in tags for h in TRADFI_HINTS):
                onboard = datetime.fromtimestamp(s["onboardDate"] / 1000, timezone.utc)
                result["symbols"].append({
                    "symbol": s["symbol"], "source": "exchangeInfo",
                    "onboardDate": onboard.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "contractType": s.get("contractType"),
                    "underlyingType": s.get("underlyingType"),
                    "underlyingSubType": s.get("underlyingSubType"),
                    "status": s.get("status"),
                })
    else:
        # Without exchangeInfo there is no metadata to classify symbols, so only the
        # examples named in the request are checked against the public archive.
        for sym in STOCK_EXAMPLES:
            try:
                months = archive_months(sym)
            except Exception as e:
                result["symbols"].append({"symbol": sym, "source": "archive listing",
                                          "error": error_text(e)})
                continue
            if months:
                result["symbols"].append({"symbol": sym, "source": "archive listing",
                                          "onboardDate": None, "archive_months": months})
    return result


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("data_daily"))
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between Yahoo requests")
    args = ap.parse_args()

    status = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "yahoo": {}, "fred": {}}
    for t in YAHOO_TICKERS:
        status["yahoo"][t] = res = fetch_yahoo(t, args.out_dir)
        print(f"yahoo {t}: {res['status']} {res.get('error', '')}", file=sys.stderr)
        time.sleep(args.pause)
    for sid in FRED_IDS:
        status["fred"][sid] = res = fetch_fred(sid, args.out_dir)
        print(f"fred {sid}: {res['status']} {res.get('error', '')}", file=sys.stderr)
    status["binance_stock_perps"] = perps = find_stock_perps()
    print(f"binance exchangeInfo: {perps['exchange_info']}", file=sys.stderr)
    print(f"stock/index-linked symbols: {[s['symbol'] for s in perps['symbols']]}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "fetch_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
