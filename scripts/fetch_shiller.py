#!/usr/bin/env python3
"""Download Robert Shiller's monthly U.S. stock market data (ie_data.xls) and save
data_daily/shiller/ie_data.xls plus a tidy CSV data_daily/shiller/shiller_monthly.csv
with columns: date, price, dividend, earnings, cpi, gs10, cape.

Usage:
    pip install pandas xlrd
    python3 scripts/fetch_shiller.py --out-dir data_daily
    # 자동 다운로드가 막히면 shillerdata.com 에서 ie_data.xls 를 받아 경로를 주면 변환만 한다
    python3 scripts/fetch_shiller.py --out-dir data_daily --xls ~/Downloads/ie_data.xls
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fetch_daily_data import error_text, http_get  # same folder (scripts/)

URLS = ["http://www.econ.yale.edu/~shiller/data/ie_data.xls",
        "https://www.econ.yale.edu/~shiller/data/ie_data.xls"]


def parse(xls: bytes) -> pd.DataFrame:
    return parse_raw(pd.read_excel(io.BytesIO(xls), sheet_name="Data", header=None))


def parse_raw(raw: pd.DataFrame) -> pd.DataFrame:
    hdr = next(i for i in range(40) if str(raw.iat[i, 0]).strip() == "Date")
    head = [str(x).strip() for x in raw.iloc[hdr]]

    def find(*names: str, contains: str | None = None) -> int:
        for j, h in enumerate(head):
            if h in names or (contains and contains in h):
                return j
        raise KeyError(f"column {names or contains} not found in header {head}")

    cols = {"price": find("P"), "dividend": find("D"), "earnings": find("E"), "cpi": find("CPI"),
            "gs10": find(contains="GS10"), "cape": find("CAPE", contains="P/E10")}
    body = raw.iloc[hdr + 1:]
    d = pd.to_numeric(body[0], errors="coerce")
    ok = d.notna() & (d > 1800)
    year = d[ok].astype(int)
    month = ((d[ok] - year) * 100).round().astype(int)
    out = pd.DataFrame({"date": [f"{y:04d}-{m:02d}-01" for y, m in zip(year, month)]})
    for k, j in cols.items():
        out[k] = pd.to_numeric(body.loc[ok, j], errors="coerce").to_numpy()
    if not month.between(1, 12).all():
        raise ValueError("unexpected date format in column 'Date'")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("data_daily"))
    ap.add_argument("--xls", type=Path, help="already downloaded ie_data.xls (skip download)")
    args = ap.parse_args()
    out = args.out_dir / "shiller"
    out.mkdir(parents=True, exist_ok=True)
    status = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    data = None
    if args.xls:
        data = args.xls.expanduser().read_bytes(); status["source"] = str(args.xls)
    else:
        for url in URLS:
            try:
                data = http_get(url); status["source"] = url
                break
            except Exception as e:
                status.setdefault("errors", []).append(f"{url}: {error_text(e)}")
    if data is None:
        status["status"] = "failed"
        (out / "fetch_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        print("Shiller download failed. Download ie_data.xls from shillerdata.com and rerun with --xls PATH", file=sys.stderr)
        return 1
    (out / "ie_data.xls").write_bytes(data)
    try:
        df = parse(data)
    except Exception as e:
        status.update(status="failed", error=error_text(e))
        (out / "fetch_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        print(f"parse failed: {e} (raw file kept at {out / 'ie_data.xls'})", file=sys.stderr)
        return 1
    df.to_csv(out / "shiller_monthly.csv", index=False)
    last = {c: df.loc[df[c].notna(), "date"].iloc[-1] for c in ("price", "dividend", "earnings", "cpi", "gs10", "cape")}
    status.update(status="ok", rows=len(df), first=df.date.iloc[0], last_non_empty=last)
    (out / "fetch_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(f"shiller ok: {len(df)} rows, last non-empty {last}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
