#!/usr/bin/env python3
"""Write data_daily/MANIFEST.md: per-file row counts, first/last dates,
duplicates, missing values and the list of failed downloads.

Reads data_daily/fetch_status.json (from scripts/fetch_daily_data.py), the CSVs
under data_daily/ and the Binance CSVs under data/.

Usage:
    python3 scripts/write_manifest.py --daily-dir data_daily --binance-dir data --binance-end 2026-09
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def max_gap(dates: pd.Series) -> str:
    d = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    if len(d) < 2:
        return "-"
    diffs = d.diff().dt.days
    i = int(diffs.idxmax())
    return f"{int(diffs[i])} ({d[i - 1]:%Y-%m-%d}→{d[i]:%Y-%m-%d})"


def yahoo_rows(daily: Path, status: dict) -> list[str]:
    out = []
    for ticker, st in status["yahoo"].items():
        if st["status"] != "ok":
            continue
        path = daily / "yahoo" / f"{ticker}.csv"
        df = pd.read_csv(path, dtype={"date": str})
        empty = int(df[["open", "high", "low", "close", "adj_close", "volume"]].isna().any(axis=1).sum())
        out.append(f"| `yahoo/{path.name}` | {len(df):,} | {df.date.iloc[0]} | {df.date.iloc[-1]} | "
                   f"{int(df.date.duplicated().sum())} | {empty} | {max_gap(df.date)} | {st.get('timezone', '')} |")
    return out


def fred_rows(daily: Path, status: dict) -> list[str]:
    out = []
    for sid, st in status["fred"].items():
        if st["status"] != "ok":
            continue
        path = daily / "fred" / f"{sid}.csv"
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        date, value = df.columns[0], df.columns[1]
        empty = int(df[value].isin(["", "."]).sum())
        out.append(f"| `fred/{path.name}` | {len(df):,} | {df[date].iloc[0]} | {df[date].iloc[-1]} | "
                   f"{int(df[date].duplicated().sum())} | {empty} | {max_gap(df[date])} |")
    return out


def kline_row(path: Path, end: pd.Timestamp) -> str:
    df = pd.read_csv(path, usecols=["open_time", "count"])
    t = pd.to_datetime(df.open_time, utc=True)
    expected = int((end - t.min()) / pd.Timedelta(hours=1)) + 1
    missing = expected - t.nunique()
    return (f"| `{path.as_posix()}` | {len(df):,} | {df.open_time.iloc[0][:16]}Z | {df.open_time.iloc[-1][:16]}Z | "
            f"{int(t.duplicated().sum())} | {missing} | {int((df['count'] == 0).sum())} |")


def funding_row(path: Path) -> str:
    df = pd.read_csv(path)
    ft = pd.to_datetime(df.funding_time, utc=True)
    hours = df.funding_interval_hours
    delta_h = ft.diff().dt.total_seconds() / 3600
    same_interval = hours.eq(hours.shift())
    missing = int(((delta_h[same_interval] / hours[same_interval]) - 1).clip(lower=0).sum())
    changes = int((~same_interval).sum() - 1)
    return (f"| `{path.as_posix()}` | {len(df):,} | {df.funding_time.iloc[0][:16]}Z | "
            f"{df.funding_time.iloc[-1][:16]}Z | {int(df.calc_time.duplicated().sum())} | "
            f"{int(ft.duplicated().sum())} | {missing} | {changes} | "
            f"{', '.join(map(str, sorted(hours.unique())))} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--daily-dir", type=Path, default=Path("data_daily"))
    ap.add_argument("--binance-dir", type=Path, default=Path("data"))
    ap.add_argument("--binance-end", default="2026-09", help="last requested month of the Binance build")
    args = ap.parse_args()

    status = json.loads((args.daily_dir / "fetch_status.json").read_text())
    end = (pd.Timestamp(args.binance_end + "-01", tz="UTC") + pd.offsets.MonthBegin(1)
           - pd.Timedelta(hours=1))
    perps = status["binance_stock_perps"]
    dropped = sorted({c for st in status["yahoo"].values() for c in st.get("dropped_columns", [])})
    symbols = " ".join(p.name[:-len("_1h.csv")] for p in sorted(args.binance_dir.glob("*_1h.csv")))

    L = ["# MANIFEST", "",
         f"- 다운로드 완료 시각 (UTC): {status['generated']}",
         "- 재생성 순서:",
         "  1. `python3 scripts/fetch_daily_data.py --out-dir data_daily`",
         f"  2. `python3 scripts/build_binance_um_dataset.py --symbols {symbols} "
         f"--start 2020-01 --end {args.binance_end} --interval 1h --out-dir data`",
         "  3. `python3 scripts/write_manifest.py`", ""]

    L += ["## 실패 목록", ""]
    failures = [f"- yahoo `{t}`: {st['error']}" for t, st in status["yahoo"].items() if st["status"] != "ok"]
    failures += [f"- FRED `{s}`: {st['error']}" for s, st in status["fred"].items() if st["status"] != "ok"]
    ei = perps["exchange_info"]
    if ei.get("status") != "ok":
        failures.append(f"- Binance `{ei['url']}`: {ei.get('error')} "
                        "(접속 지역 제한으로 거부됨. 우회하지 않았다)")
    failures += [f"- Binance 아카이브 목록 `{s['symbol']}`: {s['error']}" for s in perps["symbols"] if "error" in s]
    L += failures or ["- 없음"]
    L.append("")

    L += ["## 1) Yahoo Finance (yfinance) 일봉", "",
          "`period=\"max\", interval=\"1d\", auto_adjust=False, actions=True`. "
          "열: `date,open,high,low,close,adj_close,volume,dividends,stock_splits`. "
          "`date`는 거래소 현지 기준 날짜다 (시간대 열 참고).", ""]
    if dropped:
        L += [f"- 요청한 열에 없는 yfinance 열은 제외했다: {', '.join(f'`{c}`' for c in dropped)}"]
    L += [f"- 다운로드 시각({status['generated']})에 거래 중이던 시장(FX `=X`, 선물 `=F` 등)은 "
          "마지막 행이 미완성 봉일 수 있다.",
          "- '빈 값 행' = open/high/low/close/adj_close/volume 중 빈 값이 있는 행. "
          "휴장일 달력과 대조하지 않았으므로 날짜 누락은 '최대 간격(일)'으로만 표시한다.", "",
          "| 파일 | 행 수 | 첫 날짜 | 마지막 날짜 | 중복 날짜 | 빈 값 행 | 최대 간격(일) | 시간대 |",
          "|---|---:|---|---|---:|---:|---|---|"]
    L += yahoo_rows(args.daily_dir, status)
    L.append("")

    L += ["## 2) FRED", "",
          "`https://fred.stlouisfed.org/graph/fredgraph.csv?id=ID` 응답을 그대로 저장했다. "
          "'빈 값' = 값이 비어 있거나 `.`인 행 (FRED의 휴일·미발표 표시).", "",
          "| 파일 | 행 수 | 첫 날짜 | 마지막 날짜 | 중복 날짜 | 빈 값 | 최대 간격(일) |",
          "|---|---:|---|---|---:|---:|---|"]
    L += fred_rows(args.daily_dir, status)
    L.append("")

    L += ["## 3) Binance USDⓈ-M 1h 캔들 · 펀딩비", "",
          "`scripts/build_binance_um_dataset.py`로 생성 (2020-01 ~ " + args.binance_end + " 요청, "
          "상장 전 월은 아카이브에 없어 건너뜀). 상세 점검과 원본 중복 제거 수는 `data/README.md`에 있다.", "",
          f"- 누락 봉 = 첫 봉부터 {end:%Y-%m-%dT%H:%M}Z까지의 1h 슬롯 중 없는 수.",
          "- 체결 0건 봉 = 행은 있지만 `count = 0` (거래소 점검 등으로 생긴 자리채움 봉).", "",
          "| 파일 | 행 수 | 첫 시각 | 마지막 시각 | 중복 | 누락 봉 | 체결 0건 봉 |",
          "|---|---:|---|---|---:|---:|---:|"]
    klines = sorted(args.binance_dir.glob("*_1h.csv"))
    L += [kline_row(p, end) for p in klines]
    L += ["", "- 누락 회차 = 같은 주기가 이어지는 구간에서 정산 간격 / `funding_interval_hours` - 1의 합.",
          "- 같은 정시 2건 = `calc_time`이 다른 두 기록이 같은 `funding_time`으로 반올림된 경우 (둘 다 남김).", "",
          "| 파일 | 행 수 | 첫 시각 | 마지막 시각 | 중복 (calc_time) | 같은 정시 2건 | 누락 회차 | 주기 변경 | 펀딩 주기(h) |",
          "|---|---:|---|---|---:|---:|---:|---:|---|"]
    L += [funding_row(p) for p in sorted(args.binance_dir.glob("*_funding.csv"))]
    L.append("")

    L += ["## 4) Binance 주식·지수 연동 심볼", ""]
    if ei.get("status") == "ok":
        L.append(f"`{ei['url']}`에서 찾은 심볼:")
    else:
        L += [f"`{ei['url']}` 호출이 실패해 ({ei.get('error')}) 전체 목록과 `onboardDate`를 얻지 못했다. "
              "대신 요청에 예시로 적힌 심볼만 공개 아카이브(data.binance.vision) 목록에서 확인했다. "
              "따라서 아래가 주식·지수 연동 심볼 전체는 아닐 수 있다."]
    L += ["", "| 심볼 | onboardDate | 확인 경로 | 아카이브 월 | 첫 1h 봉 | 첫 펀딩 | 파일 |", "|---|---|---|---|---|---|---|"]
    for s in perps["symbols"]:
        if "error" in s:
            continue
        sym = s["symbol"]
        k, f = args.binance_dir / f"{sym}_1h.csv", args.binance_dir / f"{sym}_funding.csv"
        first_k = pd.read_csv(k, nrows=1).open_time[0][:16] + "Z" if k.exists() else "-"
        first_f = pd.read_csv(f, nrows=1).funding_time[0][:16] + "Z" if f.exists() else "-"
        months = s.get("archive_months") or []
        month_txt = f"{months[0]} ~ {months[-1]}" if months else "-"
        files = ", ".join(f"`{p.as_posix()}`" for p in (k, f) if p.exists()) or "없음"
        L.append(f"| {sym} | {s.get('onboardDate') or '확인 불가'} | {s['source']} | {month_txt} | "
                 f"{first_k} | {first_f} | {files} |")
    L.append("")

    (args.daily_dir / "MANIFEST.md").write_text("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
