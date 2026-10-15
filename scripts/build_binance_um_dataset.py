#!/usr/bin/env python3
"""Build per-symbol CSVs from Binance USDⓈ-M futures monthly archives.

Downloads the public monthly zip files (klines + fundingRate) from
data.binance.vision, verifies each against its published SHA256 CHECKSUM,
normalises column names and timestamps, removes duplicates, checks for
missing bars / funding events and writes a README describing the result.

Only static public archive files are fetched; no Binance REST/WebSocket API
(and nothing that needs authentication or places orders) is called.

Usage:
    python3 scripts/build_binance_um_dataset.py \
        --symbols BTCUSDT ETHUSDT --start 2020-01 --end 2026-09 \
        --raw-dir .cache/binance_um --out-dir data
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/monthly"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
KLINE_OUT_COLS = KLINE_COLS[:-1]  # "ignore" carries no information
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]
FUNDING_OUT_COLS = ["funding_time", "calc_time", "funding_interval_hours", "last_funding_rate"]

# Header spellings seen across Binance archive generations, mapped to canonical names.
HEADER_ALIASES = {
    "open time": "open_time", "close time": "close_time",
    "quote asset volume": "quote_volume", "quote_asset_volume": "quote_volume",
    "number of trades": "count", "trades": "count",
    "taker buy base asset volume": "taker_buy_volume",
    "taker_buy_base_asset_volume": "taker_buy_volume",
    "taker buy quote asset volume": "taker_buy_quote_volume",
    "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
    "fundingtime": "calc_time", "funding_time": "calc_time",
    "fundingrate": "last_funding_rate", "funding_rate": "last_funding_rate",
}

US_PER_MS = 1_000
US_PER_HOUR = 3_600_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
INTERVAL_US = {"1m": 60_000_000, "5m": 300_000_000, "15m": 900_000_000,
               "1h": US_PER_HOUR, "4h": 4 * US_PER_HOUR, "1d": 24 * US_PER_HOUR}


# ---------------------------------------------------------------- helpers

def month_range(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_start_us(month: str) -> int:
    y, m = map(int, month.split("-"))
    return int((datetime(y, m, 1, tzinfo=timezone.utc) - EPOCH).total_seconds()) * 1_000_000


def next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y + 1:04d}-01" if m == 12 else f"{y:04d}-{m + 1:02d}"


def to_us(raw: str) -> tuple[int, str]:
    """Return (epoch microseconds, detected unit) for a ms or µs epoch value."""
    v = int(raw)
    if v >= 10**15:   # µs epoch (>= 2001-09 in µs; a ms value would be year 33658+)
        return v, "us"
    if v >= 10**12:   # ms epoch (2001-09 .. 2286)
        return v * US_PER_MS, "ms"
    raise ValueError(f"unrecognised timestamp magnitude: {raw!r}")


def iso(us: int) -> str:
    """UTC ISO-8601 with millisecond precision, e.g. 2020-01-01T00:59:59.999Z."""
    if us % US_PER_MS:
        raise ValueError(f"sub-millisecond timestamp would be truncated: {us}")
    dt = EPOCH + timedelta(microseconds=us)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def compress_months(months: list[str]) -> str:
    """['2020-01','2020-02','2020-04'] -> '2020-01~2020-02, 2020-04'."""
    if not months:
        return "-"
    runs, start, prev = [], months[0], months[0]
    for m in months[1:]:
        if m != next_month(prev):
            runs.append((start, prev))
            start = m
        prev = m
    runs.append((start, prev))
    return ", ".join(a if a == b else f"{a}~{b}" for a, b in runs)


# ---------------------------------------------------------------- download

def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes | None:
    """GET a URL; None on 404. Retries transient failures with backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": "binance-um-archive-builder/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))
    return None


def archive_url(kind: str, symbol: str, month: str, interval: str) -> str:
    if kind == "klines":
        return f"{BASE_URL}/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    return f"{BASE_URL}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"


def archive_path(raw_dir: Path, kind: str, symbol: str, month: str, interval: str) -> Path:
    return raw_dir / kind / symbol / Path(archive_url(kind, symbol, month, interval)).name


def download_one(url: str, dest: Path) -> dict:
    checksum = http_get(url + ".CHECKSUM")
    expected = checksum.decode().split()[0].lower() if checksum else None
    if dest.exists() and expected and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
        return {"status": "ok", "checksum": "verified", "cached": True}
    for _ in range(2):
        data = http_get(url)
        if data is None:
            return {"status": "missing", "checksum": None, "cached": False}
        if expected is None or hashlib.sha256(data).hexdigest() == expected:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return {"status": "ok", "checksum": "verified" if expected else "unavailable",
                    "cached": False}
    raise RuntimeError(f"SHA256 mismatch after retry: {url}")


def download_all(jobs: list[tuple[str, str, str]], raw_dir: Path, interval: str,
                 workers: int) -> dict:
    def run(job):
        kind, symbol, month = job
        url = archive_url(kind, symbol, month, interval)
        return job, download_one(url, archive_path(raw_dir, kind, symbol, month, interval))

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (job, res) in enumerate(pool.map(run, jobs), 1):
            results[job] = res
            if i % 50 == 0 or i == len(jobs):
                print(f"  downloaded {i}/{len(jobs)}", file=sys.stderr)
    return results


# ---------------------------------------------------------------- parsing

def read_zip_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected one CSV member, found {names}")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
            return [row for row in csv.reader(text) if row and any(c.strip() for c in row)]


def split_header(rows: list[list[str]], canonical: list[str], name: str) -> tuple[bool, list]:
    """Detect an optional header row and validate it against the canonical columns."""
    if not rows:
        return False, rows
    first = rows[0]
    if first[0].strip().lstrip("-").isdigit():
        return False, rows
    names = [HEADER_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in first]
    if names != canonical:
        raise ValueError(f"{name}: unexpected header {first}")
    return True, rows[1:]


def find_gaps(times: list[int], start: int, end: int, step: int) -> list[tuple[int, int, int]]:
    """Missing slots on a regular grid in [start, end). Returns (first, last, count)."""
    gaps, prev = [], start - step
    for t in times + [end]:
        if t - prev > step:
            gaps.append((prev + step, t - step, (t - prev) // step - 1))
        prev = t
    return gaps


def same_values(a: list[str], b: list[str]) -> bool:
    return all(Decimal(x) == Decimal(y) for x, y in zip(a, b))


# ---------------------------------------------------------------- builders

def build_klines(symbol: str, months: list[str], raw_dir: Path, interval: str,
                 out_path: Path) -> dict:
    step = INTERVAL_US[interval]
    start, end = month_start_us(months[0]), month_start_us(next_month(months[-1]))
    files, rows = [], []
    for month in months:
        path = archive_path(raw_dir, "klines", symbol, month, interval)
        if not path.exists():
            files.append({"month": month, "present": False})
            continue
        has_header, body = split_header(read_zip_rows(path), KLINE_COLS, path.name)
        units = set()
        for r in body:
            if len(r) != len(KLINE_COLS):
                raise ValueError(f"{path.name}: expected {len(KLINE_COLS)} columns, got {r}")
            o, uo = to_us(r[0])
            c, uc = to_us(r[6])
            units.update((uo, uc))
            rows.append((o, c, r))
        files.append({"month": month, "present": True, "header": has_header,
                      "units": sorted(units), "rows": len(body)})

    rows.sort(key=lambda x: x[0])  # stable: earlier month file wins on duplicates
    kept, seen = [], {}
    dup_exact = dup_conflict = out_of_range = misaligned = bad_close = 0
    conflicts = []
    for o, c, r in rows:
        if not start <= o < end:
            out_of_range += 1
            continue
        values = [r[i] for i in range(1, 11) if i != 6]
        if o in seen:
            prev_c, prev_values = seen[o]
            if prev_c == c and same_values(prev_values, values):
                dup_exact += 1
            else:
                dup_conflict += 1
                conflicts.append(iso(o))
            continue
        seen[o] = (c, values)
        if (o - start) % step:
            misaligned += 1
        if not (o < c < o + step):
            bad_close += 1
        kept.append((o, c, r))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(KLINE_OUT_COLS)
        for o, c, r in kept:
            w.writerow([iso(o), r[1], r[2], r[3], r[4], r[5], iso(c), r[7], r[8], r[9], r[10]])

    aligned = [o for o, _, _ in kept if (o - start) % step == 0]
    gaps = find_gaps(aligned, start, end, step)
    # Placeholder bars (e.g. during exchange maintenance): slot present but no trades.
    no_trade = [iso(o) for o, _, r in kept if int(r[8]) == 0]
    return {
        "symbol": symbol, "kind": "klines", "path": out_path, "files": files,
        "raw_rows": len(rows), "rows": len(kept), "expected": (end - start) // step,
        "dup_exact": dup_exact, "dup_conflict": dup_conflict, "conflicts": conflicts,
        "out_of_range": out_of_range, "misaligned": misaligned, "bad_close": bad_close,
        "no_trade": no_trade,
        "missing": sum(g[2] for g in gaps), "gaps": gaps,
        "first": iso(kept[0][0]) if kept else None, "last": iso(kept[-1][0]) if kept else None,
    }


def build_funding(symbol: str, months: list[str], raw_dir: Path, out_path: Path) -> dict:
    start, end = month_start_us(months[0]), month_start_us(next_month(months[-1]))
    files, rows = [], []
    for month in months:
        path = archive_path(raw_dir, "fundingRate", symbol, month, "")
        if not path.exists():
            files.append({"month": month, "present": False})
            continue
        has_header, body = split_header(read_zip_rows(path), FUNDING_COLS, path.name)
        units = set()
        for r in body:
            if len(r) != len(FUNDING_COLS):
                raise ValueError(f"{path.name}: expected {len(FUNDING_COLS)} columns, got {r}")
            t, unit = to_us(r[0])
            units.add(unit)
            # Funding settles on the hour; calc_time carries a few ms of jitter.
            nominal = (t + US_PER_HOUR // 2) // US_PER_HOUR * US_PER_HOUR
            rows.append((nominal, t, int(r[1]), r[2]))
        files.append({"month": month, "present": True, "header": has_header,
                      "units": sorted(units), "rows": len(body)})

    rows.sort(key=lambda x: x[0])
    kept, seen = [], {}
    dup_exact = dup_conflict = out_of_range = 0
    conflicts = []
    for nominal, t, hours, rate in rows:
        if not start <= nominal < end:
            out_of_range += 1
            continue
        if nominal in seen:
            if seen[nominal][0] == hours and Decimal(seen[nominal][1]) == Decimal(rate):
                dup_exact += 1
            else:
                dup_conflict += 1
                conflicts.append(iso(nominal))
            continue
        seen[nominal] = (hours, rate)
        kept.append((nominal, t, hours, rate))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(FUNDING_OUT_COLS)
        for nominal, t, hours, rate in kept:
            w.writerow([iso(nominal), iso(t), hours, rate])

    # Missing events: walk consecutive settlements using the reported interval.
    gaps, irregular = [], []
    if kept:
        first_n, _, first_h, _ = kept[0]
        lead = (first_n - start) // (first_h * US_PER_HOUR)
        if lead:
            gaps.append((start, first_n - first_h * US_PER_HOUR, lead))
        for (pn, _, ph, _), (cn, _, ch, _) in zip(kept, kept[1:]):
            delta = cn - pn
            if delta in (ph * US_PER_HOUR, ch * US_PER_HOUR):
                continue
            step = ch * US_PER_HOUR
            if delta > step and delta % step == 0:
                gaps.append((pn + step, cn - step, delta // step - 1))
            else:
                irregular.append((iso(pn), iso(cn), delta / US_PER_HOUR))
        last_n, _, last_h, _ = kept[-1]
        step = last_h * US_PER_HOUR
        trail = -(-(end - last_n) // step) - 1
        if trail > 0:
            gaps.append((last_n + step, last_n + trail * step, trail))

    offsets_ms = [abs(t - n) / US_PER_MS for n, t, _, _ in kept]
    return {
        "symbol": symbol, "kind": "funding", "path": out_path, "files": files,
        "raw_rows": len(rows), "rows": len(kept),
        "dup_exact": dup_exact, "dup_conflict": dup_conflict, "conflicts": conflicts,
        "out_of_range": out_of_range, "missing": sum(g[2] for g in gaps), "gaps": gaps,
        "irregular": irregular, "intervals": sorted({h for _, _, h, _ in kept}),
        "max_offset_ms": max(offsets_ms, default=0),
        "nonzero_offsets": sum(1 for x in offsets_ms if x),
        "first": iso(kept[0][0]) if kept else None, "last": iso(kept[-1][0]) if kept else None,
    }


# ---------------------------------------------------------------- README

def fmt_gap(g: tuple[int, int, int], step_label: str) -> str:
    first, last, count = g
    if first == last:
        return f"`{iso(first)}` ({count}{step_label})"
    return f"`{iso(first)}` ~ `{iso(last)}` ({count}{step_label})"


def write_readme(path: Path, args, months: list[str], dl: dict, stats: list[dict]) -> None:
    rel = lambda p: p.relative_to(path.parent).as_posix()
    n_ok = sum(1 for r in dl.values() if r["status"] == "ok")
    n_missing = sum(1 for r in dl.values() if r["status"] == "missing")
    n_verified = sum(1 for r in dl.values() if r["checksum"] == "verified")
    missing_files = sorted(f"{k}/{s}/{m}" for (k, s, m), r in dl.items() if r["status"] == "missing")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    L = []
    L += [f"# Binance USDⓈ-M 선물 {args.interval} 캔들 · 펀딩비 데이터", ""]
    L += [f"- 출처: <{BASE_URL}/> (공개 정적 아카이브 zip. 인증·주문 API는 호출하지 않음)",
          f"- 심볼: {', '.join(args.symbols)}",
          f"- 기간: {months[0]} ~ {months[-1]} (UTC, 월별 파일 {len(months)}개/심볼/종류)",
          f"- 생성 시각: {generated}",
          "- 재생성: `python3 scripts/build_binance_um_dataset.py "
          f"--symbols {' '.join(args.symbols)} --start {args.start} --end {args.end} "
          f"--interval {args.interval} --out-dir data`", ""]

    L += ["## 파일", "", "| 파일 | 행 수 | 첫 시각 (UTC) | 마지막 시각 (UTC) |", "|---|---:|---|---|"]
    for s in stats:
        L.append(f"| `{rel(s['path'])}` | {s['rows']:,} | {s['first']} | {s['last']} |")
    L.append("")

    L += ["## 열 정의", "", f"### `{{symbol}}_{args.interval}.csv` (klines)", "",
          "| 열 | 설명 |", "|---|---|",
          "| `open_time` | 봉 시작 시각 (UTC ISO-8601, ms) |",
          "| `open`, `high`, `low`, `close` | 시가·고가·저가·종가 (USDT) |",
          "| `volume` | 거래량 (기초자산 수량) |",
          "| `close_time` | 봉 종료 시각 (UTC ISO-8601, ms, `open_time + 1h - 1ms`) |",
          "| `quote_volume` | 거래대금 (USDT) |",
          "| `count` | 체결 건수 |",
          "| `taker_buy_volume` | 테이커 매수 거래량 (기초자산) |",
          "| `taker_buy_quote_volume` | 테이커 매수 거래대금 (USDT) |", "",
          "원본의 `ignore` 열(항상 0)은 제거했다.", "",
          "### `{symbol}_funding.csv` (fundingRate)", "",
          "| 열 | 설명 |", "|---|---|",
          "| `funding_time` | 펀딩 정산 시각. `calc_time`을 가장 가까운 정시로 반올림한 값 (UTC ISO-8601) |",
          "| `calc_time` | 원본 계산 시각 그대로 (UTC ISO-8601, ms) |",
          "| `funding_interval_hours` | 펀딩 주기 (시간) |",
          "| `last_funding_rate` | 해당 회차 펀딩비 (소수, 0.0001 = 0.01%) |", ""]

    L += ["## 변환 규칙", "",
          "- **열 이름 통일**: 헤더가 있는 파일은 헤더를 정규 열 이름과 대조해 검증하고, "
          "헤더가 없는 파일은 열 개수를 검증한 뒤 같은 정규 열 이름을 붙였다.",
          "- **타임스탬프 단위**: 값의 자릿수로 판별한다 (`>= 10^15` → µs, `>= 10^12` → ms). "
          "모든 값을 µs로 맞춘 뒤 UTC ISO-8601 문자열(`YYYY-MM-DDTHH:MM:SS.sssZ`)로 변환했다. "
          "ms 미만 자릿수가 있으면 잘리지 않도록 스크립트가 오류로 중단한다.",
          "- **숫자 값**: 원본 문자열을 그대로 기록했다 (부동소수점 재포맷 없음). "
          "따라서 같은 열에서도 `7225`와 `7225.00`처럼 소수 자릿수 표기가 다를 수 있다.",
          "- **중복 제거**: 같은 `open_time`(펀딩은 `funding_time`)이 여러 번 나오면 먼저 나온 행(이른 월 파일)을 남긴다. "
          "값이 숫자로 같으면 '동일 중복', 다르면 '충돌 중복'으로 집계한다.",
          "- **범위 밖 행**: 요청 기간 밖의 행은 제외하고 개수만 집계한다.",
          "- 각 zip은 같은 경로의 `.CHECKSUM`(SHA256)과 대조해 검증했다.", ""]

    L += ["### 원본 파일 형식", "", "| 데이터 | 헤더 없음 (월) | 헤더 있음 (월) | 타임스탬프 단위 |",
          "|---|---|---|---|"]
    for s in stats:
        present = [f for f in s["files"] if f["present"]]
        no_hdr = [f["month"] for f in present if not f["header"]]
        hdr = [f["month"] for f in present if f["header"]]
        units = {}
        for f in present:
            units.setdefault("/".join(f["units"]), []).append(f["month"])
        unit_txt = "; ".join(f"{u}: {compress_months(ms)}" for u, ms in units.items())
        L.append(f"| {s['symbol']} {s['kind']} | {compress_months(no_hdr)} | "
                 f"{compress_months(hdr)} | {unit_txt} |")
    L.append("")

    L += ["## 다운로드 결과", "",
          f"- 요청 zip: {len(dl)}개 (심볼 {len(args.symbols)} × 종류 2 × {len(months)}개월)",
          f"- 존재: {n_ok}개, 아카이브에 없음(404): {n_missing}개",
          f"- SHA256 검증 통과: {n_verified}/{n_ok}"]
    if missing_files:
        L.append(f"- 없는 파일: {', '.join(missing_files)}")
    L.append("")

    L += ["## 품질 점검", "", "### 캔들 (klines)", "",
          "| 심볼 | 원본 행 | 동일 중복 | 충돌 중복 | 범위 밖 | 정시 아님 | close_time 이상 | 최종 행 | 기대 봉 수 | 누락 봉 | 체결 0건 봉 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in (x for x in stats if x["kind"] == "klines"):
        L.append(f"| {s['symbol']} | {s['raw_rows']:,} | {s['dup_exact']} | {s['dup_conflict']} | "
                 f"{s['out_of_range']} | {s['misaligned']} | {s['bad_close']} | {s['rows']:,} | "
                 f"{s['expected']:,} | {s['missing']} | {len(s['no_trade'])} |")
    L.append("")
    L.append(f"기대 봉 수 = {months[0]}-01T00:00Z부터 {months[-1]} 말일 23:00Z까지의 {args.interval} 슬롯 수. "
             "'체결 0건 봉'은 행은 있지만 `count = 0`인 봉으로, 시가=고가=저가=종가=직전 종가이고 "
             "거래량이 0인 자리채움 봉이다 (거래소 점검 시간 등). 누락으로 세지 않고 그대로 두었다.")
    L.append("")
    for s in (x for x in stats if x["kind"] == "klines"):
        L.append(f"**{s['symbol']} 누락 구간** ({len(s['gaps'])}개 구간, {s['missing']}봉)")
        L.append("")
        L += [f"- {fmt_gap(g, '봉')}" for g in s["gaps"]] or ["- 없음"]
        if s["no_trade"]:
            L.append(f"- 체결 0건 봉: {', '.join(f'`{t}`' for t in s['no_trade'])}")
        if s["conflicts"]:
            L.append(f"- 충돌 중복 시각: {', '.join(s['conflicts'])}")
        L.append("")

    L += ["### 펀딩비 (fundingRate)", "",
          "| 심볼 | 원본 행 | 동일 중복 | 충돌 중복 | 범위 밖 | 최종 행 | 펀딩 주기(h) | 누락 회차 | 불규칙 간격 | calc_time 지터 최대 |",
          "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for s in (x for x in stats if x["kind"] == "funding"):
        L.append(f"| {s['symbol']} | {s['raw_rows']:,} | {s['dup_exact']} | {s['dup_conflict']} | "
                 f"{s['out_of_range']} | {s['rows']:,} | {', '.join(map(str, s['intervals']))} | "
                 f"{s['missing']} | {len(s['irregular'])} | {s['max_offset_ms']:.0f} ms |")
    L.append("")
    L.append("누락 회차는 연속한 정산 시각의 간격을 `funding_interval_hours`와 비교해 계산했다 "
             "(기간 시작·끝의 빈 회차 포함). 지터는 `calc_time`과 `funding_time`(정시) 사이의 차이다.")
    L.append("")
    for s in (x for x in stats if x["kind"] == "funding"):
        L.append(f"**{s['symbol']} 펀딩 누락/불규칙** "
                 f"(누락 {s['missing']}회, 불규칙 {len(s['irregular'])}건, "
                 f"지터가 0이 아닌 행 {s['nonzero_offsets']:,}개)")
        L.append("")
        L += [f"- 누락 {fmt_gap(g, '회')}" for g in s["gaps"]]
        L += [f"- 불규칙 `{a}` → `{b}` ({h:g}시간 간격)" for a, b, h in s["irregular"]]
        if not s["gaps"] and not s["irregular"]:
            L.append("- 없음")
        if s["conflicts"]:
            L.append(f"- 충돌 중복 시각: {', '.join(s['conflicts'])}")
        L.append("")

    path.write_text("\n".join(L))


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--start", default="2020-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2026-09", help="last month, YYYY-MM")
    ap.add_argument("--interval", default="1h", choices=sorted(INTERVAL_US))
    ap.add_argument("--raw-dir", type=Path, default=Path(".cache/binance_um"),
                    help="where downloaded zip files are cached")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    months = month_range(args.start, args.end)
    jobs = [(kind, sym, m) for sym in args.symbols for kind in ("klines", "fundingRate") for m in months]
    print(f"downloading {len(jobs)} archives into {args.raw_dir}", file=sys.stderr)
    dl = download_all(jobs, args.raw_dir, args.interval, args.workers)

    stats = []
    for sym in args.symbols:
        stats.append(build_klines(sym, months, args.raw_dir, args.interval,
                                  args.out_dir / f"{sym}_{args.interval}.csv"))
        stats.append(build_funding(sym, months, args.raw_dir, args.out_dir / f"{sym}_funding.csv"))
    write_readme(args.out_dir / "README.md", args, months, dl, stats)

    for s in stats:
        print(f"{s['path']}: rows={s['rows']} dup_exact={s['dup_exact']} "
              f"dup_conflict={s['dup_conflict']} out_of_range={s['out_of_range']} "
              f"missing={s['missing']} gaps={len(s['gaps'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
