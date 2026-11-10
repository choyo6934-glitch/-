"""2단계-1: BTC 순노출 기준 시간별 패널(목표 변수·행동 라벨·특징)과 체결 단위 특징.

- BTC 장부 = XBTUSD + XBT 만기물(모두 1달러 역계약). XBT7D_U110(주간 이진 계약)은 제외.
- 시각 h는 '정시 마감'을 뜻한다. h의 특징은 [.., h) 봉만 쓰고, 라벨은 (h, h+1h]의 체결로 만든다.
- 봉인 구간(h ≥ 2021-07-01) 행은 panel_sealed.parquet로 따로 저장하고, 규칙 추출 코드는 열지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CACHE, SAT, SPLITS
from load_aoa import load_executions, load_wallet
from prices import bitstamp_1m, bitstamp_bars, binance_um_1h, binance_funding

BTC_SYMS_EXCLUDE = {"XBT7D_U110"}
SEALED_START = pd.Timestamp(SPLITS["sealed"][0], tz="UTC")


def btc_symbols(symbols) -> list[str]:
    return sorted(s for s in symbols if s.startswith("XBT") and s not in BTC_SYMS_EXCLUDE)


def contract_specs(ex: pd.DataFrame) -> pd.DataFrame:
    """종목별 계약 유형: inverse(XBT/계약 = 1/가격) 또는 linear(XBT/계약 = m·가격)."""
    tr = ex[(ex.exectype == "Trade") & (ex.lastqty > 0) & (ex.lastpx > 0)]
    g = tr.groupby("symbol")
    inv = (tr.execcost.abs() * tr.lastpx / tr.lastqty / SAT).groupby(tr.symbol)
    lin = (tr.execcost.abs() / (tr.lastqty * tr.lastpx)).groupby(tr.symbol)
    spec = pd.DataFrame({"inv_med": inv.median(), "inv_cv": inv.std() / inv.median(),
                         "lin_med": lin.median(), "lin_cv": lin.std() / lin.median(), "n": g.size()})
    # 선형 배수는 정확히 일정(변동계수 ~1e-16)하므로 먼저 판정한다. 가격이 거의 안 변한 종목은
    # 역계약 비율도 우연히 일정해 보이므로 역계약 판정을 나중에 한다.
    spec["kind"] = np.where(spec.lin_cv.fillna(1) < 1e-9, "linear",
                            np.where(spec.inv_cv.fillna(1) < 0.01, "inverse", "unknown"))
    spec["mult"] = np.where(spec.kind == "linear", spec.lin_med, SAT)
    return spec


def btc_fill_book(ex: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """BTC 체결을 시간순으로 합쳐 순수량 Q와 순노출 에피소드(0→0, 방향 전환 분리)를 만든다."""
    syms = btc_symbols(ex.symbol.unique())
    d = ex[ex.symbol.isin(syms) & ex.exectype.isin(["Trade", "Settlement"])].copy()
    d = d.sort_values(["t", "row_order"], kind="mergesort").reset_index(drop=True)
    sgn = np.where(d["side"].fillna("") == "Buy", 1.0, -1.0)
    d["q"] = sgn * d["lastqty"].to_numpy()
    q = d["q"].to_numpy()
    Q = np.cumsum(q)
    Q[np.abs(Q) < 1e-9] = 0
    d["Q_after"] = Q
    Qb = np.concatenate([[0.0], Q[:-1]])
    d["Q_before"] = Qb
    # 진입(증가)·청산(감소) 분할
    inc = np.where((Qb == 0) | (np.sign(q) == np.sign(Qb)), np.abs(q), np.maximum(0, np.abs(q) - np.abs(Qb)))
    dec = np.abs(q) - inc
    d["inc_qty"] = inc
    d["dec_qty"] = dec
    # 에피소드 번호: Q가 0에서 벗어나는 체결 또는 0을 지나가는 체결에서 새로 시작
    new_ep = ((Qb == 0) & (Q != 0)) | ((Qb != 0) & (Q != 0) & (np.sign(Q) != np.sign(Qb)))
    ep = np.cumsum(new_ep)
    # 감소 체결은 이전 에피소드에 귀속(방향 전환 체결은 닫는 쪽과 여는 쪽 모두 가짐)
    d["ep_open"] = np.where(inc > 0, ep, -1)
    d["ep_close"] = np.where(dec > 0, np.where(new_ep, ep - 1, ep), -1)
    d["cash"] = -d["execcost"].fillna(0) - d["execcomm"].fillna(0)
    d["xbt"] = d["execcost"].abs() / SAT
    d["maker"] = (d["lastliquidityind"] == "AddedLiquidity").to_numpy(dtype=bool, na_value=False)
    return d


def btc_episodes(fb: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    """순노출 에피소드 단위 요약(평단·최대 명목·손익·진입 주문 수 등)."""
    rows = {}
    for r in fb.itertuples(index=False):
        tot = abs(r.q)
        if r.ep_open >= 0:
            e = rows.setdefault(r.ep_open, dict(ep=r.ep_open, dir=int(np.sign(r.Q_after)), t_open=r.t, t_close=pd.NaT,
                                                entry_q=0.0, entry_inv=0.0, exit_q=0.0, exit_inv=0.0, cash=0.0,
                                                entry_orders=[], first_entry_t=r.t, last_entry_t=r.t, max_absQ=0.0,
                                                maker_xbt=0.0, xbt=0.0, n_fills=0))
            f = r.inc_qty / tot
            e["entry_q"] += r.inc_qty
            e["entry_inv"] += r.inc_qty / r.lastpx
            e["cash"] += f * r.cash
            if not e["entry_orders"] or e["entry_orders"][-1] != r.orderid:
                e["entry_orders"].append(r.orderid)
            e["last_entry_t"] = r.t
            e["max_absQ"] = max(e["max_absQ"], abs(r.Q_after))
            e["maker_xbt"] += f * r.xbt * r.maker
            e["xbt"] += f * r.xbt
            e["n_fills"] += 1
        if r.ep_close >= 0:
            e = rows[r.ep_close]
            f = r.dec_qty / tot
            e["exit_q"] += r.dec_qty
            e["exit_inv"] += r.dec_qty / r.lastpx
            e["cash"] += f * r.cash
            e["maker_xbt"] += f * r.xbt * r.maker
            e["xbt"] += f * r.xbt
            e["n_fills"] += 1
            if (r.Q_after == 0) or (np.sign(r.Q_after) != e["dir"]):
                e["t_close"] = r.t
    ep = pd.DataFrame(rows.values())
    # 펀딩: BTC 종목 Funding 행을 그 시각 열린 에피소드에 귀속
    syms = btc_symbols(ex.symbol.unique())
    fu = ex[(ex.exectype == "Funding") & ex.symbol.isin(syms)][["t", "execcomm"]].sort_values("t")
    starts = ep["t_open"].to_numpy()
    ends = ep["t_close"].fillna(pd.Timestamp("2100-01-01", tz="UTC")).to_numpy()
    idx = np.searchsorted(starts, fu["t"].to_numpy(), side="right") - 1
    ok = (idx >= 0) & (fu["t"].to_numpy() <= ends[np.clip(idx, 0, None)])
    fund = pd.Series(-fu["execcomm"].to_numpy()[ok]).groupby(idx[ok]).sum()
    ep["funding"] = fund.reindex(range(len(ep))).fillna(0).to_numpy()
    ep["pnl_btc"] = (ep["cash"] + ep["funding"]) / SAT
    ep["avg_entry"] = ep["entry_q"] / ep["entry_inv"]          # 역계약 평단(조화평균)
    ep["avg_exit"] = ep["exit_q"] / ep["exit_inv"].replace(0, np.nan)
    ep["n_entry_orders"] = ep["entry_orders"].map(len)
    ep = ep.drop(columns=["entry_orders"])
    ep["hold_h"] = (ep["t_close"] - ep["t_open"]).dt.total_seconds() / 3600
    ep["entry_span_h"] = (ep["last_entry_t"] - ep["first_entry_t"]).dt.total_seconds() / 3600
    ep["maker_share"] = ep["maker_xbt"] / ep["xbt"]
    return ep


def equity_hourly(ex: pd.DataFrame, pm: pd.DataFrame, w: pd.DataFrame, hours: pd.DatetimeIndex,
                  p_btc: pd.Series) -> pd.DataFrame:
    """시간별 평가자산(BTC) = 순입금 + 누적 실현 + 미실현. BTC 종목은 Bitstamp, ETH는 2020년부터 바이낸스,
    나머지 종목은 직전 체결가로 평가."""
    spec = contract_specs(ex)
    m = pm.merge(ex[["row_order", "lastpx"]], on="row_order", how="left")
    m = m.sort_values(["t", "row_order"], kind="mergesort")
    # 순입금
    comp = w[(w.transactstatus == "Completed") & w.transacttype.isin(["Deposit", "Withdrawal"])]
    dep = comp.groupby("date")["amount"].sum().cumsum()
    dep_h = dep.reindex(dep.index.union(hours)).ffill().reindex(hours).fillna(0)
    # 누적 실현(모든 종목)
    rc = m.groupby("t")["realized_sat"].sum().cumsum()
    pos_t = rc.index.searchsorted(hours, side="left") - 1  # t < h
    real_h = np.where(pos_t >= 0, rc.to_numpy()[np.clip(pos_t, 0, None)], 0.0)
    # 미실현
    try:
        eth = binance_um_1h("ETHUSDT")["close"]
        eth.index = eth.index + pd.Timedelta(hours=1)  # 봉 마감 시각으로
    except Exception:
        eth = pd.Series(dtype=float)
    unreal = np.zeros(len(hours))
    for s, g in m[m.exectype != "Funding"].groupby("symbol"):
        k = np.searchsorted(g["t"].to_numpy(), hours.to_numpy(), side="left") - 1
        valid = k >= 0
        kk = np.clip(k, 0, None)
        Q = np.where(valid, g["qty_after"].to_numpy()[kk], 0.0)
        C = np.where(valid, g["cost_after"].to_numpy()[kk], 0.0)
        if not np.any(Q != 0):
            continue
        last_px = np.where(valid, g["lastpx"].to_numpy()[kk], np.nan)
        kind = spec.loc[s, "kind"] if s in spec.index else "unknown"
        if s.startswith("XBT") and s not in BTC_SYMS_EXCLUDE:
            P = p_btc.reindex(hours).to_numpy()
        elif s.startswith("ETHUSD") and len(eth):
            P = eth.reindex(hours).to_numpy()
            P = np.where(np.isnan(P), last_px, P)
        else:
            P = last_px
        if kind == "inverse":
            U = -C - Q * SAT / P
        else:
            U = -C + Q * spec.loc[s, "mult"] * P
        unreal += np.where(Q != 0, np.nan_to_num(U), 0.0)
    eq = (dep_h.to_numpy() + real_h + unreal) / SAT
    return pd.DataFrame({"equity_btc": eq, "unreal_btc": unreal / SAT, "net_dep_btc": dep_h.to_numpy() / SAT,
                         "realized_cum_btc": real_h / SAT}, index=hours)


def market_features(h1: pd.DataFrame, m15: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """정시 h 기준 특징. h1은 시작 시각 색인 1시간봉, h에서 쓸 수 있는 마지막 봉은 [h−1h, h)."""
    b = h1.copy()
    b.index = b.index + pd.Timedelta(hours=1)  # 마감 시각 색인
    c = b["close"]
    lc = np.log(c)
    f = pd.DataFrame(index=b.index)
    f["close"] = c
    for k, name in [(1, "r_1h"), (4, "r_4h"), (24, "r_1d"), (168, "r_7d")]:
        f[name] = lc - lc.shift(k)
    q = m15.copy()
    q.index = q.index + pd.Timedelta(minutes=15)
    lq = np.log(q["close"])
    r15 = (lq - lq.shift(1)).reindex(b.index)
    f["r_15m"] = r15
    prev_c = c.shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - prev_c).abs(), (b["low"] - prev_c).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean()
    f["atr24"] = tr.rolling(24).mean()
    f["atr_pct"] = f["atr14"] / c
    f["rv24"] = (lc - lc.shift(1)).rolling(24).std() * np.sqrt(24 * 365)
    f["rv168"] = (lc - lc.shift(1)).rolling(168).std() * np.sqrt(24 * 365)
    for n, name in [(24, "pos24"), (168, "pos7d")]:
        hi = b["high"].rolling(n).max()
        lo = b["low"].rolling(n).min()
        f[name] = (c - lo) / (hi - lo)
        f[f"range{n}_atr"] = (hi - lo) / f["atr14"]
    # ATR 배수로 잰 움직임
    for k, name in [(1, "m_1h_atr"), (4, "m_4h_atr"), (24, "m_1d_atr"), (168, "m_7d_atr")]:
        f[name] = (c - c.shift(k)) / f["atr14"]
    f["vol_ratio24"] = b["volume"] / b["volume"].rolling(24 * 7).mean()
    f["hour"] = f.index.hour
    f["dow"] = f.index.dayofweek
    return f.reindex(hours)


def bitmex_funding_series(ex: pd.DataFrame, fb: pd.DataFrame) -> pd.Series:
    """Funding 행의 commission은 계정 기준 부호(지급 +). 시장 펀딩률 = commission × 포지션 방향."""
    fu = ex[(ex.exectype == "Funding") & (ex.symbol == "XBTUSD")][["t", "commission", "execcost"]].sort_values("t")
    # XBTUSD(역계약): execcost < 0 이면 롱
    sign = np.where(fu["execcost"] < 0, 1.0, -1.0)
    s = pd.Series(fu["commission"].to_numpy() * sign, index=fu["t"])
    return s[~s.index.duplicated(keep="last")]


def build_panel():
    ex = load_executions()
    w = load_wallet()
    pm = pd.read_parquet(CACHE / "exec_posmap.parquet")
    h1 = bitstamp_bars("1h")
    m15 = bitstamp_bars("15min")
    fb = btc_fill_book(ex, pm)
    eps = btc_episodes(fb, ex)

    hours = pd.date_range("2018-03-05 00:00", "2022-01-01 00:00", freq="1h", tz="UTC")
    feats = market_features(h1, m15, hours)
    p_btc = feats["close"]
    eq = equity_hourly(ex, pm, w, hours, p_btc)

    # 순 BTC 수량(Q)과 다음 시각
    t_f = fb["t"].to_numpy()
    k = np.searchsorted(t_f, hours.to_numpy(), side="left") - 1
    Q = np.where(k >= 0, fb["Q_after"].to_numpy()[np.clip(k, 0, None)], 0.0)
    panel = pd.DataFrame(index=hours)
    panel["Q"] = Q
    panel = panel.join(eq).join(feats)
    panel["notional_btc"] = panel["Q"] / panel["close"]
    panel["L"] = panel["notional_btc"] / panel["equity_btc"]
    panel["flat"] = panel["L"].abs() < 0.02
    panel["side"] = np.where(panel["flat"], 0, np.sign(panel["L"])).astype(int)

    # (h, h+1] 체결 집계
    hb = pd.Series(hours.searchsorted(fb["t"], side="left"), index=fb.index)  # 체결이 속한 시각 h+1의 위치
    fb["h_next_idx"] = hb.to_numpy()
    agg = fb.groupby("h_next_idx").agg(inc_qty=("inc_qty", "sum"), dec_qty=("dec_qty", "sum"),
                                       n_fill=("q", "size"))
    agg["n_inc_orders"] = fb[fb.inc_qty > 0].groupby("h_next_idx")["orderid"].nunique()
    agg["n_dec_orders"] = fb[fb.dec_qty > 0].groupby("h_next_idx")["orderid"].nunique()
    agg = agg.fillna(0)
    nxt = pd.DataFrame(index=range(len(hours)))
    nxt = nxt.join(agg)
    # 위치 j의 체결은 (hours[j-1], hours[j]] 구간 → 시각 hours[j-1]의 '다음 1시간 행동'
    lab = nxt.shift(-1).fillna(0)
    lab.index = hours
    for c in ("inc_qty", "dec_qty", "n_fill", "n_inc_orders", "n_dec_orders"):
        panel["nx_" + c] = lab[c].to_numpy()
    panel["L_next"] = panel["L"].shift(-1)
    panel["side_next"] = panel["side"].shift(-1)
    panel["Q_next"] = panel["Q"].shift(-1)
    panel["dL"] = panel["L_next"] - panel["L"]
    panel["dL_trade"] = (panel["Q_next"] - panel["Q"]) / panel["close"].shift(-1) / panel["equity_btc"].shift(-1)

    # 에피소드 상태(정시 h 기준): 평단, 보유 시간, 진입 주문 수, 마지막 진입 이후 시간
    inc = fb[fb.inc_qty > 0].copy()
    inc["cum_q"] = inc.groupby("ep_open")["inc_qty"].cumsum()
    inc["cum_inv"] = (inc["inc_qty"] / inc["lastpx"]).groupby(inc["ep_open"]).cumsum()
    inc["new_order"] = inc["orderid"] != inc.groupby("ep_open")["orderid"].shift(1)
    inc["cum_orders"] = inc.groupby("ep_open")["new_order"].cumsum()
    ki = np.searchsorted(inc["t"].to_numpy(), hours.to_numpy(), side="left") - 1
    ok = ki >= 0
    kc = np.clip(ki, 0, None)
    ep_id = np.where(ok, inc["ep_open"].to_numpy()[kc], -1)
    # 현재 h에 열린 에피소드인지 확인
    cur_ep = np.where(Q != 0, ep_id, -1)
    panel["ep"] = cur_ep
    avg_e = np.where(ok, inc["cum_q"].to_numpy()[kc] / inc["cum_inv"].to_numpy()[kc], np.nan)
    panel["avg_entry"] = np.where(cur_ep >= 0, avg_e, np.nan)
    panel["n_entry_orders"] = np.where(cur_ep >= 0, inc["cum_orders"].to_numpy()[kc], 0)
    last_inc_t = pd.to_datetime(np.where(ok, inc["t"].to_numpy()[kc], np.datetime64("NaT")), utc=True)
    panel["h_since_entry_fill"] = np.where(cur_ep >= 0, (hours - last_inc_t).total_seconds() / 3600, np.nan)
    t_open = eps.set_index("ep")["t_open"]
    to = pd.to_datetime(t_open.reindex(cur_ep).to_numpy(), utc=True)
    panel["hold_h"] = np.where(cur_ep >= 0, (hours - to).total_seconds() / 3600, np.nan)
    d = np.sign(panel["Q"])
    panel["dist_avg_atr"] = d * (panel["close"] - panel["avg_entry"]) / panel["atr14"]   # + 이면 이익 방향
    panel["dist_avg_pct"] = d * (panel["close"] / panel["avg_entry"] - 1)
    panel["upnl_eq"] = np.where(cur_ep >= 0, panel["Q"] * (1 / panel["avg_entry"] - 1 / panel["close"]) / panel["equity_btc"], 0.0)
    # 펀딩
    bfs = bitmex_funding_series(ex, fb)
    panel["fund_bitmex"] = bfs.reindex(bfs.index.union(hours)).ffill().reindex(hours).to_numpy()
    try:
        bf = binance_funding("BTCUSDT")
        panel["fund_binance"] = bf.reindex(bf.index.union(hours)).ffill().reindex(hours).to_numpy()
    except Exception:
        panel["fund_binance"] = np.nan
    # 구간
    per = pd.Series("other", index=hours)
    for name, (a, b) in SPLITS.items():
        per[(hours > pd.Timestamp(a, tz="UTC")) & (hours <= pd.Timestamp(b, tz="UTC"))] = name
    panel["period"] = per.to_numpy()
    panel.index.name = "h"

    open_part = panel[panel.index < SEALED_START]
    sealed_part = panel[panel.index >= SEALED_START]
    open_part.to_parquet(CACHE / "panel_open.parquet")
    sealed_part.to_parquet(CACHE / "panel_sealed.parquet")
    fb.to_parquet(CACHE / "btc_fill_book.parquet", index=False)
    eps.to_parquet(CACHE / "btc_episodes.parquet", index=False)
    return panel, fb, eps


if __name__ == "__main__":
    panel, fb, eps = build_panel()
    op = panel[panel.index < SEALED_START]
    print(op.shape)
    print(op[["Q", "equity_btc", "L", "close", "atr14", "pos24", "fund_bitmex"]].describe().T.round(4))
    print("episodes", len(eps), "open eps", eps.t_close.isna().sum())
