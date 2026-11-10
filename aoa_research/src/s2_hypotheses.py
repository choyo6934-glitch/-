"""2단계-2: 출발 가설 재검증(구간별). 봉인 구간(2021H2)은 --sealed 플래그 없이는 계산하지 않는다.

체결 단위 특징은 Bitstamp 1분봉으로, 체결 직전에 마감된 1분봉까지만 쓴다.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from config import CACHE, RESULTS, SAT, SPLITS
from load_aoa import load_executions
from prices import bitstamp_1m

SEALED_START = pd.Timestamp(SPLITS["sealed"][0], tz="UTC")


def asof(series: pd.Series, times) -> np.ndarray:
    """times 시각 이전(포함)의 마지막 값. series 색인은 정렬되어 있어야 한다."""
    idx = series.index.searchsorted(pd.DatetimeIndex(times), side="right") - 1
    vals = series.to_numpy()
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)


def period_of(t: pd.Series) -> pd.Series:
    out = pd.Series("other", index=t.index)
    for name, (a, b) in SPLITS.items():
        out[(t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(b, tz="UTC"))] = name
    return out


def minute_features(t: pd.Series, m: pd.DataFrame) -> pd.DataFrame:
    """체결 시각 t 직전에 마감된 1분봉 기준: 1시간·7일 수익률, 24시간 범위 내 위치."""
    c = m["close"]
    hi24 = m["high"].rolling(1440, min_periods=1000).max()
    lo24 = m["low"].rolling(1440, min_periods=1000).min()
    idx = m.index.searchsorted(t.dt.floor("min"), side="left") - 1   # 체결 분 이전의 마지막 완성 봉
    idx = np.clip(idx, 0, len(m) - 1)
    cc = c.to_numpy()
    out = pd.DataFrame(index=t.index)
    out["px_prev"] = cc[idx]
    out["r_1h"] = np.log(cc[idx] / cc[np.clip(idx - 60, 0, None)])
    out["r_7d"] = np.log(cc[idx] / cc[np.clip(idx - 10080, 0, None)])
    out["pos24"] = (cc[idx] - lo24.to_numpy()[idx]) / (hi24.to_numpy()[idx] - lo24.to_numpy()[idx])
    return out


def order_type_stats(ex: pd.DataFrame, periods: list[str]) -> pd.DataFrame:
    tr = ex[ex.exectype == "Trade"].copy()
    tr["period"] = period_of(tr["t"])
    tr["xbt"] = tr["execcost"].abs() / SAT
    tr["maker"] = (tr["lastliquidityind"] == "AddedLiquidity").to_numpy(dtype=bool, na_value=False)
    rows = []
    for p in periods:
        d = tr[tr.period == p]
        if not len(d):
            continue
        v = d.groupby("ordtype")["xbt"].sum() / d["xbt"].sum()
        rows.append(dict(period=p, fills=len(d), orders=d.orderid.nunique(),
                         limit_value_share=v.get("Limit", 0), market_value_share=v.get("Market", 0),
                         stop_value_share=v.get("Stop", 0) + v.get("StopLimit", 0),
                         n_stop_orders=d[d.ordtype.isin(["Stop", "StopLimit"])].orderid.nunique(),
                         maker_value_share=(d.xbt * d.maker).sum() / d.xbt.sum(), maker_fill_share=d.maker.mean(),
                         liquidation_fills=int((d.text == "Liquidation").fillna(False).astype(bool).sum())))
    return pd.DataFrame(rows)


def entry_fill_stats(fb: pd.DataFrame, m: pd.DataFrame, periods: list[str]) -> pd.DataFrame:
    e = fb[fb.inc_qty > 0].copy()
    e["period"] = period_of(e["t"])
    e["val"] = e["inc_qty"] / e["lastpx"]  # BTC
    e["dir"] = np.sign(e["q"])
    f = minute_features(e["t"], m)
    e = e.join(f)
    rows = []
    for p in periods:
        d = e[(e.period == p) & e.r_1h.notna()]
        if not len(d):
            continue
        contra = np.sign(d.r_1h) == -d.dir
        with7d = np.sign(d.r_7d) == d.dir
        b, s = d[d.dir > 0], d[d.dir < 0]
        rows.append(dict(period=p, entry_fills=len(d), entry_value_btc=d.val.sum(),
                         contra_1h_value_share=(d.val * contra).sum() / d.val.sum(),
                         with_7d_trend_value_share=(d.val * with7d).sum() / d.val.sum(),
                         buy_pos24_mean_vw=np.average(b.pos24, weights=b.val) if len(b) else np.nan,
                         sell_pos24_mean_vw=np.average(s.pos24, weights=s.val) if len(s) else np.nan,
                         buy_pos24_mean=b.pos24.mean(), sell_pos24_mean=s.pos24.mean(),
                         long_value_share=b.val.sum() / d.val.sum()))
    return pd.DataFrame(rows), e


def episode_stats(fb: pd.DataFrame, eps: pd.DataFrame, panel: pd.DataFrame, h1: pd.DataFrame,
                  periods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = eps.copy()
    eps["period"] = period_of(eps["t_open"])
    # 진입 시점 평가자산과 최대 레버리지(시간별 패널)
    pe = panel[["equity_btc", "L", "ep"]].copy()
    eq_open = asof(pe["equity_btc"], eps["t_open"])
    eps["equity_open"] = eq_open
    eps["max_notional_btc"] = np.nan
    # 최대 명목(BTC) ≈ 최대 |Q| / 평단
    eps["max_notional_btc"] = eps["max_absQ"] / eps["avg_entry"]
    eps["lev_max_at_open_eq"] = eps["max_notional_btc"] / eps["equity_open"]
    eps["kept"] = eps["lev_max_at_open_eq"] >= 0.02
    # 진입 주문 단위: 주문별 진입 명목 / 진입 시점 평가자산
    inc = fb[fb.inc_qty > 0].copy()
    inc["val"] = inc["inc_qty"] / inc["lastpx"]
    od = inc.groupby(["ep_open", "orderid"], sort=False).agg(t=("t", "first"), val=("val", "sum"),
                                                              q=("inc_qty", "sum"), inv=("val", "sum"))
    od = od.reset_index()
    od["px"] = od["q"] / od["inv"]
    od = od.sort_values(["ep_open", "t"]).reset_index(drop=True)
    od["k"] = od.groupby("ep_open").cumcount()
    od["cum_q"] = od.groupby("ep_open")["q"].cumsum()
    od["cum_inv"] = od.groupby("ep_open")["inv"].cumsum()
    od["avg_before"] = (od["cum_q"] - od["q"]) / (od["cum_inv"] - od["inv"])
    dir_map = eps.set_index("ep")["dir"]
    od["dir"] = od["ep_open"].map(dir_map)
    # 물타기: 롱은 평단보다 낮은 가격, 숏은 평단보다 높은 가격에서 추가(포지션이 손실 중일 때의 추가)
    od["avg_down"] = np.where(od["k"] > 0, od["dir"] * (od["px"] - od["avg_before"]) < 0, np.nan)
    od["add_dist_pct"] = np.where(od["k"] > 0, od["dir"] * (od["px"] / od["avg_before"] - 1), np.nan)
    eqs = panel["equity_btc"]
    od["equity"] = asof(eqs, od["t"])
    od["size_eq"] = od["val"] / od["equity"]
    od["period"] = period_of(od["t"])
    # 최대 역행(MAE): 시간봉 고가·저가와 그 시각의 이동 평단
    b = h1.copy()
    b.index = b.index + pd.Timedelta(hours=1)
    pnl = panel[["ep", "avg_entry", "Q"]].join(b[["high", "low"]])
    pnl = pnl[pnl.ep >= 0]
    adv = np.where(pnl.Q > 0, (pnl.avg_entry - pnl.low) / pnl.avg_entry, (pnl.high - pnl.avg_entry) / pnl.avg_entry)
    mae = pd.Series(adv, index=pnl.index).groupby(pnl.ep.to_numpy()).max()
    eps["mae_pct"] = eps["ep"].map(mae)
    rows = []
    for p in periods:
        e = eps[(eps.period == p) & eps.kept & eps.t_close.notna()]
        o = od[(od.period == p) & od.ep_open.isin(e.ep)]
        if not len(e):
            continue
        win = e.pnl_btc > 0
        big = e.mae_pct > 0.10
        rows.append(dict(period=p, n_episodes=len(e), n_episodes_all=int(((eps.period == p) & eps.t_close.notna()).sum()),
                         entry_orders_median=e.n_entry_orders.median(), entry_span_h_median=e.entry_span_h.median(),
                         order_size_eq_median=o.size_eq.median(), lev_max_median=e.lev_max_at_open_eq.median(),
                         hold_h_median=e.hold_h.median(), hold_min_median=e.hold_h.median() * 60,
                         adds_avg_down_share=o.loc[o.k > 0, "avg_down"].mean(),
                         adds_avg_down_value_share=(o.loc[o.k > 0, "avg_down"] * o.loc[o.k > 0, "val"]).sum() / o.loc[o.k > 0, "val"].sum(),
                         win_rate=win.mean(), avg_win_btc=e.loc[win, "pnl_btc"].mean(),
                         avg_loss_btc=e.loc[~win, "pnl_btc"].mean(), pnl_sum_btc=e.pnl_btc.sum(),
                         maker_share_median=e.maker_share.median(),
                         mae10_n=int(big.sum()), mae10_win_rate=(e.loc[big, "pnl_btc"] > 0).mean(),
                         mae10_pnl_sum_btc=e.loc[big, "pnl_btc"].sum(),
                         mae_le10_pnl_sum_btc=e.loc[~big, "pnl_btc"].sum()))
    return pd.DataFrame(rows), eps, od


def stage1_position_stats(periods: list[str]) -> pd.DataFrame:
    """1단계 종목별 포지션(2% 이상) 기준 승률·평균 손익(모든 종목)."""
    pos = pd.read_parquet(CACHE / "positions.parquet")
    pos["period"] = period_of(pos["t_open"])
    rows = []
    for p in periods:
        d = pos[(pos.period == p) & pos.kept & pos.t_close.notna()]
        if not len(d):
            continue
        win = d.pnl_xbt > 0
        rows.append(dict(period=p, n_positions=len(d), win_rate=win.mean(), avg_win_btc=d.loc[win, "pnl_xbt"].mean(),
                         avg_loss_btc=d.loc[~win, "pnl_xbt"].mean(), hold_h_median=d.hold_h.median(),
                         hold_min_median=d.hold_h.median() * 60, pnl_sum_btc=d.pnl_xbt.sum(),
                         lev_max_median=d.size_ratio.median(), entry_orders_median=d.n_entry_orders.median()))
    return pd.DataFrame(rows)


def main(sealed: bool = False):
    periods = ["y2018", "train", "valid"] + (["sealed"] if sealed else [])
    ex = load_executions()
    fb = pd.read_parquet(CACHE / "btc_fill_book.parquet")
    eps = pd.read_parquet(CACHE / "btc_episodes.parquet")
    panel = pd.read_parquet(CACHE / "panel_open.parquet")
    if sealed:
        panel = pd.concat([panel, pd.read_parquet(CACHE / "panel_sealed.parquet")])
    else:
        ex = ex[ex.t < SEALED_START]
        fb = fb[fb.t < SEALED_START]
        eps = eps[eps.t_open < SEALED_START]
    from prices import bitstamp_bars
    h1 = bitstamp_bars("1h")
    m = bitstamp_1m()
    ot = order_type_stats(ex, periods)
    es, efills = entry_fill_stats(fb, m, periods)
    ep, eps2, od = episode_stats(fb, eps, panel, h1, periods)
    s1 = stage1_position_stats(periods)
    tag = "_with_sealed" if sealed else ""
    ot.to_csv(RESULTS / f"s2_h_order_types{tag}.csv", index=False)
    es.to_csv(RESULTS / f"s2_h_entry_fills{tag}.csv", index=False)
    ep.to_csv(RESULTS / f"s2_h_episodes{tag}.csv", index=False)
    s1.to_csv(RESULTS / f"s2_h_stage1_positions{tag}.csv", index=False)
    eps2.to_parquet(CACHE / f"btc_episodes_stats{tag}.parquet", index=False)
    od.to_parquet(CACHE / f"btc_entry_orders{tag}.parquet", index=False)
    pd.set_option("display.width", 250)
    for name, d in [("order types", ot), ("entry fills", es), ("BTC episodes", ep), ("stage1 positions", s1)]:
        print("==", name)
        print(d.round(4).T.to_string())


if __name__ == "__main__":
    main(sealed="--sealed" in sys.argv)
