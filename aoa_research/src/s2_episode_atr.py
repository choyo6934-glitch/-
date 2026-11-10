"""2단계-4: 에피소드 단위 ATR 배수 통계(규칙집 수치의 근거). 학습(2019~2020)·검증(2021H1)만.

ATR = 진입 직전 마감 1시간봉 기준 ATR14(Bitstamp). 모든 거리는 이 ATR의 배수.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CACHE, RESULTS
from prices import bitstamp_bars


def asof_idx(index: pd.DatetimeIndex, times) -> np.ndarray:
    return index.searchsorted(pd.DatetimeIndex(times), side="right") - 1


def build():
    eps = pd.read_parquet(CACHE / "btc_episodes_stats.parquet")
    od = pd.read_parquet(CACHE / "btc_entry_orders.parquet")
    panel = pd.read_parquet(CACHE / "panel_open.parquet")
    h1 = bitstamp_bars("1h")
    b = h1.copy()
    b.index = b.index + pd.Timedelta(hours=1)   # 마감 시각 색인

    e = eps[eps.kept & eps.t_close.notna() & eps.period.isin(["y2018", "train", "valid"])].copy()
    k = asof_idx(panel.index, e["t_open"])        # 진입 직전 정시(마감 봉)
    for c in ("atr14", "close", "pos24", "pos7d", "m_1h_atr", "m_4h_atr", "m_1d_atr", "m_7d_atr", "atr_pct", "equity_btc"):
        e[c + "_0"] = panel[c].to_numpy()[k]
    first = od[od.k == 0].set_index("ep_open")
    e["px0"] = e["ep"].map(first["px"])
    e["offset_atr"] = e["dir"] * (e["close_0"] - e["px0"]) / e["atr14_0"]          # +: 롱은 직전 종가보다 아래에서 매수
    e["exit_profit_atr"] = e["dir"] * (e["avg_exit"] - e["avg_entry"]) / e["atr14_0"]
    e["exit_profit_pct"] = e["dir"] * (e["avg_exit"] / e["avg_entry"] - 1)
    e["mae_atr"] = e["mae_pct"] * e["avg_entry"] / e["atr14_0"]
    # 최대 유리 이동(MFE): 보유 중 시간봉 고가·저가 기준(최종 평단 대비)
    hi = b["high"].to_numpy()
    lo = b["low"].to_numpy()
    s_idx = b.index.searchsorted(e["t_open"].dt.floor("h") + pd.Timedelta(hours=1), side="left")
    e_idx = b.index.searchsorted(e["t_close"].dt.floor("h") + pd.Timedelta(hours=1), side="left")
    mfe = []
    for s0, s1, d, a in zip(s_idx, e_idx, e["dir"].to_numpy(), e["avg_entry"].to_numpy()):
        s1 = max(s1, s0)
        if d > 0:
            mfe.append(hi[s0:s1 + 1].max() / a - 1)
        else:
            mfe.append(1 - lo[s0:s1 + 1].min() / a)
    e["mfe_pct"] = mfe
    e["mfe_atr"] = e["mfe_pct"] * e["avg_entry"] / e["atr14_0"]
    # 추가 주문 간격(ATR): 물타기 추가가 평단에서 얼마나 떨어져 있었나
    od2 = od[od.k > 0].copy()
    atr_map = e.set_index("ep")["atr14_0"]
    od2["atr"] = od2["ep_open"].map(atr_map)
    od2["add_dist_atr"] = od2["add_dist_pct"] * od2["avg_before"] / od2["atr"]   # −: 물타기(불리한 쪽)
    od2 = od2[od2.atr.notna()]
    od2["gap_h"] = od2.groupby("ep_open")["t"].diff().dt.total_seconds() / 3600
    od2 = od2.merge(e[["ep", "period"]], left_on="ep_open", right_on="ep", how="left", suffixes=("", "_e"))
    return e, od2


def summarize(e: pd.DataFrame, od2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for per in ("y2018", "train", "valid"):
        d = e[e.period == per]
        o = od2[od2.period_e == per] if "period_e" in od2 else od2[od2.period == per]
        if not len(d):
            continue
        win = d.pnl_btc > 0
        q = lambda s, x: float(np.nanquantile(s, x)) if len(s.dropna()) else np.nan
        rows.append(dict(
            period=per, n=len(d),
            atr_pct_median=d.atr_pct_0.median(),
            long_first_pos24_median=d.loc[d.dir > 0, "pos24_0"].median(),
            short_first_pos24_median=d.loc[d.dir < 0, "pos24_0"].median(),
            long_first_pos24_q75=q(d.loc[d.dir > 0, "pos24_0"], 0.75),
            short_first_pos24_q25=q(d.loc[d.dir < 0, "pos24_0"], 0.25),
            long_m1h_atr_median=d.loc[d.dir > 0, "m_1h_atr_0"].median(),
            short_m1h_atr_median=d.loc[d.dir < 0, "m_1h_atr_0"].median(),
            offset_atr_median=d.offset_atr.median(),
            entry_orders_median=d.n_entry_orders.median(),
            entry_span_h_median=d.entry_span_h.median(),
            add_dist_atr_median=o.add_dist_atr.median(),
            avgdown_add_dist_atr_median=o.loc[o.add_dist_atr < 0, "add_dist_atr"].median(),
            add_gap_h_median=o.gap_h.median(),
            order_size_eq_median=o.size_eq.median(),
            lev_max_median=d.lev_max_at_open_eq.median(),
            lev_max_q90=q(d.lev_max_at_open_eq, 0.9),
            hold_h_median=d.hold_h.median(), hold_h_q75=q(d.hold_h, 0.75), hold_h_q90=q(d.hold_h, 0.9),
            win_rate=win.mean(),
            win_exit_profit_atr_median=d.loc[win, "exit_profit_atr"].median(),
            loss_exit_profit_atr_median=d.loc[~win, "exit_profit_atr"].median(),
            win_mfe_atr_median=d.loc[win, "mfe_atr"].median(),
            all_mae_atr_median=d.mae_atr.median(),
            win_mae_atr_q90=q(d.loc[win, "mae_atr"], 0.9),
            loss_mae_atr_median=d.loc[~win, "mae_atr"].median(),
        ))
    return pd.DataFrame(rows)


def mae_buckets(e: pd.DataFrame) -> pd.DataFrame:
    """MAE(ATR 배수) 구간별 승률·손익: 꼬리 손실 차단 문턱의 근거."""
    bins = [0, 2, 4, 6, 8, 10, 15, 20, 30, 1e9]
    e = e.copy()
    e["mae_bin"] = pd.cut(e["mae_atr"], bins, right=False)
    g = e.groupby(["period", "mae_bin"], observed=True).agg(n=("pnl_btc", "size"), win_rate=("pnl_btc", lambda s: (s > 0).mean()),
                                                         pnl_sum_btc=("pnl_btc", "sum"),
                                                         pnl_eq_sum=("pnl_eq", "sum"))
    return g.reset_index()


if __name__ == "__main__":
    e, od2 = build()
    e["pnl_eq"] = e["pnl_btc"] / e["equity_btc_0"]
    s = summarize(e, od2)
    mb = mae_buckets(e)
    s.to_csv(RESULTS / "s2_episode_atr_summary.csv", index=False)
    mb.to_csv(RESULTS / "s2_mae_buckets.csv", index=False)
    e.to_parquet(CACHE / "btc_episodes_atr.parquet", index=False)
    od2.to_parquet(CACHE / "btc_add_orders_atr.parquet", index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).T.to_string())
    print(mb[mb.period != "y2018"].round(3).to_string())
