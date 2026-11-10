"""2단계-6: 재현도. 고정 규칙집 A(R1)를 Bitstamp BTC 1시간봉에 돌린 포지션 경로 vs 그의 실제 BTC 순노출.

지표(PREREG 2절): 방향 일치율(둘 다 포지션 / 하나라도 포지션 / 전체), 레버리지 상관계수, 규칙 손익 대 실제 손익,
기준선(직전 1시간 역방향, 7일 추세, 항상 롱, 동전 던지기) 대비 방향 일치율.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bt_engine import Costs, run
from config import CACHE, RESULTS, SPLITS
from prices import bitstamp_bars
from rulebook import FROZEN

FLAT = 0.02


def rule_path(version: str = "A") -> pd.Series:
    bars = bitstamp_bars("1h")
    bars = bars[bars.index < pd.Timestamp("2022-01-01", tz="UTC")]
    costs = Costs(maker=0.0002, taker=0.0005, slip=0.0002, funding_on=False)
    res, tr = run(bars, FROZEN[version], costs, fund=None, trade_start=pd.Timestamp("2019-01-01", tz="UTC"))
    lev = res["lev"].copy()
    lev.index = lev.index + pd.Timedelta(hours=1)   # 봉 마감 시각 = 패널의 h
    return lev, tr


def his_daily_index(panel: pd.DataFrame) -> pd.Series:
    d = panel[["equity_btc", "net_dep_btc"]].resample("1D").last()
    r = (d.equity_btc.diff() - d.net_dep_btc.diff()) / d.equity_btc.shift(1)
    return r


def fidelity(panel: pd.DataFrame, lev_rule: pd.Series, period: str, lev_rule_b: pd.Series | None = None) -> dict:
    p = panel[panel.period == period].copy()
    p["L_rule"] = lev_rule.reindex(p.index).fillna(0.0)
    s_his = np.where(p.L.abs() >= FLAT, np.sign(p.L), 0)
    s_rule = np.where(p.L_rule.abs() >= FLAT, np.sign(p.L_rule), 0)
    both = (s_his != 0) & (s_rule != 0)
    either = (s_his != 0) | (s_rule != 0)
    his_in = s_his != 0
    # 시간별 역계약 수익률(BTC 기준): L × (1 − P_h / P_{h+1})
    ret_next = (1 - p["close"] / p["close"].shift(-1)).fillna(0.0)
    r_his = p["L"].clip(-20, 20) * ret_next
    r_rule = p["L_rule"] * ret_next
    led = his_daily_index(p)
    out = dict(
        period=period, hours=len(p),
        his_in_position_share=his_in.mean(), rule_in_position_share=(s_rule != 0).mean(),
        dir_agree_both_in=(s_his[both] == s_rule[both]).mean() if both.any() else np.nan, hours_both_in=int(both.sum()),
        state_agree_either_in=(s_his[either] == s_rule[either]).mean(),
        state_agree_all=(s_his == s_rule).mean(),
        corr_L=float(np.corrcoef(p["L"].clip(-20, 20), p["L_rule"])[0, 1]),
        r2_L=float(np.corrcoef(p["L"].clip(-20, 20), p["L_rule"])[0, 1] ** 2),
        his_mtm_return_btc=float(np.prod(1 + r_his) - 1),
        rule_mtm_return_btc=float(np.prod(1 + r_rule) - 1),
        his_ledger_return_btc=float(np.prod(1 + led.dropna()) - 1),
        # 기준선(그가 포지션이 있는 시간에서 방향 일치율)
        base_contra_1h=(s_his[his_in] == -np.sign(p["r_1h"].to_numpy()[his_in])).mean(),
        base_trend_7d=(s_his[his_in] == np.sign(p["r_7d"].to_numpy()[his_in])).mean(),
        base_always_long=(s_his[his_in] == 1).mean(),
        base_coin=0.5,
        rule_agree_on_his_in=(s_his[his_in] == s_rule[his_in]).mean(),
    )
    # 그의 MTM 손익을 규칙과 방향이 같은 시간/다른 시간으로 나눔(설명 못 하는 부분)
    same = both & (s_his == s_rule)
    diff = his_in & ~same
    out["his_mtm_logret_when_rule_agrees"] = float(np.log1p(r_his[same]).sum())
    out["his_mtm_logret_otherwise"] = float(np.log1p(r_his[diff]).sum())
    # 보조: 달러 기준 노출. BTC 증거금 계좌의 달러 노출 = 1 + L, USDT 규칙의 달러 노출 = L_rule
    x_his = 1 + p["L"].clip(-20, 20)
    out["his_usd_expo_mean"] = float(x_his.mean())
    out["his_usd_short_share"] = float((x_his < -FLAT).mean())
    out["his_usd_hedged_share"] = float((x_his.abs() < 0.2).mean())
    sx_his = np.where(x_his.abs() >= FLAT, np.sign(x_his), 0)
    for tag, lr in (("A", p["L_rule"]), ("B", None if lev_rule_b is None else lev_rule_b.reindex(p.index).fillna(0.0))):
        if lr is None:
            continue
        sx_rule = np.where(lr.abs() >= FLAT, np.sign(lr), 0)
        out[f"usd_state_agree_all_{tag}"] = float((sx_his == sx_rule).mean())
        out[f"usd_corr_{tag}"] = float(np.corrcoef(x_his, lr)[0, 1])
    out["usd_state_agree_all_always_long1x"] = float((sx_his == 1).mean())
    return out


def run_open_periods():
    panel = pd.read_parquet(CACHE / "panel_open.parquet")
    lev, tr = rule_path("A")
    lev_b, _ = rule_path("B")
    rows = [fidelity(panel, lev, per, lev_b) for per in ("train", "valid")]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "s2_fidelity_train_valid.csv", index=False)
    # 규칙 거래 요약(그의 에피소드와 비교용)
    t = tr[(tr.entry_t >= pd.Timestamp("2019-01-01", tz="UTC")) & (tr.entry_t < pd.Timestamp("2021-07-01", tz="UTC"))]
    t = t.assign(period=np.where(t.entry_t < pd.Timestamp("2021-01-01", tz="UTC"), "train", "valid"))
    summ = t.groupby("period").agg(n=("ret", "size"), win_rate=("ret", lambda s: (s > 0).mean()),
                                   hold_bars_median=("bars", "median"), tranches_median=("tranches", "median"),
                                   long_share=("dir", lambda s: (s > 0).mean()), sum_ret=("ret", "sum"))
    summ.to_csv(RESULTS / "s2_fidelity_rule_trades.csv")
    pd.set_option("display.width", 250)
    print(df.round(4).T.to_string())
    print(summ.round(4).to_string())
    return df


if __name__ == "__main__":
    run_open_periods()
