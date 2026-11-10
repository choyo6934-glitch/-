"""4단계: 고정 규칙집(R1) 백테스트 — 봉인·표본 외 구간을 한 번에 실행한다.

버전 × 자산 × 구간 × 비용(0/기본/2배) 지표, B&H·C1 비교, 무작위 진입 1,000회, ±20% 이웃(3^5−1개),
부트스트랩 신뢰구간, 사전 고정 합격 기준 판정. 결과는 results/s4_*.csv.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

from bt_engine import Params, run
from bt_lib import boot_ci, buy_hold, c1_benchmark, cost_case, daily_returns, load_asset, metrics, random_entries, \
    window_equity
from config import BT_WINDOWS_COIN, BT_WINDOWS_STOCK, RESULTS, SEED
from exp_log import log
from rulebook import ASSETS, FROZEN, NEIGHBOR_GROUPS

COST_CASES = {"zero": 0.0, "base": 1.0, "double": 2.0}


def windows_for(version):
    return BT_WINDOWS_COIN if version in ("A", "B") else BT_WINDOWS_STOCK


def required_windows(version):
    return ["sealed", "oos"] if version in ("A", "B") else ["oos"]


def trade_start_for(version):
    return pd.Timestamp(BT_WINDOWS_COIN["dev"][0], tz="UTC") if version in ("A", "B") \
        else pd.Timestamp(BT_WINDOWS_STOCK["pre2022"][0])


def neighbor_params(p: Params):
    keys = list(NEIGHBOR_GROUPS)
    for combo in itertools.product([0.8, 1.0, 1.2], repeat=len(keys)):
        if all(m == 1.0 for m in combo):
            continue
        q = Params(**{**p.__dict__})
        for g, m in zip(keys, combo):
            for attr in NEIGHBOR_GROUPS[g]:
                setattr(q, attr, getattr(p, attr) * m)
        yield dict(zip(keys, combo)), q


def main():
    t0 = time.time()
    mrows, rrows, nrows, brows, vrows, trades_out = [], [], [], [], [], []
    for version, assets in ASSETS.items():
        stock = version == "C"
        wins = windows_for(version)
        for asset in assets:
            bars, fund, base_costs, ann = load_asset(version, asset)
            p = FROZEN[version]
            ts = trade_start_for(version)
            runs = {}
            for cname, k in COST_CASES.items():
                costs = cost_case(base_costs, k, len(bars))
                res, tr = run(bars, p, costs, fund=fund, trade_start=ts)
                runs[cname] = (res, tr, costs)
                for wname, (a, b) in wins.items():
                    m = metrics(res, tr, a, b, ann, stock)
                    bh, _ = buy_hold(bars, a, b, costs, ann, stock)
                    c1, _ = c1_benchmark(bars, a, b, costs, ann, fund=fund if version == "A" else None, stock=stock)
                    for who, mm in (("strategy", m), ("buy_hold", bh), ("C1", c1)):
                        mrows.append(dict(version=version, asset=asset, window=wname, cost=cname, who=who, **mm))
            res, tr, costs = runs["base"]
            tr2 = tr.copy()
            tr2.insert(0, "asset", asset)
            tr2.insert(0, "version", version)
            trades_out.append(tr2)
            # 무작위 진입(기본 비용), 필수 구간 + 개발/참고 구간
            for wname, (a, b) in wins.items():
                tz = bars.index.tz
                aa = pd.Timestamp(a, tz=tz) if tz else pd.Timestamp(a)
                bb = pd.Timestamp(b, tz=tz) if tz else pd.Timestamp(b)
                tw = tr[(tr.entry_t >= aa) & (tr.entry_t < bb)]
                if not len(tw):
                    rrows.append(dict(version=version, asset=asset, window=wname, n_trades=0))
                    continue
                cost_side = float(tw.cost.sum() / tw.notional.sum()) if tw.notional.sum() > 0 else 0.0
                sims = random_entries(bars, tr, a, b, cost_side, fund=fund if version == "A" else None,
                                      n_sim=1000, seed=SEED)
                stat = float(tw.ret.sum())
                rrows.append(dict(version=version, asset=asset, window=wname, n_trades=len(tw), strat_sum_ret=stat,
                                  rand_mean=float(sims.mean()), rand_p05=float(np.quantile(sims, 0.05)),
                                  rand_p50=float(np.quantile(sims, 0.5)), rand_p95=float(np.quantile(sims, 0.95)),
                                  percentile=float((sims < stat).mean()), cost_side=cost_side))
            # 부트스트랩(기본 비용)
            for wname, (a, b) in wins.items():
                eqw, _ = window_equity(res, a, b)
                r = daily_returns(eqw.dropna(), 1.0, stock)
                tz = bars.index.tz
                aa = pd.Timestamp(a, tz=tz) if tz else pd.Timestamp(a)
                bb = pd.Timestamp(b, tz=tz) if tz else pd.Timestamp(b)
                tw = tr[(tr.entry_t >= aa) & (tr.entry_t < bb)]
                ci = boot_ci(r, tw.ret.to_numpy(), ann)
                brows.append(dict(version=version, asset=asset, window=wname, **ci))
            # ±20% 이웃(표본 외 구간, 기본 비용)
            a, b = wins["oos"]
            for combo, q in neighbor_params(p):
                rq, tq = run(bars, q, base_costs, fund=fund, trade_start=ts)
                mq = metrics(rq, tq, a, b, ann, stock)
                nrows.append(dict(version=version, asset=asset, **{f"m_{k}": v for k, v in combo.items()},
                                  total_return=mq.get("total_return"), pf=mq.get("pf"), sharpe=mq.get("sharpe"),
                                  mdd=mq.get("mdd"), n_trades=mq.get("n_trades")))
            print(f"{version} {asset} done {time.time() - t0:.0f}s", flush=True)

    M = pd.DataFrame(mrows)
    R = pd.DataFrame(rrows)
    N = pd.DataFrame(nrows)
    B = pd.DataFrame(brows)
    T = pd.concat(trades_out, ignore_index=True)
    M.to_csv(RESULTS / "s4_metrics.csv", index=False)
    R.to_csv(RESULTS / "s4_random_entries.csv", index=False)
    N.to_csv(RESULTS / "s4_neighbors.csv", index=False)
    B.to_csv(RESULTS / "s4_bootstrap.csv", index=False)
    T.drop(columns=["entry_i", "exit_i"]).to_csv(RESULTS / "s4_trades_base.csv", index=False)

    # 합격 판정(PREREG 9절)
    for version, assets in ASSETS.items():
        for asset in assets:
            row = dict(version=version, asset=asset)
            ok_all = True
            for wname in required_windows(version):
                s = M[(M.version == version) & (M.asset == asset) & (M.window == wname) & (M.cost == "base")]
                st = s[s.who == "strategy"].iloc[0]
                bh = s[s.who == "buy_hold"].iloc[0]
                rr = R[(R.version == version) & (R.asset == asset) & (R.window == wname)].iloc[0]
                c1 = bool(st.n_trades >= 20 and st.pf >= 1.2)
                c2 = bool(st.sharpe > bh.sharpe and abs(st.mdd) < abs(bh.mdd))
                c3 = bool(rr.get("n_trades", 0) > 0 and rr.strat_sum_ret > rr.rand_p95)
                row.update({f"{wname}_pf": st.pf, f"{wname}_n": st.n_trades, f"{wname}_c1_pf": c1,
                            f"{wname}_sharpe": st.sharpe, f"{wname}_bh_sharpe": bh.sharpe, f"{wname}_mdd": st.mdd,
                            f"{wname}_bh_mdd": bh.mdd, f"{wname}_c2_sharpe_mdd": c2,
                            f"{wname}_rand_pct": rr.get("percentile"), f"{wname}_c3_random": c3})
                ok_all &= c1 and c2 and c3
            nb = N[(N.version == version) & (N.asset == asset)]
            share_pos = float((nb.total_return > 0).mean())
            c4 = share_pos > 0.5
            row.update(neighbors_positive_share=share_pos, c4_neighbors=c4)
            ok_all &= c4
            row["verdict"] = "합격" if ok_all else "불합격"
            vrows.append(row)
            log("s4", f"{version}-{asset}", "고정 규칙집 R1 최종 백테스트(봉인·표본 외 1회 실행)", window="+".join(required_windows(version)),
                version=version, asset=asset, cost="base",
                metrics={k: (float(v) if isinstance(v, (bool, int, float, np.floating, np.bool_)) else str(v)) for k, v in row.items()},
                decision=row["verdict"])
    V = pd.DataFrame(vrows)
    V.to_csv(RESULTS / "s4_verdict.csv", index=False)
    pd.set_option("display.width", 250)
    print(V.T.to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
