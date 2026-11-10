"""3단계 개발 구간 점검: 규칙집 설계안을 코인 개발 구간(2020-02 ~ 2021-06)에서만 돌린다.

PREREG 3절: 개발 구간은 구현 오류 점검과 최대 5회 설계 수정에만 쓴다. 매 실행을 실험 기록에 남긴다.
봉인(2021H2)·표본 외(2022~) 구간 지표는 여기서 계산하지 않는다.
"""
from __future__ import annotations

import sys

import pandas as pd

from bt_engine import Params, run
from bt_lib import load_asset, metrics, buy_hold
from config import BT_WINDOWS_COIN
from exp_log import log

DEV = BT_WINDOWS_COIN["dev"]


def version_params(base: Params, version: str) -> Params:
    p = Params(**{**base.__dict__})
    if version == "A":
        p.allow_short, p.flip, p.u = True, True, base.u
    else:
        p.allow_short, p.flip = False, False
        p.u = min(base.u, 1.0 / base.N)   # 레버리지 없음
    return p


def dev_run(design_id: str, description: str, base: Params, record=True):
    rows = []
    for version in ("A", "B"):
        for asset in ("BTC", "ETH"):
            bars, fund, costs, ann = load_asset(version, asset)
            p = version_params(base, version)
            res, tr = run(bars, p, costs, fund=fund, trade_start=pd.Timestamp(DEV[0], tz="UTC"))
            m = metrics(res, tr, DEV[0], DEV[1], ann)
            bh, _ = buy_hold(bars, DEV[0], DEV[1], costs, ann)
            t = tr[(tr.entry_t >= pd.Timestamp(DEV[0], tz="UTC")) & (tr.entry_t < pd.Timestamp(DEV[1], tz="UTC"))]
            reasons = t.reason_txt.value_counts().to_dict() if len(t) else {}
            keep = {k: round(float(v), 4) for k, v in m.items() if k in
                    ("total_return", "cagr", "mdd", "sharpe", "n_trades", "win_rate", "payoff", "pf", "exposure",
                     "avg_hold_bars", "long_share", "worst_day")}
            keep["bh_sharpe"] = round(bh["sharpe"], 3)
            keep["bh_mdd"] = round(bh["mdd"], 3)
            keep["exit_reasons"] = reasons
            rows.append(dict(design=design_id, version=version, asset=asset, **keep))
            if record:
                log("s3_dev", design_id, description, window="dev", version=version, asset=asset, cost="base",
                    metrics=keep, decision="개발 구간 점검")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["exit_reasons"]).to_string())
    for r in rows:
        print(r["version"], r["asset"], r["exit_reasons"])
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A0"
    if which == "A0":
        dev_run("A0", "초기안: 학습 구간 행동 통계 그대로(손실 청산 규칙 없음). 날짜 해상도 버그 수정 후", Params())
    elif which == "R1":
        dev_run("R1", "수정1: 12봉(손실 에피소드 보유 중앙값) 이후 청산 목표를 평단∓0.42ATR(손실 에피소드 청산 중앙값)로",
                Params(T_l=12, k_l=0.42))
