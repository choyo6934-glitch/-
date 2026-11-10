"""2단계 봉인 테스트(2021-07-01 ~ 2021-12-31). 한 번만 실행된다(results/SEALED_OPENED.json 잠금).

열기 전 고정한 것: 규칙집 R1(커밋 17bd00f), 결정별 선택 나무(s2_best_models.pkl), 기준선 정의(PREREG).
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from config import CACHE, RESULTS
from exp_log import log

LOCK = RESULTS / "SEALED_OPENED.json"


def main():
    if LOCK.exists():
        print("봉인 구간은 이미 열렸다:", LOCK.read_text())
        sys.exit(1)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    LOCK.write_text(json.dumps({"opened_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "git_head": commit,
                                "rulebook_commit": "17bd00f"}, ensure_ascii=False, indent=1))

    from s2_fidelity import fidelity, rule_path
    from s2_rules import prepare, task_rows, TASKS

    panel_open = pd.read_parquet(CACHE / "panel_open.parquet")
    panel_sealed = pd.read_parquet(CACHE / "panel_sealed.parquet")
    panel = pd.concat([panel_open, panel_sealed])
    # 1) 재현도
    lev_a, tr_a = rule_path("A")
    lev_b, _ = rule_path("B")
    fid = pd.DataFrame([fidelity(panel, lev_a, "sealed", lev_b)])
    fid.to_csv(RESULTS / "s2_fidelity_sealed.csv", index=False)
    # 2) 한 단계 예측(선택된 나무)
    with open(CACHE / "s2_best_models.pkl", "rb") as f:
        models = pickle.load(f)
    p = prepare(panel)
    ps = p[p.period == "sealed"]
    rows = []
    for label in ("best", "best_mkt"):
        for task, b in models[label].items():
            d, y = task_rows(ps, task)
            if not len(y) or y.nunique() < 2:
                rows.append(dict(model=label, task=task, n=len(y), base=y.mean() if len(y) else np.nan))
                continue
            X = d[b["feats"]].astype(float).fillna(-999)
            pr = b["model"].predict_proba(X)[:, 1]
            rows.append(dict(model=label, task=task, feats=b["set"], depth=b["depth"], n=len(y), base=y.mean(),
                             auc_sealed=roc_auc_score(y, pr), bacc_sealed=balanced_accuracy_score(y, pr > 0.5),
                             auc_valid=b["auc_valid"]))
    one = pd.DataFrame(rows)
    one.to_csv(RESULTS / "s2_onestep_sealed.csv", index=False)
    # 3) 방향 기준선(봉인)
    d, y = task_rows(ps, "direction")
    base = dict(period="sealed", n=len(y), long_share=y.mean(),
                acc_contra_1h=((d["r_1h"] < 0).astype(int) == y).mean(),
                acc_trend_7d=((d["r_7d"] > 0).astype(int) == y).mean(),
                acc_always_long=y.mean(), acc_always_short_posthoc=1 - y.mean(), acc_coin=0.5)
    pd.DataFrame([base]).to_csv(RESULTS / "s2_direction_baselines_sealed.csv", index=False)
    print(fid.round(4).T.to_string())
    print(one.round(3).to_string())
    print(base)
    log("s2_sealed", "SEALED", "봉인 구간 1회 개방: 재현도·한 단계 예측·방향 기준선", window="sealed", version="A",
        asset="BTC", metrics={k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                              for k, v in fid.iloc[0].items()}, decision="보고만(합격 판정 없음)")


if __name__ == "__main__":
    main()
