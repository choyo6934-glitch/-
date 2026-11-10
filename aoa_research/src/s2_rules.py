"""2단계-3: 결정별 규칙 추출(임계값 규칙·깊이 3 이하 결정나무)과 재현도.

- 학습(2019~2020)으로 적합, 검증(2021H1)으로 깊이·특징 묶음 선택. 봉인 구간은 s2_sealed.py에서 한 번만 연다.
- 결정: entry(무포지션→진입), direction(진입 방향), add(같은 방향 추가), tp(이익 중 축소), sl(손실 중 축소),
  exit_tp/exit_sl(전량 청산), flip(방향 전환).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text

from config import CACHE, RESULTS, SEED

FEATS_ATR = ["m_1h_atr", "m_4h_atr", "m_1d_atr", "m_7d_atr", "pos24", "pos7d", "range24_atr", "range168_atr"]
FEATS_MKT_EXTRA = ["r_15m", "r_1h", "r_4h", "r_1d", "r_7d", "atr_pct", "rv24", "rv168", "vol_ratio24",
                   "fund_bitmex", "hour", "dow"]
FEATS_POS = ["absL", "dist_avg_atr", "upnl_eq", "hold_h", "n_entry_orders", "h_since_entry_fill"]
FEATS_POS_STATE = ["absL", "dist_avg_atr", "upnl_eq", "hold_h", "n_entry_orders"]   # 본인 활동 시점 특징 제외
FEATS_FLAT = ["h_since_exit", "last_dir"]

_POS_TASK_SETS = {"atr": FEATS_ATR + FEATS_POS, "all": FEATS_ATR + FEATS_MKT_EXTRA + FEATS_POS,
                  "mkt_atr": FEATS_ATR + FEATS_POS_STATE, "mkt_all": FEATS_ATR + FEATS_MKT_EXTRA + FEATS_POS_STATE}
TASKS = {
    "entry": dict(sets={"atr": FEATS_ATR + FEATS_FLAT, "all": FEATS_ATR + FEATS_MKT_EXTRA + FEATS_FLAT,
                        "mkt_atr": FEATS_ATR, "mkt_all": FEATS_ATR + FEATS_MKT_EXTRA}),
    "direction": dict(sets={"atr": FEATS_ATR + FEATS_FLAT, "all": FEATS_ATR + FEATS_MKT_EXTRA + FEATS_FLAT,
                            "mkt_atr": FEATS_ATR, "mkt_all": FEATS_ATR + FEATS_MKT_EXTRA}),
    "add": dict(sets=_POS_TASK_SETS),
    "tp": dict(sets=_POS_TASK_SETS),
    "sl": dict(sets=_POS_TASK_SETS),
    "exit_tp": dict(sets=_POS_TASK_SETS),
    "exit_sl": dict(sets=_POS_TASK_SETS),
    "flip": dict(sets=_POS_TASK_SETS),
}


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["absL"] = p["L"].abs()
    # 무포지션 상태에서: 마지막으로 포지션이 있던 시각 이후 경과 시간과 그 방향
    inpos = ~p["flat"]
    last_in = pd.Series(np.where(inpos, np.arange(len(p)), np.nan), index=p.index).ffill()
    p["h_since_exit"] = np.where(inpos, 0.0, np.arange(len(p)) - last_in.to_numpy())
    p["h_since_exit"] = p["h_since_exit"].fillna(9999).clip(upper=24 * 30)
    last_dir = pd.Series(np.where(inpos, p["side"], np.nan), index=p.index).ffill().fillna(0)
    p["last_dir"] = np.where(inpos, p["side"], last_dir)
    nxt_flat = p["flat"].shift(-1).fillna(True).astype(bool)
    side_next = p["side_next"].fillna(0)
    p["y_entry"] = (p["flat"] & ~nxt_flat).astype(int)
    p["y_long"] = (side_next > 0).astype(int)
    same_dir_next = (side_next == p["side"]) & ~p["flat"]
    p["y_add"] = ((p["nx_inc_qty"] > 0) & same_dir_next).astype(int)
    p["y_reduce"] = (p["nx_dec_qty"] > 0).astype(int)
    p["y_exit"] = ((~p["flat"]) & nxt_flat).astype(int)
    p["y_flip"] = ((~p["flat"]) & (side_next == -p["side"])).astype(int)
    return p


def task_rows(p: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    ok = p["L_next"].notna()
    if task == "entry":
        d = p[ok & p["flat"]]
        y = d["y_entry"]
    elif task == "direction":
        d = p[ok & (p["y_entry"] == 1)]
        y = d["y_long"]
    elif task == "add":
        d = p[ok & ~p["flat"] & (p["y_flip"] == 0) & (p["y_exit"] == 0)]
        y = d["y_add"]
    elif task in ("tp", "exit_tp"):
        d = p[ok & ~p["flat"] & (p["upnl_eq"] > 0) & (p["y_flip"] == 0)]
        y = d["y_reduce"] if task == "tp" else d["y_exit"]
    elif task in ("sl", "exit_sl"):
        d = p[ok & ~p["flat"] & (p["upnl_eq"] <= 0) & (p["y_flip"] == 0)]
        y = d["y_reduce"] if task == "sl" else d["y_exit"]
    elif task == "flip":
        d = p[ok & ~p["flat"]]
        y = d["y_flip"]
    else:
        raise ValueError(task)
    return d, y


def prevalence_threshold(prob: np.ndarray, base_rate: float) -> float:
    """학습 구간 예측 양성 비율이 실제 기저율과 같아지는 확률 문턱값."""
    if base_rate <= 0:
        return 1.0
    return float(np.quantile(prob, 1 - base_rate))


def fit_task(p: pd.DataFrame, task: str):
    tr = p[p.period == "train"]
    va = p[p.period == "valid"]
    dtr, ytr = task_rows(tr, task)
    dva, yva = task_rows(va, task)
    rows = []
    best = None
    best_mkt = None
    for set_name, feats in TASKS[task]["sets"].items():
        Xtr = dtr[feats].astype(float).fillna(-999)
        Xva = dva[feats].astype(float).fillna(-999)
        for depth in (1, 2, 3):
            leaf = max(30, int(0.01 * len(Xtr)))
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight="balanced",
                                       random_state=SEED)
            m.fit(Xtr, ytr)
            ptr = m.predict_proba(Xtr)[:, 1]
            pva = m.predict_proba(Xva)[:, 1] if len(Xva) else np.array([])
            thr = prevalence_threshold(ptr, ytr.mean())
            auc_tr = roc_auc_score(ytr, ptr) if ytr.nunique() > 1 else np.nan
            auc_va = roc_auc_score(yva, pva) if len(yva) and yva.nunique() > 1 else np.nan
            ba_tr = balanced_accuracy_score(ytr, ptr > 0.5)
            ba_va = balanced_accuracy_score(yva, pva > 0.5) if len(yva) else np.nan
            r = dict(task=task, feats=set_name, depth=depth, n_train=len(ytr), n_valid=len(yva),
                     base_train=ytr.mean(), base_valid=yva.mean() if len(yva) else np.nan,
                     auc_train=auc_tr, auc_valid=auc_va, bacc_train=ba_tr, bacc_valid=ba_va, thr_prev=thr)
            rows.append(r)
            key = (round(float(np.nan_to_num(auc_va, nan=-1)), 3), -depth, set_name.endswith("atr"))
            cand = dict(model=m, feats=feats, set=set_name, depth=depth, thr_prev=thr, thr_half=0.5,
                        base_train=float(ytr.mean()), auc_valid=auc_va)
            if best is None or key > best[0]:
                best = (key, cand)
            if set_name.startswith("mkt") and (best_mkt is None or key > best_mkt[0]):
                best_mkt = (key, cand)
    return pd.DataFrame(rows), best[1], best_mkt[1]


def baselines_direction(p: pd.DataFrame) -> pd.DataFrame:
    """진입 방향 기준선(사전 고정): 직전 1시간 역방향, 7일 추세 방향, 항상 롱, 동전 던지기."""
    rows = []
    for per in ("train", "valid"):
        d, y = task_rows(p[p.period == per], "direction")
        contra = (d["r_1h"] < 0).astype(int)
        trend = (d["r_7d"] > 0).astype(int)
        rows.append(dict(period=per, n=len(y), long_share=y.mean(),
                         acc_contra_1h=(contra == y).mean(), acc_trend_7d=(trend == y).mean(),
                         acc_always_long=y.mean(), acc_always_short_posthoc=1 - y.mean(), acc_coin=0.5))
    return pd.DataFrame(rows)


def direction_threshold_scan(p: pd.DataFrame) -> pd.DataFrame:
    """단일 임계값 방향 규칙: 학습 구간에서 특징·문턱·부호를 고르고 검증 구간 정확도를 본다."""
    dtr, ytr = task_rows(p[p.period == "train"], "direction")
    dva, yva = task_rows(p[p.period == "valid"], "direction")
    feats = FEATS_ATR + FEATS_MKT_EXTRA
    rows = []
    for f in feats:
        x = dtr[f].astype(float)
        best = None
        for q in np.linspace(0.05, 0.95, 19):
            thr = float(np.nanquantile(x, q))
            for sgn in (1, -1):
                pred = (sgn * (x - thr) < 0).astype(int)   # sgn=1: x<thr 이면 롱
                acc = balanced_accuracy_score(ytr, pred)
                if best is None or acc > best[0]:
                    best = (acc, thr, sgn)
        acc, thr, sgn = best
        ptr = (sgn * (x - thr) < 0).astype(int)
        pv = (sgn * (dva[f].astype(float) - thr) < 0).astype(int)
        rows.append(dict(feature=f, rule=f"롱 if {f} {'<' if sgn == 1 else '>'} {thr:.4g}, 아니면 숏",
                         bacc_train=acc, acc_train=(ptr == ytr).mean(), bacc_valid=balanced_accuracy_score(yva, pv),
                         acc_valid=(pv == yva).mean(), long_pred_share_valid=pv.mean()))
    return pd.DataFrame(rows).sort_values("bacc_train", ascending=False)


def tree_rules_text(model, feats) -> str:
    return export_text(model, feature_names=list(feats), decimals=3, show_weights=True)


def run():
    p = prepare(pd.read_parquet(CACHE / "panel_open.parquet"))
    p = p[p.period.isin(["train", "valid"])]
    allm = []
    best = {}
    best_mkt = {}
    for task in TASKS:
        m, b, bm = fit_task(p, task)
        allm.append(m)
        best[task] = b
        best_mkt[task] = bm
    metrics = pd.concat(allm, ignore_index=True)
    metrics.to_csv(RESULTS / "s2_rule_model_selection.csv", index=False)
    base = baselines_direction(p)
    base.to_csv(RESULTS / "s2_direction_baselines.csv", index=False)
    with open(CACHE / "s2_best_models.pkl", "wb") as f:
        pickle.dump({"best": best, "best_mkt": best_mkt}, f)
    txt = []
    for label, dct in (("전체 특징 최선", best), ("시장 상태 전용 최선", best_mkt)):
        txt.append(f"\n## {label}\n")
        for task, b in dct.items():
            txt.append(f"### {task}  (특징={b['set']}, 깊이={b['depth']}, 검증 AUC={b['auc_valid']:.3f}, "
                       f"학습 기저율={b['base_train']:.4f}, 기저율 맞춤 문턱={b['thr_prev']:.3f})\n")
            txt.append(tree_rules_text(b["model"], b["feats"]))
    uni = direction_threshold_scan(p)
    uni.to_csv(RESULTS / "s2_direction_threshold_scan.csv", index=False)
    print(uni.head(15).round(3).to_string())
    (RESULTS / "s2_rules_trees.txt").write_text("\n".join(txt))
    pd.set_option("display.width", 250)
    print(metrics.round(3).to_string())
    print(base.round(3).to_string())
    print("\n".join(txt))
    return p, best


if __name__ == "__main__":
    run()
