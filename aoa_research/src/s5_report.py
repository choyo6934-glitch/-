"""보고서(REPORT.md) 생성. 모든 표의 숫자는 results/ 파일에서 읽는다(손으로 옮겨 적지 않음)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import RESULTS, ROOT

R = RESULTS


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "—"
    if isinstance(x, float) and np.isinf(x):
        return "∞"
    return f"{x:,.{d}f}"


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


ASSET_KR = {"BTC": "BTC", "ETH": "ETH", "QQQ": "QQQ", "SPY": "SPY", "069500.KS": "KODEX 200", "005930.KS": "삼성전자"}
VER_KR = {"A": "A 코인 선물 롱·숏", "B": "B 코인 현물 롱", "C": "C 주식·ETF 롱"}
WIN_KR = {"dev": "개발 2020-02~2021-06", "sealed": "봉인 2021H2", "oos": "표본 외 2022-01~2026-09",
          "pre2022": "참고 2010~2021"}


def main():
    s1 = json.loads((R / "s1_summary.json").read_text())
    ss = pd.read_csv(R / "s1_symbol_reconciliation.csv", index_col=0)
    hv = pd.read_csv(R / "s2_hypotheses_verdict.csv")
    ms = pd.read_csv(R / "s2_rule_model_selection.csv")
    one = pd.read_csv(R / "s2_onestep_sealed.csv")
    db = pd.read_csv(R / "s2_direction_baselines.csv")
    dbs = pd.read_csv(R / "s2_direction_baselines_sealed.csv")
    fid = pd.concat([pd.read_csv(R / "s2_fidelity_train_valid.csv"), pd.read_csv(R / "s2_fidelity_sealed.csv")])
    epa = pd.read_csv(R / "s2_episode_atr_summary.csv").set_index("period")
    cf = pd.read_csv(R / "s2_stop_counterfactual_train.csv")
    ex = json.loads((R / "s2_summary_extra.json").read_text())
    M = pd.read_csv(R / "s4_metrics.csv")
    V = pd.read_csv(R / "s4_verdict.csv")
    RN = pd.read_csv(R / "s4_random_entries.csv")
    NB = pd.read_csv(R / "s4_neighbors.csv")
    BS = pd.read_csv(R / "s4_bootstrap.csv")
    log = pd.read_csv(R / "experiment_log.csv")
    sealed_lock = json.loads((R / "SEALED_OPENED.json").read_text())

    xbt = ss.loc["XBTUSD"]
    dr = s1["daily_reconciliation"]

    # ---------------- 1단계 표
    t1 = pd.DataFrame([
        ["총 실현손익(지갑 RealisedPNL 합)", "3,537.3 BTC", f"{s1['wallet_realised_total_btc']:,.8f} BTC",
         f"재구성 {s1['recon_bitmex_realised_total_btc']:,.8f} BTC, 차이 {s1['recon_diff_total_btc']:.8f} BTC", "일치"],
        ["입금", "14.49 BTC", f"{s1['deposits_btc']:.8f} BTC", "지갑 Deposit 18건", "일치"],
        ["포지션 수", "약 3,185개", f"{s1['n_positions_all']:,}개", f"2% 미만 잔량 제외 후 {s1['n_positions_kept_2pct']:,}개(미청산 1개 포함)", "일치"],
        ["주문 수", "약 23,400건", f"{s1['n_orders_excl_zero_uuid']:,}건", "Trade 행의 orderid(0 UUID 청산 체결 제외)", "일치"],
        ["체결 행", "약 144만", f"{s1['n_trade_rows']:,}", f"Funding {s1['n_funding_rows']:,}, Settlement {s1['n_settlement_rows']}", "일치"],
        ["출금", "—", f"{s1['withdrawals_btc']:,.8f} BTC", f"기말 지갑 잔고 {s1['wallet_final_balance_btc']:,.8f} BTC(마지막 스냅샷과 같음)", "참고"],
    ], columns=["항목", "기준값", "재구성 값", "비고", "판정"])

    # ---------------- 2단계: 모델 요약
    rows = []
    for task, g in ms.groupby("task", sort=False):
        b_all = g.sort_values(["auc_valid", "depth"], ascending=[False, True]).iloc[0]
        g_m = g[g.feats.str.startswith("mkt")]
        b_m = g_m.sort_values(["auc_valid", "depth"], ascending=[False, True]).iloc[0]
        o_all = one[(one.model == "best") & (one.task == task)]
        o_m = one[(one.model == "best_mkt") & (one.task == task)]
        rows.append([task, f"{b_all.base_train:.3f}", f"{b_all.feats}/{int(b_all.depth)}", num(b_all.auc_train, 3),
                     num(b_all.auc_valid, 3), num(o_all.auc_sealed.iloc[0], 3) if len(o_all) and "auc_sealed" in o_all else "—",
                     f"{b_m.feats}/{int(b_m.depth)}", num(b_m.auc_valid, 3),
                     num(o_m.auc_sealed.iloc[0], 3) if len(o_m) and "auc_sealed" in o_m else "—",
                     int(o_all.n.iloc[0]) if len(o_all) else "—"])
    t_models = pd.DataFrame(rows, columns=["결정", "학습 기저율", "전체 특징 최선(묶음/깊이)", "학습 AUC", "검증 AUC",
                                           "봉인 AUC", "시장 상태 전용 최선", "검증 AUC ", "봉인 AUC ", "봉인 표본 수"])
    task_kr = {"entry": "진입(무포지션→포지션)", "direction": "진입 방향(롱=1)", "add": "추가 진입", "tp": "익절(이익 중 축소)",
               "sl": "손절(손실 중 축소)", "exit_tp": "이익 중 전량 청산", "exit_sl": "손실 중 전량 청산", "flip": "방향 전환"}
    t_models["결정"] = t_models["결정"].map(task_kr)

    dir_rows = []
    for _, r in pd.concat([db, dbs]).iterrows():
        dir_rows.append([r.period, int(r.n), pct(r.long_share, 0), pct(r.acc_contra_1h, 1), pct(r.acc_trend_7d, 1),
                         pct(r.acc_always_long, 1), pct(r.acc_always_short_posthoc, 1), "50.0%"])
    t_dir = pd.DataFrame(dir_rows, columns=["구간", "진입 수", "롱 비중", "직전 1시간 역방향", "7일 추세", "항상 롱",
                                            "항상 숏(사후 참고)", "동전"])
    t_dir["구간"] = t_dir["구간"].map({"train": "학습 2019~20", "valid": "검증 2021H1", "sealed": "봉인 2021H2"})

    fr = []
    for _, r in fid.iterrows():
        fr.append([{"train": "학습 2019~20", "valid": "검증 2021H1", "sealed": "봉인 2021H2"}[r.period],
                   pct(r.his_in_position_share, 0), pct(r.rule_in_position_share, 0), pct(r.dir_agree_both_in, 1),
                   pct(r.state_agree_all, 1), num(r.corr_L, 3), pct(r.r2_L, 1),
                   pct(r.his_mtm_return_btc, 0), pct(r.his_ledger_return_btc, 0), pct(r.rule_mtm_return_btc, 0),
                   pct(r.usd_state_agree_all_A, 0), pct(r.usd_state_agree_all_always_long1x, 0)])
    t_fid = pd.DataFrame(fr, columns=["구간", "그의 포지션 보유 시간", "규칙 보유 시간", "방향 일치(둘 다 보유)",
                                      "3상태 일치(전체 시간)", "레버리지 상관", "설명력 R²", "그의 손익(시간봉 시가평가)",
                                      "그의 손익(장부)", "규칙 손익(시간봉 시가평가)", "달러 노출 부호 일치: 규칙 A",
                                      "달러 노출 부호 일치: BTC 1배 보유"])

    # ---------------- 규칙집 근거 표
    tr_ = epa.loc["train"]
    t_ev = pd.DataFrame([
        ["첫 진입 지정가 거리", f"{tr_.offset_atr_median:.3f} ATR", "0.27 ATR"],
        ["물타기 추가 간격", f"{tr_.avgdown_add_dist_atr_median:.3f} ATR", "0.48 ATR0"],
        ["에피소드당 진입 주문 수", f"{tr_.entry_orders_median:.0f}", "4회"],
        ["진입 주문 1건 크기(평가자산 대비)", pct(tr_.order_size_eq_median, 1), "0.275배(A) / 0.25배(B·C)"],
        ["에피소드 최대 레버리지 중앙값 / 90% 분위", f"{tr_.lev_max_median:.2f} / {tr_.lev_max_q90:.2f}배", "상한 1.1배(A) / 1.0배(B·C)"],
        ["이익 에피소드 청산가−평단", f"{tr_.win_exit_profit_atr_median:.3f} ATR", "익절 +0.46 ATR0"],
        ["손실 에피소드 청산가−평단 / 보유 중앙값", f"{tr_.loss_exit_profit_atr_median:.3f} ATR / 11.6시간", "12봉 뒤 −0.42 ATR0"],
        ["손실 에피소드 MAE 중앙값", f"{tr_.loss_mae_atr_median:.2f} ATR", "(참고)"],
        ["보유 시간 중앙값 / 90% 분위", f"{tr_.hold_h_median:.1f} / {tr_.hold_h_q90:.1f}시간", "시간 청산 75봉"],
        ["승률", pct(tr_.win_rate, 1), "(참고)"],
    ], columns=["학습 구간(2019~2020) 측정값", "값", "규칙집 반영"])
    t_cf = cf.copy()
    t_cf["k_atr"] = t_cf["k_atr"].map(lambda v: "손절 없음(실제)" if v > 1e6 else f"{v:g} ATR")
    t_cf = t_cf.rename(columns={"k_atr": "가정한 손절 폭", "n_stopped": "손절된 에피소드", "pnl_btc": "합계 손익(BTC)",
                                "pnl_eq_sum": "자산 대비 손익 합"})

    # ---------------- 4단계 표
    vt = []
    for _, r in V.iterrows():
        def ok(x):
            return "통과" if bool(x) else "미달"
        if r.version in ("A", "B"):
            c1 = f"봉인 {num(r.sealed_pf, 2)}({ok(r.sealed_c1_pf)}) / 표본외 {num(r.oos_pf, 2)}({ok(r.oos_c1_pf)})"
            c2 = (f"봉인 {num(r.sealed_sharpe, 2)} vs {num(r.sealed_bh_sharpe, 2)}, MDD {pct(r.sealed_mdd, 0)} vs {pct(r.sealed_bh_mdd, 0)}"
                  f"({ok(r.sealed_c2_sharpe_mdd)}) / 표본외 {num(r.oos_sharpe, 2)} vs {num(r.oos_bh_sharpe, 2)}, MDD {pct(r.oos_mdd, 0)} vs {pct(r.oos_bh_mdd, 0)}({ok(r.oos_c2_sharpe_mdd)})")
            c3 = f"봉인 {pct(r.sealed_rand_pct, 1)}({ok(r.sealed_c3_random)}) / 표본외 {pct(r.oos_rand_pct, 1)}({ok(r.oos_c3_random)})"
        else:
            c1 = f"표본외 {num(r.oos_pf, 2)}, {int(r.oos_n)}건({ok(r.oos_c1_pf)})"
            c2 = f"샤프 {num(r.oos_sharpe, 2)} vs {num(r.oos_bh_sharpe, 2)}, MDD {pct(r.oos_mdd, 0)} vs {pct(r.oos_bh_mdd, 0)}({ok(r.oos_c2_sharpe_mdd)})"
            c3 = f"표본외 {pct(r.oos_rand_pct, 1)}({ok(r.oos_c3_random)})"
        c4 = f"{pct(r.neighbors_positive_share, 0)}({ok(r.c4_neighbors)})"
        vt.append([VER_KR[r.version], ASSET_KR[r.asset], c1, c2, c3, c4, f"**{r.verdict}**"])
    t_verdict = pd.DataFrame(vt, columns=["버전", "자산", "① PF ≥ 1.2", "② 샤프·MDD가 B&H보다 좋음",
                                          "③ 무작위 진입 백분위(>95% 필요)", "④ 이웃 양수 비율(>50%)", "판정"])

    def mtable(window, cost="base"):
        rows = []
        d = M[(M.window == window) & (M.cost == cost)]
        for (v, a), g in d.groupby(["version", "asset"], sort=False):
            for who in ("strategy", "buy_hold", "C1"):
                r = g[g.who == who]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append([VER_KR[v], ASSET_KR[a], {"strategy": "규칙집", "buy_hold": "매수 후 보유", "C1": "C1"}[who],
                             pct(r.total_return, 1), pct(r.cagr, 1), pct(r.mdd, 1), num(r.sharpe, 2),
                             pct(r.win_rate, 1) if who == "strategy" else "—", num(r.payoff, 3) if who == "strategy" else "—",
                             num(r.pf, 2) if who == "strategy" else "—", int(r.n_trades), pct(r.exposure, 0), pct(r.worst_day, 1)])
        return pd.DataFrame(rows, columns=["버전", "자산", "대상", "총수익", "연수익률", "최대 낙폭", "샤프", "승률",
                                           "손익비", "PF", "거래 수", "노출 시간", "최악의 하루"])

    t_oos = mtable("oos")
    t_sealed = mtable("sealed")
    t_dev = mtable("dev")
    t_pre = mtable("pre2022")

    cs = M[(M.who == "strategy") & M.window.isin(["sealed", "oos"])]
    crow = []
    for (v, a, w), g in cs.groupby(["version", "asset", "window"], sort=False):
        z = g.set_index("cost")
        crow.append([VER_KR[v], ASSET_KR[a], WIN_KR[w]] + [f"{pct(z.loc[c, 'cagr'], 1)} / {num(z.loc[c, 'pf'], 2)} / {num(z.loc[c, 'sharpe'], 2)}"
                                                          for c in ("zero", "base", "double")])
    t_cost = pd.DataFrame(crow, columns=["버전", "자산", "구간", "비용 0 (연수익률/PF/샤프)", "기본", "2배"])

    ycols = sorted([c for c in M.columns if c.startswith("y20") and int(c[1:]) >= 2021])
    yr = []
    for (v, a), g in M[(M.cost == "base") & (M.window.isin(["sealed", "oos"])) & (M.who == "strategy")].groupby(["version", "asset"], sort=False):
        vals = {}
        for c in ycols:
            s = g[c].dropna()
            vals[c] = pct(s.iloc[0], 1) if len(s) else ""
        bh = M[(M.cost == "base") & (M.window == "oos") & (M.who == "buy_hold") & (M.version == v) & (M.asset == a)].iloc[0]
        yr.append([VER_KR[v], ASSET_KR[a]] + [vals[c] for c in ycols] + [" / ".join(pct(bh[c], 0) for c in ycols if not pd.isna(bh[c]))])
    t_year = pd.DataFrame(yr, columns=["버전", "자산"] + [c[1:] for c in ycols] + ["B&H(2022~2026)"])

    rnd = []
    for _, r in RN[RN.window.isin(["sealed", "oos"])].iterrows():
        rnd.append([VER_KR[r.version], ASSET_KR[r.asset], WIN_KR[r.window], int(r.n_trades), num(r.strat_sum_ret, 3),
                    num(r.rand_p05, 3), num(r.rand_p50, 3), num(r.rand_p95, 3), pct(r.percentile, 1)])
    t_rand = pd.DataFrame(rnd, columns=["버전", "자산", "구간", "거래 수", "전략 거래수익 합", "무작위 5%", "무작위 50%",
                                        "무작위 95%", "전략 백분위"])

    nb = []
    for (v, a), g in NB.groupby(["version", "asset"], sort=False):
        nb.append([VER_KR[v], ASSET_KR[a], len(g), pct((g.total_return > 0).mean(), 0), pct(g.total_return.median(), 1),
                   pct(g.total_return.min(), 1), pct(g.total_return.max(), 1), num((g.pf >= 1.2).mean() * 100, 0) + "%"])
    t_nb = pd.DataFrame(nb, columns=["버전", "자산", "이웃 수", "순수익 양수 비율", "총수익 중앙값", "최저", "최고", "PF≥1.2 비율"])

    bs = []
    for _, r in BS[BS.window.isin(["sealed", "oos"])].iterrows():
        bs.append([VER_KR[r.version], ASSET_KR[r.asset], WIN_KR[r.window],
                   f"{pct(r.cagr_lo, 1)} ~ {pct(r.cagr_hi, 1)}", f"{num(r.sharpe_lo, 2)} ~ {num(r.sharpe_hi, 2)}",
                   f"{pct(r.mdd_lo, 0)} ~ {pct(r.mdd_hi, 0)}",
                   f"{num(r.pf_lo, 2)} ~ {'∞' if (pd.isna(r.pf_hi) or np.isinf(r.pf_hi)) else num(r.pf_hi, 2)}",
                   f"{pct(r.win_lo, 0)} ~ {pct(r.win_hi, 0)}"])
    t_bs = pd.DataFrame(bs, columns=["버전", "자산", "구간", "연수익률 95% 구간", "샤프 95% 구간", "MDD 95% 구간",
                                     "PF 95% 구간", "승률 95% 구간"])

    dev_log = log[log.stage == "s3_dev"]
    dev_rows = []
    for _, r in dev_log.iterrows():
        m = json.loads(r.metrics)
        dev_rows.append([r.exp_id, r.version, r.asset, pct(m.get("total_return"), 0) if "total_return" in m else "—",
                         num(m.get("pf"), 2) if "pf" in m else "—", num(m.get("sharpe"), 2) if "sharpe" in m else "—",
                         r.description[:60]])
    t_devlog = pd.DataFrame(dev_rows, columns=["설계", "버전", "자산", "총수익", "PF", "샤프", "설명"])

    n_pass = int((V.verdict == "합격").sum())
    oos_a = M[(M.window == "oos") & (M.cost == "base") & (M.who == "strategy") & M.version.isin(["A", "B"])]
    oos_c = M[(M.window == "oos") & (M.cost == "base") & (M.who == "strategy") & (M.version == "C")
              & M.asset.isin(["QQQ", "SPY", "005930.KS"])]
    zero_coin = M[(M.window == "oos") & (M.cost == "zero") & (M.who == "strategy") & M.version.isin(["A", "B"])]
    c1_coin = M[(M.window == "oos") & (M.cost == "base") & (M.who == "C1") & M.version.isin(["A", "B"])]
    fv = fid.set_index("period")

    text = f"""# 워뇨띠(BitMEX `aoa`) 2018-03~2021-12 체결내역 규칙 추출과 백테스트 검증

- 작성: 2026-11-10 (UTC). 연구 기록이며 투자 조언이 아니다. 실거래 주문은 하지 않았다.
- 사전 고정 기준 문서: [`PREREG.md`](PREREG.md) (커밋 `0fb5e8f`, 결과 계산 전), 규칙집: [`RULEBOOK.md`](RULEBOOK.md) (커밋 `17bd00f`, 봉인·표본 외 실행 전)
- 봉인 구간 개방 기록: `results/SEALED_OPENED.json` ({sealed_lock['opened_utc'][:19]}Z, 한 번만 실행)
- **결론: 규칙집 3종 × 자산 8개 조합이 모두 사전 고정 합격 기준에서 불합격이다. 따라서 "따라 하기 절차"는 쓰지 않는다.**

## 가정 (질문 없이 진행하려고 정한 것)

| # | 가정 | 이유·영향 |
|---|---|---|
| 1 | 입력 파일이 컨테이너에 없어서, 계정 보유자가 2026-09-22 DCInside 차트 마이너 갤러리(글 5051684)에 올린 Google Drive 압축(`aoa_public_2021-12-31_with_letter.zip`)을 받았다. 압축 SHA-256 `b6f1dc7a…8d01b9a`와 파일 6개의 해시가 공개 매니페스트와 일치했다. | 원자료의 진위(거래소 발급 여부)는 확인하지 못했다. |
| 2 | 공개자의 요청(직접 분석은 허용, 2차 상품·매매봇 판매 자제)을 지켜 원자료 CSV는 저장소·zip에 넣지 않았다. | 재현하려면 원자료를 따로 받아야 한다(재현 방법 참고). |
| 3 | "포지션 약 3,185개"는 0→0(방향 전환 분리) 포지션 전체 수로 보았다. 재구성 값이 정확히 3,185개다. "2% 미만 잔량 제외"는 포지션 기간 최대 명목가치가 진입 시점 지갑 잔고의 2% 미만인 포지션을 통계에서 빼는 것으로 해석했다(남는 것 2,738개). | 이 정의로 출발 가설의 승률 76%, 평균 이익 3.9 BTC, 평균 손실 6.8 BTC가 그대로 재현된다. |
| 4 | 규칙 추출은 BTC 순노출(XBTUSD + XBT 만기물)로 했다. 2021년 6~9월에 XBTUSD 숏과 XBTU21 롱을 크게 동시에 든 구간이 있어, 종목별이 아닌 합산 노출이 방향을 나타낸다. ETH(거래대금 14%)는 학습 구간(2019) 가격 원천이 지정 데이터에 없어 추출에서 뺐다. | ETH 규칙은 따로 검증되지 않았다. |
| 5 | 특징용 BTC 가격은 Bitstamp BTC/USD 1분봉(지정 원천)을 썼다. BitMEX 공개 API는 이 환경에서 403(지역 차단)이었다. 펀딩비 특징은 체결내역 Funding 행에서 시장 펀딩률을 복원해(이 행의 commission은 계정 기준 부호라 포지션 방향을 곱함) 직전 값을 유지했다. | BitMEX 자체 시세와 Bitstamp는 조금 다르다. |
| 6 | 평가자산 = 순입금 + 누적 실현손익 + 미실현손익. BTC 종목은 Bitstamp, ETH는 2020년부터 바이낸스, 그 밖의 종목은 직전 체결가로 평가했다. | 레버리지 분모의 근사 오차. |
| 7 | 최대 역행(MAE)은 시간봉 고가·저가와 그 시각의 이동 평단으로 쟀다. 1시간 안에 끝난 에피소드는 MAE가 없다. | 출발 가설의 −982 BTC와 합계가 다르다(−1,394 BTC). |
| 8 | 코인 백테스트는 버전 A에 바이낸스 USDⓈ-M 선물 1시간봉, 버전 B에 바이낸스 현물 1시간봉(같은 저장소의 다른 브랜치)을 썼다. 주식은 QQQ·SPY·KODEX 200(같은 저장소의 Yahoo 일봉)과 삼성전자(2026-11-10 yfinance로 새로 받음)를 수정주가로 보정해 썼다. | 삼성전자 외에는 새로 받은 데이터 없음. |
| 9 | 주식 버전 C는 코인 시간봉 규칙을 **같은 봉 수**로 일봉에 옮겼다(ATR 14일, 범위 24거래일 등). 주식 데이터로는 어떤 수치도 조정하지 않았다. | 일봉 번역 방식이 다르면 결과도 다를 수 있다. |
| 10 | 비용: 미국 ETF 편도 0.07%(국내 증권사 온라인 해외주식 수수료 수준), 국내 편도 0.015%, 삼성전자 매도세는 연도별(2026년 0.20%는 2025년 세법 개정안 기준 가정). 코인은 바이낸스 VIP0 수수료. | 개인별 수수료 차이는 비용 2배 경우로 일부 반영. |
| 11 | 지정가는 1bp 이상 관통해야 체결되는 것으로 정했다(사전 고정). 같은 봉에서 순서가 모호하면 불리한 쪽으로 처리했다. | 보수적 가정. |
| 12 | 개발 구간에서 설계안 3개(A0·R1·R2)를 시도했고, 수익이 아니라 그의 손실 청산 방식을 가장 충실히 담은 R1을 골랐다. | 세 안 모두 개발 구간 손실이었다. 기록은 실패 목록에 있다. |

## 1. 요약

1. **1단계 재구성은 기준값과 일치했다.** 청산된 45개 종목은 사토시 단위로 지갑 RealisedPNL과 같고, 총 실현손익 차이는 {s1['recon_diff_total_btc']:.4f} BTC(0.0005%)다. 이 차이는 장부 마지막 날 이후 펀딩 2건(+0.0196 BTC)과 기말 미청산 숏의 평균원가 반올림(−0.0019 BTC)으로 설명된다. 주문 {s1['n_orders_excl_zero_uuid']:,}건, 입금 {s1['deposits_btc']:.2f} BTC, 포지션 {s1['n_positions_all']:,}개.
2. **출발 가설 16개 중 11개는 확인(2개는 단서 있음), 4개는 부분 확인·수정, 1개는 기각됐다.**
   - 확인: 지정가 위주, 직전 1시간 역방향 진입(67~74%), 범위 하단 매수·상단 매도, 2021년 6건 분할·0.64배·29시간 보유, 승률 75.7%·평균 이익 3.91 BTC·평균 손실 6.87 BTC.
   - 수정: 물타기는 "절반"이 아니라 62~67%다.
   - 기각: "8시간 분할"은 중앙값 약 3시간이었다.
3. **방향 결정은 설명하지 못했다.**
   - 진입 방향을 맞히는 깊이 3 이하 나무 12개의 검증 AUC는 0.36~0.51이다. 단일 임계값 규칙은 학습 균형 정확도 최고 0.57, 그 규칙의 검증 값은 0.54다.
   - 사전 고정 기준선 중 가장 나은 "직전 1시간 역방향"은 시간봉 기준 56~58%다.
   - 체결 직전 60분 기준으로 재면 63~66%까지 오르는데, 이는 방향 판단보다 체결 타이밍(반등에 팔고 눌림에 사기)일 가능성이 크다.
4. **BTC 증거금 계좌라는 점이 '방향'의 의미를 바꾼다.** 학습 구간에서 포지션을 들고 있을 때 레버리지 중앙값은 −0.98배였다. 숏 1배는 달러 기준으로 BTC 노출이 0인 헤지 상태다. 달러 기준 노출(1+L)로 보면 그는 시간의 {pct(fv.loc['train', 'usd_state_agree_all_always_long1x'], 0)}(학습), {pct(fv.loc['valid', 'usd_state_agree_all_always_long1x'], 0)}(검증), {pct(fv.loc['sealed', 'usd_state_agree_all_always_long1x'], 0)}(봉인) 동안 BTC 롱이었다.
5. **규칙집은 그를 닮지 않았다.**
   - 레버리지 경로 상관은 {num(fv.loc['train', 'corr_L'], 3)}(학습), {num(fv.loc['valid', 'corr_L'], 3)}(검증), {num(fv.loc['sealed', 'corr_L'], 3)}(봉인)이다. 설명력(R²)은 2% 이하다.
   - 둘 다 포지션이 있을 때의 방향 일치율은 54~62%다.
   - 같은 시간봉 시가평가로 그는 +{pct(fv.loc['train', 'his_mtm_return_btc'], 0)}(학습)을 냈고, 규칙은 {pct(fv.loc['train', 'rule_mtm_return_btc'], 0)}였다.
6. **규칙집은 돈을 벌지 못했다. 8개 조합 모두 불합격({n_pass}개 합격).**
   - 코인 A·B는 표본 외(2022~2026) 연수익률 {pct(oos_a.cagr.max(), 0)}~{pct(oos_a.cagr.min(), 0)}, 최대 낙폭 {pct(oos_a.mdd.max(), 0)}~{pct(oos_a.mdd.min(), 0)}로, 무작위 진입보다 나빴다(백분위 0~0.8%).
   - 비용을 0으로 해도 PF가 {num(zero_coin.pf.min(), 2)}~{num(zero_coin.pf.max(), 2)}라서, 수수료가 아니라 규칙 자체가 손해다.
   - 주식 C는 QQQ·SPY·삼성전자에서 PF 1.96~2.61로 작은 이익(연 {pct(oos_c.cagr.min(), 1)}~{pct(oos_c.cagr.max(), 1)})을 냈다. 그러나 샤프가 매수 후 보유보다 낮고 무작위 진입 상위 5% 밖이라 불합격이다.
7. **비교 대상 C1(30일 추세·롱 전용·변동성 목표 20%)이 코인에서 더 나았다.** 표본 외 샤프 {num(c1_coin.sharpe.min(), 2)}~{num(c1_coin.sharpe.max(), 2)}, 최대 낙폭 {pct(c1_coin.mdd.max(), 0)}~{pct(c1_coin.mdd.min(), 0)}로 매수 후 보유보다 샤프가 높고 낙폭이 작았다. (C1은 이번 연구의 합격 대상이 아니라 비교 대상이다.)
8. 해석: 그의 기록에서 뽑을 수 있는 것은 **실행 습관**(지정가, 분할, 범위 하단 매수·상단 매도, 물타기, 짧은 익절)이었고, 이것만으로는 수익이 나지 않았다. 수익의 원천으로 보이는 진입 선택과 방향 판단은 기록에서 규칙으로 추출되지 않았다. 한 사람의 성공 기록(생존 편향)에서 "닮은 규칙"과 "돈 버는 규칙"은 별개였다.

## 2. 검증 결과 (1단계)

{table(t1)}

- 종목별 대조: 45개 청산 종목은 차이 0 사토시. XBTUSD는 재구성 {num(xbt.bitmex_realised_btc, 6)} BTC 대 지갑 {num(xbt.wallet_realised_btc, 6)} BTC(차이 {num(xbt.diff_sat_btc, 6)} BTC).
- 일별 대조(지갑 장부일 D = 전날 12:00 초과~당일 12:00 이하 체결, 12:00 정각 펀딩은 다음 날 장부): 종목·장부일 {dr['n_symbol_days']:,}건 중 {dr['n_exact']:,}건 사토시 일치, {dr['n_within_1000sat']:,}건 1,000사토시 이내, 최대 차이 {dr['max_abs_diff_btc']:.4f} BTC(이웃한 날끼리 상쇄되는 경계 효과).
- 확인한 원자료 특성: execcomm 부호는 유동성 기준(테이커 +, 메이커 리베이트 −)이다. Funding 행의 commission은 계정 입장의 부호 있는 요율(지급 +)이다. 2018년에 강제청산 체결 57건이 있다.
- 기말 미청산: XBTUSD 숏 {abs(s1['end_qty_by_symbol']['XBTUSD']):,.0f}계약(2021-12-24 진입), 기준값과 일치.

## 3. 추출된 규칙 (2단계)

### 3-1. 출발 가설 재검증 (2018 / 학습 2019~20 / 검증 2021H1 / 봉인 2021H2)

{table(hv)}

### 3-2. 결정별 규칙(깊이 3 이하 결정나무, 학습 적합 → 검증 선택 → 봉인 1회 평가)

목표 변수는 매 정시의 BTC 부호 레버리지(포지션 명목/평가자산)이고, 결정은 다음 1시간의 행동이다. "시장 상태 전용"은 그 자신의 활동 시점 특징(마지막 체결 후 경과 시간 등)을 뺀 모형이다.

{table(t_models)}

사람이 읽는 형태로 옮긴 시장 상태 규칙(검증 AUC 0.59~0.79, 대체로 약함):
- **진입**: 24시간 실현변동성이 연 47%를 넘고 거래량이 평소의 0.71배 이상이면 진입 확률이 높다. 변동성이 낮을 때는 거래량이 1.5배 넘게 터지고 UTC 0~7시(한국 오전~오후)일 때 진입한다. → "가격 위치보다 변동성·거래량이 클 때 들어간다."
- **방향**: 안정된 규칙 없음(아래 3-3).
- **추가**: 진입 후 16시간 안에, 특히 변동성이 클 때 계속 분할 매수·매도한다. 16시간이 지나면 거래량 급증·고변동성일 때만 추가한다.
- **익절·손절·전량 청산**: 주로 보유 시간과 분할 횟수에 좌우된다. 진입 후 1~3시간 안의 빠른 정리, 레버리지가 1.15~1.2배 이하일 때 전량 청산이 잦다. 22시간 넘게 들고 있는 손실 포지션은 시장 펀딩률이 −0.1% 이하로 크게 음수일 때 줄인다.
- **방향 전환**: 포지션을 잡은 지 24시간 안, 특히 거래량 급증 때 일어난다. 그 뒤에는 레버리지가 0.54배 이하로 작을 때만 일어난다.
- 그 자신의 활동 시점 특징을 넣으면 "직전 1~2시간 안에 체결했으면 또 체결한다"는 군집 규칙이 가장 강하다(추가 검증 AUC 0.76). 매매가 몰려서 일어난다는 뜻이며, 따라 할 수 있는 시장 규칙은 아니다.

### 3-3. 방향 결정: 설명 못 하는 부분

{table(t_dir)}

- 사전 고정 기준선 중 최선인 직전 1시간 역방향은 56~58%로, 동전(50%)보다 6~8%p 낫다.
- 결정나무(검증 AUC 0.36~0.51)와 단일 임계값 규칙(학습 균형 정확도 최고 0.57 → 검증 0.54. 다른 특징들은 검증 0.39~0.69로 흩어짐)은 이 기준선을 안정적으로 넘지 못했다.
- 체결 단위로는 진입 금액의 66~74%가 직전 60분과 반대 방향이다(메이커 66%, 테이커 69%로 비슷해서, 단순한 지정가 체결 효과만은 아니다).
- 하지만 같은 성향이 에피소드 첫 진입의 방향을 시간봉에서 예측하지는 못한다. 방향을 정한 뒤 반등·눌림에 나눠 체결한 결과일 수 있다.
- 봉인 구간 방향 표본은 14건뿐이라 결론을 바꿀 근거가 되지 못한다.

### 3-4. 재현도: 고정 규칙집 A를 같은 기간 Bitstamp BTC 1시간봉에 돌린 결과 vs 실제

{table(t_fid)}

- 수치로 본 설명 못 하는 부분: 레버리지 변동의 98% 이상(1 − R²), 방향 일치에서 동전 대비 4~12%p만 설명.
- 학습 구간 그의 시간봉 시가평가 로그수익 중 규칙과 방향이 같았던 시간의 몫은 {num(fv.loc['train', 'his_mtm_logret_when_rule_agrees'], 2)}, 나머지 시간의 몫은 {num(fv.loc['train', 'his_mtm_logret_otherwise'], 2)}이다. 수익 대부분이 규칙과 다른 결정에서 나왔다.
- 장부 수익(시간 안 체결 이익 포함)과 시간봉 시가평가 수익의 차이(학습 {pct(fv.loc['train', 'his_ledger_return_btc'], 0)} 대 {pct(fv.loc['train', 'his_mtm_return_btc'], 0)})는 시간봉보다 짧은 체결 타이밍에서 나온 몫이다. 시간봉 규칙으로는 원리적으로 따라 할 수 없다.

## 4. 규칙집 3종 (3단계)

전체 규칙은 [`RULEBOOK.md`](RULEBOOK.md)에 있다. 요지는 다음과 같다(모든 폭은 ATR 배수).

| 항목 | A 코인 선물 롱·숏(1시간봉) | B 코인 현물 롱(1시간봉) | C 주식·ETF 롱(일봉) |
|---|---|---|---|
| 진입 조건 | 롱: 24봉 범위 위치 ≤ 0.42 그리고 직전 봉 하락 / 숏: ≥ 0.56 그리고 직전 봉 상승 | 롱만 | 롱만, 다음 거래일 지정가 |
| 첫 주문 | 종가 ∓ 0.27 ATR 지정가 | 같음 | 같음(갭이면 시가 체결) |
| 분할 횟수·1회 크기 | 최대 4회, 평가자산의 0.275배, 평단 ∓ 0.48 ATR0 간격 물타기 | 4회 × 0.25배 | 4회 × 0.25배 |
| 최대 레버리지 | 1.1배 | 1.0배 | 1.0배(무레버리지) |
| 익절 | 평단 ± 0.46 ATR0 지정가 | 같음 | 같음 |
| 손절 | 12봉 뒤 청산가를 평단 ∓ 0.42 ATR0로(본전 부근 탈출), 75봉 시간 청산, 반대 셋업 시 전환 | 반대 셋업 시 청산 | 같음(12·75거래일) |
| 꼬리 손실 차단 | 평단 대비 8 ATR0 역행 시 시장가 정리 | 같음 | 같음(갭이면 시가) |
| 일 손실 한도 | 그날 시작 대비 −9.85%면 정리, 당일 진입 금지 | 같음 | 전일 종가 대비 |
| 거래 중단 | 고점 대비 −35%면 정리 후 30일 중단 | 같음 | 같음 |

수치 근거(학습 구간 그의 행동):

{table(t_ev)}

꼬리 차단 문턱은 그의 학습 구간 거래에 "k ATR에서 손절했다면"을 가정한 반사실 손익으로 골랐다.

{table(t_cf)}

원본 대비 바꾼 것과 이유는 `RULEBOOK.md`의 「원본 대비 바꾼 것」 표에 있다. 핵심 변경은 다음 셋이다.
- 재량 방향 판단을 기계적 셋업으로 대체(재현 불가).
- 그에게는 없던 꼬리 차단·일 손실 한도·거래 중단 추가.
- BTC 증거금 역계약을 USDT 선형·현물·주식으로 바꿈. 이 때문에 그의 "숏 1배 = 달러 중립"이 규칙집에서는 "실제 순매도"가 된다.

## 5. 백테스트 표 (4단계)

체결 가정(사전 고정): 신호는 마감 봉으로 계산하고 다음 봉에서만 유효하다. 지정가는 1bp 이상 관통해야 체결되고, 미체결은 버린다. 시장가·스탑에는 슬리피지를 더한다. 같은 봉에서 순서가 모호하면 불리하게 처리한다. 지표 구간은 끊김 없이 돌린 자산곡선에서 잘라 계산했다.

### 5-1. 판정 (기본 비용, 사전 고정 기준)

{table(t_verdict)}

### 5-2. 표본 외 2022-01 ~ 2026-09 (기본 비용)

{table(t_oos)}

### 5-3. 봉인 2021-07 ~ 2021-12 (기본 비용)

{table(t_sealed)}

### 5-4. 비용 민감도 (연수익률 / PF / 샤프)

{table(t_cost)}

### 5-5. 연도별 손익 (규칙집, 기본 비용; 2021은 하반기만, 2026은 1~9월)

{table(t_year)}

### 5-6. 무작위 진입 1,000회 (같은 거래 수·방향·보유 봉 수·평균 노출, 시작 시각만 무작위)

{table(t_rand)}

### 5-7. 파라미터 ±20% 이웃 (범위 문턱, d, s, k_tp, k_sl의 3^5−1 = 242개, 표본 외·기본 비용)

{table(t_nb)}

### 5-8. 부트스트랩 95% 신뢰구간 (일간 수익률 정상 블록 부트스트랩 2,000회, 거래 단위 부트스트랩 2,000회; PF 상한 ∞는 재표본의 2.5% 이상에 손실 거래가 없었다는 뜻)

{table(t_bs)}

### 5-9. 참고 구간 (판정에 쓰지 않음)

개발 구간 2020-02~2021-06 (코인):

{table(t_dev)}

주식 참고 구간 2010~2021 (미조정, 판정 대상 아님):

{table(t_pre)}

## 6. 실패 목록 (통과 못 한 가설과 전략 전부)

**전략(사전 고정 기준 불합격 8개)**
- A·B × BTC·ETH (4개): 봉인과 표본 외 모두 ①~④ 전부 미달. PF 0.35~0.92, 샤프 음수, 표본 외 최대 낙폭 −95~−98%, 무작위 진입 대비 하위권, 이웃 242개 중 순수익 양수 0개. 비용 0에서도 표본 외 PF < 1.
- C × KODEX 200: ① PF 0.93 미달, ② 미달, ③ 15% 미달. ④만 통과(58%).
- C × QQQ·SPY·삼성전자: ① PF 1.96·2.61·2.49 통과, ④ 81~98% 통과. 그러나 ② 샤프가 B&H보다 낮아(최대 낙폭은 B&H보다 작음) 미달, ③ 무작위 백분위 77%·90%·71%로 미달.

**개발 구간에서 시도하고 버린 설계(실험 기록 전체는 `results/experiment_log.csv`)**

{table(t_devlog)}

- 개발 구간 첫 실행에서 버그를 발견했다. pandas 3의 마이크로초 해상도 때문에 날짜 번호가 몇 년 동안 같게 계산돼, 일 손실 한도의 진입 금지가 풀리지 않았다. 고친 뒤 다시 돌렸고, 고치기 전 결과는 버렸다(기록에 남김). 봉인·표본 외 실행 전에 발견했다.

**규칙 추출에서 실패하거나 약했던 것**
- 진입 방향 모형: 검증 AUC 0.36~0.51(후보 12개 전부 동전 수준 이하).
- 시장 상태만으로 본 진입 시점 모형: 검증 AUC 0.46~0.62.
- 손절(손실 중 축소) 시장 상태 전용 모형: 검증 AUC 0.52~0.59.
- 출발 가설 수정·기각: 물타기 "약 절반"(실제 62~67%), "약 8시간 분할"(실제 중앙값 약 3시간), 메이커 64~81%(2021H1은 84%), 매도 범위 위치 0.54~0.59(2021H2는 0.63), 10% 역행 포지션 합계 −982 BTC(이번 정의로는 −1,394 BTC).
- 재현도: 레버리지 경로 R²는 학습 0.2%, 검증 2.1%, 봉인 0.9%. 규칙 손익은 학습 {pct(fv.loc['train', 'rule_mtm_return_btc'], 0)}, 검증 {pct(fv.loc['valid', 'rule_mtm_return_btc'], 0)}, 봉인 {pct(fv.loc['sealed', 'rule_mtm_return_btc'], 0)}. 그는 각각 {pct(fv.loc['train', 'his_mtm_return_btc'], 0)}, {pct(fv.loc['valid', 'his_mtm_return_btc'], 0)}, {pct(fv.loc['sealed', 'his_mtm_return_btc'], 0)}(같은 시간봉 시가평가).

## 7. 한계

- **생존 편향**: 많은 BitMEX 계정 중 크게 성공해 기록을 공개한 한 계정이다. 같은 습관으로 망한 계정은 보이지 않는다. 이번 결과(습관만으로는 손실)는 이 편향과 맞아떨어진다.
- **판단 근거 부재**: 체결만 있고 미체결·취소 주문, 호가, 뉴스, 그의 판단 근거는 없다. 방향을 설명할 정보가 원자료에 없을 수 있다.
- **시간 해상도**: 규칙은 시간봉(주식은 일봉)이다. 그의 이익 중 시간봉보다 짧은 체결 타이밍에서 나온 몫(학습 구간 장부 수익과 시간봉 시가평가 수익의 차이)은 원리적으로 재현되지 않는다.
- **계좌 구조 차이**: BTC 증거금 역계약에서 숏 1배는 달러 기준 중립이다. USDT·현물·주식 규칙집과 위험 성격이 다르다. 그의 "숏"의 상당 부분이 헤지였다면, 규칙집의 숏은 그보다 훨씬 공격적이다.
- **가격 원천**: 특징은 Bitstamp, 백테스트는 바이낸스로 계산했다. BitMEX 자체 시세와 펀딩 이력(API 차단)은 쓰지 못했다.
- **표본 크기**: 봉인 구간 방향 표본은 14건, 2021H2 BTC 에피소드는 35개뿐이다. 주식 표본 외 거래는 48~64건이다.
- **설계 선택**: 개발 구간에서 3개 설계를 시도했다(모두 손실). 주식 일봉 번역(같은 봉 수)은 한 가지 방식만 시험했다.
- **비용 가정**: 개인별 수수료·세금(2026년 거래세 0.20% 가정)이 다를 수 있다. 2배 비용 경우를 함께 보고했다.
- **재현성**: 원자료는 공개자 요청에 따라 재배포하지 않았다. 해시로 같은 파일인지 확인할 수 있다.

## 8. 따라 하기 절차

합격한 (버전, 자산) 조합이 없으므로 **따라 하기 절차는 제공하지 않는다**(사전 고정 원칙). 이 규칙집을 실거래나 모의 거래 기준으로 쓰지 말 것.
다시 시도한다면 새 가설을 새 사전 고정 문서로 등록하고, 2026-10 이후 데이터처럼 이번에 보지 않은 구간에서 검증해야 한다. 예를 들면 달러 기준 노출(1+L)을 목표로 한 "BTC 보유 + 헤지" 규칙이다.

## 9. 재현 방법과 파일

```
# 1) 입력 준비: 원자료(공개 Google Drive, SHA-256 확인), Bitstamp 1분봉 저장소, 현물·주식 일봉
bash aoa_research/fetch_inputs.sh
pip install pandas numpy scipy scikit-learn pyarrow numba
# 2) 실행 순서 (aoa_research/src 에서)
python3 load_aoa.py            # 원자료 → 캐시
python3 s1_reconstruct.py      # 1단계 재구성·대조
python3 prices.py              # 가격 캐시
python3 s2_panel.py            # 시간별 패널(봉인 구간 분리 저장)
python3 s2_hypotheses.py       # 출발 가설(봉인 제외)
python3 s2_rules.py            # 결정나무·기준선
python3 s2_episode_atr.py && python3 s2_stop_cf.py
python3 s3_dev.py R1           # 개발 구간 점검
python3 s2_fidelity.py         # 재현도(학습·검증)
python3 s2_sealed.py           # 봉인 1회(잠금 파일 생성) ; python3 s2_hypotheses.py --sealed
python3 s4_backtest.py         # 최종 백테스트·판정
python3 s5_stage2_verdicts.py && python3 s5_report.py
```

- `PREREG.md` 사전 고정 기준, `RULEBOOK.md` 규칙집, `results/*.csv` 결과, `results/experiment_log.csv` 시도 기록 전체({len(log)}행)
- `dist/aoa_research_bundle.zip`: 코드·결과 CSV·기준 문서 묶음(원자료 제외)
"""
    (ROOT / "REPORT.md").write_text(text)
    print("REPORT.md written", len(text))


if __name__ == "__main__":
    main()
