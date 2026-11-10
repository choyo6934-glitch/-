"""2단계 출발 가설 판정표(results/s2_hypotheses_verdict.csv)와 실험 기록 추가. 수치는 결과 파일에서 읽는다."""
from __future__ import annotations

import json

import pandas as pd

from config import RESULTS
from exp_log import log


def main():
    ot = pd.read_csv(RESULTS / "s2_h_order_types_with_sealed.csv").set_index("period")
    ef = pd.read_csv(RESULTS / "s2_h_entry_fills_with_sealed.csv").set_index("period")
    ep = pd.read_csv(RESULTS / "s2_h_episodes_with_sealed.csv").set_index("period")
    s1 = pd.read_csv(RESULTS / "s2_h_stage1_positions_with_sealed.csv").set_index("period")
    ex = json.loads((RESULTS / "s2_summary_extra.json").read_text())
    P = ["y2018", "train", "valid", "sealed"]

    def f(series, fmt="{:.2f}"):
        return " / ".join(fmt.format(series.get(p)) for p in P)

    rows = [
        ("H1a", "거래대금 99% 이상 지정가", f(ot["limit_value_share"] * 100, "{:.1f}%"),
         "2019년 이후 확인(99.7~100%). 2018년은 98.3%"),
        ("H1b", "메이커 비중 64~81%", f(ot["maker_value_share"] * 100, "{:.0f}%"),
         "부분 확인: 2021H1은 84%로 범위 밖"),
        ("H1c", "스탑 주문 거의 없음", f(ot["n_stop_orders"], "{:.0f}건"), "확인(2021년 0건)"),
        ("H2a", "진입 금액의 61~75%가 직전 1시간과 반대", f(ef["contra_1h_value_share"] * 100, "{:.0f}%"),
         "확인. 단 시간봉 단위 진입 방향 정확도는 56~58%로 낮음(체결 직전 수십 분의 타이밍 효과)"),
        ("H2b", "7일 추세 기준 약 50:50", f(ef["with_7d_trend_value_share"] * 100, "{:.0f}%"), "확인(43~49%)"),
        ("H3a", "매수는 24시간 범위 하단(평균 0.33~0.43)", f(ef["buy_pos24_mean_vw"]), "확인"),
        ("H3b", "매도는 상단(0.54~0.59)", f(ef["sell_pos24_mean_vw"]), "부분 확인: 2021H2는 0.63"),
        ("H4a", "2021년 포지션당 진입 주문 6건(중앙값)", f"2021 전체 {ex['stage1_pos_2021']['entry_orders_median']:.0f}건(종목별 포지션)",
         "확인"),
        ("H4b", "약 8시간에 분할", f"2021 BTC 에피소드 첫~마지막 진입 {ex['btc_ep_2021']['entry_span_h_median']:.1f}시간",
         "기각: 중앙값 약 3시간"),
        ("H4c", "1건은 잔고의 약 11%", f"2021 BTC 진입 주문 {ex['btc_ep_2021']['order_size_eq_median'] * 100:.1f}%",
         "대체로 확인(14.5%, H2만 10.4%)"),
        ("H4d", "레버리지 중앙값 0.66배", f"2021 종목별 포지션 {ex['stage1_pos_2021']['lev_max_median']:.2f}배", "확인"),
        ("H4e", "보유 중앙값 29시간", f"2021 종목별 포지션 {ex['stage1_pos_2021']['hold_h_median']:.1f}시간", "확인"),
        ("H5", "추가 진입의 약 절반이 물타기", f(ep["adds_avg_down_share"] * 100, "{:.0f}%"),
         "수정: 절반보다 많음(2019~2021 62~67%)"),
        ("H6", "승률 76%, 평균 이익 3.9 BTC 대 평균 손실 6.8 BTC",
         f"전 기간 2% 이상 포지션 {ex['stage1_all']['win_rate'] * 100:.1f}%, {ex['stage1_all']['avg_win']:.2f} 대 {ex['stage1_all']['avg_loss']:.2f}",
         "확인"),
        ("H7", "평단 대비 10% 넘게 밀린 포지션: 승률 24%, 합계 −982 BTC",
         f"BTC 에피소드 {ex['mae10_all']['n']}개, 승률 {ex['mae10_all']['win_rate'] * 100:.1f}%, 합계 {ex['mae10_all']['pnl_sum']:.0f} BTC",
         "승률은 확인, 합계는 MAE 정의(시간봉 고저가·이동 평단)에 따라 더 큼"),
        ("H8", "2018년은 13분 보유·3.7배 단타", f"보유 {s1.loc['y2018', 'hold_min_median']:.1f}분, 레버리지 {s1.loc['y2018', 'lev_max_median']:.2f}배(종목별)",
         "확인"),
    ]
    df = pd.DataFrame(rows, columns=["id", "출발 가설", "측정값(2018 / 2019~20 / 2021H1 / 2021H2 또는 별도 표기)", "판정"])
    df.to_csv(RESULTS / "s2_hypotheses_verdict.csv", index=False)
    for r in rows:
        log("s2", r[0], f"출발 가설 재검증: {r[1]}", window="2018/train/valid/sealed", version="", asset="BTC",
            metrics={"measured": r[2]}, decision=r[3])
    ms = pd.read_csv(RESULTS / "s2_rule_model_selection.csv")
    for task, g in ms.groupby("task"):
        best = g.sort_values("auc_valid", ascending=False).iloc[0]
        log("s2", f"tree-{task}", f"결정나무(깊이≤3) {task}: 후보 {len(g)}개 중 검증 AUC 최고",
            window="train→valid", asset="BTC",
            metrics={"best_feats": best.feats, "depth": int(best.depth), "auc_valid": float(best.auc_valid),
                     "n_candidates": int(len(g))},
            decision="방향은 AUC≈0.5로 설명 못 함" if task == "direction" else "약한 설명력")
    print(df.to_string())


if __name__ == "__main__":
    main()
