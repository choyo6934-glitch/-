"""고정된 규칙집 파라미터(설계 R1). RULEBOOK.md와 같은 값이다. 이 파일은 봉인·표본 외 실행 전에 커밋한다.

모든 폭은 ATR 배수, 창은 봉 수. 수치 근거는 2019~2020 학습 구간의 그의 행동 통계(results/s2_*.csv).
"""
from __future__ import annotations

from bt_engine import Params

# 학습 구간 근거 (results/s2_h_entry_fills.csv, s2_episode_atr_summary.csv, s2_stop_counterfactual_train.csv)
EVIDENCE = {
    "theta_L": "매수 진입 체결의 24시간 범위 위치(금액 가중 평균) 0.4249",
    "theta_S_gap": "매도 진입 체결의 범위 위치 0.5639 → 1−0.5639 = 0.44",
    "d": "첫 진입 주문가와 직전 1시간 종가의 거리 중앙값 0.273 ATR",
    "s": "물타기 추가 주문의 평단 대비 거리 중앙값 0.478 ATR",
    "k_tp": "이익 에피소드의 청산 평균가−평단 중앙값 0.456 ATR",
    "k_sl": "그의 거래에 k ATR 손절을 가정한 반사실 손익이 최대인 k = 8 ATR",
    "N": "에피소드당 진입 주문 수 중앙값 4",
    "Lmax": "에피소드 최대 레버리지 중앙값 1.10배 (A). B·C는 레버리지 없음 1.0배",
    "T_max": "보유 시간 90% 분위 75.5시간",
    "T_l": "손실 에피소드 보유 시간 중앙값 11.6시간",
    "k_l": "손실 에피소드 청산 평균가−평단 중앙값 −0.42 ATR",
    "daily_loss": "그의 일간 수익률 1% 분위 −9.85%",
    "halt_dd": "그의 학습 구간 최대 낙폭 −35%",
}

BASE = dict(theta_L=0.42, theta_S_gap=0.44, d=0.27, s=0.48, k_tp=0.46, k_sl=8.0, N=4, T_max=75, T_l=12, k_l=0.42,
            daily_loss=0.0985, halt_dd=0.35, halt_days=30.0, pen=0.0001, range_n=24, atr_n=14)

FROZEN = {
    "A": Params(**BASE, u=1.10 / 4, allow_short=True, flip=True),
    "B": Params(**BASE, u=1.00 / 4, allow_short=False, flip=False),
    "C": Params(**BASE, u=1.00 / 4, allow_short=False, flip=False),
}

# ±20% 이웃 검사에 쓰는 핵심 파라미터(PREREG 8절, 백테스트 전에 고정)
NEIGHBOR_KEYS = ["theta_L", "theta_S_gap", "d", "s", "k_tp", "k_sl"]
# theta_L과 theta_S_gap은 '범위 문턱' 하나로 묶어 같은 배수를 준다 → 실질 파라미터 5개
NEIGHBOR_GROUPS = {"theta": ["theta_L", "theta_S_gap"], "d": ["d"], "s": ["s"], "k_tp": ["k_tp"], "k_sl": ["k_sl"]}

ASSETS = {"A": ["BTC", "ETH"], "B": ["BTC", "ETH"], "C": ["QQQ", "SPY", "069500.KS", "005930.KS"]}
