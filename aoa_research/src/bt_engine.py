"""백테스트 엔진(단일 자산, 한 번에 한 포지션). PREREG 4절의 체결 가정을 따른다.

- 신호는 봉 i 마감값으로 계산, 주문은 봉 i+1에서만 유효(유효기간 1봉, 미체결은 버림).
- 지정가 매수: low ≤ 가격×(1−pen) 일 때만 체결, 체결가 = min(가격, 시가). 매도는 대칭.
- 시장가(시간 손절·전환·한도 청산): 다음 봉 시가 ± 슬리피지, 테이커 수수료.
- 꼬리 손실 차단(스탑): 발동가와 시가 중 불리한 값 ± 슬리피지, 테이커.
- 같은 봉에서 모호하면 불리하게: 추가 체결 후 스탑 확인, 스탑이면 익절 없음, 추가와 익절이 같이 가능하면 추가만 체결.
- 자산 = 현금 + 수량×종가 (선물은 USDT 선형, 현물·주식은 수량 ≥ 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from numba import njit

REASONS = {1: "익절", 2: "꼬리손실차단", 3: "시간청산", 4: "반대신호(전환/청산)", 5: "일손실한도", 6: "낙폭중단",
           7: "기말평가", 8: "본전부근탈출"}


@dataclass
class Params:
    theta_L: float = 0.42      # 롱 셋업: 24봉 범위 위치 ≤ theta_L
    theta_S_gap: float = 0.44  # 숏 셋업: 범위 위치 ≥ 1 − theta_S_gap (= 0.56)
    d: float = 0.27            # 첫 지정가: 종가 ∓ d·ATR
    s: float = 0.48            # 추가 지정가: 평단 ∓ s·ATR0
    k_tp: float = 0.46         # 익절 지정가: 평단 ± k_tp·ATR0
    k_sl: float = 8.0          # 꼬리 손실 차단: 평단 ∓ k_sl·ATR0
    N: int = 4                 # 최대 분할 횟수
    u: float = 0.275           # 1회 크기(평가자산 대비 명목)
    T_max: int = 75            # 최대 보유 봉 수
    T_l: int = 0               # 이 봉 수 이상 보유하면 청산 목표를 평단 ∓ k_l·ATR0로 낮춤(0이면 사용 안 함)
    k_l: float = 0.42          # 손실 감수 탈출 폭(ATR0 배수)
    allow_short: bool = True
    flip: bool = True          # 반대 셋업 시 전환(A). False면 청산만(B·C)
    daily_loss: float = 0.0985
    halt_dd: float = 0.35
    halt_days: float = 30.0
    pen: float = 0.0001        # 지정가 관통 여유(1bp)
    range_n: int = 24          # 범위 창(봉)
    atr_n: int = 14

    def scaled(self, **mult):
        p = Params(**asdict(self))
        for k, m in mult.items():
            v = getattr(p, k) * m
            setattr(p, k, int(round(v)) if isinstance(getattr(self, k), int) and not isinstance(getattr(self, k), bool) else v)
        return p


@dataclass
class Costs:
    maker: float = 0.0002
    taker: float = 0.0005
    slip: float = 0.0002
    funding_on: bool = True
    funding_pay_mult: float = 1.0
    sell_tax: np.ndarray | None = None   # 봉별 매도 거래세율(주식)

    def times(self, k: float) -> "Costs":
        tax = None if self.sell_tax is None else self.sell_tax * k
        return Costs(self.maker * k, self.taker * k, self.slip * k, self.funding_on and k > 0,
                     1.0 if k <= 1 else k, tax)


def indicators(o, h, l, c, atr_n=14, range_n=24):
    n = len(c)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
    atr = np.full(n, np.nan)
    cs = np.cumsum(np.nan_to_num(tr))
    atr[atr_n:] = (cs[atr_n:] - cs[:-atr_n]) / atr_n
    import pandas as pd
    hi = pd.Series(h).rolling(range_n).max().to_numpy()
    lo = pd.Series(l).rolling(range_n).min().to_numpy()
    rng = hi - lo
    pos = np.where(rng > 0, (c - lo) / np.where(rng > 0, rng, 1.0), 0.5)
    pos[np.isnan(hi)] = np.nan
    m1 = (c - prev_c) / atr
    return atr, pos, m1


@njit(cache=True)
def _run(o, h, l, c, atr, pos, m1, day_id, tdays, fund, sell_tax, trade_ok,
         theta_L, theta_S, d, s, k_tp, k_sl, N, u, T_max, T_l, k_l, allow_short, flip, daily_loss, halt_dd, halt_days, pen,
         maker, taker, slip, funding_on, fund_pay_mult, init_eq):
    n = len(c)
    eq = np.full(n, np.nan)
    lev = np.zeros(n)
    # 거래 기록
    tr_ent = np.zeros(n, np.int64); tr_ex = np.zeros(n, np.int64); tr_dir = np.zeros(n, np.int64)
    tr_pnl = np.zeros(n); tr_ret = np.zeros(n); tr_maxexpo = np.zeros(n); tr_avgexpo = np.zeros(n)
    tr_k = np.zeros(n, np.int64); tr_reason = np.zeros(n, np.int64); tr_cost = np.zeros(n); tr_notional = np.zeros(n)
    ntr = 0
    cash = init_eq; qty = 0.0; avg = 0.0; a0 = 0.0; k = 0; held = 0
    ep_start = -1; ep_cash0 = 0.0; ep_eq0 = 0.0; ep_cost = 0.0; ep_maxexpo = 0.0; ep_expo_sum = 0.0; ep_notional = 0.0
    # 대기 주문
    pe_dir = 0; pe_px = 0.0; pe_qty = 0.0; pe_atr = 0.0
    pa = False; pa_px = 0.0; pa_qty = 0.0
    ptp = False; ptp_px = 0.0; ptp_scratch = False
    pmc = False; pmc_reason = 0
    day_start_eq = init_eq; cur_day = -1; block_day = -1
    peak = init_eq; halt_until = -1.0
    eq[0] = init_eq
    for i in range(1, n):
        oi = o[i]; hi = h[i]; li = l[i]; ci = c[i]
        closed_reason = 0
        fpx = 0.0
        # 0) 펀딩(봉 시가 시점 보유분)
        if funding_on and fund[i] != 0.0 and qty != 0.0:
            pay = qty * oi * fund[i]
            if pay > 0:
                pay *= fund_pay_mult
            cash -= pay
        # 1) 시장가 청산
        if pmc and qty != 0.0:
            px = oi * (1 - slip) if qty > 0 else oi * (1 + slip)
            fee = abs(qty) * px * (taker + (sell_tax[i] if qty > 0 else 0.0))
            cash += qty * px - fee
            ep_cost += fee + abs(qty) * oi * slip
            ep_notional += abs(qty) * px
            closed_reason = pmc_reason
            qty = 0.0
        pmc = False
        if closed_reason != 0:
            tr_ent[ntr] = ep_start; tr_ex[ntr] = i; tr_dir[ntr] = tr_dir[ntr]
            tr_pnl[ntr] = cash - ep_cash0; tr_ret[ntr] = (cash - ep_cash0) / ep_eq0
            tr_maxexpo[ntr] = ep_maxexpo; tr_avgexpo[ntr] = ep_expo_sum / max(held, 1); tr_k[ntr] = k
            tr_reason[ntr] = closed_reason; tr_cost[ntr] = ep_cost; tr_notional[ntr] = ep_notional
            ntr += 1
            k = 0; held = 0; closed_reason = 0
        # 2) 보유 중: 추가 → 스탑 → 익절
        if qty != 0.0:
            dirn = 1.0 if qty > 0 else -1.0
            add_hit = False
            if pa:
                if dirn > 0 and li <= pa_px * (1 - pen):
                    add_hit = True
                    fpx = min(pa_px, oi)
                elif dirn < 0 and hi >= pa_px * (1 + pen):
                    add_hit = True
                    fpx = max(pa_px, oi)
                if add_hit:
                    q = pa_qty * dirn
                    fee = abs(q) * fpx * maker
                    avg = (avg * abs(qty) + fpx * abs(q)) / (abs(qty) + abs(q))
                    cash -= q * fpx + fee
                    qty += q
                    k += 1
                    ep_cost += fee; ep_notional += abs(q) * fpx
            S = avg - k_sl * a0 if dirn > 0 else avg + k_sl * a0
            stop_hit = (dirn > 0 and li <= S) or (dirn < 0 and hi >= S)
            if stop_hit:
                if dirn > 0:
                    base = min(S, oi)
                    px = base * (1 - slip)
                else:
                    base = max(S, oi)
                    px = base * (1 + slip)
                fee = abs(qty) * px * (taker + (sell_tax[i] if qty > 0 else 0.0))
                cash += qty * px - fee
                ep_cost += fee + abs(qty) * base * slip; ep_notional += abs(qty) * px
                qty = 0.0
                closed_reason = 2
            elif ptp and not add_hit:
                tp_hit = (dirn > 0 and hi >= ptp_px * (1 + pen)) or (dirn < 0 and li <= ptp_px * (1 - pen))
                if tp_hit:
                    px = max(ptp_px, oi) if dirn > 0 else min(ptp_px, oi)
                    fee = abs(qty) * px * (maker + (sell_tax[i] if qty > 0 else 0.0))
                    cash += qty * px - fee
                    ep_cost += fee; ep_notional += abs(qty) * px
                    qty = 0.0
                    closed_reason = 8 if ptp_scratch else 1
            if closed_reason != 0:
                tr_ent[ntr] = ep_start; tr_ex[ntr] = i; tr_dir[ntr] = int(dirn)
                tr_pnl[ntr] = cash - ep_cash0; tr_ret[ntr] = (cash - ep_cash0) / ep_eq0
                tr_maxexpo[ntr] = ep_maxexpo; tr_avgexpo[ntr] = ep_expo_sum / max(held, 1); tr_k[ntr] = k
                tr_reason[ntr] = closed_reason; tr_cost[ntr] = ep_cost; tr_notional[ntr] = ep_notional
                ntr += 1
                k = 0; held = 0; closed_reason = 0
        # 3) 무포지션: 진입 지정가
        elif pe_dir != 0:
            hit = False
            if pe_dir > 0 and li <= pe_px * (1 - pen):
                hit = True
                fpx = min(pe_px, oi)
            elif pe_dir < 0 and hi >= pe_px * (1 + pen):
                hit = True
                fpx = max(pe_px, oi)
            if hit:
                eq_before = cash  # 무포지션이므로 자산 = 현금
                q = pe_qty * pe_dir
                fee = abs(q) * fpx * maker
                ep_start = i; ep_cash0 = cash; ep_eq0 = eq_before; ep_cost = fee; ep_maxexpo = 0.0
                ep_expo_sum = 0.0; ep_notional = abs(q) * fpx
                tr_dir[ntr] = pe_dir
                cash -= q * fpx + fee
                qty = q; avg = fpx; a0 = pe_atr; k = 1; held = 0
                # 같은 봉 스탑(불리한 순서 가정)
                S = avg - k_sl * a0 if pe_dir > 0 else avg + k_sl * a0
                if (pe_dir > 0 and li <= S) or (pe_dir < 0 and hi >= S):
                    base = S
                    px = base * (1 - slip) if pe_dir > 0 else base * (1 + slip)
                    fee2 = abs(qty) * px * (taker + (sell_tax[i] if qty > 0 else 0.0))
                    cash += qty * px - fee2
                    ep_cost += fee2 + abs(qty) * base * slip; ep_notional += abs(qty) * px
                    tr_ent[ntr] = ep_start; tr_ex[ntr] = i; tr_dir[ntr] = pe_dir
                    tr_pnl[ntr] = cash - ep_cash0; tr_ret[ntr] = (cash - ep_cash0) / ep_eq0
                    tr_maxexpo[ntr] = abs(q) * fpx / ep_eq0; tr_avgexpo[ntr] = abs(q) * fpx / ep_eq0; tr_k[ntr] = 1
                    tr_reason[ntr] = 2; tr_cost[ntr] = ep_cost; tr_notional[ntr] = ep_notional
                    ntr += 1
                    qty = 0.0; k = 0
        pe_dir = 0; pa = False; ptp = False
        # 4) 마감 평가
        e = cash + qty * ci
        eq[i] = e
        lev[i] = qty * ci / e if e > 0 else 0.0
        if qty != 0.0:
            held += 1
            expo = abs(qty) * ci / e if e > 0 else 0.0
            ep_expo_sum += expo
            if expo > ep_maxexpo:
                ep_maxexpo = expo
        if e <= 0:
            # 파산: 이후 거래 중단
            for j in range(i + 1, n):
                eq[j] = e
            break
        # 5) 일 손실 한도·낙폭 중단
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            day_start_eq = eq[i - 1] if not np.isnan(eq[i - 1]) else e
        if e > peak:
            peak = e
        if e <= day_start_eq * (1 - daily_loss):
            block_day = cur_day
            if qty != 0.0:
                pmc = True; pmc_reason = 5
        if e <= peak * (1 - halt_dd) and halt_until < tdays[i]:
            if qty != 0.0:
                pmc = True; pmc_reason = 6
            halt_until = tdays[i] + halt_days
            peak = e
        # 6) 시간 청산
        if qty != 0.0 and not pmc and held >= T_max:
            pmc = True; pmc_reason = 3
        # 7) 다음 봉 주문
        can_trade = trade_ok[i] and (block_day != cur_day) and (tdays[i] >= halt_until) and not np.isnan(atr[i]) \
            and not np.isnan(pos[i]) and atr[i] > 0
        if not can_trade:
            continue
        long_setup = pos[i] <= theta_L and m1[i] < 0
        short_setup = allow_short and pos[i] >= theta_S and m1[i] > 0
        if qty == 0.0 and not pmc:
            if long_setup:
                pe_dir = 1; pe_px = ci - d * atr[i]; pe_qty = u * e / pe_px; pe_atr = atr[i]
            elif short_setup:
                pe_dir = -1; pe_px = ci + d * atr[i]; pe_qty = u * e / pe_px; pe_atr = atr[i]
        elif qty != 0.0 and not pmc:
            dirn = 1 if qty > 0 else -1
            opposite = (dirn > 0 and pos[i] >= theta_S and m1[i] > 0) or (dirn < 0 and pos[i] <= theta_L and m1[i] < 0)
            if opposite:
                pmc = True; pmc_reason = 4
                if flip and ((dirn > 0 and allow_short) or dirn < 0):
                    pe_dir = -dirn
                    pe_px = ci + d * atr[i] if pe_dir < 0 else ci - d * atr[i]
                    pe_qty = u * e / pe_px; pe_atr = atr[i]
            else:
                if k < N:
                    pa = True
                    pa_px = avg - s * a0 if dirn > 0 else avg + s * a0
                    pa_qty = u * e / pa_px
                ptp = True
                ptp_scratch = T_l > 0 and held >= T_l
                if ptp_scratch:
                    # 오래 끈 포지션: 본전 부근(평단 ∓ k_l·ATR0)에서 빠져나온다(그의 손실 거래 청산 방식)
                    ptp_px = avg - k_l * a0 if dirn > 0 else avg + k_l * a0
                else:
                    ptp_px = avg + k_tp * a0 if dirn > 0 else avg - k_tp * a0
    # 기말 미청산은 마지막 종가로 평가해 기록
    if qty != 0.0 and ep_start >= 0:
        tr_ent[ntr] = ep_start; tr_ex[ntr] = n - 1; tr_dir[ntr] = 1 if qty > 0 else -1
        e = cash + qty * c[n - 1]
        tr_pnl[ntr] = e - ep_cash0; tr_ret[ntr] = (e - ep_cash0) / ep_eq0
        tr_maxexpo[ntr] = ep_maxexpo; tr_avgexpo[ntr] = ep_expo_sum / max(held, 1); tr_k[ntr] = k
        tr_reason[ntr] = 7; tr_cost[ntr] = ep_cost; tr_notional[ntr] = ep_notional
        ntr += 1
    return (eq, lev, tr_ent[:ntr], tr_ex[:ntr], tr_dir[:ntr], tr_pnl[:ntr], tr_ret[:ntr], tr_maxexpo[:ntr],
            tr_avgexpo[:ntr], tr_k[:ntr], tr_reason[:ntr], tr_cost[:ntr], tr_notional[:ntr])


def run(bars, p: Params, costs: Costs, fund=None, trade_start=None, init_eq=1.0, day_index=None):
    """bars: DataFrame(open, high, low, close), 색인 = 봉 시작 시각."""
    import pandas as pd
    o = bars["open"].to_numpy(float); h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float); c = bars["close"].to_numpy(float)
    atr, pos, m1 = indicators(o, h, l, c, p.atr_n, p.range_n)
    idx = bars.index
    # 날짜 계산은 해상도(ns/us)와 무관하게 Timedelta 나눗셈으로 한다
    epoch = pd.Timestamp("1970-01-01", tz=idx.tz) if idx.tz is not None else pd.Timestamp("1970-01-01")
    days_f = np.asarray((idx - epoch) / pd.Timedelta(days=1), dtype=float)
    day_id = np.floor(days_f).astype(np.int64) if day_index is None else day_index
    tdays = days_f - days_f[0]
    f = np.zeros(len(c)) if fund is None else fund
    tax = np.zeros(len(c)) if costs.sell_tax is None else costs.sell_tax
    ok = np.ones(len(c), dtype=np.bool_)
    if trade_start is not None:
        ok = (idx >= trade_start)
        ok = np.asarray(ok, dtype=np.bool_)
    out = _run(o, h, l, c, atr, pos, m1, day_id, tdays, f, tax, ok,
               p.theta_L, 1 - p.theta_S_gap, p.d, p.s, p.k_tp, p.k_sl, int(p.N), p.u, int(p.T_max), int(p.T_l), p.k_l,
               p.allow_short, p.flip, p.daily_loss, p.halt_dd, p.halt_days, p.pen,
               costs.maker, costs.taker, costs.slip, costs.funding_on, costs.funding_pay_mult, init_eq)
    eq, lev = out[0], out[1]
    trades = pd.DataFrame({"entry_i": out[2], "exit_i": out[3], "dir": out[4], "pnl": out[5], "ret": out[6],
                           "max_expo": out[7], "avg_expo": out[8], "tranches": out[9], "reason": out[10],
                           "cost": out[11], "notional": out[12]})
    if len(trades):
        trades["entry_t"] = idx[trades.entry_i.to_numpy()]
        trades["exit_t"] = idx[trades.exit_i.to_numpy()]
        trades["bars"] = trades.exit_i - trades.entry_i + 1
        trades["reason_txt"] = trades.reason.map(REASONS)
    res = pd.DataFrame({"equity": eq, "lev": lev}, index=idx)
    return res, trades
