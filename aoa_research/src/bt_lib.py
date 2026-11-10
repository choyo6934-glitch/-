"""백테스트 공통: 자산·비용 준비, 구간 지표, 비교 대상(B&H, C1, 무작위 진입), 부트스트랩."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bt_engine import Costs, Params, run
from config import SEED
from prices import binance_funding, binance_spot_1h, binance_um_1h, daily_stock

# ---------------------------------------------------------------- 자산과 비용(PREREG 5절)

SLIP = {"BTC": 0.0002, "ETH": 0.0003}


def kr_sell_tax(idx: pd.DatetimeIndex) -> np.ndarray:
    """코스피 상장 주식 매도 거래세(증권거래세+농특세) 연도별. ETF는 면제."""
    t = np.zeros(len(idx))
    d = pd.DatetimeIndex(idx)
    t[d < pd.Timestamp("2019-06-03")] = 0.0030
    t[(d >= pd.Timestamp("2019-06-03")) & (d < pd.Timestamp("2021-01-01"))] = 0.0025
    t[(d >= pd.Timestamp("2021-01-01")) & (d < pd.Timestamp("2023-01-01"))] = 0.0023
    t[(d >= pd.Timestamp("2023-01-01")) & (d < pd.Timestamp("2024-01-01"))] = 0.0020
    t[(d >= pd.Timestamp("2024-01-01")) & (d < pd.Timestamp("2025-01-01"))] = 0.0018
    t[(d >= pd.Timestamp("2025-01-01")) & (d < pd.Timestamp("2026-01-01"))] = 0.0015
    t[d >= pd.Timestamp("2026-01-01")] = 0.0020
    return t


def funding_array(bars: pd.DataFrame, symbol: str) -> np.ndarray:
    f = binance_funding(symbol)
    f = f.groupby(f.index.floor("h")).sum()
    return f.reindex(bars.index).fillna(0.0).to_numpy()


def load_asset(version: str, asset: str):
    """반환: bars, fund(배열 또는 None), base Costs, 연환산 일수."""
    if version == "A":
        sym = f"{asset}USDT"
        bars = binance_um_1h(sym)
        fund = funding_array(bars, sym)
        costs = Costs(maker=0.0002, taker=0.0005, slip=SLIP[asset], funding_on=True)
        return bars, fund, costs, 365
    if version == "B":
        bars = binance_spot_1h(f"{asset}USDT")
        costs = Costs(maker=0.0010, taker=0.0010, slip=SLIP[asset], funding_on=False)
        return bars, None, costs, 365
    if version == "C":
        bars = daily_stock(asset)
        if asset in ("QQQ", "SPY"):
            costs = Costs(maker=0.0007, taker=0.0007, slip=0.0002, funding_on=False)
        elif asset == "069500.KS":
            costs = Costs(maker=0.00015, taker=0.00015, slip=0.0005, funding_on=False)
        else:  # 삼성전자
            costs = Costs(maker=0.00015, taker=0.00015, slip=0.0005, funding_on=False, sell_tax=kr_sell_tax(bars.index))
        return bars, None, costs, 252
    raise ValueError(version)


def zero_costs(c: Costs, n: int) -> Costs:
    return Costs(0.0, 0.0, 0.0, False, 1.0, None if c.sell_tax is None else np.zeros(n))


def cost_case(c: Costs, k: float, n: int) -> Costs:
    if k == 0:
        return zero_costs(c, n)
    return c.times(k)


# ---------------------------------------------------------------- 지표

def _ts(x, tz):
    t = pd.Timestamp(x)
    if tz is not None and t.tzinfo is None:
        t = t.tz_localize(tz)
    if tz is None and t.tzinfo is not None:
        t = t.tz_localize(None)
    return t


def window_equity(res: pd.DataFrame, a, b) -> pd.Series:
    """구간 [a, b)의 자산곡선(구간 시작 직전 종가를 시작값으로 1로 정규화)."""
    tz = res.index.tz
    a, b = _ts(a, tz), _ts(b, tz)
    e = res["equity"]
    inside = e[(e.index >= a) & (e.index < b)]
    before = e[e.index < a]
    start = before.iloc[-1] if len(before) else inside.iloc[0]
    return inside / start, start


def daily_returns(eqw: pd.Series, start_val: float = 1.0, stock: bool = False) -> pd.Series:
    if stock:
        d = eqw
    else:
        d = eqw.resample("1D").last().dropna()
    prev = d.shift(1)
    prev.iloc[0] = start_val
    return d / prev - 1


def metrics(res: pd.DataFrame, trades: pd.DataFrame, a, b, ann: int, stock: bool = False) -> dict:
    eqw, _ = window_equity(res, a, b)
    eqw = eqw.dropna()
    if len(eqw) < 2:
        return {}
    r = daily_returns(eqw, 1.0, stock)
    days = (eqw.index[-1] - eqw.index[0]).total_seconds() / 86400 + (1 if stock else 1 / 24)
    total = eqw.iloc[-1] - 1
    cagr = (eqw.iloc[-1]) ** (365.25 / days) - 1 if eqw.iloc[-1] > 0 else -1.0
    curve = pd.concat([pd.Series([1.0]), eqw.reset_index(drop=True)])
    mdd = float((curve / curve.cummax() - 1).min())
    sharpe = float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else np.nan
    tz = res.index.tz
    aa, bb = _ts(a, tz), _ts(b, tz)
    t = trades[(trades.entry_t >= aa) & (trades.entry_t < bb)] if len(trades) else trades
    wins = t[t.ret > 0] if len(t) else t
    losses = t[t.ret <= 0] if len(t) else t
    gp = wins.ret.sum() if len(t) else 0.0
    gl = -losses.ret.sum() if len(t) else 0.0
    lev = res["lev"][(res.index >= aa) & (res.index < bb)]
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    out = dict(total_return=total, cagr=cagr, mdd=mdd, sharpe=sharpe, n_trades=len(t),
               win_rate=(t.ret > 0).mean() if len(t) else np.nan,
               payoff=(wins.ret.mean() / -losses.ret.mean()) if len(wins) and len(losses) and losses.ret.mean() < 0 else np.nan,
               pf=(gp / gl) if gl > 0 else (np.inf if gp > 0 else np.nan),
               sum_trade_ret=t.ret.sum() if len(t) else 0.0,
               exposure=float((lev.abs() > 1e-12).mean()) if len(lev) else np.nan,
               worst_day=float(r.min()), best_day=float(r.max()),
               avg_hold_bars=t.bars.mean() if len(t) else np.nan,
               long_share=(t.dir > 0).mean() if len(t) else np.nan)
    for y, v in yearly.items():
        out[f"y{y}"] = v
    return out


# ---------------------------------------------------------------- 비교 대상

def buy_hold(bars: pd.DataFrame, a, b, costs: Costs, ann: int, stock=False) -> tuple[dict, pd.DataFrame]:
    tz = bars.index.tz
    a, b = _ts(a, tz), _ts(b, tz)
    w = bars[(bars.index >= a) & (bars.index < b)]
    px0 = w["open"].iloc[0] * (1 + costs.slip)
    fee0 = costs.taker
    qty = (1 - fee0) / px0
    eq = qty * w["close"]
    tax_end = 0.0 if costs.sell_tax is None else float(costs.sell_tax[bars.index.get_indexer([w.index[-1]])[0]])
    eq.iloc[-1] = eq.iloc[-1] * (1 - costs.taker - costs.slip - tax_end)
    res = pd.DataFrame({"equity": eq, "lev": 1.0})
    # 구간 시작값 1을 기준으로
    before = pd.DataFrame({"equity": [1.0], "lev": [0.0]}, index=[a - pd.Timedelta(seconds=1)])
    res2 = pd.concat([before, res])
    m = metrics(res2, pd.DataFrame(columns=["entry_t", "ret", "dir", "bars"]), a, b, ann, stock)
    m["n_trades"] = 1
    return m, res2


def c1_benchmark(bars: pd.DataFrame, a, b, costs: Costs, ann: int, fund=None, stock=False,
                 lookback_days=30, vol_target=0.20, band=0.05) -> tuple[dict, pd.DataFrame]:
    """C1: 30일 추세(롱 전용) + 변동성 목표 20%, 레버리지 없음, 5%p 띠 바깥에서만 다음 봉 시가 조정."""
    tz = bars.index.tz
    a, b = _ts(a, tz), _ts(b, tz)
    if stock:
        daily = bars["close"]
    else:
        daily = bars["close"].resample("1D").last().dropna()
    lr = np.log(daily).diff()
    if stock:
        ref = daily.asof(daily.index - pd.Timedelta(days=lookback_days))
        ref = pd.Series(ref.to_numpy(), index=daily.index)
        vol = lr.rolling(21).std() * np.sqrt(ann)
    else:
        ref = daily.shift(lookback_days)
        vol = lr.rolling(lookback_days).std() * np.sqrt(ann)
    target = ((daily > ref) * np.minimum(1.0, vol_target / vol)).fillna(0.0)
    # 일 마감 신호 → 다음 봉 시가 실행
    if stock:
        sig_at_bar = target.shift(1).reindex(bars.index).fillna(0.0)
        is_exec = pd.Series(True, index=bars.index)
    else:
        nxt = target.copy()
        nxt.index = nxt.index + pd.Timedelta(days=1)   # 다음 날 00:00 봉 시가에 실행
        sig_at_bar = nxt.reindex(bars.index)
        is_exec = sig_at_bar.notna()
        sig_at_bar = sig_at_bar.ffill().fillna(0.0)
    w = bars[(bars.index >= a) & (bars.index < b)]
    o = w["open"].to_numpy(); c = w["close"].to_numpy()
    tg = sig_at_bar.reindex(w.index).to_numpy(); ex = is_exec.reindex(w.index).fillna(False).to_numpy()
    f = np.zeros(len(w)) if fund is None else pd.Series(fund, index=bars.index).reindex(w.index).fillna(0).to_numpy()
    tax = np.zeros(len(w)) if costs.sell_tax is None else pd.Series(costs.sell_tax, index=bars.index).reindex(w.index).to_numpy()
    cash, qty = 1.0, 0.0
    eq = np.zeros(len(w)); lev = np.zeros(len(w))
    n_reb = 0
    for i in range(len(w)):
        if costs.funding_on and f[i] != 0 and qty != 0:
            pay = qty * o[i] * f[i]
            cash -= pay * (costs.funding_pay_mult if pay > 0 else 1)
        if ex[i]:
            e_open = cash + qty * o[i]
            cur_w = qty * o[i] / e_open if e_open > 0 else 0
            if abs(tg[i] - cur_w) > band or (tg[i] == 0 and qty > 0):
                tq = tg[i] * e_open / o[i]
                dq = tq - qty
                px = o[i] * (1 + costs.slip) if dq > 0 else o[i] * (1 - costs.slip)
                fee = abs(dq) * px * (costs.taker + (tax[i] if dq < 0 else 0))
                cash -= dq * px + fee
                qty = tq
                n_reb += 1
        eq[i] = cash + qty * c[i]
        lev[i] = qty * c[i] / eq[i]
    res = pd.DataFrame({"equity": eq, "lev": lev}, index=w.index)
    before = pd.DataFrame({"equity": [1.0], "lev": [0.0]}, index=[a - pd.Timedelta(seconds=1)])
    res2 = pd.concat([before, res])
    m = metrics(res2, pd.DataFrame(columns=["entry_t", "ret", "dir", "bars"]), a, b, ann, stock)
    m["n_trades"] = n_reb
    return m, res2


def random_entries(bars: pd.DataFrame, trades: pd.DataFrame, a, b, cost_side: float, fund=None,
                   n_sim=1000, seed=SEED) -> np.ndarray:
    """전략 거래(방향, 보유 봉 수, 평균 노출)를 같은 구간 안에서 겹치지 않게 무작위 배치. 반환: 거래 순수익률 합계 분포."""
    tz = bars.index.tz
    a, b = _ts(a, tz), _ts(b, tz)
    w = bars[(bars.index >= a) & (bars.index < b)]
    o = w["open"].to_numpy()
    c_last = w["close"].to_numpy()[-1]
    W = len(w)
    t = trades[(trades.entry_t >= a) & (trades.entry_t < b)]
    if not len(t):
        return np.zeros(n_sim)
    dur = np.minimum(t["bars"].to_numpy().astype(int), W - 1)
    dirs = t["dir"].to_numpy().astype(float)
    expo = t["avg_expo"].to_numpy().astype(float)
    F = W - dur.sum()
    if F < 0:
        scale = (W - 1) / dur.sum()
        dur = np.maximum(1, np.floor(dur * scale).astype(int))
        F = W - dur.sum()
    f = np.zeros(W) if fund is None else pd.Series(fund, index=bars.index).reindex(w.index).fillna(0).to_numpy()
    cf = np.concatenate([[0.0], np.cumsum(f)])
    px_ext = np.concatenate([o, [c_last]])
    rng = np.random.default_rng(seed)
    out = np.zeros(n_sim)
    n = len(dur)
    for sim in range(n_sim):
        perm = rng.permutation(n)
        dd = dur[perm]
        gaps = np.sort(rng.integers(0, F + 1, size=n))
        starts = gaps + np.concatenate([[0], np.cumsum(dd)[:-1]])
        ends = np.minimum(starts + dd, W)
        r = dirs[perm] * expo[perm] * (px_ext[ends] / px_ext[starts] - 1)
        r -= expo[perm] * 2 * cost_side
        if fund is not None:
            # 엔진과 같게: 진입 다음 봉부터 청산 봉 시가까지의 펀딩(롱이 양수 지급)
            r -= dirs[perm] * expo[perm] * (cf[np.minimum(ends + 1, W)] - cf[np.minimum(starts + 1, W)])
        out[sim] = r.sum()
    return out


# ---------------------------------------------------------------- 부트스트랩

def stationary_bootstrap(r: np.ndarray, n_boot=2000, mean_block=10, seed=SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(r)
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    for bidx in range(n_boot):
        j = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                j = rng.integers(n)
            elif t > 0:
                j = (j + 1) % n
            idx[bidx, t] = j
    return r[idx]


def boot_ci(r: pd.Series, trades_ret: np.ndarray, ann: int, n_boot=2000, seed=SEED) -> dict:
    rr = r.to_numpy()
    if len(rr) < 20:
        return {}
    sims = stationary_bootstrap(rr, n_boot, 10, seed)
    eqs = np.cumprod(1 + sims, axis=1)
    years = len(rr) / ann
    cagr = eqs[:, -1] ** (1 / years) - 1
    sh = sims.mean(1) / sims.std(1) * np.sqrt(ann)
    mdd = (eqs / np.maximum.accumulate(eqs, axis=1) - 1).min(1)
    out = {"cagr_lo": np.nanquantile(cagr, 0.025), "cagr_hi": np.nanquantile(cagr, 0.975),
           "sharpe_lo": np.nanquantile(sh, 0.025), "sharpe_hi": np.nanquantile(sh, 0.975),
           "mdd_lo": np.nanquantile(mdd, 0.025), "mdd_hi": np.nanquantile(mdd, 0.975)}
    if len(trades_ret) >= 5:
        rng = np.random.default_rng(seed + 1)
        tb = trades_ret[rng.integers(0, len(trades_ret), size=(n_boot, len(trades_ret)))]
        gp = np.where(tb > 0, tb, 0).sum(1)
        gl = -np.where(tb <= 0, tb, 0).sum(1)
        pf = np.where(gl > 0, gp / np.where(gl > 0, gl, 1), np.inf)
        out.update(pf_lo=np.nanquantile(pf, 0.025), pf_hi=np.nanquantile(pf, 0.975),
                   win_lo=np.quantile((tb > 0).mean(1), 0.025), win_hi=np.quantile((tb > 0).mean(1), 0.975))
    return out
