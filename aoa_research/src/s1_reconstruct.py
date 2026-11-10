"""1단계: 체결 → 포지션 재구성과 지갑 RealisedPNL 대조.

포지션 정의(PREREG 1절)
- 종목별로 Trade·Settlement 수량을 누적(매수 +, 매도 −). 0 → 0 구간이 포지션 1개.
- 한 체결이 0을 지나가면 비례 분할: 앞 부분은 기존 포지션을 닫고, 남는 부분이 새 포지션을 연다.
- Funding 행은 수량을 바꾸지 않고 그 시각에 열려 있는 포지션에 귀속.
- 현금흐름(사토시) = −execcost(Trade·Settlement) − execcomm(전체).

손익 대조
- BitMEX 방식(평균 원가)으로 실현 손익을 따로 계산해, 기말 미청산 포지션의 원가가 빠진 값을 지갑 RealisedPNL과 비교한다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import RESULTS, CACHE, SAT, ZERO_UUID
from load_aoa import load_executions, load_wallet


def wallet_balance_series(w: pd.DataFrame) -> pd.Series:
    """지갑 잔고(사토시)의 계단 함수. RealisedPNL은 그날 12:00 UTC, 입출금은 그날 00:00으로 둔다."""
    c = w[w.transactstatus == "Completed"].copy()
    c["book_t"] = c["date"] + pd.to_timedelta(np.where(c.transacttype == "RealisedPNL", 12, 0), unit="h")
    c = c.sort_values(["book_t", "file_row"], kind="mergesort")
    s = c.groupby("book_t")["amount"].sum().cumsum()
    return s


def reconstruct(df: pd.DataFrame):
    df = df.sort_values(["symbol", "t", "row_order"], kind="mergesort").reset_index(drop=True)
    sym = df["symbol"].to_numpy(dtype=object)
    et = df["exectype"].to_numpy(dtype=object)
    side = df["side"].fillna("").to_numpy(dtype=object)
    q_abs = df["lastqty"].fillna(0).to_numpy(dtype=float)
    px = df["lastpx"].to_numpy(dtype=float)
    cost = df["execcost"].fillna(0).to_numpy(dtype=float)
    comm = df["execcomm"].fillna(0).to_numpy(dtype=float)
    oid = df["orderid"].fillna("").to_numpy(dtype=object)
    liq = df["lastliquidityind"].fillna("").to_numpy(dtype=object)
    txt = df["text"].fillna("").to_numpy(dtype=object)
    ordtype = df["ordtype"].fillna("").to_numpy(dtype=object)
    t = df["t"].to_numpy()
    n = len(df)

    pos_id_of_row = np.full(n, -1, dtype=np.int64)       # 행이 귀속된 포지션 (분할 체결은 닫는 쪽)
    pos_id_open_of_row = np.full(n, -1, dtype=np.int64)  # 분할 체결의 새 포지션
    qty_after = np.zeros(n)
    realized_row = np.zeros(n)   # 행별 BitMEX식 실현액(매매 실현 + 수수료 + 펀딩), 사토시
    cost_after = np.zeros(n)     # 행 처리 후 열린 수량의 원가(execcost 합), 사토시

    positions = []
    sym_summary = {}

    cur_sym = None
    for i in range(n):
        s = sym[i]
        if s != cur_sym:
            if cur_sym is not None:
                sym_summary[cur_sym] = (qty, avg_cost, realized, fees, funding, cash)
            cur_sym = s
            qty = 0.0
            avg_cost = 0.0     # 열린 수량의 execcost 합 (BitMEX 평균 원가)
            realized = 0.0     # 평균 원가 기준 실현 매매손익(사토시)
            fees = 0.0
            funding = 0.0
            cash = 0.0
            cur = None
        e = et[i]
        if e == "Funding":
            f = -comm[i]
            funding += f
            cash += f
            realized_row[i] = f
            if cur is not None:
                cur["funding"] += f
                cur["n_funding"] += 1
                pos_id_of_row[i] = cur["pos_id"]
            qty_after[i] = qty
            cost_after[i] = avg_cost
            continue
        # Trade / Settlement
        sgn = 1.0 if side[i] == "Buy" else -1.0
        q = sgn * q_abs[i]
        c = cost[i]
        fee = -comm[i]
        fees += fee
        cash += -c + fee
        realized_row[i] = fee
        if q == 0:
            qty_after[i] = qty
            cost_after[i] = avg_cost
            continue
        xbt_per_ct = abs(c) / q_abs[i] / SAT if q_abs[i] > 0 else np.nan
        is_maker = liq[i] == "AddedLiquidity"
        if qty == 0 or np.sign(q) == np.sign(qty):
            # 진입 또는 추가
            if qty == 0:
                cur = dict(pos_id=len(positions), symbol=s, dir=int(np.sign(q)), t_open=t[i], t_close=None,
                           n_fills=0, n_entry_fills=0, n_exit_fills=0, entry_orders=[], exit_orders=[],
                           entry_qty=0.0, entry_xbt=0.0, entry_px_qty=0.0, exit_qty=0.0, exit_xbt=0.0,
                           exit_px_qty=0.0, max_abs_qty=0.0, max_notional_xbt=0.0,
                           trade_cash=0.0, fees=0.0, funding=0.0, n_funding=0,
                           maker_fills=0, maker_xbt=0.0, total_xbt=0.0, n_liq=0, n_stop=0, n_market=0,
                           n_settle=0)
                positions.append(cur)
            qty += q
            avg_cost += c
            cur["n_fills"] += 1
            cur["n_entry_fills"] += 1
            if not cur["entry_orders"] or cur["entry_orders"][-1] != oid[i]:
                cur["entry_orders"].append(oid[i])
            cur["entry_qty"] += abs(q)
            cur["entry_xbt"] += abs(c) / SAT
            cur["entry_px_qty"] += abs(q) * px[i]
            cur["trade_cash"] += -c
            cur["fees"] += fee
            cur["max_abs_qty"] = max(cur["max_abs_qty"], abs(qty))
            cur["max_notional_xbt"] = max(cur["max_notional_xbt"], abs(qty) * xbt_per_ct)
            cur["maker_fills"] += int(is_maker)
            cur["maker_xbt"] += abs(c) / SAT * is_maker
            cur["total_xbt"] += abs(c) / SAT
            cur["n_liq"] += int(txt[i] == "Liquidation")
            cur["n_stop"] += int(ordtype[i] in ("Stop", "StopLimit"))
            cur["n_market"] += int(ordtype[i] == "Market")
            cur["n_settle"] += int(e == "Settlement")
            pos_id_of_row[i] = cur["pos_id"]
            qty_after[i] = qty
            cost_after[i] = avg_cost
            continue
        # 감소(또는 방향 전환)
        close_q = min(abs(q), abs(qty))
        frac = close_q / abs(q)
        removed_cost = avg_cost * close_q / abs(qty)
        realized += -(removed_cost + frac * c)
        realized_row[i] += -(removed_cost + frac * c)
        avg_cost -= removed_cost
        qty += np.sign(q) * close_q
        cur["n_fills"] += 1
        cur["n_exit_fills"] += 1
        if not cur["exit_orders"] or cur["exit_orders"][-1] != oid[i]:
            cur["exit_orders"].append(oid[i])
        cur["exit_qty"] += close_q
        cur["exit_xbt"] += frac * abs(c) / SAT
        cur["exit_px_qty"] += close_q * px[i]
        cur["trade_cash"] += -frac * c
        cur["fees"] += frac * fee
        cur["maker_fills"] += int(is_maker)
        cur["maker_xbt"] += frac * abs(c) / SAT * is_maker
        cur["total_xbt"] += frac * abs(c) / SAT
        cur["n_liq"] += int(txt[i] == "Liquidation")
        cur["n_stop"] += int(ordtype[i] in ("Stop", "StopLimit"))
        cur["n_market"] += int(ordtype[i] == "Market")
        cur["n_settle"] += int(e == "Settlement")
        pos_id_of_row[i] = cur["pos_id"]
        if abs(qty) < 1e-9:
            qty = 0.0
            avg_cost = 0.0
            cur["t_close"] = t[i]
            cur = None
        rem = abs(q) - close_q
        if rem > 1e-9:
            # 방향 전환: 남는 수량으로 새 포지션
            q2 = np.sign(q) * rem
            c2 = (1 - frac) * c
            cur = dict(pos_id=len(positions), symbol=s, dir=int(np.sign(q2)), t_open=t[i], t_close=None,
                       n_fills=1, n_entry_fills=1, n_exit_fills=0, entry_orders=[oid[i]], exit_orders=[],
                       entry_qty=rem, entry_xbt=abs(c2) / SAT, entry_px_qty=rem * px[i], exit_qty=0.0,
                       exit_xbt=0.0, exit_px_qty=0.0, max_abs_qty=rem, max_notional_xbt=rem * xbt_per_ct,
                       trade_cash=-c2, fees=(1 - frac) * fee, funding=0.0, n_funding=0,
                       maker_fills=int(is_maker), maker_xbt=abs(c2) / SAT * is_maker, total_xbt=abs(c2) / SAT,
                       n_liq=int(txt[i] == "Liquidation"), n_stop=int(ordtype[i] in ("Stop", "StopLimit")),
                       n_market=int(ordtype[i] == "Market"), n_settle=int(e == "Settlement"), flipped_in=True)
            positions.append(cur)
            positions[cur["pos_id"] - 1]["flipped_out"] = True
            qty = q2
            avg_cost = c2
            pos_id_open_of_row[i] = cur["pos_id"]
        qty_after[i] = qty
        cost_after[i] = avg_cost
    sym_summary[cur_sym] = (qty, avg_cost, realized, fees, funding, cash)

    pos = pd.DataFrame(positions)
    pos["n_entry_orders"] = pos["entry_orders"].map(lambda x: len(set(x)))
    pos["n_exit_orders"] = pos["exit_orders"].map(lambda x: len(set(x)))
    pos = pos.drop(columns=["entry_orders", "exit_orders"])
    pos["pnl_xbt"] = (pos["trade_cash"] + pos["fees"] + pos["funding"]) / SAT
    pos["trade_pnl_xbt"] = pos["trade_cash"] / SAT
    pos["fees_xbt"] = pos["fees"] / SAT
    pos["funding_xbt"] = pos["funding"] / SAT
    pos["avg_entry_px"] = pos["entry_px_qty"] / pos["entry_qty"]
    pos["avg_exit_px"] = pos["exit_px_qty"] / pos["exit_qty"].replace(0, np.nan)
    pos["t_open"] = pd.to_datetime(pos["t_open"], utc=True)
    pos["t_close"] = pd.to_datetime(pos["t_close"], utc=True)
    pos["hold_h"] = (pos["t_close"] - pos["t_open"]).dt.total_seconds() / 3600
    pos["maker_share_xbt"] = pos["maker_xbt"] / pos["total_xbt"]
    for c in ("flipped_in", "flipped_out"):
        pos[c] = pos.get(c, False)
        pos[c] = pos[c].fillna(False).astype(bool)

    ss = pd.DataFrame.from_dict(sym_summary, orient="index",
                                columns=["end_qty", "open_cost", "realized_trade", "fees", "funding", "cash"])
    ss["bitmex_realised"] = ss["realized_trade"] + ss["fees"] + ss["funding"]
    df["pos_id"] = pos_id_of_row
    df["pos_id_open"] = pos_id_open_of_row
    df["qty_after"] = qty_after
    df["realized_sat"] = realized_row
    df["cost_after"] = cost_after
    return df, pos, ss


def daily_reconciliation(df: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """지갑 장부일 D는 (D−1 12:00, D 12:00] 사이 체결과 [D−1 12:00, D 12:00) 사이 펀딩을 담는다
    (12:00 정각 펀딩은 다음 날 장부로 간다). 종목·장부일별로 재구성 실현액과 비교."""
    d = df[["symbol", "t", "exectype", "realized_sat"]].copy()
    x = d["t"] - pd.Timedelta(hours=12)
    d["book_date"] = np.where(d["exectype"] == "Funding", x.dt.floor("D") + pd.Timedelta(days=1), x.dt.ceil("D"))
    d["book_date"] = pd.to_datetime(d["book_date"], utc=True)
    mine = d.groupby(["symbol", "book_date"])["realized_sat"].sum()
    comp = w[(w.transactstatus == "Completed") & (w.transacttype == "RealisedPNL")]
    led = comp.groupby(["address", "date"])["amount"].sum()
    led.index = led.index.set_names(["symbol", "book_date"])
    m = pd.concat([mine.rename("recon_sat"), led.rename("wallet_sat")], axis=1).fillna(0)
    m = m[(m.recon_sat != 0) | (m.wallet_sat != 0)]
    m["diff_sat"] = m["recon_sat"] - m["wallet_sat"]
    return m


def main():
    ex = load_executions()
    w = load_wallet()
    df, pos, ss = reconstruct(ex)

    # 지갑 대조
    comp = w[w.transactstatus == "Completed"]
    wr = comp[comp.transacttype == "RealisedPNL"].groupby("address")["amount"].sum()
    ss["wallet_realised"] = wr.reindex(ss.index).fillna(0)
    ss["diff_sat"] = ss["bitmex_realised"] - ss["wallet_realised"]
    for c in ["open_cost", "realized_trade", "fees", "funding", "cash", "bitmex_realised", "wallet_realised", "diff_sat"]:
        ss[c + "_btc"] = ss[c] / SAT
    ss = ss.sort_values("wallet_realised", ascending=False)
    extra_wallet_syms = sorted(set(wr.index) - set(ss.index))

    deposits = comp[comp.transacttype == "Deposit"]["amount"].sum() / SAT
    withdrawals = comp[comp.transacttype == "Withdrawal"]["amount"].sum() / SAT
    total_wallet_pnl = wr.sum() / SAT
    final_bal = comp["amount"].sum() / SAT
    last_wb = w["walletbalance"].dropna().iloc[-1] / SAT

    tr = ex[ex.exectype == "Trade"]
    n_orders = tr.loc[tr.orderid != ZERO_UUID, "orderid"].nunique()

    # 2% 잔량 제외 (진입 시점 지갑 잔고 기준)
    bal = wallet_balance_series(w)
    idx = bal.index.searchsorted(pos["t_open"], side="right") - 1
    pos["wallet_at_open_xbt"] = np.where(idx >= 0, bal.to_numpy()[np.clip(idx, 0, None)], np.nan) / SAT
    pos["size_ratio"] = pos["max_notional_xbt"] / pos["wallet_at_open_xbt"]
    pos["kept"] = pos["size_ratio"] >= 0.02
    n_all = len(pos)
    n_kept = int(pos["kept"].sum())

    open_end = pos[pos["t_close"].isna()]
    daily = daily_reconciliation(df, w)
    daily.to_csv(RESULTS / "s1_daily_reconciliation.csv")
    in_ledger = daily.reset_index()
    in_ledger = in_ledger[in_ledger.book_date <= w["date"].max()]
    summary_daily = dict(
        n_symbol_days=int(len(in_ledger)),
        n_exact=int((in_ledger.diff_sat.abs() < 0.5).sum()),
        n_within_1000sat=int((in_ledger.diff_sat.abs() <= 1000).sum()),
        max_abs_diff_btc=float(in_ledger.diff_sat.abs().max() / SAT),
        sum_abs_diff_btc=float(in_ledger.diff_sat.abs().sum() / SAT),
        unbooked_after_ledger_end_btc=float(daily.reset_index().query("book_date > @w.date.max()").recon_sat.sum() / SAT),
    )
    summary = dict(
        daily_reconciliation=summary_daily,
        n_exec_rows=int(len(ex)), n_trade_rows=int(len(tr)),
        n_funding_rows=int((ex.exectype == "Funding").sum()), n_settlement_rows=int((ex.exectype == "Settlement").sum()),
        n_orders_excl_zero_uuid=int(n_orders),
        deposits_btc=round(deposits, 8), withdrawals_btc=round(withdrawals, 8),
        wallet_realised_total_btc=round(total_wallet_pnl, 8),
        recon_bitmex_realised_total_btc=round(ss["bitmex_realised"].sum() / SAT, 8),
        recon_cash_total_btc=round(ss["cash"].sum() / SAT, 8),
        recon_diff_total_btc=round(ss["diff_sat"].sum() / SAT, 8),
        max_abs_symbol_diff_btc=round(ss["diff_sat"].abs().max() / SAT, 8),
        wallet_final_balance_btc=round(final_bal, 8), wallet_last_snapshot_btc=round(last_wb, 8),
        wallet_symbols_without_fills=extra_wallet_syms,
        n_positions_all=n_all, n_positions_kept_2pct=n_kept,
        n_positions_open_at_end=int(len(open_end)),
        open_at_end=open_end[["symbol", "dir", "t_open", "max_abs_qty"]].astype(str).to_dict("records"),
        end_qty_by_symbol={k: float(v) for k, v in ss["end_qty"].items() if v != 0},
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(ss[["end_qty", "open_cost_btc", "realized_trade_btc", "fees_btc", "funding_btc", "cash_btc",
              "bitmex_realised_btc", "wallet_realised_btc", "diff_sat_btc"]].round(6).to_string())

    RESULTS.mkdir(exist_ok=True, parents=True)
    ss.to_csv(RESULTS / "s1_symbol_reconciliation.csv")
    with open(RESULTS / "s1_summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=str)
    pos.to_parquet(CACHE / "positions.parquet", index=False)
    df[["row_order", "symbol", "t", "exectype", "pos_id", "pos_id_open", "qty_after", "cost_after",
        "realized_sat"]].to_parquet(
        CACHE / "exec_posmap.parquet", index=False)
    return df, pos, ss, summary


if __name__ == "__main__":
    main()
