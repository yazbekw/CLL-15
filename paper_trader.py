"""Paper trading engine — checks signal, manages positions.

This mirrors backtest_core.run_backtest logic exactly, but executes live.
"""
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config_paper import (
    SYMBOLS, STARTING_EQUITY, MAX_CONCURRENT_POSITIONS, MAX_GROSS_EXPOSURE,
    DAILY_STOP, COOLDOWN_BARS_AFTER_LOSS, ROUND_TRIP_COST, MAX_CORR,
    MIN_HOLD_BARS, MAX_HOLD_BARS, TRAIL_ACTIVATION_ATR, ATR_TRAIL_MULT,
    MIN_DAILY_VOLUME_USD, MIN_FUNDING_ADVERSE,
    BTC_SHOCK_1H, BTC_SHOCK_1D,
)
from data import load_all
from strategy_trend import TrendV2
import paper_db as db


def _correlation_matrix(data, ts, lookback_bars=180):
    rets = {}
    for s, df in data.items():
        if ts not in df.index:
            continue
        pos = df.index.get_loc(ts)
        start = max(0, pos - lookback_bars)
        sub = df.iloc[start:pos]
        if len(sub) < 30:
            continue
        rets[s] = np.log(sub["close"]).diff().dropna()
    if len(rets) < 2:
        return pd.DataFrame()
    return pd.DataFrame(rets).corr()


def _load_state():
    equity = float(db.get_state("equity", STARTING_EQUITY))
    peak = float(db.get_state("peak_equity", STARTING_EQUITY))
    last_bar_ts = int(db.get_state("last_bar_ts", "0"))
    return equity, peak, last_bar_ts


def run_check(verbose=True) -> dict:
    """Run one paper-trading cycle. Returns summary dict."""
    t0 = time.time()
    log = []
    def _log(msg):
        if verbose:
            print(msg)
        log.append(msg)

    _log(f"\n=== paper check @ {datetime.now(timezone.utc).isoformat()} ===")

    # 1) Load recent data (14 days enough for indicators on 4H)
    try:
        data, funding = load_all(days=60, with_funding=True, force_refresh=False)
    except Exception as e:
        _log(f"  load failed: {e}")
        return {"ok": False, "error": str(e), "log": log}

    if "BTCUSDT" not in data:
        _log("  BTC missing — abort")
        return {"ok": False, "error": "BTC missing", "log": log}

    engine = TrendV2(data, funding)
    idx = engine.idx

    # Find the most recent CLOSED 4H bar. CoinEx updates bar in progress,
    # so we use the last fully-closed one.
    now_ms = int(time.time() * 1000)
    last_closed_ts = None
    for i in range(len(idx) - 1, -1, -1):
        ts = idx[i]
        ts_ms = int(ts.value // 10**6)
        if ts_ms + 4 * 3600 * 1000 <= now_ms:
            last_closed_ts = ts
            break
    if last_closed_ts is None:
        _log("  no closed bar yet")
        return {"ok": False, "error": "no closed bar", "log": log}

    equity, peak, last_bar_ts_seen = _load_state()
    current_bar_ms = int(last_closed_ts.value // 10**6)
    is_new_bar = current_bar_ms > last_bar_ts_seen

    _log(f"  last_closed_bar = {last_closed_ts}")
    _log(f"  is_new_bar      = {is_new_bar}  (last seen: {last_bar_ts_seen})")
    _log(f"  equity          = {equity:.2f}")

    # 2) Manage existing positions
    open_positions = db.get_open_positions()
    _log(f"  open positions  = {len(open_positions)}")

    for p in open_positions:
        sym = p["sym"]
        side = p["side"]
        ind = engine.ind.get(sym)
        if ind is None or last_closed_ts not in ind["close"].index:
            continue

        c = float(ind["close"].loc[last_closed_ts])

        # Update bars, PnL
        prev_bars = p["bars"]
        # We need entry bar reference for first bar
        entry_ts = pd.Timestamp(p["entry_ts"])
        if prev_bars == 0:
            ref_prev = p["entry_price"]
        else:
            # Find previous bar in idx
            try:
                pos_i = idx.get_loc(last_closed_ts)
                prev_ts = idx[pos_i - 1]
                ref_prev = float(ind["close"].loc[prev_ts]) if prev_ts in ind["close"].index else c
            except KeyError:
                ref_prev = c

        ret_bar = (c / ref_prev) - 1.0
        sign = +1.0 if side == "long" else -1.0
        new_pnl = p["pnl"] + sign * ret_bar * p["notional"]

        # Funding since last bar
        fund_pnl = p["funding_pnl"]
        f = funding.get(sym)
        if f is not None and not f.empty:
            try:
                pos_i = idx.get_loc(last_closed_ts)
                prev_ts = idx[pos_i - 1] if pos_i > 0 else last_closed_ts
                hit = f.loc[(f.index > prev_ts) & (f.index <= last_closed_ts)]
                if not hit.empty:
                    total_rate = float(hit["funding_rate"].sum())
                    add = -sign * total_rate * p["notional"]
                    new_pnl += add
                    fund_pnl += add
            except Exception:
                pass

        # Update trailing stop
        atr = float(ind["atr"].loc[last_closed_ts]) if last_closed_ts in ind["atr"].index else np.nan
        stop = p["stop"]
        initial_stop = p["initial_stop"]
        if np.isfinite(atr) and atr > 0:
            pa = (c - p["entry_price"]) / atr if side == "long" else (p["entry_price"] - c) / atr
            if pa >= TRAIL_ACTIVATION_ATR:
                if side == "long":
                    new_stop = c - ATR_TRAIL_MULT * atr
                    if new_stop > stop:
                        stop = float(new_stop)
                else:
                    new_stop = c + ATR_TRAIL_MULT * atr
                    if new_stop < stop:
                        stop = float(new_stop)

        # Check exit
        reason = None
        new_bars = prev_bars + 1
        if side == "long" and c <= initial_stop:
            reason = "SL"
        elif side == "short" and c >= initial_stop:
            reason = "SL"
        if reason is None and new_bars >= MIN_HOLD_BARS:
            if new_bars >= MAX_HOLD_BARS:
                reason = "TIME"
            elif side == "long" and stop > initial_stop and c <= stop:
                reason = "TRAIL"
            elif side == "short" and stop < initial_stop and c >= stop:
                reason = "TRAIL"
            else:
                ll_x = ind["ll_exit"].loc[last_closed_ts] if last_closed_ts in ind["ll_exit"].index else np.nan
                hh_x = ind["hh_exit"].loc[last_closed_ts] if last_closed_ts in ind["hh_exit"].index else np.nan
                if side == "long" and np.isfinite(ll_x) and c < ll_x:
                    reason = "DONCHIAN"
                elif side == "short" and np.isfinite(hh_x) and c > hh_x:
                    reason = "DONCHIAN"

        if reason:
            cost = ROUND_TRIP_COST * p["notional"]
            net = new_pnl - cost
            equity += net
            peak = max(peak, equity)

            db.insert_trade({
                "sym": sym, "side": side,
                "entry_ts": p["entry_ts"], "exit_ts": last_closed_ts.isoformat(),
                "entry_price": p["entry_price"], "exit_price": c,
                "notional": p["notional"], "pnl": new_pnl,
                "cost": cost, "net_pnl": net, "reason": reason,
                "bars": new_bars, "equity_after": equity,
            })
            db.delete_position(sym)
            db.set_state("equity", equity)
            db.set_state("peak_equity", peak)
            _log(f"  EXIT {sym} {side}  reason={reason}  net={net:+.2f}  equity={equity:.2f}")
        else:
            db.update_position(sym, bars=new_bars, pnl=new_pnl,
                               funding_pnl=fund_pnl, stop=stop)
            _log(f"  HOLD {sym} {side}  bars={new_bars}  pnl={new_pnl:+.2f}  stop={stop:.4f}")

    # 3) Entry scan (only on new bar close)
    if is_new_bar:
        open_positions = db.get_open_positions()
        if len(open_positions) < MAX_CONCURRENT_POSITIONS:
            gross = sum(p["notional"] for p in open_positions) / equity
            if gross < MAX_GROSS_EXPOSURE:
                if engine._btc_regime_ok(last_closed_ts):
                    bias_up = engine.btc_bias_up(last_closed_ts)
                    if bias_up is not None:
                        corr = _correlation_matrix(data, last_closed_ts) if len(open_positions) > 0 else None
                        cands = []
                        for sym in data.keys():
                            if any(p["sym"] == sym for p in open_positions):
                                continue
                            if not engine._liquidity_ok(sym, last_closed_ts):
                                continue
                            sig = engine.check_entry(sym, last_closed_ts)
                            if sig is None:
                                continue
                            if bias_up is True and sig["side"] == "short":
                                continue
                            if bias_up is False and sig["side"] == "long":
                                continue
                            if not engine._funding_ok(sym, sig["side"], last_closed_ts):
                                continue
                            if corr is not None and not corr.empty and sym in corr.columns:
                                conflict = False
                                for p in open_positions:
                                    other = p["sym"]
                                    if other in corr.columns and corr.loc[sym, other] > MAX_CORR:
                                        conflict = True
                                        break
                                if conflict:
                                    continue
                            cands.append((sym, sig))

                        if cands:
                            cands.sort(key=lambda x: x[1]["atr"] / x[1]["ref_price"], reverse=True)
                            slots = MAX_CONCURRENT_POSITIONS - len(open_positions)
                            for sym, sig in cands:
                                if slots <= 0:
                                    break
                                ind = engine.ind[sym]
                                # Use current close as entry proxy (in real trading, market order at next open)
                                entry_price = float(ind["close"].loc[last_closed_ts])
                                stop = engine.initial_stop(sig["side"], entry_price, sig["atr"])
                                notional = engine.position_size(sig["side"], entry_price, stop, equity)
                                if notional <= 0:
                                    continue
                                room = MAX_GROSS_EXPOSURE * equity - sum(p["notional"] for p in open_positions)
                                if room <= 0:
                                    break
                                notional = min(notional, room)
                                db.insert_position({
                                    "sym": sym, "side": sig["side"],
                                    "entry_ts": last_closed_ts.isoformat(),
                                    "entry_price": entry_price,
                                    "stop": stop, "initial_stop": stop,
                                    "atr": sig["atr"], "notional": notional,
                                })
                                open_positions.append({"sym": sym, "notional": notional})
                                slots -= 1
                                _log(f"  ENTRY {sym} {sig['side']}  @ {entry_price:.4f}  "
                                     f"notional={notional:.2f}")

        db.set_state("last_bar_ts", current_bar_ms)

    # 4) Update equity curve
    db.append_equity(time.time(), equity)

    elapsed = time.time() - t0
    _log(f"  check done in {elapsed:.1f}s")

    return {
        "ok": True, "log": log,
        "equity": equity,
        "open_positions": len(db.get_open_positions()),
        "last_bar": last_closed_ts.isoformat(),
        "is_new_bar": is_new_bar,
    }
