"""Trend Following v5 — Donchian + MA + BTC bias + ATR sizing."""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    MA_WINDOW, DONCHIAN_ENTRY, DONCHIAN_EXIT, ATR_WINDOW,
    ATR_STOP_MULT, ATR_TRAIL_MULT, TRAIL_ACTIVATION_ATR, BTC_BIAS_MA,
    MIN_ATR_PCT, RISK_PER_TRADE, MAX_POSITION_NOTIONAL,
    MIN_FUNDING_ADVERSE, MIN_HOLD_BARS, MAX_HOLD_BARS,
    BTC_SHOCK_1H, BTC_SHOCK_1D,
    MIN_DAILY_VOLUME_USD, LIQUIDITY_LOOKBACK_DAYS, BARS_PER_DAY,
)


def compute_atr(df, window=ATR_WINDOW):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


class TrendV2:
    def __init__(self, data: Dict[str, pd.DataFrame], funding: Dict[str, pd.DataFrame]):
        self.raw = data
        self.funding = funding
        self.symbols = list(data.keys())   # active symbols only
        self.idx = self._common_index()
        self.ind = self._build_indicators()

    def _common_index(self):
        idx = None
        for df in self.raw.values():
            idx = df.index if idx is None else idx.intersection(df.index)
        return idx.sort_values()

    def _build_indicators(self):
        out = {}
        for sym, df in self.raw.items():
            d = df.reindex(self.idx)
            ma = d["close"].rolling(MA_WINDOW).mean()
            hh = d["high"].rolling(DONCHIAN_ENTRY).max().shift(1)
            ll = d["low"].rolling(DONCHIAN_ENTRY).min().shift(1)
            hh_exit = d["high"].rolling(DONCHIAN_EXIT).max().shift(1)
            ll_exit = d["low"].rolling(DONCHIAN_EXIT).min().shift(1)
            atr = compute_atr(d, ATR_WINDOW)
            vol_usd = d["value"].rolling(LIQUIDITY_LOOKBACK_DAYS * BARS_PER_DAY).mean() * BARS_PER_DAY
            out[sym] = {
                "close": d["close"], "open": d["open"],
                "high": d["high"], "low": d["low"],
                "ma": ma, "hh": hh, "ll": ll,
                "hh_exit": hh_exit, "ll_exit": ll_exit,
                "atr": atr, "vol_usd": vol_usd,
            }

        # BTC bias (simple MA)
        btc = out["BTCUSDT"]
        btc_close = btc["close"]
        btc_ma = btc_close.rolling(BTC_BIAS_MA).mean()
        out["_btc_bias"] = (btc_close > btc_ma)

        btc_ret_1 = btc_close.pct_change(1)
        btc_ret_6 = btc_close.pct_change(6)
        out["_btc_shock_1h"] = btc_ret_1.abs() > BTC_SHOCK_1H
        out["_btc_shock_1d"] = btc_ret_6.abs() > BTC_SHOCK_1D
        return out

    def _liquidity_ok(self, sym, ts):
        ind = self.ind[sym]
        if ts not in ind["vol_usd"].index:
            return False
        v = ind["vol_usd"].loc[ts]
        return bool(np.isfinite(v) and v >= MIN_DAILY_VOLUME_USD)

    def _funding_ok(self, sym, side, ts):
        f = self.funding.get(sym)
        if f is None or f.empty:
            return True
        window = f.loc[(f.index <= ts) & (f.index > ts - pd.Timedelta(hours=24))]
        if window.empty:
            return True
        avg = window["funding_rate"].mean()
        if side == "long":
            return avg <= MIN_FUNDING_ADVERSE
        return avg >= -MIN_FUNDING_ADVERSE

    def _btc_regime_ok(self, ts):
        if ts not in self.idx:
            return False
        if self.ind["_btc_shock_1h"].loc[ts]:
            return False
        if self.ind["_btc_shock_1d"].loc[ts]:
            return False
        return True

    def btc_bias_up(self, ts):
        s = self.ind.get("_btc_bias")
        if s is None or ts not in s.index:
            return None
        v = s.loc[ts]
        if pd.isna(v):
            return None
        return bool(v)

    def check_entry(self, sym, ts):
        ind = self.ind[sym]
        if ts not in ind["close"].index:
            return None
        c = ind["close"].loc[ts]
        ma = ind["ma"].loc[ts]
        hh = ind["hh"].loc[ts]
        ll = ind["ll"].loc[ts]
        atr = ind["atr"].loc[ts]
        if not np.isfinite([c, ma, hh, ll, atr]).all():
            return None
        if atr <= 0 or (atr / c) < MIN_ATR_PCT:
            return None

        long_break = c > hh and c > ma
        short_break = c < ll and c < ma
        if long_break and not short_break:
            return {"side": "long", "atr": float(atr), "ref_price": float(c)}
        if short_break and not long_break:
            return {"side": "short", "atr": float(atr), "ref_price": float(c)}
        return None

    def check_exit(self, position, ts):
        sym = position["sym"]
        side = position["side"]
        ind = self.ind[sym]
        if ts not in ind["close"].index:
            return None
        c = ind["close"].loc[ts]

        initial_stop = position.get("initial_stop", position["stop"])

        if side == "long" and c <= initial_stop:
            return "SL"
        if side == "short" and c >= initial_stop:
            return "SL"

        if position["bars"] < MIN_HOLD_BARS:
            return None
        if position["bars"] >= MAX_HOLD_BARS:
            return "TIME"

        if side == "long" and position["stop"] > initial_stop and c <= position["stop"]:
            return "TRAIL"
        if side == "short" and position["stop"] < initial_stop and c >= position["stop"]:
            return "TRAIL"

        hh_exit = ind["hh_exit"].loc[ts]
        ll_exit = ind["ll_exit"].loc[ts]
        if side == "long" and np.isfinite(ll_exit) and c < ll_exit:
            return "DONCHIAN"
        if side == "short" and np.isfinite(hh_exit) and c > hh_exit:
            return "DONCHIAN"

        return None

    def update_trailing_stop(self, position, ts):
        ind = self.ind[position["sym"]]
        if ts not in ind["close"].index:
            return
        c = ind["close"].loc[ts]
        atr = ind["atr"].loc[ts]
        if not np.isfinite(atr) or atr <= 0:
            return

        if position["side"] == "long":
            profit_atr = (c - position["entry_price"]) / atr
        else:
            profit_atr = (position["entry_price"] - c) / atr

        if profit_atr < TRAIL_ACTIVATION_ATR:
            return

        if position["side"] == "long":
            new_stop = c - ATR_TRAIL_MULT * atr
            if new_stop > position["stop"]:
                position["stop"] = float(new_stop)
        else:
            new_stop = c + ATR_TRAIL_MULT * atr
            if new_stop < position["stop"]:
                position["stop"] = float(new_stop)

    def initial_stop(self, side, entry_price, atr):
        if side == "long":
            return entry_price - ATR_STOP_MULT * atr
        return entry_price + ATR_STOP_MULT * atr

    def position_size(self, side, entry_price, stop_price, equity):
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0
        risk_usd = equity * RISK_PER_TRADE
        units = risk_usd / risk_per_unit
        notional = units * entry_price
        return float(min(notional, equity * MAX_POSITION_NOTIONAL))
