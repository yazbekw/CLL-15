import os
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, render_template_string
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__)

BASE_URL = "https://api.coinex.com/v2"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"]
PAIR_LIST = [(SYMBOLS[i], SYMBOLS[j]) for i in range(len(SYMBOLS)) for j in range(i + 1, len(SYMBOLS))]
PERIOD = "15min"
BAR_MS = 15 * 60 * 1000
LOOKBACK_DAYS_DEFAULT = 180
TRAIN_DAYS = 14
TRAIN_BARS = TRAIN_DAYS * 24 * 4
HORIZON = 4
RETRAIN_EVERY = 24  # 6 hours
ENTRY_THRESHOLD = 0.0035
TAKE_PROFIT = 0.0070
STOP_LOSS = -0.0045
EARLY_EXIT_THRESHOLD = 0.0010
MAX_HOLD_BARS = 8
BTC_15M_SHOCK = 0.012
BTC_1H_SHOCK = 0.025
MAX_GROSS_EXPOSURE = 0.60
RISK_PER_TRADE = 0.003
DAILY_STOP = -0.015
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_BARS = 24
FEE_TAKER = float(os.getenv("FEE_TAKER", "0.0005"))
SLIPPAGE = float(os.getenv("SLIPPAGE", "0.0002"))
ROUND_TRIP_COST = 4 * (FEE_TAKER + SLIPPAGE)  # two legs x entry/exit
REQUEST_TIMEOUT = 20

_session = requests.Session()
_session.headers.update({"User-Agent": "CLL15-research/1.0"})


def get_json(path: str, params: dict, retries: int = 3):
    last_err = None
    for k in range(retries):
        try:
            r = _session.get(BASE_URL + path, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if obj.get("code") != 0:
                raise RuntimeError(obj.get("message", "CoinEx API error"))
            return obj.get("data", [])
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (k + 1))
    raise last_err


def fetch_klines(market: str, days: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)
    rows = []
    cursor_end = end_ms
    # The API allows max 1000 candles per request.
    while cursor_end > start_ms:
        chunk = get_json(
            "/futures/kline",
            {"market": market, "period": PERIOD, "limit": 1000,
             "start_time": start_ms, "end_time": cursor_end},
        )
        if not chunk:
            break
        rows.extend(chunk)
        ts = [int(x["created_at"]) for x in chunk]
        earliest = min(ts)
        if earliest <= start_ms or len(chunk) < 1000:
            break
        cursor_end = earliest - 1
        time.sleep(0.05)
    if not rows:
        raise RuntimeError(f"No K-line data returned for {market}")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["created_at"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    for c in ["open", "close", "high", "low", "volume", "value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().drop_duplicates("ts").sort_values("ts").set_index("ts")
    return df[["open", "close", "high", "low", "volume", "value"]]


def fetch_funding_history(market: str, days: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 60 * 60 * 1000)
    rows = []
    page = 1
    while True:
        chunk = get_json(
            "/futures/funding-rate-history",
            {"market": market, "start_time": start_ms, "end_time": end_ms,
             "page": page, "limit": 1000},
        )
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        page += 1
        if page > 20:
            break
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["funding_time"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    df["funding_rate"] = pd.to_numeric(df["actual_funding_rate"], errors="coerce").fillna(0.0)
    return df[["ts", "funding_rate"]].drop_duplicates("ts").sort_values("ts").set_index("ts")


def make_features(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = pd.DataFrame(index=a.index.union(b.index)).sort_index()
    x["ra"] = np.log(a["close"]).diff().reindex(x.index)
    x["rb"] = np.log(b["close"]).diff().reindex(x.index)
    # Dynamic hedge ratio from rolling 14-day covariance/variance, lagged 1 bar to avoid leakage.
    cov = x["ra"].rolling(TRAIN_BARS).cov(x["rb"])
    var = x["rb"].rolling(TRAIN_BARS).var()
    beta = (cov / var.replace(0, np.nan)).clip(0.1, 5.0).shift(1)
    beta = beta.fillna(1.0)
    x["beta"] = beta
    x["spread"] = x["ra"] - beta * x["rb"]

    feats = pd.DataFrame(index=x.index)
    for col in ["ra", "rb", "spread"]:
        for lag in range(1, 5):
            feats[f"{col}_lag{lag}"] = x[col].shift(lag)
    feats["rel_4"] = (x["ra"].rolling(4).sum() - x["rb"].rolling(4).sum()).shift(1)
    feats["rel_16"] = (x["ra"].rolling(16).sum() - x["rb"].rolling(16).sum()).shift(1)
    feats["vol_a_16"] = x["ra"].rolling(16).std().shift(1)
    feats["vol_b_16"] = x["rb"].rolling(16).std().shift(1)
    feats["beta"] = beta

    # Future spread return for the next HORIZON bars.
    # Explicit summation avoids accidental alignment/look-ahead mistakes.
    future = x["ra"] - beta * x["rb"]
    target = sum(future.shift(-h) for h in range(1, HORIZON + 1))
    return feats.replace([np.inf, -np.inf], np.nan), target, beta


def build_dataset(data: Dict[str, pd.DataFrame], a_name: str, b_name: str):
    feats, target, beta = make_features(data[a_name], data[b_name])
    btc = data["BTCUSDT"]
    btc_r = np.log(btc["close"]).diff()
    feats["btc_15m"] = btc_r.reindex(feats.index).shift(1)
    feats["btc_1h"] = btc_r.rolling(4).sum().reindex(feats.index).shift(1)
    feats["btc_vol_1h"] = btc_r.rolling(4).std().reindex(feats.index).shift(1)
    return feats, target, beta


def fit_predict(train_x, train_y, x_row):
    mask = train_x.notna().all(axis=1) & train_y.notna()
    tx = train_x.loc[mask]
    ty = train_y.loc[mask]
    if len(tx) < 500 or x_row.isna().any():
        return np.nan
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10.0))
    ])
    model.fit(tx.values, ty.values)
    return float(model.predict(x_row.values.reshape(1, -1))[0])


def pair_forecasts_at(data, ts, cache):
    results = []
    # cache[(a,b)] = (features,target,beta)
    for a, b in PAIR_LIST:
        feats, target, beta = cache[(a, b)]
        if ts not in feats.index:
            continue
        x_row = feats.loc[ts]
        train_end_pos = feats.index.get_loc(ts)
        start_pos = max(0, train_end_pos - TRAIN_BARS)
        train_x = feats.iloc[start_pos:train_end_pos]
        train_y = target.iloc[start_pos:train_end_pos]
        pred = fit_predict(train_x, train_y, x_row)
        if not np.isfinite(pred):
            continue
        bval = float(beta.loc[ts]) if np.isfinite(beta.loc[ts]) else 1.0
        results.append({"a": a, "b": b, "forecast": pred, "beta": bval})
    return results


def backtest(days: int = LOOKBACK_DAYS_DEFAULT):
    data = {}
    funding = {}
    for s in SYMBOLS:
        data[s] = fetch_klines(s, days)
        funding[s] = fetch_funding_history(s, days)

    common_start = max(df.index.min() for df in data.values())
    common_end = min(df.index.max() for df in data.values())
    idx = data["BTCUSDT"].loc[common_start:common_end].index
    if len(idx) < TRAIN_BARS + 100:
        raise RuntimeError("Not enough common history for the requested backtest.")

    cache = {pair: build_dataset(data, *pair) for pair in PAIR_LIST}
    # Refit only every RETRAIN_EVERY bars and reuse forecasts in between; recalculated features remain leak-free.
    forecasts = {}
    last_forecast_pos = -10**9
    last_forecast = []

    trades = []
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    position = None
    daily = {}
    cooldown = 0
    consecutive_losses = 0

    for pos in range(TRAIN_BARS + 5, len(idx) - 1):
        ts = idx[pos]
        next_ts = idx[pos + 1]
        if cooldown > 0:
            cooldown -= 1

        if pos - last_forecast_pos >= RETRAIN_EVERY or not last_forecast:
            last_forecast = pair_forecasts_at(data, ts, cache)
            last_forecast_pos = pos

        btc_close = data["BTCUSDT"].loc[ts, "close"]
        btc_prev = data["BTCUSDT"].loc[idx[pos-1], "close"]
        btc_1h_prev = data["BTCUSDT"].loc[idx[pos-4], "close"] if pos >= 4 else btc_close
        btc15 = abs(btc_close / btc_prev - 1)
        btc1h = abs(btc_close / btc_1h_prev - 1)
        shocked = btc15 > BTC_15M_SHOCK or btc1h > BTC_1H_SHOCK

        # Manage existing position using close-to-close spread return at current bar.
        if position is not None:
            a, b = position["a"], position["b"]
            beta = position["beta"]
            wa = 1.0 / (1.0 + beta)
            wb = beta / (1.0 + beta)
            if position["bars"] == 0:
                ra = data[a].loc[ts, "close"] / position["entry_open_a"] - 1
                rb = data[b].loc[ts, "close"] / position["entry_open_b"] - 1
            else:
                ra = data[a].loc[ts, "close"] / data[a].loc[idx[pos-1], "close"] - 1
                rb = data[b].loc[ts, "close"] / data[b].loc[idx[pos-1], "close"] - 1
            pnl_bar = wa * ra - wb * rb
            position["pnl"] += pnl_bar
            position["bars"] += 1

            # Approximate funding cost using actual historical funding events inside the bar.
            for sym, notional, direction in [(a, wa, 1), (b, wb, -1)]:
                f = funding[sym]
                hit = f.loc[(f.index > idx[pos-1]) & (f.index <= ts)]
                if not hit.empty:
                    # positive funding: long pays, short receives
                    position["pnl"] += -direction * notional * float(hit["funding_rate"].sum())

            exit_reason = None
            if position["pnl"] >= TAKE_PROFIT:
                exit_reason = "TP"
            elif position["pnl"] <= STOP_LOSS:
                exit_reason = "SL"
            elif position["bars"] >= MAX_HOLD_BARS:
                exit_reason = "TIME"
            else:
                pair_now = next((z for z in last_forecast if z["a"] == a and z["b"] == b), None)
                if pair_now is not None and pair_now["forecast"] < EARLY_EXIT_THRESHOLD:
                    exit_reason = "WEAK_SIGNAL"

            if exit_reason:
                # Full round-trip costs, applied to gross exposure.
                net = position["pnl"] - ROUND_TRIP_COST
                equity *= (1.0 + net)
                peak = max(peak, equity)
                dd = equity / peak - 1.0
                max_dd = min(max_dd, dd)
                date_key = ts.date().isoformat()
                daily.setdefault(date_key, 0.0)
                daily[date_key] += net
                if net < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                trades.append({
                    "entry": position["entry"].isoformat(), "exit": ts.isoformat(),
                    "long": a, "short": b, "beta": round(beta, 3),
                    "gross_forecast": round(position["forecast"] * 100, 3),
                    "pnl_pct": round(net * 100, 4), "reason": exit_reason,
                    "bars": position["bars"],
                })
                position = None
                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    cooldown = COOLDOWN_BARS
                    consecutive_losses = 0
            continue

        # Daily loss lock.
        today_net = daily.get(ts.date().isoformat(), 0.0)
        if today_net <= DAILY_STOP or cooldown > 0 or shocked:
            continue

        candidates = [z for z in last_forecast if z["forecast"] >= ENTRY_THRESHOLD]
        if not candidates:
            continue
        best = max(candidates, key=lambda z: z["forecast"])

        # Require the expected spread to be materially above modeled execution costs.
        if best["forecast"] <= ROUND_TRIP_COST + 0.0015:
            continue

        # Enter at next bar open. Store forecast as known at ts only.
        position = {
            "a": best["a"], "b": best["b"], "beta": best["beta"],
            "forecast": best["forecast"], "entry": next_ts,
            "bars": 0, "pnl": 0.0,
        }

    if position is not None:
        ts = idx[-1]
        net = position["pnl"] - ROUND_TRIP_COST
        equity *= (1 + net)
        trades.append({"entry": position["entry"].isoformat(), "exit": ts.isoformat(),
                       "long": position["a"], "short": position["b"], "beta": round(position["beta"], 3),
                       "gross_forecast": round(position["forecast"] * 100, 3), "pnl_pct": round(net * 100, 4),
                       "reason": "END", "bars": position["bars"]})

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        summary = {"equity": 1.0, "return_pct": 0.0, "trades": 0, "win_rate_pct": 0.0,
                   "profit_factor": 0.0, "max_drawdown_pct": 0.0}
    else:
        wins = tdf[tdf.pnl_pct > 0]["pnl_pct"].sum()
        losses = -tdf[tdf.pnl_pct < 0]["pnl_pct"].sum()
        pf = float(wins / losses) if losses > 0 else float("inf")
        summary = {"equity": round(equity, 6), "return_pct": round((equity - 1) * 100, 3),
                   "trades": int(len(tdf)), "win_rate_pct": round(float((tdf.pnl_pct > 0).mean() * 100), 2),
                   "profit_factor": None if not np.isfinite(pf) else round(pf, 3),
                   "max_drawdown_pct": round(max_dd * 100, 3),
                   "avg_trade_pct": round(float(tdf.pnl_pct.mean()), 4)}
    return summary, tdf.to_dict(orient="records")


def live_signal(days: int = TRAIN_DAYS + 2):
    data = {s: fetch_klines(s, days) for s in SYMBOLS}
    common_end = min(df.index.max() for df in data.values())
    cache = {pair: build_dataset(data, *pair) for pair in PAIR_LIST}
    forecasts = pair_forecasts_at(data, common_end, cache)
    forecasts = sorted(forecasts, key=lambda x: x["forecast"], reverse=True)
    return {"timestamp": common_end.isoformat(), "round_trip_cost_pct": ROUND_TRIP_COST * 100,
            "top_pairs": [{**x, "forecast_pct": round(x["forecast"] * 100, 4)} for x in forecasts[:10]]}


HTML = r"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLL-15 Crypto Research</title>
<style>
body{font-family:Arial,sans-serif;background:#0e1117;color:#eee;margin:0;padding:20px} .wrap{max-width:1100px;margin:auto}
.card{background:#171b24;border:1px solid #2b3240;border-radius:14px;padding:18px;margin:12px 0} h1{margin:0 0 8px}
button{background:#2f81f7;color:#fff;border:0;border-radius:9px;padding:10px 15px;cursor:pointer} input{background:#0e1117;color:#fff;border:1px solid #3b4352;border-radius:8px;padding:9px;width:90px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.metric{font-size:22px;font-weight:700}.muted{color:#9da7b5;font-size:13px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid #2b3240;font-size:14px}.status{white-space:pre-wrap}
</style></head><body><div class="wrap">
<h1>CLL-15 · Cross-Crypto Lead-Lag</h1><div class="muted">Research/backtest only — no live orders are sent.</div>
<div class="card"><label>Backtest days <input id="days" value="180" type="number" min="30" max="365"></label> <button onclick="runBT()">Run backtest</button> <button onclick="signal()">Current signal</button><div id="status" class="status"></div></div>
<div id="summary" class="card" style="display:none"></div><div id="pairs" class="card" style="display:none"></div>
</div><script>
async function runBT(){const s=document.getElementById('status');s.textContent='Downloading data and running walk-forward backtest...';try{const d=await (await fetch('/api/backtest?days='+document.getElementById('days').value)).json();if(d.error)throw Error(d.error);const x=d.summary;document.getElementById('summary').style.display='block';document.getElementById('summary').innerHTML='<div class="grid">'+[['Return',x.return_pct+'%'],['Trades',x.trades],['Win rate',x.win_rate_pct+'%'],['Profit factor',x.profit_factor],['Max drawdown',x.max_drawdown_pct+'%'],['Avg trade',x.avg_trade_pct+'%']].map(a=>'<div class="card"><div class="muted">'+a[0]+'</div><div class="metric">'+a[1]+'</div></div>').join('')+'</div>';document.getElementById('pairs').style.display='block';document.getElementById('pairs').innerHTML='<h3>Last trades</h3><table><tr><th>Entry</th><th>Exit</th><th>Long</th><th>Short</th><th>Forecast</th><th>PnL</th><th>Reason</th></tr>'+d.trades.slice(-50).reverse().map(t=>'<tr><td>'+t.entry+'</td><td>'+t.exit+'</td><td>'+t.long+'</td><td>'+t.short+'</td><td>'+t.gross_forecast+'%</td><td>'+t.pnl_pct+'%</td><td>'+t.reason+'</td></tr>').join('')+'</table>';s.textContent='Finished.'}catch(e){s.textContent='Error: '+e.message}}
async function signal(){const s=document.getElementById('status');s.textContent='Calculating current signal...';try{const d=await (await fetch('/api/signal')).json();if(d.error)throw Error(d.error);document.getElementById('pairs').style.display='block';document.getElementById('pairs').innerHTML='<h3>Current ranked pairs</h3><div class="muted">Last common candle: '+d.timestamp+' · Round-trip cost assumption: '+d.round_trip_cost_pct.toFixed(3)+'%</div><table><tr><th>Long</th><th>Short</th><th>Forecast</th><th>Beta</th><th>Eligible</th></tr>'+d.top_pairs.map(t=>'<tr><td>'+t.a+'</td><td>'+t.b+'</td><td>'+t.forecast_pct+'%</td><td>'+t.beta.toFixed(3)+'</td><td>'+(t.forecast>=0.35?'YES':'NO')+'</td></tr>').join('')+'</table>';s.textContent='Finished.'}catch(e){s.textContent='Error: '+e.message}}
</script></body></html>
"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.get("/api/backtest")
def api_backtest():
    try:
        days = int(request.args.get("days", LOOKBACK_DAYS_DEFAULT))
        days = max(30, min(days, 365))
        summary, trades = backtest(days)
        return jsonify({"summary": summary, "trades": trades})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/signal")
def api_signal():
    try:
        return jsonify(live_signal())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
