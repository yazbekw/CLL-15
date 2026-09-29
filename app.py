"""Paper trading web UI + background scheduler. Deploy on Render."""
import os
import threading
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string, request

from config_paper import (
    CHECK_INTERVAL_SECONDS, WEB_REFRESH_SECONDS,
    PAPER_STARTING_EQUITY, SYMBOLS,
)
import paper_db as db
from paper_trader import run_check

app = Flask(__name__)

# ---------- Background scheduler ----------
_scheduler_lock = threading.Lock()
_scheduler_started = False
_last_run_info = {"ts": None, "result": None}


def _scheduler_loop():
    print("[scheduler] started")
    while True:
        try:
            print(f"[scheduler] running check @ {datetime.now(timezone.utc).isoformat()}")
            res = run_check(verbose=False)
            _last_run_info["ts"] = datetime.now(timezone.utc).isoformat()
            _last_run_info["result"] = res
            print(f"[scheduler] done  equity={res.get('equity')}  "
                  f"open={res.get('open_positions')}")
        except Exception as e:
            print(f"[scheduler] error: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)


def ensure_scheduler():
    global _scheduler_started
    with _scheduler_lock:
        if not _scheduler_started:
            t = threading.Thread(target=_scheduler_loop, daemon=True)
            t.start()
            _scheduler_started = True


# ---------- Startup ----------
db.init_db()
ensure_scheduler()


# ---------- Routes ----------
@app.get("/")
def home():
    return render_template_string(HTML)


@app.get("/api/status")
def api_status():
    equity = float(db.get_state("equity", PAPER_STARTING_EQUITY))
    peak = float(db.get_state("peak_equity", PAPER_STARTING_EQUITY))
    positions = db.get_open_positions()
    trades = db.get_trades(limit=50)
    curve = db.get_equity_curve(limit=2000)

    total_ret = (equity - PAPER_STARTING_EQUITY) / PAPER_STARTING_EQUITY * 100
    dd = (equity / peak - 1.0) * 100 if peak > 0 else 0

    wins = [t for t in trades if t["net_pnl"] > 0]
    losses = [t for t in trades if t["net_pnl"] < 0]
    n = len(trades)
    win_rate = 100 * len(wins) / n if n else 0
    pf = (sum(t["net_pnl"] for t in wins) / -sum(t["net_pnl"] for t in losses)
          if losses and sum(t["net_pnl"] for t in losses) != 0 else None)

    return jsonify({
        "equity": round(equity, 2),
        "peak_equity": round(peak, 2),
        "starting_equity": PAPER_STARTING_EQUITY,
        "total_return_pct": round(total_ret, 3),
        "current_dd_pct": round(dd, 3),
        "open_positions": positions,
        "recent_trades": trades,
        "equity_curve": curve,
        "stats": {
            "total_trades": n,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": None if pf is None else round(pf, 3),
        },
        "scheduler": {
            "interval_s": CHECK_INTERVAL_SECONDS,
            "last_run_ts": _last_run_info["ts"],
            "last_run_ok": (_last_run_info["result"] or {}).get("ok"),
            "last_run_equity": (_last_run_info["result"] or {}).get("equity"),
        },
        "universe": SYMBOLS,
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.post("/api/run-now")
def api_run_now():
    try:
        res = run_check(verbose=True)
        _last_run_info["ts"] = datetime.now(timezone.utc).isoformat()
        _last_run_info["result"] = res
        return jsonify(res)
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e),
                        "traceback": traceback.format_exc()}), 500


@app.get("/health")
def health():
    return "OK", 200

@app.get("/")
def home():
    return render_template_string(HTML, refresh=WEB_REFRESH_SECONDS)
    
# ---------- HTML ----------
HTML = r"""
<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLL-15 Paper Trader</title>
<style>
body{font-family:system-ui,Arial,sans-serif;background:#0e1117;color:#eee;margin:0;padding:20px}
.wrap{max-width:1200px;margin:auto}
.card{background:#171b24;border:1px solid #2b3240;border-radius:14px;padding:18px;margin:12px 0}
h1{margin:0 0 8px}h3{margin:6px 0 12px}
button{background:#2f81f7;color:#fff;border:0;border-radius:9px;padding:10px 16px;cursor:pointer;margin-right:8px}
button:disabled{opacity:.5;cursor:not-allowed}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.metric{font-size:24px;font-weight:700}
.muted{color:#9da7b5;font-size:13px}
.pos{color:#4ade80}.neg{color:#f87171}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:8px;border-bottom:1px solid #2b3240;font-size:13px}
.badge{padding:2px 8px;border-radius:6px;font-size:12px}
.long{background:#1c3a2c;color:#4ade80}
.short{background:#3a1c1c;color:#f87171}
.reason{padding:2px 6px;border-radius:4px;background:#2b3240;font-size:11px}
canvas{width:100%;height:200px;background:#0e1117;border-radius:8px}
</style></head><body><div class="wrap">
<h1>CLL-15 · Paper Trader</h1>
<div class="muted">Trend Following v5 · 4H · No real orders sent</div>

<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
    <div id="sched" class="muted">Loading...</div>
    <div>
      <button id="btnRun" onclick="runNow()">Run check now</button>
      <span class="muted">Auto-refresh <span id="cntdown">60</span>s</span>
    </div>
  </div>
</div>

<div id="metrics" class="card"></div>

<div class="card">
  <h3>Equity curve</h3>
  <canvas id="chart" width="1000" height="200"></canvas>
</div>

<div class="card" id="positionsCard">
  <h3>Open positions</h3>
  <div id="positions"></div>
</div>

<div class="card" id="tradesCard">
  <h3>Recent trades</h3>
  <div id="trades"></div>
</div>

</div>
<script>
let countdown = {{ refresh }};

async function load() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    render(d);
  } catch (e) {
    console.error(e);
  }
}

function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  const s = (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
  return `<span class="${v>=0?'pos':'neg'}">${s}</span>`;
}

function render(d) {
  // Scheduler info
  const sch = d.scheduler;
  document.getElementById('sched').innerHTML =
    `Last check: ${sch.last_run_ts || '—'} · equity=${sch.last_run_equity ?? '—'} · interval=${sch.interval_s}s`;

  // Metrics
  const m = d.stats;
  document.getElementById('metrics').innerHTML =
    '<div class="grid">' +
    card('Equity', d.equity + ' USDT', '') +
    card('Total return', fmtPct(d.total_return_pct), '') +
    card('Current DD', fmtPct(d.current_dd_pct), '') +
    card('Trades', m.total_trades, '') +
    card('Win rate', m.win_rate_pct + '%', '') +
    card('Profit factor', m.profit_factor ?? '—', '') +
    card('Open', d.open_positions.length, '') +
    '</div>';

  // Positions
  if (d.open_positions.length === 0) {
    document.getElementById('positions').innerHTML =
      '<div class="muted">No open positions.</div>';
  } else {
    let html = '<table><tr><th>Symbol</th><th>Side</th><th>Entry</th>' +
               '<th>Stop</th><th>Notional</th><th>Bars</th><th>PnL</th></tr>';
    for (const p of d.open_positions) {
      const cls = p.side === 'long' ? 'long' : 'short';
      const pnl = p.pnl || 0;
      const pnlCls = pnl >= 0 ? 'pos' : 'neg';
      html += `<tr><td>${p.sym}</td>
        <td><span class="badge ${cls}">${p.side}</span></td>
        <td>${(+p.entry_price).toFixed(4)}</td>
        <td>${(+p.stop).toFixed(4)}</td>
        <td>${(+p.notional).toFixed(2)}</td>
        <td>${p.bars}</td>
        <td class="${pnlCls}">${pnl.toFixed(2)}</td></tr>`;
    }
    html += '</table>';
    document.getElementById('positions').innerHTML = html;
  }

  // Trades
  if (!d.recent_trades || d.recent_trades.length === 0) {
    document.getElementById('trades').innerHTML =
      '<div class="muted">No closed trades yet.</div>';
  } else {
    let html = '<table><tr><th>Symbol</th><th>Side</th><th>Entry</th>' +
               '<th>Exit</th><th>Net PnL</th><th>Reason</th><th>Bars</th></tr>';
    for (const t of d.recent_trades) {
      const pnlCls = t.net_pnl >= 0 ? 'pos' : 'neg';
      html += `<tr><td>${t.sym}</td>
        <td>${t.side}</td>
        <td>${(+t.entry_price).toFixed(4)}</td>
        <td>${(+t.exit_price).toFixed(4)}</td>
        <td class="${pnlCls}">${(+t.net_pnl).toFixed(2)}</td>
        <td><span class="reason">${t.reason}</span></td>
        <td>${t.bars}</td></tr>`;
    }
    html += '</table>';
    document.getElementById('trades').innerHTML = html;
  }

  // Chart
  drawChart(d.equity_curve);
}

function card(label, val, extra) {
  return `<div class="card"><div class="muted">${label}</div>` +
         `<div class="metric">${val}</div>${extra || ''}</div>`;
}

function drawChart(curve) {
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!curve || curve.length < 2) return;
  const vals = curve.map(p => p.equity);
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 1;
  const pad = 20;
  const xStep = (w - 2 * pad) / (vals.length - 1);
  // baseline
  ctx.strokeStyle = '#2b3240';
  ctx.beginPath();
  ctx.moveTo(pad, h / 2); ctx.lineTo(w - pad, h / 2); ctx.stroke();
  // curve
  ctx.strokeStyle = '#2f81f7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  vals.forEach((v, i) => {
    const x = pad + i * xStep;
    const y = h - pad - (v - min) / range * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // labels
  ctx.fillStyle = '#9da7b5';
  ctx.font = '11px sans-serif';
  ctx.fillText(max.toFixed(2), 2, 12);
  ctx.fillText(min.toFixed(2), 2, h - 6);
}

async function runNow() {
  const b = document.getElementById('btnRun');
  b.disabled = true; b.textContent = 'Running...';
  try {
    const r = await fetch('/api/run-now', {method: 'POST'});
    const d = await r.json();
    if (!d.ok) alert('Error: ' + (d.error || 'unknown'));
    await load();
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    b.disabled = false; b.textContent = 'Run check now';
  }
}

// Auto-refresh
function tick() {
  countdown--;
  if (countdown <= 0) { countdown = {{ refresh }}; load(); }
  document.getElementById('cntdown').textContent = countdown;
}

load();
setInterval(tick, 1000);
</script></body></html>
"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
