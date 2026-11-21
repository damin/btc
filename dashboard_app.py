from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, Response, jsonify, request

from btc_24h_predictor import (
    BTC24HPredictorPipeline,
    DemoSnapshotAdapter,
    ExtremeBoundaryEngine,
    PredictionOutput,
)
from live_adapter import GLOBAL_EXCHANGE_CIRCUIT, ProductionAPIAdapter


def _load_local_dotenv() -> None:
    """零第三方依赖加载同目录 .env 文件（不覆盖系统已注入的环境变量）"""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ and val != "":
            os.environ[key] = val


_load_local_dotenv()

REFRESH_INTERVAL_SEC = int(os.getenv("REFRESH_INTERVAL_SEC", "300"))


# =============================================================================
# 1. 后台 5 分钟轮询与状态缓存管理 (Carbon State Manager)
# =============================================================================

def _demo_strike_gex_bars() -> List[Dict[str, Any]]:
    sample = [
        (54000, 0.4, -4.8),
        (55000, 0.7, -9.2),
        (56000, 1.1, -11.5),
        (57000, 1.8, -14.6),
        (57500, 2.2, -21.4),
        (58000, 3.5, -16.8),
        (59000, 5.4, -13.1),
        (60000, 8.9, -15.4),
        (61000, 11.2, -10.6),
        (61500, 10.4, -10.1),
        (62000, 15.8, -6.2),
        (63000, 24.6, -4.1),
        (64000, 16.5, -2.3),
        (65000, 12.1, -1.2),
        (66000, 7.8, -0.6),
    ]
    return [
        {
            "strike": k,
            "call_gex_m": c,
            "put_gex_m": p,
            "net_gex_m": round(c + p, 2),
        }
        for k, c, p in sample
    ]


def _evaluate_alert_state(
    pred: PredictionOutput,
    candle_1h_close: float,
    oi_1h_change_pct: float,
) -> Dict[str, Any]:
    inv_msg = ExtremeBoundaryEngine.check_invalidation(
        pred=pred,
        candle_1h_close=candle_1h_close,
        oi_1h_change_pct=oi_1h_change_pct,
    )
    spot = pred.spot_price

    if inv_msg:
        return {
            "level": "CRITICAL",
            "title": "结构熔断已触发 · 切换 Q3 趋势模式",
            "message": inv_msg,
            "candle_1h_close": candle_1h_close,
            "oi_1h_change_pct": oi_1h_change_pct,
        }

    if spot >= pred.wick_ceiling_zone[0]:
        return {
            "level": "WARNING",
            "title": "逼近极端轧空上限 · 密切盯防 1H 收线",
            "message": f"现价 ${spot:,.1f} 已进入 Wick Ceiling 警戒带，若 1H 实体收破 ${pred.invalidation_bull_trigger:,.1f} 且 OI 扩张超过 1% 将触发向上熔断。",
            "candle_1h_close": candle_1h_close,
            "oi_1h_change_pct": oi_1h_change_pct,
        }
    if spot <= pred.wick_floor_zone[1]:
        return {
            "level": "WARNING",
            "title": "逼近极端插针底线 · 密切盯防 1H 收线",
            "message": f"现价 ${spot:,.1f} 已下探 Wick Floor 警戒带，若 1H 实体收破 ${pred.invalidation_bear_trigger:,.1f} 且 OI 扩张超过 1% 将触发向下熔断。",
            "candle_1h_close": candle_1h_close,
            "oi_1h_change_pct": oi_1h_change_pct,
        }

    return {
        "level": "NORMAL",
        "title": "结构正常 · 当前四象限模型有效",
        "message": f"1H 收盘价 ${candle_1h_close:,.1f}（1H OI 变化 {oi_1h_change_pct:+.2f}%）处于安全边界内，可按 24H 磁吸带执行区间策略。",
        "candle_1h_close": candle_1h_close,
        "oi_1h_change_pct": oi_1h_change_pct,
    }


class DashboardStateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refreshing = False
        self.mode: str = "DEMO" if os.getenv("USE_DEMO", "0") == "1" else "LIVE"
        self.exchange_source: str = os.getenv("EXCHANGE_SOURCE", "auto").strip().lower()
        if self.exchange_source not in ("auto", "okx", "deribit"):
            self.exchange_source = "auto"
        self.proxy: Optional[str] = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        self.manual_etf_netflow_usd: float = 0.0
        self.manual_polymarket_bias: Optional[float] = None

        self.snapshot: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.last_updated_epoch: float = 0.0
        self.next_refresh_epoch: float = time.time() + REFRESH_INTERVAL_SEC

    async def _fetch_payload_async(
        self,
        mode: str,
        exchange_source: str,
        proxy: Optional[str],
        etf_netflow: float,
        poly_bias: Optional[float],
    ) -> Dict[str, Any]:
        if mode == "DEMO":
            adapter = DemoSnapshotAdapter()
            pipeline = BTC24HPredictorPipeline(adapter=adapter)
            pred = await pipeline.run_once()
            tech = await adapter.fetch_technical()
            deriv = await adapter.fetch_derivatives()
            senti = await adapter.fetch_sentiment()
            alert = _evaluate_alert_state(pred, candle_1h_close=60180.0, oi_1h_change_pct=-0.65)
            return {
                "mode": "DEMO",
                "active_exchange": "DEMO_SNAPSHOT",
                "failover_reason": None,
                "prediction": asdict(pred),
                "regime_label": pred.primary_regime.value,
                "market_meta": {
                    "dvol_pct": round(deriv.dvol_annualized * 100.0, 2),
                    "net_gex_m": round(deriv.net_gex_usd / 1e6, 2),
                    "gamma_flip": deriv.gamma_flip_price,
                    "call_wall": deriv.call_wall,
                    "put_wall": deriv.put_wall,
                    "avwap": tech.avwap_anchor,
                    "funding_rate_pct": round(deriv.funding_rate_8h * 100.0, 4),
                    "oi_24h_pct": deriv.oi_change_24h_pct,
                    "fear_greed": senti.fear_greed_index,
                    "active_contracts": 148,
                },
                "strike_gex_bars": _demo_strike_gex_bars(),
                "alert": alert,
                "config": {
                    "exchange_source": exchange_source,
                    "etf_netflow_m": round(etf_netflow / 1e6, 2),
                    "polymarket_bias": poly_bias,
                },
            }

        async with ProductionAPIAdapter(
            okx_inst_id="BTC-USDT-SWAP",
            deribit_perp="BTC-PERPETUAL",
            exchange_source=exchange_source,
            circuit=GLOBAL_EXCHANGE_CIRCUIT,
            proxy=proxy,
            manual_etf_netflow_usd=etf_netflow,
            manual_polymarket_bias=poly_bias,
        ) as adapter:
            pipeline = BTC24HPredictorPipeline(adapter=adapter)
            pred = await pipeline.run_once()
            gex = adapter.last_gex_profile
            alert = _evaluate_alert_state(
                pred,
                candle_1h_close=adapter.last_1h_close,
                oi_1h_change_pct=adapter.oi_1h_change_pct,
            )
            return {
                "mode": "LIVE",
                "active_exchange": adapter.active_exchange,
                "failover_reason": adapter.failover_reason,
                "prediction": asdict(pred),
                "regime_label": pred.primary_regime.value,
                "market_meta": {
                    "dvol_pct": round(adapter.latest_dvol * 100.0, 2),
                    "net_gex_m": round((gex.net_gex_usd / 1e6), 2) if gex else 0.0,
                    "gamma_flip": gex.gamma_flip_price if gex else pred.spot_price,
                    "call_wall": gex.call_wall if gex else pred.core_high_zone[1],
                    "put_wall": gex.put_wall if gex else pred.core_low_zone[0],
                    "avwap": adapter.latest_avwap,
                    "funding_rate_pct": round(adapter.latest_funding_rate * 100.0, 4),
                    "oi_24h_pct": adapter.latest_oi_24h_pct,
                    "fear_greed": adapter.latest_fng,
                    "active_contracts": gex.active_contracts if gex else 0,
                },
                "strike_gex_bars": adapter.strike_gex_bars,
                "alert": alert,
                "config": {
                    "exchange_source": exchange_source,
                    "etf_netflow_m": round(etf_netflow / 1e6, 2),
                    "polymarket_bias": poly_bias,
                },
            }

    def refresh_now(self) -> Tuple[bool, Optional[str]]:
        with self._lock:
            if self._refreshing:
                return False, "已有刷新任务正在执行中"
            self._refreshing = True
            mode = self.mode
            exchange_source = self.exchange_source
            proxy = self.proxy
            etf_netflow = self.manual_etf_netflow_usd
            poly_bias = self.manual_polymarket_bias

        try:
            payload = asyncio.run(
                self._fetch_payload_async(mode, exchange_source, proxy, etf_netflow, poly_bias)
            )
            now_ep = time.time()
            with self._lock:
                self.snapshot = payload
                self.last_error = None
                self.last_updated_epoch = now_ep
                self.next_refresh_epoch = now_ep + REFRESH_INTERVAL_SEC
            return True, None
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.last_error = err_msg
                self.next_refresh_epoch = time.time() + REFRESH_INTERVAL_SEC
            return False, err_msg
        finally:
            with self._lock:
                self._refreshing = False

    def update_config(
        self,
        mode: Optional[str] = None,
        exchange_source: Optional[str] = None,
        etf_netflow_m: Optional[float] = None,
        polymarket_bias: Optional[str] = None,
    ) -> None:
        with self._lock:
            if mode in ("LIVE", "DEMO"):
                self.mode = mode
            if exchange_source in ("auto", "okx", "deribit"):
                self.exchange_source = exchange_source
                if exchange_source in ("auto", "okx"):
                    GLOBAL_EXCHANGE_CIRCUIT.reset_primary()
            if etf_netflow_m is not None:
                self.manual_etf_netflow_usd = float(etf_netflow_m) * 1e6
            if polymarket_bias is not None:
                text = str(polymarket_bias).strip()
                self.manual_polymarket_bias = None if text == "" else max(-1.0, min(1.0, float(text)))

    def get_state_json(self) -> Dict[str, Any]:
        with self._lock:
            now_ep = time.time()
            seconds_to_next = max(0, int(self.next_refresh_epoch - now_ep))
            return {
                "ok": self.snapshot is not None,
                "mode": self.mode,
                "exchange_source": self.exchange_source,
                "primary_cooldown_sec": GLOBAL_EXCHANGE_CIRCUIT.remaining_cooldown_sec,
                "refreshing": self._refreshing,
                "refresh_interval_sec": REFRESH_INTERVAL_SEC,
                "seconds_to_next_refresh": seconds_to_next,
                "last_updated_utc": (
                    datetime.fromtimestamp(self.last_updated_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    if self.last_updated_epoch > 0
                    else "--"
                ),
                "error": self.last_error,
                "data": self.snapshot,
            }


store = DashboardStateStore()
_poller_started = False
_poller_lock = threading.Lock()


def _background_poller() -> None:
    while True:
        store.refresh_now()
        time.sleep(REFRESH_INTERVAL_SEC)


def ensure_poller_started() -> None:
    global _poller_started
    with _poller_lock:
        if not _poller_started:
            t = threading.Thread(target=_background_poller, name="btc24h-poller", daemon=True)
            t.start()
            _poller_started = True


# =============================================================================
# 2. Flask Web 应用与 API 路由
# =============================================================================

app = Flask(__name__)


@app.route("/")
def index() -> Response:
    ensure_poller_started()
    return Response(HTML_TEMPLATE, mimetype="text/html; charset=utf-8")


@app.route("/healthz", methods=["GET"])
def healthz():
    snap = store.snapshot or {}
    return jsonify(
        {
            "status": "ok",
            "mode": store.mode,
            "active_exchange": snap.get("active_exchange", "INITIALIZING"),
        }
    )


@app.route("/api/snapshot", methods=["GET"])
def api_snapshot():
    ensure_poller_started()
    return jsonify(store.get_state_json())


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    ok, err = store.refresh_now()
    state = store.get_state_json()
    state["manual_refresh_ok"] = ok
    if err:
        state["error"] = err
    return jsonify(state)


@app.route("/api/config", methods=["POST"])
def api_config():
    body = request.get_json(silent=True) or {}
    try:
        store.update_config(
            mode=body.get("mode"),
            exchange_source=body.get("exchange_source"),
            etf_netflow_m=body.get("etf_netflow_m"),
            polymarket_bias=body.get("polymarket_bias"),
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": f"参数格式错误: {exc}"}), 400

    store.refresh_now()
    return jsonify(store.get_state_json())


# =============================================================================
# 3. 内嵌暗色量化终端前端 (零构建单文件 HTML + Chart.js)
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BTC 24H 极值磁吸带 & Deribit GEX 实时看板 (V3)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0b0f17;
      --panel: #121826;
      --panel-2: #182032;
      --border: #253047;
      --text: #e6edf7;
      --muted: #8b9bb4;
      --green: #10b981;
      --red: #f43f5e;
      --amber: #f59e0b;
      --blue: #38bdf8;
      --purple: #a78bfa;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      padding: 18px 22px 32px;
      line-height: 1.45;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .container { max-width: 1440px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
    .topbar {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px;
    }
    .brand h1 { font-size: 18px; font-weight: 700; letter-spacing: 0.3px; }
    .brand .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
    .pill {
      padding: 5px 11px; border-radius: 999px; font-size: 12px; font-weight: 600;
      border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
    }
    .pill.live { border-color: rgba(16,185,129,0.5); color: var(--green); }
    .pill.demo { border-color: rgba(245,158,11,0.5); color: var(--amber); }
    .pill.alt  { border-color: rgba(56,189,248,0.55); color: var(--blue); }
    button {
      background: #1d4ed8; color: #fff; border: none; border-radius: 8px; padding: 7px 14px;
      font-size: 12px; font-weight: 600; cursor: pointer; transition: opacity 0.15s;
    }
    button:hover { opacity: 0.9; }
    button.secondary { background: var(--panel-2); border: 1px solid var(--border); color: var(--text); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .info-bar {
      display: none; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.45);
      color: #bae6fd; border-radius: 10px; padding: 9px 14px; font-size: 12.5px;
    }
    .error-bar {
      display: none; background: rgba(244,63,94,0.14); border: 1px solid rgba(244,63,94,0.5);
      color: #fecdd3; border-radius: 10px; padding: 10px 14px; font-size: 12.5px;
    }
    .alert-banner {
      border-radius: 12px; padding: 14px 18px; border: 1px solid var(--border);
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
    }
    .alert-banner.NORMAL { background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.45); }
    .alert-banner.WARNING { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.55); }
    .alert-banner.CRITICAL {
      background: rgba(244,63,94,0.18); border-color: rgba(244,63,94,0.85);
      animation: pulse-red 1.8s infinite;
    }
    @keyframes pulse-red {
      0%, 100% { box-shadow: 0 0 0 0 rgba(244,63,94,0.35); }
      50% { box-shadow: 0 0 18px 2px rgba(244,63,94,0.55); }
    }
    .alert-title { font-size: 14px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .alert-desc { font-size: 12.5px; color: var(--text); opacity: 0.9; margin-top: 3px; }
    .alert-triggers { display: flex; gap: 14px; font-size: 12px; }
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
    .card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
    }
    .kpi-label { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
    .kpi-val { font-size: 22px; font-weight: 700; margin-top: 5px; }
    .kpi-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .main-grid { display: grid; grid-template-columns: 1.05fr 1.35fr; gap: 16px; }
    @media (max-width: 1040px) { .main-grid { grid-template-columns: 1fr; } }
    .panel-header {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;
      border-bottom: 1px solid var(--border); padding-bottom: 10px;
    }
    .panel-title { font-size: 14.5px; font-weight: 700; }
    .panel-meta { font-size: 12px; color: var(--muted); }
    .ladder-wrap {
      position: relative; height: 420px; width: 100%; background: #0d1320;
      border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    }
    .zone-box {
      position: absolute; left: 10px; right: 10px; border-radius: 6px;
      display: flex; justify-content: space-between; align-items: center;
      padding: 0 12px; font-size: 12px; font-weight: 600; pointer-events: none;
    }
    .zone-wick-ceil { background: rgba(244,63,94,0.16); border: 1px dashed rgba(244,63,94,0.65); color: #fda4af; }
    .zone-core-high { background: rgba(16,185,129,0.16); border: 1px solid rgba(16,185,129,0.65); color: #6ee7b7; }
    .zone-core-low  { background: rgba(56,189,248,0.16); border: 1px solid rgba(56,189,248,0.65); color: #7dd3fc; }
    .zone-wick-floor{ background: rgba(245,158,11,0.16); border: 1px dashed rgba(245,158,11,0.65); color: #fcd34d; }
    .price-line {
      position: absolute; left: 0; right: 0; height: 0; border-top: 2px solid #fff;
      display: flex; justify-content: flex-end; align-items: center; pointer-events: none;
    }
    .price-tag {
      transform: translateY(-50%); margin-right: 10px; padding: 2px 8px; border-radius: 4px;
      font-size: 11.5px; font-weight: 700; white-space: nowrap;
    }
    .line-spot { border-top: 2px solid #ffffff; z-index: 10; }
    .line-spot .price-tag { background: #ffffff; color: #0b0f17; }
    .line-flip { border-top: 1.5px dotted var(--purple); z-index: 6; justify-content: flex-start; }
    .line-flip .price-tag { background: rgba(167,139,250,0.22); border: 1px solid var(--purple); color: #ddd6fe; margin-left: 10px; }
    .line-avwap { border-top: 1.5px dotted var(--amber); z-index: 6; justify-content: center; }
    .line-avwap .price-tag { background: rgba(245,158,11,0.22); border: 1px solid var(--amber); color: #fde68a; }
    .line-inv { border-top: 1.5px solid var(--red); z-index: 7; justify-content: flex-start; }
    .line-inv .price-tag { background: rgba(244,63,94,0.85); color: #fff; margin-left: 10px; font-size: 11px; }
    .chart-box { position: relative; height: 420px; width: 100%; }
    .footer-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; }
    @media (max-width: 900px) { .footer-grid { grid-template-columns: 1fr; } }
    .form-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    .field label { font-size: 11.5px; color: var(--muted); }
    .field input, .field select {
      background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
      border-radius: 8px; padding: 7px 10px; font-size: 13px; width: 175px;
    }
    .diag-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px 16px; font-size: 12.5px; }
    .diag-item span:first-child { color: var(--muted); margin-right: 6px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="brand">
        <h1>BTC 24H 极值磁吸带 &amp; Deribit GEX 监控终端</h1>
        <div class="sub mono">最后更新: <span id="lastUpdated">--</span> · 下次 5 分钟自动刷新: <span id="countdown">--</span>s</div>
      </div>
      <div class="controls">
        <span id="exchangePill" class="pill">--</span>
        <span id="modePill" class="pill">--</span>
        <button id="btnToggleMode" class="secondary" onclick="toggleMode()">切换 DEMO / LIVE</button>
        <button id="btnRefresh" onclick="manualRefresh()">立即刷新</button>
      </div>
    </div>

    <div id="failoverBar" class="info-bar"></div>
    <div id="errorBar" class="error-bar"></div>

    <div id="alertBanner" class="alert-banner NORMAL">
      <div>
        <div class="alert-title" id="alertTitle">正在初始化行情与期权链...</div>
        <div class="alert-desc" id="alertMsg">等待后台协程完成首次 OKX + Deribit 并发抓取。</div>
      </div>
      <div class="alert-triggers mono">
        <div>向上熔断: <strong id="invBull" style="color:var(--red)">--</strong></div>
        <div>向下熔断: <strong id="invBear" style="color:var(--red)">--</strong></div>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="card">
        <div class="kpi-label">BTC 现价 / 24H 预期波幅 (EM)</div>
        <div class="kpi-val mono" id="kpiSpot">--</div>
        <div class="kpi-sub mono" id="kpiEm">EM: -- | DVOL: --</div>
      </div>
      <div class="card">
        <div class="kpi-label">四象限状态路由 (Regime)</div>
        <div class="kpi-val" id="kpiRegime" style="font-size:17px;color:var(--blue)">--</div>
        <div class="kpi-sub" id="kpiWeekend">周末薄盘叠加: --</div>
      </div>
      <div class="card">
        <div class="kpi-label">Deribit 净 Gamma (Net GEX)</div>
        <div class="kpi-val mono" id="kpiGex">--</div>
        <div class="kpi-sub mono" id="kpiWalls">Flip: -- | Put/Call Wall: --</div>
      </div>
      <div class="card">
        <div class="kpi-label">舆情反身性得分 &amp; 资金费率</div>
        <div class="kpi-val mono" id="kpiSent">--</div>
        <div class="kpi-sub mono" id="kpiFund">费率: -- | F&amp;G: --</div>
      </div>
    </div>

    <div class="main-grid">
      <div class="card">
        <div class="panel-header">
          <div class="panel-title">24H 极值磁吸带与关键结构锚点</div>
          <div class="panel-meta mono" id="ladderMeta">Wick / Core / AVWAP / Flip</div>
        </div>
        <div id="ladderCanvas" class="ladder-wrap"></div>
      </div>

      <div class="card">
        <div class="panel-header">
          <div class="panel-title">Deribit 各行权价 Net GEX 分布 ($M / 1% Move)</div>
          <div class="panel-meta mono" id="gexChartMeta">正值=做市商吸波阻力 · 负值=做市商助推加速</div>
        </div>
        <div class="chart-box">
          <canvas id="gexChart"></canvas>
        </div>
      </div>
    </div>

    <div class="footer-grid">
      <div class="card">
        <div class="panel-header">
          <div class="panel-title">永续数据源路由 &amp; 盘中手工因子注入</div>
          <div class="panel-meta">美国直连免代理 · 支持 OKX / Deribit 实时切换与 ETF 因子补充</div>
        </div>
        <div class="form-row">
          <div class="field">
            <label>永续合约数据源</label>
            <select id="selectSource">
              <option value="auto">AUTO (OKX→Deribit)</option>
              <option value="okx">OKX (BTC-USDT-SWAP)</option>
              <option value="deribit">Deribit (BTC-PERP)</option>
            </select>
          </div>
          <div class="field">
            <label>现货 ETF 净流入 ($M)</label>
            <input id="inputEtf" type="number" step="1" placeholder="如 -124.5" value="0" />
          </div>
          <div class="field">
            <label>Polymarket 偏移 (-1~+1)</label>
            <input id="inputPoly" type="number" step="0.05" min="-1" max="1" placeholder="留空自动" />
          </div>
          <button onclick="applyConfig()">应用并重算</button>
        </div>
      </div>

      <div class="card">
        <div class="panel-header">
          <div class="panel-title">系统诊断与权重遥测</div>
          <div class="panel-meta mono" id="diagContracts">活跃合约: --</div>
        </div>
        <div class="diag-list mono" id="diagBox">
          <div class="diag-item"><span>模型权重:</span><strong>--</strong></div>
        </div>
      </div>
    </div>
  </div>

<script>
  let gexChart = null;
  let currentMode = "LIVE";
  let countdownSec = 300;

  const fmtUSD = (v) => (v == null || isNaN(v)) ? "--" : "$" + Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 });

  function renderLadder(pred, meta) {
    const el = document.getElementById("ladderCanvas");
    const points = [
      pred.invalidation_bull_trigger,
      pred.wick_ceiling_zone[1],
      pred.core_high_zone[1],
      pred.spot_price,
      pred.core_low_zone[0],
      pred.wick_floor_zone[0],
      pred.invalidation_bear_trigger,
      meta.gamma_flip,
      meta.avwap,
    ].filter((x) => typeof x === "number" && x > 0);

    const rawMax = Math.max(...points);
    const rawMin = Math.min(...points);
    const pad = Math.max((rawMax - rawMin) * 0.08, pred.em_24h * 0.15);
    const maxP = rawMax + pad;
    const minP = rawMin - pad;
    const span = maxP - minP || 1;
    const yPct = (price) => Math.max(2, Math.min(98, ((maxP - price) / span) * 100));

    const zoneHtml = (cls, label, low, high) => {
      const top = yPct(high);
      const bottom = yPct(low);
      const h = Math.max(bottom - top, 5.5);
      return `<div class="zone-box ${cls}" style="top:${top}%;height:${h}%;">
        <span>${label}</span>
        <span class="mono">${fmtUSD(low)} ~ ${fmtUSD(high)}</span>
      </div>`;
    };

    const lineHtml = (cls, text, price) => {
      const top = yPct(price);
      return `<div class="price-line ${cls}" style="top:${top}%;">
        <span class="price-tag mono">${text}: ${fmtUSD(price)}</span>
      </div>`;
    };

    el.innerHTML = [
      zoneHtml("zone-wick-ceil", "极端轧空上限 (Wick Ceiling)", pred.wick_ceiling_zone[0], pred.wick_ceiling_zone[1]),
      zoneHtml("zone-core-high", "预期高点磁吸 (Core High)", pred.core_high_zone[0], pred.core_high_zone[1]),
      zoneHtml("zone-core-low", "预期低点磁吸 (Core Low)", pred.core_low_zone[0], pred.core_low_zone[1]),
      zoneHtml("zone-wick-floor", "极端插针底线 (Wick Floor)", pred.wick_floor_zone[0], pred.wick_floor_zone[1]),
      lineHtml("line-inv", "向上熔断", pred.invalidation_bull_trigger),
      lineHtml("line-inv", "向下熔断", pred.invalidation_bear_trigger),
      meta.gamma_flip ? lineHtml("line-flip", "Gamma Flip", meta.gamma_flip) : "",
      meta.avwap ? lineHtml("line-avwap", "AVWAP", meta.avwap) : "",
      lineHtml("line-spot", "SPOT 现价", pred.spot_price),
    ].join("");
  }

  function renderGexChart(bars, meta) {
    const ctx = document.getElementById("gexChart").getContext("2d");
    const labels = bars.map((b) => {
      let tag = "$" + (b.strike / 1000).toFixed(b.strike % 1000 === 0 ? 0 : 1) + "k";
      if (Math.abs(b.strike - meta.call_wall) < 1) tag += " [CW]";
      if (Math.abs(b.strike - meta.put_wall) < 1) tag += " [PW]";
      return tag;
    });
    const values = bars.map((b) => b.net_gex_m);
    const bgColors = values.map((v) => v >= 0 ? "rgba(16, 185, 129, 0.78)" : "rgba(244, 63, 94, 0.78)");
    const borderColors = values.map((v) => v >= 0 ? "#10b981" : "#f43f5e");

    if (gexChart) gexChart.destroy();

    gexChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Net GEX ($M / 1% Move)",
          data: values,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#182032",
            borderColor: "#334155",
            borderWidth: 1,
            callbacks: {
              label: (ctxItem) => {
                const b = bars[ctxItem.dataIndex];
                return [
                  ` Net GEX : ${b.net_gex_m >= 0 ? "+" : ""}${b.net_gex_m.toFixed(2)}M`,
                  ` Call GEX: +${b.call_gex_m.toFixed(2)}M`,
                  ` Put GEX : ${b.put_gex_m.toFixed(2)}M`,
                ];
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: "#94a3b8", font: { size: 11 } }, grid: { color: "rgba(37, 48, 71, 0.35)" } },
          y: {
            ticks: { color: "#94a3b8", callback: (v) => (v >= 0 ? "+" : "") + v + "M" },
            grid: { color: "rgba(37, 48, 71, 0.55)" },
          },
        },
      },
    });
  }

  function applyStateToUI(resp) {
    currentMode = resp.mode || "LIVE";
    countdownSec = resp.seconds_to_next_refresh ?? 300;
    document.getElementById("lastUpdated").textContent = resp.last_updated_utc || "--";
    document.getElementById("countdown").textContent = countdownSec;
    if (resp.exchange_source) {
      document.getElementById("selectSource").value = resp.exchange_source;
    }

    const pill = document.getElementById("modePill");
    pill.textContent = currentMode === "LIVE" ? "● LIVE 实盘" : "● DEMO 沙盘";
    pill.className = "pill " + (currentMode === "LIVE" ? "live" : "demo");

    const errBar = document.getElementById("errorBar");
    if (resp.error) {
      errBar.style.display = "block";
      errBar.textContent = "接口拉取告警: " + resp.error;
    } else {
      errBar.style.display = "none";
    }

    if (!resp.data) return;

    const { prediction: pred, market_meta: meta, alert, strike_gex_bars: bars, regime_label, active_exchange, failover_reason } = resp.data;

    const exPill = document.getElementById("exchangePill");
    exPill.textContent = "永续源: " + (active_exchange || "--");
    exPill.className = "pill " + (active_exchange === "OKX" ? "live" : (active_exchange === "DERIBIT" ? "alt" : "demo"));

    const fBar = document.getElementById("failoverBar");
    if (failover_reason && currentMode === "LIVE") {
      fBar.style.display = "block";
      const cd = resp.primary_cooldown_sec > 0 ? `（OKX 冷却探测剩余 ${Math.ceil(resp.primary_cooldown_sec / 60)} 分钟）` : "";
      fBar.textContent = `数据源路由状态: ${failover_reason}${cd}`;
    } else {
      fBar.style.display = "none";
    }

    const banner = document.getElementById("alertBanner");
    banner.className = "alert-banner " + (alert.level || "NORMAL");
    document.getElementById("alertTitle").textContent = alert.title;
    document.getElementById("alertMsg").textContent = alert.message;
    document.getElementById("invBull").textContent = fmtUSD(pred.invalidation_bull_trigger);
    document.getElementById("invBear").textContent = fmtUSD(pred.invalidation_bear_trigger);

    document.getElementById("kpiSpot").textContent = fmtUSD(pred.spot_price);
    document.getElementById("kpiEm").textContent = `EM: ±${fmtUSD(pred.em_24h)} | DVOL: ${meta.dvol_pct}%`;

    document.getElementById("kpiRegime").textContent = regime_label;
    document.getElementById("kpiWeekend").textContent = `周末薄盘叠加: ${pred.weekend_overlay ? "已激活 (缓冲带加宽30%)" : "未触发"}`;

    const gexEl = document.getElementById("kpiGex");
    gexEl.textContent = `${meta.net_gex_m >= 0 ? "+" : ""}${meta.net_gex_m.toFixed(2)}M`;
    gexEl.style.color = meta.net_gex_m >= 0 ? "var(--green)" : "var(--red)";
    document.getElementById("kpiWalls").textContent = `Flip: ${fmtUSD(meta.gamma_flip)} | PW ${fmtUSD(meta.put_wall)} / CW ${fmtUSD(meta.call_wall)}`;

    document.getElementById("kpiSent").textContent = `${pred.raw_sentiment >= 0 ? "+" : ""}${pred.raw_sentiment.toFixed(2)} → ${pred.adjusted_sentiment >= 0 ? "+" : ""}${pred.adjusted_sentiment.toFixed(2)}`;
    document.getElementById("kpiFund").textContent = `8H费率: ${meta.funding_rate_pct}% | F&G: ${meta.fear_greed} | 24H OI: ${meta.oi_24h_pct >= 0 ? "+" : ""}${meta.oi_24h_pct}%`;

    renderLadder(pred, meta);
    renderGexChart(bars || [], meta);

    document.getElementById("diagContracts").textContent = `Deribit 活跃合约: ${meta.active_contracts}`;
    const d = pred.diagnostics || {};
    document.getElementById("diagBox").innerHTML = `
      <div class="diag-item"><span>合成权重:</span><strong>${d.weights || "--"}</strong></div>
      <div class="diag-item"><span>费率信号:</span><strong>${d.funding_signal || "--"}</strong></div>
      <div class="diag-item"><span>AVWAP 偏离:</span><strong>${d.avwap_distance_pct || "--"}</strong></div>
      <div class="diag-item"><span>1H OI 变化:</span><strong>${alert.oi_1h_change_pct >= 0 ? "+" : ""}${alert.oi_1h_change_pct}%</strong></div>
    `;
  }

  async function fetchSnapshot() {
    try {
      const res = await fetch("/api/snapshot");
      const json = await res.json();
      applyStateToUI(json);
    } catch (e) {
      console.error("Snapshot fetch failed:", e);
    }
  }

  async function manualRefresh() {
    const btn = document.getElementById("btnRefresh");
    btn.disabled = true;
    btn.textContent = "刷新中...";
    try {
      const res = await fetch("/api/refresh", { method: "POST" });
      const json = await res.json();
      applyStateToUI(json);
    } finally {
      btn.disabled = false;
      btn.textContent = "立即刷新";
    }
  }

  async function toggleMode() {
    const nextMode = currentMode === "LIVE" ? "DEMO" : "LIVE";
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: nextMode }),
    });
    const json = await res.json();
    applyStateToUI(json);
  }

  async function applyConfig() {
    const etfVal = parseFloat(document.getElementById("inputEtf").value || "0");
    const polyVal = document.getElementById("inputPoly").value;
    const srcVal = document.getElementById("selectSource").value;
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exchange_source: srcVal,
        etf_netflow_m: isNaN(etfVal) ? 0 : etfVal,
        polymarket_bias: polyVal,
      }),
    });
    const json = await res.json();
    applyStateToUI(json);
  }

  setInterval(() => {
    if (countdownSec > 0) {
      countdownSec -= 1;
      document.getElementById("countdown").textContent = countdownSec;
    } else {
      fetchSnapshot();
    }
  }, 1000);

  fetchSnapshot().then(() => {
    setTimeout(() => {
      if (document.getElementById("kpiSpot").textContent === "--") {
        fetchSnapshot();
      }
    }, 2500);
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    ensure_poller_started()
    port = int(os.getenv("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
