from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
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
from live_adapter import (
    GLOBAL_ETF_CACHE,
    GLOBAL_EXCHANGE_CIRCUIT,
    ProductionAPIAdapter,
)


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
# 2026-11-22 18:00:00 EST (UTC-5) 对应的毫秒时间戳，用于拉取完整对比窗口历史 K 线
LOCKED_FORECAST_START_MS = 1795388400000


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


def _build_demo_candles(start_ms: int) -> List[Dict[str, float]]:
    """DEMO 沙盘模式下生成平滑演示 15m K 线，便于离线预览预测 vs 实盘对比效果"""
    waypoints = [
        (0.0, 61248.2), (1.5, 61020.0), (4.0, 61510.0), (6.5, 61480.0),
        (9.0, 62890.0), (11.5, 63320.0), (13.5, 61840.0), (16.0, 59820.0),
        (18.5, 58590.0), (20.5, 58910.0),
    ]
    bars: List[Dict[str, float]] = []
    prev_c = 61248.2
    for step in range(82):  # 约 20.5 小时 (82 根 15m K线)
        h = step * 0.25
        ts = start_ms + step * 15 * 60 * 1000
        base = waypoints[-1][1]
        for i in range(len(waypoints) - 1):
            h0, p0 = waypoints[i]
            h1, p1 = waypoints[i + 1]
            if h0 <= h <= h1:
                t = (h - h0) / (h1 - h0)
                mu = (1.0 - math.cos(t * math.pi)) / 2.0
                base = p0 * (1.0 - mu) + p1 * mu
                break
        wave = math.sin(step * 0.85) * 145.0 + math.cos(step * 0.37) * 80.0
        close_p = round(base + wave, 1)
        open_p = prev_c
        high_p = round(max(open_p, close_p) + abs(math.sin(step * 1.3)) * 210.0 + 45.0, 1)
        low_p = round(min(open_p, close_p) - abs(math.cos(step * 1.1)) * 220.0 - 45.0, 1)
        vol = round(180.0 + abs(close_p - open_p) * 1.35 + (420.0 if 13.0 <= h <= 18.0 else 60.0), 1)
        bars.append({"ts": ts, "o": open_p, "h": high_p, "l": low_p, "c": close_p, "v": vol})
        prev_c = close_p
    return bars


def _http_get_json(url: str, proxy: Optional[str], timeout: float = 8.0) -> Any:
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "BTC-24H-GEX-Dashboard/4.1",
            "Accept": "application/json",
        },
    )
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_okx_15m_candles(proxy: Optional[str], limit: int = 240) -> List[Dict[str, float]]:
    url = f"https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar=15m&limit={limit}"
    payload = _http_get_json(url, proxy=proxy)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not rows:
        raise RuntimeError(f"OKX candles empty: {payload}")
    candles = [
        {
            "ts": int(r[0]),
            "o": float(r[1]),
            "h": float(r[2]),
            "l": float(r[3]),
            "c": float(r[4]),
            "v": float(r[5]),
        }
        for r in rows
    ]
    candles.sort(key=lambda x: x["ts"])
    return candles


def _fetch_deribit_15m_candles(
    proxy: Optional[str],
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, float]]:
    url = (
        "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
        f"?instrument_name=BTC-PERPETUAL&start_timestamp={start_ms}&end_timestamp={end_ms}&resolution=15"
    )
    payload = _http_get_json(url, proxy=proxy)
    res = payload.get("result") if isinstance(payload, dict) else None
    if not res or res.get("status") != "ok" or not res.get("ticks"):
        raise RuntimeError(f"Deribit chart data invalid: {payload}")
    candles = [
        {
            "ts": int(ts),
            "o": float(o),
            "h": float(h),
            "l": float(l),
            "c": float(c),
            "v": float(v or 0.0),
        }
        for ts, o, h, l, c, v in zip(
            res["ticks"],
            res["open"],
            res["high"],
            res["low"],
            res["close"],
            res.get("volume", [0.0] * len(res["ticks"])),
        )
    ]
    candles.sort(key=lambda x: x["ts"])
    return candles


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
        self._candle_lock = threading.Lock()
        self._refreshing = False
        self.mode: str = "DEMO" if os.getenv("USE_DEMO", "0") == "1" else "LIVE"
        self.exchange_source: str = os.getenv("EXCHANGE_SOURCE", "auto").strip().lower()
        if self.exchange_source not in ("auto", "okx", "deribit"):
            self.exchange_source = "auto"
        self.proxy: Optional[str] = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        # None = 自动抓取 SoSoValue/Farside；填数值 = 手动覆盖
        self.manual_etf_netflow_usd: Optional[float] = None
        self.manual_polymarket_bias: Optional[float] = None

        self.snapshot: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.last_updated_epoch: float = 0.0
        self.next_refresh_epoch: float = time.time() + REFRESH_INTERVAL_SEC

        # 15m K 线短期缓存（避免前端频繁刷新触发交易所限频）
        self._candle_cache_payload: Optional[Dict[str, Any]] = None
        self._candle_cache_epoch: float = 0.0

    def invalidate_candle_cache(self) -> None:
        with self._candle_lock:
            self._candle_cache_payload = None
            self._candle_cache_epoch = 0.0

    def get_candles_json(self, start_ms: Optional[int] = None) -> Dict[str, Any]:
        now_ep = time.time()
        with self._lock:
            mode = self.mode
            exchange_source = self.exchange_source
            proxy = self.proxy
            snap = self.snapshot
            snap_ep = self.last_updated_epoch

        if mode == "DEMO":
            base_start = start_ms or LOCKED_FORECAST_START_MS
            return {
                "ok": True,
                "source": "DEMO_SNAPSHOT",
                "candles": _build_demo_candles(base_start),
            }

        with self._candle_lock:
            if (
                self._candle_cache_payload is not None
                and (now_ep - self._candle_cache_epoch) < 45.0
                and start_ms is None
            ):
                return self._candle_cache_payload

        now_ms = int(now_ep * 1000)
        query_start_ms = min(start_ms or LOCKED_FORECAST_START_MS, now_ms - 48 * 3600 * 1000)

        candles: List[Dict[str, float]] = []
        used_source = "OKX"
        err_notes: List[str] = []

        order = ["deribit", "okx"] if exchange_source == "deribit" else ["okx", "deribit"]
        if exchange_source == "auto" and GLOBAL_EXCHANGE_CIRCUIT.remaining_cooldown_sec > 0:
            order = ["deribit", "okx"]

        for src in order:
            try:
                if src == "okx":
                    candles = _fetch_okx_15m_candles(proxy=proxy, limit=240)
                    used_source = "OKX (BTC-USDT-SWAP)"
                else:
                    candles = _fetch_deribit_15m_candles(
                        proxy=proxy,
                        start_ms=query_start_ms,
                        end_ms=now_ms,
                    )
                    used_source = "DERIBIT (BTC-PERPETUAL)"
                if candles:
                    break
            except Exception as exc:
                err_notes.append(f"{src.upper()}: {exc}")

        # 用快照最新现价修正最后一根 K 线收盘价，确保与顶部 KPI 现价完全一致
        if candles and snap and snap.get("prediction") and snap_ep > 0:
            spot_now = float(snap["prediction"]["spot_price"])
            snap_ms = int(snap_ep * 1000)
            last_bar = candles[-1]
            if snap_ms >= last_bar["ts"]:
                last_bar["c"] = spot_now
                last_bar["h"] = max(last_bar["h"], spot_now)
                last_bar["l"] = min(last_bar["l"], spot_now)

        result = {
            "ok": len(candles) > 0,
            "source": used_source if candles else "UNAVAILABLE",
            "candles": candles,
            "error": " | ".join(err_notes) if not candles and err_notes else None,
        }

        if candles and start_ms is None:
            with self._candle_lock:
                self._candle_cache_payload = result
                self._candle_cache_epoch = now_ep

        return result

    async def _fetch_payload_async(
        self,
        mode: str,
        exchange_source: str,
        proxy: Optional[str],
        etf_override_usd: Optional[float],
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
                    "etf_netflow_m": round(tech.spot_etf_netflow_usd / 1e6, 2),
                    "etf_date": "2026-11-20",
                    "etf_source": "Demo Snapshot",
                },
                "strike_gex_bars": _demo_strike_gex_bars(),
                "alert": alert,
                "config": {
                    "exchange_source": exchange_source,
                    "etf_override_m": round(etf_override_usd / 1e6, 2) if etf_override_usd is not None else None,
                    "polymarket_bias": poly_bias,
                },
            }

        async with ProductionAPIAdapter(
            okx_inst_id="BTC-USDT-SWAP",
            deribit_perp="BTC-PERPETUAL",
            exchange_source=exchange_source,
            circuit=GLOBAL_EXCHANGE_CIRCUIT,
            etf_cache=GLOBAL_ETF_CACHE,
            proxy=proxy,
            manual_etf_netflow_usd=etf_override_usd,
            manual_polymarket_bias=poly_bias,
        ) as adapter:
            pipeline = BTC24HPredictorPipeline(adapter=adapter)
            pred = await pipeline.run_once()
            gex = adapter.last_gex_profile
            etf = adapter.latest_etf
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
                    "etf_netflow_m": round(etf.netflow_usd / 1e6, 2),
                    "etf_date": etf.trade_date,
                    "etf_source": etf.source,
                },
                "strike_gex_bars": adapter.strike_gex_bars,
                "alert": alert,
                "config": {
                    "exchange_source": exchange_source,
                    "etf_override_m": round(etf_override_usd / 1e6, 2) if etf_override_usd is not None else None,
                    "polymarket_bias": poly_bias,
                },
            }

    def refresh_now(self, force_etf_refresh: bool = False) -> Tuple[bool, Optional[str]]:
        with self._lock:
            if self._refreshing:
                return False, "已有刷新任务正在执行中"
            self._refreshing = True
            mode = self.mode
            exchange_source = self.exchange_source
            proxy = self.proxy
            etf_override = self.manual_etf_netflow_usd
            poly_bias = self.manual_polymarket_bias

        if force_etf_refresh:
            GLOBAL_ETF_CACHE.clear()

        try:
            payload = asyncio.run(
                self._fetch_payload_async(mode, exchange_source, proxy, etf_override, poly_bias)
            )
            now_ep = time.time()
            with self._lock:
                self.snapshot = payload
                self.last_error = None
                self.last_updated_epoch = now_ep
                self.next_refresh_epoch = now_ep + REFRESH_INTERVAL_SEC
            self.invalidate_candle_cache()
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
        etf_netflow_m: Optional[Any] = None,
        polymarket_bias: Optional[Any] = None,
    ) -> None:
        with self._lock:
            if mode in ("LIVE", "DEMO"):
                self.mode = mode
            if exchange_source in ("auto", "okx", "deribit"):
                self.exchange_source = exchange_source
                if exchange_source in ("auto", "okx"):
                    GLOBAL_EXCHANGE_CIRCUIT.reset_primary()
            if etf_netflow_m is not None:
                text = str(etf_netflow_m).strip()
                self.manual_etf_netflow_usd = None if text == "" else float(text) * 1e6
            if polymarket_bias is not None:
                text = str(polymarket_bias).strip()
                self.manual_polymarket_bias = None if text == "" else max(-1.0, min(1.0, float(text)))
        self.invalidate_candle_cache()

    def get_state_json(self) -> Dict[str, Any]:
        with self._lock:
            now_ep = time.time()
            seconds_to_next = max(0, int(self.next_refresh_epoch - now_ep))
            return {
                "ok": self.snapshot is not None,
                "mode": self.mode,
                "exchange_source": self.exchange_source,
                "primary_cooldown_sec": GLOBAL_EXCHANGE_CIRCUIT.remaining_cooldown_sec,
                "etf_last_error": GLOBAL_ETF_CACHE.last_error,
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
            "etf_source": (snap.get("market_meta") or {}).get("etf_source", "--"),
        }
    )


@app.route("/api/snapshot", methods=["GET"])
def api_snapshot():
    ensure_poller_started()
    return jsonify(store.get_state_json())


@app.route("/api/candles", methods=["GET"])
def api_candles():
    ensure_poller_started()
    start_arg = request.args.get("start_ms", "").strip()
    start_ms = int(start_arg) if start_arg.isdigit() else None
    return jsonify(store.get_candles_json(start_ms=start_ms))


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    ok, err = store.refresh_now(force_etf_refresh=True)
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

    store.refresh_now(force_etf_refresh=(str(body.get("etf_netflow_m", "")).strip() == ""))
    return jsonify(store.get_state_json())


# =============================================================================
# 3. 内嵌暗色量化终端前端 (V4.1: 集成 24H 预测路径 vs 实时行情同图对比)
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BTC 24H 极值磁吸带 & Deribit GEX 实时看板 (V4.1)</title>
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
    button.secondary.active { border-color: var(--blue); color: var(--blue); background: rgba(56,189,248,0.12); }
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
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
    .card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
    }
    .kpi-label { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
    .kpi-val { font-size: 21px; font-weight: 700; margin-top: 5px; }
    .kpi-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .main-grid { display: grid; grid-template-columns: 1.05fr 1.35fr; gap: 16px; }
    @media (max-width: 1040px) { .main-grid { grid-template-columns: 1fr; } }
    .panel-header {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;
      border-bottom: 1px solid var(--border); padding-bottom: 10px; flex-wrap: wrap; gap: 10px;
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

    /* ====== 预测 vs 实时行情同图对比样式 ====== */
    .compare-toolbar {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 10px; margin-bottom: 10px;
    }
    .compare-metrics { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .cmp-stat {
      background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px;
      padding: 5px 11px; font-size: 12px;
    }
    .cmp-stat span { color: var(--muted); margin-right: 6px; }
    .compare-legend {
      display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--muted);
    }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; }
    .legend-swatch { width: 20px; height: 0; display: inline-block; }
    .compare-canvas-wrap {
      position: relative; height: 430px; width: 100%; background: #0d1320;
      border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    }
    @media (max-width: 768px) { .compare-canvas-wrap { height: 340px; } }
    #forecastCompareCanvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
    .compare-tooltip {
      position: absolute; display: none; pointer-events: none; z-index: 20;
      background: rgba(13, 19, 32, 0.95); border: 1px solid #334155; border-radius: 8px;
      padding: 8px 11px; font-size: 12px; line-height: 1.55; color: var(--text);
      box-shadow: 0 8px 22px rgba(0,0,0,0.5); white-space: nowrap;
    }

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
        <h1>BTC 24H 极值磁吸带 & Deribit GEX 监控终端</h1>
        <div class="sub mono">最后更新: <span id="lastUpdated">--</span> · 下次 5 分钟自动刷新: <span id="countdown">--</span>s</div>
      </div>
      <div class="controls">
        <span id="etfPill" class="pill">--</span>
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
        <div class="alert-title" id="alertTitle">正在初始化行情、期权链与美国现货 ETF 流量...</div>
        <div class="alert-desc" id="alertMsg">等待后台协程完成首次 OKX + Deribit + ETF 自动抓取。</div>
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
        <div class="kpi-label">美国现货 ETF 单日净流入 (自动)</div>
        <div class="kpi-val mono" id="kpiEtf">--</div>
        <div class="kpi-sub mono" id="kpiEtfMeta">日期: -- | 来源: --</div>
      </div>
      <div class="card">
        <div class="kpi-label">舆情反身性得分 & 资金费率</div>
        <div class="kpi-val mono" id="kpiSent">--</div>
        <div class="kpi-sub mono" id="kpiFund">费率: -- | F&G: --</div>
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

    <!-- ====== 新增：24H 模拟预测走势 vs 实时行情同图对比模块 ====== -->
    <div class="card">
      <div class="panel-header">
        <div>
          <div class="panel-title">BTC 24H 极值磁吸带预测路径 vs 实时行情追踪对比</div>
          <div class="panel-meta mono" id="compareWindowMeta">
            窗口: 2026-11-22 18:00 ~ 2026-11-23 24:00 (EST) · K线源: 初始化中...
          </div>
        </div>
        <div class="controls">
          <button id="btnBandLocked" class="secondary active" onclick="setBandMode('locked')">锁定快照磁吸带</button>
          <button id="btnBandLive" class="secondary" onclick="setBandMode('live')">叠加实时磁吸带</button>
          <button id="btnReanchor" class="secondary" onclick="reanchorFromCurrentState()">按当前盘面重锚24H</button>
          <button id="btnResetScript" class="secondary" style="display:none;" onclick="resetToBaselineScript()">恢复初始剧本</button>
        </div>
      </div>

      <div class="compare-toolbar">
        <div class="compare-metrics mono">
          <div class="cmp-stat"><span>实时现价:</span><strong id="cmpLiveSpot" style="color:var(--blue)">--</strong></div>
          <div class="cmp-stat"><span>同期预测点位:</span><strong id="cmpPredSpot">--</strong></div>
          <div class="cmp-stat"><span>路径偏离度:</span><strong id="cmpDelta">--</strong></div>
          <div class="cmp-stat"><span>剧本验证状态:</span><strong id="cmpStatus">--</strong></div>
        </div>
        <div class="compare-legend">
          <span class="legend-item">
            <i class="legend-swatch" style="border-top:2px dashed rgba(226,232,240,0.85);"></i> 模拟预测走势 (Forecast)
          </span>
          <span class="legend-item">
            <i class="legend-swatch" style="border-top:2.5px solid #38bdf8;"></i> 实时收盘轨迹 + 15m 影线包络 (Live)
          </span>
          <span class="legend-item">
            <i class="legend-swatch" style="border-top:1.5px dashed #a78bfa;"></i> Gamma Flip
          </span>
          <span class="legend-item">
            <i class="legend-swatch" style="border-top:1.5px dashed #f59e0b;"></i> AVWAP
          </span>
        </div>
      </div>

      <div class="compare-canvas-wrap">
        <canvas id="forecastCompareCanvas"></canvas>
        <div id="compareTooltip" class="compare-tooltip mono"></div>
      </div>
    </div>

    <div class="footer-grid">
      <div class="card">
        <div class="panel-header">
          <div class="panel-title">数据源路由 & 可选手动覆盖</div>
          <div class="panel-meta">现货 ETF 已启用自动抓取 · 下方输入框留空即保持全自动</div>
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
            <label>ETF 净流入覆盖 ($M，留空=自动)</label>
            <input id="inputEtf" type="number" step="0.1" placeholder="留空自动抓取" />
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
  let latestSnapshotData = null;

  const fmtUSD = (v) => (v == null || isNaN(v)) ? "--" : "$" + Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 });

  // =========================================================================
  // 24H 预测路径 vs 实时行情同图对比引擎
  // =========================================================================
  const BASELINE_SCRIPT = {
    isCustomReanchor: false,
    windowLabel: "2026-11-22 18:00 ~ 2026-11-23 24:00 (EST)",
    startTs: Date.parse("2026-11-22T18:00:00-05:00"),
    endTs: Date.parse("2026-11-24T00:00:00-05:00"),
    levels: {
      cbUp: 65774.7,
      wickCeil: [64818.0, 65447.5],
      coreHigh: [62885.3, 63602.7],
      flip: 61248.9,
      avwap: 59527.0,
      coreLow: [58057.3, 58774.8],
      wickFloor: [56317.4, 56945.2],
      cbDown: 56035.8,
    },
    waypoints: [
      [0.0,  61248.2], [1.5,  60910.0], [4.0,  61420.0], [5.5,  61320.0],
      [8.0,  62350.0], [9.5,  63060.0], [10.5, 62960.0], [12.0, 63180.0],
      [14.0, 61080.0], [15.0, 60850.0], [16.3, 59280.0], [18.0, 58260.0],
      [19.2, 58560.0], [20.2, 58190.0], [21.6, 58740.0], [23.6, 58290.0],
      [26.5, 58840.0], [28.0, 59080.0], [30.0, 59680.0],
    ],
    annotations: [
      { h: 7.2,  price: 63920, text: "① 今晚~周一晨: 站稳 Flip 激发负 Gamma 助推，冲高测试 Core High" },
      { h: 16.2, price: 60450, text: "② 周一美盘: 多头踩踏 & ETF 承压，回踩 Core Low 寻底" },
      { h: 22.8, price: 57560, text: "③ 周一晚: Core Low 企稳，回归中枢震荡" },
    ],
  };

  let activeScript = JSON.parse(JSON.stringify(BASELINE_SCRIPT));
  let bandMode = "locked"; // "locked" | "live"
  let candleSeries = [];   // [{ ts, o, h, l, c, v }]
  let candleSourceLabel = "初始化中...";
  let compareHoverX = null;
  let liveWs = null;

  function getActiveLevels() {
    if (bandMode === "live" && latestSnapshotData && latestSnapshotData.prediction) {
      const p = latestSnapshotData.prediction;
      const m = latestSnapshotData.market_meta || {};
      return {
        cbUp: p.invalidation_bull_trigger,
        wickCeil: p.wick_ceiling_zone,
        coreHigh: p.core_high_zone,
        flip: m.gamma_flip || p.spot_price,
        avwap: m.avwap || p.spot_price,
        coreLow: p.core_low_zone,
        wickFloor: p.wick_floor_zone,
        cbDown: p.invalidation_bear_trigger,
      };
    }
    return activeScript.levels;
  }

  function interpolateForecast(ts) {
    const pts = activeScript.waypoints.map(([h, p]) => ({
      ts: activeScript.startTs + h * 3600 * 1000,
      price: p,
    }));
    if (ts <= pts[0].ts) return pts[0].price;
    const last = pts[pts.length - 1];
    if (ts >= last.ts) return last.price;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      if (ts >= a.ts && ts <= b.ts) {
        const t = (ts - a.ts) / (b.ts - a.ts);
        const mu = (1 - Math.cos(t * Math.PI)) / 2;
        return a.price * (1 - mu) + b.price * mu;
      }
    }
    return last.price;
  }

  function setBandMode(mode) {
    bandMode = mode;
    document.getElementById("btnBandLocked").classList.toggle("active", mode === "locked");
    document.getElementById("btnBandLive").classList.toggle("active", mode === "live");
    renderComparisonChart();
  }

  function reanchorFromCurrentState() {
    if (!latestSnapshotData || !latestSnapshotData.prediction) return;
    const p = latestSnapshotData.prediction;
    const m = latestSnapshotData.market_meta || {};
    const nowMs = Date.now();
    const spot = p.spot_price;
    const flip = m.gamma_flip || spot;
    const avwap = m.avwap || spot;
    const coreHMid = (p.core_high_zone[0] + p.core_high_zone[1]) / 2;
    const coreLMid = (p.core_low_zone[0] + p.core_low_zone[1]) / 2;
    const bullBias = spot >= flip;

    activeScript = {
      isCustomReanchor: true,
      windowLabel: "动态重锚窗口: 当前时刻 → 未来 24H",
      startTs: nowMs - 2 * 3600 * 1000, // 保留前 2 小时实盘衔接
      endTs: nowMs + 24 * 3600 * 1000,
      levels: {
        cbUp: p.invalidation_bull_trigger,
        wickCeil: [...p.wick_ceiling_zone],
        coreHigh: [...p.core_high_zone],
        flip,
        avwap,
        coreLow: [...p.core_low_zone],
        wickFloor: [...p.wick_floor_zone],
        cbDown: p.invalidation_bear_trigger,
      },
      waypoints: bullBias
        ? [
            [0.0, spot], [2.0, spot], [5.5, (spot + coreHMid) / 2], [9.0, coreHMid],
            [13.5, avwap], [18.0, coreLMid + (avwap - coreLMid) * 0.25], [22.0, avwap], [26.0, (avwap + flip) / 2],
          ]
        : [
            [0.0, spot], [2.0, spot], [6.0, (spot + coreLMid) / 2], [10.0, coreLMid],
            [14.5, avwap], [19.5, (avwap + coreHMid) / 2], [23.0, flip], [26.0, avwap],
          ],
      annotations: [],
    };
    document.getElementById("btnResetScript").style.display = "inline-block";
    fetchCandles();
  }

  function resetToBaselineScript() {
    activeScript = JSON.parse(JSON.stringify(BASELINE_SCRIPT));
    document.getElementById("btnResetScript").style.display = "none";
    fetchCandles();
  }

  function updateCompareMetaAndMetrics() {
    document.getElementById("compareWindowMeta").textContent =
      `窗口: ${activeScript.windowLabel} · K线源: ${candleSourceLabel}`;

    const liveSpot = (candleSeries.length ? candleSeries[candleSeries.length - 1].c : null)
      ?? (latestSnapshotData?.prediction?.spot_price ?? null);
    if (liveSpot == null) return;

    const latestTs = candleSeries.length ? candleSeries[candleSeries.length - 1].ts : Date.now();
    const clampedTs = Math.min(Math.max(latestTs, activeScript.startTs), activeScript.endTs);
    const predSpot = interpolateForecast(clampedTs);
    const diff = liveSpot - predSpot;
    const diffPct = (diff / predSpot) * 100;
    const lv = getActiveLevels();

    document.getElementById("cmpLiveSpot").textContent = fmtUSD(liveSpot);
    document.getElementById("cmpPredSpot").textContent = fmtUSD(predSpot);

    const deltaEl = document.getElementById("cmpDelta");
    const sign = diff >= 0 ? "+" : "-";
    deltaEl.textContent = `${sign}$${Math.abs(diff).toFixed(1)} (${diff >= 0 ? "+" : ""}${diffPct.toFixed(2)}%)`;
    deltaEl.style.color = diff >= 0 ? "var(--green)" : "var(--red)";

    const statusEl = document.getElementById("cmpStatus");
    if (liveSpot >= lv.cbUp || liveSpot <= lv.cbDown) {
      statusEl.textContent = "触发熔断警戒";
      statusEl.style.color = "var(--red)";
    } else if (Math.abs(diffPct) <= 1.5) {
      statusEl.textContent = "吻合预测通道";
      statusEl.style.color = "var(--green)";
    } else if (diff > 0) {
      statusEl.textContent = "强于预测基准 (偏多运行)";
      statusEl.style.color = "var(--blue)";
    } else {
      statusEl.textContent = "弱于预测基准 (回踩承压)";
      statusEl.style.color = "var(--amber)";
    }
  }

  function renderComparisonChart() {
    const canvas = document.getElementById("forecastCompareCanvas");
    const tooltip = document.getElementById("compareTooltip");
    if (!canvas) return;
    const wrap = canvas.parentElement;
    const rect = wrap.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 18, right: 68, bottom: 36, left: 14 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const lv = getActiveLevels();
    const startTs = activeScript.startTs;
    const endTs = activeScript.endTs;

    const visibleCandles = candleSeries.filter(
      (c) => c.ts >= startTs - 15 * 60 * 1000 && c.ts <= endTs + 15 * 60 * 1000
    );

    const allPrices = [
      lv.cbUp, lv.cbDown, lv.wickCeil[1], lv.wickFloor[0],
      ...activeScript.waypoints.map((w) => w[1]),
      ...visibleCandles.flatMap((c) => [c.h, c.l]),
    ].filter((v) => typeof v === "number" && !isNaN(v));

    const minP = Math.min(...allPrices) - 450;
    const maxP = Math.max(...allPrices) + 450;

    const xOf = (ts) => pad.left + ((ts - startTs) / (endTs - startTs)) * plotW;
    const yOf = (p) => pad.top + ((maxP - p) / (maxP - minP)) * plotH;

    ctx.clearRect(0, 0, W, H);

    // 1. 绘制磁吸带区间色块
    const drawZone = (range, fill, border, label) => {
      const yTop = yOf(range[1]);
      const yBot = yOf(range[0]);
      ctx.fillStyle = fill;
      ctx.fillRect(pad.left, yTop, plotW, Math.max(2, yBot - yTop));
      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, yTop); ctx.lineTo(pad.left + plotW, yTop);
      ctx.moveTo(pad.left, yBot); ctx.lineTo(pad.left + plotW, yBot);
      ctx.stroke();
      ctx.fillStyle = border;
      ctx.font = "600 11px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText(`${label} ${fmtUSD(range[0])} ~ ${fmtUSD(range[1])}`, pad.left + 8, yTop + 13);
    };

    drawZone(lv.wickCeil,  "rgba(244,63,94,0.12)",  "rgba(253,164,175,0.72)", "极端轧空上限 (Wick Ceiling)");
    drawZone(lv.coreHigh,  "rgba(16,185,129,0.12)", "rgba(110,231,183,0.72)", "预期高点磁吸 (Core High)");
    drawZone(lv.coreLow,   "rgba(56,189,248,0.12)", "rgba(125,211,252,0.72)", "预期低点磁吸 (Core Low)");
    drawZone(lv.wickFloor, "rgba(245,158,11,0.12)", "rgba(252,211,77,0.72)",  "极端插针底线 (Wick Floor)");

    // 2. 背景价格网格与右侧刻度
    const stepP = (maxP - minP) > 14000 ? 2000 : 1000;
    const firstGridP = Math.ceil(minP / stepP) * stepP;
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (let p = firstGridP; p <= maxP; p += stepP) {
      const y = yOf(p);
      ctx.strokeStyle = "rgba(37, 48, 71, 0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.fillStyle = "#8b9bb4";
      ctx.fillText("$" + p.toLocaleString("en-US"), pad.left + plotW + 6, y + 4);
    }

    // 3. 底部时间轴网格（按美东时间 EST 标注）
    const totalHours = (endTs - startTs) / 3600000;
    const hourStep = totalHours > 26 ? 4 : 3;
    for (let h = 0; h <= totalHours; h += hourStep) {
      const ts = startTs + h * 3600 * 1000;
      const x = xOf(ts);
      ctx.strokeStyle = "rgba(37, 48, 71, 0.40)";
      ctx.beginPath();
      ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();

      const label = new Intl.DateTimeFormat("en-US", {
        timeZone: "America/New_York",
        month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
      }).format(new Date(ts)).replace(", ", " ");
      ctx.fillStyle = "#8b9bb4";
      const textX = Math.min(Math.max(pad.left, x - 34), pad.left + plotW - 68);
      ctx.fillText(label, textX, pad.top + plotH + 21);
    }

    // 4. 绘制关键水平参考线 (熔断线、Gamma Flip、AVWAP)
    const drawRefLine = (price, color, dash, tagText) => {
      if (!price) return;
      const y = yOf(price);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.setLineDash(dash);
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.font = "600 11px ui-monospace, monospace";
      const tw = ctx.measureText(tagText).width;
      ctx.fillStyle = "rgba(11, 15, 23, 0.82)";
      ctx.fillRect(pad.left + plotW - tw - 12, y - 14, tw + 8, 13);
      ctx.fillStyle = color;
      ctx.fillText(tagText, pad.left + plotW - tw - 8, y - 4);
      ctx.restore();
    };

    drawRefLine(lv.cbUp,   "#f43f5e", [4, 4], `向上熔断 ${fmtUSD(lv.cbUp)}`);
    drawRefLine(lv.flip,   "#a78bfa", [5, 4], `Gamma Flip ${fmtUSD(lv.flip)}`);
    drawRefLine(lv.avwap,  "#f59e0b", [5, 4], `AVWAP ${fmtUSD(lv.avwap)}`);
    drawRefLine(lv.cbDown, "#f43f5e", [4, 4], `向下熔断 ${fmtUSD(lv.cbDown)}`);

    // 5. 底部 15m 实时成交量柱
    if (visibleCandles.length > 1) {
      const maxVol = Math.max(...visibleCandles.map((c) => c.v || 1), 1);
      const volMaxH = plotH * 0.14;
      const barW = Math.max(2, (plotW / ((endTs - startTs) / (15 * 60 * 1000))) * 0.68);
      visibleCandles.forEach((c) => {
        if (c.ts < startTs || c.ts > endTs) return;
        const x = xOf(c.ts);
        const bh = ((c.v || 0) / maxVol) * volMaxH;
        ctx.fillStyle = c.c >= c.o ? "rgba(16, 185, 129, 0.28)" : "rgba(244, 63, 94, 0.28)";
        ctx.fillRect(x - barW / 2, pad.top + plotH - bh, barW, bh);
      });
    }

    // 6. 绘制模拟预测曲线 (白色虚线)
    ctx.save();
    ctx.strokeStyle = "rgba(226, 232, 240, 0.82)";
    ctx.lineWidth = 2.1;
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    const sampleCount = 200;
    for (let i = 0; i <= sampleCount; i++) {
      const ts = startTs + ((endTs - startTs) * i) / sampleCount;
      const x = xOf(ts);
      const y = yOf(interpolateForecast(ts));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 阶段剧本文字标注 (仅在宽屏且存在剧本标注时显示)
    if (W >= 820 && activeScript.annotations && activeScript.annotations.length) {
      ctx.save();
      ctx.font = "11.5px -apple-system, BlinkMacSystemFont, sans-serif";
      activeScript.annotations.forEach((ann) => {
        const ax = xOf(startTs + ann.h * 3600 * 1000);
        const ay = yOf(ann.price);
        const tw = ctx.measureText(ann.text).width;
        const bx = Math.min(Math.max(pad.left + 8, ax - tw / 2), pad.left + plotW - tw - 12);
        ctx.fillStyle = "rgba(13, 19, 32, 0.82)";
        ctx.fillRect(bx - 5, ay - 12, tw + 10, 17);
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(ann.text, bx, ay);
      });
      ctx.restore();
    }

    // 7. 绘制实时行情 15m 影线包络 + 实时收盘轨迹 (高亮青蓝实线)
    const clipped = visibleCandles.filter((c) => c.ts >= startTs && c.ts <= endTs);
    if (clipped.length > 0) {
      // 15m 高低影线包络带 (直观展示是否插针触碰磁吸带)
      ctx.save();
      ctx.fillStyle = "rgba(56, 189, 248, 0.16)";
      ctx.beginPath();
      clipped.forEach((c, idx) => {
        const x = xOf(c.ts), y = yOf(c.h);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      for (let i = clipped.length - 1; i >= 0; i--) {
        ctx.lineTo(xOf(clipped[i].ts), yOf(clipped[i].l));
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // 实时收盘主线
      ctx.save();
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.6;
      ctx.shadowColor = "rgba(56, 189, 248, 0.45)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      clipped.forEach((c, idx) => {
        const x = xOf(c.ts), y = yOf(c.c);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // 最新实时价格圆点与当前进度竖线
      const lastBar = clipped[clipped.length - 1];
      const lx = xOf(lastBar.ts);
      const ly = yOf(lastBar.c);

      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(lx, pad.top); ctx.lineTo(lx, pad.top + plotH);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(lx, ly, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // 8. 鼠标悬停十字光标与实时误差浮窗
    if (compareHoverX !== null && compareHoverX >= pad.left && compareHoverX <= pad.left + plotW) {
      const hoverTs = startTs + ((compareHoverX - pad.left) / plotW) * (endTs - startTs);
      const predAtHover = interpolateForecast(hoverTs);

      ctx.save();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(compareHoverX, pad.top);
      ctx.lineTo(compareHoverX, pad.top + plotH);
      ctx.stroke();
      ctx.restore();

      let liveRowHtml = `<span style="color:#64748b">尚未到达该时刻</span>`;
      if (clipped.length > 0) {
        let nearest = clipped[0];
        for (const bar of clipped) {
          if (Math.abs(bar.ts - hoverTs) < Math.abs(nearest.ts - hoverTs)) nearest = bar;
        }
        if (Math.abs(nearest.ts - hoverTs) <= 18 * 60 * 1000) {
          const d = nearest.c - predAtHover;
          const dp = (d / predAtHover) * 100;
          const col = d >= 0 ? "#10b981" : "#f43f5e";
          liveRowHtml = `<strong style="color:#38bdf8">${fmtUSD(nearest.c)}</strong> <span style="color:${col}">(${d >= 0 ? "+" : ""}${d.toFixed(1)} / ${d >= 0 ? "+" : ""}${dp.toFixed(2)}%)</span><br/><span>15m 高/低:</span> ${fmtUSD(nearest.h)} / ${fmtUSD(nearest.l)}`;
        }
      }

      const estTimeStr = new Intl.DateTimeFormat("zh-CN", {
        timeZone: "America/New_York",
        month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
      }).format(new Date(hoverTs));

      tooltip.style.display = "block";
      tooltip.innerHTML = `
        <div><span style="color:#8b9bb4">美东时间:</span> <strong>${estTimeStr} EST</strong></div>
        <div><span style="color:#8b9bb4">预测点位:</span> <strong>${fmtUSD(predAtHover)}</strong></div>
        <div><span style="color:#8b9bb4">实盘成交:</span> ${liveRowHtml}</div>
      `;
      const tipLeft = (compareHoverX + 14 + tooltip.offsetWidth > W - 10)
        ? Math.max(8, compareHoverX - tooltip.offsetWidth - 14)
        : compareHoverX + 14;
      tooltip.style.left = tipLeft + "px";
      tooltip.style.top = (pad.top + 10) + "px";
    } else {
      tooltip.style.display = "none";
    }
  }

  function pushLiveTick(price, tsMs) {
    if (!price || isNaN(price)) return;
    const ts = tsMs || Date.now();
    const bucket = Math.floor(ts / (15 * 60 * 1000)) * (15 * 60 * 1000);
    if (!candleSeries.length) {
      candleSeries.push({ ts: bucket, o: price, h: price, l: price, c: price, v: 0 });
    } else {
      const last = candleSeries[candleSeries.length - 1];
      if (bucket > last.ts) {
        candleSeries.push({ ts: bucket, o: last.c, h: Math.max(last.c, price), l: Math.min(last.c, price), c: price, v: 0 });
      } else {
        last.c = price;
        last.h = Math.max(last.h, price);
        last.l = Math.min(last.l, price);
      }
    }
    updateCompareMetaAndMetrics();
    renderComparisonChart();
  }

  async function fetchCandles() {
    try {
      const qs = activeScript.isCustomReanchor ? `?start_ms=${Math.floor(activeScript.startTs)}` : "";
      const res = await fetch("/api/candles" + qs);
      const json = await res.json();
      if (json && Array.isArray(json.candles) && json.candles.length > 0) {
        candleSeries = json.candles;
        candleSourceLabel = json.source || "OKX/DERIBIT";
      } else if (json && json.error) {
        candleSourceLabel = "使用快照点位回传";
      }
      updateCompareMetaAndMetrics();
      renderComparisonChart();
    } catch (e) {
      console.warn("Candles fetch warning:", e);
    }
  }

  function connectRealtimeTickerWs() {
    if (liveWs) {
      try { liveWs.close(); } catch (_) {}
      liveWs = null;
    }
    if (currentMode !== "LIVE") return;

    const connectDeribitWs = () => {
      try {
        const ws = new WebSocket("wss://www.deribit.com/ws/api/v2");
        liveWs = ws;
        ws.onopen = () => {
          ws.send(JSON.stringify({
            jsonrpc: "2.0", id: 1, method: "public/subscribe",
            params: { channels: ["ticker.BTC-PERPETUAL.100ms"] },
          }));
        };
        ws.onmessage = (ev) => {
          if (currentMode !== "LIVE") return;
          const msg = JSON.parse(ev.data);
          const data = msg?.params?.data;
          if (data && data.last_price) {
            pushLiveTick(Number(data.last_price), Number(data.timestamp || Date.now()));
          }
        };
      } catch (_) {}
    };

    try {
      const ws = new WebSocket("wss://ws.okx.com:8443/ws/v5/public");
      liveWs = ws;
      let gotTick = false;
      const fallbackTimer = setTimeout(() => {
        if (!gotTick && currentMode === "LIVE") {
          try { ws.close(); } catch (_) {}
          connectDeribitWs();
        }
      }, 6000);

      ws.onopen = () => {
        ws.send(JSON.stringify({ op: "subscribe", args: [{ channel: "tickers", instId: "BTC-USDT-SWAP" }] }));
      };
      ws.onmessage = (ev) => {
        if (currentMode !== "LIVE") return;
        const msg = JSON.parse(ev.data);
        if (msg.data && msg.data[0] && msg.data[0].last) {
          gotTick = true;
          clearTimeout(fallbackTimer);
          pushLiveTick(Number(msg.data[0].last), Number(msg.data[0].ts || Date.now()));
        }
      };
      ws.onerror = () => {
        clearTimeout(fallbackTimer);
        if (!gotTick) connectDeribitWs();
      };
    } catch (_) {
      connectDeribitWs();
    }
  }

  (function bindCompareCanvasEvents() {
    const canvas = document.getElementById("forecastCompareCanvas");
    const updateHover = (clientX) => {
      const rect = canvas.getBoundingClientRect();
      compareHoverX = clientX - rect.left;
      renderComparisonChart();
    };
    canvas.addEventListener("mousemove", (e) => updateHover(e.clientX));
    canvas.addEventListener("touchmove", (e) => {
      if (e.touches && e.touches[0]) updateHover(e.touches[0].clientX);
    }, { passive: true });
    canvas.addEventListener("mouseleave", () => {
      compareHoverX = null;
      renderComparisonChart();
    });
    canvas.addEventListener("touchend", () => {
      compareHoverX = null;
      renderComparisonChart();
    });
    window.addEventListener("resize", renderComparisonChart);
  })();

  // =========================================================================
  // 原有极值磁吸阶梯 & GEX 柱状图渲染逻辑
  // =========================================================================
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
    const prevMode = currentMode;
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

    if (prevMode !== currentMode) {
      connectRealtimeTickerWs();
      fetchCandles();
    }

    if (!resp.data) return;
    latestSnapshotData = resp.data;

    const { prediction: pred, market_meta: meta, alert, strike_gex_bars: bars, regime_label, active_exchange, failover_reason } = resp.data;

    const exPill = document.getElementById("exchangePill");
    exPill.textContent = "永续源: " + (active_exchange || "--");
    exPill.className = "pill " + (active_exchange === "OKX" ? "live" : (active_exchange === "DERIBIT" ? "alt" : "demo"));

    const etfVal = Number(meta.etf_netflow_m || 0);
    const etfPill = document.getElementById("etfPill");
    etfPill.textContent = `ETF: ${etfVal >= 0 ? "+" : ""}${etfVal.toFixed(1)}M`;
    etfPill.className = "pill " + (etfVal > 0 ? "live" : (etfVal < 0 ? "demo" : "alt"));

    const fBar = document.getElementById("failoverBar");
    const notices = [];
    if (failover_reason && currentMode === "LIVE") {
      const cd = resp.primary_cooldown_sec > 0 ? `（OKX 冷却剩余 ${Math.ceil(resp.primary_cooldown_sec / 60)}m）` : "";
      notices.push(`永续路由: ${failover_reason}${cd}`);
    }
    if (resp.etf_last_error && currentMode === "LIVE" && String(meta.etf_source || "").startsWith("Auto Failed")) {
      notices.push(`ETF 自动抓取提示: ${resp.etf_last_error}（在 .env 配置免费 SOSO_API_KEY 即可 100% 稳定自动拉取）`);
    }
    if (notices.length > 0) {
      fBar.style.display = "block";
      fBar.textContent = notices.join("  |  ");
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

    const etfEl = document.getElementById("kpiEtf");
    etfEl.textContent = `${etfVal >= 0 ? "+" : ""}$${Math.abs(etfVal).toFixed(1)}M`;
    etfEl.style.color = etfVal > 0 ? "var(--green)" : (etfVal < 0 ? "var(--red)" : "var(--text)");
    document.getElementById("kpiEtfMeta").textContent = `交易日: ${meta.etf_date || "--"} | ${meta.etf_source || "--"}`;

    document.getElementById("kpiSent").textContent = `${pred.raw_sentiment >= 0 ? "+" : ""}${pred.raw_sentiment.toFixed(2)} → ${pred.adjusted_sentiment >= 0 ? "+" : ""}${pred.adjusted_sentiment.toFixed(2)}`;
    document.getElementById("kpiFund").textContent = `8H费率: ${meta.funding_rate_pct}% | F&G: ${meta.fear_greed} | 24H OI: ${meta.oi_24h_pct >= 0 ? "+" : ""}${meta.oi_24h_pct}%`;

    renderLadder(pred, meta);
    renderGexChart(bars || [], meta);

    if (currentMode === "LIVE" && pred.spot_price) {
      pushLiveTick(pred.spot_price, Date.now());
    } else {
      updateCompareMetaAndMetrics();
      renderComparisonChart();
    }

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
      await fetchCandles();
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
    await fetchCandles();
  }

  async function applyConfig() {
    const etfStr = document.getElementById("inputEtf").value;
    const polyStr = document.getElementById("inputPoly").value;
    const srcVal = document.getElementById("selectSource").value;
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exchange_source: srcVal,
        etf_netflow_m: etfStr,
        polymarket_bias: polyStr,
      }),
    });
    const json = await res.json();
    applyStateToUI(json);
    await fetchCandles();
  }

  setInterval(() => {
    if (countdownSec > 0) {
      countdownSec -= 1;
      document.getElementById("countdown").textContent = countdownSec;
    } else {
      fetchSnapshot();
      fetchCandles();
    }
  }, 1000);

  renderComparisonChart();
  Promise.all([fetchSnapshot(), fetchCandles()]).then(() => {
    connectRealtimeTickerWs();
    setTimeout(() => {
      if (document.getElementById("kpiSpot").textContent === "--") {
        fetchSnapshot();
        fetchCandles();
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
