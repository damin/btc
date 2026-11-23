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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
    _EST_TZ = ZoneInfo("America/New_York")
except Exception:
    _EST_TZ = timezone(timedelta(hours=-5))

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
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ and val != "":
            os.environ[key] = val


_load_local_dotenv()

REFRESH_INTERVAL_SEC = int(os.getenv("REFRESH_INTERVAL_SEC", "300"))
SESSION_CACHE_FILE = Path(__file__).resolve().parent / ".forecast_session_state.json"

BASELINE_START_MS = int(datetime(2026, 11, 22, 18, 0, 0, tzinfo=_EST_TZ).timestamp() * 1000)
BASELINE_END_MS = BASELINE_START_MS + 30 * 3600 * 1000


# =============================================================================
# 1. 24H 预测会话与跨日自动轮转引擎 (Auto-Rollover Session Engine)
# =============================================================================

def _fmt_est_window(start_ms: int, end_ms: int) -> str:
    s_dt = datetime.fromtimestamp(start_ms / 1000.0, tz=_EST_TZ)
    e_dt = datetime.fromtimestamp(end_ms / 1000.0, tz=_EST_TZ)
    s_str = s_dt.strftime("%Y-%m-%d %H:%M")
    if e_dt.hour == 0 and e_dt.minute == 0:
        e_str = f"{(e_dt - timedelta(days=1)).strftime('%Y-%m-%d')} 24:00"
    else:
        e_str = e_dt.strftime("%Y-%m-%d %H:%M")
    return f"{s_str} ~ {e_str} (EST)"


def _initial_locked_session() -> Dict[str, Any]:
    return {
        "sessionId": "baseline-20261122",
        "isManual": False,
        "autoRollover": True,
        "windowLabel": "2026-11-22 18:00 ~ 2026-11-23 24:00 (EST)",
        "startTs": BASELINE_START_MS,
        "endTs": BASELINE_END_MS,
        "levels": {
            "cbUp": 65774.7,
            "wickCeil": [64818.0, 65447.5],
            "coreHigh": [62885.3, 63602.7],
            "flip": 61248.9,
            "avwap": 59527.0,
            "coreLow": [58057.3, 58774.8],
            "wickFloor": [56317.4, 56945.2],
            "cbDown": 56035.8,
        },
        "waypoints": [
            [0.0, 61248.2], [1.5, 60910.0], [4.0, 61420.0], [5.5, 61320.0],
            [8.0, 62350.0], [9.5, 63060.0], [10.5, 62960.0], [12.0, 63180.0],
            [14.0, 61080.0], [15.0, 60850.0], [16.3, 59280.0], [18.0, 58260.0],
            [19.2, 58560.0], [20.2, 58190.0], [21.6, 58740.0], [23.6, 58290.0],
            [26.5, 58840.0], [28.0, 59080.0], [30.0, 59680.0],
        ],
        "annotations": [
            {"h": 7.2, "price": 63920.0, "text": "① 站稳 Flip 激发负 Gamma 助推，冲高测试 Core High"},
            {"h": 16.2, "price": 60450.0, "text": "② 美盘多头换手 & ETF 承压，回踩 Core Low 寻底"},
            {"h": 22.8, "price": 57560.0, "text": "③ Core Low 企稳，回归中枢震荡"},
        ],
    }


def _generate_24h_session(
    payload: Dict[str, Any],
    start_ms: int,
    end_ms: int,
    session_id: str,
    is_manual: bool = False,
    anchor_open_price: Optional[float] = None,
) -> Dict[str, Any]:
    pred = payload["prediction"]
    meta = payload["market_meta"]

    spot = float(anchor_open_price if anchor_open_price is not None else pred["spot_price"])
    raw_flip = float(meta.get("gamma_flip") or spot)
    raw_avwap = float(meta.get("avwap") or spot)
    net_gex_m = float(meta.get("net_gex_m") or 0.0)
    etf_m = float(meta.get("etf_netflow_m") or 0.0)
    fund_pct = float(meta.get("funding_rate_pct") or 0.0)

    ch_low, ch_high = float(pred["core_high_zone"][0]), float(pred["core_high_zone"][1])
    cl_low, cl_high = float(pred["core_low_zone"][0]), float(pred["core_low_zone"][1])
    ch_mid, cl_mid = round((ch_low + ch_high) / 2.0, 1), round((cl_low + cl_high) / 2.0, 1)

    safe_flip = min(max(raw_flip, cl_low), ch_high)
    safe_avwap = min(max(raw_avwap, cl_low), ch_high)
    equilibrium = round(safe_avwap * 0.55 + safe_flip * 0.45, 1)
    total_h = max(24.0, round((end_ms - start_ms) / 3600000.0, 2))
    scale = total_h / 24.0

    if net_gex_m < 0 and spot >= raw_flip:
        dip_target = cl_mid if ((etf_m < -100.0) or (fund_pct > 0.005)) else round((safe_avwap + cl_high) / 2.0, 1)
        waypoints = [
            [0.0, round(spot, 1)],
            [round(2.5 * scale, 2), round(spot + (ch_low - spot) * 0.45, 1)],
            [round(6.5 * scale, 2), ch_mid],
            [round(8.5 * scale, 2), round(ch_high - (ch_high - ch_low) * 0.22, 1)],
            [round(12.5 * scale, 2), round(safe_flip, 1)],
            [round(16.5 * scale, 2), dip_target],
            [round(19.5 * scale, 2), round((dip_target + safe_avwap) / 2.0, 1)],
            [total_h, equilibrium],
        ]
        annotations = [
            {"h": round(5.8 * scale, 1), "price": round(ch_high + (ch_high - ch_low) * 0.45, 1), "text": "① 初段: 站稳 Flip 负 Gamma 助推上探 Core High"},
            {"h": round(14.2 * scale, 1), "price": round((safe_flip + dip_target) / 2.0, 1), "text": "② 美盘: 获利盘与 ETF 博弈，回踩支撑换手"},
            {"h": round(20.2 * scale, 1), "price": round(min(dip_target, cl_high) - (cl_high - cl_low) * 0.45, 1), "text": "③ 尾段: 抛压消化企稳，向均值中枢回归"},
        ]
    elif net_gex_m < 0 and spot < raw_flip:
        waypoints = [
            [0.0, round(spot, 1)],
            [round(2.8 * scale, 2), round(spot - (spot - cl_high) * 0.55, 1)],
            [round(6.5 * scale, 2), cl_mid],
            [round(8.8 * scale, 2), round(cl_low + (cl_high - cl_low) * 0.25, 1)],
            [round(13.5 * scale, 2), round(safe_avwap, 1)],
            [round(17.5 * scale, 2), round((safe_avwap + safe_flip) / 2.0, 1)],
            [round(20.5 * scale, 2), round(safe_flip, 1)],
            [total_h, equilibrium],
        ]
        annotations = [
            {"h": round(5.8 * scale, 1), "price": round(cl_low - (cl_high - cl_low) * 0.48, 1), "text": "① 初段: Flip 下方负 Gamma 助跌下探 Core Low"},
            {"h": round(14.0 * scale, 1), "price": round(safe_avwap + (ch_low - cl_high) * 0.18, 1), "text": "② 中段: 空头回补修复，反抽测试 AVWAP"},
            {"h": round(20.0 * scale, 1), "price": round(safe_flip + (ch_low - cl_high) * 0.22, 1), "text": "③ 尾段: 围绕 Flip 与中枢区间收敛震荡"},
        ]
    else:
        first_leg = ch_low if spot <= equilibrium else cl_high
        second_leg = cl_high if spot <= equilibrium else ch_low
        waypoints = [
            [0.0, round(spot, 1)],
            [round(4.5 * scale, 2), round((spot + first_leg) / 2.0, 1)],
            [round(8.5 * scale, 2), round(first_leg, 1)],
            [round(13.0 * scale, 2), equilibrium],
            [round(17.5 * scale, 2), round(second_leg, 1)],
            [round(21.0 * scale, 2), round(safe_avwap, 1)],
            [total_h, equilibrium],
        ]
        annotations = [
            {"h": round(7.2 * scale, 1), "price": round(ch_high + (ch_high - ch_low) * 0.35, 1), "text": "① 正 Gamma 做市商高抛吸波，冲高受阻 Core 内沿"},
            {"h": round(16.8 * scale, 1), "price": round(cl_low - (cl_high - cl_low) * 0.35, 1), "text": "② 低位对冲买盘托底，全天围绕 AVWAP 收敛"},
        ]

    return {
        "sessionId": session_id,
        "isManual": is_manual,
        "autoRollover": True,
        "windowLabel": _fmt_est_window(start_ms, end_ms),
        "startTs": start_ms,
        "endTs": end_ms,
        "levels": {
            "cbUp": float(pred["invalidation_bull_trigger"]),
            "wickCeil": [float(pred["wick_ceiling_zone"][0]), float(pred["wick_ceiling_zone"][1])],
            "coreHigh": [ch_low, ch_high],
            "flip": raw_flip,
            "avwap": raw_avwap,
            "coreLow": [cl_low, cl_high],
            "wickFloor": [float(pred["wick_floor_zone"][0]), float(pred["wick_floor_zone"][1])],
            "cbDown": float(pred["invalidation_bear_trigger"]),
        },
        "waypoints": waypoints,
        "annotations": annotations,
    }


def _load_persisted_session() -> Optional[Dict[str, Any]]:
    if not SESSION_CACHE_FILE.is_file():
        return None
    try:
        data = json.loads(SESSION_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "startTs" in data and "endTs" in data and "waypoints" in data:
            return data
    except Exception:
        pass
    return None


def _save_persisted_session(session: Dict[str, Any]) -> None:
    try:
        SESSION_CACHE_FILE.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# 2. 状态管理与交易所 K 线缓存 (Carbon State Manager)
# =============================================================================

def _demo_strike_gex_bars() -> List[Dict[str, Any]]:
    sample = [
        (54000, 0.4, -4.8), (55000, 0.7, -9.2), (56000, 1.1, -11.5),
        (57000, 1.8, -14.6), (57500, 2.2, -21.4), (58000, 3.5, -16.8),
        (59000, 5.4, -13.1), (60000, 8.9, -15.4), (61000, 11.2, -10.6),
        (61500, 10.4, -10.1), (62000, 15.8, -6.2), (63000, 24.6, -4.1),
        (64000, 16.5, -2.3), (65000, 12.1, -1.2), (66000, 7.8, -0.6),
    ]
    return [{"strike": k, "call_gex_m": c, "put_gex_m": p, "net_gex_m": round(c + p, 2)} for k, c, p in sample]


def _build_demo_candles(session: Dict[str, Any]) -> List[Dict[str, float]]:
    start_ms, end_ms = int(session["startTs"]), int(session["endTs"])
    waypoints = session["waypoints"]
    total_steps = max(24, int(((end_ms - start_ms) / (15 * 60 * 1000)) * 0.65))
    bars: List[Dict[str, float]] = []
    prev_c = float(waypoints[0][1])
    for step in range(total_steps):
        h = step * 0.25
        ts = start_ms + step * 15 * 60 * 1000
        base = float(waypoints[-1][1])
        for i in range(len(waypoints) - 1):
            h0, p0 = float(waypoints[i][0]), float(waypoints[i][1])
            h1, p1 = float(waypoints[i + 1][0]), float(waypoints[i + 1][1])
            if h0 <= h <= h1:
                mu = (1.0 - math.cos(((h - h0) / max(h1 - h0, 1e-6)) * math.pi)) / 2.0
                base = p0 * (1.0 - mu) + p1 * mu
                break
        close_p = round(base + math.sin(step * 0.85) * 140.0 + math.cos(step * 0.37) * 75.0, 1)
        open_p = prev_c
        high_p = round(max(open_p, close_p) + abs(math.sin(step * 1.3)) * 195.0 + 40.0, 1)
        low_p = round(min(open_p, close_p) - abs(math.cos(step * 1.1)) * 205.0 - 40.0, 1)
        vol = round(180.0 + abs(close_p - open_p) * 1.35 + (380.0 if 11.0 <= h <= 17.0 else 60.0), 1)
        bars.append({"ts": ts, "o": open_p, "h": high_p, "l": low_p, "c": close_p, "v": vol})
        prev_c = close_p
    return bars


def _http_get_json(url: str, proxy: Optional[str], timeout: float = 8.0) -> Any:
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": "QuantumOrb-GEX-HUD/5.0", "Accept": "application/json"})
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_okx_15m_candles(proxy: Optional[str], limit: int = 240) -> List[Dict[str, float]]:
    url = f"https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar=15m&limit={limit}"
    payload = _http_get_json(url, proxy=proxy)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not rows:
        raise RuntimeError(f"OKX candles empty: {payload}")
    candles = [{"ts": int(r[0]), "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4]), "v": float(r[5])} for r in rows]
    candles.sort(key=lambda x: x["ts"])
    return candles


def _fetch_deribit_15m_candles(proxy: Optional[str], start_ms: int, end_ms: int) -> List[Dict[str, float]]:
    url = (
        "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
        f"?instrument_name=BTC-PERPETUAL&start_timestamp={start_ms}&end_timestamp={end_ms}&resolution=15"
    )
    payload = _http_get_json(url, proxy=proxy)
    res = payload.get("result") if isinstance(payload, dict) else None
    if not res or res.get("status") != "ok" or not res.get("ticks"):
        raise RuntimeError(f"Deribit chart data invalid: {payload}")
    candles = [
        {"ts": int(ts), "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v or 0.0)}
        for ts, o, h, l, c, v in zip(res["ticks"], res["open"], res["high"], res["low"], res["close"], res.get("volume", [0.0] * len(res["ticks"])))
    ]
    candles.sort(key=lambda x: x["ts"])
    return candles


def _evaluate_alert_state(pred: PredictionOutput, candle_1h_close: float, oi_1h_change_pct: float) -> Dict[str, Any]:
    inv_msg = ExtremeBoundaryEngine.check_invalidation(pred=pred, candle_1h_close=candle_1h_close, oi_1h_change_pct=oi_1h_change_pct)
    spot = pred.spot_price
    if inv_msg:
        return {"level": "CRITICAL", "title": "SHIELD BREACHED · 结构熔断触发切换趋势模式", "message": inv_msg, "candle_1h_close": candle_1h_close, "oi_1h_change_pct": oi_1h_change_pct}
    if spot >= pred.wick_ceiling_zone[0]:
        return {"level": "WARNING", "title": "UPPER SHIELD ALERT · 逼近极端轧空上限", "message": f"现价 ${spot:,.1f} 已进入 Wick Ceiling 警戒带，1H 收破 ${pred.invalidation_bull_trigger:,.1f} 且 OI>+1% 将触发向上熔断。", "candle_1h_close": candle_1h_close, "oi_1h_change_pct": oi_1h_change_pct}
    if spot <= pred.wick_floor_zone[1]:
        return {"level": "WARNING", "title": "LOWER SHIELD ALERT · 逼近极端插针底线", "message": f"现价 ${spot:,.1f} 已进入 Wick Floor 警戒带，1H 收破 ${pred.invalidation_bear_trigger:,.1f} 且 OI>+1% 将触发向下熔断。", "candle_1h_close": candle_1h_close, "oi_1h_change_pct": oi_1h_change_pct}
    return {"level": "NORMAL", "title": "SHIELD INTEGRITY NOMINAL · 四象限磁吸模型有效", "message": f"1H 收盘 ${candle_1h_close:,.1f} (1H OI {oi_1h_change_pct:+.2f}%) 处于装甲边界内，执行 24H 磁吸区间策略。", "candle_1h_close": candle_1h_close, "oi_1h_change_pct": oi_1h_change_pct}


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
        self.manual_etf_netflow_usd: Optional[float] = None
        self.manual_polymarket_bias: Optional[float] = None
        self.snapshot: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.last_updated_epoch: float = 0.0
        self.next_refresh_epoch: float = time.time() + REFRESH_INTERVAL_SEC
        persisted = _load_persisted_session()
        self.forecast_session: Dict[str, Any] = persisted if persisted else _initial_locked_session()
        self._candle_cache_payload: Optional[Dict[str, Any]] = None
        self._candle_cache_epoch: float = 0.0

    def invalidate_candle_cache(self) -> None:
        with self._candle_lock:
            self._candle_cache_payload = None
            self._candle_cache_epoch = 0.0

    def _build_auto_calendar_session(self, payload: Dict[str, Any], now_ms: int, candles: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        if now_ms < BASELINE_END_MS:
            return _initial_locked_session()
        now_est = datetime.fromtimestamp(now_ms / 1000.0, tz=_EST_TZ)
        day_start_dt = now_est.replace(hour=0, minute=0, second=0, microsecond=0)
        start_ms = int(day_start_dt.timestamp() * 1000)
        end_ms = int((day_start_dt + timedelta(days=1)).timestamp() * 1000)
        anchor_open: Optional[float] = None
        if candles:
            nearest = min(candles, key=lambda c: abs(int(c["ts"]) - start_ms))
            if abs(int(nearest["ts"]) - start_ms) <= 30 * 60 * 1000:
                anchor_open = float(nearest["o"])
        return _generate_24h_session(payload, start_ms, end_ms, f"auto-{day_start_dt.strftime('%Y%m%d')}", False, anchor_open)

    def _check_and_rollover_session(self, payload: Dict[str, Any], now_ms: int, candles: Optional[List[Dict[str, float]]] = None) -> None:
        with self._lock:
            expired = now_ms >= int(self.forecast_session.get("endTs", 0))
        if not expired:
            return
        next_sess = self._build_auto_calendar_session(payload=payload, now_ms=now_ms, candles=candles)
        with self._lock:
            if now_ms >= int(self.forecast_session.get("endTs", 0)):
                self.forecast_session = next_sess
                _save_persisted_session(next_sess)
        self.invalidate_candle_cache()

    def manual_reanchor_now(self, now_ms: Optional[int] = None) -> Dict[str, Any]:
        cur_ms = now_ms or int(time.time() * 1000)
        start_ms = (cur_ms // (15 * 60 * 1000)) * (15 * 60 * 1000)
        end_ms = start_ms + 24 * 3600 * 1000
        with self._lock:
            snap = self.snapshot
        if not snap:
            return self.get_state_json()
        new_sess = _generate_24h_session(snap, start_ms, end_ms, f"manual-{start_ms}", True)
        with self._lock:
            self.forecast_session = new_sess
            _save_persisted_session(new_sess)
        self.invalidate_candle_cache()
        return self.get_state_json()

    def restore_auto_schedule(self, now_ms: Optional[int] = None) -> Dict[str, Any]:
        cur_ms = now_ms or int(time.time() * 1000)
        with self._lock:
            snap = self.snapshot
        if not snap:
            auto_sess = _initial_locked_session()
        else:
            candles_resp = self.get_candles_json()
            c_list = candles_resp.get("candles") if isinstance(candles_resp, dict) else None
            auto_sess = self._build_auto_calendar_session(payload=snap, now_ms=cur_ms, candles=c_list)
        with self._lock:
            self.forecast_session = auto_sess
            _save_persisted_session(auto_sess)
        self.invalidate_candle_cache()
        return self.get_state_json()

    def get_candles_json(self, start_ms: Optional[int] = None) -> Dict[str, Any]:
        now_ep = time.time()
        with self._lock:
            mode, exchange_source, proxy = self.mode, self.exchange_source, self.proxy
            snap, snap_ep, sess = self.snapshot, self.last_updated_epoch, dict(self.forecast_session)
        if mode == "DEMO":
            return {"ok": True, "source": "DEMO_SIMULATOR", "candles": _build_demo_candles(sess)}
        with self._candle_lock:
            if self._candle_cache_payload is not None and (now_ep - self._candle_cache_epoch) < 45.0 and start_ms is None:
                return self._candle_cache_payload

        now_ms = int(now_ep * 1000)
        query_start_ms = min(start_ms or int(sess.get("startTs", BASELINE_START_MS)), now_ms - 48 * 3600 * 1000)
        candles: List[Dict[str, float]] = []
        used_source, err_notes = "OKX", []
        order = ["deribit", "okx"] if (exchange_source == "deribit" or (exchange_source == "auto" and GLOBAL_EXCHANGE_CIRCUIT.remaining_cooldown_sec > 0)) else ["okx", "deribit"]
        for src in order:
            try:
                if src == "okx":
                    candles = _fetch_okx_15m_candles(proxy=proxy, limit=240)
                    used_source = "OKX (BTC-USDT-SWAP)"
                else:
                    candles = _fetch_deribit_15m_candles(proxy=proxy, start_ms=query_start_ms, end_ms=now_ms)
                    used_source = "DERIBIT (BTC-PERPETUAL)"
                if candles:
                    break
            except Exception as exc:
                err_notes.append(f"{src.upper()}: {exc}")

        if candles and snap and snap.get("prediction") and snap_ep > 0:
            spot_now = float(snap["prediction"]["spot_price"])
            if int(snap_ep * 1000) >= candles[-1]["ts"]:
                candles[-1]["c"] = spot_now
                candles[-1]["h"] = max(candles[-1]["h"], spot_now)
                candles[-1]["l"] = min(candles[-1]["l"], spot_now)

        result = {"ok": len(candles) > 0, "source": used_source if candles else "UNAVAILABLE", "candles": candles, "error": " | ".join(err_notes) if not candles and err_notes else None}
        if candles and start_ms is None:
            with self._candle_lock:
                self._candle_cache_payload = result
                self._candle_cache_epoch = now_ep
        return result

    async def _fetch_payload_async(self, mode: str, exchange_source: str, proxy: Optional[str], etf_override_usd: Optional[float], poly_bias: Optional[float]) -> Dict[str, Any]:
        if mode == "DEMO":
            adapter = DemoSnapshotAdapter()
            pipeline = BTC24HPredictorPipeline(adapter=adapter)
            pred = await pipeline.run_once()
            tech, deriv, senti = await adapter.fetch_technical(), await adapter.fetch_derivatives(), await adapter.fetch_sentiment()
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
                "config": {"exchange_source": exchange_source, "etf_override_m": round(etf_override_usd / 1e6, 2) if etf_override_usd is not None else None, "polymarket_bias": poly_bias},
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
            gex, etf = adapter.last_gex_profile, adapter.latest_etf
            alert = _evaluate_alert_state(pred, candle_1h_close=adapter.last_1h_close, oi_1h_change_pct=adapter.oi_1h_change_pct)
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
                "config": {"exchange_source": exchange_source, "etf_override_m": round(etf_override_usd / 1e6, 2) if etf_override_usd is not None else None, "polymarket_bias": poly_bias},
            }

    def refresh_now(self, force_etf_refresh: bool = False) -> Tuple[bool, Optional[str]]:
        with self._lock:
            if self._refreshing:
                return False, "已有刷新任务正在执行中"
            self._refreshing = True
            mode, exchange_source, proxy = self.mode, self.exchange_source, self.proxy
            etf_override, poly_bias = self.manual_etf_netflow_usd, self.manual_polymarket_bias
        if force_etf_refresh:
            GLOBAL_ETF_CACHE.clear()
        try:
            payload = asyncio.run(self._fetch_payload_async(mode, exchange_source, proxy, etf_override, poly_bias))
            now_ep = time.time()
            now_ms = int(now_ep * 1000)
            with self._lock:
                self.snapshot = payload
                self.last_error = None
                self.last_updated_epoch = now_ep
                self.next_refresh_epoch = now_ep + REFRESH_INTERVAL_SEC
                need_rollover = now_ms >= int(self.forecast_session.get("endTs", 0))
            self.invalidate_candle_cache()
            if need_rollover:
                candles_resp = self.get_candles_json()
                c_list = candles_resp.get("candles") if isinstance(candles_resp, dict) else None
                self._check_and_rollover_session(payload=payload, now_ms=now_ms, candles=c_list)
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

    def update_config(self, mode: Optional[str] = None, exchange_source: Optional[str] = None, etf_netflow_m: Optional[Any] = None, polymarket_bias: Optional[Any] = None) -> None:
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
        now_ep = time.time()
        now_ms = int(now_ep * 1000)
        with self._lock:
            snap = self.snapshot
            need_rollover = snap is not None and now_ms >= int(self.forecast_session.get("endTs", 0))
        if need_rollover and snap is not None:
            self._check_and_rollover_session(payload=snap, now_ms=now_ms)
        with self._lock:
            return {
                "ok": self.snapshot is not None,
                "mode": self.mode,
                "exchange_source": self.exchange_source,
                "primary_cooldown_sec": GLOBAL_EXCHANGE_CIRCUIT.remaining_cooldown_sec,
                "etf_last_error": GLOBAL_ETF_CACHE.last_error,
                "refreshing": self._refreshing,
                "refresh_interval_sec": REFRESH_INTERVAL_SEC,
                "seconds_to_next_refresh": max(0, int(self.next_refresh_epoch - now_ep)),
                "last_updated_utc": datetime.fromtimestamp(self.last_updated_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if self.last_updated_epoch > 0 else "--",
                "error": self.last_error,
                "data": self.snapshot,
                "forecast_session": self.forecast_session,
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
            threading.Thread(target=_background_poller, name="btc24h-poller", daemon=True).start()
            _poller_started = True


# =============================================================================
# 3. Flask Web 路由
# =============================================================================

app = Flask(__name__)


@app.route("/")
def index() -> Response:
    ensure_poller_started()
    return Response(HTML_TEMPLATE, mimetype="text/html; charset=utf-8")


@app.route("/healthz", methods=["GET"])
def healthz():
    snap = store.snapshot or {}
    return jsonify({
        "status": "ok",
        "ui_version": "5.0-quantum-orb",
        "mode": store.mode,
        "active_exchange": snap.get("active_exchange", "INITIALIZING"),
        "etf_source": (snap.get("market_meta") or {}).get("etf_source", "--"),
        "forecast_window": store.forecast_session.get("windowLabel", "--"),
    })


@app.route("/api/snapshot", methods=["GET"])
def api_snapshot():
    ensure_poller_started()
    return jsonify(store.get_state_json())


@app.route("/api/candles", methods=["GET"])
def api_candles():
    ensure_poller_started()
    start_arg = request.args.get("start_ms", "").strip()
    return jsonify(store.get_candles_json(start_ms=int(start_arg) if start_arg.isdigit() else None))


@app.route("/api/reanchor", methods=["POST"])
def api_reanchor():
    ensure_poller_started()
    if store.snapshot is None:
        store.refresh_now()
    return jsonify(store.manual_reanchor_now())


@app.route("/api/reanchor/auto", methods=["POST"])
def api_reanchor_auto():
    ensure_poller_started()
    if store.snapshot is None:
        store.refresh_now()
    return jsonify(store.restore_auto_schedule())


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
# 4. QuantumOrb V5.0 战术指挥舱前端 (ORB + RADAR + SHIELD + SCANNER + CARTOGRAPHER + GUARD)
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QUANTUM·GEX // BTC 24H 战术指挥舱 V5.0</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #050811;
      --panel: rgba(11, 17, 30, 0.86);
      --panel-2: rgba(17, 25, 43, 0.92);
      --border: #1b2a47;
      --cyan: #00f0ff;
      --green: #00ff9d;
      --red: #ff2a5f;
      --amber: #ffb800;
      --purple: #b47cff;
      --text: #e8f1ff;
      --muted: #7b90b2;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      background-image:
        radial-gradient(circle at 50% 18%, rgba(0, 240, 255, 0.08), transparent 48%),
        linear-gradient(rgba(27, 42, 71, 0.22) 1px, transparent 1px),
        linear-gradient(90deg, rgba(27, 42, 71, 0.22) 1px, transparent 1px);
      background-size: 100% 100%, 36px 36px, 36px 36px;
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      padding: 14px 18px 28px;
      line-height: 1.42;
      overflow-x: hidden;
    }
    .mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .container { max-width: 1520px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }

    /* 战术切角装甲卡片 */
    .hud-card {
      position: relative;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 13px 15px;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45), inset 0 0 24px rgba(0, 240, 255, 0.025);
      backdrop-filter: blur(8px);
    }
    .hud-card::before, .hud-card::after {
      content: ""; position: absolute; width: 10px; height: 10px; pointer-events: none;
    }
    .hud-card::before { top: -1px; left: -1px; border-top: 2px solid var(--cyan); border-left: 2px solid var(--cyan); border-top-left-radius: 8px; }
    .hud-card::after { bottom: -1px; right: -1px; border-bottom: 2px solid var(--cyan); border-right: 2px solid var(--cyan); border-bottom-right-radius: 8px; }

    .mod-head {
      display: flex; justify-content: space-between; align-items: center; gap: 8px;
      padding-bottom: 9px; margin-bottom: 10px; border-bottom: 1px solid rgba(27, 42, 71, 0.85); flex-wrap: wrap;
    }
    .mod-tag {
      display: inline-flex; align-items: center; gap: 7px; font-size: 13px; font-weight: 700;
      letter-spacing: 0.6px; color: #fff; text-transform: uppercase;
    }
    .mod-badge {
      font-size: 10px; padding: 2px 6px; border-radius: 4px; background: rgba(0, 240, 255, 0.14);
      border: 1px solid rgba(0, 240, 255, 0.45); color: var(--cyan); font-weight: 700;
    }
    .mod-sub { font-size: 11.5px; color: var(--muted); }

    /* 顶栏 */
    .topbar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }
    .brand-wrap { display: flex; align-items: center; gap: 12px; }
    .pulse-dot {
      width: 11px; height: 11px; border-radius: 50%; background: var(--green);
      box-shadow: 0 0 12px var(--green); animation: blink 1.4s infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.45; transform: scale(0.82); } }
    .brand-title { font-size: 17px; font-weight: 800; letter-spacing: 0.8px; background: linear-gradient(90deg, #fff, var(--cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .brand-sub { font-size: 11.5px; color: var(--muted); margin-top: 2px; }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .pill {
      padding: 4px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 700;
      border: 1px solid var(--border); background: var(--panel-2); color: var(--text);
    }
    .pill.live { border-color: rgba(0, 255, 157, 0.55); color: var(--green); }
    .pill.demo { border-color: rgba(255, 184, 0, 0.55); color: var(--amber); }
    .pill.alt  { border-color: rgba(0, 240, 255, 0.55); color: var(--cyan); }
    button {
      background: linear-gradient(135deg, #0284c7, #00b4d8); color: #fff; border: 1px solid rgba(0,240,255,0.45);
      border-radius: 7px; padding: 6px 13px; font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.15s;
    }
    button:hover { filter: brightness(1.12); box-shadow: 0 0 12px rgba(0, 240, 255, 0.35); }
    button.secondary { background: var(--panel-2); border: 1px solid var(--border); color: var(--text); box-shadow: none; }
    button.secondary.active { border-color: var(--cyan); color: var(--cyan); background: rgba(0, 240, 255, 0.13); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    .info-bar, .error-bar { display: none; border-radius: 8px; padding: 8px 13px; font-size: 12px; }
    .info-bar { background: rgba(0, 240, 255, 0.1); border: 1px solid rgba(0, 240, 255, 0.4); color: #bae6fd; }
    .error-bar { background: rgba(255, 42, 95, 0.14); border: 1px solid rgba(255, 42, 95, 0.55); color: #fecdd3; }

    /* SHIELD 护盾装甲条 */
    .shield-grid {
      display: grid; grid-template-columns: 1fr 1.45fr 1fr; gap: 14px; align-items: center;
    }
    @media (max-width: 980px) { .shield-grid { grid-template-columns: 1fr; } }
    .shield-wing { background: rgba(7, 11, 20, 0.72); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; }
    .shield-row { display: flex; justify-content: space-between; font-size: 11.5px; margin-bottom: 5px; }
    .shield-track { height: 7px; background: #0c1424; border-radius: 99px; overflow: hidden; border: 1px solid #1e293b; }
    .shield-fill { height: 100%; width: 50%; transition: width 0.35s ease; }
    .shield-center { text-align: center; padding: 4px 8px; }
    .shield-center.NORMAL .sh-title { color: var(--green); }
    .shield-center.WARNING .sh-title { color: var(--amber); }
    .shield-center.CRITICAL .sh-title { color: var(--red); text-shadow: 0 0 12px var(--red); }
    .sh-title { font-size: 13.5px; font-weight: 800; letter-spacing: 0.5px; }
    .sh-msg { font-size: 12px; color: var(--muted); margin-top: 3px; }

    /* 核心战术三舱: RADAR | ORB | SCANNER+GUARD */
    .tactical-deck { display: grid; grid-template-columns: 1.05fr 1.15fr 1.1fr; gap: 14px; }
    @media (max-width: 1180px) { .tactical-deck { grid-template-columns: 1fr; } }

    /* 1) RADAR 极值磁吸声呐 */
    .radar-wrap {
      position: relative; height: 390px; background: #060a14; border: 1px solid var(--border);
      border-radius: 8px; overflow: hidden;
    }
    .radar-sweep {
      position: absolute; left: 0; right: 0; height: 56px; pointer-events: none; z-index: 2;
      background: linear-gradient(180deg, transparent, rgba(0, 240, 255, 0.11), rgba(0, 240, 255, 0.28), transparent);
      animation: radarScan 4.2s linear infinite;
    }
    @keyframes radarScan { 0% { top: -60px; } 100% { top: 100%; } }
    .zone-box {
      position: absolute; left: 8px; right: 8px; border-radius: 5px; z-index: 3;
      display: flex; justify-content: space-between; align-items: center;
      padding: 0 10px; font-size: 11.5px; font-weight: 700; pointer-events: none;
    }
    .zone-wick-ceil { background: rgba(255, 42, 95, 0.15); border: 1px dashed rgba(255, 42, 95, 0.7); color: #fda4af; }
    .zone-core-high { background: rgba(0, 255, 157, 0.14); border: 1px solid rgba(0, 255, 157, 0.65); color: #6ee7b7; }
    .zone-core-low  { background: rgba(0, 240, 255, 0.14); border: 1px solid rgba(0, 240, 255, 0.65); color: #7dd3fc; }
    .zone-wick-floor{ background: rgba(255, 184, 0, 0.15); border: 1px dashed rgba(255, 184, 0, 0.7); color: #fde047; }
    .price-line {
      position: absolute; left: 0; right: 0; height: 0; z-index: 5;
      display: flex; justify-content: flex-end; align-items: center; pointer-events: none; transition: top 0.22s ease-out;
    }
    .price-tag {
      transform: translateY(-50%); margin-right: 8px; padding: 2px 7px; border-radius: 4px;
      font-size: 11px; font-weight: 700; white-space: nowrap;
    }
    .line-spot { border-top: 2px solid #fff; z-index: 10; box-shadow: 0 0 10px #fff; }
    .line-spot .price-tag { background: #fff; color: #050811; }
    .line-flip { border-top: 1.5px dashed var(--purple); justify-content: flex-start; }
    .line-flip .price-tag { background: rgba(180,124,255,0.22); border: 1px solid var(--purple); color: #e9d5ff; margin-left: 8px; }
    .line-avwap { border-top: 1.5px dashed var(--amber); justify-content: center; }
    .line-avwap .price-tag { background: rgba(255,184,0,0.2); border: 1px solid var(--amber); color: #fef08a; }
    .line-inv { border-top: 1.5px solid var(--red); justify-content: flex-start; }
    .line-inv .price-tag { background: rgba(255,42,95,0.86); color: #fff; margin-left: 8px; font-size: 10.5px; }
    .radar-foot { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--muted); margin-top: 8px; }

    /* 2) ORB 中央量子核心引擎 */
    .orb-stage {
      position: relative; height: 270px; display: flex; justify-content: center; align-items: center;
      background: radial-gradient(circle, rgba(0,240,255,0.07) 0%, rgba(5,8,17,0.92) 72%);
      border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    }
    #orbCanvas { width: 100%; height: 100%; display: block; }
    .orb-overlay-top {
      position: absolute; top: 10px; left: 12px; right: 12px; display: flex; justify-content: space-between; align-items: center; pointer-events: none;
    }
    .verdict-chip {
      padding: 3px 9px; border-radius: 4px; font-size: 11px; font-weight: 800; letter-spacing: 0.6px;
      border: 1px solid var(--cyan); background: rgba(0, 240, 255, 0.16); color: var(--cyan);
    }
    .orb-overlay-bottom {
      position: absolute; bottom: 10px; left: 12px; right: 12px; display: flex; justify-content: space-between; align-items: flex-end; pointer-events: none;
    }
    .spot-hero { font-size: 27px; font-weight: 800; letter-spacing: -0.5px; text-shadow: 0 0 16px rgba(0,240,255,0.45); transition: color 0.18s; }
    .tick-up { color: var(--green) !important; text-shadow: 0 0 18px rgba(0,255,157,0.75) !important; }
    .tick-down { color: var(--red) !important; text-shadow: 0 0 18px rgba(255,42,95,0.75) !important; }
    .orb-kpi-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 10px; }
    .mini-box { background: rgba(8, 13, 24, 0.85); border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px; }
    .mini-lbl { font-size: 10.5px; color: var(--muted); text-transform: uppercase; }
    .mini-val { font-size: 13.5px; font-weight: 700; margin-top: 2px; }

    /* 3) SCANNER & GUARD 右侧大单雷达与哨兵 */
    .guard-strip { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 10px; }
    .ob-bar-wrap { background: rgba(8, 13, 24, 0.85); border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px; margin-bottom: 10px; }
    .ob-labels { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px; }
    .ob-track { height: 7px; background: rgba(255, 42, 95, 0.55); border-radius: 99px; overflow: hidden; }
    .ob-bid-fill { height: 100%; width: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); transition: width 0.2s ease; }
    .scanner-terminal {
      height: 224px; background: #04070e; border: 1px solid #16223b; border-radius: 7px;
      padding: 8px 10px; overflow-y: auto; font-size: 11.5px; display: flex; flex-direction: column; gap: 5px;
    }
    .scan-line { display: flex; justify-content: space-between; gap: 8px; border-bottom: 1px dashed rgba(27,42,71,0.45); padding-bottom: 3px; }
    .scan-buy { color: var(--green); }
    .scan-sell { color: var(--red); }
    .scan-sys { color: var(--cyan); }

    /* CARTOGRAPHER 24H 全息轨迹对比 */
    .compare-toolbar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 10px; }
    .compare-metrics { display: flex; flex-wrap: wrap; gap: 8px; }
    .cmp-stat { background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 12px; }
    .cmp-stat span { color: var(--muted); margin-right: 5px; }
    .compare-legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 11.5px; color: var(--muted); }
    .legend-item { display: inline-flex; align-items: center; gap: 5px; }
    .legend-swatch { width: 18px; height: 0; display: inline-block; }
    .compare-canvas-wrap {
      position: relative; height: 410px; width: 100%; background: #060a14;
      border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    }
    #forecastCompareCanvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
    .compare-tooltip {
      position: absolute; display: none; pointer-events: none; z-index: 20;
      background: rgba(8, 13, 25, 0.96); border: 1px solid var(--cyan); border-radius: 7px;
      padding: 8px 11px; font-size: 11.5px; line-height: 1.55; color: var(--text); white-space: nowrap;
    }

    /* 底部双栏: GEX 光柱 + 配置与权重诊断 */
    .bottom-deck { display: grid; grid-template-columns: 1.35fr 1fr; gap: 14px; }
    @media (max-width: 1020px) { .bottom-deck { grid-template-columns: 1fr; } }
    .chart-box { position: relative; height: 300px; width: 100%; }
    .form-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end; margin-bottom: 14px; }
    .field { display: flex; flex-direction: column; gap: 4px; }
    .field label { font-size: 11px; color: var(--muted); }
    .field input, .field select {
      background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
      border-radius: 6px; padding: 6px 9px; font-size: 12.5px; width: 156px;
    }
    .diag-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px 12px; font-size: 12px; background: rgba(6,10,20,0.75); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
    .diag-item span:first-child { color: var(--muted); margin-right: 5px; }
  </style>
</head>
<body>
  <div class="container">
    <!-- 顶栏战术状态 -->
    <div class="hud-card topbar">
      <div class="brand-wrap">
        <div class="pulse-dot" id="wsPulseDot"></div>
        <div>
          <div class="brand-title mono">QUANTUM·GEX // BTC 24H TACTICAL COMMAND V5.0</div>
          <div class="brand-sub mono">快照更新: <span id="lastUpdated">--</span> · 周期重算: <span id="countdown">--</span>s · 实时流: <span id="wsHzPill" style="color:var(--cyan)">CONNECTING...</span></div>
        </div>
      </div>
      <div class="controls">
        <span id="etfPill" class="pill mono">ETF: --</span>
        <span id="exchangePill" class="pill mono">SRC: --</span>
        <span id="modePill" class="pill mono">--</span>
        <button id="btnToggleMode" class="secondary" onclick="toggleMode()">切换 DEMO / LIVE</button>
        <button id="btnRefresh" onclick="manualRefresh()">立即同步期权链</button>
      </div>
    </div>

    <div id="failoverBar" class="info-bar mono"></div>
    <div id="errorBar" class="error-bar mono"></div>

    <!-- 模块 1: SHIELD 1H 结构熔断双向装甲 -->
    <div class="hud-card">
      <div class="shield-grid">
        <div class="shield-wing mono">
          <div class="shield-row">
            <span style="color:var(--amber)">🛡️ LOWER SHIELD (向下熔断缓冲)</span>
            <strong id="bearShieldPct" style="color:var(--green)">--%</strong>
          </div>
          <div class="shield-track"><div id="bearShieldFill" class="shield-fill" style="background:linear-gradient(90deg,var(--red),var(--green));"></div></div>
          <div class="shield-row" style="margin-top:5px;margin-bottom:0;color:var(--muted);">
            <span>触发线: <strong id="invBear" style="color:var(--red)">--</strong></span>
            <span>条件: 1H收破 + OI>1%</span>
          </div>
        </div>

        <div id="shieldBanner" class="shield-center NORMAL">
          <div class="sh-title mono" id="alertTitle">INITIALIZING QUANTUM TACTICAL MATRIX...</div>
          <div class="sh-msg" id="alertMsg">正在建立 OKX / Deribit 毫秒级流与美国现货 ETF 遥测通道...</div>
        </div>

        <div class="shield-wing mono">
          <div class="shield-row">
            <span style="color:var(--cyan)">🛡️ UPPER SHIELD (向上熔断缓冲)</span>
            <strong id="bullShieldPct" style="color:var(--green)">--%</strong>
          </div>
          <div class="shield-track"><div id="bullShieldFill" class="shield-fill" style="background:linear-gradient(90deg,var(--green),var(--cyan));"></div></div>
          <div class="shield-row" style="margin-top:5px;margin-bottom:0;color:var(--muted);">
            <span>触发线: <strong id="invBull" style="color:var(--red)">--</strong></span>
            <span>条件: 1H收破 + OI>1%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 核心战术三联舱: RADAR | ORB | SCANNER & GUARD -->
    <div class="tactical-deck">
      <!-- 模块 2: RADAR -->
      <div class="hud-card">
        <div class="mod-head">
          <div class="mod-tag"><span class="mod-badge">RADAR</span> 24H 极值磁吸声呐舱</div>
          <div class="mod-sub mono">100ms 实时游标锁定</div>
        </div>
        <div class="radar-wrap">
          <div class="radar-sweep"></div>
          <div id="radarLadder" style="position:relative;width:100%;height:100%;"></div>
        </div>
        <div class="radar-foot mono">
          <span>距 Flip: <strong id="radarFlipDist" style="color:var(--purple)">--</strong></span>
          <span>距 AVWAP: <strong id="radarAvwapDist" style="color:var(--amber)">--</strong></span>
        </div>
      </div>

      <!-- 模块 3: ORB 核心引力引擎 -->
      <div class="hud-card">
        <div class="mod-head">
          <div class="mod-tag"><span class="mod-badge">ORB</span> 量子 Gamma 核心引擎</div>
          <div class="mod-sub mono" id="kpiWeekend">周末薄盘: --</div>
        </div>
        <div class="orb-stage">
          <canvas id="orbCanvas"></canvas>
          <div class="orb-overlay-top mono">
            <div>
              <div style="font-size:10.5px;color:var(--muted);">RESONANCE SCORE</div>
              <div style="font-size:22px;font-weight:800;color:var(--cyan);" id="orbScoreVal">-- / 100</div>
            </div>
            <span class="verdict-chip" id="orbVerdictBadge">CALIBRATING</span>
          </div>
          <div class="orb-overlay-bottom mono">
            <div>
              <div style="font-size:10.5px;color:var(--muted);">BTC REALTIME SPOT</div>
              <div class="spot-hero" id="kpiSpot">--</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:11px;color:var(--cyan);font-weight:700;" id="kpiRegime">--</div>
              <div style="font-size:11px;color:var(--muted);" id="kpiEm">EM: -- | DVOL: --</div>
            </div>
          </div>
        </div>
        <div class="orb-kpi-grid mono">
          <div class="mini-box">
            <div class="mini-lbl">Deribit Net GEX (1% Move)</div>
            <div class="mini-val" id="kpiGex">--</div>
          </div>
          <div class="mini-box">
            <div class="mini-lbl">Gamma Flip / Put·Call Wall</div>
            <div class="mini-val" id="kpiWalls" style="font-size:11.5px;">--</div>
          </div>
        </div>
      </div>

      <!-- 模块 4 & 5: GUARD + SCANNER -->
      <div class="hud-card">
        <div class="mod-head">
          <div class="mod-tag"><span class="mod-badge">SCANNER · GUARD</span> 主力扫单 & 机构哨兵</div>
          <span class="pill alt mono" id="crowdTrapBadge" style="font-size:10.5px;padding:2px 8px;">杠杆态: --</span>
        </div>

        <div class="guard-strip mono">
          <div class="mini-box">
            <div class="mini-lbl">美国现货 ETF 净流量</div>
            <div class="mini-val" id="kpiEtf">--</div>
            <div style="font-size:10.5px;color:var(--muted);margin-top:2px;" id="kpiEtfMeta">--</div>
          </div>
          <div class="mini-box">
            <div class="mini-lbl">反身性情绪 / 费率 & F&G</div>
            <div class="mini-val" id="kpiSent">--</div>
            <div style="font-size:10.5px;color:var(--muted);margin-top:2px;" id="kpiFund">--</div>
          </div>
        </div>

        <div class="ob-bar-wrap mono">
          <div class="ob-labels">
            <span style="color:var(--green)">五档买墙 BID <strong id="obBidPct">50%</strong></span>
            <span style="color:var(--muted)">ORDERBOOK DEPTH</span>
            <span style="color:var(--red)"><strong id="obAskPct">50%</strong> ASK 五档卖墙</span>
          </div>
          <div class="ob-track"><div id="obBidBar" class="ob-bid-fill"></div></div>
        </div>

        <div class="scanner-terminal mono" id="scannerLog">
          <div class="scan-line scan-sys"><span>[SYSTEM] QuantumOrb 实时大单与关键位雷达已就绪...</span><span>READY</span></div>
        </div>
      </div>
    </div>

    <!-- 模块 6: CARTOGRAPHER 全息 24H 预测路径 vs 实时行情对比 -->
    <div class="hud-card">
      <div class="mod-head">
        <div>
          <div class="mod-tag"><span class="mod-badge">CARTOGRAPHER</span> 24H 全息预测路径 vs 15m 实盘轨迹追踪</div>
          <div class="mod-sub mono" id="compareWindowMeta" style="margin-top:3px;">窗口: 初始化中...</div>
        </div>
        <div class="controls">
          <button id="btnBandLocked" class="secondary active" onclick="setBandMode('locked')">锁定周期磁吸带</button>
          <button id="btnBandLive" class="secondary" onclick="setBandMode('live')">叠加实时磁吸带</button>
          <button id="btnReanchor" class="secondary" onclick="reanchorFromCurrentState()">按当前盘面重锚24H</button>
          <button id="btnResetScript" class="secondary" style="display:none;" onclick="resetToAutoSchedule()">恢复自动日历窗口</button>
        </div>
      </div>

      <div class="compare-toolbar">
        <div class="compare-metrics mono">
          <div class="cmp-stat"><span>实时现价:</span><strong id="cmpLiveSpot" style="color:var(--cyan)">--</strong></div>
          <div class="cmp-stat"><span>同期预测价:</span><strong id="cmpPredSpot">--</strong></div>
          <div class="cmp-stat"><span>实时偏离度:</span><strong id="cmpDelta">--</strong></div>
          <div class="cmp-stat"><span>轨迹判定:</span><strong id="cmpStatus">--</strong></div>
        </div>
        <div class="compare-legend">
          <span class="legend-item"><i class="legend-swatch" style="border-top:2px dashed rgba(232,241,255,0.85);"></i> 24H 预测基准 (Forecast)</span>
          <span class="legend-item"><i class="legend-swatch" style="border-top:2.5px solid #00f0ff;"></i> 实盘轨迹 + 15m 影线包络 (Live)</span>
          <span class="legend-item"><i class="legend-swatch" style="border-top:1.5px dashed #b47cff;"></i> Gamma Flip</span>
          <span class="legend-item"><i class="legend-swatch" style="border-top:1.5px dashed #ffb800;"></i> AVWAP</span>
        </div>
      </div>

      <div class="compare-canvas-wrap">
        <canvas id="forecastCompareCanvas"></canvas>
        <div id="compareTooltip" class="compare-tooltip mono"></div>
      </div>
    </div>

    <!-- 底部舱: Deribit 行权价 GEX 能量分布 + 路由控制与遥测 -->
    <div class="bottom-deck">
      <div class="hud-card">
        <div class="mod-head">
          <div class="mod-tag"><span class="mod-badge">GEX MATRIX</span> Deribit 各行权价净 Gamma 能量柱 ($M / 1% Move)</div>
          <div class="mod-sub mono">绿=正GEX吸波阻力 · 红=负GEX助推加速</div>
        </div>
        <div class="chart-box"><canvas id="gexChart"></canvas></div>
      </div>

      <div class="hud-card">
        <div class="mod-head">
          <div class="mod-tag"><span class="mod-badge">CONTROL · TELEMETRY</span> 路由覆盖与权重诊断</div>
          <div class="mod-sub mono" id="diagContracts">活跃期权合约: --</div>
        </div>
        <div class="form-row">
          <div class="field">
            <label>永续主数据源</label>
            <select id="selectSource">
              <option value="auto">AUTO (OKX→Deribit)</option>
              <option value="okx">OKX (BTC-USDT-SWAP)</option>
              <option value="deribit">Deribit (BTC-PERP)</option>
            </select>
          </div>
          <div class="field">
            <label>ETF 净流入覆盖 ($M)</label>
            <input id="inputEtf" type="number" step="0.1" placeholder="留空全自动" />
          </div>
          <div class="field">
            <label>Polymarket 偏移 (-1~+1)</label>
            <input id="inputPoly" type="number" step="0.05" min="-1" max="1" placeholder="留空自动" />
          </div>
          <button onclick="applyConfig()">应用并重算</button>
        </div>
        <div class="diag-list mono" id="diagBox">
          <div class="diag-item"><span>合成权重:</span><strong>--</strong></div>
          <div class="diag-item"><span>费率信号:</span><strong>--</strong></div>
          <div class="diag-item"><span>AVWAP 偏离:</span><strong>--</strong></div>
          <div class="diag-item"><span>1H OI 变化:</span><strong>--</strong></div>
        </div>
      </div>
    </div>
  </div>

<script>
  let gexChart = null;
  let currentMode = "LIVE";
  let countdownSec = 300;
  let latestSnapshotData = null;
  let lastRealtimeSpot = null;
  let tickCountWindow = 0;
  let demoSimTimer = null;

  const fmtUSD = (v) => (v == null || isNaN(v)) ? "--" : "$" + Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 });

  // =========================================================================
  // 1. 🎯 ORB 60FPS 3D 粒子引力核心球引擎
  // =========================================================================
  const orbState = {
    netGexM: -25.0,
    dvolPct: 55.0,
    whaleShockwave: 0.0,
    rotY: 0,
    rotX: 0.28,
    points: [],
  };

  (function initOrbParticles() {
    const N = 150;
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < N; i++) {
      const y = 1 - (i / (N - 1)) * 2;
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = golden * i;
      orbState.points.push({ x: Math.cos(theta) * r, y, z: Math.sin(theta) * r, phase: Math.random() * Math.PI * 2 });
    }
  })();

  function triggerOrbPulse(intensity = 1.0) {
    orbState.whaleShockwave = Math.min(1.6, Math.max(orbState.whaleShockwave, intensity));
  }

  function animateOrb() {
    const canvas = document.getElementById("orbCanvas");
    if (canvas) {
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.floor(rect.width * dpr) || canvas.height !== Math.floor(rect.height * dpr)) {
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const W = rect.width, H = rect.height;
      const cx = W / 2, cy = H / 2;
      const baseR = Math.min(W, H) * 0.30;

      ctx.clearRect(0, 0, W, H);

      const negGamma = orbState.netGexM < 0;
      const primaryRGB = negGamma ? "255, 42, 95" : "0, 255, 157";
      const accentRGB = negGamma ? "180, 124, 255" : "0, 240, 255";
      const speed = 0.004 + Math.min(0.022, (orbState.dvolPct / 100) * 0.018);
      orbState.rotY += speed;
      const nowT = performance.now() * 0.003;

      // 中心辉光
      const grad = ctx.createRadialGradient(cx, cy, 4, cx, cy, baseR * 1.55);
      grad.addColorStop(0, `rgba(${primaryRGB}, 0.28)`);
      grad.addColorStop(0.55, `rgba(${accentRGB}, 0.08)`);
      grad.addColorStop(1, "rgba(5, 8, 17, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, baseR * 1.55, 0, Math.PI * 2);
      ctx.fill();

      // 外围战术反向旋转刻度环
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-orbState.rotY * 0.55);
      ctx.strokeStyle = `rgba(${accentRGB}, 0.32)`;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([8, 6, 2, 6]);
      ctx.beginPath();
      ctx.arc(0, 0, baseR * 1.26, 0, Math.PI * 2);
      ctx.stroke();

      ctx.rotate(orbState.rotY * 1.1);
      ctx.strokeStyle = `rgba(${primaryRGB}, 0.42)`;
      ctx.setLineDash([18, 14]);
      ctx.beginPath();
      ctx.arc(0, 0, baseR * 1.12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // 大单冲击波环
      if (orbState.whaleShockwave > 0.02) {
        const swR = baseR * (1.0 + (1.6 - orbState.whaleShockwave) * 0.65);
        ctx.strokeStyle = `rgba(0, 240, 255, ${Math.min(0.85, orbState.whaleShockwave * 0.6)})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, swR, 0, Math.PI * 2);
        ctx.stroke();
        orbState.whaleShockwave *= 0.93;
      }

      // 3D 粒子投影
      const cosY = Math.cos(orbState.rotY), sinY = Math.sin(orbState.rotY);
      const cosX = Math.cos(orbState.rotX), sinX = Math.sin(orbState.rotX);
      const breath = negGamma ? (1 + Math.sin(nowT * 1.7) * 0.065) : (1 + Math.sin(nowT * 0.8) * 0.025);

      const projected = orbState.points.map((pt) => {
        const pulse = 1 + Math.sin(nowT + pt.phase) * (negGamma ? 0.05 : 0.018);
        const px = pt.x * pulse, py = pt.y * pulse, pz = pt.z * pulse;
        const x1 = px * cosY - pz * sinY;
        const z1 = pz * cosY + px * sinY;
        const y2 = py * cosX - z1 * sinX;
        const z2 = z1 * cosX + py * sinX;
        const persp = 2.6 / (2.6 - z2 * 0.65);
        return { sx: cx + x1 * baseR * breath * persp, sy: cy + y2 * baseR * breath * persp, z: z2, persp };
      }).sort((a, b) => a.z - b.z);

      for (const p of projected) {
        const alpha = Math.max(0.14, Math.min(0.95, (p.z + 1.15) / 2.1));
        const rad = Math.max(1.1, 2.1 * p.persp);
        ctx.fillStyle = p.z > 0.25 ? `rgba(${primaryRGB}, ${alpha})` : `rgba(${accentRGB}, ${alpha * 0.8})`;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, rad, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    requestAnimationFrame(animateOrb);
  }

  // 计算 0-100 量子战术稳定性/共振评分
  function updateOrbScoreAndShields(spot) {
    if (!latestSnapshotData || !latestSnapshotData.prediction) return;
    const pred = latestSnapshotData.prediction;
    const meta = latestSnapshotData.market_meta || {};
    const bullTrig = pred.invalidation_bull_trigger;
    const bearTrig = pred.invalidation_bear_trigger;
    const span = Math.max(bullTrig - bearTrig, 1);

    const bearDistPct = Math.max(0, ((spot - bearTrig) / spot) * 100);
    const bullDistPct = Math.max(0, ((bullTrig - spot) / spot) * 100);
    const bearBarW = Math.max(4, Math.min(100, ((spot - bearTrig) / (span * 0.65)) * 100));
    const bullBarW = Math.max(4, Math.min(100, ((bullTrig - spot) / (span * 0.65)) * 100));

    document.getElementById("bearShieldPct").textContent = bearDistPct.toFixed(2) + "%";
    document.getElementById("bullShieldPct").textContent = bullDistPct.toFixed(2) + "%";
    document.getElementById("bearShieldFill").style.width = bearBarW + "%";
    document.getElementById("bullShieldFill").style.width = bullBarW + "%";

    // Radar 实时百分比距离
    if (meta.gamma_flip) {
      const fd = ((spot - meta.gamma_flip) / meta.gamma_flip) * 100;
      document.getElementById("radarFlipDist").textContent = `${fd >= 0 ? "+" : ""}${fd.toFixed(2)}%`;
    }
    if (meta.avwap) {
      const ad = ((spot - meta.avwap) / meta.avwap) * 100;
      document.getElementById("radarAvwapDist").textContent = `${ad >= 0 ? "+" : ""}${ad.toFixed(2)}%`;
    }

    // 0-100 综合裁决分
    const minMarginPct = Math.min(bearDistPct, bullDistPct);
    let score = 52 + Math.min(28, minMarginPct * 4.2);
    if ((meta.net_gex_m || 0) >= 0) score += 12;
    else score -= Math.min(16, Math.abs(meta.net_gex_m) * 0.22);
    if (Math.abs(meta.funding_rate_pct || 0) > 0.01) score -= 6;
    score = Math.round(Math.max(8, Math.min(98, score)));

    const scoreEl = document.getElementById("orbScoreVal");
    const badgeEl = document.getElementById("orbVerdictBadge");
    scoreEl.textContent = `${score} / 100`;

    if (spot >= bullTrig || spot <= bearTrig || minMarginPct < 0.8) {
      badgeEl.textContent = "DANGER · 临近熔断极值";
      badgeEl.style.borderColor = "var(--red)";
      badgeEl.style.color = "var(--red)";
      scoreEl.style.color = "var(--red)";
    } else if ((meta.net_gex_m || 0) < -15) {
      badgeEl.textContent = "VOLATILE · 负GEX助推态";
      badgeEl.style.borderColor = "var(--amber)";
      badgeEl.style.color = "var(--amber)";
      scoreEl.style.color = "var(--amber)";
    } else if (score >= 75) {
      badgeEl.textContent = "CLEAN · 稳态磁吸收敛";
      badgeEl.style.borderColor = "var(--green)";
      badgeEl.style.color = "var(--green)";
      scoreEl.style.color = "var(--green)";
    } else {
      badgeEl.textContent = "ACTIVE · 磁吸区间博弈";
      badgeEl.style.borderColor = "var(--cyan)";
      badgeEl.style.color = "var(--cyan)";
      scoreEl.style.color = "var(--cyan)";
    }
  }

  // =========================================================================
  // 2. 📊 SCANNER 实时战术扫描日志流
  // =========================================================================
  function appendScannerLog(tag, text, cls = "scan-sys") {
    const box = document.getElementById("scannerLog");
    if (!box) return;
    const tStr = new Date().toLocaleTimeString("en-GB", { hour12: false });
    const row = document.createElement("div");
    row.className = `scan-line ${cls}`;
    row.innerHTML = `<span>[${tStr}] ${text}</span><strong>${tag}</strong>`;
    box.prepend(row);
    while (box.children.length > 45) box.removeChild(box.lastChild);
  }

  // =========================================================================
  // 3. 🗺️ CARTOGRAPHER 24H 预测路径 vs 实时行情同图对比引擎
  // =========================================================================
  const DEFAULT_SESSION = {
    sessionId: "baseline-20261122",
    isManual: false,
    autoRollover: true,
    windowLabel: "2026-11-22 18:00 ~ 2026-11-23 24:00 (EST)",
    startTs: Date.parse("2026-11-22T18:00:00-05:00"),
    endTs: Date.parse("2026-11-24T00:00:00-05:00"),
    levels: {
      cbUp: 65774.7, wickCeil: [64818.0, 65447.5], coreHigh: [62885.3, 63602.7],
      flip: 61248.9, avwap: 59527.0, coreLow: [58057.3, 58774.8], wickFloor: [56317.4, 56945.2], cbDown: 56035.8,
    },
    waypoints: [
      [0.0, 61248.2], [1.5, 60910.0], [4.0, 61420.0], [5.5, 61320.0], [8.0, 62350.0],
      [9.5, 63060.0], [10.5, 62960.0], [12.0, 63180.0], [14.0, 61080.0], [15.0, 60850.0],
      [16.3, 59280.0], [18.0, 58260.0], [19.2, 58560.0], [20.2, 58190.0], [21.6, 58740.0],
      [23.6, 58290.0], [26.5, 58840.0], [28.0, 59080.0], [30.0, 59680.0],
    ],
    annotations: [
      { h: 7.2, price: 63920, text: "① 站稳 Flip 激发负 Gamma 助推，冲高测试 Core High" },
      { h: 16.2, price: 60450, text: "② 美盘多头换手 & ETF 承压，回踩 Core Low 寻底" },
      { h: 22.8, price: 57560, text: "③ Core Low 企稳，回归中枢震荡" },
    ],
  };

  let activeScript = JSON.parse(JSON.stringify(DEFAULT_SESSION));
  let bandMode = "locked";
  let candleSeries = [];
  let candleSourceLabel = "初始化中...";
  let compareHoverX = null;
  let liveWs = null;
  let rolloverCheckBusy = false;

  function getActiveLevels() {
    if (bandMode === "live" && latestSnapshotData && latestSnapshotData.prediction) {
      const p = latestSnapshotData.prediction;
      const m = latestSnapshotData.market_meta || {};
      return {
        cbUp: p.invalidation_bull_trigger, wickCeil: p.wick_ceiling_zone, coreHigh: p.core_high_zone,
        flip: m.gamma_flip || p.spot_price, avwap: m.avwap || p.spot_price,
        coreLow: p.core_low_zone, wickFloor: p.wick_floor_zone, cbDown: p.invalidation_bear_trigger,
      };
    }
    return activeScript.levels;
  }

  function interpolateForecast(ts) {
    const pts = activeScript.waypoints.map(([h, p]) => ({ ts: activeScript.startTs + h * 3600 * 1000, price: p }));
    if (ts <= pts[0].ts) return pts[0].price;
    const last = pts[pts.length - 1];
    if (ts >= last.ts) return last.price;
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      if (ts >= a.ts && ts <= b.ts) {
        const mu = (1 - Math.cos(((ts - a.ts) / Math.max(b.ts - a.ts, 1)) * Math.PI)) / 2;
        return a.price * (1 - mu) + b.price * mu;
      }
    }
    return last.price;
  }

  function setBandMode(mode) {
    bandMode = mode;
    document.getElementById("btnBandLocked").classList.toggle("active", mode === "locked");
    document.getElementById("btnBandLive").classList.toggle("active", mode === "live");
    updateCompareMetaAndMetrics();
    renderComparisonChart();
  }

  async function reanchorFromCurrentState() {
    const btn = document.getElementById("btnReanchor");
    btn.disabled = true;
    try {
      const res = await fetch("/api/reanchor", { method: "POST" });
      applyStateToUI(await res.json());
      appendScannerLog("REANCHOR", "已按最新盘面重新锚定未来 24H 全息推演路径", "scan-sys");
      await fetchCandles();
    } finally {
      btn.disabled = false;
    }
  }

  async function resetToAutoSchedule() {
    const btn = document.getElementById("btnResetScript");
    btn.disabled = true;
    try {
      const res = await fetch("/api/reanchor/auto", { method: "POST" });
      applyStateToUI(await res.json());
      appendScannerLog("AUTO-CAL", "已恢复美东自然日自动跨日轮转窗口", "scan-sys");
      await fetchCandles();
    } finally {
      btn.disabled = false;
    }
  }

  function updateCompareMetaAndMetrics() {
    const remH = (Math.max(0, activeScript.endTs - Date.now()) / 3600000).toFixed(1);
    const modeTag = activeScript.isManual ? `[手动重锚 · ${remH}h后自动续期]` : `[自动跨日轮转 · 剩余${remH}h推演下一日]`;
    document.getElementById("compareWindowMeta").textContent = `窗口: ${activeScript.windowLabel} ${modeTag} · K线源: ${candleSourceLabel}`;
    document.getElementById("btnResetScript").style.display = activeScript.isManual ? "inline-block" : "none";

    const liveSpot = lastRealtimeSpot ?? (candleSeries.length ? candleSeries[candleSeries.length - 1].c : null) ?? (latestSnapshotData?.prediction?.spot_price ?? null);
    if (liveSpot == null) return;

    const latestTs = candleSeries.length ? candleSeries[candleSeries.length - 1].ts : Date.now();
    const predSpot = interpolateForecast(Math.min(Math.max(latestTs, activeScript.startTs), activeScript.endTs));
    const diff = liveSpot - predSpot;
    const diffPct = (diff / predSpot) * 100;
    const lv = getActiveLevels();

    document.getElementById("cmpLiveSpot").textContent = fmtUSD(liveSpot);
    document.getElementById("cmpPredSpot").textContent = fmtUSD(predSpot);

    const deltaEl = document.getElementById("cmpDelta");
    deltaEl.textContent = `${diff >= 0 ? "+" : "-"}$${Math.abs(diff).toFixed(1)} (${diff >= 0 ? "+" : ""}${diffPct.toFixed(2)}%)`;
    deltaEl.style.color = diff >= 0 ? "var(--green)" : "var(--red)";

    const statusEl = document.getElementById("cmpStatus");
    if (liveSpot >= lv.cbUp || liveSpot <= lv.cbDown) {
      statusEl.textContent = "BREACH · 触发熔断警戒"; statusEl.style.color = "var(--red)";
    } else if (Math.abs(diffPct) <= 1.5) {
      statusEl.textContent = "LOCKED · 吻合预测通道"; statusEl.style.color = "var(--green)";
    } else if (diff > 0) {
      statusEl.textContent = "OVERPERFORM · 强于预测基准"; statusEl.style.color = "var(--cyan)";
    } else {
      statusEl.textContent = "UNDERPERFORM · 弱于预测承压"; statusEl.style.color = "var(--amber)";
    }
  }

  function renderComparisonChart() {
    const canvas = document.getElementById("forecastCompareCanvas");
    const tooltip = document.getElementById("compareTooltip");
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const W = rect.width, H = rect.height;
    const pad = { top: 18, right: 68, bottom: 34, left: 14 };
    const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;

    const lv = getActiveLevels();
    const startTs = activeScript.startTs, endTs = activeScript.endTs;
    const visibleCandles = candleSeries.filter((c) => c.ts >= startTs - 900000 && c.ts <= endTs + 900000);
    const allPrices = [lv.cbUp, lv.cbDown, lv.wickCeil[1], lv.wickFloor[0], ...activeScript.waypoints.map((w) => w[1]), ...visibleCandles.flatMap((c) => [c.h, c.l])].filter((v) => typeof v === "number" && !isNaN(v));
    const minP = Math.min(...allPrices) - 450, maxP = Math.max(...allPrices) + 450;

    const xOf = (ts) => pad.left + ((ts - startTs) / (endTs - startTs)) * plotW;
    const yOf = (p) => pad.top + ((maxP - p) / (maxP - minP)) * plotH;

    ctx.clearRect(0, 0, W, H);

    const drawZone = (range, fill, border, label) => {
      const yTop = yOf(range[1]), yBot = yOf(range[0]);
      ctx.fillStyle = fill;
      ctx.fillRect(pad.left, yTop, plotW, Math.max(2, yBot - yTop));
      ctx.strokeStyle = border;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, yTop); ctx.lineTo(pad.left + plotW, yTop);
      ctx.moveTo(pad.left, yBot); ctx.lineTo(pad.left + plotW, yBot);
      ctx.stroke();
      ctx.fillStyle = border;
      ctx.font = "600 11px sans-serif";
      ctx.fillText(`${label} ${fmtUSD(range[0])}~${fmtUSD(range[1])}`, pad.left + 8, yTop + 13);
    };

    drawZone(lv.wickCeil,  "rgba(255,42,95,0.11)",  "rgba(253,164,175,0.72)", "Wick Ceiling");
    drawZone(lv.coreHigh,  "rgba(0,255,157,0.11)",  "rgba(110,231,183,0.72)", "Core High");
    drawZone(lv.coreLow,   "rgba(0,240,255,0.11)",  "rgba(125,211,252,0.72)", "Core Low");
    drawZone(lv.wickFloor, "rgba(255,184,0,0.11)",  "rgba(253,224,71,0.72)",  "Wick Floor");

    const stepP = (maxP - minP) > 14000 ? 2000 : 1000;
    ctx.font = "11px monospace";
    for (let p = Math.ceil(minP / stepP) * stepP; p <= maxP; p += stepP) {
      const y = yOf(p);
      ctx.strokeStyle = "rgba(27, 42, 71, 0.55)";
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
      ctx.fillStyle = "#7b90b2";
      ctx.fillText("$" + p.toLocaleString("en-US"), pad.left + plotW + 6, y + 4);
    }

    const totalHours = (endTs - startTs) / 3600000;
    const hourStep = totalHours > 26 ? 4 : 3;
    for (let h = 0; h <= totalHours; h += hourStep) {
      const ts = startTs + h * 3600 * 1000, x = xOf(ts);
      ctx.strokeStyle = "rgba(27, 42, 71, 0.45)";
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
      const label = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(ts)).replace(", ", " ");
      ctx.fillStyle = "#7b90b2";
      ctx.fillText(label, Math.min(Math.max(pad.left, x - 34), pad.left + plotW - 68), pad.top + plotH + 20);
    }

    const drawRefLine = (price, color, dash, tagText) => {
      if (!price) return;
      const y = yOf(price);
      ctx.save();
      ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
      ctx.font = "600 10.5px monospace";
      const tw = ctx.measureText(tagText).width;
      ctx.fillStyle = "rgba(5, 8, 17, 0.86)";
      ctx.fillRect(pad.left + plotW - tw - 12, y - 13, tw + 8, 12);
      ctx.fillStyle = color;
      ctx.fillText(tagText, pad.left + plotW - tw - 8, y - 3);
      ctx.restore();
    };

    drawRefLine(lv.cbUp,   "#ff2a5f", [4, 4], `向上熔断 ${fmtUSD(lv.cbUp)}`);
    drawRefLine(lv.flip,   "#b47cff", [5, 4], `Gamma Flip ${fmtUSD(lv.flip)}`);
    drawRefLine(lv.avwap,  "#ffb800", [5, 4], `AVWAP ${fmtUSD(lv.avwap)}`);
    drawRefLine(lv.cbDown, "#ff2a5f", [4, 4], `向下熔断 ${fmtUSD(lv.cbDown)}`);

    if (visibleCandles.length > 1) {
      const maxVol = Math.max(...visibleCandles.map((c) => c.v || 1), 1);
      const barW = Math.max(2, (plotW / ((endTs - startTs) / 900000)) * 0.66);
      visibleCandles.forEach((c) => {
        if (c.ts < startTs || c.ts > endTs) return;
        const bh = ((c.v || 0) / maxVol) * (plotH * 0.14);
        ctx.fillStyle = c.c >= c.o ? "rgba(0, 255, 157, 0.28)" : "rgba(255, 42, 95, 0.28)";
        ctx.fillRect(xOf(c.ts) - barW / 2, pad.top + plotH - bh, barW, bh);
      });
    }

    // 预测虚线
    ctx.save();
    ctx.strokeStyle = "rgba(232, 241, 255, 0.85)";
    ctx.lineWidth = 2; ctx.setLineDash([6, 5]);
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const ts = startTs + ((endTs - startTs) * i) / 200;
      const x = xOf(ts), y = yOf(interpolateForecast(ts));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    if (W >= 820 && activeScript.annotations && activeScript.annotations.length) {
      ctx.save();
      ctx.font = "11.5px sans-serif";
      activeScript.annotations.forEach((ann) => {
        const ax = xOf(startTs + ann.h * 3600 * 1000);
        const ay = Math.min(Math.max(pad.top + 22, yOf(ann.price)), pad.top + plotH - 12);
        const tw = ctx.measureText(ann.text).width;
        const bx = Math.min(Math.max(pad.left + 8, ax - tw / 2), pad.left + plotW - tw - 12);
        ctx.fillStyle = "rgba(6, 10, 20, 0.86)";
        ctx.fillRect(bx - 5, ay - 12, tw + 10, 17);
        ctx.fillStyle = "#cbd5e1";
        ctx.fillText(ann.text, bx, ay);
      });
      ctx.restore();
    }

    // 实盘霓虹轨迹 + 15m 影线包络
    const clipped = visibleCandles.filter((c) => c.ts >= startTs && c.ts <= endTs);
    if (clipped.length > 0) {
      ctx.save();
      ctx.fillStyle = "rgba(0, 240, 255, 0.15)";
      ctx.beginPath();
      clipped.forEach((c, idx) => { const x = xOf(c.ts), y = yOf(c.h); if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      for (let i = clipped.length - 1; i >= 0; i--) ctx.lineTo(xOf(clipped[i].ts), yOf(clipped[i].l));
      ctx.closePath(); ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = "#00f0ff"; ctx.lineWidth = 2.6;
      ctx.shadowColor = "rgba(0, 240, 255, 0.65)"; ctx.shadowBlur = 10;
      ctx.beginPath();
      clipped.forEach((c, idx) => { const x = xOf(c.ts), y = yOf(c.c); if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke();

      const lastBar = clipped[clipped.length - 1];
      const lx = xOf(lastBar.ts), ly = yOf(lastBar.c);
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "rgba(0, 240, 255, 0.35)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(lx, pad.top); ctx.lineTo(lx, pad.top + plotH); ctx.stroke();

      ctx.setLineDash([]); ctx.fillStyle = "#00f0ff";
      ctx.beginPath(); ctx.arc(lx, ly, 4.8, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.restore();
    }

    if (compareHoverX !== null && compareHoverX >= pad.left && compareHoverX <= pad.left + plotW) {
      const hoverTs = startTs + ((compareHoverX - pad.left) / plotW) * (endTs - startTs);
      const predAtHover = interpolateForecast(hoverTs);
      ctx.save();
      ctx.strokeStyle = "rgba(0, 240, 255, 0.45)"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(compareHoverX, pad.top); ctx.lineTo(compareHoverX, pad.top + plotH); ctx.stroke();
      ctx.restore();

      let liveRowHtml = `<span style="color:#64748b">尚未到达该时刻</span>`;
      if (clipped.length > 0) {
        let nearest = clipped[0];
        for (const bar of clipped) if (Math.abs(bar.ts - hoverTs) < Math.abs(nearest.ts - hoverTs)) nearest = bar;
        if (Math.abs(nearest.ts - hoverTs) <= 18 * 60 * 1000) {
          const d = nearest.c - predAtHover, dp = (d / predAtHover) * 100;
          const col = d >= 0 ? "#00ff9d" : "#ff2a5f";
          liveRowHtml = `<strong style="color:#00f0ff">${fmtUSD(nearest.c)}</strong> <span style="color:${col}">(${d >= 0 ? "+" : ""}${d.toFixed(1)} / ${d >= 0 ? "+" : ""}${dp.toFixed(2)}%)</span><br/><span>15m 高/低:</span> ${fmtUSD(nearest.h)} / ${fmtUSD(nearest.l)}`;
        }
      }
      const estTimeStr = new Intl.DateTimeFormat("zh-CN", { timeZone: "America/New_York", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(hoverTs));
      tooltip.style.display = "block";
      tooltip.innerHTML = `<div><span style="color:#7b90b2">美东时间:</span> <strong>${estTimeStr} EST</strong></div><div><span style="color:#7b90b2">预测基准:</span> <strong>${fmtUSD(predAtHover)}</strong></div><div><span style="color:#7b90b2">实盘轨迹:</span> ${liveRowHtml}</div>`;
      tooltip.style.left = ((compareHoverX + 14 + tooltip.offsetWidth > W - 10) ? Math.max(8, compareHoverX - tooltip.offsetWidth - 14) : compareHoverX + 14) + "px";
      tooltip.style.top = (pad.top + 10) + "px";
    } else {
      tooltip.style.display = "none";
    }
  }

  // =========================================================================
  // 4. 🛰️ RADAR 声呐阶梯 & GEX 光柱图
  // =========================================================================
  let radarScale = { minP: 55000, span: 10000 };

  function renderRadarLadder(pred, meta) {
    const el = document.getElementById("radarLadder");
    const spot = lastRealtimeSpot || pred.spot_price;
    const points = [pred.invalidation_bull_trigger, pred.wick_ceiling_zone[1], pred.core_high_zone[1], spot, pred.core_low_zone[0], pred.wick_floor_zone[0], pred.invalidation_bear_trigger, meta.gamma_flip, meta.avwap].filter((x) => typeof x === "number" && x > 0);
    const rawMax = Math.max(...points), rawMin = Math.min(...points);
    const pad = Math.max((rawMax - rawMin) * 0.08, pred.em_24h * 0.15);
    const maxP = rawMax + pad, minP = rawMin - pad;
    radarScale = { maxP, span: (maxP - minP) || 1 };
    const yPct = (price) => Math.max(2, Math.min(98, ((radarScale.maxP - price) / radarScale.span) * 100));

    const zoneHtml = (cls, label, low, high) => {
      const top = yPct(high), bottom = yPct(low);
      return `<div class="zone-box ${cls}" style="top:${top}%;height:${Math.max(bottom - top, 5.5)}%;"><span>${label}</span><span class="mono">${fmtUSD(low)}~${fmtUSD(high)}</span></div>`;
    };
    const lineHtml = (idAttr, cls, text, price) => `<div ${idAttr ? `id="${idAttr}"` : ""} class="price-line ${cls}" style="top:${yPct(price)}%;"><span class="price-tag mono">${text}: ${fmtUSD(price)}</span></div>`;

    el.innerHTML = [
      zoneHtml("zone-wick-ceil", "Wick Ceiling", pred.wick_ceiling_zone[0], pred.wick_ceiling_zone[1]),
      zoneHtml("zone-core-high", "Core High", pred.core_high_zone[0], pred.core_high_zone[1]),
      zoneHtml("zone-core-low", "Core Low", pred.core_low_zone[0], pred.core_low_zone[1]),
      zoneHtml("zone-wick-floor", "Wick Floor", pred.wick_floor_zone[0], pred.wick_floor_zone[1]),
      lineHtml("", "line-inv", "向上熔断", pred.invalidation_bull_trigger),
      lineHtml("", "line-inv", "向下熔断", pred.invalidation_bear_trigger),
      meta.gamma_flip ? lineHtml("", "line-flip", "Flip", meta.gamma_flip) : "",
      meta.avwap ? lineHtml("", "line-avwap", "AVWAP", meta.avwap) : "",
      lineHtml("radarSpotLine", "line-spot", "SPOT", spot),
    ].join("");
  }

  function updateRadarSpotCursor(spot) {
    const line = document.getElementById("radarSpotLine");
    if (!line || !radarScale.span) return;
    const top = Math.max(2, Math.min(98, ((radarScale.maxP - spot) / radarScale.span) * 100));
    line.style.top = top + "%";
    const tag = line.querySelector(".price-tag");
    if (tag) tag.textContent = "SPOT: " + fmtUSD(spot);
  }

  function renderGexChart(bars, meta) {
    if (!window.Chart) return;
    const ctx = document.getElementById("gexChart").getContext("2d");
    const labels = bars.map((b) => {
      let tag = "$" + (b.strike / 1000).toFixed(b.strike % 1000 === 0 ? 0 : 1) + "k";
      if (Math.abs(b.strike - meta.call_wall) < 1) tag += " [CW]";
      if (Math.abs(b.strike - meta.put_wall) < 1) tag += " [PW]";
      return tag;
    });
    const values = bars.map((b) => b.net_gex_m);
    if (gexChart) gexChart.destroy();
    gexChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: values.map((v) => v >= 0 ? "rgba(0, 255, 157, 0.72)" : "rgba(255, 42, 95, 0.72)"),
          borderColor: values.map((v) => v >= 0 ? "#00ff9d" : "#ff2a5f"),
          borderWidth: 1, borderRadius: 3,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0b111e", borderColor: "#00f0ff", borderWidth: 1,
            callbacks: { label: (item) => { const b = bars[item.dataIndex]; return [` Net GEX: ${b.net_gex_m >= 0 ? "+" : ""}${b.net_gex_m.toFixed(2)}M`, ` Call: +${b.call_gex_m.toFixed(2)}M | Put: ${b.put_gex_m.toFixed(2)}M`]; } },
          },
        },
        scales: {
          x: { ticks: { color: "#7b90b2", font: { size: 10.5 } }, grid: { color: "rgba(27,42,71,0.35)" } },
          y: { ticks: { color: "#7b90b2", callback: (v) => (v >= 0 ? "+" : "") + v + "M" }, grid: { color: "rgba(27,42,71,0.5)" } },
        },
      },
    });
  }

  // =========================================================================
  // 5. 毫秒级 WebSocket 引擎 (Tickers + Whale Trades + Books5 深度)
  // =========================================================================
  function pushLiveTick(price, tsMs) {
    if (!price || isNaN(price)) return;
    const prev = lastRealtimeSpot;
    lastRealtimeSpot = price;
    tickCountWindow += 1;

    const spotEl = document.getElementById("kpiSpot");
    spotEl.textContent = fmtUSD(price);
    if (prev != null && price !== prev) {
      const cls = price > prev ? "tick-up" : "tick-down";
      spotEl.classList.remove("tick-up", "tick-down");
      spotEl.classList.add(cls);
      setTimeout(() => spotEl.classList.remove("tick-up", "tick-down"), 240);

      // 检测实时穿越 Gamma Flip
      const flip = latestSnapshotData?.market_meta?.gamma_flip;
      if (flip && ((prev < flip && price >= flip) || (prev > flip && price <= flip))) {
        triggerOrbPulse(1.4);
        appendScannerLog("FLIP CROSS", `现价实时${price >= flip ? "向上突破" : "向下跌破"} Gamma Flip (${fmtUSD(flip)})`, price >= flip ? "scan-buy" : "scan-sell");
      }
    }

    updateRadarSpotCursor(price);
    updateOrbScoreAndShields(price);

    const ts = tsMs || Date.now();
    const bucket = Math.floor(ts / 900000) * 900000;
    if (!candleSeries.length) {
      candleSeries.push({ ts: bucket, o: price, h: price, l: price, c: price, v: 0 });
    } else {
      const last = candleSeries[candleSeries.length - 1];
      if (bucket > last.ts) candleSeries.push({ ts: bucket, o: last.c, h: Math.max(last.c, price), l: Math.min(last.c, price), c: price, v: 0 });
      else { last.c = price; last.h = Math.max(last.h, price); last.l = Math.min(last.l, price); }
    }
    updateCompareMetaAndMetrics();
    renderComparisonChart();
  }

  function updateOrderbookBar(bidVol, askVol) {
    const total = (bidVol + askVol) || 1;
    const bidPct = Math.round((bidVol / total) * 100);
    document.getElementById("obBidPct").textContent = bidPct + "%";
    document.getElementById("obAskPct").textContent = (100 - bidPct) + "%";
    document.getElementById("obBidBar").style.width = bidPct + "%";
  }

  function startDemoStream() {
    if (demoSimTimer) clearInterval(demoSimTimer);
    if (currentMode !== "DEMO") return;
    document.getElementById("wsHzPill").textContent = "DEMO SIM · 100ms";
    demoSimTimer = setInterval(() => {
      if (currentMode !== "DEMO") return;
      const base = lastRealtimeSpot || latestSnapshotData?.prediction?.spot_price || 60500;
      const nextP = Math.round((base + (Math.random() - 0.495) * 42) * 10) / 10;
      pushLiveTick(nextP, Date.now());
      updateOrderbookBar(40 + Math.random() * 60, 40 + Math.random() * 60);
      if (Math.random() < 0.28) {
        const isBuy = Math.random() > 0.48;
        const usdK = (25 + Math.random() * 190).toFixed(0);
        if (usdK > 110) triggerOrbPulse(1.1);
        appendScannerLog(isBuy ? `BUY $${usdK}K` : `SELL $${usdK}K`, `DEMO 永续主动${isBuy ? "吃单扫货" : "砸盘卖出"} @ ${fmtUSD(nextP)}`, isBuy ? "scan-buy" : "scan-sell");
      }
    }, 950);
  }

  function connectRealtimeTickerWs() {
    if (liveWs) { try { liveWs.close(); } catch (_) {} liveWs = null; }
    if (demoSimTimer) { clearInterval(demoSimTimer); demoSimTimer = null; }
    if (currentMode !== "LIVE") { startDemoStream(); return; }

    const connectDeribitWs = () => {
      try {
        const ws = new WebSocket("wss://www.deribit.com/ws/api/v2");
        liveWs = ws;
        ws.onopen = () => {
          document.getElementById("wsHzPill").textContent = "DERIBIT WS · LIVE";
          ws.send(JSON.stringify({ jsonrpc: "2.0", id: 1, method: "public/subscribe", params: { channels: ["ticker.BTC-PERPETUAL.100ms", "trades.BTC-PERPETUAL.100ms"] } }));
        };
        ws.onmessage = (ev) => {
          if (currentMode !== "LIVE") return;
          const msg = JSON.parse(ev.data);
          const ch = msg?.params?.channel || "";
          const data = msg?.params?.data;
          if (ch.startsWith("ticker.") && data?.last_price) {
            pushLiveTick(Number(data.last_price), Number(data.timestamp || Date.now()));
            if (data.best_bid_amount && data.best_ask_amount) updateOrderbookBar(Number(data.best_bid_amount), Number(data.best_ask_amount));
          } else if (ch.startsWith("trades.") && Array.isArray(data)) {
            data.forEach((tr) => {
              const notional = Number(tr.amount || 0);
              if (notional >= 20000) {
                const isBuy = tr.direction === "buy";
                if (notional >= 80000) triggerOrbPulse(1.2);
                appendScannerLog(`${isBuy ? "BUY" : "SELL"} $${(notional / 1000).toFixed(0)}K`, `Deribit 主动${isBuy ? "买入" : "卖出"} @ ${fmtUSD(tr.price)}`, isBuy ? "scan-buy" : "scan-sell");
              }
            });
          }
        };
      } catch (_) {}
    };

    try {
      const ws = new WebSocket("wss://ws.okx.com:8443/ws/v5/public");
      liveWs = ws;
      let gotTick = false;
      const fallbackTimer = setTimeout(() => {
        if (!gotTick && currentMode === "LIVE") { try { ws.close(); } catch (_) {} connectDeribitWs(); }
      }, 5500);

      ws.onopen = () => {
        document.getElementById("wsHzPill").textContent = "OKX WS · 100ms LIVE";
        ws.send(JSON.stringify({
          op: "subscribe",
          args: [
            { channel: "tickers", instId: "BTC-USDT-SWAP" },
            { channel: "trades", instId: "BTC-USDT-SWAP" },
            { channel: "books5", instId: "BTC-USDT-SWAP" },
          ],
        }));
      };
      ws.onmessage = (ev) => {
        if (currentMode !== "LIVE") return;
        const msg = JSON.parse(ev.data);
        const ch = msg?.arg?.channel;
        if (ch === "tickers" && msg.data?.[0]?.last) {
          gotTick = true; clearTimeout(fallbackTimer);
          pushLiveTick(Number(msg.data[0].last), Number(msg.data[0].ts || Date.now()));
        } else if (ch === "books5" && msg.data?.[0]) {
          const bids = (msg.data[0].bids || []).reduce((s, r) => s + Number(r[1] || 0), 0);
          const asks = (msg.data[0].asks || []).reduce((s, r) => s + Number(r[1] || 0), 0);
          updateOrderbookBar(bids, asks);
        } else if (ch === "trades" && Array.isArray(msg.data)) {
          msg.data.forEach((tr) => {
            const px = Number(tr.px || 0), szContracts = Number(tr.sz || 0);
            // OKX BTC-USDT-SWAP 1张 = 0.01 BTC
            const notional = px * szContracts * 0.01;
            if (notional >= 20000) {
              const isBuy = tr.side === "buy";
              if (notional >= 80000) triggerOrbPulse(1.25);
              appendScannerLog(`${isBuy ? "BUY" : "SELL"} $${(notional / 1000).toFixed(0)}K`, `OKX 永续主力${isBuy ? "扫单买入" : "主动砸盘"} @ ${fmtUSD(px)}`, isBuy ? "scan-buy" : "scan-sell");
            }
          });
        }
      };
      ws.onerror = () => { clearTimeout(fallbackTimer); if (!gotTick) connectDeribitWs(); };
    } catch (_) {
      connectDeribitWs();
    }
  }

  // =========================================================================
  // 6. 状态同步与控制交互
  // =========================================================================
  function applyStateToUI(resp) {
    const prevMode = currentMode;
    currentMode = resp.mode || "LIVE";
    countdownSec = resp.seconds_to_next_refresh ?? 300;
    document.getElementById("lastUpdated").textContent = resp.last_updated_utc || "--";
    document.getElementById("countdown").textContent = countdownSec;
    if (resp.exchange_source) document.getElementById("selectSource").value = resp.exchange_source;

    let sessionChanged = false;
    if (resp.forecast_session && resp.forecast_session.sessionId !== activeScript.sessionId) {
      activeScript = resp.forecast_session;
      sessionChanged = true;
      appendScannerLog("ROLLOVER", `预测窗口已同步切换至 ${activeScript.windowLabel}`, "scan-sys");
    }

    const pill = document.getElementById("modePill");
    pill.textContent = currentMode === "LIVE" ? "● LIVE 实盘" : "● DEMO 沙盘";
    pill.className = "pill mono " + (currentMode === "LIVE" ? "live" : "demo");

    const errBar = document.getElementById("errorBar");
    if (resp.error) { errBar.style.display = "block"; errBar.textContent = "接口告警: " + resp.error; }
    else errBar.style.display = "none";

    if (prevMode !== currentMode) { connectRealtimeTickerWs(); fetchCandles(); }
    else if (sessionChanged) fetchCandles();

    if (!resp.data) return;
    latestSnapshotData = resp.data;
    const { prediction: pred, market_meta: meta, alert, strike_gex_bars: bars, regime_label, active_exchange, failover_reason } = resp.data;

    orbState.netGexM = Number(meta.net_gex_m || 0);
    orbState.dvolPct = Number(meta.dvol_pct || 50);

    const exPill = document.getElementById("exchangePill");
    exPill.textContent = "SRC: " + (active_exchange || "--");
    exPill.className = "pill mono " + (active_exchange === "OKX" ? "live" : (active_exchange === "DERIBIT" ? "alt" : "demo"));

    const etfVal = Number(meta.etf_netflow_m || 0);
    const etfPill = document.getElementById("etfPill");
    etfPill.textContent = `ETF: ${etfVal >= 0 ? "+" : ""}${etfVal.toFixed(1)}M`;
    etfPill.className = "pill mono " + (etfVal > 0 ? "live" : (etfVal < 0 ? "demo" : "alt"));

    const fBar = document.getElementById("failoverBar");
    const notices = [];
    if (failover_reason && currentMode === "LIVE") notices.push(`永续路由: ${failover_reason}`);
    if (resp.etf_last_error && currentMode === "LIVE" && String(meta.etf_source || "").startsWith("Auto Failed")) notices.push(`ETF 提示: ${resp.etf_last_error}`);
    if (notices.length > 0) { fBar.style.display = "block"; fBar.textContent = notices.join("  |  "); }
    else fBar.style.display = "none";

    const shieldBanner = document.getElementById("shieldBanner");
    shieldBanner.className = "shield-center " + (alert.level || "NORMAL");
    document.getElementById("alertTitle").textContent = alert.title;
    document.getElementById("alertMsg").textContent = alert.message;
    document.getElementById("invBull").textContent = fmtUSD(pred.invalidation_bull_trigger);
    document.getElementById("invBear").textContent = fmtUSD(pred.invalidation_bear_trigger);

    if (lastRealtimeSpot == null || currentMode === "DEMO") lastRealtimeSpot = pred.spot_price;
    document.getElementById("kpiSpot").textContent = fmtUSD(lastRealtimeSpot);
    document.getElementById("kpiEm").textContent = `EM: ±${fmtUSD(pred.em_24h)} | DVOL: ${meta.dvol_pct}%`;
    document.getElementById("kpiRegime").textContent = regime_label;
    document.getElementById("kpiWeekend").textContent = `周末薄盘: ${pred.weekend_overlay ? "激活 (+30%)" : "未触发"}`;

    const gexEl = document.getElementById("kpiGex");
    gexEl.textContent = `${meta.net_gex_m >= 0 ? "+" : ""}${meta.net_gex_m.toFixed(2)}M`;
    gexEl.style.color = meta.net_gex_m >= 0 ? "var(--green)" : "var(--red)";
    document.getElementById("kpiWalls").textContent = `Flip ${fmtUSD(meta.gamma_flip)} | PW ${fmtUSD(meta.put_wall)} / CW ${fmtUSD(meta.call_wall)}`;

    const etfEl = document.getElementById("kpiEtf");
    etfEl.textContent = `${etfVal >= 0 ? "+" : "-"}$${Math.abs(etfVal).toFixed(1)}M`;
    etfEl.style.color = etfVal > 0 ? "var(--green)" : (etfVal < 0 ? "var(--red)" : "var(--text)");
    document.getElementById("kpiEtfMeta").textContent = `${meta.etf_date || "--"} · ${meta.etf_source || "--"}`;

    document.getElementById("kpiSent").textContent = `${pred.raw_sentiment >= 0 ? "+" : ""}${pred.raw_sentiment.toFixed(2)} → ${pred.adjusted_sentiment >= 0 ? "+" : ""}${pred.adjusted_sentiment.toFixed(2)}`;
    document.getElementById("kpiFund").textContent = `费率 ${meta.funding_rate_pct}% | F&G ${meta.fear_greed} | 24H OI ${meta.oi_24h_pct >= 0 ? "+" : ""}${meta.oi_24h_pct}%`;

    const d = pred.diagnostics || {};
    const trapEl = document.getElementById("crowdTrapBadge");
    trapEl.textContent = `杠杆态: ${d.funding_signal || "正常"}`;
    trapEl.className = "pill mono " + (String(d.funding_signal || "").includes("拥挤") ? "demo" : "live");

    renderRadarLadder(pred, meta);
    renderGexChart(bars || [], meta);
    updateOrbScoreAndShields(lastRealtimeSpot);

    if (currentMode === "LIVE" && pred.spot_price) pushLiveTick(pred.spot_price, Date.now());
    else { updateCompareMetaAndMetrics(); renderComparisonChart(); }

    document.getElementById("diagContracts").textContent = `Deribit 活跃合约: ${meta.active_contracts}`;
    document.getElementById("diagBox").innerHTML = `
      <div class="diag-item"><span>合成权重:</span><strong>${d.weights || "--"}</strong></div>
      <div class="diag-item"><span>费率信号:</span><strong>${d.funding_signal || "--"}</strong></div>
      <div class="diag-item"><span>AVWAP偏离:</span><strong>${d.avwap_distance_pct || "--"}</strong></div>
      <div class="diag-item"><span>1H OI变化:</span><strong>${alert.oi_1h_change_pct >= 0 ? "+" : ""}${alert.oi_1h_change_pct}%</strong></div>
    `;
  }

  async function fetchCandles() {
    try {
      const res = await fetch(`/api/candles?start_ms=${Math.floor(activeScript.startTs)}`);
      const json = await res.json();
      if (json && Array.isArray(json.candles) && json.candles.length > 0) {
        candleSeries = json.candles;
        candleSourceLabel = json.source || "OKX/DERIBIT";
      } else if (json && json.error) candleSourceLabel = "使用实时点位回传";
      updateCompareMetaAndMetrics();
      renderComparisonChart();
    } catch (e) { console.warn("Candles warning:", e); }
  }

  async function fetchSnapshot() {
    try {
      const res = await fetch("/api/snapshot");
      applyStateToUI(await res.json());
    } catch (e) { console.error("Snapshot failed:", e); }
  }

  async function manualRefresh() {
    const btn = document.getElementById("btnRefresh");
    btn.disabled = true; btn.textContent = "同步中...";
    try {
      const res = await fetch("/api/refresh", { method: "POST" });
      applyStateToUI(await res.json());
      appendScannerLog("SYNC", "已强制同步 Deribit 全量期权链与 ETF 缓存", "scan-sys");
      await fetchCandles();
    } finally {
      btn.disabled = false; btn.textContent = "立即同步期权链";
    }
  }

  async function toggleMode() {
    const nextMode = currentMode === "LIVE" ? "DEMO" : "LIVE";
    lastRealtimeSpot = null;
    const res = await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: nextMode }) });
    applyStateToUI(await res.json());
    appendScannerLog("MODE", `系统已切换至 ${nextMode} 模式`, "scan-sys");
    await fetchCandles();
  }

  async function applyConfig() {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exchange_source: document.getElementById("selectSource").value,
        etf_netflow_m: document.getElementById("inputEtf").value,
        polymarket_bias: document.getElementById("inputPoly").value,
      }),
    });
    applyStateToUI(await res.json());
    appendScannerLog("CONFIG", "数据源路由与覆盖参数已更新重算", "scan-sys");
    await fetchCandles();
  }

  (function bindCompareEvents() {
    const canvas = document.getElementById("forecastCompareCanvas");
    const updateHover = (clientX) => { compareHoverX = clientX - canvas.getBoundingClientRect().left; renderComparisonChart(); };
    canvas.addEventListener("mousemove", (e) => updateHover(e.clientX));
    canvas.addEventListener("touchmove", (e) => { if (e.touches?.[0]) updateHover(e.touches[0].clientX); }, { passive: true });
    canvas.addEventListener("mouseleave", () => { compareHoverX = null; renderComparisonChart(); });
    canvas.addEventListener("touchend", () => { compareHoverX = null; renderComparisonChart(); });
    window.addEventListener("resize", renderComparisonChart);
  })();

  setInterval(() => {
    if (Date.now() >= activeScript.endTs + 2000 && !rolloverCheckBusy) {
      rolloverCheckBusy = true;
      fetchSnapshot().finally(() => { setTimeout(() => { rolloverCheckBusy = false; }, 10000); });
    }
    if (countdownSec > 0) {
      countdownSec -= 1;
      document.getElementById("countdown").textContent = countdownSec;
      if (countdownSec % 30 === 0) updateCompareMetaAndMetrics();
    } else {
      fetchSnapshot();
      fetchCandles();
    }
    tickCountWindow = 0;
  }, 1000);

  requestAnimationFrame(animateOrb);
  renderComparisonChart();
  Promise.all([fetchSnapshot(), fetchCandles()]).then(() => {
    connectRealtimeTickerWs();
    setTimeout(() => {
      if (document.getElementById("kpiSpot").textContent === "--") { fetchSnapshot(); fetchCandles(); }
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
