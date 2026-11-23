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
SESSION_CACHE_FILE = Path(__file__).resolve().parent / ".forecast_session_state.json"

# 初始验证基准窗口: 2026-11-22 18:00 EST ~ 2026-11-23 24:00 EST
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
        prev_day = (e_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        e_str = f"{prev_day} 24:00"
    else:
        e_str = e_dt.strftime("%Y-%m-%d %H:%M")
    return f"{s_str} ~ {e_str} (EST)"


def _initial_locked_session() -> Dict[str, Any]:
    """2026-11-23 24:00 (EST) 到期前保持锁定的基准验证剧本"""
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
            {"h": 7.2, "price": 63920.0, "text": "① 今晚~周一晨: 站稳 Flip 激发负 Gamma 助推，冲高测试 Core High"},
            {"h": 16.2, "price": 60450.0, "text": "② 周一美盘: 多头踩踏 & ETF 承压，回踩 Core Low 寻底"},
            {"h": 22.8, "price": 57560.0, "text": "③ 周一晚: Core Low 企稳，回归中枢震荡"},
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
    """根据四象限状态、净 GEX、Gamma Flip、AVWAP 与磁吸带生成未来 24H 推演剧本"""
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
    ch_mid = round((ch_low + ch_high) / 2.0, 1)
    cl_mid = round((cl_low + cl_high) / 2.0, 1)

    safe_flip = min(max(raw_flip, cl_low), ch_high)
    safe_avwap = min(max(raw_avwap, cl_low), ch_high)
    equilibrium = round(safe_avwap * 0.55 + safe_flip * 0.45, 1)
    total_h = max(24.0, round((end_ms - start_ms) / 3600000.0, 2))
    scale = total_h / 24.0

    if net_gex_m < 0 and spot >= raw_flip:
        # 负 Gamma + Flip 上方：亚欧盘顺势冲高 Core High，美盘检验 ETF 与多头拥挤度后回踩企稳
        deep_pullback = (etf_m < -100.0) or (fund_pct > 0.005)
        dip_target = cl_mid if deep_pullback else round((safe_avwap + cl_high) / 2.0, 1)
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
            {"h": round(5.8 * scale, 1), "price": round(ch_high + (ch_high - ch_low) * 0.45, 1),
             "text": "① 初段: 站稳 Flip 激发负 Gamma 助推，上探 Core High"},
            {"h": round(14.2 * scale, 1), "price": round((safe_flip + dip_target) / 2.0, 1),
             "text": "② 美盘: 获利盘与 ETF 流量博弈，回踩支撑带换手"},
            {"h": round(20.2 * scale, 1), "price": round(min(dip_target, cl_high) - (cl_high - cl_low) * 0.45, 1),
             "text": "③ 尾段: 抛压消化后企稳，向中枢均值回归"},
        ]
    elif net_gex_m < 0 and spot < raw_flip:
        # 负 Gamma + Flip 下方：先顺惯性下探 Core Low 扫流动性，再反抽 AVWAP / Flip
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
            {"h": round(5.8 * scale, 1), "price": round(cl_low - (cl_high - cl_low) * 0.48, 1),
             "text": "① 初段: Flip 下方负 Gamma 助跌，下探 Core Low 寻底"},
            {"h": round(14.0 * scale, 1), "price": round(safe_avwap + (ch_low - cl_high) * 0.18, 1),
             "text": "② 中段: 空头回补修复，反抽测试 AVWAP"},
            {"h": round(20.0 * scale, 1), "price": round(safe_flip + (ch_low - cl_high) * 0.22, 1),
             "text": "③ 尾段: 围绕 Flip 与中枢区间收敛震荡"},
        ]
    else:
        # 正 Gamma 吸波环境：做市商高抛低吸压制波动，围绕 Core 内沿均值回归
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
            {"h": round(7.2 * scale, 1), "price": round(ch_high + (ch_high - ch_low) * 0.35, 1),
             "text": "① 正 Gamma 做市商高抛吸波，冲高受阻于 Core 内沿"},
            {"h": round(16.8 * scale, 1), "price": round(cl_low - (cl_high - cl_low) * 0.35, 1),
             "text": "② 低位对冲买盘托底，全天围绕 AVWAP 均值收敛"},
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
# 2. 后台 5 分钟轮询与状态缓存管理 (Carbon State Manager)
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


def _build_demo_candles(session: Dict[str, Any]) -> List[Dict[str, float]]:
    """DEMO 沙盘模式下根据当前活动预测窗口生成平滑演示 15m K 线"""
    start_ms = int(session["startTs"])
    end_ms = int(session["endTs"])
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
                t = (h - h0) / max(h1 - h0, 1e-6)
                mu = (1.0 - math.cos(t * math.pi)) / 2.0
                base = p0 * (1.0 - mu) + p1 * mu
                break
        wave = math.sin(step * 0.85) * 140.0 + math.cos(step * 0.37) * 75.0
        close_p = round(base + wave, 1)
        open_p = prev_c
        high_p = round(max(open_p, close_p) + abs(math.sin(step * 1.3)) * 195.0 + 40.0, 1)
        low_p = round(min(open_p, close_p) - abs(math.cos(step * 1.1)) * 205.0 - 40.0, 1)
        vol = round(180.0 + abs(close_p - open_p) * 1.35 + (380.0 if 11.0 <= h <= 17.0 else 60.0), 1)
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
            "User-Agent": "BTC-24H-GEX-Dashboard/4.2",
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
