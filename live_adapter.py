from __future__ import annotations

import asyncio
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

from btc_24h_predictor import (
    BTC24HPredictorPipeline,
    BaseMarketDataAdapter,
    DerivativesSnapshot,
    EventShock,
    SentimentSnapshot,
    TechnicalSnapshot,
)


# =============================================================================
# 1. 美国现货 BTC ETF 净流入自动抓取引擎 (SoSoValue API -> Farside HTML 瀑布流)
# =============================================================================

@dataclass(frozen=True)
class ETFFlowSnapshot:
    netflow_usd: float      # 最新交易日净流入 (USD)，流出为负
    trade_date: str         # 对应交易日 (YYYY-MM-DD)
    source: str             # "SoSoValue API" | "Farside Investors" | "Manual Override" | "Unavailable"
    fetched_at_mono: float  # 缓存时间戳


class _FarsideTableParser(HTMLParser):
    """零依赖标准库 HTML 解析器，专门提取 Farside Investors ETF 流量表格的所有行"""

    def __init__(self) -> None:
        super().__init__()
        self._in_tr = False
        self._in_cell = False
        self._cur_cell: List[str] = []
        self._cur_row: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "tr":
            self._in_tr = True
            self._cur_row = []
        elif t in ("td", "th") and self._in_tr:
            self._in_cell = True
            self._cur_cell = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            self._in_cell = False
            text = "".join(self._cur_cell).strip()
            self._cur_row.append(text)
        elif t == "tr" and self._in_tr:
            self._in_tr = False
            if self._cur_row:
                self.rows.append(self._cur_row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cur_cell.append(data)


class SpotETFFlowParser:
    """
    标准化解析 SoSoValue v1 JSON 与 Farside Investors HTML 表格，
    确保跳过 T+1 未结算空行与汇总尾行，始终返回最新已公布交易日的真实净流入。
    """

    DATE_PATTERN = re.compile(r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$")

    @staticmethod
    def parse_sosovalue_json(payload: Dict[str, Any]) -> Optional[ETFFlowSnapshot]:
        if payload.get("code") != 0:
            return None
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return None

        # 按 date 降序排序，跳过 T+1 尚未结算 (total_net_inflow 为 None) 的最新空行
        rows = sorted(
            [r for r in data if isinstance(r, dict) and r.get("date")],
            key=lambda r: str(r.get("date")),
            reverse=True,
        )
        for row in rows:
            val = row.get("total_net_inflow")
            if val is None or val == "":
                continue
            try:
                flow_usd = float(val)
            except (TypeError, ValueError):
                continue
            return ETFFlowSnapshot(
                netflow_usd=round(flow_usd, 2),
                trade_date=str(row["date"])[:10],
                source="SoSoValue API",
                fetched_at_mono=time.monotonic(),
            )
        return None

    @classmethod
    def _parse_farside_number_millions(cls, raw: str) -> Optional[float]:
        s = raw.strip().replace(",", "").replace("$", "")
        if not s or s in ("-", "—", "N/A", "n/a", "0.0", "0"):
            return None
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1].strip()
        try:
            val = float(s)
        except ValueError:
            return None
        return -val if neg else val

    @classmethod
    def parse_farside_html(cls, html: str) -> Optional[ETFFlowSnapshot]:
        parser = _FarsideTableParser()
        parser.feed(html)

        latest: Optional[Tuple[datetime, float]] = None
        for row in parser.rows:
            if len(row) < 3:
                continue
            date_cell = row[0].strip()
            if not cls.DATE_PATTERN.match(date_cell):
                continue
            try:
                dt = datetime.strptime(date_cell, "%d %b %Y")
            except ValueError:
                continue

            # 最后一列为 Total (单位: US$ millions)
            total_m = cls._parse_farside_number_millions(row[-1])
            if total_m is None:
                continue

            if latest is None or dt > latest[0]:
                latest = (dt, total_m)

        if latest is None:
            return None

        dt_best, total_m_best = latest
        return ETFFlowSnapshot(
            netflow_usd=round(total_m_best * 1_000_000.0, 2),
            trade_date=dt_best.strftime("%Y-%m-%d"),
            source="Farside Investors",
            fetched_at_mono=time.monotonic(),
        )


class ETFFlowCache:
    """日频 ETF 数据进程级缓存（默认 30 分钟 TTL），避免每 5 分钟轮询浪费 API 配额"""

    def __init__(self, ttl_sec: float = 1800.0) -> None:
        self.ttl_sec = ttl_sec
        self.cached: Optional[ETFFlowSnapshot] = None
        self.last_error: Optional[str] = None

    def get_fresh(self) -> Optional[ETFFlowSnapshot]:
        if self.cached is None:
            return None
        if (time.monotonic() - self.cached.fetched_at_mono) <= self.ttl_sec:
            return self.cached
        return None

    def put(self, snap: ETFFlowSnapshot) -> None:
        self.cached = snap
        self.last_error = None

    def clear(self) -> None:
        self.cached = None
        self.last_error = None


GLOBAL_ETF_CACHE = ETFFlowCache(ttl_sec=1800.0)


# =============================================================================
# 2. Deribit 期权链 Black-Scholes Gamma & GEX 实时计算引擎
# =============================================================================

@dataclass(frozen=True)
class ParsedOption:
    instrument_name: str
    strike: float
    option_type: str        # "C" or "P"
    t_years: float          # 距到期剩余年化时间
    open_interest: float    # 持仓量 (BTC 张数)
    mark_iv: float          # 隐含波动率 (小数，如 0.65)
    underlying_price: float


@dataclass(frozen=True)
class GEXProfile:
    net_gex_usd: float      # 当前现价下的净 GEX (每 1% 现货波动对应美元对冲量)
    gamma_flip_price: float # 净 GEX = 0 的正负翻转位
    call_wall: float        # 近期最大正 Gamma 看涨期权行权价
    put_wall: float         # 近期最大负 Gamma 看跌期权行权价
    active_contracts: int   # 参与计算的活跃合约数量


class DeribitGEXCalculator:
    """
    零第三方数值库依赖，基于 Black-76 (期货标的 r=0) 计算 Deribit BTC 期权 GEX 结构。
    做市商头寸标准假设：做市商多 Call Gamma (+)、空 Put Gamma (-)。
    """

    INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

    @staticmethod
    def parse_instrument(
        raw: Dict[str, Any],
        now_utc: datetime,
        max_dte_days: float = 14.0,
    ) -> Optional[ParsedOption]:
        name = raw.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) != 4 or parts[0] != "BTC":
            return None

        oi = float(raw.get("open_interest") or 0.0)
        mark_iv_raw = float(raw.get("mark_iv") or 0.0)
        underlying = float(raw.get("underlying_price") or 0.0)
        if oi <= 0.0 or mark_iv_raw <= 0.0 or underlying <= 0.0:
            return None

        try:
            expiry_dt = datetime.strptime(parts[1], "%d%b%y").replace(
                hour=8, minute=0, second=0, tzinfo=timezone.utc
            )
            strike = float(parts[2])
            opt_type = parts[3].upper()
        except ValueError:
            return None

        if opt_type not in ("C", "P"):
            return None

        seconds_left = (expiry_dt - now_utc).total_seconds()
        if seconds_left <= 300.0:
            return None

        dte_days = seconds_left / 86400.0
        if dte_days > max_dte_days:
            return None

        sigma = mark_iv_raw / 100.0 if mark_iv_raw > 3.0 else mark_iv_raw

        return ParsedOption(
            instrument_name=name,
            strike=strike,
            option_type=opt_type,
            t_years=seconds_left / (365.0 * 86400.0),
            open_interest=oi,
            mark_iv=sigma,
            underlying_price=underlying,
        )

    @classmethod
    def bs_gamma(cls, spot: float, strike: float, t_years: float, sigma: float) -> float:
        if spot <= 0.0 or strike <= 0.0 or t_years <= 0.0 or sigma <= 0.0:
            return 0.0
        sqrt_t = math.sqrt(t_years)
        denom = sigma * sqrt_t
        if denom <= 1e-8:
            return 0.0
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / denom
        pdf = cls.INV_SQRT_2PI * math.exp(-0.5 * d1 * d1)
        return pdf / (spot * denom)

    @classmethod
    def compute_gex_at_spot(cls, options: List[ParsedOption], spot: float) -> float:
        total_gex = 0.0
        spot_sq_1pct = spot * spot * 0.01
        for opt in options:
            gamma = cls.bs_gamma(spot, opt.strike, opt.t_years, opt.mark_iv)
            sign = 1.0 if opt.option_type == "C" else -1.0
            total_gex += sign * gamma * opt.open_interest * spot_sq_1pct
        return total_gex

    @classmethod
    def build_strike_bars(
        cls,
        raw_summaries: List[Dict[str, Any]],
        spot_price: float,
        now_utc: Optional[datetime] = None,
        max_dte_days: float = 14.0,
    ) -> List[Dict[str, Any]]:
        now = now_utc or datetime.now(timezone.utc)
        options = [
            p
            for item in raw_summaries
            if (p := cls.parse_instrument(item, now, max_dte_days=max_dte_days)) is not None
        ]
        if len(options) < 10:
            options = [
                p
                for item in raw_summaries
                if (p := cls.parse_instrument(item, now, max_dte_days=35.0)) is not None
            ]

        buckets: Dict[float, Dict[str, float]] = {}
        spot_sq_1pct = spot_price * spot_price * 0.01
        low_cut, high_cut = spot_price * 0.84, spot_price * 1.16

        for opt in options:
            if not (low_cut <= opt.strike <= high_cut):
                continue
            gamma = cls.bs_gamma(spot_price, opt.strike, opt.t_years, opt.mark_iv)
            gex_usd = gamma * opt.open_interest * spot_sq_1pct
            slot = buckets.setdefault(opt.strike, {"call_gex": 0.0, "put_gex": 0.0})
            if opt.option_type == "C":
                slot["call_gex"] += gex_usd
            else:
                slot["put_gex"] -= gex_usd

        bars: List[Dict[str, Any]] = []
        for strike in sorted(buckets.keys()):
            c_val = buckets[strike]["call_gex"]
            p_val = buckets[strike]["put_gex"]
            bars.append(
                {
                    "strike": strike,
                    "call_gex_m": round(c_val / 1e6, 3),
                    "put_gex_m": round(p_val / 1e6, 3),
                    "net_gex_m": round((c_val + p_val) / 1e6, 3),
                }
            )
        return bars

    @classmethod
    def analyze_chain(
        cls,
        raw_summaries: List[Dict[str, Any]],
        spot_price: float,
        now_utc: Optional[datetime] = None,
        max_dte_days: float = 14.0,
    ) -> GEXProfile:
        now = now_utc or datetime.now(timezone.utc)
        options = [
            parsed
            for item in raw_summaries
            if (parsed := cls.parse_instrument(item, now, max_dte_days=max_dte_days)) is not None
        ]

        if len(options) < 10:
            options = [
                parsed
                for item in raw_summaries
                if (parsed := cls.parse_instrument(item, now, max_dte_days=35.0)) is not None
            ]

        if not options:
            return GEXProfile(
                net_gex_usd=0.0,
                gamma_flip_price=spot_price,
                call_wall=round(spot_price * 1.05, 1),
                put_wall=round(spot_price * 0.95, 1),
                active_contracts=0,
            )

        call_gex_by_strike: Dict[float, float] = {}
        put_gex_by_strike: Dict[float, float] = {}
        net_gex_now = 0.0
        spot_sq_1pct = spot_price * spot_price * 0.01

        for opt in options:
            gamma = cls.bs_gamma(spot_price, opt.strike, opt.t_years, opt.mark_iv)
            gex_abs = gamma * opt.open_interest * spot_sq_1pct
            if opt.option_type == "C":
                net_gex_now += gex_abs
                if spot_price * 0.98 <= opt.strike <= spot_price * 1.25:
                    call_gex_by_strike[opt.strike] = call_gex_by_strike.get(opt.strike, 0.0) + gex_abs
            else:
                net_gex_now -= gex_abs
                if spot_price * 0.75 <= opt.strike <= spot_price * 1.02:
                    put_gex_by_strike[opt.strike] = put_gex_by_strike.get(opt.strike, 0.0) + gex_abs

        call_wall = (
            max(call_gex_by_strike.items(), key=lambda kv: kv[1])[0]
            if call_gex_by_strike
            else round(spot_price * 1.05, 1)
        )
        put_wall = (
            max(put_gex_by_strike.items(), key=lambda kv: kv[1])[0]
            if put_gex_by_strike
            else round(spot_price * 0.95, 1)
        )

        low_grid = spot_price * 0.85
        high_grid = spot_price * 1.15
        steps = 60
        step_size = (high_grid - low_grid) / steps

        flip_candidates: List[float] = []
        prev_p = low_grid
        prev_gex = cls.compute_gex_at_spot(options, prev_p)

        for i in range(1, steps + 1):
            curr_p = low_grid + i * step_size
            curr_gex = cls.compute_gex_at_spot(options, curr_p)
            if (prev_gex <= 0.0 <= curr_gex) or (prev_gex >= 0.0 >= curr_gex):
                diff = curr_gex - prev_gex
                if abs(diff) > 1e-9:
                    ratio = -prev_gex / diff
                    flip_candidates.append(prev_p + ratio * step_size)
            prev_p, prev_gex = curr_p, curr_gex

        gamma_flip = (
            min(flip_candidates, key=lambda p: abs(p - spot_price))
            if flip_candidates
            else spot_price
        )

        return GEXProfile(
            net_gex_usd=round(net_gex_now, 2),
            gamma_flip_price=round(gamma_flip, 1),
            call_wall=round(call_wall, 1),
            put_wall=round(put_wall, 1),
            active_contracts=len(options),
        )


# =============================================================================
# 3. 美国直连双源容灾路由 (OKX Primary -> Deribit Perpetual Fallback)
# =============================================================================

@dataclass(frozen=True)
class NormalizedTechRaw:
    exchange: str
    spot_price: float
    low_24h: float
    high_24h: float
    klines_1h_asc: List[Tuple[int, float, float, float, float, float]]
    last_1h_close: float
    depth_1pct_usd: float


@dataclass(frozen=True)
class NormalizedDerivRaw:
    exchange: str
    mark_price: float
    funding_rate_8h: float
    oi_change_24h_pct: float
    oi_change_1h_pct: float


class ExchangeFailoverCircuit:
    def __init__(self, cooldown_sec: float = 900.0) -> None:
        self.cooldown_sec = cooldown_sec
        self._primary_blocked_until: float = 0.0
        self.last_failover_reason: Optional[str] = None

    def is_primary_open(self) -> bool:
        return time.monotonic() >= self._primary_blocked_until

    def trip_primary(self, reason: str) -> None:
        self._primary_blocked_until = time.monotonic() + self.cooldown_sec
        self.last_failover_reason = reason

    def reset_primary(self) -> None:
        self._primary_blocked_until = 0.0
        self.last_failover_reason = None

    @property
    def remaining_cooldown_sec(self) -> int:
        return max(0, int(self._primary_blocked_until - time.monotonic()))


GLOBAL_EXCHANGE_CIRCUIT = ExchangeFailoverCircuit(cooldown_sec=900.0)


# =============================================================================
# 4. 生产级异步 API 适配器 (V4: OKX + Deribit + 自动现货 ETF 净流入抓取)
# =============================================================================

class ProductionAPIAdapter(BaseMarketDataAdapter):
    """
    全异步公开接口实盘适配器 (V4)：
    - 主永续数据源: OKX V5 (BTC-USDT-SWAP) -> 备源 Deribit (BTC-PERPETUAL)
    - 期权波动率源: Deribit 公开期权 API (真实 DVOL + 期权链 GEX / Flip / Call&Put Wall)
    - 现货 ETF 源 : 自动瀑布流抓取 —— 优先 SoSoValue 免费官方 API (配置 SOSO_API_KEY)
                   -> 零密钥自动解析 Farside Investors 官方日度流量表
                   -> 支持网页端手动覆盖，留空自动恢复实时抓取
    - 舆情基准源  : Alternative.me 14日恐惧与贪婪指数
    """

    OKX_API = "https://www.okx.com"
    DERIBIT_API = "https://www.deribit.com/api/v2"
    ALT_FNG_API = "https://api.alternative.me/fng/"
    SOSO_ETF_API = "https://openapi.sosovalue.com/openapi/v1/etfs/summary-history"
    FARSIDE_ETF_URL = "https://farside.co.uk/btc/"

    FAILOVER_HTTP_CODES = {403, 418, 429, 451, 500, 502, 503, 504}

    def __init__(
        self,
        okx_inst_id: str = "BTC-USDT-SWAP",
        okx_ct_val_btc: float = 0.01,
        deribit_perp: str = "BTC-PERPETUAL",
        exchange_source: Optional[str] = None,
        circuit: Optional[ExchangeFailoverCircuit] = None,
        etf_cache: Optional[ETFFlowCache] = None,
        soso_api_key: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout_sec: float = 10.0,
        depth_baseline_usd: float = 28_000_000.0,
        manual_etf_netflow_usd: Optional[float] = None,  # None = 自动抓取；非 None = 手动覆盖
        manual_polymarket_bias: Optional[float] = None,
        manual_events: Optional[List[EventShock]] = None,
    ):
        self.okx_inst_id = okx_inst_id
        self.okx_ct_val_btc = okx_ct_val_btc
        self.deribit_perp = deribit_perp
        self.exchange_source = (exchange_source or os.getenv("EXCHANGE_SOURCE", "auto")).strip().lower()
        if self.exchange_source not in ("auto", "okx", "deribit"):
            self.exchange_source = "auto"

        self.circuit = circuit or GLOBAL_EXCHANGE_CIRCUIT
        self.etf_cache = etf_cache or GLOBAL_ETF_CACHE
        self.soso_api_key = (soso_api_key or os.getenv("SOSO_API_KEY") or "").strip() or None
        self.proxy = proxy or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        self.timeout_sec = timeout_sec
        self.depth_baseline_usd = depth_baseline_usd
        self.manual_etf_netflow_usd = manual_etf_netflow_usd
        self.manual_polymarket_bias = manual_polymarket_bias
        self.manual_events = manual_events or []

        self._session: Optional[Any] = None
        self._cached_spot: Optional[float] = None
        self._cached_24h_range: Tuple[float, float] = (0.0, 0.0)
        self._deribit_book_cache: Optional[Dict[str, Any]] = None
        self._etf_inflight: Optional[asyncio.Task[ETFFlowSnapshot]] = None

        # 实时遥测字段
        self.active_exchange: str = "DERIBIT" if self.exchange_source == "deribit" else "OKX"
        self.failover_reason: Optional[str] = self.circuit.last_failover_reason
        self.last_gex_profile: Optional[GEXProfile] = None
        self.strike_gex_bars: List[Dict[str, Any]] = []
        self.last_1h_close: float = 0.0
        self.oi_1h_change_pct: float = 0.0
        self.latest_oi_24h_pct: float = 0.0
        self.latest_funding_rate: float = 0.0
        self.latest_dvol: float = 0.0
        self.latest_avwap: float = 0.0
        self.latest_fng: int = 50
        self.latest_etf: ETFFlowSnapshot = ETFFlowSnapshot(
            netflow_usd=0.0, trade_date="--", source="Initializing", fetched_at_mono=0.0
        )

    async def __aenter__(self) -> ProductionAPIAdapter:
        if aiohttp is None:
            raise RuntimeError("请先安装 aiohttp: pip install aiohttp")
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
            headers={"User-Agent": "Mozilla/5.0 (compatible; CarbonQuant-BTC24H/4.0)"},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> Any:
        if aiohttp is None:
            raise RuntimeError("请先安装 aiohttp: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                headers={"User-Agent": "Mozilla/5.0 (compatible; CarbonQuant-BTC24H/4.0)"},
            )
        return self._session

    async def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        session = await self._ensure_session()
        async with session.get(url, params=params, headers=headers, proxy=self.proxy) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _get_text(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        session = await self._ensure_session()
        async with session.get(url, headers=headers, proxy=self.proxy) as resp:
            resp.raise_for_status()
            return await resp.text()

    # -------------------------------------------------------------------------
    # 美国现货 BTC ETF 净流入自动抓取逻辑 (Tier 1 SoSoValue -> Tier 2 Farside)
    # -------------------------------------------------------------------------

    async def _fetch_spot_etf_netflow(self) -> ETFFlowSnapshot:
        if self._etf_inflight is None:
            self._etf_inflight = asyncio.create_task(self._fetch_spot_etf_netflow_impl())
        return await self._etf_inflight

    async def _fetch_spot_etf_netflow_impl(self) -> ETFFlowSnapshot:
        # 1) 若用户在网页端显式填入了手动覆盖值，优先采用手动值
        if self.manual_etf_netflow_usd is not None:
            snap = ETFFlowSnapshot(
                netflow_usd=round(self.manual_etf_netflow_usd, 2),
                trade_date="手动覆盖",
                source="Manual Override",
                fetched_at_mono=time.monotonic(),
            )
            self.latest_etf = snap
            return snap

        # 2) 命中 30 分钟进程级缓存直接返回
        cached = self.etf_cache.get_fresh()
        if cached is not None:
            self.latest_etf = cached
            return cached

        errors: List[str] = []

        # 3) Tier 1: SoSoValue 官方免费 API (需在 .env 填入 SOSO_API_KEY，零反爬风险)
        if self.soso_api_key:
            try:
                payload = await self._get_json(
                    self.SOSO_ETF_API,
                    params={"symbol": "BTC", "country_code": "US", "limit": 7},
                    headers={"x-soso-api-key": self.soso_api_key},
                )
                snap = SpotETFFlowParser.parse_sosovalue_json(payload)
                if snap is not None:
                    self.etf_cache.put(snap)
                    self.latest_etf = snap
                    return snap
                errors.append("SoSoValue 未返回有效净流入记录")
            except Exception as exc:
                errors.append(f"SoSoValue 异常: {type(exc).__name__}")

        # 4) Tier 2: Farside Investors 官方页面零密钥自动解析
        try:
            html = await self._get_text(
                self.FARSIDE_ETF_URL,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/130.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            snap = SpotETFFlowParser.parse_farside_html(html)
            if snap is not None:
                self.etf_cache.put(snap)
                self.latest_etf = snap
                return snap
            errors.append("Farside 页面解析无有效数值")
        except Exception as exc:
            errors.append(f"Farside 抓取受限: {type(exc).__name__}")

        # 5) 兜底：若存在过期旧缓存则降级沿用，避免突然归零
        if self.etf_cache.cached is not None:
            self.etf_cache.last_error = "; ".join(errors)
            self.latest_etf = self.etf_cache.cached
            return self.etf_cache.cached

        self.etf_cache.last_error = "; ".join(errors) or "未获取到 ETF 数据"
        fallback = ETFFlowSnapshot(
            netflow_usd=0.0,
            trade_date="--",
            source="Auto Failed (建议配置免费 SOSO_API_KEY)",
            fetched_at_mono=time.monotonic(),
        )
        self.latest_etf = fallback
        return fallback

    # -------------------------------------------------------------------------
    # OKX & Deribit 永续合约抓取与路由
    # -------------------------------------------------------------------------

    @classmethod
    def _classify_failover_error(cls, exc: Exception) -> Optional[str]:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return "OKX 请求超时 (Timeout)"
        if aiohttp is not None and isinstance(exc, aiohttp.ClientResponseError):
            if exc.status in cls.FAILOVER_HTTP_CODES:
                return f"OKX HTTP {exc.status} ({exc.message or 'Error'})"
        if aiohttp is not None and isinstance(exc, aiohttp.ClientError):
            return f"OKX 网络连接异常 ({type(exc).__name__})"
        status = getattr(exc, "status", None)
        if isinstance(status, int) and status in cls.FAILOVER_HTTP_CODES:
            return f"OKX HTTP {status}"
        if isinstance(exc, RuntimeError) and str(exc).startswith("OKX "):
            return str(exc)
        return None

    @staticmethod
    def _calc_anchored_vwap(
        klines_asc: List[Tuple[int, float, float, float, float, float]]
    ) -> float:
        if not klines_asc:
            return 0.0
        anchor_idx = max(range(len(klines_asc)), key=lambda i: klines_asc[i][5])
        pv_sum = 0.0
        vol_sum = 0.0
        for bar in klines_asc[anchor_idx:]:
            typical = (bar[2] + bar[3] + bar[4]) / 3.0
            vol = bar[5]
            pv_sum += typical * vol
            vol_sum += vol
        return round(pv_sum / vol_sum, 1) if vol_sum > 0 else round(klines_asc[-1][4], 1)

    @staticmethod
    def _calc_okx_depth_1pct_usd(
        bids: List[List[Any]],
        asks: List[List[Any]],
        spot: float,
        ct_val_btc: float,
    ) -> float:
        lower_bound = spot * 0.99
        upper_bound = spot * 1.01
        bid_usd = sum(
            float(row[0]) * float(row[1]) * ct_val_btc
            for row in bids
            if float(row[0]) >= lower_bound
        )
        ask_usd = sum(
            float(row[0]) * float(row[1]) * ct_val_btc
            for row in asks
            if float(row[0]) <= upper_bound
        )
        return bid_usd + ask_usd

    @staticmethod
    def _calc_deribit_depth_1pct_usd(
        bids: List[List[Any]],
        asks: List[List[Any]],
        spot: float,
    ) -> float:
        lower_bound = spot * 0.99
        upper_bound = spot * 1.01
        bid_usd = sum(float(row[1]) for row in bids if float(row[0]) >= lower_bound)
        ask_usd = sum(float(row[1]) for row in asks if float(row[0]) <= upper_bound)
        return bid_usd + ask_usd

    @staticmethod
    def _unwrap_okx_data(payload: Dict[str, Any], endpoint_name: str) -> List[Any]:
        code = str(payload.get("code", ""))
        if code != "0":
            raise RuntimeError(f"OKX {endpoint_name} 返回异常 code={code}: {payload.get('msg')}")
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"OKX {endpoint_name} 返回空数据")
        return data

    async def _fetch_okx_technical_raw(self) -> NormalizedTechRaw:
        ticker_req = self._get_json(
            f"{self.OKX_API}/api/v5/market/ticker",
            params={"instId": self.okx_inst_id},
        )
        candles_req = self._get_json(
            f"{self.OKX_API}/api/v5/market/candles",
            params={"instId": self.okx_inst_id, "bar": "1H", "limit": "48"},
        )
        books_req = self._get_json(
            f"{self.OKX_API}/api/v5/market/books",
            params={"instId": self.okx_inst_id, "sz": "400"},
        )
        ticker_resp, candles_resp, books_resp = await asyncio.gather(
            ticker_req, candles_req, books_req
        )

        ticker = self._unwrap_okx_data(ticker_resp, "market/ticker")[0]
        candles_desc = self._unwrap_okx_data(candles_resp, "market/candles")
        book = self._unwrap_okx_data(books_resp, "market/books")[0]

        spot = float(ticker["last"])
        low_24h = float(ticker["low24h"])
        high_24h = float(ticker["high24h"])

        last_1h_close = spot
        for row in candles_desc:
            if len(row) >= 9 and str(row[8]) == "1":
                last_1h_close = float(row[4])
                break
        else:
            if len(candles_desc) >= 2:
                last_1h_close = float(candles_desc[1][4])

        klines_asc: List[Tuple[int, float, float, float, float, float]] = []
        for row in reversed(candles_desc):
            vol_btc = float(row[6]) if len(row) > 6 and row[6] else float(row[5]) * self.okx_ct_val_btc
            klines_asc.append(
                (int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), vol_btc)
            )

        depth_usd = self._calc_okx_depth_1pct_usd(
            book.get("bids", []),
            book.get("asks", []),
            spot=spot,
            ct_val_btc=self.okx_ct_val_btc,
        )

        return NormalizedTechRaw(
            exchange="OKX",
            spot_price=spot,
            low_24h=low_24h,
            high_24h=high_24h,
            klines_1h_asc=klines_asc,
            last_1h_close=last_1h_close,
            depth_1pct_usd=depth_usd,
        )

    async def _fetch_okx_derivatives_raw(self) -> NormalizedDerivRaw:
        funding_req = self._get_json(
            f"{self.OKX_API}/api/v5/public/funding-rate",
            params={"instId": self.okx_inst_id},
        )
        mark_req = self._get_json(
            f"{self.OKX_API}/api/v5/public/mark-price",
            params={"instType": "SWAP", "instId": self.okx_inst_id},
        )
        oi_hist_req = self._get_json(
            f"{self.OKX_API}/api/v5/rubik/stat/contracts/open-interest-history",
            params={"instId": self.okx_inst_id, "period": "1H", "limit": "25"},
        )

        funding_resp, mark_resp, oi_resp = await asyncio.gather(
            funding_req, mark_req, oi_hist_req, return_exceptions=True
        )

        if isinstance(funding_resp, Exception):
            raise funding_resp
        funding_row = self._unwrap_okx_data(funding_resp, "public/funding-rate")[0]
        funding_rate = float(funding_row["fundingRate"])

        if isinstance(mark_resp, Exception):
            mark_price = self._cached_spot or 0.0
        else:
            mark_row = self._unwrap_okx_data(mark_resp, "public/mark-price")[0]
            mark_price = float(mark_row["markPx"])

        oi_change_24h = 0.0
        oi_change_1h = 0.0
        if not isinstance(oi_resp, Exception):
            try:
                oi_rows = self._unwrap_okx_data(oi_resp, "rubik/open-interest-history")
                if len(oi_rows) >= 2:
                    col_idx = 3 if len(oi_rows[0]) > 3 and oi_rows[0][3] else (2 if len(oi_rows[0]) > 2 else 1)
                    oi_new = float(oi_rows[0][col_idx])
                    oi_prev_1h = float(oi_rows[1][col_idx])
                    oi_old_24h = float(oi_rows[-1][col_idx])
                    if oi_old_24h > 0:
                        oi_change_24h = round(((oi_new - oi_old_24h) / oi_old_24h) * 100.0, 2)
                    if oi_prev_1h > 0:
                        oi_change_1h = round(((oi_new - oi_prev_1h) / oi_prev_1h) * 100.0, 2)
            except Exception:
                oi_change_24h, oi_change_1h = 0.0, 0.0

        return NormalizedDerivRaw(
            exchange="OKX",
            mark_price=mark_price,
            funding_rate_8h=funding_rate,
            oi_change_24h_pct=oi_change_24h,
            oi_change_1h_pct=oi_change_1h,
        )

    async def _fetch_deribit_perp_book(self) -> Dict[str, Any]:
        if self._deribit_book_cache is not None:
            return self._deribit_book_cache
        payload = await self._get_json(
            f"{self.DERIBIT_API}/public/get_order_book",
            params={"instrument_name": self.deribit_perp, "depth": 200},
        )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Deribit get_order_book 返回异常")
        self._deribit_book_cache = result
        return result

    async def _fetch_deribit_technical_raw(self) -> NormalizedTechRaw:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - 49 * 3600 * 1000

        book_req = self._fetch_deribit_perp_book()
        chart_req = self._get_json(
            f"{self.DERIBIT_API}/public/get_tradingview_chart_data",
            params={
                "instrument_name": self.deribit_perp,
                "start_timestamp": start_ms,
                "end_timestamp": now_ms,
                "resolution": "60",
            },
        )
        book, chart_resp = await asyncio.gather(book_req, chart_req)

        spot = float(book.get("last_price") or book.get("mark_price") or 0.0)
        stats = book.get("stats") or {}
        low_24h = float(stats.get("low") or spot)
        high_24h = float(stats.get("high") or spot)

        chart = chart_resp.get("result") or {}
        ticks = chart.get("ticks") or []
        opens = chart.get("open") or []
        highs = chart.get("high") or []
        lows = chart.get("low") or []
        closes = chart.get("close") or []
        vols = chart.get("volume") or []

        n = min(len(ticks), len(opens), len(highs), len(lows), len(closes), len(vols))
        klines_asc: List[Tuple[int, float, float, float, float, float]] = [
            (int(ticks[i]), float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i]), float(vols[i]))
            for i in range(n)
        ]

        last_1h_close = klines_asc[-2][4] if len(klines_asc) >= 2 else spot
        depth_usd = self._calc_deribit_depth_1pct_usd(
            book.get("bids", []),
            book.get("asks", []),
            spot=spot,
        )

        return NormalizedTechRaw(
            exchange="DERIBIT",
            spot_price=spot,
            low_24h=low_24h,
            high_24h=high_24h,
            klines_1h_asc=klines_asc,
            last_1h_close=last_1h_close,
            depth_1pct_usd=depth_usd,
        )

    async def _fetch_deribit_derivatives_raw(self) -> NormalizedDerivRaw:
        book = await self._fetch_deribit_perp_book()
        mark_price = float(book.get("mark_price") or book.get("last_price") or 0.0)
        funding_8h = float(book.get("funding_8h") or 0.0)
        return NormalizedDerivRaw(
            exchange="DERIBIT",
            mark_price=mark_price,
            funding_rate_8h=funding_8h,
            oi_change_24h_pct=0.0,
            oi_change_1h_pct=0.0,
        )

    async def _route_exchange_technical(self) -> NormalizedTechRaw:
        if self.exchange_source == "deribit":
            self.active_exchange = "DERIBIT"
            self.failover_reason = "手动指定 EXCHANGE_SOURCE=deribit"
            return await self._fetch_deribit_technical_raw()

        if self.exchange_source == "okx":
            self.active_exchange = "OKX"
            self.failover_reason = None
            return await self._fetch_okx_technical_raw()

        if not self.circuit.is_primary_open():
            self.active_exchange = "DERIBIT"
            self.failover_reason = self.circuit.last_failover_reason
            return await self._fetch_deribit_technical_raw()

        try:
            raw = await self._fetch_okx_technical_raw()
            self.circuit.reset_primary()
            self.active_exchange = "OKX"
            self.failover_reason = None
            return raw
        except Exception as exc:
            reason = self._classify_failover_error(exc)
            if reason is None:
                raise
            self.circuit.trip_primary(reason)
            self.active_exchange = "DERIBIT"
            self.failover_reason = f"{reason} -> 已自动切换 Deribit BTC-PERPETUAL"
            return await self._fetch_deribit_technical_raw()

    async def _route_exchange_derivatives(self) -> NormalizedDerivRaw:
        if self.exchange_source == "deribit":
            self.active_exchange = "DERIBIT"
            return await self._fetch_deribit_derivatives_raw()

        if self.exchange_source == "okx":
            self.active_exchange = "OKX"
            return await self._fetch_okx_derivatives_raw()

        if not self.circuit.is_primary_open():
            self.active_exchange = "DERIBIT"
            self.failover_reason = self.circuit.last_failover_reason
            return await self._fetch_deribit_derivatives_raw()

        try:
            raw = await self._fetch_okx_derivatives_raw()
            self.circuit.reset_primary()
            self.active_exchange = "OKX"
            return raw
        except Exception as exc:
            reason = self._classify_failover_error(exc)
            if reason is None:
                raise
            self.circuit.trip_primary(reason)
            self.active_exchange = "DERIBIT"
            self.failover_reason = f"{reason} -> 已自动切换 Deribit BTC-PERPETUAL"
            return await self._fetch_deribit_derivatives_raw()

    # -------------------------------------------------------------------------
    # BaseMarketDataAdapter 标准接口实现
    # -------------------------------------------------------------------------

    async def fetch_technical(self) -> TechnicalSnapshot:
        raw, etf_snap = await asyncio.gather(
            self._route_exchange_technical(),
            self._fetch_spot_etf_netflow(),
        )
        self._cached_spot = raw.spot_price
        self._cached_24h_range = (raw.low_24h, raw.high_24h)
        self.last_1h_close = raw.last_1h_close

        avwap = self._calc_anchored_vwap(raw.klines_1h_asc)
        self.latest_avwap = avwap
        depth_ratio = round(raw.depth_1pct_usd / self.depth_baseline_usd, 3)

        return TechnicalSnapshot(
            spot_price=raw.spot_price,
            avwap_anchor=avwap,
            recent_24h_low=raw.low_24h,
            recent_24h_high=raw.high_24h,
            spot_etf_netflow_usd=etf_snap.netflow_usd,
            orderbook_depth_ratio=depth_ratio,
            timestamp=datetime.now(timezone.utc),
        )

    async def _fetch_deribit_dvol(self) -> float:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - 12 * 3600 * 1000
        payload = await self._get_json(
            f"{self.DERIBIT_API}/public/get_volatility_index_data",
            params={
                "currency": "BTC",
                "start_timestamp": start_ms,
                "end_timestamp": now_ms,
                "resolution": "3600",
            },
        )
        candles = payload.get("result", {}).get("data", [])
        if not candles:
            return 0.60
        latest_close = float(candles[-1][4])
        return round(latest_close / 100.0 if latest_close > 3.0 else latest_close, 4)

    async def _fetch_liquidation_clusters(
        self,
        spot: float,
        dvol: float,
        funding_rate: float,
    ) -> Tuple[float, float, float]:
        em_1d = spot * (dvol / math.sqrt(365.0))
        low_24h, high_24h = self._cached_24h_range

        skew = 0.85 if funding_rate < 0 else 1.15
        upper_liq = max(high_24h * 1.008, spot + em_1d * 0.95 * skew)
        lower_liq = min(
            low_24h * 1.015 if low_24h > 0 else spot - em_1d * 1.05,
            spot - em_1d * (1.10 / skew),
        )
        range_pct = (high_24h - low_24h) / spot if spot > 0 and high_24h > low_24h else 0.03
        est_liq_usd = round(max(80_000_000.0, (range_pct / 0.05) * 450_000_000.0), 0)
        return round(upper_liq, 1), round(lower_liq, 1), est_liq_usd

    async def fetch_derivatives(self) -> DerivativesSnapshot:
        exchange_req = self._route_exchange_derivatives()
        dvol_req = self._fetch_deribit_dvol()
        opt_chain_req = self._get_json(
            f"{self.DERIBIT_API}/public/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "option"},
        )

        deriv_raw, dvol, opt_resp = await asyncio.gather(
            exchange_req, dvol_req, opt_chain_req
        )

        spot = self._cached_spot or deriv_raw.mark_price
        self.latest_funding_rate = deriv_raw.funding_rate_8h
        self.latest_dvol = dvol
        self.latest_oi_24h_pct = deriv_raw.oi_change_24h_pct
        self.oi_1h_change_pct = deriv_raw.oi_change_1h_pct

        raw_options = opt_resp.get("result", [])
        gex_profile = DeribitGEXCalculator.analyze_chain(
            raw_summaries=raw_options,
            spot_price=spot,
            max_dte_days=14.0,
        )
        self.last_gex_profile = gex_profile
        self.strike_gex_bars = DeribitGEXCalculator.build_strike_bars(
            raw_summaries=raw_options,
            spot_price=spot,
            max_dte_days=14.0,
        )

        upper_liq, lower_liq, liq_24h_usd = await self._fetch_liquidation_clusters(
            spot=spot, dvol=dvol, funding_rate=deriv_raw.funding_rate_8h
        )

        return DerivativesSnapshot(
            dvol_annualized=dvol,
            net_gex_usd=gex_profile.net_gex_usd,
            gamma_flip_price=gex_profile.gamma_flip_price,
            call_wall=gex_profile.call_wall,
            put_wall=gex_profile.put_wall,
            upper_liq_cluster=upper_liq,
            lower_liq_cluster=lower_liq,
            liq_24h_total_usd=liq_24h_usd,
            funding_rate_8h=deriv_raw.funding_rate_8h,
            oi_change_24h_pct=deriv_raw.oi_change_24h_pct,
        )

    async def fetch_sentiment(self) -> SentimentSnapshot:
        try:
            fng_resp = await self._get_json(self.ALT_FNG_API, params={"limit": 14})
            values = [int(item["value"]) for item in fng_resp.get("data", [])]
        except Exception:
            values = [50]

        latest_fng = values[0] if values else 50
        self.latest_fng = latest_fng

        if len(values) >= 5:
            mean_v = sum(values) / len(values)
            variance = sum((v - mean_v) ** 2 for v in values) / len(values)
            std_v = max(math.sqrt(variance), 5.0)
            z_score = round((latest_fng - mean_v) / std_v, 2)
        else:
            z_score = round((latest_fng - 50.0) / 15.0, 2)

        if latest_fng <= 15:
            z_score = min(z_score, -2.55)
        elif latest_fng >= 85:
            z_score = max(z_score, 2.55)

        poly_bias = (
            self.manual_polymarket_bias
            if self.manual_polymarket_bias is not None
            else round((latest_fng - 50.0) / 50.0 * 0.6, 3)
        )

        # 复用单次并发 ETF 抓取结果，消除冷启动竞态，并将 ±$100M 以上流向自动映射为事件冲击因子
        etf_snap = await self._fetch_spot_etf_netflow()
        events = list(self.manual_events)
        if not events and abs(etf_snap.netflow_usd) >= 100_000_000.0:
            etf_m = etf_snap.netflow_usd / 1_000_000.0
            etf_score = max(-0.85, min(0.85, etf_m / 300.0))
            events.append(
                EventShock(
                    title=f"US Spot BTC ETF ({etf_snap.trade_date}) {etf_m:+.1f}M",
                    score=round(etf_score, 3),
                    hours_ago=3.0,
                    half_life_hours=4.0,
                )
            )

        return SentimentSnapshot(
            polymarket_bias=poly_bias,
            social_zscore_24h=z_score,
            fear_greed_index=latest_fng,
            events=events,
        )


async def run_live() -> None:
    async with ProductionAPIAdapter(
        okx_inst_id="BTC-USDT-SWAP",
        deribit_perp="BTC-PERPETUAL",
        exchange_source=os.getenv("EXCHANGE_SOURCE", "auto"),
        proxy=os.getenv("HTTPS_PROXY"),
    ) as adapter:
        pipeline = BTC24HPredictorPipeline(adapter=adapter)
        result = await pipeline.run_once()
        gex = adapter.last_gex_profile
        etf = adapter.latest_etf
        print("=" * 76)
        print(f"BTC 24H 实时预测系统 [LIVE V4] | UTC: {result.timestamp}")
        print(f"永续源: {adapter.active_exchange} | 现货 ETF: {etf.netflow_usd/1e6:+.1f}M ({etf.trade_date} · {etf.source})")
        print(f"现价: ${result.spot_price:,.1f} | EM: ±${result.em_24h:,.1f} | 1H收盘: ${adapter.last_1h_close:,.1f}")
        if gex:
            print(
                f"Deribit ({gex.active_contracts} 合约): "
                f"Net GEX=${gex.net_gex_usd/1e6:+.2f}M | Flip=${gex.gamma_flip_price:,.1f} | "
                f"PW=${gex.put_wall:,.1f} | CW=${gex.call_wall:,.1f}"
            )
        print(f"状态路由: {result.primary_regime.value}")
        print(f"极端轧空上限: ${result.wick_ceiling_zone[0]:,.1f} ~ ${result.wick_ceiling_zone[1]:,.1f}")
        print(f"预期高点磁吸: ${result.core_high_zone[0]:,.1f} ~ ${result.core_high_zone[1]:,.1f}")
        print(f"预期低点磁吸: ${result.core_low_zone[0]:,.1f} ~ ${result.core_low_zone[1]:,.1f}")
        print(f"极端插针底线: ${result.wick_floor_zone[0]:,.1f} ~ ${result.wick_floor_zone[1]:,.1f}")
        print("=" * 76)


if __name__ == "__main__":
    asyncio.run(run_live())
