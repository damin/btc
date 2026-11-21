from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 1. 领域数据契约层 (Domain Data Contracts)
# =============================================================================

class MarketRegime(str, Enum):
    Q1_POSITIVE_GAMMA = "Q1_正Gamma均值回归"
    Q2_EXTREME_EXHAUSTION = "Q2_极端情绪背离插针V反"
    Q3_NEGATIVE_GAMMA_TREND = "Q3_负Gamma单边共振"
    Q4_LOW_LIQUIDITY_VACUUM = "Q4_低流动性周末真空"


@dataclass(frozen=True)
class EventShock:
    """突发宏观/ETF/黑天鹅事件冲击"""
    title: str
    score: float               # [-1.0, +1.0], 负值偏空，正值偏多
    hours_ago: float           # 距离事件发生已过去的小时数
    half_life_hours: float = 3.0


@dataclass(frozen=True)
class SentimentSnapshot:
    """舆情与预测市场原始快照"""
    polymarket_bias: float     # [-1.0, +1.0] 预测市场隐含方向偏移
    social_zscore_24h: float   # 社交媒体情绪 24H 滚动 Z-Score
    fear_greed_index: int      # 0 - 100
    events: List[EventShock] = field(default_factory=list)


@dataclass(frozen=True)
class DerivativesSnapshot:
    """衍生品微观结构快照"""
    dvol_annualized: float     # Deribit DVOL 年化隐含波动率 (如 0.62 代表 62%)
    net_gex_usd: float         # 净 Gamma Exposure (正值=正Gamma, 负值=负Gamma)
    gamma_flip_price: float    # 正负 Gamma 翻转位
    call_wall: float           # 近期最大看涨期权持仓执行价
    put_wall: float            # 近期最大看跌期权持仓执行价
    upper_liq_cluster: float   # 上方密集空头清算磁吸位
    lower_liq_cluster: float   # 下方密集多头清算磁吸位
    liq_24h_total_usd: float   # 过去 24H 全网爆仓总额
    funding_rate_8h: float     # 资金费率 (如 -0.0007)
    oi_change_24h_pct: float   # 24H 未平仓合约变化率 (%)


@dataclass(frozen=True)
class TechnicalSnapshot:
    """现货/永续技术面与流动性快照"""
    spot_price: float
    avwap_anchor: float        # 锚定在最近大清算极值或周线开盘的 AVWAP
    recent_24h_low: float      # 过去 24H 最低插针价
    recent_24h_high: float     # 过去 24H 最高价
    spot_etf_netflow_usd: float
    orderbook_depth_ratio: float  # 当前 ±1% 盘口深度 / 30日均值 (低于 0.75 为薄盘)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class MarketBundle:
    """统一喂给预测引擎的完整切片"""
    technical: TechnicalSnapshot
    derivatives: DerivativesSnapshot
    sentiment: SentimentSnapshot


@dataclass(frozen=True)
class PredictionOutput:
    """24H 极值预测结构化输出"""
    timestamp: str
    spot_price: float
    primary_regime: MarketRegime
    weekend_overlay: bool
    em_24h: float
    raw_sentiment: float
    adjusted_sentiment: float
    core_low_zone: Tuple[float, float]
    wick_floor_zone: Tuple[float, float]
    core_high_zone: Tuple[float, float]
    wick_ceiling_zone: Tuple[float, float]
    invalidation_bull_trigger: float
    invalidation_bear_trigger: float
    diagnostics: Dict[str, str]


# =============================================================================
# 2. 异步数据采集接口层 (Data Ingestion Adapters)
# =============================================================================

class BaseMarketDataAdapter(ABC):
    """统一数据源抽象接口：实盘挂载 OKX / Deribit / SoSoValue / Farside 异步公开接口"""

    @abstractmethod
    async def fetch_technical(self) -> TechnicalSnapshot:
        """对接 OKX / Deribit 永续合约 K 线、深度盘口及自动抓取的美国现货 ETF 净流入"""
        raise NotImplementedError

    @abstractmethod
    async def fetch_derivatives(self) -> DerivativesSnapshot:
        """对接 Deribit 期权链 (GEX/DVOL) + 永续资金费率与 OI 变化"""
        raise NotImplementedError

    @abstractmethod
    async def fetch_sentiment(self) -> SentimentSnapshot:
        """对接恐惧与贪婪指数 + Polymarket 偏移与突发事件流"""
        raise NotImplementedError

    async def fetch_bundle(self) -> MarketBundle:
        """并发拉取三路数据，压缩端到端延迟"""
        tech, deriv, senti = await asyncio.gather(
            self.fetch_technical(),
            self.fetch_derivatives(),
            self.fetch_sentiment(),
        )
        return MarketBundle(technical=tech, derivatives=deriv, sentiment=senti)


class DemoSnapshotAdapter(BaseMarketDataAdapter):
    """内置 2026-11-21 实战快照，用于本地验证与回测调试"""

    async def fetch_technical(self) -> TechnicalSnapshot:
        return TechnicalSnapshot(
            spot_price=60270.0,
            avwap_anchor=61850.0,
            recent_24h_low=55130.0,
            recent_24h_high=60996.0,
            spot_etf_netflow_usd=-124_500_000.0,
            orderbook_depth_ratio=0.68,
            timestamp=datetime(2026, 11, 21, 20, 45, tzinfo=timezone.utc),
        )

    async def fetch_derivatives(self) -> DerivativesSnapshot:
        return DerivativesSnapshot(
            dvol_annualized=0.62,
            net_gex_usd=-38_000_000.0,
            gamma_flip_price=61500.0,
            call_wall=63000.0,
            put_wall=57500.0,
            upper_liq_cluster=62600.0,
            lower_liq_cluster=57800.0,
            liq_24h_total_usd=783_000_000.0,
            funding_rate_8h=-0.0007,
            oi_change_24h_pct=-2.48,
        )

    async def fetch_sentiment(self) -> SentimentSnapshot:
        return SentimentSnapshot(
            polymarket_bias=-0.28,
            social_zscore_24h=-2.75,
            fear_greed_index=6,
            events=[
                EventShock(
                    title="US Spot BTC ETF 单日净流出 1.245 亿美元",
                    score=-0.55,
                    hours_ago=5.0,
                    half_life_hours=3.0,
                )
            ],
        )


# =============================================================================
# 3. 舆情反身性与时间衰减计算引擎 (Sentiment Engine)
# =============================================================================

class SentimentReflexivityEngine:
    """
    将三路舆情合成为标准化得分 S_raw，并检测极端情绪背离以输出 S_adj
    权重: 预测市场赔率 45% | 社媒加速度 35% | 突发事件衰减 20%
    """

    W_POLYMARKET = 0.45
    W_SOCIAL = 0.35
    W_EVENT = 0.20

    @staticmethod
    def _clip(val: float, low: float = -1.0, high: float = 1.0) -> float:
        return max(low, min(high, val))

    def compute_event_impact(self, events: List[EventShock]) -> float:
        total = 0.0
        for ev in events:
            if ev.hours_ago > 6.0 or ev.half_life_hours <= 0:
                continue
            decay_lambda = math.log(2.0) / ev.half_life_hours
            total += ev.score * math.exp(-decay_lambda * ev.hours_ago)
        return self._clip(total)

    def evaluate(self, bundle: MarketBundle) -> Tuple[float, float, bool]:
        senti = bundle.sentiment
        tech = bundle.technical
        deriv = bundle.derivatives

        social_norm = math.tanh(senti.social_zscore_24h / 2.0)
        event_score = self.compute_event_impact(senti.events)

        s_raw = self._clip(
            self.W_POLYMARKET * senti.polymarket_bias
            + self.W_SOCIAL * social_norm
            + self.W_EVENT * event_score
        )

        exhaustion_triggered = False
        s_adj = s_raw

        bearish_exhaustion = (
            senti.social_zscore_24h <= -2.5
            and senti.fear_greed_index <= 15
            and tech.spot_price > tech.recent_24h_low * 1.03
            and (deriv.liq_24h_total_usd >= 400_000_000 or deriv.oi_change_24h_pct < -2.0)
        )
        bullish_exhaustion = (
            senti.social_zscore_24h >= 2.5
            and senti.fear_greed_index >= 85
            and tech.spot_price < tech.recent_24h_high * 0.97
            and deriv.funding_rate_8h > 0.0008
        )

        if bearish_exhaustion:
            exhaustion_triggered = True
            squeeze_boost = 0.18 if deriv.funding_rate_8h < 0 else 0.08
            s_adj = self._clip(-0.45 * s_raw + squeeze_boost)
        elif bullish_exhaustion:
            exhaustion_triggered = True
            s_adj = self._clip(-0.45 * s_raw - 0.12)

        return round(s_raw, 4), round(s_adj, 4), exhaustion_triggered


# =============================================================================
# 4. 四象限状态路由器 (Four-Quadrant Regime Router)
# =============================================================================

@dataclass(frozen=True)
class RegimeWeights:
    w_stat_em: float
    w_gamma_wall: float
    w_liq_cluster: float
    em_multiplier: float
    wick_buffer_mult: float
    alpha_tilt: float


class FourQuadrantRouter:
    """根据 Gamma 结构、舆情极值与流动性状态路由至对应权重模板"""

    WEIGHT_TABLE: Dict[MarketRegime, RegimeWeights] = {
        MarketRegime.Q1_POSITIVE_GAMMA: RegimeWeights(
            w_stat_em=0.35, w_gamma_wall=0.45, w_liq_cluster=0.20,
            em_multiplier=0.80, wick_buffer_mult=0.45, alpha_tilt=0.25,
        ),
        MarketRegime.Q2_EXTREME_EXHAUSTION: RegimeWeights(
            w_stat_em=0.35, w_gamma_wall=0.15, w_liq_cluster=0.50,
            em_multiplier=1.00, wick_buffer_mult=0.75, alpha_tilt=0.35,
        ),
        MarketRegime.Q3_NEGATIVE_GAMMA_TREND: RegimeWeights(
            w_stat_em=0.50, w_gamma_wall=0.15, w_liq_cluster=0.35,
            em_multiplier=1.60, wick_buffer_mult=0.90, alpha_tilt=0.50,
        ),
        MarketRegime.Q4_LOW_LIQUIDITY_VACUUM: RegimeWeights(
            w_stat_em=0.40, w_gamma_wall=0.25, w_liq_cluster=0.35,
            em_multiplier=0.80, wick_buffer_mult=1.15, alpha_tilt=0.30,
        ),
    }

    def route(
        self,
        bundle: MarketBundle,
        s_raw: float,
        exhaustion_triggered: bool,
    ) -> Tuple[MarketRegime, bool, RegimeWeights]:
        tech = bundle.technical
        deriv = bundle.derivatives

        is_weekend_thin = (
            tech.timestamp.weekday() in (5, 6)
            and tech.orderbook_depth_ratio < 0.75
        )

        if exhaustion_triggered:
            regime = MarketRegime.Q2_EXTREME_EXHAUSTION
        elif (
            deriv.net_gex_usd < 0
            and tech.spot_price < deriv.gamma_flip_price
            and abs(s_raw) >= 0.45
            and abs(tech.spot_etf_netflow_usd) >= 100_000_000
        ):
            regime = MarketRegime.Q3_NEGATIVE_GAMMA_TREND
        elif deriv.net_gex_usd >= 0 and abs(s_raw) < 0.40:
            regime = MarketRegime.Q1_POSITIVE_GAMMA
        elif is_weekend_thin:
            regime = MarketRegime.Q4_LOW_LIQUIDITY_VACUUM
        else:
            regime = MarketRegime.Q3_NEGATIVE_GAMMA_TREND if deriv.net_gex_usd < 0 else MarketRegime.Q1_POSITIVE_GAMMA

        base_w = self.WEIGHT_TABLE[regime]

        if is_weekend_thin and regime != MarketRegime.Q4_LOW_LIQUIDITY_VACUUM:
            adjusted_w = RegimeWeights(
                w_stat_em=base_w.w_stat_em,
                w_gamma_wall=base_w.w_gamma_wall,
                w_liq_cluster=base_w.w_liq_cluster,
                em_multiplier=round(base_w.em_multiplier * 0.90, 3),
                wick_buffer_mult=round(base_w.wick_buffer_mult * 1.30, 3),
                alpha_tilt=base_w.alpha_tilt,
            )
            return regime, True, adjusted_w

        return regime, is_weekend_thin, base_w


# =============================================================================
# 5. 24H 极值磁吸带合成与失效熔断引擎 (Extreme Boundary Synthesizer)
# =============================================================================

class ExtremeBoundaryEngine:
    """合成 24H 核心磁吸区间与插针极限缓冲带，并提供结构失效熔断判定"""

    def synthesize(
        self,
        bundle: MarketBundle,
        regime: MarketRegime,
        weekend_overlay: bool,
        weights: RegimeWeights,
        s_raw: float,
        s_adj: float,
    ) -> PredictionOutput:
        tech = bundle.technical
        deriv = bundle.derivatives
        p0 = tech.spot_price

        em_24h = p0 * (deriv.dvol_annualized / math.sqrt(365.0))
        scaled_em = em_24h * weights.em_multiplier

        stat_high = p0 + scaled_em * (1.0 + weights.alpha_tilt * s_adj)
        stat_low = p0 - scaled_em * (1.0 - weights.alpha_tilt * s_adj)

        h_core = (
            weights.w_stat_em * stat_high
            + weights.w_gamma_wall * deriv.call_wall
            + weights.w_liq_cluster * deriv.upper_liq_cluster
        )
        l_core = (
            weights.w_stat_em * stat_low
            + weights.w_gamma_wall * deriv.put_wall
            + weights.w_liq_cluster * deriv.lower_liq_cluster
        )

        magnet_half_width = 0.20 * em_24h
        wick_extension = weights.wick_buffer_mult * em_24h

        core_high_zone = (round(h_core - magnet_half_width, 1), round(h_core + magnet_half_width, 1))
        core_low_zone = (round(l_core - magnet_half_width, 1), round(l_core + magnet_half_width, 1))

        wick_floor_outer = min(l_core - wick_extension, tech.recent_24h_low)
        wick_floor_inner = max(wick_floor_outer + 0.35 * em_24h, l_core - wick_extension * 0.85)
        wick_floor_zone = (round(wick_floor_outer, 1), round(min(wick_floor_inner, core_low_zone[0]), 1))

        wick_ceil_inner = h_core + wick_extension * 0.75
        wick_ceil_outer = h_core + wick_extension * 1.05
        wick_ceiling_zone = (round(wick_ceil_inner, 1), round(wick_ceil_outer, 1))

        inv_bull = round(wick_ceiling_zone[1] * 1.005, 1)
        inv_bear = round(wick_floor_zone[0] * 0.995, 1)

        return PredictionOutput(
            timestamp=tech.timestamp.isoformat(),
            spot_price=p0,
            primary_regime=regime,
            weekend_overlay=weekend_overlay,
            em_24h=round(em_24h, 2),
            raw_sentiment=s_raw,
            adjusted_sentiment=s_adj,
            core_low_zone=core_low_zone,
            wick_floor_zone=wick_floor_zone,
            core_high_zone=core_high_zone,
            wick_ceiling_zone=wick_ceiling_zone,
            invalidation_bull_trigger=inv_bull,
            invalidation_bear_trigger=inv_bear,
            diagnostics={
                "weights": f"EM={weights.w_stat_em}, GammaWall={weights.w_gamma_wall}, Liq={weights.w_liq_cluster}",
                "funding_signal": "空头拥挤(易轧空)" if deriv.funding_rate_8h < 0 else "多头拥挤",
                "avwap_distance_pct": f"{((tech.avwap_anchor - p0) / p0) * 100:+.2f}%",
            },
        )

    @staticmethod
    def check_invalidation(
        pred: PredictionOutput,
        candle_1h_close: float,
        oi_1h_change_pct: float,
    ) -> Optional[str]:
        """执行层熔断检查：1H 实体收破插针极值且 OI 同步扩张 -> 强制切换 Q3 趋势模式"""
        if candle_1h_close > pred.invalidation_bull_trigger and oi_1h_change_pct > 1.0:
            return "BULL_BREAKOUT_INVALIDATION: 向上突破极端轧空上限且主力增仓，切换 Q3 顺势追多"
        if candle_1h_close < pred.invalidation_bear_trigger and oi_1h_change_pct > 1.0:
            return "BEAR_BREAKDOWN_INVALIDATION: 向下跌破插针底线且空头增仓，切换 Q3 顺势追空"
        return None


# =============================================================================
# 6. 顶层编排器与执行入口 (Pipeline Orchestrator)
# =============================================================================

class BTC24HPredictorPipeline:
    def __init__(self, adapter: BaseMarketDataAdapter):
        self.adapter = adapter
        self.sentiment_engine = SentimentReflexivityEngine()
        self.router = FourQuadrantRouter()
        self.boundary_engine = ExtremeBoundaryEngine()

    async def run_once(self) -> PredictionOutput:
        bundle = await self.adapter.fetch_bundle()
        s_raw, s_adj, exhausted = self.sentiment_engine.evaluate(bundle)
        regime, weekend_overlay, weights = self.router.route(bundle, s_raw, exhausted)
        return self.boundary_engine.synthesize(
            bundle=bundle,
            regime=regime,
            weekend_overlay=weekend_overlay,
            weights=weights,
            s_raw=s_raw,
            s_adj=s_adj,
        )


async def main() -> None:
    pipeline = BTC24HPredictorPipeline(adapter=DemoSnapshotAdapter())
    result = await pipeline.run_once()
    print("=" * 68)
    print(f"BTC 24H 极值趋势预测引擎 [DEMO] | 时间戳: {result.timestamp}")
    print(f"当前现价: ${result.spot_price:,.1f} | 24H 预期波幅 (EM): ±${result.em_24h:,.1f}")
    print(f"状态路由: {result.primary_regime.value} (周末薄盘叠加: {result.weekend_overlay})")
    print(f"极端轧空上限: ${result.wick_ceiling_zone[0]:,.1f} ~ ${result.wick_ceiling_zone[1]:,.1f}")
    print(f"预期高点磁吸: ${result.core_high_zone[0]:,.1f} ~ ${result.core_high_zone[1]:,.1f}")
    print(f"预期低点磁吸: ${result.core_low_zone[0]:,.1f} ~ ${result.core_low_zone[1]:,.1f}")
    print(f"极端插针底线: ${result.wick_floor_zone[0]:,.1f} ~ ${result.wick_floor_zone[1]:,.1f}")
    print("=" * 68)


if __name__ == "__main__":
    asyncio.run(main())
