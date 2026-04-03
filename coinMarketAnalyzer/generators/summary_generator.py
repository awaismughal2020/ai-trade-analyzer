"""
Summary Generator
Creates human-readable summaries from analysis metrics
"""

from typing import Dict, Any
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.whale_engine import WhaleMetrics
from engines.technical_engine import TechnicalSignals
from engines.holder_metrics import HolderStats

# Setup logging
logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generates human-readable summaries from trading analysis
    """
    
    def __init__(self):
        """Initialize Summary Generator"""
        logger.info("Summary Generator initialized")
    
    def generate_summary(
        self,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats,
        recent_analysis: dict = None,
        token_type: str = "meme",
        degraded: bool = False,
        market_context = None,
        market_data: dict = None
    ) -> str:
        """
        Generate comprehensive summary text framed within a 24-hour effectiveness window.
        
        Perps-aware: Skips holder distribution and concentration sections for perps tokens
        (perps don't have traditional holders/gini/top10 concentration).
        Instead, includes perps-specific context like large trader activity and funding.
        
        Args:
            whale_metrics: Whale analysis metrics
            technical_signals: Technical indicator signals
            holder_stats: Holder distribution statistics
            recent_analysis: 24h-focused analysis metrics (optional)
            token_type: 'meme' or 'perps' — controls which sections to include
            degraded: True if response is degraded (missing whale/holder data)
            market_context: MarketContext with global market data (optional)
            market_data: Market data dict (optional, used for perps whale flow)
            
        Returns:
            Human-readable summary string framed in ±24h context
        """
        is_perps = token_type.lower() == "perps"
        parts = []
        
        # Degraded mode disclaimer (prepend if data is missing)
        if degraded and not is_perps:
            parts.append("[LIMITED DATA] Whale and holder data unavailable — analysis based on price action and technical indicators only.")
        
        # 24h context header
        recent_24h_part = self._summarize_24h_context(recent_analysis)
        if recent_24h_part:
            parts.append(recent_24h_part)
        
        # Whale / large trader behavior summary
        if is_perps:
            whale_part = self._summarize_large_trader_behavior(whale_metrics)
        else:
            whale_part = self._summarize_whale_behavior(whale_metrics)
        parts.append(whale_part)
        
        # Market-wide whale flow (perps only)
        if is_perps and market_data:
            whale_flow_part = self._summarize_whale_flow(market_data)
            if whale_flow_part:
                parts.append(whale_flow_part)
        
        # Holder distribution and concentration (meme only — not applicable to perps)
        if not is_perps and not degraded:
            holder_part = self._summarize_holder_distribution(whale_metrics, holder_stats)
            parts.append(holder_part)
            
            concentration_part = self._summarize_concentration(whale_metrics)
            parts.append(concentration_part)
        
        # Market trend conclusion (prefer 24h price trend when available)
        trend_part = self._summarize_market_trend(technical_signals, whale_metrics, recent_analysis=recent_analysis)
        parts.append(trend_part)
        
        # Dominant whale inactivity warning (meme only, skip if degraded - no whale data)
        if not is_perps and not degraded:
            dominant_whale_warning = self._summarize_dominant_whale_inactivity(whale_metrics)
            if dominant_whale_warning:
                parts.append(dominant_whale_warning)
        
        # Global market context (CryptoRank)
        market_ctx_part = self._summarize_market_context(market_context)
        if market_ctx_part:
            parts.append(market_ctx_part)
        
        # 24h forward-looking footer
        parts.append("Signal effective for the next 24 hours.")
        
        return " ".join(parts)
    
    def _summarize_24h_context(self, recent_analysis: dict = None) -> str:
        """
        Generate a summary of the last 24 hours of market activity.
        
        This frames the entire analysis within the 24h effectiveness window.
        
        Args:
            recent_analysis: Dict with 24h-focused metrics from _compute_recent_24h_analysis
            
        Returns:
            Summary string for 24h context, or empty string if no data
        """
        if not recent_analysis or "error" in recent_analysis:
            return "In the last 24 hours:"
        
        parts = []
        # Use actual candle coverage when we have less than 24h so we don't mislead (e.g. new tokens)
        window_h = recent_analysis.get("window_actual_hours")
        if window_h is not None and window_h < 24:
            h_str = str(int(window_h)) if window_h == int(window_h) else f"{round(window_h, 1)}"
            time_phrase = f"Over the last {h_str}h"
        else:
            time_phrase = "Over the last 24h"
        
        # Price change
        price_change = recent_analysis.get("price_change_24h_pct")
        
        if price_change is not None:
            if abs(price_change) < 0.5:
                parts.append(f"{time_phrase}, price is flat ({price_change:+.1f}%)")
            elif price_change > 0:
                parts.append(f"{time_phrase}, price is up {price_change:+.1f}%")
            else:
                parts.append(f"{time_phrase}, price is down {price_change:+.1f}%")
        else:
            wh = recent_analysis.get("window_actual_hours")
            if wh is not None and wh < 23:
                parts.append(f"In the last {wh} hours")
            else:
                parts.append("In the last 24 hours")
        
        # Volume trend
        vol_trend = recent_analysis.get("volume_trend_24h")
        if vol_trend:
            if vol_trend == "increasing":
                parts.append("with increasing volume")
            elif vol_trend == "decreasing":
                parts.append("with declining volume")
            else:
                parts.append("with stable volume")
        
        # Buy/sell pressure
        buy_pressure = recent_analysis.get("buy_pressure_24h")
        if buy_pressure is not None:
            if buy_pressure > 0.6:
                parts.append("and strong buy pressure.")
            elif buy_pressure > 0.5:
                parts.append("and moderate buy pressure.")
            elif buy_pressure < 0.4:
                parts.append("and strong sell pressure.")
            else:
                parts.append("and moderate sell pressure.")
        else:
            parts[-1] = parts[-1] + "."
        
        return " ".join(parts)
    
    def _summarize_whale_behavior(self, metrics: WhaleMetrics) -> str:
        """Summarize whale accumulation/distribution behavior"""
        data_source = getattr(metrics, 'whale_data_source', 'recent')
        is_stale = getattr(metrics, 'is_whale_data_stale', False)
        
        # Handle "holding but not trading" scenario (whale holding significant position)
        if data_source == "holding":
            if metrics.whale_state == "Accumulation":
                return "Whales are holding significant positions (no recent trades but still accumulating),"
            elif metrics.whale_state == "Distribution":
                return "Whales have distributed but may still hold positions,"
            else:
                return "Whales are holding stable positions (no recent trades but still holding),"
        
        # Check if whale data is stale (no recent activity and no significant holdings)
        stale_suffix = ""
        if is_stale and data_source == "lifetime":
            stale_suffix = " (based on lifetime data - no recent whale trades)"
        
        # Special case: If no whale trading activity (both volumes are 0), emphasize concentration
        if metrics.whale_buy_volume == 0 and metrics.whale_sell_volume == 0:
            # Check if we have whales holding
            if metrics.confirmed_whale_count > 0:
                return "Whale concentration present (no recent trading activity),"
            else:
                return "No significant whale activity detected,"
        
        if metrics.whale_state == "Accumulation":
            # Determine strength
            net_flow = metrics.whale_net_volume
            acc_pressure = metrics.whale_buy_volume
            
            if net_flow > 0 and acc_pressure > metrics.whale_sell_volume * 3:
                return f"Whales are accumulating strongly{stale_suffix},"
            elif net_flow > 0:
                return f"Whales are accumulating moderately{stale_suffix},"
            else:
                return f"Whales show slight accumulation{stale_suffix},"
        
        elif metrics.whale_state == "Distribution":
            if metrics.whale_sell_volume > metrics.whale_buy_volume * 3:
                return f"Whales are distributing heavily{stale_suffix},"
            elif metrics.whale_sell_volume > metrics.whale_buy_volume * 1.5:
                return f"Whales are distributing moderately{stale_suffix},"
            else:
                return f"Whales show slight distribution{stale_suffix},"
        
        else:  # Stability
            if is_stale and data_source == "lifetime":
                return "Whale activity is stable (no recent whale trades in last 10 days),"
            return "Whale activity is stable,"
    
    def _summarize_large_trader_behavior(self, metrics: WhaleMetrics) -> str:
        """
        Summarize large trader behavior for perpetual contracts.
        
        Uses whale_buy_volume / whale_sell_volume / whale_state from WhaleMetrics
        but frames them as 'large traders' rather than 'whales', which is more
        appropriate terminology for perps/derivatives markets.
        """
        # No activity
        if metrics.whale_buy_volume == 0 and metrics.whale_sell_volume == 0:
            return "No significant large trader activity in the last 24h."
        
        buy_vol = metrics.whale_buy_volume
        sell_vol = metrics.whale_sell_volume
        net_vol = metrics.whale_net_volume
        
        # Format volumes for readability
        def fmt_vol(v):
            if abs(v) >= 1_000_000:
                return f"${abs(v)/1_000_000:.1f}M"
            elif abs(v) >= 1_000:
                return f"${abs(v)/1_000:.0f}K"
            else:
                return f"${abs(v):.0f}"
        
        if metrics.whale_state == "Accumulation":
            if buy_vol > sell_vol * 3:
                strength = "aggressively going long"
            elif buy_vol > sell_vol * 1.5:
                strength = "net long"
            else:
                strength = "slightly net long"
            return f"Large traders are {strength} ({fmt_vol(buy_vol)} buys vs {fmt_vol(sell_vol)} sells, net {fmt_vol(net_vol)})."
        
        elif metrics.whale_state == "Distribution":
            if sell_vol > buy_vol * 3:
                strength = "aggressively going short"
            elif sell_vol > buy_vol * 1.5:
                strength = "net short"
            else:
                strength = "slightly net short"
            return f"Large traders are {strength} ({fmt_vol(sell_vol)} sells vs {fmt_vol(buy_vol)} buys, net {fmt_vol(abs(net_vol))})."
        
        else:  # Stability
            return f"Large trader flow is balanced ({fmt_vol(buy_vol)} buys vs {fmt_vol(sell_vol)} sells)."
    
    def _summarize_whale_flow(self, market_data: dict) -> str:
        """Summarize market-wide whale flow for perps (from block-liquidity whale/flow)."""
        net_avg = market_data.get('whale_flow_net_avg', 0)
        count = market_data.get('whale_flow_count_whales', 0)
        long_vol = market_data.get('whale_flow_long_volume', 0)
        short_vol = market_data.get('whale_flow_short_volume', 0)

        if count == 0 and long_vol == 0 and short_vol == 0:
            return ""

        def fmt_vol(v):
            if abs(v) >= 1_000_000_000:
                return f"${abs(v)/1_000_000_000:.1f}B"
            elif abs(v) >= 1_000_000:
                return f"${abs(v)/1_000_000:.0f}M"
            elif abs(v) >= 1_000:
                return f"${abs(v)/1_000:.0f}K"
            return f"${abs(v):.0f}"

        if net_avg > 0.02:
            direction = "net long"
        elif net_avg < -0.02:
            direction = "net short"
        else:
            direction = "balanced"

        vol_part = ""
        if long_vol > 0 or short_vol > 0:
            vol_part = f" ({fmt_vol(long_vol)} long vs {fmt_vol(short_vol)} short)"

        return f"Market-wide whale flow is {direction}{vol_part} with {count} whales active."
    
    def _summarize_holder_distribution(
        self,
        whale_metrics: WhaleMetrics,
        holder_stats: HolderStats
    ) -> str:
        """Summarize holder growth and distribution"""
        growth = whale_metrics.holder_growth_24h
        active = holder_stats.active_holders
        
        if growth > 10:
            growth_text = "holder growth is very positive"
        elif growth > 5:
            growth_text = "holder growth is positive"
        elif growth > 0:
            growth_text = "holder growth is slightly positive"
        elif growth > -5:
            growth_text = "holder count is stable"
        else:
            growth_text = "holder count is declining"
        
        # Add holder count context
        if active > 1000:
            count_text = f" ({active:,} active holders),"
        elif active > 500:
            count_text = f" ({active} active holders),"
        elif active > 100:
            count_text = f" ({active} holders),"
        else:
            count_text = " (low holder count),"
        
        return growth_text + count_text
    
    def _summarize_concentration(self, metrics: WhaleMetrics) -> str:
        """Summarize supply concentration health"""
        gini = metrics.gini_coefficient
        top10 = metrics.top10_hold_percent
        dev = metrics.dev_hold_percent
        sniper = metrics.sniper_hold_percent
        
        # Overall assessment
        issues = []
        
        if gini > 0.6:
            issues.append("high inequality")
        if top10 > 65:
            issues.append("high top 10 concentration")
        if dev > 10:
            issues.append("high dev holding")
        if sniper > 5:
            issues.append("significant sniper presence")
        
        if len(issues) == 0:
            if gini < 0.4 and top10 < 30:
                return "and supply concentration is healthy."
            else:
                return "and supply concentration is moderate."
        elif len(issues) == 1:
            return f"but there is {issues[0]}."
        else:
            return f"but there are concerns: {', '.join(issues)}."
    
    def _summarize_market_trend(
        self,
        technical: TechnicalSignals,
        whale_metrics: WhaleMetrics,
        recent_analysis: dict = None
    ) -> str:
        """Summarize overall market trend.

        Prefers the 24h price-derived trend_24h when available (strongly_bullish,
        bullish, strongly_bearish, bearish). Falls back to the technical + whale +
        RSI heuristic when trend_24h is absent or sideways.
        """
        # Use 24h price trend when present and directional
        if recent_analysis:
            trend_24h = recent_analysis.get("trend_24h")
            _TREND_MAP = {
                "strongly_bullish": "Market trend strongly bullish.",
                "bullish": "Market trend leaning bullish.",
                "strongly_bearish": "Market trend strongly bearish.",
                "bearish": "Market trend leaning bearish.",
            }
            if trend_24h in _TREND_MAP:
                return _TREND_MAP[trend_24h]

        # Fallback: combine technical indicators and whale signals
        tech_signal = technical.overall_signal
        tech_strength = technical.signal_strength
        whale_state = whale_metrics.whale_state
        
        bullish_count = 0
        bearish_count = 0
        
        if tech_signal == "bullish":
            bullish_count += 1
        elif tech_signal == "bearish":
            bearish_count += 1
        
        if whale_state == "Accumulation":
            bullish_count += 1
        elif whale_state == "Distribution":
            bearish_count += 1
        
        if technical.rsi_signal == "oversold":
            bullish_count += 1
        elif technical.rsi_signal == "overbought":
            bearish_count += 1
        
        if bullish_count > bearish_count + 1:
            if tech_strength > 0.7:
                return "Market trend strongly bullish."
            else:
                return "Market trend leaning bullish."
        elif bearish_count > bullish_count + 1:
            if tech_strength > 0.7:
                return "Market trend strongly bearish."
            else:
                return "Market trend leaning bearish."
        else:
            return "Market trend neutral/mixed."
    
    def _summarize_dominant_whale_inactivity(self, metrics: WhaleMetrics) -> str:
        """
        Generate warning about dominant whale inactivity or aging.
        
        This helps users understand when whale activity signals may be misleading
        because the top holders (dominant whales) haven't traded recently.
        
        Status levels:
        - ACTIVE: All dominant whales traded within 3 days
        - AGING: Dominant whales traded 3-5 days ago (borderline)
        - PARTIALLY_INACTIVE: Some dominant whales inactive for 5+ days
        - FULLY_INACTIVE: All dominant whales inactive for 5+ days
        
        Args:
            metrics: Whale analysis metrics
            
        Returns:
            Warning string if dominant whales are inactive/aging, empty string otherwise
        """
        dominant_status = getattr(metrics, 'dominant_whale_status', 'UNKNOWN')
        dominant_inactive_pct = getattr(metrics, 'dominant_whale_inactive_holding_pct', 0.0)
        dominant_aging_pct = getattr(metrics, 'dominant_whale_aging_holding_pct', 0.0)
        top_holder_hours = getattr(metrics, 'top_holder_last_activity_hours', None)
        
        if dominant_status == "UNKNOWN" or dominant_status == "ACTIVE":
            return ""
        
        # Convert hours to days for readability
        if top_holder_hours is not None:
            days_inactive = top_holder_hours / 24
            time_str = f"{days_inactive:.0f} days" if days_inactive >= 1 else f"{top_holder_hours:.0f} hours"
        else:
            time_str = "unknown period"
        
        if dominant_status == "FULLY_INACTIVE":
            return (f"WARNING: All dominant whales (holding {dominant_inactive_pct:.0f}% of supply) "
                   f"have been inactive for {time_str}. Whale signals may be unreliable.")
        elif dominant_status == "PARTIALLY_INACTIVE":
            if dominant_inactive_pct >= 50:
                return (f"CAUTION: Top holder ({dominant_inactive_pct:.0f}% of supply) "
                       f"has not traded in {time_str}. Activity may be from smaller whales only.")
            else:
                return (f"Note: Some dominant whales ({dominant_inactive_pct:.0f}% of supply) "
                       f"are inactive for 5+ days. Monitor closely.")
        elif dominant_status == "AGING":
            if dominant_aging_pct >= 50:
                return (f"⏳ Note: Top holder ({dominant_aging_pct:.0f}% of supply) last traded {time_str} ago. "
                       f"Whale activity is aging - signal reliability reduced.")
            else:
                return (f"⏳ Note: Some dominant whales ({dominant_aging_pct:.0f}% of supply) "
                       f"haven't traded in 3-5 days. Whale signal is aging.")
        
        return ""
    
    def _summarize_market_context(self, market_context) -> str:
        """Summarize global market conditions from CryptoRank data"""
        if not market_context or getattr(market_context, 'is_default', True):
            return ""
        
        fg = market_context.fear_greed
        fg_change = market_context.fear_greed_change
        btc_dom = market_context.btc_dominance
        btc_dom_change = market_context.btc_dominance_change
        mcap_change = market_context.total_market_cap_change
        alt_idx = market_context.altcoin_index
        alt_change = market_context.altcoin_index_change
        
        if fg <= 15:
            sentiment = "Extreme fear"
        elif fg <= 30:
            sentiment = "Fear"
        elif fg <= 55:
            sentiment = "Neutral"
        elif fg <= 75:
            sentiment = "Greed"
        else:
            sentiment = "Extreme greed"
        
        fg_dir = "improving" if fg_change > 0 else "worsening" if fg_change < 0 else "stable"
        btc_dir = "rising" if btc_dom_change > 0.1 else "declining" if btc_dom_change < -0.1 else "stable"
        alt_dir = f"+{alt_change}" if alt_change > 0 else str(alt_change)
        mcap_dir = f"+{mcap_change:.2f}" if mcap_change > 0 else f"{mcap_change:.2f}"
        
        return (
            f"Market Conditions: {sentiment} (F&G: {fg}, {fg_dir}). "
            f"BTC dominance {btc_dom:.1f}% ({btc_dir}). "
            f"Total market cap {mcap_dir}%. "
            f"Altcoin index: {alt_idx} ({alt_dir})."
        )
    
    def generate_short_summary(
        self,
        signal: str,
        confidence: float,
        whale_state: str,
        phase: str
    ) -> str:
        """
        Generate a short one-line summary
        
        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Confidence level (0-1)
            whale_state: Whale behavior state
            phase: Token phase
            
        Returns:
            Short summary string
        """
        confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        
        if signal == "BUY":
            return f"{phase} token with {whale_state.lower()} whale activity. {confidence_text.capitalize()} confidence buy signal."
        elif signal == "SELL":
            return f"{phase} token with {whale_state.lower()} whale activity. {confidence_text.capitalize()} confidence sell signal."
        else:
            return f"{phase} token with {whale_state.lower()} whale activity. Neutral/hold recommendation."
    
    def generate_risk_summary(
        self,
        whale_metrics: WhaleMetrics,
        holder_stats: HolderStats
    ) -> str:
        """Generate risk-focused summary"""
        risks = []
        
        # Check various risk factors
        if whale_metrics.gini_coefficient > 0.6:
            risks.append(f"High Gini coefficient ({whale_metrics.gini_coefficient:.2f}) indicates uneven distribution")
        
        if whale_metrics.top10_hold_percent > 50:
            risks.append(f"Top 10 holders control {whale_metrics.top10_hold_percent:.1f}% of supply")
        
        if whale_metrics.dev_hold_percent > 10:
            risks.append(f"Developer holds {whale_metrics.dev_hold_percent:.1f}% - rug risk")
        
        if whale_metrics.sniper_hold_percent > 5:
            risks.append(f"Snipers hold {whale_metrics.sniper_hold_percent:.1f}% - dump risk")
        
        if whale_metrics.whale_state == "Distribution":
            risks.append("Whales are currently selling")
        
        if holder_stats.active_holders < 100:
            risks.append(f"Low holder count ({holder_stats.active_holders}) - liquidity risk")
        
        if len(risks) == 0:
            return "No significant risks identified."
        else:
            return "Risk factors: " + "; ".join(risks) + "."


if __name__ == "__main__":
    print("Summary Generator")
    print("=" * 60)
    
    # Create mock data for testing
    whale_metrics = WhaleMetrics(
        whale_buy_volume=500000,
        whale_sell_volume=100000,
        whale_net_volume=400000,
        gini_coefficient=0.42,
        top10_hold_percent=25.0,
        dev_hold_percent=3.0,
        sniper_hold_percent=1.5,
        whale_state="Accumulation",
        whale_count=5,
        confirmed_whale_count=3,
        phase="P2",
        total_holders=500,
        active_holders=450,
        holder_growth_24h=5.0
    )
    
    tech_signals = TechnicalSignals(
        rsi=45.0,
        rsi_signal="neutral",
        ema_20=0.00015,
        ema_50=0.00014,
        ema_cross_signal="bullish",
        macd_line=0.000001,
        macd_signal=0.0000005,
        macd_histogram=0.0000005,
        macd_trend="bullish",
        bb_upper=0.00018,
        bb_lower=0.00012,
        bb_position=0.5,
        volume_ratio=1.5,
        buy_pressure=0.6,
        price_momentum_5=0.08,
        price_momentum_10=0.15,
        volatility=0.02,
        overall_signal="bullish",
        signal_strength=0.7
    )
    
    holder_stats = HolderStats(
        total_holders=500,
        active_holders=450,
        gini_coefficient=0.42,
        top10_concentration=25.0,
        top20_concentration=35.0,
        top50_concentration=55.0,
        median_holding=10000.0,
        mean_holding=50000.0,
        std_holding=100000.0,
        holder_score=75.0
    )
    
    generator = SummaryGenerator()
    
    summary = generator.generate_summary(whale_metrics, tech_signals, holder_stats)
    print(f"\nFull Summary:")
    print(f"  {summary}")
    
    short = generator.generate_short_summary("BUY", 0.83, "Accumulation", "P2")
    print(f"\nShort Summary:")
    print(f"  {short}")
    
    risk = generator.generate_risk_summary(whale_metrics, holder_stats)
    print(f"\nRisk Summary:")
    print(f"  {risk}")

