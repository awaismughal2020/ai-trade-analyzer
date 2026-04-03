"""
Entry Timing Engine - Feature 2
Analyzes optimal entry timing for trades using multiple signal components

This engine combines 5 analyzers to determine the best moment to enter a trade:
1. Momentum Analysis - RSI and price momentum
2. Volume Analysis - Buy/sell ratio and volume patterns
3. Liquidity Analysis - Market liquidity and depth (requires Birdeye)
4. Smart Money Analysis - Top trader behavior (requires Birdeye)
5. Whale Analysis - Whale accumulation/distribution patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import timing_config

# Setup logging
logger = logging.getLogger(__name__)


class TimingRecommendation(Enum):
    """Entry and Exit timing recommendations"""
    # Entry recommendations
    ENTER_NOW = "ENTER_NOW"
    WAIT = "WAIT"
    DO_NOT_ENTER = "DO_NOT_ENTER"
    
    # Exit recommendations
    EXIT_NOW = "EXIT_NOW"
    WAIT_TO_EXIT = "WAIT_TO_EXIT"
    EXIT = "EXIT"


@dataclass
class TimingSignal:
    """
    Complete timing signal with recommendation and component scores
    
    Attributes:
        recommendation: The timing recommendation (ENTER_NOW, WAIT, EXIT_NOW, EXIT, etc.)
        wait_minutes: Suggested wait time in minutes (0 if immediate action recommended)
        confidence: Confidence in the recommendation (0.0 to 1.0)
        reason: Human-readable explanation of the recommendation
        potential_improvement_pct: Estimated % improvement if waiting (score-derived; not a literal expected price move).
        miss_risk_pct: Risk of missing the opportunity by waiting (score-derived; not a probability).
        Component scores (each -1 to +1):
            - Positive = favorable for immediate entry
            - Negative = suggests waiting
    """
    recommendation: TimingRecommendation
    wait_minutes: Optional[int]
    confidence: float
    reason: str
    potential_improvement_pct: float
    miss_risk_pct: float
    
    # Component scores (-1 to +1)
    momentum_score: float
    volume_score: float
    liquidity_score: float
    smart_money_score: float
    whale_score: float
    
    # Final aggregated score
    final_score: float
    
    # Component reasons for debugging
    component_reasons: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'recommendation': self.recommendation.value,
            'wait_minutes': self.wait_minutes,
            'confidence': round(self.confidence, 4),
            'reason': self.reason,
            'potential_improvement_pct': round(self.potential_improvement_pct, 2),
            'miss_risk_pct': round(self.miss_risk_pct, 2),
            'scores': {
                'momentum': round(self.momentum_score, 4),
                'volume': round(self.volume_score, 4),
                'liquidity': round(self.liquidity_score, 4),
                'smart_money': round(self.smart_money_score, 4),
                'whale': round(self.whale_score, 4),
                'final': round(self.final_score, 4)
            },
            'component_reasons': self.component_reasons
        }


class EntryTimingEngine:
    """
    Analyzes market conditions to determine optimal entry timing
    
    Uses weighted combination of 5 analyzers to produce a timing score
    and recommendation for immediate entry, waiting, or urgent entry.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize Entry Timing Engine
        
        Args:
            config: Optional TimingConfig override (uses global timing_config if None)
        """
        self.config = config or timing_config
        
        # Default weights (graduated tokens - not on bonding curve)
        self.default_weights = self.config.get_weights(is_on_bonding_curve=False)
        
        # Current weights (will be set per-analysis based on bonding curve status)
        self.weights = self.default_weights.copy()
        
        # Track bonding curve status for current analysis
        self._is_on_bonding_curve = False
        
        logger.info(f"Entry Timing Engine initialized with default weights: {self.weights}")
    
    def _get_column(self, df: pd.DataFrame, *names) -> Optional[pd.Series]:
        """
        Get column from DataFrame, supporting multiple naming conventions.
        Handles both meme tokens (uppercase: Close, Volume) and perps (lowercase: close, volume).
        
        Args:
            df: DataFrame to get column from
            *names: Column names to try in order (e.g., 'Close', 'close')
            
        Returns:
            Series if column found, None otherwise
        """
        for name in names:
            if name in df.columns:
                return df[name]
        return None
    
    def _get_close_prices(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Get close prices as numpy array, handling both column name conventions"""
        col = self._get_column(df, 'Close', 'close')
        if col is not None:
            return np.array(col.values, dtype=float)
        return None
    
    def _get_volume(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Get volume column, handling both column name conventions"""
        return self._get_column(df, 'Volume', 'volume')
    
    def _get_buy_volume(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Get buy volume column, handling both column name conventions"""
        return self._get_column(df, 'BuyVolume', 'buy_volume')
    
    def _get_sell_volume(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Get sell volume column, handling both column name conventions"""
        return self._get_column(df, 'SellVolume', 'sell_volume')
    
    def analyze(
        self,
        candles_df: pd.DataFrame,
        whale_metrics: Optional[Any] = None,
        birdeye_data: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Any] = None,
        is_on_bonding_curve: bool = False,
        token_type: str = "meme",
        market_context = None,
        recent_analysis: Optional[Dict[str, Any]] = None
    ) -> TimingSignal:
        """
        Perform comprehensive entry timing analysis.
        
        NOTE: For optimal signal quality, callers should pass candles filtered to the
        last 24 hours. This ensures timing recommendations are based on the most recent
        price action and are effective for the next 24 hours. The caller (predict.py)
        handles this filtering via _filter_recent_candles().
        
        Args:
            candles_df: DataFrame with OHLCV candle data (ideally last 24h for timing focus)
            whale_metrics: WhaleMetrics object from whale engine (optional)
            birdeye_data: Dict from BirdeyeFetcher.fetch_timing_data() (optional)
            technical_signals: TechnicalSignals object (optional)
            is_on_bonding_curve: True if token is still on bonding curve (not graduated to Raydium)
                                 Affects weight distribution:
                                 - ON curve: Smart Money 10%, Whale 20%
                                 - OFF curve: Smart Money 15%, Whale 15%
            
        Returns:
            TimingSignal with recommendation and scores (effective for next 24h)
        """
        # Set bonding curve status and update weights accordingly
        self._is_on_bonding_curve = is_on_bonding_curve
        self.weights = self.config.get_weights(is_on_bonding_curve=is_on_bonding_curve)
        
        if is_on_bonding_curve:
            logger.info(f"Token is ON bonding curve - using adjusted weights: smart_money={self.weights['smart_money']:.2f}, whale={self.weights['whale']:.2f}")
        
        # Validate input data
        if candles_df is None or len(candles_df) < 10:
            logger.warning("Insufficient candle data for timing analysis")
            return self._insufficient_data_response()
        
        logger.info(f"Analyzing entry timing with {len(candles_df)} candles")
        
        # Run all analyzers
        all_scores = {}
        all_reasons = {}
        
        # 1. Momentum Analysis
        momentum_score, momentum_reasons = self._analyze_momentum(candles_df, technical_signals)
        all_scores['momentum'] = momentum_score
        all_reasons['momentum'] = momentum_reasons
        
        # 2. Volume Analysis
        volume_score, volume_reasons = self._analyze_volume(candles_df, recent_analysis=recent_analysis)
        all_scores['volume'] = volume_score
        all_reasons['volume'] = volume_reasons
        
        # 3. Liquidity Analysis (requires Birdeye data)
        liquidity_score, liquidity_reasons = self._analyze_liquidity(birdeye_data)
        all_scores['liquidity'] = liquidity_score
        all_reasons['liquidity'] = liquidity_reasons
        
        # 4. Smart Money Analysis (requires Birdeye data)
        smart_money_score, smart_money_reasons = self._analyze_smart_money(birdeye_data)
        all_scores['smart_money'] = smart_money_score
        all_reasons['smart_money'] = smart_money_reasons
        
        # 5. Whale / Large Trader Analysis
        whale_score, whale_reasons = self._analyze_whale(whale_metrics, token_type=token_type)
        all_scores['whale'] = whale_score
        all_reasons['whale'] = whale_reasons
        
        # Aggregate scores
        final_score = self._aggregate_scores(all_scores)
        
        # Make decision
        timing_signal = self._make_decision(final_score, all_scores, all_reasons)
        
        # Apply market context confidence modifier
        if market_context and not getattr(market_context, 'is_default', True):
            if getattr(market_context, 'is_extreme_fear', False):
                if timing_signal.recommendation.value == "ENTER_NOW":
                    timing_signal.confidence = max(0.3, timing_signal.confidence * 0.85)
                    timing_signal.component_reasons.setdefault("market_context", []).append(
                        f"Confidence reduced: extreme fear market (F&G={market_context.fear_greed})"
                    )
                    logger.info(
                        f"Market context: extreme fear — ENTER_NOW confidence reduced to {timing_signal.confidence:.2f}"
                    )
            elif getattr(market_context, 'is_extreme_greed', False):
                if timing_signal.recommendation.value == "ENTER_NOW":
                    timing_signal.confidence = max(0.3, timing_signal.confidence * 0.90)
                    timing_signal.component_reasons.setdefault("market_context", []).append(
                        f"Confidence reduced: extreme greed market (F&G={market_context.fear_greed})"
                    )
        
        logger.info(f"Timing analysis complete: {timing_signal.recommendation.value} "
                   f"(score: {final_score:.3f}, confidence: {timing_signal.confidence:.2f})")
        
        return timing_signal
    
    def analyze_exit_timing(
        self,
        candles_df: pd.DataFrame,
        whale_metrics: Optional[Any] = None,
        birdeye_data: Optional[Dict[str, Any]] = None,
        technical_signals: Optional[Any] = None,
        is_on_bonding_curve: bool = False,
        token_type: str = "meme",
        safety_overrides: Optional[List[Dict[str, Any]]] = None,
        recent_analysis: Optional[Dict[str, Any]] = None
    ) -> TimingSignal:
        """
        Analyze optimal EXIT timing for trades (when main signal is SELL)
        
        NOTE: For optimal signal quality, callers should pass candles filtered to the
        last 24 hours. This ensures exit timing recommendations reflect the most recent
        price action and are effective for the next 24 hours.
        
        For exit timing, the logic is INVERTED from entry timing:
        - Price UP = good time to exit (sell high)
        - Price DOWN = wait for bounce before exiting
        - High sell volume = urgent exit (others are selling)
        - Whale distribution = urgent exit
        
        Args:
            candles_df: DataFrame with OHLCV candle data (ideally last 24h for timing focus)
            whale_metrics: WhaleMetrics object from whale engine (optional)
            birdeye_data: Dict from BirdeyeFetcher.fetch_timing_data() (optional)
            technical_signals: TechnicalSignals object (optional)
            is_on_bonding_curve: True if token is still on bonding curve (not graduated to Raydium)
                                 Affects weight distribution:
                                 - ON curve: Smart Money 10%, Whale 20%
                                 - OFF curve: Smart Money 15%, Whale 15%
            safety_overrides: List of safety overrides that triggered the SELL signal (optional).
                              When present, the EXIT_NOW threshold is lowered to reflect the
                              urgency established by the safety override.
            
        Returns:
            TimingSignal with EXIT recommendation and scores (effective for next 24h)
        """
        # Set bonding curve status and update weights accordingly
        self._is_on_bonding_curve = is_on_bonding_curve
        self.weights = self.config.get_weights(is_on_bonding_curve=is_on_bonding_curve)
        
        if is_on_bonding_curve:
            logger.info(f"Token is ON bonding curve - using adjusted exit weights: smart_money={self.weights['smart_money']:.2f}, whale={self.weights['whale']:.2f}")
        
        # Validate input data
        if candles_df is None or len(candles_df) < 10:
            logger.warning("Insufficient candle data for exit timing analysis")
            return self._insufficient_exit_data_response()
        
        logger.info(f"Analyzing exit timing with {len(candles_df)} candles")
        
        # Run all analyzers with EXIT logic (inverted from entry)
        all_scores = {}
        all_reasons = {}
        
        # 1. Exit Momentum Analysis (inverted)
        momentum_score, momentum_reasons = self._analyze_exit_momentum(candles_df, technical_signals)
        all_scores['momentum'] = momentum_score
        all_reasons['momentum'] = momentum_reasons
        
        # 2. Exit Volume Analysis (inverted)
        volume_score, volume_reasons = self._analyze_exit_volume(candles_df, recent_analysis=recent_analysis)
        all_scores['volume'] = volume_score
        all_reasons['volume'] = volume_reasons
        
        # 3. Liquidity Analysis (same for exit - need liquidity to sell)
        liquidity_score, liquidity_reasons = self._analyze_liquidity(birdeye_data)
        all_scores['liquidity'] = liquidity_score
        all_reasons['liquidity'] = liquidity_reasons
        
        # 4. Smart Money Exit Analysis
        smart_money_score, smart_money_reasons = self._analyze_exit_smart_money(birdeye_data)
        all_scores['smart_money'] = smart_money_score
        all_reasons['smart_money'] = smart_money_reasons
        
        # 5. Whale / Large Trader Exit Analysis (inverted)
        whale_score, whale_reasons = self._analyze_exit_whale(whale_metrics, token_type=token_type)
        all_scores['whale'] = whale_score
        all_reasons['whale'] = whale_reasons
        
        # Aggregate scores
        final_score = self._aggregate_scores(all_scores)
        
        # Make EXIT decision (with safety override context for threshold adjustment)
        timing_signal = self._make_exit_decision(final_score, all_scores, all_reasons, safety_overrides)
        
        logger.info(f"Exit timing analysis complete: {timing_signal.recommendation.value} "
                   f"(score: {final_score:.3f}, confidence: {timing_signal.confidence:.2f})")
        
        return timing_signal
    
    def _analyze_exit_momentum(
        self,
        candles_df: pd.DataFrame,
        technical_signals: Optional[Any]
    ) -> Tuple[float, List[str]]:
        """
        Analyze price momentum for EXIT timing (INVERTED from entry)
        
        For exit: price UP = good time to exit, price DOWN = wait
        """
        score = 0.0
        reasons = []
        
        try:
            # Get close prices using helper (handles both meme and perps)
            close_prices = self._get_close_prices(candles_df)
            
            if close_prices is None:
                return 0.0, ["Close price data not available"]
            
            # Get RSI
            if technical_signals and hasattr(technical_signals, 'rsi'):
                rsi = technical_signals.rsi
            else:
                rsi = self._calculate_rsi(close_prices)
            
            if rsi is not None:
                # For EXIT: overbought = GOOD time to exit
                if rsi > self.config.RSI_EXTREME_HIGH:
                    score += 0.6
                    reasons.append(f"RSI extremely overbought ({rsi:.1f}) - excellent exit point")
                elif rsi > self.config.RSI_OVERBOUGHT:
                    score += 0.3
                    reasons.append(f"RSI overbought ({rsi:.1f}) - good exit point")
                elif rsi < self.config.RSI_EXTREME_LOW:
                    score -= 0.4
                    reasons.append(f"RSI extremely oversold ({rsi:.1f}) - wait for bounce")
                elif rsi < self.config.RSI_OVERSOLD:
                    score -= 0.2
                    reasons.append(f"RSI oversold ({rsi:.1f}) - consider waiting")
            
            # Short-window (15m) momentum for exit: spike = good exit, dip = wait
            candle_count = len(close_prices)
            n_candles_15m = 3 if candle_count > 100 else 1  # 5m: 3 candles = 15m; 1h: 1 candle approx
            if len(close_prices) >= n_candles_15m + 1:
                recent_close_15m = float(close_prices[-1])
                past_close_15m = float(close_prices[-(n_candles_15m + 1)])
                if past_close_15m > 0:
                    short_momentum = (recent_close_15m - past_close_15m) / past_close_15m
                    if short_momentum >= self.config.MOMENTUM_SPIKE_UP_THRESHOLD:
                        score += 0.5
                        reasons.append(
                            f"Price up {short_momentum*100:.1f}% (last 15m) - good exit point"
                        )
                    elif short_momentum <= self.config.MOMENTUM_DIP_DOWN_THRESHOLD:
                        score -= 0.4
                        reasons.append(
                            f"Price down {abs(short_momentum)*100:.1f}% (last 15m) - wait for recovery"
                        )

            # Full-window momentum (24h when 24h-filtered candles are passed in)
            recent_close = float(close_prices[-1])
            past_close = float(close_prices[0])
            
            # Calculate the approximate time window for the label
            # Meme = 5m candles, Perps = 1h candles
            # Detect candle interval: if >100 candles, likely 5m; otherwise likely 1h
            if candle_count > 100:
                window_hours = candle_count * 5 / 60  # 5-minute candles
            else:
                window_hours = candle_count  # 1-hour candles
            
            if past_close > 0:
                momentum = (recent_close - past_close) / past_close
                
                # Format time window label
                if window_hours >= 24:
                    time_label = f"{window_hours:.0f}h"
                elif window_hours >= 1:
                    time_label = f"{window_hours:.1f}h"
                else:
                    time_label = f"{window_hours*60:.0f}m"
                
                # For EXIT: price UP = good exit, price DOWN = wait
                if momentum > self.config.MOMENTUM_STRONG_UP:
                    score += 0.5
                    reasons.append(f"Price up {momentum*100:.1f}% ({time_label}) - excellent exit point")
                elif momentum > self.config.MOMENTUM_MILD_UP:
                    score += 0.3
                    reasons.append(f"Price up {momentum*100:.1f}% ({time_label}) - good exit point")
                elif momentum < self.config.MOMENTUM_STRONG_DOWN:
                    score -= 0.4
                    reasons.append(f"Price down {abs(momentum)*100:.1f}% ({time_label}) - wait for recovery")
                elif momentum < self.config.MOMENTUM_MILD_DOWN:
                    score -= 0.2
                    reasons.append(f"Price down {abs(momentum)*100:.1f}% ({time_label}) - consider waiting")
        
        except Exception as e:
            logger.warning(f"Error in exit momentum analysis: {e}")
            reasons.append("Exit momentum analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _analyze_exit_volume(self, candles_df: pd.DataFrame, recent_analysis: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
        """
        Analyze volume for EXIT timing.
        Falls back to 24h aggregate sell pressure from recent_analysis when per-candle data is missing.
        
        For exit: high sell volume = others exiting = URGENT exit
        """
        score = 0.0
        reasons = []
        
        try:
            # Get buy/sell volume using helpers (handles both meme and perps)
            buy_vol_series = self._get_buy_volume(candles_df)
            sell_vol_series = self._get_sell_volume(candles_df)
            
            if buy_vol_series is None or sell_vol_series is None:
                if recent_analysis and recent_analysis.get("sell_pressure_24h") is not None:
                    sell_ratio = float(recent_analysis.get("sell_pressure_24h", 0.5))
                    if sell_ratio > 0.70:
                        score += 0.5
                        reasons.append(f"Heavy selling ({sell_ratio*100:.0f}% sell, 24h aggregate) - exit urgently")
                    elif sell_ratio > self.config.VOLUME_BUY_DOMINANT:
                        score += 0.3
                        reasons.append(f"Sell pressure ({sell_ratio*100:.0f}% sell, 24h aggregate) - good time to exit")
                    elif sell_ratio < self.config.VOLUME_SELL_DOMINANT:
                        score -= 0.2
                        reasons.append(f"Buy dominant ({(1-sell_ratio)*100:.0f}% buy, 24h aggregate) - may wait")
                    else:
                        reasons.append(f"Balanced volume ({sell_ratio*100:.0f}% sell, 24h aggregate)")
                    return max(-1.0, min(1.0, score)), reasons
                return 0.0, ["Volume data not available"]
            
            window = min(60, len(candles_df))
            
            buy_vol = float(buy_vol_series.tail(window).sum())
            sell_vol = float(sell_vol_series.tail(window).sum())
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                sell_ratio = sell_vol / total_vol
                
                # For EXIT: high sell ratio = urgent exit
                if sell_ratio > 0.70:
                    score += 0.5
                    reasons.append(f"Heavy selling ({sell_ratio*100:.0f}% sell) - exit urgently")
                elif sell_ratio > self.config.VOLUME_BUY_DOMINANT:
                    score += 0.3
                    reasons.append(f"Sell pressure ({sell_ratio*100:.0f}% sell) - good time to exit")
                elif sell_ratio < self.config.VOLUME_SELL_DOMINANT:
                    score -= 0.2
                    reasons.append(f"Buy dominant ({(1-sell_ratio)*100:.0f}% buy) - may wait for higher")
            
            # Check volume for liquidity
            volume_series = self._get_volume(candles_df)
            
            if volume_series is not None and len(candles_df) >= 20:
                avg_volume = float(volume_series.tail(60).mean()) if len(candles_df) >= 60 else float(volume_series.mean())
                current_volume = float(volume_series.tail(5).mean())
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    
                    if volume_ratio < 0.3:
                        score -= 0.3
                        reasons.append(f"Very low volume ({volume_ratio:.1f}x avg) - poor exit liquidity")
                    elif volume_ratio > 2.0:
                        reasons.append(f"High volume ({volume_ratio:.1f}x avg) - good exit liquidity")
        
        except Exception as e:
            logger.warning(f"Error in exit volume analysis: {e}")
            reasons.append("Exit volume analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _analyze_exit_smart_money(self, birdeye_data: Optional[Dict]) -> Tuple[float, List[str]]:
        """
        Analyze smart money behavior for EXIT timing
        
        For exit: smart money selling = urgent exit
        """
        if birdeye_data is None or not birdeye_data.get('configured', False):
            return 0.0, ["Birdeye data not available"]
        
        score = 0.0
        reasons = []
        
        try:
            top_traders = birdeye_data.get('top_traders', [])
            
            if top_traders:
                # Birdeye API returns volumeBuy/volumeSell (not buyVolume/sellVolume)
                total_buy = sum(t.get('volumeBuy', 0) or 0 for t in top_traders[:10])
                total_sell = sum(t.get('volumeSell', 0) or 0 for t in top_traders[:10])
                
                if total_buy + total_sell > 0:
                    sell_ratio = total_sell / (total_buy + total_sell)
                    
                    # For EXIT: smart money selling = urgent exit
                    if sell_ratio > 0.70:
                        score += 0.5
                        reasons.append(f"Smart money heavily selling ({sell_ratio*100:.0f}% sell) - exit now")
                    elif sell_ratio > self.config.SMART_MONEY_BUY_DOMINANT:
                        score += 0.3
                        reasons.append(f"Smart money selling ({sell_ratio*100:.0f}% sell) - good exit")
                    elif sell_ratio < self.config.SMART_MONEY_SELL_DOMINANT:
                        score -= 0.2
                        reasons.append(f"Smart money buying ({(1-sell_ratio)*100:.0f}% buy) - may wait")
        
        except Exception as e:
            logger.warning(f"Error in exit smart money analysis: {e}")
            reasons.append("Exit smart money analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _analyze_exit_whale(self, whale_metrics: Optional[Any], token_type: str = "meme") -> Tuple[float, List[str]]:
        """
        Analyze whale / large trader behavior for EXIT timing.
        
        Uses perps-appropriate terminology ('large traders') for perps tokens.
        For exit: distributing = urgent exit
        """
        is_perps = token_type.lower() == "perps"
        actor = "Large traders" if is_perps else "Whales"
        actor_lc = "large trader" if is_perps else "whale"
        
        if whale_metrics is None:
            return 0.0, [f"{actor} metrics not available"]
        
        score = 0.0
        reasons = []
        
        try:
            whale_state = getattr(whale_metrics, 'whale_state', None)
            
            # For EXIT: distribution = urgent exit, accumulation = may wait
            if whale_state == "Distribution":
                score += 0.5
                reasons.append(f"{actor} distributing - exit urgently")
            elif whale_state == "Accumulation":
                score -= 0.3
                reasons.append(f"{actor} accumulating - may wait for higher")
            elif whale_state == "Stability":
                reasons.append(f"{actor} activity stable")
            
            # Check net flow
            net_flow = getattr(whale_metrics, 'whale_net_volume', 0)
            
            if net_flow < 0:
                score += 0.3
                reasons.append(f"Net {actor_lc} outflow ({net_flow:,.0f}) - exit signal")
            elif net_flow > 0:
                score -= 0.2
                reasons.append(f"Net {actor_lc} inflow (+{net_flow:,.0f}) - may wait")
            
            # Check buy/sell volume distribution
            buy_volume = getattr(whale_metrics, 'whale_buy_volume', 0)
            sell_volume = getattr(whale_metrics, 'whale_sell_volume', 0)
            
            if buy_volume + sell_volume > 0:
                sell_ratio = sell_volume / (buy_volume + sell_volume)
                
                if sell_ratio > self.config.WHALE_STRONG_ACCUMULATION:
                    score += 0.4
                    reasons.append(f"Strong selling pressure ({sell_ratio*100:.0f}%) - exit now")
                elif sell_ratio < self.config.WHALE_STRONG_DISTRIBUTION:
                    score -= 0.2
                    reasons.append(f"Strong buying ({(1-sell_ratio)*100:.0f}%) - may wait")
        
        except Exception as e:
            logger.warning(f"Error in exit whale analysis: {e}")
            reasons.append("Exit whale analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _make_exit_decision(
        self,
        final_score: float,
        all_scores: Dict[str, float],
        all_reasons: Dict[str, List[str]],
        safety_overrides: Optional[List[Dict[str, Any]]] = None
    ) -> TimingSignal:
        """
        Make EXIT timing decision based on aggregated score
        
        For exit:
        - Positive score = good time to exit
        - Negative score = wait for better exit
        - Very negative = urgent exit (things are bad)
        
        When safety_overrides are present, the EXIT_NOW threshold is lowered
        to EXIT_SAFETY_OVERRIDE_THRESHOLD, reflecting the urgency already
        established by the safety override (e.g., extreme concentration / rug pull risk).
        """
        # Use a lower EXIT_NOW threshold when safety overrides are active,
        # so the timing engine doesn't endlessly recommend WAIT_TO_EXIT
        # when the main signal already established urgency.
        has_safety_overrides = bool(safety_overrides)
        if has_safety_overrides:
            exit_now_threshold = self.config.EXIT_SAFETY_OVERRIDE_THRESHOLD
            logger.info(f"Safety overrides present — using lowered exit threshold "
                       f"({exit_now_threshold} instead of {self.config.EXIT_WAIT_THRESHOLD})")
        else:
            exit_now_threshold = self.config.EXIT_WAIT_THRESHOLD
        
        # Determine exit recommendation
        if final_score <= self.config.EXIT_URGENT_THRESHOLD:
            # Very negative = things are crashing, exit urgently
            recommendation = TimingRecommendation.EXIT
            wait_minutes = 0
        elif final_score >= exit_now_threshold:
            # Score exceeds threshold = exit now
            recommendation = TimingRecommendation.EXIT_NOW
            wait_minutes = 0
        else:
            # In between = wait for better exit
            recommendation = TimingRecommendation.WAIT_TO_EXIT
            wait_minutes = self._estimate_exit_wait_time(final_score, all_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(final_score, all_scores)
        
        # Compile primary reasons
        reason = self._compile_exit_reason(recommendation, final_score, all_scores, all_reasons)
        
        # Estimate exit timing impact
        potential_improvement, miss_risk = self._estimate_exit_timing_impact(
            recommendation, final_score, all_scores
        )
        
        return TimingSignal(
            recommendation=recommendation,
            wait_minutes=wait_minutes,
            confidence=confidence,
            reason=reason,
            potential_improvement_pct=potential_improvement,
            miss_risk_pct=miss_risk,
            momentum_score=all_scores.get('momentum', 0.0),
            volume_score=all_scores.get('volume', 0.0),
            liquidity_score=all_scores.get('liquidity', 0.0),
            smart_money_score=all_scores.get('smart_money', 0.0),
            whale_score=all_scores.get('whale', 0.0),
            final_score=final_score,
            component_reasons=all_reasons
        )
    
    def _estimate_exit_wait_time(self, score: float, all_scores: Optional[Dict[str, float]] = None) -> int:
        """
        Estimate how long to wait before exiting based on score and component analysis.
        
        Dynamic calculation: Uses individual component scores to fine-tune exit
        wait time (e.g. heavy selling = exit faster, whale distributing = exit faster).
        Falls back to config-based thresholds if component scores are unavailable.
        
        Works identically for meme and perps tokens since both share EntryTimingEngine.
        """
        # Dynamic calculation from component results
        if all_scores:
            try:
                momentum = all_scores.get('momentum', 0.0)
                volume = all_scores.get('volume', 0.0)
                whale = all_scores.get('whale', 0.0)
                smart_money = all_scores.get('smart_money', 0.0)
                
                # Base wait: scale from score magnitude
                base_wait = int(abs(score) * 60)
                
                adjustment = 0
                
                # Heavy selling by others = exit faster
                if volume > 0.3:
                    adjustment -= 10
                
                # Whale distributing = exit faster
                if whale > 0.3:
                    adjustment -= 10
                
                # Price rising = good exit point, reduce wait
                if momentum > 0.3:
                    adjustment -= 5
                elif momentum < -0.3:
                    adjustment += 5
                
                # Smart money selling = exit faster
                if smart_money > 0.3:
                    adjustment -= 5
                
                dynamic_wait = base_wait + adjustment
                
                # Clamp between config min/max as safety bounds
                return max(self.config.EXIT_WAIT_TIME_SHORT,
                           min(self.config.EXIT_WAIT_TIME_LONG, dynamic_wait))
            except Exception:
                pass  # Fall through to config-based fallback
        
        # Fallback: use TimingConfig thresholds
        if score >= self.config.EXIT_SCORE_MEDIUM:
            return self.config.EXIT_WAIT_TIME_SHORT
        elif score >= self.config.EXIT_SCORE_SHORT:
            return self.config.EXIT_WAIT_TIME_MEDIUM
        else:
            return self.config.EXIT_WAIT_TIME_LONG
    
    def _compile_exit_reason(
        self,
        recommendation: TimingRecommendation,
        final_score: float,
        all_scores: Dict[str, float],
        all_reasons: Dict[str, List[str]]
    ) -> str:
        """Compile exit timing reason"""
        sorted_components = sorted(
            all_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]
        
        top_reasons = []
        for component, score in sorted_components:
            component_reasons = all_reasons.get(component, [])
            if component_reasons:
                top_reasons.append(component_reasons[0])
        
        if not top_reasons:
            if recommendation == TimingRecommendation.EXIT:
                return "Market conditions deteriorating - exit immediately"
            elif recommendation == TimingRecommendation.EXIT_NOW:
                return "Good exit conditions - sell now"
            else:
                return "Wait for better exit price"
        
        return "; ".join(top_reasons)
    
    def _estimate_exit_timing_impact(
        self,
        recommendation: TimingRecommendation,
        final_score: float,
        all_scores: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Estimate potential price improvement by waiting to exit.
        Values are score-based heuristics; for WAIT_TO_EXIT, miss_risk is
        capped relative to potential_improvement (bounded ratio).

        Returns:
            Tuple of (potential_improvement_pct, miss_risk_pct)
            - potential_improvement: how much better price you might get by waiting
            - miss_risk: risk of price dropping further if you wait
        """
        if recommendation == TimingRecommendation.EXIT:
            # Urgent = high risk of further drop
            potential_improvement = 0.0
            miss_risk = min(25.0, abs(final_score) * 30.0)
        elif recommendation == TimingRecommendation.WAIT_TO_EXIT:
            # Floor improvement; cap miss_risk by improvement (bounded ratio)
            potential_improvement = max(1.0, min(8.0, abs(final_score) * 5.0 + 2.0))
            miss_risk = max(3.0, min(20.0, potential_improvement * 2.5))
        else:
            # EXIT_NOW = balanced
            potential_improvement = max(0.5, 3.0 - final_score * 2.0)
            miss_risk = max(3.0, 8.0 - final_score * 5.0)
        
        return potential_improvement, miss_risk
    
    def _insufficient_exit_data_response(self) -> TimingSignal:
        """Return neutral exit signal when data is insufficient"""
        return TimingSignal(
            recommendation=TimingRecommendation.EXIT_NOW,
            wait_minutes=0,
            confidence=0.3,
            reason="Insufficient data for exit timing - recommend immediate exit",
            potential_improvement_pct=0.0,
            miss_risk_pct=10.0,
            momentum_score=0.0,
            volume_score=0.0,
            liquidity_score=0.0,
            smart_money_score=0.0,
            whale_score=0.0,
            final_score=0.0,
            component_reasons={}
        )
    
    def _analyze_momentum(
        self,
        candles_df: pd.DataFrame,
        technical_signals: Optional[Any]
    ) -> Tuple[float, List[str]]:
        """
        Analyze price momentum to determine entry timing
        
        Returns:
            Tuple of (score, list of reasons)
            - Positive score: good time to enter
            - Negative score: wait for better entry
        """
        score = 0.0
        reasons = []
        
        try:
            # Get close prices using helper (handles both meme and perps)
            close_prices = self._get_close_prices(candles_df)
            
            if close_prices is None:
                return 0.0, ["Close price data not available"]
            
            # Get RSI from technical signals or calculate
            if technical_signals and hasattr(technical_signals, 'rsi'):
                rsi = technical_signals.rsi
            else:
                rsi = self._calculate_rsi(close_prices)
            
            if rsi is not None:
                # Check RSI against thresholds
                if rsi > self.config.RSI_EXTREME_HIGH:
                    score -= 0.6
                    reasons.append(f"RSI extremely overbought ({rsi:.1f})")
                elif rsi > self.config.RSI_OVERBOUGHT:
                    score -= 0.3
                    reasons.append(f"RSI overbought ({rsi:.1f})")
                elif rsi < self.config.RSI_EXTREME_LOW:
                    score += 0.6
                    reasons.append(f"RSI extremely oversold ({rsi:.1f})")
                elif rsi < self.config.RSI_OVERSOLD:
                    score += 0.3
                    reasons.append(f"RSI oversold ({rsi:.1f})")
            
            # Short-window (15m) momentum: spike -> WAIT, dip -> ENTER
            candle_count = len(close_prices)
            n_candles_15m = 3 if candle_count > 100 else 1  # 5m: 3 candles = 15m; 1h: 1 candle approx
            if len(close_prices) >= n_candles_15m + 1:
                recent_close = float(close_prices[-1])
                past_close_15m = float(close_prices[-(n_candles_15m + 1)])
                if past_close_15m > 0:
                    short_momentum = (recent_close - past_close_15m) / past_close_15m
                    if short_momentum >= self.config.MOMENTUM_SPIKE_UP_THRESHOLD:
                        score -= 0.4
                        reasons.append(
                            f"Price up {short_momentum*100:.1f}% (last 15m) - wait for pullback"
                        )
                    elif short_momentum <= self.config.MOMENTUM_DIP_DOWN_THRESHOLD:
                        score += 0.5
                        reasons.append(
                            f"Price down {abs(short_momentum)*100:.1f}% (last 15m) - dip opportunity"
                        )

            # Calculate price momentum over the full window (caller should pass 24h-filtered candles)
            recent_close = float(close_prices[-1])
            past_close = float(close_prices[0])
            
            # Calculate the approximate time window for the label
            if candle_count > 100:
                window_hours = candle_count * 5 / 60  # 5-minute candles
            else:
                window_hours = candle_count  # 1-hour candles
            
            if past_close > 0:
                momentum = (recent_close - past_close) / past_close
                
                # Format time window label
                if window_hours >= 24:
                    time_label = f"{window_hours:.0f}h"
                elif window_hours >= 1:
                    time_label = f"{window_hours:.1f}h"
                else:
                    time_label = f"{window_hours*60:.0f}m"
                
                if momentum > self.config.MOMENTUM_STRONG_UP:
                    score -= 0.4
                    reasons.append(f"Price up {momentum*100:.1f}% ({time_label}) - wait for pullback")
                elif momentum > self.config.MOMENTUM_MILD_UP:
                    score -= 0.2
                    reasons.append(f"Price up {momentum*100:.1f}% ({time_label}) - mild FOMO risk")
                elif momentum < self.config.MOMENTUM_STRONG_DOWN:
                    score += 0.5
                    reasons.append(f"Price down {abs(momentum)*100:.1f}% ({time_label}) - strong dip opportunity")
                elif momentum < self.config.MOMENTUM_MILD_DOWN:
                    score += 0.3
                    reasons.append(f"Price down {abs(momentum)*100:.1f}% ({time_label}) - potential dip entry")
        
        except Exception as e:
            logger.warning(f"Error in momentum analysis: {e}")
            reasons.append("Momentum analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _analyze_volume(self, candles_df: pd.DataFrame, recent_analysis: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
        """
        Analyze volume patterns to determine entry timing.
        Falls back to 24h aggregate buy/sell from recent_analysis when per-candle data is missing.
        
        Returns:
            Tuple of (score, list of reasons)
        """
        score = 0.0
        reasons = []
        
        try:
            # Get buy/sell volume using helpers (handles both meme and perps)
            buy_vol_series = self._get_buy_volume(candles_df)
            sell_vol_series = self._get_sell_volume(candles_df)
            
            if buy_vol_series is None or sell_vol_series is None:
                if recent_analysis and recent_analysis.get("buy_pressure_24h") is not None:
                    buy_ratio = float(recent_analysis.get("buy_pressure_24h", 0.5))
                    if buy_ratio > self.config.VOLUME_BUY_DOMINANT:
                        score += 0.3
                        reasons.append(f"Buy dominant ({buy_ratio*100:.0f}% buy, 24h aggregate)")
                    elif buy_ratio < self.config.VOLUME_SELL_DOMINANT:
                        score -= 0.3
                        reasons.append(f"Sell dominant ({(1-buy_ratio)*100:.0f}% sell, 24h aggregate)")
                    else:
                        reasons.append(f"Balanced volume ({buy_ratio*100:.0f}% buy, 24h aggregate)")
                    return max(-1.0, min(1.0, score)), reasons
                return 0.0, ["Volume data not available"]
            
            # Use last N candles for analysis
            window = min(60, len(candles_df))
            
            buy_vol = float(buy_vol_series.tail(window).sum())
            sell_vol = float(sell_vol_series.tail(window).sum())
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                buy_ratio = buy_vol / total_vol
                
                if buy_ratio > self.config.VOLUME_BUY_DOMINANT:
                    score += 0.3
                    reasons.append(f"Buy dominant ({buy_ratio*100:.0f}% buy volume)")
                elif buy_ratio < self.config.VOLUME_SELL_DOMINANT:
                    score -= 0.3
                    reasons.append(f"Sell dominant ({(1-buy_ratio)*100:.0f}% sell volume)")
            
            # Detect volume surge
            volume_series = self._get_volume(candles_df)
            close_prices = self._get_close_prices(candles_df)
            
            if volume_series is not None and len(candles_df) >= 20:
                avg_volume = float(volume_series.tail(60).mean()) if len(candles_df) >= 60 else float(volume_series.mean())
                current_volume = float(volume_series.tail(5).mean())
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    
                    if volume_ratio > self.config.VOLUME_SURGE_THRESHOLD and close_prices is not None and len(close_prices) >= 6:
                        # Check if price is rising or falling with surge
                        price_change = (float(close_prices[-1]) - float(close_prices[-6])) / float(close_prices[-6])
                        
                        if price_change > 0.02:
                            score -= 0.3
                            reasons.append(f"Volume surge {volume_ratio:.1f}x with rising price - FOMO risk")
                        elif price_change < -0.02:
                            score += 0.3
                            reasons.append(f"Volume surge {volume_ratio:.1f}x with falling price - absorption")
                        else:
                            reasons.append(f"Volume surge {volume_ratio:.1f}x (neutral)")
                    elif volume_ratio < 0.5:
                        score -= 0.1
                        reasons.append(f"Low volume ({volume_ratio:.1f}x avg) - poor liquidity")
        
        except Exception as e:
            logger.warning(f"Error in volume analysis: {e}")
            reasons.append("Volume analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _analyze_liquidity(self, birdeye_data: Optional[Dict]) -> Tuple[float, List[str]]:
        """
        Analyze liquidity conditions from Birdeye data
        
        Returns:
            Tuple of (score, list of reasons)
        """
        if birdeye_data is None or not birdeye_data.get('configured', False):
            return 0.0, ["Birdeye data not available"]
        
        score = 0.0
        reasons = []
        
        try:
            # Get liquidity from price_data or market_data
            liquidity = 0.0
            market_cap = 0.0
            
            if birdeye_data.get('price_data'):
                liquidity = birdeye_data['price_data'].get('liquidity', 0) or 0
            
            if birdeye_data.get('market_data'):
                market_cap = birdeye_data['market_data'].get('marketCap', 0) or 0
                if liquidity == 0:
                    liquidity = birdeye_data['market_data'].get('liquidity', 0) or 0
            
            # Analyze liquidity levels
            if liquidity > self.config.LIQUIDITY_HIGH:
                score += 0.3
                reasons.append(f"High liquidity (${liquidity:,.0f})")
            elif liquidity < self.config.LIQUIDITY_LOW:
                score -= 0.5
                reasons.append(f"Low liquidity (${liquidity:,.0f}) - high slippage risk")
            else:
                reasons.append(f"Moderate liquidity (${liquidity:,.0f})")
            
            # Analyze liquidity to market cap ratio
            if market_cap > 0 and liquidity > 0:
                ratio = liquidity / market_cap
                
                if ratio > self.config.LIQUIDITY_MCAP_HEALTHY:
                    score += 0.2
                    reasons.append(f"Healthy liq/mcap ratio ({ratio*100:.1f}%)")
                elif ratio < self.config.LIQUIDITY_MCAP_THIN:
                    score -= 0.3
                    reasons.append(f"Thin liq/mcap ratio ({ratio*100:.1f}%) - rug risk")
        
        except Exception as e:
            logger.warning(f"Error in liquidity analysis: {e}")
            reasons.append("Liquidity analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _analyze_smart_money(self, birdeye_data: Optional[Dict]) -> Tuple[float, List[str]]:
        """
        Analyze top trader (smart money) behavior from Birdeye data
        
        Returns:
            Tuple of (score, list of reasons)
        """
        if birdeye_data is None or not birdeye_data.get('configured', False):
            return 0.0, ["Birdeye data not available"]
        
        score = 0.0
        reasons = []
        
        try:
            # Analyze top traders
            top_traders = birdeye_data.get('top_traders', [])
            
            if top_traders:
                # Birdeye API returns volumeBuy/volumeSell (not buyVolume/sellVolume)
                total_buy = sum(t.get('volumeBuy', 0) or 0 for t in top_traders[:10])
                total_sell = sum(t.get('volumeSell', 0) or 0 for t in top_traders[:10])
                
                if total_buy + total_sell > 0:
                    buy_ratio = total_buy / (total_buy + total_sell)
                    
                    if buy_ratio > self.config.SMART_MONEY_BUY_DOMINANT:
                        score += 0.4
                        reasons.append(f"Smart money buying ({buy_ratio*100:.0f}% buy)")
                    elif buy_ratio < self.config.SMART_MONEY_SELL_DOMINANT:
                        score -= 0.4
                        reasons.append(f"Smart money selling ({(1-buy_ratio)*100:.0f}% sell)")
                    else:
                        reasons.append("Smart money neutral")
            
            # Analyze trade data (8h vs 24h comparison)
            trade_data = birdeye_data.get('trade_data')
            
            if trade_data:
                # Birdeye API returns volume_buy_8h format (not buy8h)
                buy_8h = trade_data.get('volume_buy_8h', 0) or trade_data.get('buy8h', 0) or 0
                sell_8h = trade_data.get('volume_sell_8h', 0) or trade_data.get('sell8h', 0) or 0
                buy_24h = trade_data.get('volume_buy_24h', 0) or trade_data.get('buy24h', 0) or 0
                sell_24h = trade_data.get('volume_sell_24h', 0) or trade_data.get('sell24h', 0) or 0
                
                # Calculate momentum acceleration
                if buy_24h + sell_24h > 0 and buy_8h + sell_8h > 0:
                    ratio_24h = buy_24h / (buy_24h + sell_24h)
                    ratio_8h = buy_8h / (buy_8h + sell_8h)
                    
                    if ratio_8h > ratio_24h * self.config.SMART_MONEY_ACCELERATION:
                        score += 0.3
                        reasons.append("Buy pressure accelerating (8h > 24h)")
                    elif ratio_8h < ratio_24h * self.config.SMART_MONEY_DECELERATION:
                        score -= 0.3
                        reasons.append("Buy pressure decelerating (8h < 24h)")
        
        except Exception as e:
            logger.warning(f"Error in smart money analysis: {e}")
            reasons.append("Smart money analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _analyze_whale(self, whale_metrics: Optional[Any], token_type: str = "meme") -> Tuple[float, List[str]]:
        """
        Analyze whale / large trader behavior from WhaleMetrics.
        
        Uses perps-appropriate terminology ('large traders') for perps tokens
        and 'whales' for meme tokens.
        
        Returns:
            Tuple of (score, list of reasons)
        """
        is_perps = token_type.lower() == "perps"
        actor = "Large traders" if is_perps else "Whales"
        actor_lc = "large trader" if is_perps else "whale"
        
        if whale_metrics is None:
            return 0.0, [f"{actor} metrics not available"]
        
        score = 0.0
        reasons = []
        
        try:
            # Check whale state
            whale_state = getattr(whale_metrics, 'whale_state', None)
            
            if whale_state == "Accumulation":
                score += 0.5
                reasons.append(f"{actor} accumulating - bullish")
            elif whale_state == "Distribution":
                score -= 0.5
                reasons.append(f"{actor} distributing - bearish")
            elif whale_state == "Stability":
                reasons.append(f"{actor} activity stable")
            
            # Check net volume
            net_flow = getattr(whale_metrics, 'whale_net_volume', 0)
            
            if net_flow > 0:
                score += 0.2
                reasons.append(f"Net {actor_lc} inflow (+{net_flow:,.0f})")
            elif net_flow < 0:
                score -= 0.2
                reasons.append(f"Net {actor_lc} outflow ({net_flow:,.0f})")
            
            # Check buy vs sell volume
            buy_volume = getattr(whale_metrics, 'whale_buy_volume', 0)
            sell_volume = getattr(whale_metrics, 'whale_sell_volume', 0)
            
            if buy_volume + sell_volume > 0:
                buy_ratio = buy_volume / (buy_volume + sell_volume)
                
                if buy_ratio > self.config.WHALE_STRONG_ACCUMULATION:
                    score += 0.3
                    reasons.append(f"Strong buying pressure ({buy_ratio*100:.0f}%)")
                elif buy_ratio < self.config.WHALE_STRONG_DISTRIBUTION:
                    score -= 0.3
                    reasons.append(f"Strong selling pressure ({(1-buy_ratio)*100:.0f}%)")
        
        except Exception as e:
            logger.warning(f"Error in whale analysis: {e}")
            reasons.append("Whale analysis error")
        
        return max(-1.0, min(1.0, score)), reasons
    
    def _aggregate_scores(self, all_scores: Dict[str, float]) -> float:
        """
        Aggregate all component scores using configured weights
        
        Args:
            all_scores: Dict mapping component name to score
            
        Returns:
            Final weighted score
        """
        final_score = 0.0
        
        for component, weight in self.weights.items():
            score = all_scores.get(component, 0.0)
            final_score += score * weight
        
        return final_score
    
    def _make_decision(
        self,
        final_score: float,
        all_scores: Dict[str, float],
        all_reasons: Dict[str, List[str]]
    ) -> TimingSignal:
        """
        Make timing decision based on aggregated score
        
        Args:
            final_score: Aggregated score from all components
            all_scores: Individual component scores
            all_reasons: Reasons from each component
            
        Returns:
            TimingSignal with complete recommendation
        """
        # Determine recommendation
        if final_score <= self.config.WAIT_THRESHOLD:
            recommendation = TimingRecommendation.WAIT
            wait_minutes = self._estimate_wait_time(final_score, all_scores)
        elif final_score >= self.config.URGENT_THRESHOLD:
            recommendation = TimingRecommendation.ENTER_NOW
            wait_minutes = 0
        elif final_score < 0:
            # Negative score: components suggest caution (e.g. "wait for pullback")
            recommendation = TimingRecommendation.WAIT
            wait_minutes = self._estimate_wait_time(final_score, all_scores)
        else:
            recommendation = TimingRecommendation.ENTER_NOW
            wait_minutes = 0
        
        # Calculate confidence
        confidence = self._calculate_confidence(final_score, all_scores)
        
        # Compile primary reasons
        reason = self._compile_primary_reason(recommendation, final_score, all_scores, all_reasons)
        
        # Estimate potential improvement and miss risk
        potential_improvement, miss_risk = self._estimate_timing_impact(
            recommendation, final_score, all_scores
        )
        
        return TimingSignal(
            recommendation=recommendation,
            wait_minutes=wait_minutes,
            confidence=confidence,
            reason=reason,
            potential_improvement_pct=potential_improvement,
            miss_risk_pct=miss_risk,
            momentum_score=all_scores.get('momentum', 0.0),
            volume_score=all_scores.get('volume', 0.0),
            liquidity_score=all_scores.get('liquidity', 0.0),
            smart_money_score=all_scores.get('smart_money', 0.0),
            whale_score=all_scores.get('whale', 0.0),
            final_score=final_score,
            component_reasons=all_reasons
        )
    
    def _estimate_wait_time(self, score: float, all_scores: Optional[Dict[str, float]] = None) -> int:
        """
        Estimate how long to wait based on score magnitude and component analysis.
        
        Dynamic calculation: Uses individual component scores to fine-tune
        wait time (e.g. momentum dip = enter sooner, whale distribution = wait longer).
        Falls back to config-based thresholds if component scores are unavailable.
        
        Works identically for meme and perps tokens since both share EntryTimingEngine.
        """
        if score >= self.config.WAIT_THRESHOLD:
            return 0
        
        # Dynamic calculation from component results
        if all_scores:
            try:
                momentum = all_scores.get('momentum', 0.0)
                volume = all_scores.get('volume', 0.0)
                whale = all_scores.get('whale', 0.0)
                smart_money = all_scores.get('smart_money', 0.0)
                liquidity = all_scores.get('liquidity', 0.0)
                
                # Base wait: scale from score magnitude (range ~0.2-1.0 -> ~12-60 min)
                base_wait = int(abs(score) * 60)
                
                adjustment = 0
                
                # Strong downward momentum = dip in progress -> enter sooner
                if momentum < -0.3:
                    adjustment -= 5
                elif momentum > 0.1:
                    adjustment += 5
                
                # High sell volume = capitulation may be near -> enter sooner
                if volume < -0.3:
                    adjustment -= 5
                
                # Whale accumulation despite negative score -> enter sooner
                if whale > 0.2:
                    adjustment -= 5
                elif whale < -0.3:
                    adjustment += 10
                
                # Smart money buying -> enter sooner
                if smart_money > 0.2:
                    adjustment -= 5
                elif smart_money < -0.3:
                    adjustment += 5
                
                # Low liquidity -> wait longer (slippage risk)
                if liquidity < -0.3:
                    adjustment += 5
                
                dynamic_wait = base_wait + adjustment
                
                # Clamp between config min/max as safety bounds
                return max(self.config.WAIT_TIME_SHORT,
                           min(self.config.WAIT_TIME_LONG, dynamic_wait))
            except Exception:
                pass  # Fall through to config-based fallback
        
        # Fallback: use TimingConfig thresholds
        if score >= self.config.WAIT_SCORE_SHORT:
            return self.config.WAIT_TIME_SHORT
        elif score >= self.config.WAIT_SCORE_MEDIUM:
            return self.config.WAIT_TIME_MEDIUM
        else:
            return self.config.WAIT_TIME_LONG
    
    def _calculate_confidence(
        self,
        final_score: float,
        all_scores: Dict[str, float]
    ) -> float:
        """
        Calculate confidence in the recommendation
        
        Based on:
        - Score magnitude (stronger = more confident)
        - Score agreement (more components agreeing = more confident)
        """
        # Base confidence from score magnitude
        magnitude_confidence = min(0.9, abs(final_score) * 2 + 0.3)
        
        # Agreement bonus: count how many components agree with direction
        positive_count = sum(1 for s in all_scores.values() if s > 0.1)
        negative_count = sum(1 for s in all_scores.values() if s < -0.1)
        total_active = positive_count + negative_count
        
        if total_active > 0:
            agreement = max(positive_count, negative_count) / total_active
            agreement_bonus = (agreement - 0.5) * 0.2  # -0.1 to +0.1
        else:
            agreement_bonus = 0.0
        
        confidence = max(0.3, min(0.95, magnitude_confidence + agreement_bonus))
        
        return confidence
    
    def _compile_primary_reason(
        self,
        recommendation: TimingRecommendation,
        final_score: float,
        all_scores: Dict[str, float],
        all_reasons: Dict[str, List[str]]
    ) -> str:
        """Compile the top 2 most impactful reasons into a reason string"""
        # Find the two most impactful components
        sorted_components = sorted(
            all_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]
        
        top_reasons = []
        for component, score in sorted_components:
            component_reasons = all_reasons.get(component, [])
            if component_reasons:
                top_reasons.append(component_reasons[0])
        
        if not top_reasons:
            if recommendation == TimingRecommendation.WAIT:
                return "Market conditions suggest waiting for better entry"
            elif recommendation == TimingRecommendation.ENTER_NOW:
                return "Good entry conditions - act now"
            else:
                return "Market conditions support entry"
        
        return "; ".join(top_reasons)
    
    def _estimate_timing_impact(
        self,
        recommendation: TimingRecommendation,
        final_score: float,
        all_scores: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Estimate the potential improvement by waiting and risk of missing.
        Values are score-based heuristics; for WAIT, miss_risk is capped relative
        to potential_improvement so the ratio is bounded (e.g. at most 2.5x).

        Returns:
            Tuple of (potential_improvement_pct, miss_risk_pct)
        """
        if recommendation == TimingRecommendation.WAIT:
            # Floor improvement so WAIT never shows tiny %; cap miss_risk by improvement (bounded ratio)
            potential_improvement = max(1.0, min(5.0, abs(final_score) * 4.0))
            miss_risk = max(3.0, min(15.0, potential_improvement * 2.5))
        elif recommendation == TimingRecommendation.ENTER_NOW and final_score >= self.config.URGENT_THRESHOLD:
            # Strong entry signal - low improvement potential, high miss risk
            potential_improvement = 0.5
            miss_risk = min(30.0, final_score * 40.0)
        else:
            # ENTER_NOW - balanced
            potential_improvement = max(0.5, abs(final_score) * 2.0)
            miss_risk = max(3.0, 10.0 - abs(final_score) * 5.0)
        
        return potential_improvement, miss_risk
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> Optional[float]:
        """
        Calculate RSI from price array
        
        Args:
            prices: Array of closing prices
            period: RSI period (default 14)
            
        Returns:
            RSI value or None if insufficient data
        """
        if len(prices) < period + 1:
            return None
        
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_gain == 0 and avg_loss == 0:
                return 50.0
            if avg_loss == 0:
                return 99.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except Exception:
            return None
    
    def _insufficient_data_response(self) -> TimingSignal:
        """Return neutral signal when data is insufficient"""
        return TimingSignal(
            recommendation=TimingRecommendation.ENTER_NOW,
            wait_minutes=0,
            confidence=0.3,
            reason="Insufficient data for timing analysis - using neutral recommendation",
            potential_improvement_pct=0.0,
            miss_risk_pct=0.0,
            momentum_score=0.0,
            volume_score=0.0,
            liquidity_score=0.0,
            smart_money_score=0.0,
            whale_score=0.0,
            final_score=0.0,
            component_reasons={}
        )


    # ------------------------------------------------------------------
    # Public API for coerced-WAIT scenarios (called from predict.py)
    # ------------------------------------------------------------------

    def estimate_coerced_wait_minutes(
        self,
        *,
        reason: str,
        timing_signal: "TimingSignal",
        recent_analysis: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Return a data-driven wait estimate when predict.py coerces an
        immediate recommendation to WAIT.

        ``reason`` selects the proxy-score strategy:
          * ``"hold_enter_now"``  – ENTER_NOW overridden because signal is HOLD
          * ``"degraded_wait"``   – WAIT with missing whale/holder data
          * ``"invariant_exit_to_wait"`` – exit-style rec coerced for non-SELL signal

        The returned minutes are a *suggested re-check window*, not a
        guarantee that market conditions will flip.
        """
        all_scores = {
            "momentum": timing_signal.momentum_score,
            "volume": timing_signal.volume_score,
            "liquidity": timing_signal.liquidity_score,
            "smart_money": timing_signal.smart_money_score,
            "whale": timing_signal.whale_score,
        }
        fs = timing_signal.final_score

        if reason == "hold_enter_now":
            # Engine said ENTER_NOW (fs ≥ URGENT_THRESHOLD) but signal is HOLD.
            # Map to a proxy below WAIT_THRESHOLD; stronger ENTER_NOW → less
            # negative proxy → shorter wait (the edge is "hotter", re-check sooner).
            overshoot = max(0.0, fs - self.config.URGENT_THRESHOLD)
            depth = max(0.0, 0.55 - overshoot * 1.0)
            proxy = self.config.WAIT_THRESHOLD - 0.05 - depth
            base = self._estimate_wait_time(proxy, all_scores)

        elif reason == "degraded_wait":
            if fs < self.config.WAIT_THRESHOLD:
                base = self._estimate_wait_time(fs, all_scores)
            else:
                proxy = self.config.WAIT_THRESHOLD - 0.05 - min(0.35, abs(fs) * 0.5)
                base = self._estimate_wait_time(proxy, all_scores)

        elif reason == "invariant_exit_to_wait":
            # Coerced from exit-style rec: use exit wait estimation with a
            # proxy score in the exit WAIT_TO_EXIT band.
            mid = (self.config.EXIT_URGENT_THRESHOLD + self.config.EXIT_WAIT_THRESHOLD) / 2
            proxy = mid + min(0.15, abs(fs) * 0.3)
            base = self._estimate_exit_wait_time(proxy, all_scores)

        else:
            base = self.config.WAIT_TIME_MEDIUM

        # Optional recent_analysis adjustment (meme-friendly)
        adjustment = 0
        if recent_analysis:
            vol = recent_analysis.get("volatility_24h")
            if vol is not None and vol > 0.04:
                adjustment += min(self.config.COERCED_WAIT_VOLATILITY_BONUS,
                                  int((vol - 0.04) * 200))
            trend = recent_analysis.get("trend_24h", "")
            if trend.startswith("strongly_"):
                adjustment -= 3

        minutes = base + adjustment
        return max(self.config.WAIT_TIME_SHORT,
                   min(self.config.COERCED_WAIT_MAX_MINUTES, minutes))


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    print("Entry Timing Engine - Test")
    print("=" * 60)
    
    # Generate sample candle data
    np.random.seed(42)
    n = 500
    
    # Simulated price with trend
    base_price = 0.0001
    trend = np.cumsum(np.random.randn(n) * 0.000001)
    close = base_price + trend + np.abs(np.random.randn(n) * 0.000001)
    
    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1s'),
        'Open': close * 0.999,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.exponential(1000000, n),
        'BuyVolume': np.random.exponential(600000, n),
        'SellVolume': np.random.exponential(400000, n)
    })
    
    # Test the engine
    engine = EntryTimingEngine()
    signal = engine.analyze(candles_df=df)
    
    print(f"\nTiming Analysis Result:")
    print(f"  Recommendation: {signal.recommendation.value}")
    print(f"  Confidence: {signal.confidence:.2f}")
    print(f"  Wait Minutes: {signal.wait_minutes}")
    print(f"  Reason: {signal.reason}")
    print(f"\nComponent Scores:")
    print(f"  Momentum: {signal.momentum_score:.3f}")
    print(f"  Volume: {signal.volume_score:.3f}")
    print(f"  Liquidity: {signal.liquidity_score:.3f}")
    print(f"  Smart Money: {signal.smart_money_score:.3f}")
    print(f"  Whale: {signal.whale_score:.3f}")
    print(f"  Final Score: {signal.final_score:.3f}")
    print(f"\nPotential Improvement: {signal.potential_improvement_pct:.1f}%")
    print(f"Miss Risk: {signal.miss_risk_pct:.1f}%")

