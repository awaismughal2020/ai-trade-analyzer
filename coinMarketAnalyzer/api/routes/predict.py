"""
Prediction API Routes
Endpoints for meme and perps token predictions
"""

import asyncio
import dataclasses
import logging
import time
from typing import Optional, Dict, Any, Union, List, Literal
from datetime import datetime, timedelta
from pathlib import Path

import sentry_sdk
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import TokenType, get_config, meme_config, perps_config, safety_config, ALGO_VERSION, PROMPT_VERSION
from api.validators import validate_token_address, validate_user_address, validate_date_field
from api.response_schema import filter_response_by_token_type
from core.circuit_breaker import sentry_fallback_warning

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request model for prediction"""
    token_address: str = Field(..., min_length=2, max_length=44, description="Token address or ticker symbol")
    token_type: Literal["meme", "perps"] = Field("meme", description="Token type: 'meme' or 'perps'")
    user_address: Optional[str] = Field(None, min_length=32, max_length=44, description="User wallet address for profiling")
    user_from_date: Optional[str] = Field(None, description="User trades start date (YYYY-MM-DD)")
    user_to_date: Optional[str] = Field(None, description="User trades end date (YYYY-MM-DD)")

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, values):
        if isinstance(values, dict):
            if "token_type" in values and isinstance(values["token_type"], str):
                values["token_type"] = values["token_type"].lower().strip()
        return values

    @model_validator(mode="after")
    def check_formats(self) -> "PredictionRequest":
        self.token_address = validate_token_address(self.token_address, self.token_type)
        if self.user_address:
            self.user_address = validate_user_address(self.user_address, self.token_type)
        if self.user_from_date:
            self.user_from_date = validate_date_field(self.user_from_date)
        if self.user_to_date:
            self.user_to_date = validate_date_field(self.user_to_date)
        return self


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    model_config = {"extra": "allow"}
    
    signal: str
    confidence: float
    token_address: str
    token_type: str
    timestamp: str
    summary: str
    algo_version: str = ALGO_VERSION
    prompt_version: str = PROMPT_VERSION
    
    # Signal effectiveness window (±24h)
    signal_effectiveness: Optional[Dict[str, Any]] = None
    
    # Layer breakdown
    layer_breakdown: Optional[Dict[str, Any]] = None
    layers_used: Optional[int] = None
    agreement_level: Optional[Union[str, float]] = None
    
    # Whale metrics
    whale_buy_volume: Optional[float] = None
    whale_sell_volume: Optional[float] = None
    whale_net_volume: Optional[float] = None
    whale_volume_unit: Optional[str] = None
    whale_data_source: Optional[str] = None
    is_whale_data_stale: Optional[bool] = None
    
    # Dominant whale tracking
    dominant_whale_status: Optional[str] = None
    dominant_whale_inactive_holding_pct: Optional[float] = None
    dominant_whale_aging_holding_pct: Optional[float] = None
    top_holder_last_activity_hours: Optional[float] = None
    
    # Holder metrics
    gini_coefficient: Optional[float] = None
    top10_hold_percent: Optional[float] = None
    dev_hold_percent: Optional[float] = None
    sniper_hold_percent: Optional[float] = None
    phase: Optional[str] = None
    
    # Safety overrides
    safety_overrides_applied: Optional[List[Dict[str, Any]]] = None
    
    # Perps-specific
    perps_metrics: Optional[Dict[str, Any]] = None
    user_perps_profile: Optional[Dict[str, Any]] = None
    
    # Timing
    timing: Optional[Dict[str, Any]] = None
    
    # Candle data freshness/source
    candle_data_freshness: Optional[Dict[str, Any]] = None
    candle_data_source: Optional[str] = None
    
    # Data quality
    data_quality: Optional[Dict[str, Any]] = None
    
    # User assessment
    user_risk_assessment: Optional[Dict[str, Any]] = None
    user_analytics: Optional[Dict[str, Any]] = None

    # Execution time (server-side)
    execution_time_seconds: Optional[float] = None


# =============================================================================
# Confidence Grade Mapping
# =============================================================================

def _confidence_grade(confidence: float) -> str:
    """Map a raw confidence value (0-1) to a human-readable grade."""
    if confidence >= 0.80:
        return "Strong"
    elif confidence >= 0.65:
        return "Moderate"
    elif confidence >= 0.50:
        return "Cautious"
    return "Weak"


# =============================================================================
# Components (lazy loaded)
# =============================================================================

_components: Optional[Dict[str, Any]] = None


def get_components():
    """Initialize and return system components (lazy loading)"""
    global _components
    
    if _components is not None:
        return _components
    
    logger.info("Initializing prediction components...")
    
    from core.data_fetcher import DataFetcher, PerpsDataFetcher
    from core.data_fetcher_birdeye import BirdeyeFetcher
    from core.market_context_fetcher import MarketContextFetcher
    from engines.whale_engine import WhaleEngine
    from engines.technical_engine import TechnicalIndicatorEngine
    from engines.holder_metrics import HolderMetricsCalculator
    from engines.entry_timing import EntryTimingEngine
    from engines.user_profiler import UserProfiler
    from engines.risk_assessor import RiskAssessor
    from generators.signal_generator import SignalGenerator
    from generators.summary_generator import SummaryGenerator
    from services.openai_service import OpenAIService
    
    config = get_config()
    
    # Create shared instances so SignalGenerator can reference them for Layer 2
    data_fetcher = DataFetcher(limiter_name="predict")
    birdeye_fetcher = BirdeyeFetcher()
    
    _components = {
        "config": config,
        "data_fetcher": data_fetcher,
        "perps_fetcher": PerpsDataFetcher(),
        "birdeye_fetcher": birdeye_fetcher,
        "whale_engine": WhaleEngine(),
        "technical_engine": TechnicalIndicatorEngine(),
        "holder_metrics": HolderMetricsCalculator(),
        "entry_timing": EntryTimingEngine(),
        "user_profiler": UserProfiler(),
        "risk_assessor": RiskAssessor(),
        "signal_generator": SignalGenerator(data_fetcher=data_fetcher, birdeye_fetcher=birdeye_fetcher),
        "summary_generator": SummaryGenerator(),
        "openai_service": OpenAIService(),
        "market_context_fetcher": MarketContextFetcher()
    }
    
    logger.info("Prediction components initialized")
    
    return _components


# =============================================================================
# Prediction Endpoints
# =============================================================================

@router.post("/", response_model=PredictionResponse, response_model_exclude_none=True, summary="Get Trading Signal")
async def predict(request: PredictionRequest):
    """
    Generate a trading signal for a token
    
    This endpoint analyzes the token using multiple layers:
    1. ML Model (XGBoost)
    2. Whale Engine
    3. Technical Indicators
    4. Holder Metrics
    5. User Profile (if address provided)
    
    Returns a weighted signal (BUY/SELL/HOLD) with confidence and breakdown.
    """
    try:
        start = time.perf_counter()
        # candle_days and holder_limit are fixed internally for optimal fresh data
        # 10 days of 5m candles = 2,880 candles — plenty for all indicators and always fresh
        from config import meme_config
        result = await run_prediction(
            token_address=request.token_address,
            token_type=request.token_type,
            candle_days=meme_config.DEFAULT_CANDLE_DAYS,
            holder_limit=meme_config.DEFAULT_HOLDER_LIMIT,
            user_address=request.user_address,
            user_from_date=request.user_from_date,
            user_to_date=request.user_to_date
        )
        result["execution_time_seconds"] = round(time.perf_counter() - start, 4)
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("prediction_request", {
                "token_address": request.token_address,
                "token_type": request.token_type,
                "user_address": request.user_address,
                "user_from_date": request.user_from_date,
                "user_to_date": request.user_to_date,
            })
            scope.set_context("error_details", {
                "traceback": traceback.format_exc(),
            })
            sentry_sdk.capture_exception(e)
        logger.error(f"Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/{token_address}", response_model=PredictionResponse, response_model_exclude_none=True, summary="Get Trading Signal (GET)")
async def predict_get(
    token_address: str,
    token_type: str = Query("meme", description="Token type: 'meme' or 'perps'"),
    user_address: Optional[str] = Query(None)
):
    """
    Generate a trading signal for a token (GET method)
    
    Same as POST /predict but uses URL parameters.
    """
    token_type = token_type.lower().strip()
    try:
        start = time.perf_counter()
        from config import meme_config
        result = await run_prediction(
            token_address=token_address,
            token_type=token_type,
            candle_days=meme_config.DEFAULT_CANDLE_DAYS,
            holder_limit=meme_config.DEFAULT_HOLDER_LIMIT,
            user_address=user_address
        )
        result["execution_time_seconds"] = round(time.perf_counter() - start, 4)
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("prediction_request", {
                "token_address": token_address,
                "token_type": token_type,
                "user_address": user_address,
            })
            scope.set_context("error_details", {
                "traceback": traceback.format_exc(),
            })
            sentry_sdk.capture_exception(e)
        logger.error(f"Prediction error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# =============================================================================
# Core Prediction Logic
# =============================================================================

async def run_prediction(
    token_address: str,
    token_type: str = "meme",
    candle_days: int = 10,
    holder_limit: int = 1000,
    user_address: Optional[str] = None,
    user_from_date: Optional[str] = None,
    user_to_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete prediction pipeline for BOTH meme and perps tokens.
    
    ±24h Signal Effectiveness Strategy (applies to BOTH meme and perps):
    
    Data Fetching (recent history for fresh, accurate analysis):
    - Meme: 10 days of 5-min candles (2,880 candles), full holder/whale data
    - Perps: 500 hourly candles (~21 days), 200 funding records
    
    24h-Focused Analysis (for signal effectiveness):
    - Technical indicators (RSI, EMA, MACD, BB) computed on FULL dataset for accuracy
    - Whale/large trader analysis scoped to last 24h (meme: WHALE_DELTA_WINDOW_HOURS=24, perps: 1-day trade lookback)
    - Timing analysis uses 24h-filtered candles (meme: last 288 5-min candles, perps: last 24 1h candles)
    - 24h price change, volume, volatility, buy/sell pressure computed as recent_analysis
    - Summary text explicitly framed in 24h context
    - Signal output explicitly states it's effective for the next 24 hours
    """
    # Default user date range: last 3 months when user context is used but dates not provided
    if user_address and (user_from_date is None or user_to_date is None):
        to_dt = datetime.utcnow()
        from_dt = to_dt - timedelta(days=90)
        user_from_date = user_from_date or from_dt.strftime('%Y-%m-%dT00:00:00Z')
        user_to_date = user_to_date or to_dt.strftime('%Y-%m-%dT23:59:59Z')

    components = get_components()
    
    is_perps = token_type.lower() == "perps"
    
    # Initialize Layer 2 if user_address provided (ensure data_fetcher is wired)
    if user_address:
        components["signal_generator"].set_data_fetcher(components["data_fetcher"])
    
    logger.info(f"Running prediction for {token_address} (type={token_type})")
    
    # Fetch global market context (cached, non-blocking)
    try:
        market_context = await asyncio.to_thread(
            components["market_context_fetcher"].fetch_latest
        )
    except Exception as e:
        logger.warning(f"Failed to fetch market context: {e}")
        from core.market_context_fetcher import MarketContextFetcher
        market_context = MarketContextFetcher.get_empty_context()
    
    # Fetch data based on token type
    # NOTE: Full historical data is fetched for indicator calculation
    # but whale/large trader data is already scoped to 24h in the fetcher
    if is_perps:
        data = await _fetch_perps_data(components, token_address, user_address, user_from_date, user_to_date)
    else:
        data = await _fetch_meme_data(components, token_address, candle_days, holder_limit)
    
    # Calculate technical indicators on FULL dataset for robust analysis
    # RSI, EMA, MACD, Bollinger Bands need deep history for accuracy
    tech_signals = components["technical_engine"].calculate_all_indicators(data['candles'])
    
    # Determine staleness threshold based on token type
    stale_threshold = perps_config.CANDLE_STALE_THRESHOLD_HOURS if is_perps else meme_config.CANDLE_STALE_THRESHOLD_HOURS
    
    # Compute 24h-focused analysis metrics from the full candle data.
    # NOTE: The summary text and signal_effectiveness.recent_24h_analysis both
    # derive from this same candle set. Differences from external UIs (e.g.
    # Birdeye) may be due to rolling-window vs calendar-day, candle granularity,
    # or data-source timing.
    recent_analysis = _compute_recent_24h_analysis(data['candles'], is_perps, stale_threshold_hours=stale_threshold)

    # When candle window is shorter than 24h, prefer Birdeye's 24h change when available
    window_actual_hours = recent_analysis.get("window_actual_hours")
    if window_actual_hours is not None and window_actual_hours < 23:
        birdeye_24h = (data.get("market_data") or {}).get("price_change_24h_pct_birdeye")
        if birdeye_24h is not None:
            recent_analysis["price_change_24h_pct"] = round(float(birdeye_24h), 2)
            recent_analysis["price_change_24h_source"] = "birdeye"
            pct = recent_analysis["price_change_24h_pct"]
            if pct > 5:
                recent_analysis["trend_24h"] = "strongly_bullish"
            elif pct > 1:
                recent_analysis["trend_24h"] = "bullish"
            elif pct < -5:
                recent_analysis["trend_24h"] = "strongly_bearish"
            elif pct < -1:
                recent_analysis["trend_24h"] = "bearish"
            else:
                recent_analysis["trend_24h"] = "sideways"
        else:
            recent_analysis["price_change_24h_source"] = "candles"
    else:
        recent_analysis["price_change_24h_source"] = "candles"

    # When candles show flat price (range 0) but Birdeye has a 24h price change, use it for trend
    if (recent_analysis.get("price_24h_range_pct") or 0) == 0 and recent_analysis.get("trend_24h") == "sideways":
        birdeye_pct = (data.get("market_data") or {}).get("price_change_24h_pct_birdeye")
        if birdeye_pct is not None and birdeye_pct != 0:
            pct = round(float(birdeye_pct), 2)
            recent_analysis["price_change_24h_pct"] = pct
            recent_analysis["price_change_24h_source"] = "birdeye"
            if pct > 5:
                recent_analysis["trend_24h"] = "strongly_bullish"
            elif pct > 1:
                recent_analysis["trend_24h"] = "bullish"
            elif pct < -5:
                recent_analysis["trend_24h"] = "strongly_bearish"
            elif pct < -1:
                recent_analysis["trend_24h"] = "bearish"

    # When using Birdeye 24h trade data, enrich recent_analysis if candle-based volume/buy-sell are missing
    whale_data_src = data.get("whale_data_source") or ""
    if whale_data_src == "birdeye_24h":
        wm = data.get("whale_metrics")
        if wm is not None:
            buy24h = getattr(wm, "whale_buy_volume", 0) or 0
            sell24h = getattr(wm, "whale_sell_volume", 0) or 0
            total_vol = buy24h + sell24h
            vol_total = (recent_analysis.get("volume_24h_total") or 0)
            if total_vol > 0 and (vol_total <= 0 or recent_analysis.get("buy_sell_data_source") == "unavailable"):
                if vol_total <= 0:
                    recent_analysis["volume_24h_total"] = round(float(total_vol), 2)
                    n_candles = recent_analysis.get("candles_analyzed") or 1
                    recent_analysis["volume_24h_avg"] = round(float(total_vol) / max(1, n_candles), 2)
                recent_analysis["buy_sell_data_source"] = "birdeye_24h"
                recent_analysis["buy_pressure_24h"] = round(buy24h / total_vol, 4)
                recent_analysis["sell_pressure_24h"] = round(sell24h / total_vol, 4)
                recent_analysis["net_flow_24h"] = round(buy24h - sell24h, 2)
                # When buy is 0, ratio is 0 (100% sell); downstream should treat 0 as no buy pressure
                recent_analysis["buy_sell_ratio_24h"] = round(buy24h / (sell24h + 1e-10), 4)
    elif whale_data_src == "no_data":
        if (recent_analysis.get("volume_24h_total") or 0) <= 0 and recent_analysis.get("buy_sell_data_source") == "unavailable":
            recent_analysis["buy_sell_data_source"] = "no_data"

    # Generate trading signal (uses full indicators + 24h whale data + market context)
    trading_signal = components["signal_generator"].generate_signal(
        whale_metrics=data['whale_metrics'],
        holder_stats=data['holder_stats'],
        technical_signals=tech_signals,
        summary="",
        candle_data=data['candle_data'],
        market_data=data.get('market_data') or {},
        user_address=user_address,
        token_type=token_type,
        perps_user_profile=data.get('user_perps_profile'),
        user_from_date=user_from_date,
        user_to_date=user_to_date,
        market_context=market_context,
        recent_analysis=recent_analysis,
        target_mint=token_address,
    )
    
    # Store recent analysis on the trading signal
    trading_signal.recent_analysis = recent_analysis
    
    # Fetch user analytics if user_address provided
    user_analytics = None
    if user_address and not is_perps:
        # Meme: Fetch full user analytics via data_fetcher (mirrors HTS Layer 2)
        try:
            user_profile_data = await asyncio.to_thread(
                components["data_fetcher"].fetch_user_complete_profile,
                wallet_address=user_address,
                from_date=user_from_date,
                to_date=user_to_date
            )
            if user_profile_data and user_profile_data.get('summary'):
                summary_data = user_profile_data['summary']
                user_analytics = {
                    "wallet_address": user_address,
                    "lifetime_pnl": round(summary_data.get('lifetime_pnl', 0), 2),
                    "total_volume_usd": round(summary_data.get('total_volume_usd', 0), 2),
                    "max_drawdown_pct": round(summary_data.get('max_drawdown_pct', 0), 4),
                    "avg_r_multiple": round(summary_data.get('avg_r_multiple', 0), 4),
                    "avg_holding_time_minutes": round(summary_data.get('avg_holding_time_minutes', 0), 2),
                    "trades_count": len(user_profile_data.get('trades', [])),
                    "source": "meme"
                }
                if summary_data.get('win_rate') is not None:
                    user_analytics["win_rate"] = summary_data['win_rate']
                if summary_data.get('total_closed_pnl') is not None:
                    user_analytics["total_closed_pnl"] = round(summary_data['total_closed_pnl'], 2)
                if summary_data.get('total_fees') is not None:
                    user_analytics["total_fees"] = round(summary_data['total_fees'], 4)
                if summary_data.get('avg_leverage') is not None:
                    user_analytics["avg_leverage"] = round(summary_data['avg_leverage'], 2)
                if summary_data.get('bot_detected') is not None:
                    user_analytics["bot_detected"] = summary_data['bot_detected']
                logger.info(f"User analytics included: wallet={user_address[:8]}..., PnL=${summary_data.get('lifetime_pnl', 0):.2f}")
        except Exception as e:
            logger.warning(f"Failed to fetch user analytics: {e}")
    elif user_address and is_perps:
        # Perps: Build normalized user_analytics from already-fetched perps profile
        user_perps_profile = data.get('user_perps_profile')
        if user_perps_profile:
            user_analytics = {
                "wallet_address": user_address,
                "lifetime_pnl": user_perps_profile.get('total_pnl', 0),
                "total_volume_usd": user_perps_profile.get('total_volume', 0),
                "win_rate": user_perps_profile.get('win_rate', 0),
                "trades_count": user_perps_profile.get('total_trades', 0),
                "trader_type": user_perps_profile.get('trader_type', 'UNKNOWN'),
                "direction_bias": user_perps_profile.get('direction_bias', 'UNKNOWN'),
                "profit_factor": user_perps_profile.get('profit_factor', 0),
                "avg_trade_size": user_perps_profile.get('avg_trade_size', 0),
                "trades_per_day": user_perps_profile.get('trades_per_day', 0),
                "source": "perps"
            }
            logger.info(f"Perps user analytics included: wallet={user_address[:8]}..., PnL=${user_perps_profile.get('total_pnl', 0):.2f}")
    
    # Generate summary (will include 24h context, perps-aware, degraded-aware)
    summary = components["summary_generator"].generate_summary(
        whale_metrics=data['whale_metrics'],
        holder_stats=data['holder_stats'],
        technical_signals=tech_signals,
        recent_analysis=recent_analysis,
        token_type=token_type,
        degraded=data.get('degraded', False),
        market_context=market_context,
        market_data=data.get('market_data')
    )
    
    # Filter candles to last 24h for timing-specific analysis
    # Timing decisions should be based on recent price action, not 3-week-old data
    recent_candles = _filter_recent_candles(data['candles'], hours=24, is_perps=is_perps, stale_threshold_hours=stale_threshold)
    timing_candles = recent_candles if len(recent_candles) >= 10 else data['candles']
    
    # Get timing recommendation if entry timing engine is available
    # Use EXIT timing for SELL signals, ENTRY timing for BUY/HOLD signals
    timing_result = None
    # Build minimal birdeye_data for timing engine from available market_data
    _timing_birdeye_data = None
    _market = data.get("market_data") or {}
    _liq = _market.get("liquidity_usd")
    if _liq is not None and _liq > 0:
        _timing_birdeye_data = {
            "configured": True,
            "price_data": {"liquidity": _liq},
        }
    if components.get("entry_timing") and data.get('candles') is not None and not data.get('candles').empty:
        try:
            # Choose appropriate timing analysis based on signal
            # Use 24h-filtered candles for timing decisions (recent price action matters most)
            if trading_signal.signal == "SELL":
                # Use exit timing for SELL signals
                # Pass safety overrides so the timing engine can adjust its
                # EXIT_NOW threshold when the SELL was triggered by a safety override
                timing_signal = components["entry_timing"].analyze_exit_timing(
                    candles_df=timing_candles,
                    whale_metrics=data['whale_metrics'],
                    birdeye_data=_timing_birdeye_data,
                    technical_signals=tech_signals,
                    token_type=token_type,
                    safety_overrides=trading_signal.safety_overrides_applied,
                    recent_analysis=recent_analysis,
                )
                timing_type = "exit"
            else:
                # Use entry timing for BUY/HOLD signals
                timing_signal = components["entry_timing"].analyze(
                    candles_df=timing_candles,
                    whale_metrics=data['whale_metrics'],
                    birdeye_data=_timing_birdeye_data,
                    technical_signals=tech_signals,
                    token_type=token_type,
                    market_context=market_context,
                    recent_analysis=recent_analysis,
                )
                timing_type = "entry"
            
            # Convert TimingSignal to dict for response
            recommendation = timing_signal.recommendation.value if hasattr(timing_signal.recommendation, 'value') else str(timing_signal.recommendation)
            reason = timing_signal.reason
            confidence = timing_signal.confidence
            
            # Safety override: Adjust timing based on concerning metrics
            warnings = []
            whale_metrics = data.get('whale_metrics')
            holder_stats = data.get('holder_stats')
            # Use max(whale, holder) gini so timing text matches response gini (e.g. 1.0 when top10 ~100%)
            effective_gini = max(
                float(getattr(whale_metrics, 'gini_coefficient', 0) or 0) if whale_metrics else 0.0,
                float(getattr(holder_stats, 'gini_coefficient', 0) or 0) if holder_stats else 0.0,
            )
            
            # Meme-specific holder warnings (not applicable to perps)
            if not is_perps:
                # Check for extreme concentration (top 10 hold > 90%)
                if whale_metrics and getattr(whale_metrics, 'top10_hold_percent', 0) > 90:
                    warnings.append(f"CAUTION: Top 10 holders control {whale_metrics.top10_hold_percent:.1f}% of supply")
                    confidence *= 0.5  # Reduce confidence
                
                # Check for high Gini (inequality > 0.8) using effective gini
                if effective_gini > 0.8:
                    warnings.append(f"HIGH INEQUALITY: Gini coefficient {effective_gini:.2f}")
                
                # Check for inactive whales
                if whale_metrics and getattr(whale_metrics, 'dominant_whale_status', '') == 'FULLY_INACTIVE':
                    inactive_pct = getattr(whale_metrics, 'dominant_whale_inactive_holding_pct', 0)
                    if inactive_pct > 20:
                        warnings.append(f"INACTIVE WHALES: {inactive_pct:.1f}% held by inactive wallets")
                        confidence *= 0.7
            
            # Check for conflicting signals
            if timing_type == "entry" and recommendation == "ENTER_NOW":
                layer_breakdown = trading_signal.layer_breakdown or {}
                sell_signals = sum(1 for layer in layer_breakdown.values() if layer.get('signal') == 'SELL')
                if sell_signals >= 2:
                    warnings.append(f"CONFLICTING SIGNALS: {sell_signals} layers suggest SELL")
                    recommendation = "WAIT"
                    confidence *= 0.6
                    timing_signal.wait_minutes = components["entry_timing"].estimate_coerced_wait_minutes(
                        reason="hold_enter_now",
                        timing_signal=timing_signal,
                        recent_analysis=recent_analysis,
                    )
            
            # Align entry timing with HOLD signal — don't suggest entering when
            # the main signal says no position (fixes "Timing contradiction:
            # HOLD signal but ENTER_NOW recommendation")
            if timing_type == "entry" and recommendation == "ENTER_NOW" and trading_signal.signal == "HOLD":
                recommendation = "WAIT"
                confidence *= 0.7
                warnings.append("Timing aligned with signal: HOLD suggests no position — ENTER_NOW overridden to WAIT")
                timing_signal.wait_minutes = components["entry_timing"].estimate_coerced_wait_minutes(
                    reason="hold_enter_now",
                    timing_signal=timing_signal,
                    recent_analysis=recent_analysis,
                )
                logger.info(f"HOLD + ENTER_NOW override: recommendation changed to WAIT, wait={timing_signal.wait_minutes}m, confidence={confidence:.2f}")
            
            # Combine warnings with reason
            if warnings:
                reason = f"{reason}. {'; '.join(warnings)}"
            
            # Degraded mode: make timing conservative when whale/holder data is missing
            is_degraded = data.get('degraded', False)
            if is_degraded:
                if recommendation == "ENTER_NOW":
                    recommendation = "WAIT"
                    reason = f"[DEGRADED DATA] {reason}. Whale and holder data unavailable — timing based on price action only."
                elif recommendation == "EXIT_NOW":
                    reason = f"[DEGRADED DATA] {reason}. Whale and holder data unavailable."
                else:
                    reason = f"[DEGRADED DATA] {reason}. Analysis incomplete — whale/holder data unavailable."
                
                confidence *= 0.5
                
                # Ensure wait_minutes is set when recommendation is WAIT
                if recommendation == "WAIT" and (timing_signal.wait_minutes is None or timing_signal.wait_minutes == 0):
                    timing_signal.wait_minutes = components["entry_timing"].estimate_coerced_wait_minutes(
                        reason="degraded_wait",
                        timing_signal=timing_signal,
                        recent_analysis=recent_analysis,
                    )
                
                warnings.append("DEGRADED: Timing based on price/technical data only (no whale/holder data)")
                logger.info(f"Timing downgraded due to degraded data: {recommendation}, confidence={confidence:.2f}")
            
            # Invariant: exit-style timing should only accompany a SELL signal.
            # If a future refactor breaks this, coerce to WAIT so clients never
            # see contradictory "HOLD + EXIT" payloads.
            _EXIT_RECS = {"EXIT", "EXIT_NOW", "WAIT_TO_EXIT"}
            if trading_signal.signal != "SELL" and recommendation in _EXIT_RECS:
                logger.warning(
                    f"Timing invariant violation: signal={trading_signal.signal} "
                    f"but recommendation={recommendation} — coercing to WAIT"
                )
                recommendation = "WAIT"
                timing_type = "entry"
                confidence *= 0.5
                warnings.append(
                    f"Timing coerced from exit to WAIT (signal is {trading_signal.signal}, not SELL)"
                )
                timing_signal.wait_minutes = components["entry_timing"].estimate_coerced_wait_minutes(
                    reason="invariant_exit_to_wait",
                    timing_signal=timing_signal,
                    recent_analysis=recent_analysis,
                )

            # Classify urgency from the recommendation — immediate actions
            # should not expose a misleading "wait 0 minutes".
            _IMMEDIATE_RECS = {"ENTER_NOW", "EXIT_NOW", "EXIT"}
            timing_urgency = "immediate" if recommendation in _IMMEDIATE_RECS else "defer"
            effective_wait = None if timing_urgency == "immediate" else timing_signal.wait_minutes

            timing_result = {
                "type": timing_type,  # "entry" or "exit"
                "recommendation": recommendation,
                "confidence": confidence,
                "final_score": timing_signal.final_score,
                "reason": reason,
                "wait_minutes": effective_wait,
                "timing_urgency": timing_urgency,
                "potential_improvement_pct": timing_signal.potential_improvement_pct,
                "miss_risk_pct": timing_signal.miss_risk_pct,
                "warnings": warnings if warnings else None,
                "degraded": is_degraded,
                "component_scores": {
                    "momentum": timing_signal.momentum_score,
                    "volume": timing_signal.volume_score,
                    "liquidity": timing_signal.liquidity_score,
                    "smart_money": timing_signal.smart_money_score,
                    "whale": timing_signal.whale_score
                }
            }
        except Exception as e:
            logger.warning(f"Failed to generate timing recommendation: {e}")
    
    # Signal-timing contradiction: reduce BUY confidence when timing strongly disagrees
    timing_contradiction_applied = False
    if trading_signal.signal == "BUY" and timing_result:
        timing_score = timing_result.get("final_score", 0)
        if timing_score < safety_config.SIGNAL_TIMING_CONTRADICTION_THRESHOLD:
            original_conf = trading_signal.confidence
            penalty = safety_config.SIGNAL_TIMING_CONTRADICTION_PENALTY
            trading_signal.confidence = max(0.3, trading_signal.confidence * (1.0 - penalty))
            timing_contradiction_applied = True
            if timing_result.get("warnings") is None:
                timing_result["warnings"] = []
            timing_result["warnings"].append(
                f"SIGNAL-TIMING CONTRADICTION: BUY signal but timing score is negative "
                f"({timing_score:.2f}). Confidence reduced {original_conf:.2f} -> {trading_signal.confidence:.2f}"
            )
            logger.warning(
                f"Signal-timing contradiction: BUY + timing score {timing_score:.2f} "
                f"— confidence {original_conf:.2f} -> {trading_signal.confidence:.2f}"
            )

    # Build signal effectiveness metadata (±24h window)
    now = datetime.utcnow()
    effectiveness_hours = getattr(trading_signal, 'signal_effectiveness_hours', 24)
    # When we have partial candle data (< 24h), show actual analysis window so UI isn't misleading
    window_actual = recent_analysis.get("window_actual_hours") if recent_analysis else None
    if window_actual is not None and window_actual < 24:
        h_val = int(window_actual) if window_actual == int(window_actual) else round(window_actual, 1)
        analysis_window_label = f"Last {h_val} hours"
        data_strategy = "Candle data covers partial window; 24h price change from Birdeye when available. Signal calibrated for ±24h."
    else:
        analysis_window_label = f"Last {effectiveness_hours} hours"
        data_strategy = "Full historical data used for indicator calculation; signal calibrated for ±24h window"
    signal_effectiveness = {
        "effectiveness_hours": effectiveness_hours,
        "analysis_window": analysis_window_label,
        "prediction_window": f"Next {effectiveness_hours} hours",
        "signal_valid_from": (now - timedelta(hours=effectiveness_hours)).isoformat() + "Z",
        "signal_valid_until": (now + timedelta(hours=effectiveness_hours)).isoformat() + "Z",
        "data_strategy": data_strategy,
        "recent_24h_analysis": recent_analysis
    }
    
    # Compute risk management (stop-loss, take-profit, risk/reward)
    risk_mgmt = _compute_risk_management(
        signal=trading_signal.signal,
        tech_signals=tech_signals,
        recent_analysis=recent_analysis,
        is_perps=is_perps,
    )

    # Check if response is degraded (missing whale/holder data)
    is_degraded_response = data.get('degraded', False)
    
    # Build confidence analysis (before/after safety + grade)
    conf_before = getattr(trading_signal, 'confidence_before_safety', None)
    conf_after = trading_signal.confidence
    adjustments = []
    overrides = getattr(trading_signal, 'safety_overrides_applied', None) or []
    for ov in overrides:
        if ov.get('check') in (
            'extreme_volatility', 'extreme_price_swing', 'token_age_p1',
            'token_age_p1_few_holders', 'extreme_funding_rate', 'low_liquidity',
            'market_extreme_fear', 'market_extreme_greed', 'high_risk',
            'whale_distribution', 'top10_concentration', 'gini_coefficient',
            'high_gini_few_holders', 'holder_count', 'extreme_low_holders',
            'degraded_data_guard',
        ):
            adjustments.append(ov.get('reason', ov['check']))

    if timing_contradiction_applied:
        adjustments.append(
            f"Signal-timing contradiction penalty (-{safety_config.SIGNAL_TIMING_CONTRADICTION_PENALTY * 100:.0f}%)"
        )

    confidence_analysis = {
        "confidence_before_safety": round(conf_before, 4) if conf_before is not None else None,
        "confidence_after_safety": round(conf_after, 4),
        "confidence_grade": _confidence_grade(conf_after),
        "adjustments": adjustments if adjustments else None,
    }

    # Holder gini/top10: avoid showing gini=1.0 when top10 is low (inconsistent)
    _top10_pct = None if (is_perps or is_degraded_response) else (getattr(data['whale_metrics'], 'top10_hold_percent', 0) or getattr(data['holder_stats'], 'top10_concentration', 0))
    _whale_gini = float(getattr(data['whale_metrics'], 'gini_coefficient', 0) or 0)
    _holder_gini = float(getattr(data['holder_stats'], 'gini_coefficient', 0) or 0)
    if is_perps or is_degraded_response:
        _gini_val = None
    elif _top10_pct is not None and _top10_pct < 1.0:
        # Very low top10 with very high gini is inconsistent (likely different supply bases);
        # cap gini so we don't show "0.99 gini, 0.3% top10"
        _gini_val = min(max(_whale_gini, _holder_gini), 0.85)
    elif _top10_pct is not None and _top10_pct < 99.0:
        _gini_val = max(_whale_gini, min(_holder_gini, 0.99))
    else:
        _gini_val = max(_whale_gini, _holder_gini)

    # Build response
    response = {
        "signal": trading_signal.signal,
        "confidence": trading_signal.confidence,
        "confidence_grade": _confidence_grade(trading_signal.confidence),
        "confidence_analysis": confidence_analysis,
        "token_address": token_address,
        "token_type": token_type,
        "timestamp": now.isoformat() + "Z",
        "summary": summary,
        
        # Signal effectiveness window (±24h)
        "signal_effectiveness": signal_effectiveness,
        
        "layer_breakdown": trading_signal.layer_breakdown,
        "layers_used": trading_signal.layers_used,
        "agreement_level": trading_signal.agreement_level,
        
        # Whale metrics (based on last 24h of large trader activity)
        # When degraded, whale volumes are 0 (empty defaults) — show as 0, not null
        "whale_buy_volume": getattr(data['whale_metrics'], 'whale_buy_volume', 0),
        "whale_sell_volume": getattr(data['whale_metrics'], 'whale_sell_volume', 0),
        "whale_net_volume": getattr(data['whale_metrics'], 'whale_net_volume', 0),
        "whale_volume_unit": "USD" if is_perps else "tokens",
        "whale_data_source": data.get('whale_data_source', 'unknown'),
        "is_whale_data_stale": getattr(data['whale_metrics'], 'is_whale_data_stale', False),
        
        # Dominant whale tracking (meme only — null out when degraded to avoid empty defaults)
        "dominant_whale_status": None if (is_perps or is_degraded_response) else getattr(data['whale_metrics'], 'dominant_whale_status', None),
        "dominant_whale_inactive_holding_pct": None if (is_perps or is_degraded_response) else getattr(data['whale_metrics'], 'dominant_whale_inactive_holding_pct', None),
        "dominant_whale_aging_holding_pct": None if (is_perps or is_degraded_response) else getattr(data['whale_metrics'], 'dominant_whale_aging_holding_pct', None),
        "top_holder_last_activity_hours": None if (is_perps or is_degraded_response) else getattr(data['whale_metrics'], 'top_holder_last_activity_hours', None),
        
        # Holder metrics (meme only — null out when degraded to avoid misleading empty defaults)
        "gini_coefficient": _gini_val,
        "top10_hold_percent": _top10_pct,
        "dev_hold_percent": None if (is_perps or is_degraded_response) else getattr(data['whale_metrics'], 'dev_hold_percent', None),
        "sniper_hold_percent": None if (is_perps or is_degraded_response) else getattr(data['whale_metrics'], 'sniper_hold_percent', None),
        "phase": None if (is_perps or is_degraded_response) else getattr(data['whale_metrics'], 'phase', None),
        
        # Safety overrides
        "safety_overrides_applied": getattr(trading_signal, 'safety_overrides_applied', None),
        
        # Timing (based on last 24h of price action)
        "timing": timing_result,
        
        # Risk management suggestions (stop-loss, take-profit, R:R)
        "risk_management": risk_mgmt,
        
        # Candle data freshness indicator
        # Helps API consumers detect if analysis is based on stale data
        "candle_data_freshness": recent_analysis.get("data_freshness") if isinstance(recent_analysis, dict) else None,
        "candle_data_source": data.get('candle_data_source', 'internal_api'),
        
        # Data quality indicator (shows if response is degraded due to missing data)
        "data_quality": {
            "degraded": data.get('degraded', False),
            "reason": data.get('degraded_reason'),
            "candle_source": data.get('candle_data_source', 'internal_api'),
            "whale_source": data.get('whale_data_source', 'unknown'),
            "primary_source": data.get('primary_source'),  # "birdeye" when Birdeye-primary used, else None
        },
        
        # Global market context (CryptoRank)
        "market_context": market_context.to_dict() if hasattr(market_context, 'to_dict') else None
    }
    
    # Add perps-specific metrics
    if is_perps:
        response["perps_metrics"] = data.get('perps_metrics')
        response["user_perps_profile"] = data.get('user_perps_profile')
        response["whale_flow"] = data.get('whale_flow')
    
    # Add user assessment if user address provided
    if user_address:
        # User risk assessment comes from the signal generator's Layer 5
        if trading_signal.user_risk_assessment:
            response["user_risk_assessment"] = trading_signal.user_risk_assessment
        
        # User analytics (normalized format for both meme and perps)
        if user_analytics:
            response["user_analytics"] = user_analytics
    
    return filter_response_by_token_type(
        response, token_type,
        recurse_keys={"user_risk_assessment", "user_analytics"},
    )


async def _fetch_perps_data(
    components: Dict,
    ticker: str,
    user_address: Optional[str],
    user_from_date: Optional[str],
    user_to_date: Optional[str]
) -> Dict[str, Any]:
    """Fetch data for perps prediction"""
    perps_fetcher = components["perps_fetcher"]
    
    # Fetch perps data (fast mode: uses /historical-funding endpoint, short timeouts)
    # Run in thread to avoid blocking the event loop
    perps_data = await asyncio.to_thread(
        perps_fetcher.fetch_complete_perps_data,
        ticker=ticker,
        max_candles=500,
        max_funding=200,
        include_large_trader_analysis=True,
        fast_mode=True,
        large_trader_max_pages=2,
        market_data_timeout=10
    )
    
    candles = perps_data['candles']
    candle_source = 'internal_api'
    
    if len(candles) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
    
    # Check candle freshness for perps (no Birdeye fallback — Birdeye is Solana-specific)
    stale_threshold = perps_config.CANDLE_STALE_THRESHOLD_HOURS
    candle_freshness = _check_candle_freshness(candles, is_perps=True, stale_threshold_hours=stale_threshold)
    if candle_freshness.get("is_stale", False):
        candle_age = candle_freshness.get("latest_candle_age_hours", 0)
        logger.warning(
            f"Perps candle data is STALE ({candle_age:.1f}h old, threshold: {stale_threshold}h) "
            f"for {ticker}. No Birdeye fallback available for perps. "
            f"Analysis will use available data as-is."
        )
    
    # Get large trader metrics (whale equivalent for perps) — guard against None
    large_trader_metrics = perps_data.get('large_trader_metrics') or {}
    
    # Create whale metrics from large trader data
    whale_metrics = components["whale_engine"].create_empty_metrics()
    if large_trader_metrics:
        whale_metrics.whale_buy_volume = large_trader_metrics.get('total_buy_volume', 0)
        whale_metrics.whale_sell_volume = large_trader_metrics.get('total_sell_volume', 0)
        whale_metrics.whale_net_volume = large_trader_metrics.get('large_trader_net_volume', 0)
        whale_metrics.whale_state = large_trader_metrics.get('large_trader_state', 'Neutral')
    
    # Create holder stats (empty for perps)
    holder_stats = components["holder_metrics"].create_empty_stats()
    
    # User perps profile
    user_perps_profile = None
    if user_address:
        try:
            user_trades = await asyncio.to_thread(
                perps_fetcher.fetch_user_perps_trades,
                user_address=user_address,
                from_date=user_from_date,
                to_date=user_to_date,
                coin=ticker.split('-')[0] if '-' in ticker else ticker
            )
            if not user_trades.empty:
                user_perps_profile = perps_fetcher.analyze_user_perps_profile(user_trades)
        except Exception as e:
            logger.warning(f"Failed to fetch user perps profile: {e}")
    
    # Extract candle features
    candle_data = _extract_candle_features(candles)
    
    # Market data (guard against None — fetch_market_data returns None on timeout)
    market_data = perps_data.get('market_data') or {}
    if large_trader_metrics:
        market_data.update({
            'large_trader_state': large_trader_metrics.get('large_trader_state'),
            'large_trader_concentration': large_trader_metrics.get('top_concentration_pct'),
            'buy_sell_ratio': large_trader_metrics.get('buy_sell_ratio')
        })
    
    # Merge market-wide whale flow into market_data so ML features + safety overrides can use it
    whale_flow = perps_data.get('whale_flow') or {}
    if whale_flow.get('whale_flow_available'):
        market_data.update({
            'whale_flow_net': whale_flow.get('whale_flow_net', 0),
            'whale_flow_net_avg': whale_flow.get('whale_flow_net_avg', 0),
            'whale_flow_long_ratio': whale_flow.get('whale_flow_long_ratio', 0.5),
            'whale_flow_count_whales': whale_flow.get('whale_flow_count_whales', 0),
            'whale_flow_long_volume': whale_flow.get('whale_flow_long_volume', 0),
            'whale_flow_short_volume': whale_flow.get('whale_flow_short_volume', 0),
            'whale_flow_total_volume': whale_flow.get('whale_flow_total_volume', 0),
        })
    
    return {
        'candles': candles,
        'whale_metrics': whale_metrics,
        'holder_stats': holder_stats,
        'candle_data': candle_data,
        'market_data': market_data,
        'whale_data_source': 'perps_trades',
        'candle_data_source': candle_source,
        'candle_data_freshness': candle_freshness,
        'perps_metrics': large_trader_metrics,
        'whale_flow': whale_flow,
        'user_perps_profile': user_perps_profile
    }


def _convert_birdeye_to_standard_candles(birdeye_candles: list) -> pd.DataFrame:
    """
    Convert Birdeye OHLCV data to the standard candle DataFrame format
    used by the rest of the pipeline.
    
    Birdeye returns: [{unixTime, timestamp, o, h, l, c, v}, ...]
    Standard format: Timestamp, Open, High, Low, Close, Volume, etc.
    
    Note: BuyVolume/SellVolume are intentionally omitted because Birdeye OHLCV
    does not provide buy/sell splits. Downstream consumers handle the absence:
    - technical_engine defaults to volume*0.5 internally for buy_pressure
    - entry_timing reports "Volume data not available" (score=0) instead of misleading data
    
    Args:
        birdeye_candles: List of dicts from BirdeyeFetcher.fetch_ohlcv_range()
        
    Returns:
        Standardized DataFrame matching the format from DataFetcher.standardize_candles()
    """
    if not birdeye_candles:
        return pd.DataFrame()
    
    rows = []
    for c in birdeye_candles:
        volume = float(c.get('v', 0))
        rows.append({
            'Timestamp': pd.to_datetime(c['unixTime'], unit='s', utc=True),
            'Open': float(c.get('o', 0)),
            'High': float(c.get('h', 0)),
            'Low': float(c.get('l', 0)),
            'Close': float(c.get('c', 0)),
            'Volume': volume,
            # Birdeye OHLCV does not provide buy/sell volume split.
            # Omitting BuyVolume/SellVolume so downstream consumers
            # (technical_engine, entry_timing) handle the absence correctly:
            #   - technical_engine: defaults to volume*0.5 internally
            #   - entry_timing: returns "Volume data not available" (score=0)
            # This avoids a misleading 50/50 split appearing in the response.
            'BuyCount': 0,
            'SellCount': 0,
            'TotalSupply': 0,
            'SolUSDPrice': 0
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Timestamp').reset_index(drop=True)
    logger.info(f"Converted {len(df)} Birdeye candles to standard format "
                f"(range: {df['Timestamp'].min()} to {df['Timestamp'].max()})")
    return df


def _birdeye_holders_to_holder_df(holders: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert Birdeye holder list to DataFrame for holder_metrics and whale_engine.
    holder_metrics.calculate_all_metrics expects lastHolding or current_holding.

    Birdeye fetch_token_holders returns dicts with: wallet, balance, percentage, source
    """
    if not holders:
        return pd.DataFrame()
    rows = []
    for h in holders:
        balance = float(h.get("balance", 0) or 0)
        owner = h.get("wallet") or h.get("owner") or ""
        rows.append({"lastHolding": balance, "address": owner, "wallet": owner})
    df = pd.DataFrame(rows)
    return df


def _fetch_birdeye_candles_fallback(
    birdeye_fetcher,
    mint: str,
    candle_days: int,
    stale_threshold_hours: int = 24
) -> pd.DataFrame:
    """
    Fetch candle data from Birdeye as a fallback when the primary API returns stale data.
    
    Strategy: Birdeye limits results to ~1000 candles per request, so we use a tiered approach:
    1. Try 5m candles for the LAST 3 days (864 candles, within 1000 limit) — gives fresh, detailed data
    2. If that fails/is stale, try 1h candles for the full window — gives broad coverage
    3. If that also fails, try 15m candles for 7 days as a middle ground
    
    Args:
        birdeye_fetcher: BirdeyeFetcher instance
        mint: Token mint address
        candle_days: Number of days of candle history
        stale_threshold_hours: Hours after which data is considered stale (from config)
        
    Returns:
        Standardized candle DataFrame, or empty DataFrame if Birdeye fails
    """
    if birdeye_fetcher is None or not birdeye_fetcher.is_configured():
        logger.warning("Birdeye not configured - cannot use as candle data fallback")
        sentry_fallback_warning("birdeye", "Birdeye not configured — candle fallback unavailable", {"mint": mint[:8]})
        return pd.DataFrame()
    
    end_time = datetime.utcnow()
    best_df = pd.DataFrame()
    best_age = float('inf')
    
    # Tiered fallback: try increasingly coarser resolutions
    # Each tier stays within Birdeye's ~1000 candle limit
    attempts = [
        # Tier 1: 5m candles, last 3 days (864 candles) — most detailed, most recent
        {"interval": "5m", "days": 3, "label": "5m/3d"},
        # Tier 2: 15m candles, last 7 days (672 candles) — good balance
        {"interval": "15m", "days": min(candle_days, 7), "label": "15m/7d"},
        # Tier 3: 1h candles, full window (max ~240 for 10 days) — broadest coverage
        {"interval": "1H", "days": candle_days, "label": f"1h/{candle_days}d"},
    ]
    
    for attempt in attempts:
        try:
            start_time = end_time - timedelta(days=attempt["days"])
            label = attempt["label"]
            
            logger.info(f"BIRDEYE FALLBACK ({label}): Fetching candles for {mint[:8]}...")
            
            birdeye_candles = birdeye_fetcher.fetch_ohlcv_range(
                token_address=mint,
                start_time=start_time,
                end_time=end_time,
                interval=attempt["interval"]
            )
            
            if not birdeye_candles:
                logger.warning(f"Birdeye ({label}) returned no candle data for {mint[:8]}...")
                continue
            
            df = _convert_birdeye_to_standard_candles(birdeye_candles)
            
            if len(df) == 0:
                continue
            
            # Check freshness of the Birdeye data using the same threshold
            freshness = _check_candle_freshness(df, is_perps=False, stale_threshold_hours=stale_threshold_hours)
            age_hours = freshness.get("latest_candle_age_hours", 999)
            
            if not freshness.get("is_stale", True):
                logger.info(f"BIRDEYE FALLBACK SUCCESS ({label}): {len(df)} candles, {age_hours:.1f}h old — FRESH")
                return df
            else:
                logger.warning(f"Birdeye ({label}) data is {age_hours:.1f}h old — trying next tier...")
                # Track the freshest result as our best candidate
                if age_hours < best_age:
                    best_df = df
                    best_age = age_hours
        except Exception as e:
            logger.error(f"Birdeye ({attempt['label']}) fallback failed for {mint[:8]}...: {e}")
            continue
    
    # No tier returned fresh data — return the best we got (if any)
    if len(best_df) > 0:
        logger.warning(f"Birdeye: No fresh data found. Using best available ({best_age:.1f}h old, {len(best_df)} candles)")
        return best_df
    
    logger.warning(f"Birdeye: All fallback tiers failed for {mint[:8]}...")
    sentry_fallback_warning("birdeye", f"All Birdeye fallback tiers failed for {mint[:8]}...", {"mint": mint[:8]})
    return pd.DataFrame()


async def _fetch_meme_data_birdeye_primary(
    components: Dict,
    mint: str,
    candle_days: int,
    holder_limit: int
) -> Optional[Dict[str, Any]]:
    """
    Fetch meme data with Birdeye as primary source. Used when MEME_PRIMARY_DATA_SOURCE=birdeye.
    Runs critical path (candles + trade_data) first; on success runs token_holders, market_data, metadata in parallel.
    Returns unified dict compatible with _fetch_meme_data, or None to trigger internal fallback.
    """
    birdeye_fetcher = components.get("birdeye_fetcher")
    if not birdeye_fetcher or not birdeye_fetcher.is_configured():
        return None

    timeout_total = meme_config.MEME_BIRDEYE_PRIMARY_TIMEOUT_SECONDS
    timeout_per = meme_config.MEME_BIRDEYE_PER_CALL_TIMEOUT_SECONDS

    async def _run() -> Optional[Dict[str, Any]]:
        end_time = datetime.utcnow()
        # Critical path: candles (5m/3d) + trade_data
        start_5m = end_time - timedelta(days=3)
        try:
            candles_raw, trade_data = await asyncio.gather(
                asyncio.wait_for(
                    asyncio.to_thread(
                        birdeye_fetcher.fetch_ohlcv_range,
                        mint,
                        start_5m,
                        end_time,
                        "5m",
                    ),
                    timeout=timeout_per,
                ),
                asyncio.wait_for(
                    asyncio.to_thread(birdeye_fetcher.fetch_trade_data, mint),
                    timeout=timeout_per,
                ),
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Birdeye-primary critical path failed for {mint[:8]}...: {e}")
            return None

        candles = _convert_birdeye_to_standard_candles(candles_raw) if candles_raw else pd.DataFrame()
        if len(candles) == 0:
            logger.warning(f"Birdeye-primary: no candles for {mint[:8]}...")
            return None

        # Secondary: token_holders, liquidity, market_data (supply), metadata, creation_info
        try:
            holders_task = asyncio.wait_for(
                asyncio.to_thread(birdeye_fetcher.fetch_token_holders, mint, holder_limit),
                timeout=timeout_per,
            )
            price_liq_task = asyncio.wait_for(
                asyncio.to_thread(birdeye_fetcher.fetch_price_with_liquidity, mint),
                timeout=timeout_per,
            )
            market_task = asyncio.wait_for(
                asyncio.to_thread(birdeye_fetcher.fetch_market_data, mint),
                timeout=timeout_per,
            )
            meta_task = asyncio.wait_for(
                asyncio.to_thread(birdeye_fetcher.fetch_token_metadata, mint),
                timeout=timeout_per,
            )
            creation_task = asyncio.wait_for(
                asyncio.to_thread(birdeye_fetcher.fetch_token_creation_info, mint),
                timeout=timeout_per,
            )
            holders_list, price_liq, market_data_res, metadata_res, creation_info = await asyncio.gather(
                holders_task, price_liq_task, market_task, meta_task, creation_task
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Birdeye-primary secondary fetch failed for {mint[:8]}...: {e}")
            holders_list, price_liq, market_data_res, metadata_res, creation_info = [], None, None, None, None

        # Merge creation info into metadata so phase/age consumers see it
        if creation_info:
            if metadata_res is None:
                metadata_res = {}
            metadata_res.update(creation_info)

        holders_df = _birdeye_holders_to_holder_df(holders_list or [])
        total_supply = 1_000_000_000
        if market_data_res:
            total_supply = float(
                market_data_res.get("supply") or market_data_res.get("circulatingSupply") or total_supply
            )
        if total_supply <= 0 and len(holders_df) > 0:
            total_supply = float(holders_df["lastHolding"].sum()) or total_supply

        whale_metrics = components["whale_engine"].create_whale_metrics_from_birdeye(
            trade_data or {},
            holders_df,
            total_supply,
            metadata_res,
        )
        holder_stats = (
            components["holder_metrics"].calculate_all_metrics(holders_df, total_supply)
            if len(holders_df) > 0
            else components["holder_metrics"].create_empty_stats()
        )
        candle_data = _extract_candle_features(candles)
        meme_market_data: Dict[str, Any] = {}
        if price_liq and price_liq.get("liquidity") is not None:
            meme_market_data["liquidity_usd"] = float(price_liq["liquidity"])
        elif market_data_res and market_data_res.get("liquidity") is not None:
            meme_market_data["liquidity_usd"] = float(market_data_res["liquidity"])
        if price_liq and "priceChange24h" in price_liq:
            meme_market_data["price_change_24h_pct_birdeye"] = float(price_liq.get("priceChange24h", 0) or 0)

        return {
            "candles": candles,
            "whale_metrics": whale_metrics,
            "holder_stats": holder_stats,
            "candle_data": candle_data,
            "market_data": meme_market_data,
            "whale_data_source": getattr(whale_metrics, "whale_data_source", "birdeye_24h"),
            "candle_data_source": "birdeye",
            "primary_source": "birdeye",
            "metadata": metadata_res or {},
        }

    try:
        logger.info(f"Birdeye-primary: fetching meme data for {mint[:8]}...")
        result = await asyncio.wait_for(_run(), timeout=timeout_total)
        if result is not None:
            logger.info(f"Birdeye-primary: success for {mint[:8]}...")
        return result
    except asyncio.TimeoutError:
        logger.warning("Birdeye-primary fetch timed out; falling back to internal")
        return None
    except Exception as e:
        logger.warning(f"Birdeye-primary fetch failed: {e}; falling back to internal")
        return None


async def _fetch_meme_data(
    components: Dict,
    mint: str,
    candle_days: int,
    holder_limit: int
) -> Dict[str, Any]:
    """
    Fetch data for meme token prediction.
    
    When MEME_PRIMARY_DATA_SOURCE=birdeye, tries Birdeye first (candles, trade_data, holders, liquidity);
    on timeout, no candles, or stale candles falls back to internal API. Otherwise uses internal API first
    with Birdeye only for candle fallback and liquidity.
    
    Data strategy for ±24h signal effectiveness:
    - Recent candle history (candle_days, default 10 days of 5-min candles = 2,880 candles) for fresh data
    - If primary API returns stale candles, automatically falls back to Birdeye OHLCV
    - Whale engine internally uses WHALE_DELTA_WINDOW_HOURS=24 for delta calculation
    - The caller (run_prediction) will filter to 24h for timing analysis and compute recent_analysis
    """
    # Birdeye-primary path: try Birdeye first when configured; fall back to internal on failure/staleness
    primary_source = getattr(meme_config, "MEME_PRIMARY_DATA_SOURCE", "birdeye")
    if primary_source == "birdeye":
        birdeye_fetcher = components.get("birdeye_fetcher")
        if not birdeye_fetcher:
            logger.info("Birdeye-primary: skipped (no birdeye_fetcher in components)")
        elif not birdeye_fetcher.is_configured():
            logger.info("Birdeye-primary: skipped (Birdeye not configured — check BIRDEYE_API_KEY)")
        else:
            birdeye_result = await _fetch_meme_data_birdeye_primary(
                components, mint, candle_days, holder_limit
            )
            if birdeye_result is not None:
                stale_threshold = meme_config.CANDLE_STALE_THRESHOLD_HOURS
                freshness = _check_candle_freshness(
                    birdeye_result["candles"],
                    is_perps=False,
                    stale_threshold_hours=stale_threshold,
                )
                if not freshness.get("is_stale", True):
                    return birdeye_result
                logger.warning(
                    f"Birdeye-primary candles stale ({freshness.get('latest_candle_age_hours', 0):.1f}h); "
                    "falling back to internal"
                )
                sentry_fallback_warning(
                    "birdeye_primary",
                    "Birdeye-primary candles stale — falling back to internal API",
                    {"mint": mint[:8], "latest_candle_age_hours": freshness.get("latest_candle_age_hours")},
                )
            else:
                sentry_fallback_warning(
                    "birdeye_primary",
                    "Birdeye-primary failed or timed out — falling back to internal API",
                    {"mint": mint[:8]},
                )

    data_fetcher = components["data_fetcher"]
    
    # Fast-fail probe: check if internal API is reachable before launching 5 parallel requests
    api_reachable = await asyncio.to_thread(data_fetcher.is_api_reachable, 3)
    
    if api_reachable:
        # Fetch complete meme token data (run in thread to avoid blocking event loop)
        data = await asyncio.to_thread(
            data_fetcher.fetch_complete_data_v2,
            mint=mint,
            candle_days=candle_days,
            holder_limit=holder_limit
        )
    else:
        logger.warning(
            f"Internal API unreachable for {mint[:8]}... — skipping to Birdeye fallback"
        )
        sentry_fallback_warning(
            "internal_api",
            f"Internal API unreachable — falling back to Birdeye for candles",
            {"mint": mint[:8]},
        )
        data = {
            'candles': pd.DataFrame(),
            'user_holdings': pd.DataFrame(),
            'holders': pd.DataFrame(),
            'holders_historical': pd.DataFrame(),
            'metadata': None
        }
    
    candles = data['candles']
    candle_source = 'internal_api' if api_reachable else 'unavailable'
    stale_threshold = meme_config.CANDLE_STALE_THRESHOLD_HOURS
    
    # Check candle freshness and fallback to Birdeye if stale
    if len(candles) > 0:
        freshness = _check_candle_freshness(candles, is_perps=False, stale_threshold_hours=stale_threshold)
        primary_age = freshness.get("latest_candle_age_hours", 0)
        if freshness.get("is_stale", False):
            logger.warning(
                f"Primary API candle data is STALE ({primary_age:.1f}h old, threshold: {stale_threshold}h) "
                f"for {mint[:8]}... Attempting Birdeye fallback..."
            )
            birdeye_candles = await asyncio.to_thread(
                _fetch_birdeye_candles_fallback,
                components.get("birdeye_fetcher"),
                mint,
                candle_days,
                stale_threshold_hours=stale_threshold
            )
            if len(birdeye_candles) > 0:
                birdeye_freshness = _check_candle_freshness(birdeye_candles, is_perps=False, stale_threshold_hours=stale_threshold)
                birdeye_age = birdeye_freshness.get("latest_candle_age_hours", 999)
                
                # Use Birdeye if it's fresher than primary (even if not perfectly fresh)
                if birdeye_age < primary_age:
                    candles = birdeye_candles
                    data['candles'] = candles
                    candle_source = 'birdeye_fallback'
                    sentry_fallback_warning(
                        "birdeye",
                        f"Primary API candle data stale ({primary_age:.1f}h) — using Birdeye candles ({birdeye_age:.1f}h)",
                        {"mint": mint[:8], "primary_age_hours": primary_age, "birdeye_age_hours": birdeye_age},
                    )
                    if not birdeye_freshness.get("is_stale", True):
                        logger.info(f"Using Birdeye candle data (FRESH, {birdeye_age:.1f}h old, {len(candles)} candles)")
                    else:
                        logger.info(
                            f"Using Birdeye candle data (fresher: {birdeye_age:.1f}h vs primary {primary_age:.1f}h, "
                            f"{len(candles)} candles)"
                        )
                else:
                    logger.warning(
                        f"Birdeye data not fresher ({birdeye_age:.1f}h) than primary ({primary_age:.1f}h). "
                        f"Using primary API data."
                    )
            else:
                logger.warning("Birdeye fallback returned no data. Using primary API data as-is.")
                sentry_fallback_warning(
                    "birdeye",
                    "Birdeye fallback returned no data — using primary API data as-is",
                    {"mint": mint[:8]},
                )
    elif len(candles) == 0:
        # Primary returned nothing — try Birdeye directly
        logger.warning(f"Primary API returned 0 candles for {mint[:8]}... Trying Birdeye...")
        birdeye_candles = await asyncio.to_thread(
            _fetch_birdeye_candles_fallback,
            components.get("birdeye_fetcher"),
            mint,
            candle_days,
            stale_threshold_hours=stale_threshold
        )
        if len(birdeye_candles) > 0:
            candles = birdeye_candles
            data['candles'] = candles
            candle_source = 'birdeye_fallback'
            sentry_fallback_warning(
                "birdeye",
                f"Primary API returned 0 candles — using Birdeye ({len(candles)} candles)",
                {"mint": mint[:8]},
            )
            logger.info(f"Using Birdeye candle data ({len(candles)} candles)")
    
    if len(candles) == 0:
        raise HTTPException(status_code=404, detail=f"No candle data found for {mint} (tried primary API and Birdeye)")
    
    # HYBRID APPROACH for accurate analysis:
    # - Use 'holders' (current, filtered) for holder metrics & concentration
    # - Use 'holders_historical' for whale engine (whale trading activity)
    
    current_holders_df = data['holders']  # Current on-chain holders (filtered, no pools)
    historical_holders_df = data.get('holders_historical', pd.DataFrame())  # Historical trading data
    
    # For whale analysis: prefer user_holdings, fallback to historical, then current
    whale_analysis_df = data['user_holdings'] if len(data['user_holdings']) > 0 else historical_holders_df
    if len(whale_analysis_df) == 0:
        whale_analysis_df = current_holders_df
    
    # For holder metrics: use current holders (filtered, no pools)
    holder_metrics_df = current_holders_df
    
    # Degraded mode: if no holder data, return response with empty defaults
    # instead of a hard 404 (same pattern that perps uses)
    if len(holder_metrics_df) == 0:
        logger.warning(f"No holder data for {mint[:8]}... — using empty defaults (degraded mode)")
        whale_metrics = components["whale_engine"].create_empty_metrics()
        holder_stats = components["holder_metrics"].create_empty_stats()
        candle_data = _extract_candle_features(candles)
        return {
            'candles': candles,
            'whale_metrics': whale_metrics,
            'holder_stats': holder_stats,
            'candle_data': candle_data,
            'market_data': {},
            'whale_data_source': 'unavailable',
            'candle_data_source': candle_source,
            'degraded': True,
            'degraded_reason': 'Holder data unavailable from internal API'
        }
    
    logger.info(f"Using {len(whale_analysis_df)} holders for whale analysis, {len(holder_metrics_df)} for holder metrics")
    
    # Get total supply from holders data (from /mint/details) or fallback to metadata
    metadata = data.get('metadata') or {}

    # If metadata lacks creation time, enrich from Birdeye token_creation_info
    _creation_keys = {'CreatedAt', 'createdAt', 'created_at', 'blockUnixTime', 'blockHumanTime'}
    if not (_creation_keys & set(metadata.keys())):
        birdeye_fetcher = components.get("birdeye_fetcher")
        if birdeye_fetcher and birdeye_fetcher.is_configured():
            try:
                creation_info = await asyncio.to_thread(
                    birdeye_fetcher.fetch_token_creation_info, mint
                )
                if creation_info:
                    metadata.update(creation_info)
                    data['metadata'] = metadata
                    logger.info(f"Enriched metadata with Birdeye creation info for {mint[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to fetch Birdeye creation info for {mint[:8]}...: {e}")

    # Priority: Get totalSupply from current holders DataFrame (from /mint/details with clickhouse)
    if len(current_holders_df) > 0 and 'totalSupply' in current_holders_df.columns:
        total_supply = current_holders_df['totalSupply'].iloc[0]
        logger.info(f"Using totalSupply from /mint/details: {total_supply:,.0f}")
    else:
        total_supply = metadata.get('totalSupply', 1_000_000_000)
        logger.warning(f"Using fallback totalSupply: {total_supply:,.0f}")
        sentry_fallback_warning(
            "internal_api",
            f"totalSupply unavailable from /mint/details — using fallback {total_supply:,.0f}",
            {"mint": mint[:8], "fallback_total_supply": total_supply},
        )
    
    # Analyze whale behavior using historical trading data
    if len(whale_analysis_df) > 0:
        whale_metrics = components["whale_engine"].analyze_token(
            user_holdings_df=whale_analysis_df,
            token_metadata=metadata,
            total_supply=total_supply
        )
    else:
        logger.warning(f"No whale data for {mint[:8]}... — using empty defaults")
        whale_metrics = components["whale_engine"].create_empty_metrics()

    # Prefer Birdeye /holder/v1/distribution for top10_hold_percent (matches Birdeye portal)
    birdeye_fetcher = components.get("birdeye_fetcher")
    if birdeye_fetcher and birdeye_fetcher.is_configured():
        try:
            birdeye_top10 = await asyncio.to_thread(
                birdeye_fetcher.fetch_holder_distribution, mint, 10
            )
            if birdeye_top10 is not None and 0 <= birdeye_top10 <= 100:
                whale_metrics = dataclasses.replace(whale_metrics, top10_hold_percent=birdeye_top10)
                logger.info(f"Top10 hold % from Birdeye for {mint[:8]}...: {birdeye_top10:.2f}%")
        except Exception as e:
            logger.warning(f"Birdeye holder distribution failed for {mint[:8]}...: {e}")
    
    # Calculate holder metrics using current holders (filtered, no pools/contracts)
    holder_stats = components["holder_metrics"].calculate_all_metrics(holder_metrics_df, total_supply)
    
    # Extract candle features
    candle_data = _extract_candle_features(candles)
    
    # Fetch liquidity from Birdeye for meme tokens
    meme_market_data: Dict[str, Any] = {}
    birdeye_fetcher = components.get("birdeye_fetcher")
    if birdeye_fetcher:
        try:
            price_liq = await asyncio.to_thread(
                birdeye_fetcher.fetch_price_with_liquidity, mint
            )
            if price_liq and price_liq.get('liquidity') is not None:
                meme_market_data['liquidity_usd'] = float(price_liq['liquidity'])
                logger.info(f"Birdeye liquidity for {mint[:8]}...: ${meme_market_data['liquidity_usd']:,.2f}")
            if price_liq and "priceChange24h" in price_liq:
                meme_market_data["price_change_24h_pct_birdeye"] = float(price_liq.get("priceChange24h", 0) or 0)
        except Exception as e:
            logger.warning(f"Failed to fetch Birdeye liquidity for {mint[:8]}...: {e}")
    
    return {
        'candles': candles,
        'whale_metrics': whale_metrics,
        'holder_stats': holder_stats,
        'candle_data': candle_data,
        'market_data': meme_market_data,
        'whale_data_source': 'internal_api',
        'candle_data_source': candle_source
    }


def _extract_candle_features(candles_df) -> Dict:
    """Extract candle-based features for ML model"""
    import numpy as np
    
    if len(candles_df) == 0:
        return {}
    
    def get_col(df, *names):
        for name in names:
            if name in df.columns:
                return df[name].values
        return None
    
    latest = candles_df.iloc[-1]
    features = {}
    
    features['BuyCount'] = float(latest.get('BuyCount', latest.get('buy_count', 0)))
    features['SellCount'] = float(latest.get('SellCount', latest.get('sell_count', 0)))
    features['TotalSupply'] = float(latest.get('TotalSupply', latest.get('total_supply', 1_000_000_000)))
    
    if len(candles_df) >= 2:
        close_prices = get_col(candles_df, 'Close', 'close')
        if close_prices is not None:
            close_prices = np.array(close_prices, dtype=float)
            returns = (close_prices[-1] - close_prices[-2]) / (close_prices[-2] + 1e-10)
            features['returns'] = float(returns)
        else:
            features['returns'] = 0.0
    else:
        features['returns'] = 0.0
    
    volume_col = get_col(candles_df, 'Volume', 'volume')
    if volume_col is not None:
        volumes = np.array(volume_col, dtype=float)
        features['volume_ma'] = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    else:
        features['volume_ma'] = 0.0
    
    return features


def _check_candle_freshness(
    candles_df: pd.DataFrame,
    is_perps: bool = False,
    stale_threshold_hours: int = 24
) -> Dict[str, Any]:
    """
    Check how fresh the candle data is by examining the most recent timestamp.
    
    Args:
        candles_df: DataFrame with candle data
        is_perps: Whether this is perps data
        stale_threshold_hours: Hours after which data is considered stale (default 24).
                               Pass meme_config.CANDLE_STALE_THRESHOLD_HOURS or
                               perps_config.CANDLE_STALE_THRESHOLD_HOURS.
    
    Returns:
        Dictionary with freshness metadata:
        - latest_candle_age_hours: How old the most recent candle is
        - is_stale: True if data is older than stale_threshold_hours
        - latest_candle_timestamp: ISO timestamp of the most recent candle
        - warning: Human-readable warning if data is stale, None otherwise
    """
    result = {
        "latest_candle_age_hours": None,
        "is_stale": None,
        "latest_candle_timestamp": None,
        "warning": None
    }
    
    if candles_df is None or len(candles_df) == 0:
        result["warning"] = "No candle data available"
        result["is_stale"] = True
        return result
    
    # Detect timestamp column
    ts_col = None
    for col in ['timestamp', 'Timestamp']:
        if col in candles_df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        result["warning"] = "No timestamp column - cannot verify freshness"
        return result
    
    try:
        ts_series = candles_df[ts_col]
        if ts_series.dtype in ['int64', 'float64', 'int32', 'float32']:
            max_ts = float(ts_series.max())
            if max_ts > 1e12:
                parsed = pd.to_datetime(ts_series, unit='ms', utc=True)
            else:
                parsed = pd.to_datetime(ts_series, unit='s', utc=True)
        else:
            parsed = pd.to_datetime(ts_series, utc=True)
        
        latest = parsed.max()
        now = pd.Timestamp.now(tz='UTC')
        age_hours = (now - latest).total_seconds() / 3600
        
        result["latest_candle_age_hours"] = round(age_hours, 2)
        result["latest_candle_timestamp"] = latest.isoformat()
        result["is_stale"] = age_hours > stale_threshold_hours
        
        if age_hours > stale_threshold_hours:
            result["warning"] = (
                f"CRITICAL: Candle data is {age_hours:.1f} hours old "
                f"(threshold: {stale_threshold_hours}h)! "
                f"Analysis is based on outdated prices. "
                f"This may be caused by API limit truncating old data."
            )
        elif age_hours > 2:
            result["warning"] = (
                f"Candle data is {age_hours:.1f} hours old. "
                f"Analysis may not reflect the very latest price action, "
                f"but within acceptable range ({stale_threshold_hours}h threshold)."
            )
    except Exception as e:
        result["warning"] = f"Could not verify data freshness: {e}"
    
    return result


def _filter_recent_candles(
    candles_df: pd.DataFrame,
    hours: int = 24,
    is_perps: bool = False,
    stale_threshold_hours: int = 24
) -> pd.DataFrame:
    """
    Filter candles to the last N hours for timing-focused analysis.
    Works for BOTH meme and perps tokens.
    
    **Returned rows are always sorted chronologically (oldest first).**
    Downstream consumers (price change, volume trend) rely on this ordering.
    
    Candle resolution handling:
    - Perps (1h candles): 24 hours = 24 candles
    - Meme (5m candles): 24 hours = 288 candles (24 * 12)
    
    Timestamp column handling:
    - Perps: 'timestamp' (lowercase)
    - Meme: 'Timestamp' (uppercase)
    - Falls back to row-count if no timestamp column found
    
    Args:
        candles_df: Full candle DataFrame (meme or perps)
        hours: Number of hours to include (default: 24)
        is_perps: Whether this is perps data (affects candle count fallback)
        stale_threshold_hours: Hours after which data is considered stale (from config)
        
    Returns:
        Filtered DataFrame with only recent candles, sorted oldest-first
    """
    import numpy as np
    
    if candles_df is None or len(candles_df) == 0:
        return candles_df
    
    # Detect timestamp column
    ts_col = None
    for col in ['timestamp', 'Timestamp']:
        if col in candles_df.columns:
            ts_col = col
            break
    
    n_candles = hours if is_perps else hours * 12  # Fallback candle count (1h vs 5m)
    
    if ts_col is None:
        logger.warning("No timestamp column found in candle data - using row-count fallback")
        return candles_df.tail(n_candles).reset_index(drop=True)
    
    try:
        df = candles_df.copy()
        
        # Robust timestamp parsing: handle integers (epoch sec/ms) and strings
        ts_series = df[ts_col]
        if ts_series.dtype in ['int64', 'float64', 'int32', 'float32']:
            max_ts = float(ts_series.max())
            if max_ts > 1e12:
                df['_ts_parsed'] = pd.to_datetime(ts_series, unit='ms', utc=True)
            else:
                df['_ts_parsed'] = pd.to_datetime(ts_series, unit='s', utc=True)
        else:
            df['_ts_parsed'] = pd.to_datetime(ts_series, utc=True)
        
        # Check for NaT (failed parsing)
        nat_count = df['_ts_parsed'].isna().sum()
        if nat_count > len(df) * 0.5:
            logger.error(f"Timestamp parsing failed for {nat_count}/{len(df)} candles - using row-count fallback")
            return candles_df.tail(n_candles).reset_index(drop=True)
        
        # Sort chronologically (oldest first) — required for correct price/volume trend
        df = df.sort_values('_ts_parsed', ascending=True)
        
        # Filter to recent window
        now = pd.Timestamp.now(tz='UTC')
        cutoff = now - pd.Timedelta(hours=hours)
        recent = df[df['_ts_parsed'] >= cutoff].drop(columns=['_ts_parsed']).reset_index(drop=True)
        
        # Freshness check: how old is the most recent candle?
        latest_ts = df['_ts_parsed'].max()
        data_age_hours = (now - latest_ts).total_seconds() / 3600
        
        if data_age_hours > stale_threshold_hours:
            logger.error(
                f"STALE CANDLE DATA DETECTED: Latest candle is {data_age_hours:.1f} hours old "
                f"(threshold: {stale_threshold_hours}h, latest: {latest_ts}, now: {now}). "
                f"This likely means the API limit was too low to cover the requested date range. "
                f"Analysis will be based on outdated prices!"
            )
        
        if len(recent) < 10:
            if data_age_hours > stale_threshold_hours:
                logger.error(
                    f"Only {len(recent)} candles found in last {hours}h (need >= 10). "
                    f"Falling back to last {n_candles} rows, but DATA IS STALE ({data_age_hours:.1f}h old). "
                    f"Results will NOT reflect current market conditions!"
                )
            else:
                logger.warning(f"Only {len(recent)} candles in last {hours}h, using last {n_candles} rows")
            # Fallback: sort then take tail so we get the most recent N in chronological order
            sorted_df = df.sort_values('_ts_parsed', ascending=True).drop(columns=['_ts_parsed'])
            return sorted_df.tail(n_candles).reset_index(drop=True)
        
        if data_age_hours <= stale_threshold_hours:
            logger.info(f"Filtered to {len(recent)} candles from last {hours}h (data is {data_age_hours:.1f}h old - FRESH)")
        
        return recent
    except Exception as e:
        logger.warning(f"Failed to filter recent candles by timestamp: {e}")
        return candles_df.tail(n_candles).reset_index(drop=True)


def _compute_recent_24h_analysis(
    candles_df: pd.DataFrame,
    is_perps: bool = False,
    stale_threshold_hours: int = 24
) -> Dict[str, Any]:
    """
    Compute 24-hour focused analysis metrics from candle data.
    Works for BOTH meme and perps tokens.
    
    This provides a snapshot of what happened in the last 24 hours:
    - Price change (24h)
    - Volume profile (24h)
    - Volatility (24h)
    - Trend direction (24h)
    - Buy/sell pressure (24h)
    
    The full historical data is still used for indicator calculation,
    but these metrics show the most relevant recent activity.
    
    Column name handling:
    - Meme: Close, Volume, BuyVolume, SellVolume (uppercase)
    - Perps: close, volume, buy_volume, sell_volume (lowercase)
    
    Args:
        candles_df: Full candle DataFrame (meme: 90 days of 5m candles, perps: 500 1h candles)
        is_perps: Whether this is perps data (affects candle filtering)
        stale_threshold_hours: Hours after which data is considered stale (from config)
        
    Returns:
        Dictionary with 24h-focused metrics for the signal effectiveness window
    """
    import numpy as np
    
    if candles_df is None or len(candles_df) == 0:
        return {"error": "No candle data available"}
    
    # Get recent candles (last 24h) — already sorted oldest-first
    recent = _filter_recent_candles(candles_df, hours=24, is_perps=is_perps, stale_threshold_hours=stale_threshold_hours)
    
    if len(recent) == 0:
        return {"error": "No recent candle data available"}
    
    # Defensive chronological sort: ensures correct price/volume trend even if
    # _filter_recent_candles is changed upstream or called with unsorted data.
    for _ts_name in ('timestamp', 'Timestamp'):
        if _ts_name in recent.columns:
            try:
                _tmp = pd.to_datetime(recent[_ts_name], utc=True, errors='coerce')
                if _tmp.notna().sum() > len(recent) * 0.5:
                    recent = recent.assign(_sort_ts=_tmp).sort_values('_sort_ts', ascending=True).drop(columns=['_sort_ts']).reset_index(drop=True)
            except Exception:
                pass
            break
    
    def get_col(df, *names):
        for name in names:
            if name in df.columns:
                return np.array(df[name].values, dtype=float)
        return None
    
    # Compute data freshness: how old is the most recent candle?
    data_freshness = _check_candle_freshness(candles_df, is_perps, stale_threshold_hours=stale_threshold_hours)
    
    result = {
        "window_hours": 24,
        "candles_analyzed": len(recent),
        "total_candles_available": len(candles_df),
        "data_freshness": data_freshness
    }
    
    # Actual time span of filtered candles (so we can say "last Xh" when X < 24)
    if len(recent) >= 2:
        ts_col = None
        for col in ['timestamp', 'Timestamp']:
            if col in recent.columns:
                ts_col = col
                break
        if ts_col is not None:
            try:
                ts_series = recent[ts_col]
                first_ts = ts_series.iloc[0]
                last_ts = ts_series.iloc[-1]
                if pd.notna(first_ts) and pd.notna(last_ts):
                    if isinstance(first_ts, (int, float)) and isinstance(last_ts, (int, float)):
                        if first_ts > 1e12:
                            t_first = pd.to_datetime(first_ts, unit='ms', utc=True)
                            t_last = pd.to_datetime(last_ts, unit='ms', utc=True)
                        else:
                            t_first = pd.to_datetime(first_ts, unit='s', utc=True)
                            t_last = pd.to_datetime(last_ts, unit='s', utc=True)
                    else:
                        t_first = pd.to_datetime(first_ts, utc=True)
                        t_last = pd.to_datetime(last_ts, utc=True)
                    if hasattr(t_first, 'tzinfo') and t_first.tzinfo is None:
                        t_first = t_first.tz_localize('UTC')
                    if hasattr(t_last, 'tzinfo') and t_last.tzinfo is None:
                        t_last = t_last.tz_localize('UTC')
                    delta_sec = (t_last - t_first).total_seconds()
                    if delta_sec >= 0:
                        result["window_actual_hours"] = round(delta_sec / 3600, 1)
            except Exception:
                pass
    
    # Price change over 24h
    close_prices = get_col(recent, 'Close', 'close')
    if close_prices is not None and len(close_prices) >= 2:
        price_start = close_prices[0]
        price_end = close_prices[-1]
        price_high = float(np.max(close_prices))
        price_low = float(np.min(close_prices))
        
        result["price_change_24h_pct"] = round(((price_end - price_start) / (price_start + 1e-10)) * 100, 2)
        result["price_current"] = round(float(price_end), 8)
        result["price_24h_high"] = round(price_high, 8)
        result["price_24h_low"] = round(price_low, 8)
        result["price_24h_range_pct"] = round(((price_high - price_low) / (price_low + 1e-10)) * 100, 2)
        
        # Trend direction
        if result["price_change_24h_pct"] > 5:
            result["trend_24h"] = "strongly_bullish"
        elif result["price_change_24h_pct"] > 1:
            result["trend_24h"] = "bullish"
        elif result["price_change_24h_pct"] < -5:
            result["trend_24h"] = "strongly_bearish"
        elif result["price_change_24h_pct"] < -1:
            result["trend_24h"] = "bearish"
        else:
            result["trend_24h"] = "sideways"
    
    # Volume analysis (24h)
    volumes = get_col(recent, 'Volume', 'volume')
    if volumes is not None and len(volumes) > 0:
        result["volume_24h_total"] = round(float(np.sum(volumes)), 2)
        result["volume_24h_avg"] = round(float(np.mean(volumes)), 2)
        
        # Volume trend: compare first half vs second half of 24h
        half = len(volumes) // 2
        if half > 0:
            first_half_vol = float(np.mean(volumes[:half]))
            second_half_vol = float(np.mean(volumes[half:]))
            if first_half_vol > 0:
                vol_change = ((second_half_vol - first_half_vol) / first_half_vol) * 100
                vol_change = max(-95.0, min(95.0, vol_change))
                result["volume_trend_24h"] = "increasing" if vol_change > 10 else "decreasing" if vol_change < -10 else "stable"
                result["volume_trend_change_pct"] = round(vol_change, 2)
    
    # Buy/sell pressure (24h)
    # Note: Birdeye candles do NOT include BuyVolume/SellVolume columns,
    # so this section correctly produces no pressure data for Birdeye-sourced candles.
    buy_vol = get_col(recent, 'BuyVolume', 'buy_volume')
    sell_vol = get_col(recent, 'SellVolume', 'sell_volume')
    if buy_vol is not None and sell_vol is not None:
        total_buy = float(np.sum(buy_vol))
        total_sell = float(np.sum(sell_vol))
        total = total_buy + total_sell
        if total > 0:
            result["buy_pressure_24h"] = round(total_buy / total, 4)
            result["sell_pressure_24h"] = round(total_sell / total, 4)
            result["net_flow_24h"] = round(total_buy - total_sell, 2)
            result["buy_sell_ratio_24h"] = round(total_buy / (total_sell + 1e-10), 4)
            result["buy_sell_data_source"] = "direct"
    else:
        result["buy_sell_data_source"] = "unavailable"
    
    # Volatility (24h)
    if close_prices is not None and len(close_prices) >= 2:
        returns = np.diff(close_prices) / (close_prices[:-1] + 1e-10)
        result["volatility_24h"] = round(float(np.std(returns)), 6)
    
    return result


def _compute_risk_management(
    signal: str,
    tech_signals,
    recent_analysis: Dict,
    is_perps: bool
) -> Optional[Dict]:
    """
    Compute stop-loss, take-profit, and risk/reward ratio using
    Bollinger Bands and the 24h price range.
    """
    if signal not in ("BUY", "SELL"):
        return None

    current_price = recent_analysis.get("price_current", 0)
    if current_price <= 0:
        return None

    volume_24h = recent_analysis.get("volume_24h_total", 0)
    volatility_24h = recent_analysis.get("volatility_24h", 0)
    if volume_24h <= 0 and volatility_24h <= 0:
        min_stop_pct = safety_config.MIN_STOP_DISTANCE_PCT
        if signal == "BUY":
            stop_loss = current_price * (1.0 - min_stop_pct / 100.0)
            take_profit = current_price * 1.05
        else:
            stop_loss = current_price * (1.0 + min_stop_pct / 100.0)
            take_profit = current_price * 0.95
        reward_distance_pct = abs(take_profit - current_price) / current_price * 100
        return {
            "suggested_stop_loss": round(stop_loss, 8),
            "suggested_take_profit": round(take_profit, 8),
            "stop_distance_pct": round(min_stop_pct, 2),
            "reward_distance_pct": round(reward_distance_pct, 2),
            "risk_reward_ratio": round(reward_distance_pct / min_stop_pct, 2) if min_stop_pct > 0 else 0,
            "method": "min_stop_pct_fallback"
        }

    bb_lower = getattr(tech_signals, 'bb_lower', 0) or 0
    bb_upper = getattr(tech_signals, 'bb_upper', 0) or 0
    low_24h = recent_analysis.get("price_24h_low", 0) or 0
    high_24h = recent_analysis.get("price_24h_high", 0) or 0

    if signal == "BUY":
        stop_loss = max(bb_lower, low_24h) if bb_lower > 0 else low_24h
        take_profit = bb_upper if bb_upper > current_price else current_price * 1.05
    else:
        stop_loss = min(bb_upper, high_24h) if bb_upper > 0 else high_24h
        take_profit = bb_lower if 0 < bb_lower < current_price else current_price * 0.95

    if stop_loss <= 0:
        return None

    stop_distance_pct = abs(current_price - stop_loss) / current_price * 100

    min_stop_pct = safety_config.MIN_STOP_DISTANCE_PCT
    if stop_distance_pct < min_stop_pct:
        if signal == "BUY":
            stop_loss = current_price * (1.0 - min_stop_pct / 100.0)
        else:
            stop_loss = current_price * (1.0 + min_stop_pct / 100.0)
        stop_distance_pct = min_stop_pct

    reward_distance_pct = abs(take_profit - current_price) / current_price * 100
    risk_reward = reward_distance_pct / stop_distance_pct if stop_distance_pct > 0 else 0

    if risk_reward < safety_config.MIN_RISK_REWARD_RATIO:
        return None

    return {
        "suggested_stop_loss": round(stop_loss, 8),
        "suggested_take_profit": round(take_profit, 8),
        "stop_distance_pct": round(stop_distance_pct, 2),
        "reward_distance_pct": round(reward_distance_pct, 2),
        "risk_reward_ratio": round(risk_reward, 2),
        "method": "bollinger_bands_24h_range"
    }
