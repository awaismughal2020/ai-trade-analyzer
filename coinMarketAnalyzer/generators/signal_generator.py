"""
Signal Generator
Combines all analysis components to generate final trading signals
Enhanced with Layer 2 (User Profile) and Multi-Layer Aggregation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any, List
import logging
import traceback
from dataclasses import dataclass, field
import joblib
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    signal_config, MODELS_DIR, whale_config, wallet_classification_config,
    layer_weight_config, perps_model_config, TokenType, perps_config, meme_config,
    api_config, safety_config
)
from engines.whale_engine import WhaleMetrics
from engines.technical_engine import TechnicalSignals
from engines.holder_metrics import HolderStats

# Always import Layer Aggregator (required for multi-layer architecture per flowchart)
from .layer_aggregator import (
    LayerAggregator, LayerSignal, AggregatedSignal, 
    create_layer_signal, aggregation_to_dict
)

# Import Token Type Router for token type based routing
from engines.token_type_router import (
    TokenTypeRouter, get_router, is_perps_token, is_meme_token
)

# Layer 2 (User Profile) imports (optional - graceful degradation if not available)
try:
    from engines.user_profiler import UserProfiler, UserProfile
    from engines.risk_assessor import RiskAssessor, RiskAssessment, RiskRating
    LAYER2_AVAILABLE = True
except ImportError:
    LAYER2_AVAILABLE = False
    UserProfiler = None
    RiskAssessor = None

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Final trading signal with all metrics"""
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1
    
    # Whale metrics
    whale_buy_volume: float
    whale_sell_volume: float
    whale_net_volume: float
    
    # Distribution metrics
    gini_coefficient: float
    top10_hold_percent: float
    dev_hold_percent: float
    sniper_hold_percent: float
    
    # Phase and state
    phase: str
    whale_state: str
    
    # Summary
    summary: str
    
    # Additional context
    technical_signal: str
    risk_level: str
    timestamp: str
    
    # === Fields with default values must come after fields without defaults ===
    
    # Whale data quality flags
    is_whale_data_stale: bool = False  # True if no recent whale activity (using lifetime data)
    whale_data_source: str = "recent"  # "recent" (active trading), "holding" (holding but not trading), "lifetime" (no holdings), or "no_whales"
    
    # Layer 2 - User Risk Assessment (optional)
    user_risk_assessment: Optional[Dict[str, Any]] = None
    
    # Multi-Layer Breakdown (optional)
    layer_breakdown: Optional[Dict[str, Any]] = None
    
    # Aggregation metadata
    layers_used: int = 0
    agreement_level: float = 0.0
    
    # Safety override information
    safety_overrides_applied: Optional[List[Dict[str, Any]]] = None
    confidence_before_safety: Optional[float] = None

    # Global market context snapshot
    market_context: Optional[Dict[str, Any]] = None

    # Feature 2: Entry Timing (optional)
    timing: Optional[Dict[str, Any]] = None
    
    # Signal effectiveness window (±24h by default)
    # Signal is based on the last N hours of data and predicted to be effective for the next N hours
    signal_effectiveness_hours: int = 24
    signal_valid_from: Optional[str] = None  # ISO timestamp: signal analysis start (now - 24h)
    signal_valid_until: Optional[str] = None  # ISO timestamp: signal prediction end (now + 24h)
    
    # 24h-focused analysis metrics (recent window analysis)
    recent_analysis: Optional[Dict[str, Any]] = None


class SignalGenerator:
    """
    Generates trading signals by combining all analysis components
    Enhanced with Layer 2 (User Profile) support
    """
    
    def __init__(self, model_path: Optional[str] = None, data_fetcher=None, birdeye_fetcher=None):
        """
        Initialize Signal Generator with dual ML model support
        
        Args:
            model_path: Path to trained XGBoost model for meme tokens (optional)
            data_fetcher: DataFetcher instance for Layer 2 (optional)
            birdeye_fetcher: BirdeyeFetcher instance for historical price data (optional)
        """
        # Meme model components
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Perps model components
        self.perps_model = None
        self.perps_scaler = None
        self.perps_features = None
        self.perps_ticker_encoder = None
        
        # Token type router
        self.token_router = get_router()
        
        # Layer 2 components (optional)
        self.user_profiler = None
        self.risk_assessor = None
        self.layer_aggregator = None
        self.data_fetcher = data_fetcher
        self.birdeye_fetcher = birdeye_fetcher
        
        # Always initialize Layer Aggregator (required for multi-layer weighted combination per architecture)
        try:
            self.layer_aggregator = LayerAggregator()
            logger.info("Layer Aggregator initialized (multi-layer weighted combination enabled)")
        except Exception as e:
            logger.error(f"Layer Aggregator initialization failed: {e}")
            raise  # This is critical, so raise the error
        
        # Initialize Layer 2 (User Profile) if available
        if LAYER2_AVAILABLE:
            try:
                self.user_profiler = UserProfiler(data_fetcher=data_fetcher, birdeye_fetcher=birdeye_fetcher)
                self.risk_assessor = RiskAssessor(self.user_profiler)
                logger.info("Layer 2 (User Profile) initialized with Birdeye integration")
            except Exception as e:
                logger.warning(f"Layer 2 initialization failed: {e}")
        else:
            logger.info("Layer 2 (User Profile) not available - will use 4 layers instead of 5")
        
        # Load meme model (default)
        if model_path:
            self._load_model(model_path)
        else:
            default_path = os.path.join(MODELS_DIR, "xgboost_hybrid_model.pkl")
            if os.path.exists(default_path):
                self._load_model(default_path)
            else:
                logger.warning("No meme model found, using rule-based signal generation")
        
        # Load perps model if available
        self._load_perps_model()
        
        logger.info("Signal Generator initialized with dual model support")
    
    def set_data_fetcher(self, data_fetcher):
        """Set or update the data fetcher for Layer 2"""
        self.data_fetcher = data_fetcher
        if self.user_profiler:
            self.user_profiler.set_data_fetcher(data_fetcher)
    
    def _load_model(self, model_path: str):
        """Load trained XGBoost model for meme tokens"""
        try:
            self.model = joblib.load(model_path)
            
            scaler_path = model_path.replace("xgboost_hybrid_model.pkl", "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            features_path = model_path.replace("xgboost_hybrid_model.pkl", "feature_columns.pkl")
            if os.path.exists(features_path):
                self.feature_columns = joblib.load(features_path)
            
            logger.info(f"Meme model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load meme model: {e}")
            self.model = None
    
    def _load_perps_model(self):
        """Load trained model for perps tokens"""
        try:
            model_path = perps_model_config.get_model_path()
            
            if not os.path.exists(model_path):
                logger.info(f"Perps model not found at {model_path} - will use rule-based for perps")
                return
            
            self.perps_model = joblib.load(model_path)
            
            scaler_path = perps_model_config.get_scaler_path()
            if os.path.exists(scaler_path):
                self.perps_scaler = joblib.load(scaler_path)
            
            features_path = perps_model_config.get_features_path()
            if os.path.exists(features_path):
                self.perps_features = joblib.load(features_path)
            
            ticker_encoder_path = perps_model_config.get_ticker_encoder_path()
            if os.path.exists(ticker_encoder_path):
                self.perps_ticker_encoder = joblib.load(ticker_encoder_path)
            
            logger.info(f"Perps model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load perps model: {e}")
            self.perps_model = None
    
    def has_perps_model(self) -> bool:
        """Check if perps model is loaded"""
        return self.perps_model is not None
    
    def has_meme_model(self) -> bool:
        """Check if meme model is loaded"""
        return self.model is not None
    
    def generate_signal(
        self,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats,
        summary: str,
        candle_data: Optional[Dict] = None,
        user_address: Optional[str] = None,
        token_type: str = "meme",
        market_data: Optional[Dict] = None,
        perps_user_profile: Optional[Dict] = None,
        user_from_date: Optional[str] = None,
        user_to_date: Optional[str] = None,
        market_context = None,
        recent_analysis: Optional[Dict] = None,
        target_mint: Optional[str] = None
    ) -> TradingSignal:
        """
        Generate final trading signal from all components
        
        Args:
            whale_metrics: Whale analysis metrics
            technical_signals: Technical indicator signals
            holder_stats: Holder distribution statistics
            summary: Generated summary text
            candle_data: Raw candle data (optional)
            user_address: User wallet for personalized signals (optional, Layer 2)
            token_type: Token type for ML model routing ('meme' or 'perps')
            market_data: Additional market data for perps (funding rate, OI, etc.)
            perps_user_profile: Pre-fetched perps user profile for Layer 5 (optional, perps only)
            user_from_date: Optional start date for user profile trades (ISO format, defaults to 90 days)
            user_to_date: Optional end date for user profile trades (ISO format, defaults to now)
            market_context: MarketContext object with global market data (optional)
            target_mint: Token address for same-token history lookup (optional)
            
        Returns:
            TradingSignal object with final recommendation
        """
        # Layer 2 variables
        user_risk_assessment = None
        layer_breakdown = None
        layers_used = 0
        agreement_level = 0.0
        
        # Use layer aggregation if available (ALWAYS use multi-layer weighted combination per architecture)
        if self.layer_aggregator:
            return self._generate_layered_signal(
                whale_metrics=whale_metrics,
                technical_signals=technical_signals,
                holder_stats=holder_stats,
                summary=summary,
                candle_data=candle_data,
                user_address=user_address,
                token_type=token_type,
                market_data=market_data,
                perps_user_profile=perps_user_profile,
                user_from_date=user_from_date,
                user_to_date=user_to_date,
                market_context=market_context,
                recent_analysis=recent_analysis,
                target_mint=target_mint,
            )
        
        # Fallback: Standard signal generation (if aggregator not available)
        # Try ML-based prediction first
        if self.model is not None:
            signal, confidence = self._ml_prediction(
                whale_metrics=whale_metrics,
                technical_signals=technical_signals,
                holder_stats=holder_stats,
                candle_data=candle_data
            )
        else:
            # Fall back to rule-based prediction
            signal, confidence = self._rule_based_prediction(
                whale_metrics=whale_metrics,
                technical_signals=technical_signals,
                holder_stats=holder_stats
            )
        
        # Assess risk level
        risk_level = self._assess_risk(
            whale_metrics=whale_metrics,
            holder_stats=holder_stats
        )
        
        # === SAFETY OVERRIDES ===
        # These overrides protect against dangerous trades that the ML model may miss
        signal, confidence, safety_overrides, confidence_before_safety = self._apply_safety_overrides(
            signal, confidence, whale_metrics, technical_signals, holder_stats, risk_level,
            token_type=token_type,
            recent_analysis=recent_analysis,
            market_data=market_data
        )
        
        # Boost confidence for SELL when metrics align
        if signal == "SELL":
            sell_confirmations = 0
            if whale_metrics.gini_coefficient > whale_config.GINI_HIGH_RISK:
                sell_confirmations += 1
            if whale_metrics.top10_hold_percent > wallet_classification_config.TOP10_CONCENTRATION_HIGH_RISK * 100:
                sell_confirmations += 1
            if technical_signals.overall_signal == "bearish":
                sell_confirmations += 1
            if whale_metrics.whale_state == "Distribution":
                sell_confirmations += 1
            
            # Boost confidence based on confirmations
            if sell_confirmations >= 3:
                confidence = min(0.95, confidence + 0.1)
            elif sell_confirmations >= 2:
                confidence = min(0.90, confidence + 0.05)
        
        # === POSITIVE SIGNAL BOOSTING ===
        # When fundamentals are strong but model says SELL, reconsider
        
        buy_confirmations = 0
        if whale_metrics.top10_hold_percent < wallet_classification_config.SAFETY_GOOD_CONCENTRATION_PCT:  # Good distribution
            buy_confirmations += 1
        if whale_metrics.gini_coefficient < whale_config.GINI_HIGH_RISK:  # Reasonable equality
            buy_confirmations += 1
        if technical_signals.rsi_signal == "oversold":  # Potential bounce
            buy_confirmations += 1
        if technical_signals.overall_signal == "bullish":
            buy_confirmations += 1
        if whale_metrics.whale_state == "Accumulation":
            buy_confirmations += 2  # Strong signal
        if holder_stats.active_holders > 500:  # Good holder base
            buy_confirmations += 1
        
        # Override SELL to HOLD when fundamentals are strong
        if signal == "SELL" and buy_confirmations >= 4:
            signal = "HOLD"
            confidence = 0.55
            override_reason = f"Strong fundamentals override ({buy_confirmations} positive signals)"
            logger.info(f"Positive override: SELL -> HOLD - {override_reason}")
        
        # Consider upgrading HOLD to BUY when fundamentals are very strong
        if signal == "HOLD" and buy_confirmations >= 5 and risk_level == "Low":
            signal = "BUY"
            confidence = 0.65
            override_reason = f"Very strong fundamentals ({buy_confirmations} positive signals)"
            logger.info(f"Positive upgrade: HOLD -> BUY - {override_reason}")
        
        # Calculate signal effectiveness window (±24h for both meme and perps)
        from datetime import timedelta
        effectiveness_hours = perps_config.SIGNAL_EFFECTIVENESS_HOURS if is_perps_token(token_type) else meme_config.SIGNAL_EFFECTIVENESS_HOURS
        now_utc = datetime.utcnow()
        signal_valid_from = (now_utc - timedelta(hours=effectiveness_hours)).isoformat()
        signal_valid_until = (now_utc + timedelta(hours=effectiveness_hours)).isoformat()
        
        return TradingSignal(
            signal=signal,
            confidence=confidence,
            whale_buy_volume=whale_metrics.whale_buy_volume,
            whale_sell_volume=whale_metrics.whale_sell_volume,
            whale_net_volume=whale_metrics.whale_net_volume,
            is_whale_data_stale=getattr(whale_metrics, 'is_whale_data_stale', False),
            whale_data_source=getattr(whale_metrics, 'whale_data_source', 'recent'),
            gini_coefficient=whale_metrics.gini_coefficient,
            top10_hold_percent=whale_metrics.top10_hold_percent,
            dev_hold_percent=whale_metrics.dev_hold_percent,
            sniper_hold_percent=whale_metrics.sniper_hold_percent,
            phase=whale_metrics.phase,
            whale_state=whale_metrics.whale_state,
            summary=summary,
            technical_signal=technical_signals.overall_signal,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            user_risk_assessment=user_risk_assessment,
            layer_breakdown=layer_breakdown,
            layers_used=layers_used,
            agreement_level=agreement_level,
            safety_overrides_applied=safety_overrides if safety_overrides else None,
            confidence_before_safety=confidence_before_safety,
            signal_effectiveness_hours=effectiveness_hours,
            signal_valid_from=signal_valid_from,
            signal_valid_until=signal_valid_until
        )
    
    def _generate_layered_signal(
        self,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats,
        summary: str,
        candle_data: Optional[Dict] = None,
        user_address: Optional[str] = None,
        token_type: str = "meme",
        market_data: Optional[Dict] = None,
        perps_user_profile: Optional[Dict] = None,
        user_from_date: Optional[str] = None,
        user_to_date: Optional[str] = None,
        market_context = None,
        recent_analysis: Optional[Dict] = None,
        target_mint: Optional[str] = None
    ) -> TradingSignal:
        """
        Generate signal using multi-layer aggregation with user profile
        
        This method uses the Layer Aggregator to combine signals from:
        - ML Model (30%) - Uses meme or perps model based on token_type
        - Whale Engine (25%)
        - Technical Indicators (15%)
        - Holder Metrics (15%)
        - User Profile (15%) - Layer 2
        
        Args:
            token_type: 'meme' or 'perps' - routes to appropriate ML model
            market_data: Additional market data for perps (funding rate, OI, etc.)
            perps_user_profile: Pre-fetched perps user profile for Layer 5 (optional, perps only)
            user_from_date: Optional start date for user profile trades (ISO format, defaults to 90 days)
            user_to_date: Optional end date for user profile trades (ISO format, defaults to now)
            target_mint: Token address for same-token history lookup (optional)
        """
        is_perps = is_perps_token(token_type)
        logger.info(f"Generating layered signal for user {user_address[:8] if user_address else 'N/A'} "
                   f"(token_type={token_type}, is_perps={is_perps})...")
        
        # Build layer signals
        layer_signals = {}
        
        # 1. ML Model Layer - Route based on token_type
        ml_signal, ml_confidence = self._ml_prediction_routed(
            whale_metrics=whale_metrics,
            technical_signals=technical_signals,
            holder_stats=holder_stats,
            candle_data=candle_data,
            token_type=token_type,
            market_data=market_data,
            market_context=market_context
        )
        
        model_type = "perps" if is_perps else "meme"
        layer_signals['ml_model'] = create_layer_signal(
            'ml_model', ml_signal, ml_confidence, True,
            {'probability': ml_confidence, 'model_type': model_type}
        )
        
        # 2. Whale Engine Layer
        # Detect if whale data is from empty defaults (degraded mode)
        whale_data_source = getattr(whale_metrics, 'whale_data_source', 'recent')
        whale_data_unavailable = (
            whale_data_source in ("no_data", "unavailable") or
            (whale_metrics.whale_buy_volume == 0 and
             whale_metrics.whale_sell_volume == 0 and
             whale_metrics.whale_state == "UNKNOWN")
        )
        
        whale_signal = self._whale_to_signal(whale_metrics)
        whale_confidence = self._whale_confidence(whale_metrics)
        whale_is_valid = not whale_data_unavailable
        
        if whale_data_unavailable:
            logger.info("Whale data unavailable — marking whale_engine layer as invalid")
        
        layer_signals['whale_engine'] = create_layer_signal(
            'whale_engine', whale_signal, whale_confidence, whale_is_valid,
            {
                'state': whale_metrics.whale_state,
                'net_flow': whale_metrics.whale_net_volume,
                'top10': whale_metrics.top10_hold_percent,
                'data_available': not whale_data_unavailable
            }
        )
        
        # 3. Technical Indicators Layer
        tech_signal = self._technical_to_signal(technical_signals)
        tech_confidence = technical_signals.signal_strength
        layer_signals['technical'] = create_layer_signal(
            'technical', tech_signal, tech_confidence, True,
            {
                'overall': technical_signals.overall_signal,
                'rsi': technical_signals.rsi,
                'rsi_signal': technical_signals.rsi_signal
            }
        )
        
        # 4. Holder Metrics Layer
        # Invalid for perps (no holders) or when holder data is unavailable (degraded mode)
        holder_data_unavailable = (
            holder_stats.active_holders == 0 and
            holder_stats.total_holders == 0
        )
        holder_signal = self._holder_to_signal(holder_stats, whale_metrics)
        holder_confidence = holder_stats.holder_score / 100.0
        holder_is_valid = not is_perps and not holder_data_unavailable
        
        if holder_data_unavailable and not is_perps:
            logger.info("Holder data unavailable — marking holder_metrics layer as invalid")
        
        layer_signals['holder_metrics'] = create_layer_signal(
            'holder_metrics', holder_signal, holder_confidence, holder_is_valid,
            {
                'gini': whale_metrics.gini_coefficient,
                'holder_score': holder_stats.holder_score,
                'active_holders': holder_stats.active_holders,
                'data_available': not holder_data_unavailable
            }
        )
        
        # 5. User Profile Layer (Layer 2)
        user_risk_assessment = None
        if user_address and is_perps and perps_user_profile:
            # Perps path: Use already-fetched perps profile directly
            try:
                perps_assessment = self._assess_perps_user(
                    perps_user_profile, whale_metrics, technical_signals
                )
                
                layer_signals['user_profile'] = create_layer_signal(
                    'user_profile',
                    perps_assessment['signal'],
                    perps_assessment['confidence'],
                    True,
                    {
                        'rating': perps_assessment['rating'],
                        'message': perps_assessment['message'],
                        'profile_summary': perps_assessment['profile_summary']
                    }
                )
                
                # Store for response (matching meme format)
                user_risk_assessment = {
                    'rating': perps_assessment['rating'],
                    'confidence': perps_assessment['confidence'],
                    'signal': perps_assessment['signal'],
                    'message': perps_assessment['message'],
                    'risk_factors': perps_assessment['risk_factors'],
                    'pattern': None,
                    'profile_summary': perps_assessment['profile_summary'],
                    'same_token_history': None,
                    'similar_pattern_summary': None,
                }
            except Exception as e:
                logger.warning(f"Perps user profile assessment failed: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                layer_signals['user_profile'] = create_layer_signal(
                    'user_profile', 'HOLD', 0.3, False,
                    {'error': str(e)}
                )
        elif user_address and self.risk_assessor:
            # Meme path: Use UserProfiler-based risk assessment
            try:
                assessment = self.risk_assessor.assess_trade(
                    user_address=user_address,
                    target_mint=target_mint or "",
                    whale_metrics=whale_metrics,
                    technical_signals=technical_signals,
                    holder_stats=holder_stats,
                    from_date=user_from_date,
                    to_date=user_to_date
                )
                
                layer_signals['user_profile'] = create_layer_signal(
                    'user_profile',
                    assessment.signal,
                    assessment.confidence,
                    True,
                    {
                        'rating': assessment.rating.value,
                        'message': assessment.message,
                        'profile_summary': assessment.profile_summary
                    }
                )
                
                # Store for response
                user_risk_assessment = {
                    'rating': assessment.rating.value,
                    'confidence': round(assessment.confidence, 4),
                    'signal': assessment.signal,
                    'message': assessment.message,
                    'risk_factors': assessment.risk_factors,
                    'pattern': {
                        'type': assessment.matching_pattern.pattern_type.value,
                        'win_rate': round(assessment.matching_pattern.win_rate, 4),
                        'occurrences': assessment.matching_pattern.occurrences
                    } if assessment.matching_pattern else None,
                    'profile_summary': assessment.profile_summary,
                    'same_token_history': assessment.same_token_history,
                    'similar_pattern_summary': assessment.similar_pattern_summary,
                }
            except Exception as e:
                logger.warning(f"User profile assessment failed: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                layer_signals['user_profile'] = create_layer_signal(
                    'user_profile', 'HOLD', 0.3, False,
                    {'error': str(e)}
                )
        else:
            layer_signals['user_profile'] = create_layer_signal(
                'user_profile', 'HOLD', 0.0, False,
                {'error': 'User address not provided'}
            )
        
        # Aggregate all layers — exclude structurally-unavailable layers from
        # coverage penalty (holder_metrics never valid for perps)
        excluded = {'holder_metrics'} if is_perps else set()
        aggregation = self.layer_aggregator.aggregate(layer_signals, excluded_from_coverage=excluded)
        
        # Extract aggregated signal
        signal = aggregation.final_signal
        confidence = aggregation.final_confidence
        
        # Build layer breakdown for response
        layer_breakdown = {
            layer.layer_name: {
                'signal': layer.signal,
                'confidence': round(layer.confidence, 4),
                'base_weight': round(layer.base_weight, 4),
                'adjusted_weight': round(layer.adjusted_weight, 4),
                'contribution': round(layer.contribution, 4),
                'is_valid': layer.is_valid
            }
            for layer in aggregation.layer_contributions
        }
        
        # Assess risk level
        risk_level = self._assess_risk(whale_metrics, holder_stats)
        
        # Apply safety overrides (perps-aware: skips holder/gini checks for perps)
        signal, confidence, safety_overrides, confidence_before_safety = self._apply_safety_overrides(
            signal, confidence, whale_metrics, technical_signals, holder_stats, risk_level,
            token_type=token_type,
            market_context=market_context,
            recent_analysis=recent_analysis,
            market_data=market_data
        )
        
        # Degraded mode guard: when whale AND holder data are both unavailable,
        # we only have ML model + technical indicators (2 of 5 layers).
        # Force HOLD because we can't trust BUY/SELL without fundamental analysis.
        if whale_data_unavailable and holder_data_unavailable and not is_perps:
            if signal in ("BUY", "SELL"):
                original = signal
                signal = "HOLD"
                confidence = min(confidence, 0.5)
                safety_overrides.append({
                    'check': 'degraded_data_guard',
                    'condition': 'whale_data AND holder_data unavailable',
                    'value': f'layers_used={aggregation.layers_used}/5',
                    'original_signal': original,
                    'new_signal': 'HOLD',
                    'reason': f'Insufficient data for {original} signal — only {aggregation.layers_used} of 5 analysis layers available (missing whale, holder, user data)'
                })
                logger.warning(
                    f"Degraded data guard: {original} -> HOLD — "
                    f"only {aggregation.layers_used}/5 layers, confidence capped at {confidence:.2f}"
                )
        
        # Calculate signal effectiveness window (±24h for both meme and perps)
        from datetime import timedelta
        effectiveness_hours = perps_config.SIGNAL_EFFECTIVENESS_HOURS if is_perps else meme_config.SIGNAL_EFFECTIVENESS_HOURS
        now = datetime.utcnow()
        signal_valid_from = (now - timedelta(hours=effectiveness_hours)).isoformat()
        signal_valid_until = (now + timedelta(hours=effectiveness_hours)).isoformat()
        
        return TradingSignal(
            signal=signal,
            confidence=confidence,
            whale_buy_volume=whale_metrics.whale_buy_volume,
            whale_sell_volume=whale_metrics.whale_sell_volume,
            whale_net_volume=whale_metrics.whale_net_volume,
            is_whale_data_stale=getattr(whale_metrics, 'is_whale_data_stale', False),
            whale_data_source=getattr(whale_metrics, 'whale_data_source', 'recent'),
            gini_coefficient=whale_metrics.gini_coefficient,
            top10_hold_percent=whale_metrics.top10_hold_percent,
            dev_hold_percent=whale_metrics.dev_hold_percent,
            sniper_hold_percent=whale_metrics.sniper_hold_percent,
            phase=whale_metrics.phase,
            whale_state=whale_metrics.whale_state,
            summary=summary,
            technical_signal=technical_signals.overall_signal,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat(),
            user_risk_assessment=user_risk_assessment,
            layer_breakdown=layer_breakdown,
            layers_used=aggregation.layers_used,
            agreement_level=aggregation.agreement_level,
            safety_overrides_applied=safety_overrides if safety_overrides else None,
            confidence_before_safety=confidence_before_safety,
            market_context=market_context.to_dict() if market_context and hasattr(market_context, 'to_dict') else None,
            signal_effectiveness_hours=effectiveness_hours,
            signal_valid_from=signal_valid_from,
            signal_valid_until=signal_valid_until
        )
    
    def _whale_to_signal(self, whale_metrics: WhaleMetrics) -> str:
        """Convert whale metrics to signal"""
        if whale_metrics.whale_state == "Accumulation":
            return "BUY"
        elif whale_metrics.whale_state == "Distribution":
            return "SELL"
        return "HOLD"
    
    def _whale_confidence(self, whale_metrics: WhaleMetrics) -> float:
        """
        Calculate confidence from whale metrics
        
        Reduces confidence when:
        - Whale data is stale (no recent activity AND no significant holdings, using lifetime totals)
        - "Holding" scenario gets moderate reduction (less than stale, since holding is meaningful)
        - No whale activity detected
        - Dominant whales (top holders) are inactive (NEW)
        """
        # Based on net flow relative to total pressure
        total_pressure = (whale_metrics.whale_buy_volume + 
                         whale_metrics.whale_sell_volume)
        
        base_confidence = 0.5
        if total_pressure > 0:
            dominance = abs(whale_metrics.whale_net_volume) / total_pressure
            base_confidence = min(0.9, 0.5 + dominance * 0.4)
        
        # Adjust confidence based on data source
        data_source = getattr(whale_metrics, 'whale_data_source', 'recent')
        is_stale = getattr(whale_metrics, 'is_whale_data_stale', False)
        
        if data_source == "holding":
            # Whale is holding significant position but not actively trading
            # This is still meaningful, so reduce confidence less (to 75% of base)
            base_confidence *= 0.75
            logger.debug(f"Whale data is 'holding' (no recent trades but significant holdings), "
                        f"reducing confidence to {base_confidence:.2f}")
        elif is_stale and data_source == "lifetime":
            # Truly stale data (no recent activity AND no significant holdings)
            # Cut confidence in half - whale signal becomes less influential
            base_confidence *= 0.5
            logger.debug(f"Whale data is stale (lifetime data, no significant holdings), "
                        f"reducing confidence to {base_confidence:.2f}")
        
        # ==================== DOMINANT WHALE INACTIVITY PENALTY ====================
        # Apply additional confidence penalty when dominant whales (top holders) are inactive or aging
        # This addresses the scenario where "whale accumulating" is misleading because
        # the signal comes from small whales while the dominant holders are dormant
        dominant_status = getattr(whale_metrics, 'dominant_whale_status', 'UNKNOWN')
        dominant_inactive_pct = getattr(whale_metrics, 'dominant_whale_inactive_holding_pct', 0.0)
        dominant_aging_pct = getattr(whale_metrics, 'dominant_whale_aging_holding_pct', 0.0)
        
        if dominant_status == "FULLY_INACTIVE":
            # All dominant whales are inactive (5+ days) - apply severe penalty
            penalty = whale_config.DOMINANT_WHALE_SEVERE_PENALTY
            base_confidence *= (1.0 - penalty)
            logger.info(f"Dominant whale penalty: ALL dominant whales inactive (5+ days), "
                       f"reducing confidence by {penalty*100:.0f}% to {base_confidence:.2f}")
        elif dominant_status == "PARTIALLY_INACTIVE":
            # Some dominant whales are inactive (5+ days) - apply scaled penalty based on inactive holding %
            if dominant_inactive_pct >= whale_config.DOMINANT_WHALE_INACTIVE_HOLDING_THRESHOLD:
                # Inactive whales hold significant portion - apply severe penalty
                penalty = whale_config.DOMINANT_WHALE_SEVERE_PENALTY
            else:
                # Moderate penalty based on proportion of inactive holdings
                penalty = whale_config.DOMINANT_WHALE_CONFIDENCE_PENALTY * (dominant_inactive_pct / whale_config.DOMINANT_WHALE_INACTIVE_HOLDING_THRESHOLD)
            
            base_confidence *= (1.0 - penalty)
            logger.info(f"Dominant whale penalty: {dominant_inactive_pct:.1f}% of supply held by inactive dominant whales, "
                       f"reducing confidence by {penalty*100:.0f}% to {base_confidence:.2f}")
        elif dominant_status == "AGING":
            # Dominant whales are aging (3-5 days inactive) - apply small warning penalty
            if dominant_aging_pct >= whale_config.DOMINANT_WHALE_INACTIVE_HOLDING_THRESHOLD:
                # Aging whales hold significant portion - apply full aging penalty
                penalty = whale_config.DOMINANT_WHALE_AGING_PENALTY
            else:
                # Scaled aging penalty based on proportion of aging holdings
                penalty = whale_config.DOMINANT_WHALE_AGING_PENALTY * (dominant_aging_pct / whale_config.DOMINANT_WHALE_INACTIVE_HOLDING_THRESHOLD)
            
            base_confidence *= (1.0 - penalty)
            logger.info(f"Dominant whale aging penalty: {dominant_aging_pct:.1f}% of supply held by aging dominant whales (3-5 days), "
                       f"reducing confidence by {penalty*100:.0f}% to {base_confidence:.2f}")
        elif dominant_status == "ACTIVE":
            # All dominant whales are active - no penalty, signal is reliable
            logger.debug(f"Dominant whales are ACTIVE - no confidence penalty applied")
        
        return base_confidence
    
    def _technical_to_signal(self, technical_signals: TechnicalSignals) -> str:
        """Convert technical signals to signal"""
        if technical_signals.overall_signal == "bullish":
            return "BUY"
        elif technical_signals.overall_signal == "bearish":
            return "SELL"
        return "HOLD"
    
    def _holder_to_signal(self, holder_stats: HolderStats, whale_metrics: WhaleMetrics) -> str:
        """Convert holder metrics to signal"""
        # Good distribution = BUY signal
        if whale_metrics.gini_coefficient < whale_config.GINI_SAFE and whale_metrics.top10_hold_percent < wallet_classification_config.SAFETY_GOOD_CONCENTRATION_PCT:
            return "BUY"
        # High concentration = SELL signal
        elif whale_metrics.gini_coefficient > whale_config.GINI_HIGH_RISK or whale_metrics.top10_hold_percent > wallet_classification_config.SAFETY_MEDIUM_CONCENTRATION_PCT:
            return "SELL"
        return "HOLD"
    
    def _assess_perps_user(
        self,
        perps_user_profile: Dict,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals
    ) -> Dict[str, Any]:
        """
        Convert perps user profile metrics to a risk assessment signal for Layer 5.
        
        This is the perps equivalent of risk_assessor.assess_trade() which uses
        meme-based UserProfiler. For perps, we use the already-fetched perps profile
        containing win_rate, direction_bias, profit_factor, trader_type, etc.
        
        Args:
            perps_user_profile: Dict from PerpsDataFetcher.analyze_user_perps_profile()
            whale_metrics: Current whale/large trader metrics
            technical_signals: Current technical signals
            
        Returns:
            Dict with assessment fields: rating, confidence, signal, message,
            risk_factors, profile_summary (matching meme RiskAssessment format)
        """
        win_rate = perps_user_profile.get('win_rate', 0)
        total_pnl = perps_user_profile.get('total_pnl', 0)
        profit_factor = perps_user_profile.get('profit_factor', 0)
        total_trades = perps_user_profile.get('total_trades', 0)
        direction_bias = perps_user_profile.get('direction_bias', 'UNKNOWN')
        trader_type = perps_user_profile.get('trader_type', 'UNKNOWN')
        is_profitable = perps_user_profile.get('is_profitable', False)
        trades_per_day = perps_user_profile.get('trades_per_day', 0)
        
        risk_factors = []
        
        # Insufficient data check
        if total_trades < 5:
            return {
                'rating': 'YELLOW',
                'confidence': 0.3,
                'signal': 'HOLD',
                'signal_weight': 0.0,
                'message': f"Insufficient perps trading history ({total_trades} trades). Need at least 5 for assessment.",
                'risk_factors': ['insufficient_trade_history'],
                'profile_summary': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'trader_type': trader_type,
                    'source': 'perps'
                }
            }
        
        # Calculate composite score from user metrics
        score = 0.5  # Neutral baseline
        confidence = 0.5
        
        # Win rate component (strongest signal)
        if win_rate > 60:
            score += 0.2
            confidence += 0.1
        elif win_rate > 55:
            score += 0.1
            confidence += 0.05
        elif win_rate < 35:
            score -= 0.2
            confidence += 0.1
            risk_factors.append(f"Low win rate: {win_rate:.1f}%")
        elif win_rate < 45:
            score -= 0.1
            confidence += 0.05
            risk_factors.append(f"Below-average win rate: {win_rate:.1f}%")
        
        # Profitability component
        if is_profitable and total_pnl > 0:
            score += 0.1
        elif total_pnl < 0:
            score -= 0.1
            risk_factors.append(f"Negative PnL: ${total_pnl:.2f}")
        
        # Profit factor component
        if profit_factor > 2.0:
            score += 0.1
        elif profit_factor > 1.5:
            score += 0.05
        elif profit_factor < 0.5 and profit_factor > 0:
            score -= 0.1
            risk_factors.append(f"Low profit factor: {profit_factor:.2f}")
        
        # Direction alignment with current technical signal
        tech_direction = technical_signals.overall_signal  # bullish/bearish/neutral
        if direction_bias == "LONG_BIAS" and tech_direction == "bearish":
            risk_factors.append("User has LONG bias but technicals are bearish")
            score -= 0.05
        elif direction_bias == "SHORT_BIAS" and tech_direction == "bullish":
            risk_factors.append("User has SHORT bias but technicals are bullish")
            score -= 0.05
        elif direction_bias == "LONG_BIAS" and tech_direction == "bullish":
            score += 0.05  # Alignment bonus
        elif direction_bias == "SHORT_BIAS" and tech_direction == "bearish":
            score += 0.05  # Alignment bonus
        
        # Trade volume confidence boost (more trades = more reliable profile)
        if total_trades > 50:
            confidence += 0.1
        elif total_trades > 20:
            confidence += 0.05
        
        # Clamp values
        score = max(0.0, min(1.0, score))
        confidence = max(0.3, min(0.9, confidence))
        
        # Determine rating and signal
        if score >= 0.65:
            rating = 'GREEN'
            signal = 'BUY'
            message = (f"Favorable perps profile: {win_rate:.1f}% win rate, "
                      f"PF={profit_factor:.2f}, {trader_type}")
        elif score <= 0.35:
            rating = 'RED'
            signal = 'SELL'
            message = (f"Unfavorable perps profile: {win_rate:.1f}% win rate, "
                      f"PnL=${total_pnl:.2f}")
        else:
            rating = 'YELLOW'
            signal = 'HOLD'
            message = (f"Mixed perps profile: {win_rate:.1f}% win rate, "
                      f"PF={profit_factor:.2f}, {direction_bias}")
        
        if risk_factors:
            message += f". Risk factors: {'; '.join(risk_factors)}"
        
        logger.info(f"Perps user assessment: {rating} ({signal}), confidence: {confidence:.2f}, "
                   f"win_rate={win_rate:.1f}%, trades={total_trades}")
        
        return {
            'rating': rating,
            'confidence': round(confidence, 4),
            'signal': signal,
            'signal_weight': round(score - 0.5, 4),  # -0.5 to +0.5 range
            'message': message,
            'risk_factors': risk_factors,
            'profile_summary': {
                'total_trades': total_trades,
                'win_rate': round(win_rate, 4),
                'trader_type': trader_type,
                'direction_bias': direction_bias,
                'profit_factor': round(profit_factor, 4),
                'total_pnl': round(total_pnl, 2),
                'trades_per_day': round(trades_per_day, 2),
                'source': 'perps'
            }
        }
    
    def _apply_safety_overrides(
        self,
        signal: str,
        confidence: float,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats,
        risk_level: str,
        token_type: str = "meme",
        market_context = None,
        recent_analysis: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> tuple:
        """
        Apply safety overrides to prevent dangerous trades.
        
        Perps-aware: Holder-based and gini-based checks are SKIPPED for perps tokens
        because perps don't have traditional holders, gini coefficients, or top10 concentration.
        For perps, only whale-behavior and risk-level overrides apply.
        
        For meme tokens: All checks apply (rug pull protection via holder/gini/concentration).
        """
        original_signal = signal
        confidence_before_safety = confidence
        overrides_applied = []
        is_perps = is_perps_token(token_type)
        
        # Clamp top10_hold_percent to valid range (0-100) to match API response
        clamped_top10 = max(0, min(100, whale_metrics.top10_hold_percent))
        
        # Log the metrics for debugging
        logger.info(f"Safety override check: gini={whale_metrics.gini_coefficient:.2f}, "
                   f"top10={clamped_top10:.1f}%, holders={holder_stats.active_holders}, "
                   f"signal={signal}, token_type={token_type}")
        
        # =====================================================================
        # MEME-ONLY SAFETY OVERRIDES (holder/gini/concentration checks)
        # SKIPPED for perps (no holders/gini) and for degraded mode
        # (holder data unavailable — empty defaults would trigger false overrides)
        # =====================================================================
        holder_data_unavailable = (
            holder_stats.active_holders == 0 and
            holder_stats.total_holders == 0
        )
        if holder_data_unavailable and not is_perps:
            logger.info("Skipping holder/gini safety overrides — holder data unavailable (degraded mode)")
        
        if not is_perps and not holder_data_unavailable:
            # CRITICAL: Never BUY when top 10 holders control > threshold
            if signal == "BUY" and clamped_top10 > wallet_classification_config.SAFETY_HIGH_CONCENTRATION_PCT:
                signal = "HOLD"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                               confidence * wallet_classification_config.SAFETY_CONFIDENCE_HIGH_MULTIPLIER)
                overrides_applied.append({
                    'check': 'top10_concentration',
                    'condition': f'top10_hold_percent > 80%',
                    'value': round(clamped_top10, 1),
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'Top 10 concentration too high ({clamped_top10:.1f}%)'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - Top 10 concentration too high")
            
            # CRITICAL: Flag extreme concentration even for HOLD/SELL signals
            if clamped_top10 >= wallet_classification_config.SAFETY_EXTREME_CONCENTRATION_PCT:
                if signal == "HOLD":
                    signal = "SELL"
                    confidence = min(wallet_classification_config.SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER, confidence)
                    overrides_applied.append({
                        'check': 'extreme_concentration',
                        'condition': f'top10_hold_percent >= {wallet_classification_config.SAFETY_EXTREME_CONCENTRATION_PCT}%',
                        'value': round(clamped_top10, 1),
                        'original_signal': original_signal,
                        'new_signal': signal,
                        'reason': f'EXTREME concentration risk - Top 10 control {clamped_top10:.1f}% (rug pull risk)'
                    })
                    logger.warning(f"Safety override: {original_signal} -> {signal} - EXTREME concentration ({clamped_top10:.1f}%)")
                elif signal == "BUY":
                    if not any(o['check'] == 'top10_concentration' for o in overrides_applied):
                        overrides_applied.append({
                            'check': 'extreme_concentration',
                            'condition': f'top10_hold_percent >= {wallet_classification_config.SAFETY_EXTREME_CONCENTRATION_PCT}%',
                            'value': round(clamped_top10, 1),
                            'original_signal': original_signal,
                            'new_signal': signal,
                            'reason': f'EXTREME concentration risk - Top 10 control {clamped_top10:.1f}%'
                        })
            
            # CRITICAL: Flag when top10 is 100% - means <= 10 holders total
            if clamped_top10 == 100.0 and signal != "SELL":
                signal = "SELL"
                confidence = min(wallet_classification_config.SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER, confidence)
                if not any(o['check'] in ['extreme_concentration', 'very_few_holders_sell'] for o in overrides_applied):
                    overrides_applied.append({
                        'check': 'top10_is_all_holders',
                        'condition': f'top10_hold_percent == 100% (≤10 total holders)',
                        'value': f'{clamped_top10:.1f}%, {holder_stats.active_holders} holders',
                        'original_signal': original_signal,
                        'new_signal': signal,
                        'reason': f'Top 10 wallets control 100% of supply ({holder_stats.active_holders} total holders) - extreme rug pull risk'
                    })
                    logger.warning(f"Safety override: {original_signal} -> {signal} - Top 10 = 100% ({holder_stats.active_holders} total holders)")
            
            # CRITICAL: Never BUY when Gini > extreme threshold
            if signal == "BUY" and whale_metrics.gini_coefficient > whale_config.GINI_EXTREME_THRESHOLD:
                signal = "HOLD"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                               confidence * wallet_classification_config.SAFETY_CONFIDENCE_HIGH_MULTIPLIER)
                overrides_applied.append({
                    'check': 'gini_coefficient',
                    'condition': f'gini_coefficient > {whale_config.GINI_EXTREME_THRESHOLD}',
                    'value': round(whale_metrics.gini_coefficient, 2),
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'Extreme inequality (Gini: {whale_metrics.gini_coefficient:.2f})'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - Extreme inequality")
            
            # CRITICAL: High gini with few holders is extremely risky
            high_gini_threshold = whale_config.CONCENTRATION_MEDIUM_HIGH_PCT
            few_holders_threshold = wallet_classification_config.SAFETY_SUSPICIOUS_GINI_HOLDERS
            if signal == "BUY" and whale_metrics.gini_coefficient > high_gini_threshold and holder_stats.active_holders < few_holders_threshold:
                signal = "HOLD"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                               confidence * wallet_classification_config.SAFETY_CONFIDENCE_HIGH_MULTIPLIER)
                overrides_applied.append({
                    'check': 'high_gini_few_holders',
                    'condition': f'gini_coefficient > {high_gini_threshold} AND active_holders < {few_holders_threshold}',
                    'value': f'Gini: {whale_metrics.gini_coefficient:.2f}, Holders: {holder_stats.active_holders}',
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'High inequality ({whale_metrics.gini_coefficient:.2f}) with only {holder_stats.active_holders} holders - rug pull risk'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - High gini with few holders")
            
            # CRITICAL: Upgrade HOLD to SELL for extreme concentration with few holders
            extreme_gini_threshold = whale_config.CONCENTRATION_HIGH_PCT
            extreme_holders_threshold = wallet_classification_config.SAFETY_EXTREMELY_FEW_HOLDERS
            if signal == "HOLD" and whale_metrics.gini_coefficient >= extreme_gini_threshold and holder_stats.active_holders < extreme_holders_threshold:
                signal = "SELL"
                confidence = min(wallet_classification_config.SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER, confidence)
                overrides_applied.append({
                    'check': 'extreme_concentration_sell',
                    'condition': f'gini_coefficient >= {extreme_gini_threshold} AND active_holders < {extreme_holders_threshold}',
                    'value': f'Gini: {whale_metrics.gini_coefficient:.2f}, Holders: {holder_stats.active_holders}',
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'EXTREME concentration ({whale_metrics.gini_coefficient:.2f} gini) with only {holder_stats.active_holders} holders - likely rug pull'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - Extreme concentration forces SELL")
            
            # CRITICAL: Upgrade HOLD to SELL for tokens with VERY few holders
            very_few_holders_threshold = 5
            if signal == "HOLD" and holder_stats.active_holders <= very_few_holders_threshold and holder_stats.active_holders > 0:
                signal = "SELL"
                confidence = min(wallet_classification_config.SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER, confidence)
                overrides_applied.append({
                    'check': 'very_few_holders_sell',
                    'condition': f'active_holders <= {very_few_holders_threshold}',
                    'value': holder_stats.active_holders,
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'EXTREMELY few holders ({holder_stats.active_holders}) - high rug pull risk regardless of other metrics'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - EXTREMELY few holders ({holder_stats.active_holders})")
            
            # CRITICAL: If active_holders is 0 but total_holders > 0, everyone has sold
            if signal == "HOLD" and holder_stats.active_holders == 0 and holder_stats.total_holders > 0:
                signal = "SELL"
                confidence = min(wallet_classification_config.SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER, confidence)
                overrides_applied.append({
                    'check': 'all_holders_sold',
                    'condition': f'active_holders == 0 AND total_holders > 0',
                    'value': f'0 active holders, {holder_stats.total_holders} historical',
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'NO active holders remaining ({holder_stats.total_holders} historical holders have ALL sold) - token likely dead/rugged'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - NO active holders (all {holder_stats.total_holders} sold)")
            
            # CRITICAL: Flag when Gini is suspiciously neutral with very few holders
            neutral_gini = whale_config.GINI_NEUTRAL_DEFAULT
            if (signal == "BUY" and 
                abs(whale_metrics.gini_coefficient - neutral_gini) < wallet_classification_config.SAFETY_SUSPICIOUS_GINI_TOLERANCE and 
                holder_stats.active_holders <= wallet_classification_config.SAFETY_SUSPICIOUS_GINI_HOLDERS):
                signal = "HOLD"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                               confidence * wallet_classification_config.SAFETY_CONFIDENCE_EXTREME_MULTIPLIER)
                overrides_applied.append({
                    'check': 'suspicious_gini',
                    'condition': f'gini_coefficient ≈ {neutral_gini} AND active_holders <= {wallet_classification_config.SAFETY_SUSPICIOUS_GINI_HOLDERS}',
                    'value': f'Gini: {whale_metrics.gini_coefficient:.2f}, Holders: {holder_stats.active_holders}',
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'Suspicious: Neutral gini ({whale_metrics.gini_coefficient:.2f}) with only {holder_stats.active_holders} holders - likely missing data or calculation error'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - Suspicious gini value with few holders")
            
            # CRITICAL: Never BUY when holder count is very low
            if signal == "BUY" and holder_stats.active_holders < wallet_classification_config.SAFETY_FEW_HOLDERS:
                signal = "HOLD"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                               confidence * wallet_classification_config.SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER)
                overrides_applied.append({
                    'check': 'holder_count',
                    'condition': f'active_holders < {wallet_classification_config.SAFETY_FEW_HOLDERS}',
                    'value': holder_stats.active_holders,
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'Too few holders ({holder_stats.active_holders})'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - Too few holders")
            
            # CRITICAL: Extra safety for tokens with extremely few holders
            if signal == "BUY" and holder_stats.active_holders < wallet_classification_config.SAFETY_EXTREMELY_FEW_HOLDERS:
                signal = "HOLD"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                               confidence * wallet_classification_config.SAFETY_CONFIDENCE_EXTREME_MULTIPLIER)
                overrides_applied.append({
                    'check': 'extreme_low_holders',
                    'condition': f'active_holders < {wallet_classification_config.SAFETY_EXTREMELY_FEW_HOLDERS}',
                    'value': holder_stats.active_holders,
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'EXTREMELY few holders ({holder_stats.active_holders}) - high rug pull risk'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - EXTREMELY few holders ({holder_stats.active_holders})")
            
            # CRITICAL: Flag when top10 concentration is 0 with very few holders
            if signal == "BUY" and clamped_top10 == 0 and holder_stats.active_holders < wallet_classification_config.SAFETY_SUSPICIOUS_CONCENTRATION_HOLDERS:
                signal = "HOLD"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                               confidence * wallet_classification_config.SAFETY_CONFIDENCE_HIGH_MULTIPLIER)
                overrides_applied.append({
                    'check': 'suspicious_concentration',
                    'condition': f'top10_hold_percent == 0 AND active_holders < {wallet_classification_config.SAFETY_SUSPICIOUS_CONCENTRATION_HOLDERS}',
                    'value': f'{clamped_top10}%, {holder_stats.active_holders} holders',
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'Suspicious: 0% top10 concentration with only {holder_stats.active_holders} holders - likely missing data'
                })
                logger.warning(f"Safety override: {original_signal} -> {signal} - Suspicious concentration data")
        
        # =====================================================================
        # UNIVERSAL SAFETY OVERRIDES (apply to BOTH meme and perps)
        # These are based on whale behavior and risk level, not holder metrics
        # =====================================================================
        
        # Override BUY to HOLD when whales/large traders are distributing
        if signal == "BUY" and whale_metrics.whale_state == "Distribution":
            signal = "HOLD"
            confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_MIN, 
                           confidence * wallet_classification_config.SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER)
            overrides_applied.append({
                'check': 'whale_distribution',
                'condition': 'whale_state == "Distribution"',
                'value': whale_metrics.whale_state,
                'original_signal': original_signal,
                'new_signal': signal,
                'reason': 'Large trader/whale distribution detected'
            })
            logger.warning(f"Safety override: {original_signal} -> {signal} - Whale distribution")
        
        # Strengthen SELL signal when distribution + high concentration (meme only, perps has no top10)
        if not is_perps and whale_metrics.whale_state == "Distribution" and clamped_top10 > wallet_classification_config.SAFETY_MEDIUM_CONCENTRATION_PCT:
            if signal == "HOLD":
                signal = "SELL"
                confidence = max(wallet_classification_config.SAFETY_CONFIDENCE_HIGH_MULTIPLIER, confidence)
                overrides_applied.append({
                    'check': 'distribution_concentration',
                    'condition': f'whale_state == "Distribution" AND top10_hold_percent > {wallet_classification_config.SAFETY_MEDIUM_CONCENTRATION_PCT}%',
                    'value': f'{whale_metrics.whale_state}, {clamped_top10:.1f}%',
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': 'Distribution with high concentration'
                })
                logger.info(f"Signal strengthened: HOLD -> SELL - Distribution with high concentration")
        
        # Reduce confidence for BUY when risk is High
        if signal == "BUY" and risk_level == "High":
            old_confidence = confidence
            confidence = confidence * 0.8
            overrides_applied.append({
                'check': 'high_risk',
                'condition': 'risk_level == "High"',
                'value': risk_level,
                'original_signal': signal,
                'new_signal': signal,
                'reason': f'Confidence reduced for high-risk BUY ({old_confidence:.2f} -> {confidence:.2f})'
            })
        
        # =====================================================================
        # MARKET REGIME SAFETY OVERRIDES (CryptoRank global data)
        # =====================================================================
        if market_context and not getattr(market_context, 'is_default', True):
            fg = market_context.fear_greed
            mcap_change = market_context.total_market_cap_change
            
            # Extreme fear: two severity tiers
            if signal == "BUY" and fg <= api_config.MARKET_EXTREME_FEAR_THRESHOLD:
                crash_threshold = api_config.MARKET_FEAR_CRASH_MCAP_THRESHOLD
                if mcap_change < crash_threshold:
                    # Tier 2: extreme fear + market crash — stronger penalty
                    old_confidence = confidence
                    penalty = api_config.MARKET_FEAR_CONFIDENCE_PENALTY
                    confidence = max(0.3, confidence * (1.0 - penalty))
                    overrides_applied.append({
                        'check': 'market_extreme_fear_crash',
                        'condition': f'fear_greed <= {api_config.MARKET_EXTREME_FEAR_THRESHOLD} AND market_cap_change < {crash_threshold}%',
                        'value': f'F&G={fg}, MCap change={mcap_change:.2f}%',
                        'original_signal': signal,
                        'new_signal': signal,
                        'reason': f'Extreme fear + market crash — BUY confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
                    })
                    logger.warning(
                        f"Market regime override: extreme fear (F&G={fg}) + crash ({mcap_change:.2f}%) "
                        f"— BUY confidence {old_confidence:.2f} -> {confidence:.2f}"
                    )
                else:
                    # Tier 1: extreme fear alone — moderate penalty
                    old_confidence = confidence
                    penalty = api_config.MARKET_FEAR_ALONE_CONFIDENCE_PENALTY
                    confidence = max(0.3, confidence * (1.0 - penalty))
                    overrides_applied.append({
                        'check': 'market_extreme_fear',
                        'condition': f'fear_greed <= {api_config.MARKET_EXTREME_FEAR_THRESHOLD}',
                        'value': f'F&G={fg}, MCap change={mcap_change:.2f}%',
                        'original_signal': signal,
                        'new_signal': signal,
                        'reason': f'Extreme fear — BUY confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
                    })
                    logger.warning(
                        f"Market regime override: extreme fear (F&G={fg}) "
                        f"— BUY confidence {old_confidence:.2f} -> {confidence:.2f}"
                    )
            
            # Extreme greed: penalize BUY confidence (frothy market)
            elif signal == "BUY" and fg >= api_config.MARKET_EXTREME_GREED_THRESHOLD:
                old_confidence = confidence
                penalty = api_config.MARKET_GREED_CONFIDENCE_PENALTY
                confidence = max(0.3, confidence * (1.0 - penalty))
                overrides_applied.append({
                    'check': 'market_extreme_greed',
                    'condition': f'fear_greed >= {api_config.MARKET_EXTREME_GREED_THRESHOLD}',
                    'value': f'F&G={fg}',
                    'original_signal': signal,
                    'new_signal': signal,
                    'reason': f'Extreme greed — BUY confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
                })
                logger.warning(
                    f"Market regime override: extreme greed (F&G={fg}) "
                    f"— BUY confidence {old_confidence:.2f} -> {confidence:.2f}"
                )
            
            # BEAR market regime with low favorability: penalize BUY confidence
            market_regime = getattr(market_context, 'market_regime', None)
            market_fav = getattr(market_context, 'market_favorability', 1.0)
            fav_threshold = api_config.BEAR_MARKET_FAVORABILITY_THRESHOLD
            if signal == "BUY" and market_regime == "BEAR" and market_fav < fav_threshold:
                old_confidence = confidence
                penalty = api_config.BEAR_MARKET_CONFIDENCE_PENALTY
                confidence = max(0.3, confidence * (1.0 - penalty))
                overrides_applied.append({
                    'check': 'bear_market_regime',
                    'condition': f'market_regime == "BEAR" AND market_favorability < {fav_threshold}',
                    'value': f'regime={market_regime}, favorability={market_fav:.2f}',
                    'original_signal': signal,
                    'new_signal': signal,
                    'reason': f'BEAR market regime — BUY confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
                })
                logger.warning(
                    f"Market regime override: BEAR (favorability={market_fav:.2f}) "
                    f"— BUY confidence {old_confidence:.2f} -> {confidence:.2f}"
                )
            
            # Strengthen SELL confidence in extreme fear
            if signal == "SELL" and fg <= api_config.MARKET_EXTREME_FEAR_THRESHOLD:
                old_confidence = confidence
                boost = api_config.SELL_FEAR_CONFIDENCE_BOOST
                confidence = min(0.95, confidence * (1.0 + boost))
                overrides_applied.append({
                    'check': 'sell_extreme_fear_boost',
                    'condition': f'signal == SELL AND fear_greed <= {api_config.MARKET_EXTREME_FEAR_THRESHOLD}',
                    'value': f'F&G={fg}',
                    'original_signal': signal,
                    'new_signal': signal,
                    'reason': f'Extreme fear reinforces SELL — confidence boosted ({old_confidence:.2f} -> {confidence:.2f})'
                })
                logger.info(
                    f"SELL confidence boosted by extreme fear (F&G={fg}): "
                    f"{old_confidence:.2f} -> {confidence:.2f}"
                )
            
            # Strengthen SELL confidence in BEAR regime
            if signal == "SELL" and market_regime == "BEAR" and market_fav < fav_threshold:
                old_confidence = confidence
                boost = api_config.SELL_BEAR_CONFIDENCE_BOOST
                confidence = min(0.95, confidence * (1.0 + boost))
                overrides_applied.append({
                    'check': 'sell_bear_regime_boost',
                    'condition': f'signal == SELL AND market_regime == "BEAR" AND market_favorability < {fav_threshold}',
                    'value': f'regime={market_regime}, favorability={market_fav:.2f}',
                    'original_signal': signal,
                    'new_signal': signal,
                    'reason': f'BEAR market regime reinforces SELL — confidence boosted ({old_confidence:.2f} -> {confidence:.2f})'
                })
                logger.info(
                    f"SELL confidence boosted by BEAR regime (favorability={market_fav:.2f}): "
                    f"{old_confidence:.2f} -> {confidence:.2f}"
                )
        
        # =====================================================================
        # VOLATILITY SAFETY OVERRIDE (needs recent_analysis, token-type-aware)
        # =====================================================================
        if recent_analysis:
            vol_24h = recent_analysis.get('volatility_24h', 0)
            price_change_pct = recent_analysis.get('price_change_24h_pct', 0)

            vol_threshold = safety_config.EXTREME_VOLATILITY_THRESHOLD_PERPS if is_perps else safety_config.EXTREME_VOLATILITY_THRESHOLD_MEME
            swing_threshold = safety_config.EXTREME_PRICE_SWING_PCT_PERPS if is_perps else safety_config.EXTREME_PRICE_SWING_PCT_MEME

            if signal == "BUY" and vol_24h > vol_threshold:
                signal = "HOLD"
                overrides_applied.append({
                    'check': 'extreme_volatility',
                    'condition': f'volatility_24h > {vol_threshold} ({token_type})',
                    'value': round(vol_24h, 4),
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'Extreme 24h volatility ({vol_24h:.4f}) — too unstable for confident entry'
                })
                logger.warning(f"Safety override: BUY -> HOLD — extreme volatility ({vol_24h:.4f})")

            if signal == "BUY" and abs(price_change_pct) > swing_threshold:
                old_confidence = confidence
                confidence = max(0.3, confidence * (1.0 - safety_config.VOLATILITY_CONFIDENCE_PENALTY))
                overrides_applied.append({
                    'check': 'extreme_price_swing',
                    'condition': f'|price_change_24h_pct| > {swing_threshold}% ({token_type})',
                    'value': round(price_change_pct, 2),
                    'original_signal': signal,
                    'new_signal': signal,
                    'reason': f'Extreme 24h price swing ({price_change_pct:.1f}%) — confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
                })
                logger.warning(f"Safety override: extreme price swing ({price_change_pct:.1f}%) — confidence {old_confidence:.2f} -> {confidence:.2f}")

            if signal == "SELL" and price_change_pct < 0 and abs(price_change_pct) > swing_threshold:
                old_confidence = confidence
                confidence = min(0.95, confidence * (1.0 + safety_config.VOLATILITY_CONFIDENCE_PENALTY))
                overrides_applied.append({
                    'check': 'crash_confirms_sell',
                    'condition': f'price_change_24h_pct < -{swing_threshold}% ({token_type}) AND signal == SELL',
                    'value': round(price_change_pct, 2),
                    'original_signal': signal,
                    'new_signal': signal,
                    'reason': f'Extreme price crash ({price_change_pct:.1f}%) confirms SELL — confidence boosted ({old_confidence:.2f} -> {confidence:.2f})'
                })
                logger.info(f"SELL confidence boosted by crash ({price_change_pct:.1f}%): {old_confidence:.2f} -> {confidence:.2f}")

        # =====================================================================
        # TOKEN AGE SAFETY OVERRIDE (meme only, uses whale_metrics.phase)
        # =====================================================================
        if not is_perps and signal == "BUY" and whale_metrics.phase == "P1":
            old_confidence = confidence
            confidence = max(0.3, confidence * (1.0 - safety_config.TOKEN_AGE_P1_CONFIDENCE_PENALTY))
            overrides_applied.append({
                'check': 'token_age_p1',
                'condition': 'whale_metrics.phase == "P1" (0-3 days old)',
                'value': whale_metrics.phase,
                'original_signal': signal,
                'new_signal': signal,
                'reason': f'Token is in launch phase (P1) — high rug-pull risk. Confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
            })
            logger.warning(f"Safety override: P1 token — confidence {old_confidence:.2f} -> {confidence:.2f}")

            if holder_stats.total_holders < safety_config.TOKEN_AGE_P1_MAX_HOLDERS_FOR_HOLD:
                signal = "HOLD"
                overrides_applied.append({
                    'check': 'token_age_p1_few_holders',
                    'condition': f'P1 AND total_holders < {safety_config.TOKEN_AGE_P1_MAX_HOLDERS_FOR_HOLD}',
                    'value': holder_stats.total_holders,
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'P1 token with only {holder_stats.total_holders} holders — downgraded to HOLD'
                })
                logger.warning(f"Safety override: BUY -> HOLD — P1 token with {holder_stats.total_holders} holders")

        # =====================================================================
        # EXTREME FUNDING RATE OVERRIDE (perps only, needs market_data)
        # =====================================================================
        if is_perps and market_data:
            next_funding_rate = market_data.get('next_funding_rate', 0)
            if next_funding_rate is not None:
                try:
                    next_funding_rate = float(next_funding_rate)
                except (ValueError, TypeError):
                    next_funding_rate = 0

                if abs(next_funding_rate) > safety_config.EXTREME_FUNDING_THRESHOLD:
                    crowded_long = next_funding_rate > 0 and signal == "BUY"
                    crowded_short = next_funding_rate < 0 and signal == "SELL"
                    if crowded_long or crowded_short:
                        old_confidence = confidence
                        confidence = max(0.3, confidence * (1.0 - safety_config.FUNDING_RATE_CONFIDENCE_PENALTY))
                        side = "long" if crowded_long else "short"
                        overrides_applied.append({
                            'check': 'extreme_funding_rate',
                            'condition': f'|next_funding_rate| > {safety_config.EXTREME_FUNDING_THRESHOLD} AND signal aligns with crowded {side}',
                            'value': next_funding_rate,
                            'original_signal': signal,
                            'new_signal': signal,
                            'reason': f'Extreme funding rate ({next_funding_rate}) — crowded {side} risk. Confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
                        })
                        logger.warning(f"Safety override: extreme funding rate ({next_funding_rate}) — crowded {side}, confidence {old_confidence:.2f} -> {confidence:.2f}")

        # =====================================================================
        # WHALE FLOW CONFLICT OVERRIDE (perps only — market-wide whale positioning)
        # Penalizes confidence when signal direction conflicts with whale flow
        # =====================================================================
        if is_perps and market_data:
            whale_flow_net = market_data.get('whale_flow_net_avg', 0)
            if whale_flow_net != 0:
                whales_bearish = whale_flow_net < -0.02
                whales_bullish = whale_flow_net > 0.02
                signal_conflicts = (
                    (signal == "BUY" and whales_bearish) or
                    (signal == "SELL" and whales_bullish)
                )
                if signal_conflicts:
                    old_confidence = confidence
                    penalty = min(0.15, abs(whale_flow_net) * 2)
                    confidence = max(0.3, confidence - penalty)
                    direction = "bearish" if whales_bearish else "bullish"
                    overrides_applied.append({
                        'check': 'whale_flow_conflict',
                        'condition': f'whale_flow_net_avg={whale_flow_net:.4f} ({direction}) conflicts with {signal}',
                        'value': whale_flow_net,
                        'original_signal': original_signal,
                        'new_signal': signal,
                        'reason': f'Market-wide whale flow is {direction} (net={whale_flow_net:.4f}) — conflicts with {signal}. Confidence reduced ({old_confidence:.2f} -> {confidence:.2f})'
                    })
                    logger.warning(f"Safety override: whale flow conflict — whales {direction} vs {signal}, confidence {old_confidence:.2f} -> {confidence:.2f}")

        # =====================================================================
        # LOW LIQUIDITY SAFETY OVERRIDE (meme only, needs market_data)
        # =====================================================================
        if not is_perps and market_data:
            liquidity_usd = market_data.get('liquidity_usd', float('inf'))
            if signal == "BUY" and liquidity_usd < safety_config.MIN_LIQUIDITY_USD:
                signal = "HOLD"
                overrides_applied.append({
                    'check': 'low_liquidity',
                    'condition': f'liquidity_usd < ${safety_config.MIN_LIQUIDITY_USD:,.0f}',
                    'value': round(liquidity_usd, 2),
                    'original_signal': original_signal,
                    'new_signal': signal,
                    'reason': f'Extremely low liquidity (${liquidity_usd:,.2f}) — insufficient for safe entry'
                })
                logger.warning(f"Safety override: BUY -> HOLD — low liquidity (${liquidity_usd:,.2f})")
            elif signal == "BUY" and liquidity_usd < safety_config.LOW_LIQUIDITY_WARNING_USD:
                overrides_applied.append({
                    'check': 'low_liquidity_warning',
                    'condition': f'liquidity_usd < ${safety_config.LOW_LIQUIDITY_WARNING_USD:,.0f}',
                    'value': round(liquidity_usd, 2),
                    'original_signal': signal,
                    'new_signal': signal,
                    'reason': f'Low liquidity warning (${liquidity_usd:,.2f}) — exercise caution'
                })

        return signal, confidence, overrides_applied, confidence_before_safety
    
    def _ml_prediction_routed(
        self,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats,
        candle_data: Optional[Dict] = None,
        token_type: str = "meme",
        market_data: Optional[Dict] = None,
        market_context = None
    ) -> tuple:
        """
        Route ML prediction based on token type
        
        Args:
            token_type: 'meme' or 'perps'
            market_data: Additional market data for perps (funding rate, OI, etc.)
            market_context: MarketContext with global market data
            
        Returns:
            Tuple of (signal, confidence)
        """
        if is_perps_token(token_type):
            # Use perps model
            if self.perps_model is not None:
                return self._perps_ml_prediction(
                    technical_signals=technical_signals,
                    candle_data=candle_data,
                    market_data=market_data,
                    market_context=market_context
                )
            else:
                logger.warning("Perps model not loaded, using rule-based prediction")
                return self._rule_based_prediction(
                    whale_metrics=whale_metrics,
                    technical_signals=technical_signals,
                    holder_stats=holder_stats
                )
        else:
            # Use meme model (default)
            if self.model is not None:
                return self._ml_prediction(
                    whale_metrics=whale_metrics,
                    technical_signals=technical_signals,
                    holder_stats=holder_stats,
                    candle_data=candle_data,
                    market_context=market_context
                )
            else:
                logger.warning("Meme model not loaded, using rule-based prediction")
                return self._rule_based_prediction(
                    whale_metrics=whale_metrics,
                    technical_signals=technical_signals,
                    holder_stats=holder_stats
                )
    
    def _ml_prediction(
        self,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats,
        candle_data: Optional[Dict] = None,
        market_context = None
    ) -> tuple:
        """Generate signal using meme ML model"""
        try:
            # Build feature vector with all 36 features
            features = self._build_feature_vector(
                whale_metrics=whale_metrics,
                technical_signals=technical_signals,
                holder_stats=holder_stats,
                candle_data=candle_data,
                market_context=market_context
            )
            
            # Scale features if scaler available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Get prediction
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            # Classify signal
            if probability >= signal_config.XGBOOST_LONG_THRESHOLD:
                signal = "BUY"
                confidence = probability
            elif probability <= (1 - signal_config.XGBOOST_SHORT_THRESHOLD):
                signal = "SELL"
                confidence = 1 - probability
            else:
                signal = "HOLD"
                confidence = 1 - abs(probability - 0.5) * 2
            
            return signal, float(confidence)
            
        except Exception as e:
            logger.error(f"Meme ML prediction failed: {e}, falling back to rules")
            return self._rule_based_prediction(
                whale_metrics=whale_metrics,
                technical_signals=technical_signals,
                holder_stats=holder_stats
            )
    
    def _perps_ml_prediction(
        self,
        technical_signals: TechnicalSignals,
        candle_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        market_context = None
    ) -> tuple:
        """
        Generate signal using perps ML model
        
        The model is trained with binary classification (LONG=1 vs NOT_LONG=0):
        - Probability of LONG >= 0.5 => BUY signal
        - Probability of LONG < 0.35 => SELL signal (low LONG probability)
        - Otherwise => HOLD signal
        
        Args:
            technical_signals: Technical indicator signals
            candle_data: OHLCV candle data
            market_data: Perps-specific data (funding rate, OI, etc.)
            market_context: MarketContext with global market data
            
        Returns:
            Tuple of (signal, confidence)
        """
        try:
            # Build perps feature vector
            features = self._build_perps_feature_vector(
                technical_signals=technical_signals,
                candle_data=candle_data,
                market_data=market_data,
                market_context=market_context
            )
            
            # Check if scaler expects different number of features
            if self.perps_scaler is not None:
                expected_features = self.perps_scaler.n_features_in_
                if len(features) != expected_features:
                    logger.warning(f"Feature mismatch: built {len(features)}, scaler expects {expected_features}. Using subset.")
                    # Use first N features if we have more, pad with zeros if we have less
                    if len(features) > expected_features:
                        features = features[:expected_features]
                    else:
                        features = np.pad(features, (0, expected_features - len(features)), constant_values=0)
                features_scaled = self.perps_scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Get prediction probabilities
            probabilities = self.perps_model.predict_proba(features_scaled)[0]
            
            # Binary model: [P(NOT_LONG), P(LONG)]
            prob_not_long = probabilities[0]
            prob_long = probabilities[1]
            
            # Map binary output to trading signals
            # High confidence LONG => BUY
            # Low confidence LONG => SELL
            # Medium => HOLD
            if prob_long >= 0.55:
                signal = "BUY"
                confidence = prob_long
            elif prob_long < 0.35:
                signal = "SELL"
                confidence = prob_not_long
            else:
                signal = "HOLD"
                confidence = 1.0 - abs(prob_long - 0.5) * 2  # Higher when closer to 0.5
            
            logger.info(f"Perps ML prediction: P(LONG)={prob_long:.3f}, signal={signal}, confidence={confidence:.3f}")
            
            return signal, float(confidence)
            
        except Exception as e:
            logger.error(f"Perps ML prediction failed: {e}")
            # Return HOLD with low confidence on error
            return "HOLD", 0.5
    
    def _build_perps_feature_vector(
        self,
        technical_signals: TechnicalSignals,
        candle_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        market_context = None
    ) -> np.ndarray:
        """
        Build feature vector for perps model (56 features)
        
        The model was trained on these features in order:
        ticker_encoded, return_1h, return_4h, return_24h, return_168h,
        price_velocity, price_acceleration, rsi_7, rsi_14, rsi_21,
        rsi_oversold, rsi_overbought, macd, macd_signal, macd_histogram,
        macd_crossover, adx, strong_trend, stochastic_k, stochastic_d,
        williams_r, ema_9, ema_20, ema_50, atr, bb_width,
        high_low_range_pct, close_position_in_range, volume, volume_ma,
        volume_ratio, obv, mfi, volume_roc, volume_spike, trade_count,
        trade_intensity, funding_rate, funding_positive, open_interest,
        oi_momentum, oi_increasing, oi_volatility, oi_trend_strength,
        price_oi_divergence, next_funding_rate, funding_rate_percentile,
        price_change_24h, volume_24h, btc_correlation_24h,
        relative_strength_vs_btc, sector_momentum, liquidity_score,
        hour_of_day, day_of_week, is_weekend
        """
        if candle_data is None:
            candle_data = {}
        if market_data is None:
            market_data = {}
        
        # Helper to safely get values
        def safe_get(d, keys, default=0.0):
            for k in keys if isinstance(keys, list) else [keys]:
                if k in d and d[k] is not None:
                    try:
                        return float(d[k])
                    except:
                        pass
            return default
        
        def safe_attr(obj, attr, default=0.0):
            val = getattr(obj, attr, default)
            return float(val) if val is not None else default
        
        # Current time for time features
        now = datetime.now()
        rsi = safe_attr(technical_signals, 'rsi', 50.0)
        
        features = []
        
        # 1. ticker_encoded (0 for single token)
        features.append(0.0)
        
        # 2-5. Returns
        features.append(safe_get(candle_data, ['return_1h', 'returns'], 0.0))
        features.append(safe_get(candle_data, 'return_4h', 0.0))
        features.append(safe_get(candle_data, ['return_24h', 'returns'], 0.0))
        features.append(safe_get(candle_data, 'return_168h', 0.0))
        
        # 6-7. Price velocity and acceleration
        features.append(safe_attr(technical_signals, 'price_momentum_5', 0.0))
        features.append(safe_attr(technical_signals, 'price_momentum_10', 0.0))
        
        # 8-12. RSI features
        features.append(rsi * 0.9)  # rsi_7 approximation
        features.append(rsi)  # rsi_14
        features.append(rsi * 1.05)  # rsi_21 approximation
        features.append(1.0 if rsi < 30 else 0.0)  # rsi_oversold
        features.append(1.0 if rsi > 70 else 0.0)  # rsi_overbought
        
        # 13-16. MACD features
        macd = safe_attr(technical_signals, 'macd_line', 0.0)
        macd_signal = safe_attr(technical_signals, 'macd_signal', 0.0)
        features.append(macd)
        features.append(macd_signal)
        features.append(safe_attr(technical_signals, 'macd_histogram', macd - macd_signal))
        features.append(1.0 if macd > macd_signal else 0.0)  # macd_crossover
        
        # 17-18. ADX features
        adx = safe_attr(technical_signals, 'signal_strength', 0.5) * 100
        features.append(adx)
        features.append(1.0 if adx > 25 else 0.0)  # strong_trend
        
        # 19-21. Stochastic and Williams R
        features.append(rsi)  # stochastic_k approximation
        features.append(rsi * 0.95)  # stochastic_d approximation
        features.append(-rsi)  # williams_r approximation
        
        # 22-24. Moving averages
        features.append(safe_attr(technical_signals, 'ema_20', 0.0) * 0.9)  # ema_9
        features.append(safe_attr(technical_signals, 'ema_20', 0.0))
        features.append(safe_attr(technical_signals, 'ema_50', 0.0))
        
        # 25-28. Volatility features
        features.append(safe_get(candle_data, 'atr', 0.0))
        bb_upper = safe_attr(technical_signals, 'bb_upper', 100.0)
        bb_lower = safe_attr(technical_signals, 'bb_lower', 99.0)
        features.append(bb_upper - bb_lower)  # bb_width
        features.append(safe_get(candle_data, 'high_low_range_pct', 0.0))
        features.append(safe_attr(technical_signals, 'bb_position', 0.5))
        
        # 29-37. Volume features
        volume = safe_get(candle_data, ['volume', 'Volume'], 0.0)
        volume_ma = safe_get(candle_data, 'volume_ma', volume)
        features.append(volume)
        features.append(volume_ma)
        features.append(volume / (volume_ma + 1) if volume_ma > 0 else 1.0)  # volume_ratio
        features.append(0.0)  # obv
        features.append(50.0)  # mfi default
        features.append(0.0)  # volume_roc
        features.append(0.0)  # volume_spike
        features.append(safe_get(candle_data, 'trade_count', 0.0))
        features.append(safe_get(market_data, 'trade_intensity', 0.0))
        
        # 38-47. Perps-specific features
        funding_rate = safe_get(market_data, ['funding_rate', 'fundingRate', 'next_funding_rate'], 0.0)
        features.append(funding_rate)
        features.append(1.0 if funding_rate > 0 else 0.0)  # funding_positive
        features.append(safe_get(market_data, ['open_interest', 'openInterest'], 0.0))
        oi_momentum = safe_get(market_data, 'oi_momentum', 0.0)
        features.append(oi_momentum)
        features.append(1.0 if oi_momentum > 0 else 0.0)  # oi_increasing
        features.append(safe_get(market_data, 'oi_volatility', 0.0))
        features.append(safe_get(market_data, 'oi_trend_strength', 0.0))
        features.append(safe_get(market_data, 'price_oi_divergence', 0.0))
        features.append(safe_get(market_data, 'next_funding_rate', funding_rate))
        features.append(safe_get(market_data, 'funding_rate_percentile', 50.0))
        
        # 48-53. Market features (enriched with CryptoRank global context)
        token_price_change_24h = safe_get(market_data, ['price_change_24h', 'priceChange24H'], 0.0)
        features.append(token_price_change_24h)
        features.append(safe_get(market_data, ['volume_24h', 'volume24H'], 0.0))
        if market_context and not getattr(market_context, 'is_default', True):
            features.append(market_context.btc_dominance_change / 10.0)  # btc_correlation_24h
            features.append(token_price_change_24h - market_context.total_market_cap_change)  # relative_strength_vs_btc
            features.append(market_context.altcoin_index_change / 100.0)  # sector_momentum
        else:
            features.append(0.0)  # btc_correlation_24h
            features.append(0.0)  # relative_strength_vs_btc
            features.append(0.0)  # sector_momentum
        features.append(safe_get(market_data, 'liquidity_score', 50.0))
        
        # 54-56. Time features
        features.append(float(now.hour))
        features.append(float(now.weekday()))
        features.append(1.0 if now.weekday() >= 5 else 0.0)  # is_weekend
        
        # 57-59. Whale flow features (market-wide whale positioning from block-liquidity)
        features.append(safe_get(market_data, 'whale_flow_net', 0.0))
        features.append(safe_get(market_data, 'whale_flow_long_ratio', 0.5))
        features.append(safe_get(market_data, 'whale_flow_count_whales', 0.0))
        
        return np.array(features)
    
    def _rule_based_prediction(
        self,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats
    ) -> tuple:
        """
        Generate signal using rule-based logic (fallback)
        
        Uses weighted scoring system based on:
        - Whale behavior (40%)
        - Technical signals (30%)
        - Holder distribution (30%)
        """
        score = 0.0
        max_score = 100.0
        
        # === WHALE BEHAVIOR (40 points max) ===
        whale_score = 0
        
        # Net whale flow
        if whale_metrics.whale_state == "Accumulation":
            whale_score += 20
        elif whale_metrics.whale_state == "Distribution":
            whale_score -= 20
        
        # Accumulation vs distribution ratio
        total_pressure = (whale_metrics.whale_buy_volume + 
                         whale_metrics.whale_sell_volume)
        if total_pressure > 0:
            acc_ratio = whale_metrics.whale_buy_volume / total_pressure
            whale_score += (acc_ratio - 0.5) * 40  # -20 to +20
        
        score += min(40, max(-40, whale_score))
        
        # === TECHNICAL SIGNALS (30 points max) ===
        tech_score = 0
        
        if technical_signals.overall_signal == "bullish":
            tech_score += 20 * technical_signals.signal_strength
        elif technical_signals.overall_signal == "bearish":
            tech_score -= 20 * technical_signals.signal_strength
        
        # RSI consideration
        if technical_signals.rsi_signal == "oversold":
            tech_score += 10  # Buy opportunity
        elif technical_signals.rsi_signal == "overbought":
            tech_score -= 10  # Sell signal
        
        score += min(30, max(-30, tech_score))
        
        # === HOLDER DISTRIBUTION (30 points max) ===
        dist_score = 0
        
        # Gini coefficient (lower is better for long-term health)
        if whale_metrics.gini_coefficient < whale_config.GINI_SAFE:
            dist_score += 15
        elif whale_metrics.gini_coefficient > whale_config.GINI_HIGH_RISK:
            dist_score -= 15
        
        # Top 10 concentration (lower is better)
        if whale_metrics.top10_hold_percent < wallet_classification_config.SAFETY_EXCELLENT_CONCENTRATION_PCT:
            dist_score += 10
        elif whale_metrics.top10_hold_percent > wallet_classification_config.SAFETY_MEDIUM_CONCENTRATION_PCT:
            dist_score -= 10
        
        # Dev holding (lower is better)
        if whale_metrics.dev_hold_percent < 5:
            dist_score += 5
        elif whale_metrics.dev_hold_percent > 15:
            dist_score -= 10
        
        score += min(30, max(-30, dist_score))
        
        # === CONVERT TO SIGNAL ===
        # Score range: -100 to +100
        normalized = (score + 100) / 200  # Convert to 0-1
        
        if normalized >= 0.65:
            signal = "BUY"
            confidence = normalized
        elif normalized <= 0.35:
            signal = "SELL"
            confidence = 1 - normalized
        else:
            signal = "HOLD"
            confidence = 1 - abs(normalized - 0.5) * 2
        
        return signal, float(confidence)
    
    def _build_feature_vector(
        self,
        whale_metrics: WhaleMetrics,
        technical_signals: TechnicalSignals,
        holder_stats: HolderStats,
        candle_data: Optional[Dict] = None,
        market_context = None
    ) -> np.ndarray:
        """
        Build feature vector for ML model
        
        Must match exactly these 36 features in order:
        ['BuyCount', 'SellCount', 'TotalSupply', 'SolUSDPrice', 'returns', 
         'log_returns', 'rsi_14', 'ema_20', 'ema_50', 'ema_20_50_cross', 
         'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'macd_line', 
         'macd_signal', 'macd_histogram', 'volume_ma', 'volume_ratio', 
         'buy_sell_ratio', 'net_volume', 'net_volume_ma', 'buy_pressure', 
         'price_momentum_5', 'price_momentum_10', 'volatility', 'high_low_range', 
         'gini_coefficient', 'whale_buy_volume', 'whale_sell_volume', 
         'top_10_concentration', 'holder_count', 'sol_correlation', 
         'market_favorability', 'whale_state_encoded', 'market_regime_encoded']
        """
        # Default values for candle data if not provided
        if candle_data is None:
            candle_data = {}
        
        features = []
        
        # 1. BuyCount - from candle data
        features.append(float(candle_data.get('BuyCount', 0)))
        
        # 2. SellCount - from candle data
        features.append(float(candle_data.get('SellCount', 0)))
        
        # 3. TotalSupply - from candle data
        features.append(float(candle_data.get('TotalSupply', 1_000_000_000)))
        
        # 4. SolUSDPrice - from candle data
        features.append(float(candle_data.get('SolUSDPrice', 150.0)))
        
        # 5. returns - price return
        features.append(float(candle_data.get('returns', technical_signals.price_momentum_5)))
        
        # 6. log_returns - log of returns
        returns = candle_data.get('returns', technical_signals.price_momentum_5)
        log_returns = np.log(1 + returns) if returns > -1 else 0
        features.append(float(log_returns))
        
        # 7. rsi_14
        features.append(float(technical_signals.rsi))
        
        # 8. ema_20
        features.append(float(technical_signals.ema_20))
        
        # 9. ema_50
        features.append(float(technical_signals.ema_50))
        
        # 10. ema_20_50_cross (1 if ema_20 > ema_50, 0 otherwise)
        ema_cross = 1 if technical_signals.ema_20 > technical_signals.ema_50 else 0
        features.append(float(ema_cross))
        
        # 11. bb_upper
        features.append(float(technical_signals.bb_upper))
        
        # 12. bb_middle (average of upper and lower)
        bb_middle = (technical_signals.bb_upper + technical_signals.bb_lower) / 2
        features.append(float(bb_middle))
        
        # 13. bb_lower
        features.append(float(technical_signals.bb_lower))
        
        # 14. bb_position
        features.append(float(technical_signals.bb_position))
        
        # 15. macd_line
        features.append(float(technical_signals.macd_line))
        
        # 16. macd_signal
        features.append(float(technical_signals.macd_signal))
        
        # 17. macd_histogram
        features.append(float(technical_signals.macd_histogram))
        
        # 18. volume_ma - from candle data or estimate
        features.append(float(candle_data.get('volume_ma', 1000000)))
        
        # 19. volume_ratio
        features.append(float(technical_signals.volume_ratio))
        
        # 20. buy_sell_ratio - from candle data or use buy_pressure
        buy_sell_ratio = candle_data.get('buy_sell_ratio', 
                                          technical_signals.buy_pressure / (1 - technical_signals.buy_pressure + 1e-10))
        features.append(float(buy_sell_ratio))
        
        # 21. net_volume - from candle data
        features.append(float(candle_data.get('net_volume', 0)))
        
        # 22. net_volume_ma - from candle data
        features.append(float(candle_data.get('net_volume_ma', 0)))
        
        # 23. buy_pressure
        features.append(float(technical_signals.buy_pressure))
        
        # 24. price_momentum_5
        features.append(float(technical_signals.price_momentum_5))
        
        # 25. price_momentum_10
        features.append(float(technical_signals.price_momentum_10))
        
        # 26. volatility
        features.append(float(technical_signals.volatility))
        
        # 27. high_low_range - from candle data or estimate from volatility
        high_low_range = candle_data.get('high_low_range', technical_signals.volatility * 2)
        features.append(float(high_low_range))
        
        # 28. gini_coefficient
        features.append(float(whale_metrics.gini_coefficient))
        
        # 29. whale_buy_volume
        features.append(float(whale_metrics.whale_buy_volume))
        
        # 30. whale_sell_volume
        features.append(float(whale_metrics.whale_sell_volume))
        
        # 31. top_10_concentration
        features.append(float(whale_metrics.top10_hold_percent))
        
        # 32. holder_count
        features.append(float(holder_stats.active_holders))
        
        # 33. sol_correlation (enriched with CryptoRank: inverse BTC dominance = altcoin favorability)
        if market_context and not getattr(market_context, 'is_default', True):
            features.append(float(1.0 - (market_context.btc_dominance / 100.0)))
        else:
            features.append(float(candle_data.get('sol_correlation', 0.0)))
        
        # 34. market_favorability (enriched with CryptoRank: Fear & Greed normalized to 0-1)
        if market_context and not getattr(market_context, 'is_default', True):
            features.append(float(market_context.market_favorability))
        else:
            features.append(float(candle_data.get('market_favorability', 0.5)))
        
        # 35. whale_state_encoded
        whale_state_map = {"Accumulation": 1, "Stability": 0, "Distribution": -1}
        features.append(float(whale_state_map.get(whale_metrics.whale_state, 0)))
        
        # 36. market_regime_encoded (enriched with CryptoRank: derived from F&G level + trend)
        if market_context and not getattr(market_context, 'is_default', True):
            regime_map = {"BULL": 1, "SIDEWAYS": 0, "BEAR": -1}
            features.append(float(regime_map.get(market_context.market_regime, 0)))
        else:
            market_regime = candle_data.get('market_regime', 'SIDEWAYS')
            regime_map = {"BULL": 1, "SIDEWAYS": 0, "BEAR": -1}
            features.append(float(regime_map.get(market_regime, 0)))
        
        return np.array(features)
    
    def _assess_risk(
        self,
        whale_metrics: WhaleMetrics,
        holder_stats: HolderStats
    ) -> str:
        """Assess overall risk level"""
        risk_score = 0
        
        # High Gini = high risk
        if whale_metrics.gini_coefficient > 0.6:
            risk_score += 2
        elif whale_metrics.gini_coefficient > 0.5:
            risk_score += 1
        
        # High concentration = high risk
        if whale_metrics.top10_hold_percent > wallet_classification_config.TOP10_CONCENTRATION_HIGH_RISK * 100:
            risk_score += 2
        elif whale_metrics.top10_hold_percent > wallet_classification_config.TOP10_CONCENTRATION_MEDIUM_RISK * 100:
            risk_score += 1
        
        # High dev holding = high risk
        if whale_metrics.dev_hold_percent > wallet_classification_config.DEV_HOLDING_HIGH_RISK * 100:
            risk_score += 2
        elif whale_metrics.dev_hold_percent > wallet_classification_config.DEV_HOLDING_HIGH_RISK * 50:  # 5% = half of high risk
            risk_score += 1
        
        # Distribution state = higher risk for longs
        if whale_metrics.whale_state == "Distribution":
            risk_score += 1
        
        # Low holder count = higher risk
        if holder_stats.active_holders < wallet_classification_config.SAFETY_MEDIUM_HOLDERS:
            risk_score += 1
        
        # Classify risk
        if risk_score >= 5:
            return "High"
        elif risk_score >= 3:
            return "Medium"
        else:
            return "Low"


def convert_signal_to_dict(signal: TradingSignal) -> Dict[str, Any]:
    """Convert TradingSignal to dictionary (for JSON output)"""
    result = {
        "signal": signal.signal,
        "confidence": round(signal.confidence, 2),
        "whale_buy_volume": round(signal.whale_buy_volume, 0),
        "whale_sell_volume": round(signal.whale_sell_volume, 0),
        "whale_net_volume": round(signal.whale_net_volume, 0),
        "gini_coefficient": round(signal.gini_coefficient, 2),
        "top10_hold_percent": round(signal.top10_hold_percent, 1),
        "dev_hold_percent": round(signal.dev_hold_percent, 1),
        "sniper_hold_percent": round(signal.sniper_hold_percent, 1),
        "phase": signal.phase,
        "whale_state": signal.whale_state,
        "summary": signal.summary,
        "technical_signal": signal.technical_signal,
        "risk_level": signal.risk_level,
        "timestamp": signal.timestamp
    }
    
    # Add Layer 2 fields if present
    if signal.user_risk_assessment:
        result["user_risk_assessment"] = signal.user_risk_assessment
    
    if signal.layer_breakdown:
        result["layer_breakdown"] = signal.layer_breakdown
        result["layers_used"] = signal.layers_used
        result["agreement_level"] = round(signal.agreement_level, 4)
    
    # Add timing field if present (Feature 2)
    if signal.timing:
        result["timing"] = signal.timing
    
    # Add signal effectiveness window fields
    if signal.signal_valid_from:
        result["signal_effectiveness"] = {
            "effectiveness_hours": signal.signal_effectiveness_hours,
            "signal_valid_from": signal.signal_valid_from,
            "signal_valid_until": signal.signal_valid_until
        }
    
    # Add recent 24h analysis if present
    if signal.recent_analysis:
        result["recent_analysis"] = signal.recent_analysis
    
    # Add market context if present
    if signal.market_context:
        result["market_context"] = signal.market_context
    
    return result


if __name__ == "__main__":
    print("Signal Generator")
    print("=" * 60)
    
    # Create mock data for testing
    from engines.whale_engine import WhaleMetrics
    from engines.technical_engine import TechnicalSignals
    from engines.holder_metrics import HolderStats
    
    whale_metrics = WhaleMetrics(
        whale_buy_volume=500000,
        whale_sell_volume=100000,
        whale_net_volume=400000,
        gini_coefficient=0.45,
        top10_hold_percent=25.0,
        dev_hold_percent=3.0,
        sniper_hold_percent=2.0,
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
        gini_coefficient=0.45,
        top10_concentration=25.0,
        top20_concentration=35.0,
        top50_concentration=55.0,
        median_holding=10000.0,
        mean_holding=50000.0,
        std_holding=100000.0,
        holder_score=75.0
    )
    
    generator = SignalGenerator()
    signal = generator.generate_signal(
        whale_metrics=whale_metrics,
        technical_signals=tech_signals,
        holder_stats=holder_stats,
        summary="Test summary"
    )
    
    print(f"\nGenerated Signal:")
    print(f"  Signal: {signal.signal}")
    print(f"  Confidence: {signal.confidence:.2f}")
    print(f"  Risk Level: {signal.risk_level}")
    print(f"  Whale State: {signal.whale_state}")
    print(f"  Net Flow: {signal.whale_net_volume:,.0f}")

