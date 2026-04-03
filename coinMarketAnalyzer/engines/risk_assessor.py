"""
Risk Assessor Engine
Generates risk ratings by matching current trade context against user's historical patterns
Part of Layer 2: Pre-Execution Risk & Edge Assessment
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import user_profile_config
from engines.user_profiler import (
    UserProfiler, UserProfile, TradePattern, PatternType
)

# Setup logging
logger = logging.getLogger(__name__)


# ==================== ENUMS ====================

class RiskRating(str, Enum):
    """Risk rating levels"""
    GREEN = "GREEN"    # Favorable - matches winning patterns
    YELLOW = "YELLOW"  # Neutral - insufficient data or mixed signals
    RED = "RED"        # Unfavorable - matches losing patterns


# ==================== DATA CLASSES ====================

@dataclass
class RiskAssessment:
    """Complete risk assessment result"""
    rating: RiskRating
    confidence: float  # 0-1, how confident in rating
    
    # Signal for aggregation
    signal: str  # "BUY", "SELL", "HOLD"
    signal_weight: float  # -1 to +1 for layer aggregation
    
    # Pattern matching
    matching_pattern: Optional[TradePattern] = None
    pattern_match_score: float = 0.0
    
    # Component scores
    pattern_score: float = 0.0
    market_context_score: float = 0.5
    historical_score: float = 0.5
    
    # User-facing output
    message: str = ""
    risk_factors: List[str] = field(default_factory=list)
    
    # Profile reference
    profile_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Same-token history (user has traded this exact token before)
    same_token_history: Optional[Dict[str, Any]] = None
    
    # Structured similar-pattern summary for client rendering
    similar_pattern_summary: Optional[Dict[str, Any]] = None


@dataclass
class TradeContext:
    """Current market context for pattern matching"""
    # Token metrics
    phase: str = "UNKNOWN"  # P1, P2, P3, P4
    top10_concentration: float = 50.0
    gini_coefficient: float = 0.5
    holder_count: int = 100
    
    # Whale metrics
    whale_state: str = "Stability"  # Accumulation, Distribution, Stability
    whale_net_volume: float = 0.0
    
    # Technical signals
    rsi: float = 50.0
    rsi_signal: str = "neutral"  # oversold, neutral, overbought
    trend: str = "neutral"  # bullish, neutral, bearish
    
    # Price context
    price_near_peak: bool = False
    price_near_bottom: bool = False
    volatility: float = 0.0


# ==================== RISK ASSESSOR CLASS ====================

class RiskAssessor:
    """
    Generates risk ratings by matching current trade context
    against user's historical patterns
    """
    
    def __init__(self, user_profiler: UserProfiler = None):
        """
        Initialize Risk Assessor
        
        Args:
            user_profiler: UserProfiler instance for getting user profiles
        """
        self.user_profiler = user_profiler
        self.config = user_profile_config
        
        logger.info("Risk Assessor initialized")
    
    def set_user_profiler(self, user_profiler: UserProfiler):
        """Set the user profiler (for lazy initialization)"""
        self.user_profiler = user_profiler
    
    def assess_trade(
        self,
        user_address: str,
        target_mint: str,
        whale_metrics: Any = None,
        technical_signals: Any = None,
        holder_stats: Any = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> RiskAssessment:
        """
        Generate risk assessment for proposed trade
        
        Args:
            user_address: User's wallet address
            target_mint: Token being considered for trade
            whale_metrics: WhaleMetrics object (optional)
            technical_signals: TechnicalSignals object (optional)
            holder_stats: HolderStats object (optional)
            from_date: Optional start date for user trades (ISO format, defaults to 90 days)
            to_date: Optional end date for user trades (ISO format, defaults to now)
            
        Returns:
            RiskAssessment with rating, confidence, and details
        """
        logger.info(f"Assessing trade risk for {user_address[:8]}... on {target_mint[:8]}...")
        
        # Build current trade context
        context = self._build_context(whale_metrics, technical_signals, holder_stats)
        
        # Get user profile
        if self.user_profiler is None:
            logger.warning("UserProfiler not set, returning neutral assessment")
            return self._create_neutral_assessment("UserProfiler not configured")
        
        profile = self.user_profiler.get_profile(user_address, use_birdeye_fallback=True, from_date=from_date, to_date=to_date)
        
        # Check if profile is valid
        if not profile or not profile.is_valid:
            return self._create_neutral_assessment(
                self.config.MESSAGE_NEW_USER,
                profile_summary={'total_trades': profile.total_trades if profile else 0}
            )
        
        if profile.total_trades < self.config.MIN_TRADES_FOR_PROFILE:
            return self._create_neutral_assessment(
                self.config.MESSAGE_NEW_USER,
                profile_summary={
                    'total_trades': profile.total_trades,
                    'win_rate': profile.overall_win_rate
                }
            )
        
        # Find matching pattern
        best_match, match_score = self._find_matching_pattern(profile.patterns, context)
        
        # Calculate component scores
        pattern_score = self._calc_pattern_score(best_match, match_score)
        market_score = self._calc_market_context_score(context, profile)
        history_score = self._calc_historical_score(profile)
        
        # Weighted combination
        combined_score = (
            pattern_score * self.config.PHASE_MATCH_WEIGHT * 2 +  # Pattern most important
            market_score * self.config.CONCENTRATION_MATCH_WEIGHT * 2 +
            history_score * (1 - self.config.PHASE_MATCH_WEIGHT * 2 - self.config.CONCENTRATION_MATCH_WEIGHT * 2)
        )
        
        # Determine rating
        rating, message, risk_factors = self._determine_rating(
            combined_score,
            best_match,
            profile,
            context
        )
        
        # Calculate signal weight for layer aggregation
        signal, signal_weight = self._calc_signal_weight(rating, combined_score, match_score)
        
        # Build profile summary for response (includes new API-provided fields)
        profile_summary = {
            'total_trades': profile.total_trades,
            'win_rate': round(profile.overall_win_rate, 4),
            'trader_type': profile.trader_type.value if profile.trader_type else 'UNKNOWN',
            'avg_hold_time_hours': round(profile.avg_hold_time_hours, 2),
            'profile_confidence': round(profile.confidence, 4),
            # New API-provided fields
            'lifetime_pnl': round(profile.lifetime_pnl, 2),
            'total_volume_usd': round(profile.total_volume_usd, 2),
            'max_drawdown_pct': round(profile.max_drawdown_pct, 4),
            'avg_r_multiple': round(profile.avg_r_multiple, 4),
            'avg_holding_time_minutes': round(profile.avg_holding_time_minutes, 2)
        }
        
        same_token_history = self._build_same_token_history(profile, target_mint)
        similar_pattern_summary = self._build_similar_pattern_summary(best_match, match_score)
        
        assessment = RiskAssessment(
            rating=rating,
            confidence=match_score if best_match else profile.confidence,
            signal=signal,
            signal_weight=signal_weight,
            matching_pattern=best_match,
            pattern_match_score=match_score,
            pattern_score=pattern_score,
            market_context_score=market_score,
            historical_score=history_score,
            message=message,
            risk_factors=risk_factors,
            profile_summary=profile_summary,
            same_token_history=same_token_history,
            similar_pattern_summary=similar_pattern_summary,
        )
        
        logger.info(f"Risk assessment: {rating.value} ({signal}), confidence: {assessment.confidence:.2f}")
        
        return assessment
    
    def quick_assess(
        self,
        user_address: str,
        phase: str = "P4",
        top10_concentration: float = 50.0,
        whale_state: str = "Stability"
    ) -> RiskAssessment:
        """
        Quick assessment with minimal context (for API use)
        
        Args:
            user_address: User's wallet address
            phase: Token phase (P1-P4)
            top10_concentration: Top 10 holder concentration %
            whale_state: Current whale state
            
        Returns:
            RiskAssessment
        """
        context = TradeContext(
            phase=phase,
            top10_concentration=top10_concentration,
            whale_state=whale_state
        )
        
        # Get profile and assess
        if self.user_profiler is None:
            return self._create_neutral_assessment("UserProfiler not configured")
        
        profile = self.user_profiler.get_profile(user_address, use_birdeye_fallback=True)
        
        if not profile or not profile.is_valid:
            return self._create_neutral_assessment(self.config.MESSAGE_NEW_USER)
        
        # Simplified assessment
        best_match, match_score = self._find_matching_pattern(profile.patterns, context)
        
        if best_match:
            if best_match.win_rate >= self.config.WIN_RATE_GREEN_THRESHOLD:
                rating = RiskRating.GREEN
                signal = "BUY"
            elif best_match.win_rate <= self.config.WIN_RATE_RED_THRESHOLD:
                rating = RiskRating.RED
                signal = "SELL"
            else:
                rating = RiskRating.YELLOW
                signal = "HOLD"
        else:
            rating = RiskRating.YELLOW
            signal = "HOLD"
        
        signal_weight = self._signal_to_weight(signal, match_score)
        
        return RiskAssessment(
            rating=rating,
            confidence=match_score,
            signal=signal,
            signal_weight=signal_weight,
            matching_pattern=best_match,
            pattern_match_score=match_score,
            message=self._generate_message(rating, best_match, profile),
            profile_summary={'total_trades': profile.total_trades}
        )
    
    # ==================== PRIVATE METHODS ====================
    
    def _build_same_token_history(
        self,
        profile: UserProfile,
        target_mint: str
    ) -> Optional[Dict[str, Any]]:
        """Look up the user's past PnL on *this exact token* from their profile."""
        if not target_mint or not profile.mint_pnls:
            return {"traded_before": False}
        
        target_lower = target_mint.lower()
        matching = [p for p in profile.mint_pnls if p.mint.lower() == target_lower]
        if not matching:
            return {"traded_before": False}
        
        pnl = matching[0]
        return {
            "traded_before": True,
            "trades_count": pnl.total_trades,
            "realized_pnl": round(pnl.realized_pnl, 4),
            "pnl_percent": round(pnl.pnl_percent, 4),
            "is_winner": pnl.is_winner,
            "avg_hold_hours": round(pnl.avg_hold_time_hours, 2),
            "entry_phase": pnl.entry_phase,
        }
    
    @staticmethod
    def _build_similar_pattern_summary(
        matching_pattern: Optional[TradePattern],
        match_score: float
    ) -> Optional[Dict[str, Any]]:
        """Build a structured summary of the best-matching past pattern."""
        if matching_pattern is None:
            return {
                "has_match": False,
                "reason": "insufficient_history",
            }
        
        win_rate = matching_pattern.win_rate
        occurrences = matching_pattern.occurrences
        pattern_type = matching_pattern.pattern_type.value
        one_liner = (
            f"In {occurrences} similar past trades "
            f"({pattern_type.replace('_', ' ').lower()}), "
            f"your win rate was {win_rate:.0%}."
        )
        
        return {
            "has_match": True,
            "pattern_type": pattern_type,
            "win_rate": round(win_rate, 4),
            "occurrences": occurrences,
            "match_score": round(match_score, 4),
            "avg_pnl_percent": round(matching_pattern.avg_pnl_percent, 4),
            "one_liner": one_liner,
        }
    
    def _build_context(
        self,
        whale_metrics: Any = None,
        technical_signals: Any = None,
        holder_stats: Any = None
    ) -> TradeContext:
        """Build TradeContext from analysis objects"""
        context = TradeContext()
        
        # Extract from whale_metrics
        if whale_metrics:
            context.phase = getattr(whale_metrics, 'phase', 'P4')
            context.top10_concentration = getattr(whale_metrics, 'top10_hold_percent', 50.0)
            context.gini_coefficient = getattr(whale_metrics, 'gini_coefficient', 0.5)
            context.whale_state = getattr(whale_metrics, 'whale_state', 'Stability')
            context.whale_net_volume = getattr(whale_metrics, 'whale_net_volume', 0.0)
        
        # Extract from technical_signals
        if technical_signals:
            context.rsi = getattr(technical_signals, 'rsi', 50.0)
            context.rsi_signal = getattr(technical_signals, 'rsi_signal', 'neutral')
            context.trend = getattr(technical_signals, 'overall_signal', 'neutral')
            context.volatility = getattr(technical_signals, 'volatility', 0.0)
            
            # Determine if near price extremes
            if context.rsi > 70:
                context.price_near_peak = True
            elif context.rsi < 30:
                context.price_near_bottom = True
        
        # Extract from holder_stats
        if holder_stats:
            context.holder_count = getattr(holder_stats, 'active_holders', 100)
        
        return context
    
    def _find_matching_pattern(
        self,
        patterns: List[TradePattern],
        context: TradeContext
    ) -> Tuple[Optional[TradePattern], float]:
        """
        Find the best matching pattern from user's history
        
        Returns:
            (best_pattern, similarity_score)
        """
        if not patterns:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for pattern in patterns:
            score = self._calc_pattern_similarity(pattern, context)
            
            if score > best_score:
                best_score = score
                best_match = pattern
        
        # Only return match if above threshold
        if best_score >= self.config.PATTERN_SIMILARITY_THRESHOLD:
            return best_match, best_score
        
        return None, best_score
    
    def _calc_pattern_similarity(self, pattern: TradePattern, context: TradeContext) -> float:
        """
        Calculate similarity between pattern's typical context and current context
        
        Returns value between 0 and 1
        """
        similarity = 0.0
        typical = pattern.typical_context
        
        # Phase match (0-0.3 points)
        if typical.get('typical_phase') == context.phase:
            similarity += self.config.PHASE_MATCH_WEIGHT
        
        # Concentration match (0-0.25 points)
        typical_conc = typical.get('concentration', 50.0)
        conc_diff = abs(typical_conc - context.top10_concentration)
        similarity += max(0, self.config.CONCENTRATION_MATCH_WEIGHT - (conc_diff / 100))
        
        # Whale state match (0-0.2 points)
        if typical.get('whale_state') == context.whale_state:
            similarity += self.config.WHALE_STATE_MATCH_WEIGHT
        
        # Trend match (0-0.15 points)
        if typical.get('trend') == context.trend:
            similarity += self.config.TREND_MATCH_WEIGHT
        
        # RSI zone match (0-0.1 points)
        if typical.get('rsi_signal') == context.rsi_signal:
            similarity += self.config.RSI_MATCH_WEIGHT
        
        return min(1.0, similarity)
    
    def _calc_pattern_score(self, pattern: Optional[TradePattern], match_score: float) -> float:
        """Calculate score based on matching pattern's win rate"""
        if not pattern:
            return 0.5  # Neutral
        
        # Weight pattern's win rate by match confidence
        return pattern.win_rate * match_score + 0.5 * (1 - match_score)
    
    def _calc_market_context_score(self, context: TradeContext, profile: UserProfile) -> float:
        """
        Calculate score based on how favorable current context is for this user
        
        Returns value between 0 and 1
        """
        score = 0.5  # Start neutral
        
        # Check for FOMO risk
        if context.price_near_peak and (profile.fomo_tendency or 0) > 0.5:
            score -= 0.2  # User tends to lose when buying peaks
        
        # Check for panic sell tendency
        if context.price_near_bottom and (profile.panic_sell_tendency or 0) > 0.5:
            score += 0.1  # Don't sell at bottom
        
        # Early entry user + early phase = good
        if profile.typical_entry_phase in ["P1", "P2"] and context.phase in ["P1", "P2"]:
            score += 0.15
        
        # Flipper in volatile market = risky
        if profile.is_flipper and context.volatility > 0.1:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def _calc_historical_score(self, profile: UserProfile) -> float:
        """Calculate score based on user's overall track record"""
        # Base on overall win rate
        return profile.overall_win_rate
    
    def _determine_rating(
        self,
        combined_score: float,
        pattern: Optional[TradePattern],
        profile: UserProfile,
        context: TradeContext
    ) -> Tuple[RiskRating, str, List[str]]:
        """Determine rating, message, and risk factors"""
        risk_factors = []
        
        # Check for RED conditions
        if pattern and pattern.win_rate <= self.config.WIN_RATE_RED_THRESHOLD:
            if pattern.occurrences >= self.config.MIN_TRADES_FOR_PATTERN * 2:
                losses = pattern.losses
                total = pattern.occurrences
                message = self.config.MESSAGE_RED.format(
                    losses=losses,
                    total=total
                )
                risk_factors.append(f"LOW_WIN_PATTERN: {pattern.win_rate:.0%}")
                return RiskRating.RED, message, risk_factors
        
        # FOMO check
        if context.price_near_peak and (profile.fomo_tendency or 0) > self.config.FOMO_DETECTION_THRESHOLD:
            risk_factors.append("FOMO_RISK")
            message = "🔴 Price near local peak. Your history shows losses when buying at peaks."
            return RiskRating.RED, message, risk_factors
        
        # HIGH_CONCENTRATION check
        if context.top10_concentration > 80 and pattern:
            # Check if user historically loses in high concentration
            if pattern.pattern_type == PatternType.HIGH_CONCENTRATION and pattern.win_rate < 0.4:
                risk_factors.append("HIGH_CONCENTRATION_RISK")
                message = "🔴 High token concentration. Your history shows poor results in similar conditions."
                return RiskRating.RED, message, risk_factors
        
        # Check for GREEN conditions
        if pattern and pattern.win_rate >= self.config.WIN_RATE_GREEN_THRESHOLD:
            message = self.config.MESSAGE_GREEN.format(
                win_rate=pattern.win_rate,
                occurrences=pattern.occurrences
            )
            return RiskRating.GREEN, message, risk_factors
        
        # Default to YELLOW
        if not pattern:
            message = self.config.MESSAGE_NO_PATTERN
        else:
            message = self.config.MESSAGE_YELLOW
        
        return RiskRating.YELLOW, message, risk_factors
    
    def _calc_signal_weight(
        self,
        rating: RiskRating,
        score: float,
        match_confidence: float
    ) -> Tuple[str, float]:
        """
        Calculate signal and weight for layer aggregation
        
        Returns:
            (signal, weight) where weight is -1 to +1
        """
        # Determine signal
        if rating == RiskRating.GREEN:
            signal = "BUY"
            base_weight = 0.6
        elif rating == RiskRating.RED:
            signal = "SELL"
            base_weight = -0.6
        else:
            signal = "HOLD"
            base_weight = 0.0
        
        # Scale by confidence
        confidence_factor = min(1.0, match_confidence * 1.5)
        weight = base_weight * confidence_factor
        
        return signal, weight
    
    def _signal_to_weight(self, signal: str, confidence: float) -> float:
        """Convert signal to layer weight"""
        if signal == "BUY":
            return 0.6 * confidence
        elif signal == "SELL":
            return -0.6 * confidence
        return 0.0
    
    def _generate_message(
        self,
        rating: RiskRating,
        pattern: Optional[TradePattern],
        profile: UserProfile
    ) -> str:
        """Generate user-facing message"""
        if rating == RiskRating.GREEN and pattern:
            return self.config.MESSAGE_GREEN.format(
                win_rate=pattern.win_rate,
                occurrences=pattern.occurrences
            )
        elif rating == RiskRating.RED and pattern:
            return self.config.MESSAGE_RED.format(
                losses=pattern.losses,
                total=pattern.occurrences
            )
        else:
            return self.config.MESSAGE_YELLOW
    
    def _create_neutral_assessment(
        self,
        message: str,
        profile_summary: Dict[str, Any] = None
    ) -> RiskAssessment:
        """Create a neutral/default assessment"""
        return RiskAssessment(
            rating=RiskRating.YELLOW,
            confidence=0.3,
            signal="HOLD",
            signal_weight=0.0,
            message=message,
            risk_factors=["INSUFFICIENT_DATA"],
            profile_summary=profile_summary or {}
        )


# ==================== HELPER FUNCTIONS ====================

def assessment_to_dict(assessment: RiskAssessment) -> Dict[str, Any]:
    """Convert RiskAssessment to dictionary for JSON serialization"""
    return {
        'rating': assessment.rating.value,
        'confidence': round(assessment.confidence, 4),
        'signal': assessment.signal,
        'signal_weight': round(assessment.signal_weight, 4),
        'matching_pattern': {
            'pattern_type': assessment.matching_pattern.pattern_type.value,
            'win_rate': round(assessment.matching_pattern.win_rate, 4),
            'occurrences': assessment.matching_pattern.occurrences
        } if assessment.matching_pattern else None,
        'pattern_match_score': round(assessment.pattern_match_score, 4),
        'scores': {
            'pattern': round(assessment.pattern_score, 4),
            'market_context': round(assessment.market_context_score, 4),
            'historical': round(assessment.historical_score, 4)
        },
        'message': assessment.message,
        'risk_factors': assessment.risk_factors,
        'profile_summary': assessment.profile_summary,
        'same_token_history': assessment.same_token_history,
        'similar_pattern_summary': assessment.similar_pattern_summary,
    }


if __name__ == "__main__":
    print("Risk Assessor Engine")
    print("=" * 60)
    
    # Test with mock data
    assessor = RiskAssessor()
    
    # Create a mock assessment for testing
    test_assessment = RiskAssessment(
        rating=RiskRating.GREEN,
        confidence=0.75,
        signal="BUY",
        signal_weight=0.45,
        pattern_score=0.72,
        market_context_score=0.65,
        historical_score=0.68,
        message="🟢 This aligns with your most profitable entry conditions (72% win rate over 15 trades).",
        risk_factors=[],
        profile_summary={
            'total_trades': 45,
            'win_rate': 0.65,
            'trader_type': 'FLIPPER'
        }
    )
    
    print(f"\nTest Assessment:")
    assessment_dict = assessment_to_dict(test_assessment)
    for key, value in assessment_dict.items():
        print(f"  {key}: {value}")
    
    print("\nRisk Assessor initialized successfully!")

