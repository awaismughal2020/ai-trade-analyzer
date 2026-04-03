"""
Post-Trade Reviewer Engine (Feature 4)
Analyzes completed trades to identify mistakes and provide actionable coaching

Mistake Types Detected:
- EARLY_EXIT: Sold too early, price continued higher
- LATE_ENTRY: Bought at higher price than available in window
- FOMO_ENTRY: Bought after significant pump
- PANIC_SELL: Sold at loss with short hold time
- REVENGE_TRADE: Trade made shortly after a loss
- OVER_TRADING: Multiple trades on same token in short window
- BAD_RISK_REWARD: Poor risk-reward ratio compared to user's typical trades
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import post_trade_review_config, user_profile_config

# Setup logging
logger = logging.getLogger(__name__)


# ==================== ENUMS ====================

class MistakeType(str, Enum):
    """Types of trading mistakes that can be identified"""
    EARLY_EXIT = "EARLY_EXIT"
    LATE_ENTRY = "LATE_ENTRY"
    FOMO_ENTRY = "FOMO_ENTRY"
    PANIC_SELL = "PANIC_SELL"
    REVENGE_TRADE = "REVENGE_TRADE"
    OVER_TRADING = "OVER_TRADING"
    BAD_RISK_REWARD = "BAD_RISK_REWARD"


class Severity(str, Enum):
    """Severity levels for identified mistakes"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ==================== DATA CLASSES ====================

@dataclass
class TradeInput:
    """Input trade data for analysis"""
    trade_id: str
    symbol: str
    mint: str
    qty: float
    price: float
    executed_at: datetime
    realized_pnl: float
    is_buy: bool
    creation_timestamp: Optional[int] = None
    user_address: Optional[str] = None


@dataclass
class PricePoint:
    """A single price point in time"""
    timestamp: datetime
    price: float
    volume: Optional[float] = None


@dataclass
class OutcomeComparison:
    """Comparison of expected vs actual trade outcome"""
    price_at_trade: float
    price_after_1h: Optional[float] = None
    price_after_4h: Optional[float] = None
    price_after_24h: Optional[float] = None
    max_price_after_exit: Optional[float] = None
    min_price_after_exit: Optional[float] = None
    max_price_before_entry: Optional[float] = None
    min_price_before_entry: Optional[float] = None
    optimal_exit_price: Optional[float] = None
    optimal_entry_price: Optional[float] = None
    time_to_max: Optional[str] = None
    time_to_min: Optional[str] = None
    missed_gain_pct: float = 0.0
    missed_dip_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'price_at_trade': self.price_at_trade,
            'price_after_1h': self.price_after_1h,
            'price_after_4h': self.price_after_4h,
            'price_after_24h': self.price_after_24h,
            'max_price_after_exit': self.max_price_after_exit,
            'min_price_after_exit': self.min_price_after_exit,
            'max_price_before_entry': self.max_price_before_entry,
            'min_price_before_entry': self.min_price_before_entry,
            'optimal_exit_price': self.optimal_exit_price,
            'optimal_entry_price': self.optimal_entry_price,
            'time_to_max': self.time_to_max,
            'time_to_min': self.time_to_min,
            'missed_gain_pct': round(self.missed_gain_pct, 2),
            'missed_dip_pct': round(self.missed_dip_pct, 2)
        }


@dataclass
class Mistake:
    """An identified trading mistake"""
    type: MistakeType
    severity: Severity
    impact_pct: float
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'type': self.type.value,
            'severity': self.severity.value,
            'impact_pct': round(self.impact_pct, 2),
            'description': self.description,
            'evidence': self.evidence
        }


@dataclass
class UserContext:
    """User's historical trading context for comparison"""
    total_trades: int = 0
    win_rate: float = 0.0
    avg_hold_time_hours: float = 0.0
    avg_pnl_per_trade: float = 0.0
    typical_entry_phase: str = "UNKNOWN"
    fomo_tendency: Optional[float] = None  # None when not computed (e.g. enough trades but all 0)
    panic_sell_tendency: Optional[float] = None
    trader_type: str = "UNKNOWN"
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    recent_losses: List[Dict[str, Any]] = field(default_factory=list)
    trades_on_same_mint: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PostTradeAnalysis:
    """Complete post-trade analysis result"""
    trade_id: str
    trade_type: str  # BUY or SELL
    symbol: str
    mint: str
    
    # Outcome Analysis
    outcome: OutcomeComparison
    
    # Mistakes Identified
    mistakes: List[Mistake] = field(default_factory=list)
    primary_mistake: Optional[MistakeType] = None
    
    # Impact Estimation
    missed_opportunity_pct: float = 0.0
    monthly_impact_estimate_pct: float = 0.0
    monthly_impact_estimate_usd: float = 0.0
    
    # AI Coaching (filled by OpenAI service)
    coaching_message: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_context: Optional[UserContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'trade_id': self.trade_id,
            'trade_type': self.trade_type,
            'symbol': self.symbol,
            'mint': self.mint,
            'outcome': self.outcome.to_dict(),
            'mistakes': [m.to_dict() for m in self.mistakes],
            'primary_mistake': self.primary_mistake.value if self.primary_mistake else None,
            'missed_opportunity_pct': round(self.missed_opportunity_pct, 2),
            'monthly_impact_estimate_pct': round(self.monthly_impact_estimate_pct, 2),
            'monthly_impact_estimate_usd': round(self.monthly_impact_estimate_usd, 2),
            'coaching_message': self.coaching_message,
            'recommendations': self.recommendations,
            'confidence': round(self.confidence, 2),
            'analysis_timestamp': self.analysis_timestamp
        }


# ==================== POST-TRADE REVIEWER CLASS ====================

class PostTradeReviewer:
    """
    Analyzes completed trades to identify mistakes and provide coaching
    """
    
    def __init__(self, birdeye_fetcher=None, data_fetcher=None, user_profiler=None):
        """
        Initialize Post-Trade Reviewer
        
        Args:
            birdeye_fetcher: BirdeyeFetcher instance for price data
            data_fetcher: DataFetcher instance for user history
            user_profiler: UserProfiler instance for user patterns
        """
        self.config = post_trade_review_config
        self.birdeye_fetcher = birdeye_fetcher
        self.data_fetcher = data_fetcher
        self.user_profiler = user_profiler
        
        logger.info("Post-Trade Reviewer initialized")
    
    def set_birdeye_fetcher(self, birdeye_fetcher):
        """Set or update the Birdeye fetcher"""
        self.birdeye_fetcher = birdeye_fetcher
    
    def set_data_fetcher(self, data_fetcher):
        """Set or update the data fetcher"""
        self.data_fetcher = data_fetcher
    
    def set_user_profiler(self, user_profiler):
        """Set or update the user profiler"""
        self.user_profiler = user_profiler
    
    def analyze_trade(
        self,
        trade: TradeInput,
        price_history_before: Optional[List[PricePoint]] = None,
        price_history_after: Optional[List[PricePoint]] = None,
        user_context: Optional[UserContext] = None
    ) -> PostTradeAnalysis:
        """
        Analyze a completed trade and identify mistakes
        
        Args:
            trade: The trade to analyze
            price_history_before: Price data before the trade (for entry analysis)
            price_history_after: Price data after the trade (for exit analysis)
            user_context: User's historical trading context
            
        Returns:
            PostTradeAnalysis with identified mistakes and recommendations
        """
        logger.info(f"Analyzing trade {trade.trade_id[:16]}... ({trade.symbol}, {'BUY' if trade.is_buy else 'SELL'})")
        
        # Initialize analysis
        trade_type = "BUY" if trade.is_buy else "SELL"
        outcome = self._analyze_outcome(trade, price_history_before, price_history_after)
        mistakes = []
        
        # Detect mistakes based on trade type
        if trade.is_buy:
            # For BUY trades: check for late entry, FOMO, over-trading, revenge trade
            # LATE_ENTRY: uses price_history_after to detect dip after buying (you could have waited)
            # FOMO_ENTRY: uses price_history_before to detect buying after a pump
            late_entry_mistakes = self._detect_late_entry(trade, price_history_after, outcome)
            fomo_mistakes = self._detect_fomo_entry(trade, price_history_before)
            
            # Add late entry mistakes first (forward-looking: dip after entry)
            mistakes.extend(late_entry_mistakes)
            
            # Only add FOMO if LATE_ENTRY didn't already fire (avoid double-counting)
            if not late_entry_mistakes:
                mistakes.extend(fomo_mistakes)
        else:
            # For SELL trades: check for early exit, panic sell
            mistakes.extend(self._detect_early_exit(trade, price_history_after, outcome))
            mistakes.extend(self._detect_panic_sell(trade, user_context))
        
        # Common mistake detection (both BUY and SELL)
        if user_context:
            mistakes.extend(self._detect_revenge_trade(trade, user_context))
            mistakes.extend(self._detect_over_trading(trade, user_context))
            mistakes.extend(self._detect_bad_risk_reward(trade, user_context))
        
        # Sort mistakes by severity and impact
        mistakes = self._prioritize_mistakes(mistakes)
        
        # Determine primary mistake
        primary_mistake = mistakes[0].type if mistakes else None
        
        # Calculate missed opportunity
        missed_opportunity_pct = self._calculate_missed_opportunity(trade, outcome)
        
        # Estimate monthly impact
        monthly_impact_pct, monthly_impact_usd = self._estimate_monthly_impact(
            mistakes, 
            missed_opportunity_pct,
            trade,
            user_context
        )
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(
            price_history_before,
            price_history_after,
            user_context
        )
        
        analysis = PostTradeAnalysis(
            trade_id=trade.trade_id,
            trade_type=trade_type,
            symbol=trade.symbol,
            mint=trade.mint,
            outcome=outcome,
            mistakes=mistakes,
            primary_mistake=primary_mistake,
            missed_opportunity_pct=missed_opportunity_pct,
            monthly_impact_estimate_pct=monthly_impact_pct,
            monthly_impact_estimate_usd=monthly_impact_usd,
            confidence=confidence,
            user_context=user_context
        )
        
        logger.info(f"Analysis complete: {len(mistakes)} mistakes identified, "
                   f"primary: {primary_mistake.value if primary_mistake else 'None'}")
        
        return analysis
    
    # ==================== OUTCOME ANALYSIS ====================
    
    def _analyze_outcome(
        self,
        trade: TradeInput,
        price_history_before: Optional[List[PricePoint]],
        price_history_after: Optional[List[PricePoint]]
    ) -> OutcomeComparison:
        """Analyze what happened before and after the trade"""
        outcome = OutcomeComparison(price_at_trade=trade.price)
        
        # Analyze price history AFTER trade (for SELL analysis)
        if price_history_after and len(price_history_after) > 0:
            prices_after = [p.price for p in price_history_after]
            times_after = [p.timestamp for p in price_history_after]
            
            # Find max/min prices after trade
            max_idx = np.argmax(prices_after)
            min_idx = np.argmin(prices_after)
            
            outcome.max_price_after_exit = float(prices_after[max_idx])
            outcome.min_price_after_exit = float(prices_after[min_idx])
            
            # Calculate time to max/min
            if max_idx < len(times_after):
                time_diff = times_after[max_idx] - trade.executed_at
                outcome.time_to_max = self._format_timedelta(time_diff)
            
            if min_idx < len(times_after):
                time_diff = times_after[min_idx] - trade.executed_at
                outcome.time_to_min = self._format_timedelta(time_diff)
            
            # Get prices at specific intervals
            outcome.price_after_1h = self._get_price_at_offset(
                price_history_after, trade.executed_at, self.config.OUTCOME_WINDOW_1H
            )
            outcome.price_after_4h = self._get_price_at_offset(
                price_history_after, trade.executed_at, self.config.OUTCOME_WINDOW_4H
            )
            outcome.price_after_24h = self._get_price_at_offset(
                price_history_after, trade.executed_at, self.config.OUTCOME_WINDOW_24H
            )
            
            # Calculate missed gain for SELL trades
            if not trade.is_buy and outcome.max_price_after_exit:
                missed_gain = (outcome.max_price_after_exit - trade.price) / trade.price
                outcome.missed_gain_pct = max(0, missed_gain * 100)
                outcome.optimal_exit_price = outcome.max_price_after_exit
        
        # Analyze price history BEFORE trade (for context)
        if price_history_before and len(price_history_before) > 0:
            prices_before = [p.price for p in price_history_before]
            
            outcome.max_price_before_entry = float(max(prices_before))
            outcome.min_price_before_entry = float(min(prices_before))
        
        # Calculate missed dip for BUY trades (using post-entry data within LATE_ENTRY window)
        # A "missed dip" means: after you bought, the price dropped — you could have waited
        # Uses the same window as LATE_ENTRY detection for consistency
        if trade.is_buy and price_history_after and len(price_history_after) > 0:
            window_cutoff = trade.executed_at + timedelta(hours=self.config.LATE_ENTRY_WINDOW_HOURS)
            prices_in_window = [
                p for p in price_history_after
                if p.timestamp <= window_cutoff
            ]
            
            if prices_in_window:
                min_point = min(prices_in_window, key=lambda p: p.price)
                min_price_in_window = float(min_point.price)
                
                if min_price_in_window < trade.price:
                    missed_dip = (trade.price - min_price_in_window) / trade.price
                    outcome.missed_dip_pct = max(0, missed_dip * 100)
                    outcome.optimal_entry_price = min_price_in_window
        
        return outcome
    
    def _get_price_at_offset(
        self,
        price_history: List[PricePoint],
        trade_time: datetime,
        offset_minutes: int
    ) -> Optional[float]:
        """Get price at a specific time offset from trade"""
        target_time = trade_time + timedelta(minutes=offset_minutes)
        
        # Find closest price point to target time
        closest_price = None
        min_diff = timedelta(hours=24)
        
        for point in price_history:
            diff = abs(point.timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_price = point.price
        
        # Only return if within tolerance of target
        if min_diff <= timedelta(minutes=self.config.PRICE_MATCH_TOLERANCE_MINUTES):
            return closest_price
        return None
    
    # ==================== MISTAKE DETECTION ====================
    
    def _detect_early_exit(
        self,
        trade: TradeInput,
        price_history_after: Optional[List[PricePoint]],
        outcome: OutcomeComparison
    ) -> List[Mistake]:
        """Detect if user exited too early (SELL trades only)"""
        mistakes = []
        
        if trade.is_buy or not price_history_after:
            return mistakes
        
        if outcome.max_price_after_exit is None:
            return mistakes
        
        # Calculate how much price rose after exit
        gain_after_exit = (outcome.max_price_after_exit - trade.price) / trade.price
        
        if gain_after_exit > self.config.EARLY_EXIT_GAIN_THRESHOLD:
            # Determine severity
            if gain_after_exit >= self.config.EARLY_EXIT_SEVERITY_HIGH:
                severity = Severity.HIGH
            elif gain_after_exit >= self.config.EARLY_EXIT_SEVERITY_MEDIUM:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            mistakes.append(Mistake(
                type=MistakeType.EARLY_EXIT,
                severity=severity,
                impact_pct=gain_after_exit * 100,
                description=f"Price rose {gain_after_exit*100:.1f}% after your exit",
                evidence={
                    'your_exit_price': trade.price,
                    'max_price_after': outcome.max_price_after_exit,
                    'gain_after_exit_pct': round(gain_after_exit * 100, 2),
                    'time_to_peak': outcome.time_to_max,
                    'price_after_1h': outcome.price_after_1h,
                    'price_after_4h': outcome.price_after_4h
                }
            ))
        
        return mistakes
    
    def _detect_late_entry(
        self,
        trade: TradeInput,
        price_history_after: Optional[List[PricePoint]],
        outcome: OutcomeComparison
    ) -> List[Mistake]:
        """
        Detect if user entered too early — price dipped AFTER buying (BUY trades only).
        
        A true "late entry" means: you bought, then the price dropped, meaning you
        could have waited and gotten a better price. This looks at price_history_after
        (not before) to find if a meaningful dip occurred within the lookback window
        after the trade was executed.
        """
        mistakes = []
        
        if not trade.is_buy or not price_history_after:
            return mistakes
        
        # Filter to only the late-entry lookback window after trade
        window_cutoff = trade.executed_at + timedelta(hours=self.config.LATE_ENTRY_WINDOW_HOURS)
        prices_in_window = [
            p for p in price_history_after
            if p.timestamp <= window_cutoff
        ]
        
        if not prices_in_window:
            return mistakes
        
        # Find the minimum price in the window after entry
        min_price_after = min(p.price for p in prices_in_window)
        
        # Calculate how much cheaper the user could have bought by waiting
        dip_pct = (trade.price - min_price_after) / trade.price
        
        if dip_pct > self.config.LATE_ENTRY_BETTER_PRICE_PCT:
            # Determine severity
            if dip_pct >= self.config.SEVERITY_HIGH_THRESHOLD:
                severity = Severity.HIGH
            elif dip_pct >= self.config.SEVERITY_MEDIUM_THRESHOLD:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            # Find when the dip occurred
            min_point = min(prices_in_window, key=lambda p: p.price)
            time_to_dip = min_point.timestamp - trade.executed_at
            dip_time_str = self._format_timedelta(time_to_dip)
            
            mistakes.append(Mistake(
                type=MistakeType.LATE_ENTRY,
                severity=severity,
                impact_pct=dip_pct * 100,
                description=f"Price dipped {dip_pct*100:.1f}% within {dip_time_str} after your entry — a better price was available by waiting",
                evidence={
                    'your_entry_price': trade.price,
                    'best_price_after_entry': min_price_after,
                    'dip_pct': round(dip_pct * 100, 2),
                    'time_to_dip': dip_time_str,
                    'lookback_hours': self.config.LATE_ENTRY_WINDOW_HOURS
                }
            ))
        
        return mistakes
    
    def _detect_fomo_entry(
        self,
        trade: TradeInput,
        price_history_before: Optional[List[PricePoint]]
    ) -> List[Mistake]:
        """Detect FOMO entry - buying after a significant pump"""
        mistakes = []
        
        if not trade.is_buy or not price_history_before:
            return mistakes
        
        if len(price_history_before) < 2:
            return mistakes
        
        # Calculate price change in lookback period
        earliest_price = price_history_before[0].price
        latest_price = price_history_before[-1].price
        
        price_change = (latest_price - earliest_price) / earliest_price
        
        if price_change > self.config.FOMO_PUMP_THRESHOLD:
            # Determine severity based on pump size
            if price_change >= self.config.FOMO_SEVERITY_HIGH:
                severity = Severity.HIGH
            elif price_change >= self.config.FOMO_SEVERITY_MEDIUM:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            mistakes.append(Mistake(
                type=MistakeType.FOMO_ENTRY,
                severity=severity,
                impact_pct=price_change * 100,
                description=f"You bought after a {price_change*100:.1f}% pump in the prior {self.config.FOMO_LOOKBACK_HOURS}h",
                evidence={
                    'price_change_before_entry': round(price_change * 100, 2),
                    'lookback_hours': self.config.FOMO_LOOKBACK_HOURS,
                    'entry_price': trade.price,
                    'price_at_start_of_pump': earliest_price
                }
            ))
        
        return mistakes
    
    def _detect_panic_sell(
        self,
        trade: TradeInput,
        user_context: Optional[UserContext]
    ) -> List[Mistake]:
        """Detect panic sell - selling at a loss with short hold time"""
        mistakes = []
        
        if trade.is_buy:
            return mistakes
        
        # Check if this was a losing trade
        if trade.realized_pnl >= 0:
            return mistakes
        
        # Calculate loss percentage (estimate if not directly available)
        loss_pct = trade.realized_pnl / (trade.price * trade.qty) if trade.price * trade.qty > 0 else 0
        
        # Check if loss exceeds panic threshold
        if loss_pct < self.config.PANIC_SELL_LOSS_THRESHOLD:
            # Check hold time if we have user context
            hold_time_hours = None
            if user_context and user_context.trades_on_same_mint:
                # Find the corresponding buy trade
                for t in user_context.trades_on_same_mint:
                    if t.get('is_buy', False):
                        buy_time = t.get('executed_at')
                        if buy_time:
                            if isinstance(buy_time, str):
                                buy_time = datetime.fromisoformat(buy_time.replace('Z', '+00:00'))
                            # Make timezone-naive for comparison
                            if hasattr(buy_time, 'tzinfo') and buy_time.tzinfo is not None:
                                buy_time = buy_time.replace(tzinfo=None)
                            hold_time_hours = (trade.executed_at - buy_time).total_seconds() / 3600
                            break
            
            is_short_hold = hold_time_hours is not None and hold_time_hours < self.config.PANIC_SELL_MAX_HOLD_HOURS
            
            # Determine severity
            if abs(loss_pct) >= self.config.PANIC_SELL_SEVERITY_HIGH:
                severity = Severity.HIGH
            elif abs(loss_pct) >= self.config.PANIC_SELL_SEVERITY_MEDIUM:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            description = f"Sold at {loss_pct*100:.1f}% loss"
            if is_short_hold:
                description += f" after only {hold_time_hours:.1f}h hold time"
            
            mistakes.append(Mistake(
                type=MistakeType.PANIC_SELL,
                severity=severity,
                impact_pct=abs(loss_pct) * 100,
                description=description,
                evidence={
                    'loss_pct': round(loss_pct * 100, 2),
                    'hold_time_hours': round(hold_time_hours, 2) if hold_time_hours else None,
                    'realized_pnl': trade.realized_pnl,
                    'threshold_loss_pct': self.config.PANIC_SELL_LOSS_THRESHOLD * 100
                }
            ))
        
        return mistakes
    
    def _detect_revenge_trade(
        self,
        trade: TradeInput,
        user_context: UserContext
    ) -> List[Mistake]:
        """Detect revenge trading - trading shortly after a loss"""
        mistakes = []
        
        if not user_context.recent_losses:
            return mistakes
        
        # Check for recent losses within window
        revenge_window = timedelta(hours=self.config.REVENGE_TRADE_WINDOW_HOURS)
        
        for loss_trade in user_context.recent_losses:
            loss_time = loss_trade.get('executed_at')
            if loss_time:
                if isinstance(loss_time, str):
                    loss_time = datetime.fromisoformat(loss_time.replace('Z', '+00:00'))
                # Make timezone-naive for comparison
                if hasattr(loss_time, 'tzinfo') and loss_time.tzinfo is not None:
                    loss_time = loss_time.replace(tzinfo=None)
                
                time_since_loss = trade.executed_at - loss_time
                
                if timedelta(0) < time_since_loss <= revenge_window:
                    loss_pct = loss_trade.get('pnl_pct', 0)
                    
                    if loss_pct <= self.config.REVENGE_LOSS_THRESHOLD:
                        severity = Severity.HIGH if abs(loss_pct) > self.config.REVENGE_SEVERITY_HIGH_LOSS else Severity.MEDIUM
                        
                        mistakes.append(Mistake(
                            type=MistakeType.REVENGE_TRADE,
                            severity=severity,
                            impact_pct=0,  # Impact is behavioral, not direct
                            description=f"Trade made {time_since_loss.total_seconds()/60:.0f}min after a {loss_pct*100:.1f}% loss",
                            evidence={
                                'time_since_loss_minutes': round(time_since_loss.total_seconds() / 60, 0),
                                'previous_loss_pct': round(loss_pct * 100, 2),
                                'previous_loss_symbol': loss_trade.get('symbol', 'unknown'),
                                'revenge_window_hours': self.config.REVENGE_TRADE_WINDOW_HOURS
                            }
                        ))
                        break  # Only flag once
        
        return mistakes
    
    def _detect_over_trading(
        self,
        trade: TradeInput,
        user_context: UserContext
    ) -> List[Mistake]:
        """Detect over-trading - multiple trades on same token in short window"""
        mistakes = []
        
        if not user_context.trades_on_same_mint:
            return mistakes
        
        # Count trades within window
        overtrade_window = timedelta(hours=self.config.OVERTRADE_WINDOW_HOURS)
        trades_in_window = 0
        
        for t in user_context.trades_on_same_mint:
            trade_time = t.get('executed_at')
            if trade_time:
                if isinstance(trade_time, str):
                    trade_time = datetime.fromisoformat(trade_time.replace('Z', '+00:00'))
                # Make timezone-naive for comparison
                if hasattr(trade_time, 'tzinfo') and trade_time.tzinfo is not None:
                    trade_time = trade_time.replace(tzinfo=None)
                
                time_diff = abs(trade.executed_at - trade_time)
                if time_diff <= overtrade_window:
                    trades_in_window += 1
        
        if trades_in_window >= self.config.OVERTRADE_MIN_TRADES:
            severity = Severity.HIGH if trades_in_window >= self.config.OVERTRADE_SEVERITY_HIGH else Severity.MEDIUM
            
            mistakes.append(Mistake(
                type=MistakeType.OVER_TRADING,
                severity=severity,
                impact_pct=0,  # Impact is in fees and spread, hard to quantify
                description=f"{trades_in_window} trades on {trade.symbol} within {self.config.OVERTRADE_WINDOW_HOURS}h",
                evidence={
                    'trades_in_window': trades_in_window,
                    'window_hours': self.config.OVERTRADE_WINDOW_HOURS,
                    'symbol': trade.symbol,
                    'min_trades_threshold': self.config.OVERTRADE_MIN_TRADES
                }
            ))
        
        return mistakes
    
    def _detect_bad_risk_reward(
        self,
        trade: TradeInput,
        user_context: UserContext
    ) -> List[Mistake]:
        """Detect bad risk-reward ratio compared to user's typical trades"""
        mistakes = []
        
        if trade.is_buy:
            return mistakes  # Only analyze completed (SELL) trades
        
        if not user_context or user_context.avg_pnl_per_trade == 0:
            return mistakes
        
        # Calculate this trade's R-multiple (if we had stop loss info)
        # For now, compare PnL to average
        this_trade_pnl_pct = (trade.realized_pnl / (trade.price * trade.qty)) if trade.price * trade.qty > 0 else 0
        
        # If this trade's PnL is significantly worse than average
        if this_trade_pnl_pct < 0 and abs(this_trade_pnl_pct) > abs(user_context.avg_pnl_per_trade) * self.config.BAD_RR_MULTIPLIER:
            severity = Severity.MEDIUM
            
            mistakes.append(Mistake(
                type=MistakeType.BAD_RISK_REWARD,
                severity=severity,
                impact_pct=abs(this_trade_pnl_pct) * 100,
                description=f"This trade's loss ({this_trade_pnl_pct*100:.1f}%) is worse than your average ({user_context.avg_pnl_per_trade*100:.1f}%)",
                evidence={
                    'this_trade_pnl_pct': round(this_trade_pnl_pct * 100, 2),
                    'average_pnl_pct': round(user_context.avg_pnl_per_trade * 100, 2),
                    'user_win_rate': round(user_context.win_rate * 100, 2)
                }
            ))
        
        return mistakes
    
    # ==================== IMPACT CALCULATION ====================
    
    def _prioritize_mistakes(self, mistakes: List[Mistake]) -> List[Mistake]:
        """Sort mistakes by severity and impact"""
        severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        
        return sorted(
            mistakes,
            key=lambda m: (severity_order[m.severity], -m.impact_pct)
        )
    
    def _calculate_missed_opportunity(
        self,
        trade: TradeInput,
        outcome: OutcomeComparison
    ) -> float:
        """Calculate the total missed opportunity percentage"""
        if trade.is_buy:
            return outcome.missed_dip_pct
        else:
            return outcome.missed_gain_pct
    
    def _estimate_monthly_impact(
        self,
        mistakes: List[Mistake],
        missed_opportunity_pct: float,
        trade: TradeInput,
        user_context: Optional[UserContext]
    ) -> Tuple[float, float]:
        """Estimate monthly impact of this mistake pattern"""
        if not mistakes:
            return 0.0, 0.0
        
        # Base impact is the missed opportunity
        base_impact_pct = missed_opportunity_pct
        
        # Estimate trades per month
        trades_per_month = self.config.TRADES_PER_MONTH_ESTIMATE
        if user_context and user_context.total_trades > 0:
            # Use user's actual trading frequency if available
            # Assume data is from last 30 days for simplicity
            trades_per_month = max(user_context.total_trades, trades_per_month)
        
        # Estimate how often this mistake type occurs
        mistake_frequency = self.config.MISTAKE_FREQUENCY_ESTIMATE
        
        # Calculate monthly impact
        monthly_impact_pct = base_impact_pct * mistake_frequency
        
        # Calculate USD impact (based on trade size)
        trade_value_usd = trade.price * trade.qty
        monthly_impact_usd = trade_value_usd * (monthly_impact_pct / 100) * trades_per_month
        
        return monthly_impact_pct, monthly_impact_usd
    
    def _calculate_confidence(
        self,
        price_history_before: Optional[List[PricePoint]],
        price_history_after: Optional[List[PricePoint]],
        user_context: Optional[UserContext]
    ) -> float:
        """Calculate confidence score based on data quality"""
        confidence = self.config.CONFIDENCE_BASE
        
        candles_many = self.config.CONFIDENCE_PRICE_CANDLES_MANY
        
        # Add confidence for price data
        if price_history_after and len(price_history_after) > candles_many:
            confidence += self.config.CONFIDENCE_PRICE_AFTER_MANY
        elif price_history_after and len(price_history_after) > 0:
            confidence += self.config.CONFIDENCE_PRICE_AFTER_SOME
        
        if price_history_before and len(price_history_before) > candles_many:
            confidence += self.config.CONFIDENCE_PRICE_BEFORE_MANY
        elif price_history_before and len(price_history_before) > 0:
            confidence += self.config.CONFIDENCE_PRICE_BEFORE_SOME
        
        # Add confidence for user context
        if user_context:
            if user_context.total_trades > self.config.CONFIDENCE_USER_TRADES_MANY_THRESHOLD:
                confidence += self.config.CONFIDENCE_USER_TRADES_MANY
            elif user_context.total_trades > self.config.CONFIDENCE_USER_TRADES_SOME_THRESHOLD:
                confidence += self.config.CONFIDENCE_USER_TRADES_SOME
        
        return min(self.config.CONFIDENCE_MAX, confidence)
    
    # ==================== UTILITY METHODS ====================
    
    def _format_timedelta(self, td: timedelta) -> str:
        """Format a timedelta as a human-readable string"""
        total_seconds = int(td.total_seconds())
        
        if total_seconds < 0:
            return "N/A"
        
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


# ==================== HELPER FUNCTIONS ====================

def parse_trade_input(data: Dict[str, Any]) -> TradeInput:
    """Parse trade data from API request into TradeInput object"""
    # Parse executed_at timestamp
    executed_at = data.get('executed_at')
    if isinstance(executed_at, str):
        # Handle ISO format with Z suffix
        executed_at = executed_at.replace('Z', '+00:00')
        executed_at = datetime.fromisoformat(executed_at)
        # Make timezone-naive for consistent comparisons
        if executed_at.tzinfo is not None:
            executed_at = executed_at.replace(tzinfo=None)
    elif isinstance(executed_at, (int, float)):
        executed_at = datetime.fromtimestamp(executed_at / 1000)  # Assume milliseconds
    
    return TradeInput(
        trade_id=data.get('trade_id', ''),
        symbol=data.get('symbol', ''),
        mint=data.get('mint', ''),
        qty=float(data.get('qty', 0)),
        price=float(data.get('price', 0)),
        executed_at=executed_at,
        realized_pnl=float(data.get('realized_pnl', 0)),
        is_buy=data.get('is_buy', True),
        creation_timestamp=data.get('creation_timestamp'),
        user_address=data.get('user_address')
    )


def parse_price_history(ohlcv_data: List[Dict[str, Any]]) -> List[PricePoint]:
    """Parse OHLCV data into PricePoint list"""
    price_points = []
    
    for candle in ohlcv_data:
        timestamp = candle.get('unixTime') or candle.get('timestamp')
        if timestamp:
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
            
            # Use close price as the price point
            price = candle.get('c') or candle.get('close') or candle.get('price', 0)
            volume = candle.get('v') or candle.get('volume', 0)
            
            if price > 0:
                price_points.append(PricePoint(
                    timestamp=dt,
                    price=float(price),
                    volume=float(volume) if volume else None
                ))
    
    # Sort by timestamp
    price_points.sort(key=lambda p: p.timestamp)
    
    return price_points


if __name__ == "__main__":
    print("Post-Trade Reviewer Engine (Feature 4)")
    print("=" * 60)
    
    # Test with mock data
    reviewer = PostTradeReviewer()
    
    # Create a mock trade
    mock_trade = TradeInput(
        trade_id="test123",
        symbol="TEST",
        mint="TestMint123",
        qty=1000,
        price=0.0001,
        executed_at=datetime.now() - timedelta(hours=4),
        realized_pnl=-10.0,
        is_buy=False  # SELL trade
    )
    
    # Create mock price history (price went up after exit)
    mock_price_after = [
        PricePoint(datetime.now() - timedelta(hours=3), 0.000105),
        PricePoint(datetime.now() - timedelta(hours=2), 0.000115),
        PricePoint(datetime.now() - timedelta(hours=1), 0.000125),
        PricePoint(datetime.now(), 0.000110),
    ]
    
    # Run analysis
    analysis = reviewer.analyze_trade(
        trade=mock_trade,
        price_history_after=mock_price_after
    )
    
    print(f"\nAnalysis Result:")
    print(f"  Trade: {analysis.trade_type} {analysis.symbol}")
    print(f"  Mistakes Found: {len(analysis.mistakes)}")
    for m in analysis.mistakes:
        print(f"    - {m.type.value} ({m.severity.value}): {m.description}")
    print(f"  Missed Opportunity: {analysis.missed_opportunity_pct:.1f}%")
    print(f"  Monthly Impact: {analysis.monthly_impact_estimate_pct:.1f}%")
    print(f"  Confidence: {analysis.confidence:.2f}")
