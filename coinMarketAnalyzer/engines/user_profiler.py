"""
User Profiler Engine
Analyzes user trading history to build behavioral profiles and identify patterns
Part of Layer 2: Pre-Execution Risk & Edge Assessment
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import user_profile_config, cache_config
from core.circuit_breaker import sentry_fallback_warning

# Setup logging
logger = logging.getLogger(__name__)


# ==================== ENUMS ====================

class PatternType(str, Enum):
    """Types of trading patterns that can be identified"""
    EARLY_ENTRY = "EARLY_ENTRY"          # Enters in P1/P2 phase
    LATE_ENTRY = "LATE_ENTRY"            # Enters in P4 phase
    DIP_BUY = "DIP_BUY"                  # Buys after price drops
    FOMO = "FOMO"                        # Buys at local peaks
    QUICK_FLIP = "QUICK_FLIP"            # Holds < 24 hours
    MEDIUM_HOLD = "MEDIUM_HOLD"          # Holds 1-7 days (24h-168h)
    LONG_HOLD = "LONG_HOLD"              # Holds > 7 days
    PANIC_SELL = "PANIC_SELL"            # Sells at significant loss quickly
    HIGH_CONCENTRATION = "HIGH_CONCENTRATION"  # Enters when top 10 > 60%
    LOW_CONCENTRATION = "LOW_CONCENTRATION"    # Enters when top 10 < 30%
    WHALE_FOLLOW = "WHALE_FOLLOW"        # Trades align with whale movements
    UNKNOWN = "UNKNOWN"                   # Unclassified pattern


class TraderType(str, Enum):
    """Classification of trader behavior"""
    SNIPER = "SNIPER"      # Buys within first 60 seconds
    FLIPPER = "FLIPPER"    # Holds < 24 hours typically
    HOLDER = "HOLDER"      # Holds > 7 days typically
    MIXED = "MIXED"        # Mixed behavior


# ==================== DATA CLASSES ====================

@dataclass
class TradeRecord:
    """Single trade record from user history"""
    mint: str
    timestamp: datetime
    price: float
    amount: float
    is_buy: bool
    holding_before: float
    holding_after: float
    transaction_id: Optional[str] = None
    slot: Optional[int] = None


@dataclass
class MintPnL:
    """Profit/Loss calculation for a single mint traded by user"""
    mint: str
    total_trades: int
    total_bought_value: float      # Total SOL/USD spent
    total_sold_value: float        # Total SOL/USD received
    realized_pnl: float            # Realized profit/loss
    unrealized_holding: float      # Remaining tokens
    is_winner: bool                # Was this trade profitable?
    entry_timestamp: Optional[datetime] = None
    exit_timestamp: Optional[datetime] = None
    avg_hold_time_hours: float = 0.0
    entry_phase: str = "UNKNOWN"   # P1, P2, P3, P4
    entry_context: Dict[str, Any] = field(default_factory=dict)
    pnl_percent: float = 0.0       # Percentage gain/loss
    # Price context for pattern detection (NEW)
    price_change_before_entry: float = 0.0  # Price change % in lookback period before entry
    whale_state_at_entry: str = "UNKNOWN"   # Whale state when user entered (Accumulation/Distribution/Stability)
    is_fomo_entry: bool = False             # Bought after significant pump
    is_dip_buy: bool = False                # Bought after significant dip
    is_whale_follow: bool = False           # Bought during whale accumulation
    token_age_at_entry_days: float = 0.0    # Token age when user first bought


@dataclass
class TradePattern:
    """A recurring trading pattern identified in user history"""
    pattern_id: str
    pattern_type: PatternType
    occurrences: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl_percent: float
    total_pnl: float
    typical_context: Dict[str, Any] = field(default_factory=dict)
    sample_mints: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """Complete user trading profile"""
    wallet_address: str
    total_trades: int
    total_mints_traded: int
    
    # Overall Performance
    overall_win_rate: float
    avg_pnl_per_trade: float
    total_realized_pnl: float
    
    # API-provided values (from /user-profiling endpoint)
    lifetime_pnl: float = 0.0               # Pre-calculated from API
    total_volume_usd: float = 0.0           # Pre-calculated from API
    max_drawdown_pct: float = 0.0           # Pre-calculated from API
    avg_r_multiple: float = 0.0             # Pre-calculated from API
    avg_holding_time_minutes: float = 0.0   # Pre-calculated from API
    
    # Enriched metrics from newer /user-profiling response (optional)
    win_rate_round_trip: Optional[float] = None     # 0-100 scale
    win_rate_execution: Optional[float] = None      # 0-100 scale
    total_closed_pnl: Optional[float] = None
    total_fees: Optional[float] = None
    completed_round_trips: Optional[int] = None
    trades_long: Optional[int] = None
    trades_short: Optional[int] = None
    avg_leverage: Optional[float] = None
    bot_detected: Optional[bool] = None
    
    # Detailed PnL per mint (still used for pattern detection)
    mint_pnls: List[MintPnL] = field(default_factory=list)
    
    # Patterns Identified
    patterns: List[TradePattern] = field(default_factory=list)
    best_pattern: Optional[TradePattern] = None
    worst_pattern: Optional[TradePattern] = None
    
    # Behavioral Metrics
    avg_hold_time_hours: float = 0.0
    typical_entry_phase: str = "UNKNOWN"
    trader_type: TraderType = TraderType.MIXED
    
    # Behavioral Flags
    is_sniper: bool = False
    is_holder: bool = False
    is_flipper: bool = False
    # Optional: None when "not computed" (enough trades but all zero — insufficient pattern data)
    fomo_tendency: Optional[float] = 0.0         # 0-1, tendency to buy after pumps
    panic_sell_tendency: Optional[float] = 0.0   # 0-1, tendency to sell at losses quickly
    dip_buy_tendency: Optional[float] = 0.0      # 0-1, tendency to buy after dips (NEW)
    whale_follow_tendency: Optional[float] = 0.0 # 0-1, tendency to follow whale accumulation (NEW)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    is_valid: bool = True
    confidence: float = 0.5  # How confident are we in this profile?


# ==================== USER PROFILER CLASS ====================

class UserProfiler:
    """
    Analyzes user trading history to build behavioral profiles
    and identify recurring patterns
    """
    
    def __init__(self, data_fetcher=None, birdeye_fetcher=None):
        """
        Initialize User Profiler
        
        Args:
            data_fetcher: DataFetcher instance for API calls
            birdeye_fetcher: BirdeyeFetcher instance for historical price data (optional)
        """
        self.data_fetcher = data_fetcher
        self.birdeye_fetcher = birdeye_fetcher
        self.profile_cache: Dict[str, Tuple[UserProfile, datetime]] = {}
        self.config = user_profile_config
        
        logger.info("User Profiler initialized")
    
    def set_data_fetcher(self, data_fetcher):
        """Set the data fetcher (for lazy initialization)"""
        self.data_fetcher = data_fetcher
    
    def set_birdeye_fetcher(self, birdeye_fetcher):
        """Set the Birdeye fetcher (for lazy initialization)"""
        self.birdeye_fetcher = birdeye_fetcher
    
    def build_profile(
        self, 
        wallet_address: str, 
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        use_birdeye_fallback: bool = False
    ) -> UserProfile:
        """
        Build comprehensive profile from user's trading history using new API
        
        Uses the new /user-profiling and /user-profiling/trades endpoints to:
        1. Get pre-calculated summary stats from API
        2. Use trades data for pattern detection only
        
        Args:
            wallet_address: User's wallet address
            from_date: Optional start date for profile analysis (ISO format)
            to_date: Optional end date for profile analysis (ISO format)
            
        Returns:
            UserProfile object
        """
        logger.info(f"Building profile for {wallet_address[:8]}...")
        
        # Check cache first
        cached = self._get_from_cache(wallet_address)
        if cached:
            logger.info(f"Returning cached profile for {wallet_address[:8]}...")
            return cached
        
        # Fetch user complete profile using new API
        if self.data_fetcher is None:
            logger.error("DataFetcher not set")
            return self._create_empty_profile(wallet_address)
        
        # Use new fetch_user_complete_profile method
        profile_data = self.data_fetcher.fetch_user_complete_profile(
            wallet_address, 
            from_date=from_date, 
            to_date=to_date
        )
        
        if not profile_data:
            logger.warning(f"No profile data for {wallet_address[:8]}...")
            return self._create_empty_profile(wallet_address)
        
        # Extract summary (pre-calculated stats from API)
        summary = profile_data.get('summary', {})
        trades = profile_data.get('trades', [])
        
        if not summary and not trades:
            logger.warning(f"Empty profile data for {wallet_address[:8]}...")
            return self._create_empty_profile(wallet_address)
        
        # Get pre-calculated values from API
        lifetime_pnl = summary.get('lifetime_pnl', 0.0)
        total_volume_usd = summary.get('total_volume_usd', 0.0)
        max_drawdown_pct = summary.get('max_drawdown_pct', 0.0)
        avg_r_multiple = summary.get('avg_r_multiple', 0.0)
        avg_holding_time_minutes = summary.get('avg_holding_time_minutes', 0.0)
        
        # Convert holding time to hours
        avg_hold_time_hours = avg_holding_time_minutes / 60.0 if avg_holding_time_minutes > 0 else 0.0
        
        # Calculate total trades from trades list
        total_trades = len(trades)
        
        # Check minimum trades requirement
        if total_trades < self.config.MIN_TRADES_FOR_PROFILE:
            logger.info(f"Insufficient trades ({total_trades}) for {wallet_address[:8]}...")
            profile = self._create_empty_profile(wallet_address)
            profile.total_trades = total_trades
            profile.lifetime_pnl = lifetime_pnl
            profile.avg_holding_time_minutes = avg_holding_time_minutes
            return profile
        
        # Count unique mints traded
        unique_mints = set()
        for trade in trades:
            mint = trade.get('mint') or trade.get('token') or trade.get('token_address')
            if mint:
                unique_mints.add(mint)
        
        # Skip bulk metadata fetching to avoid performance issues with many mints
        # Token metadata (creation dates) will be fetched on-demand only if needed
        # and the system will use default values for entry phase when metadata is unavailable
        token_metadata_cache = {}
        
        # Log the number of unique mints for debugging
        logger.info(f"Found {len(unique_mints)} unique mints for {wallet_address[:8]}... (skipping bulk metadata fetch)")
        
        # Process trades for pattern detection
        # Convert trades from new API format to MintPnL format for pattern analysis
        # Pass token metadata cache to enable accurate token age calculation
        mint_pnls = self._process_trades_for_patterns(trades, token_metadata_cache, use_birdeye_fallback=use_birdeye_fallback)
        
        # Win rate: prefer backend round-trip value when present (0-100 scale);
        # fall back to trade-list calculation for legacy responses.
        api_win_rate = summary.get('win_rate')
        if api_win_rate is not None and api_win_rate > 0:
            overall_win_rate = api_win_rate / 100.0 if api_win_rate > 1 else api_win_rate
        elif trades:
            winning_trades = sum(1 for t in trades if self._is_winning_trade(t))
            overall_win_rate = winning_trades / len(trades) if trades else 0.0
        else:
            overall_win_rate = 0.0
        
        # Identify patterns from trade data
        patterns = self._identify_patterns(mint_pnls) if mint_pnls else []
        
        # Calculate behavioral metrics from mint_pnls (which have pattern info)
        behaviors = self._calculate_behaviors(mint_pnls)
        
        # Also calculate tendencies from patterns for more accuracy
        if patterns and total_trades > 0:
            for pattern in patterns:
                pattern_name = pattern.pattern_type.value if hasattr(pattern.pattern_type, 'value') else str(pattern.pattern_type)
                tendency = pattern.occurrences / total_trades
                
                if pattern_name == 'FOMO':
                    behaviors['fomo_tendency'] = max(behaviors['fomo_tendency'], tendency)
                elif pattern_name == 'PANIC_SELL':
                    behaviors['panic_sell_tendency'] = max(behaviors['panic_sell_tendency'], tendency)
                elif pattern_name == 'DIP_BUY':
                    behaviors['dip_buy_tendency'] = max(behaviors['dip_buy_tendency'], tendency)
                elif pattern_name == 'WHALE_FOLLOW':
                    behaviors['whale_follow_tendency'] = max(behaviors['whale_follow_tendency'], tendency)
        
        # Determine trader type
        trader_type = self._determine_trader_type(behaviors)
        
        # Find best and worst patterns
        best_pattern = None
        worst_pattern = None
        if patterns:
            threshold = self.config.PATTERN_WIN_RATE_THRESHOLD
            # For best pattern, require a higher minimum (10) for statistical significance
            min_occurrences_for_best = max(10, self.config.MIN_TRADES_FOR_PATTERN)
            
            # For best pattern: require minimum occurrences AND good win rate
            # Sort by (occurrences >= min, win_rate, occurrences) to prioritize statistically significant patterns
            valid_patterns = [p for p in patterns if p.occurrences >= min_occurrences_for_best and p.pattern_type != PatternType.UNKNOWN]
            if valid_patterns:
                # Among valid patterns, find the one with best (win_rate * positive_pnl) score
                patterns_sorted_by_score = sorted(
                    valid_patterns,
                    key=lambda p: (p.win_rate * (1 if p.avg_pnl_percent > 0 else 0.5), p.occurrences),
                    reverse=True
                )
                best_pattern = patterns_sorted_by_score[0] if patterns_sorted_by_score[0].win_rate > threshold else None
            
            # For worst pattern, prioritize by avg_pnl_percent (most negative) and occurrences
            # This ensures PANIC_SELL with -40% avg PnL is worse than UNKNOWN with 0% PnL
            patterns_sorted_by_pnl = sorted(
                patterns, 
                key=lambda p: (p.avg_pnl_percent, -p.occurrences)  # Most negative PnL, then most occurrences
            )
            # Filter out UNKNOWN pattern if there are other patterns with negative PnL
            worst_candidates = [p for p in patterns_sorted_by_pnl if p.pattern_type != PatternType.UNKNOWN or len(patterns) == 1]
            if worst_candidates:
                worst_pattern = worst_candidates[0]
            else:
                worst_pattern = patterns_sorted_by_pnl[-1] if patterns_sorted_by_pnl[-1].win_rate < threshold else None
        
        # Calculate avg PnL per trade using API-provided lifetime_pnl
        avg_pnl = lifetime_pnl / total_trades if total_trades > 0 else 0.0
        
        # Calculate profile confidence based on data quantity
        confidence = self._calculate_confidence(total_trades, len(mint_pnls), len(patterns))
        
        # Determine typical entry phase from per-trade (buy) phase counts.
        # Each buy trade's phase is derived from its executed_at vs creation_timestamp,
        # reflecting "in which phase the wallet entered mostly."
        typical_phase = self._compute_typical_entry_phase(trades)
        
        # Build profile using API-provided values
        profile = UserProfile(
            wallet_address=wallet_address,
            total_trades=total_trades,
            total_mints_traded=len(unique_mints),
            overall_win_rate=overall_win_rate,
            avg_pnl_per_trade=avg_pnl,
            total_realized_pnl=lifetime_pnl,  # Use API value
            # Legacy API-provided fields
            lifetime_pnl=lifetime_pnl,
            total_volume_usd=total_volume_usd,
            max_drawdown_pct=max_drawdown_pct,
            avg_r_multiple=avg_r_multiple,
            avg_holding_time_minutes=avg_holding_time_minutes,
            # Enriched metrics (may be None for legacy responses)
            win_rate_round_trip=summary.get('win_rate'),
            win_rate_execution=summary.get('win_rate_execution_based'),
            total_closed_pnl=summary.get('total_closed_pnl'),
            total_fees=summary.get('total_fees'),
            completed_round_trips=summary.get('completed_round_trips'),
            trades_long=summary.get('trades_long'),
            trades_short=summary.get('trades_short'),
            avg_leverage=summary.get('avg_leverage'),
            bot_detected=summary.get('bot_detected'),
            # Pattern detection data
            mint_pnls=mint_pnls,
            patterns=patterns,
            best_pattern=best_pattern,
            worst_pattern=worst_pattern,
            # Use API-provided hold time
            avg_hold_time_hours=avg_hold_time_hours,
            typical_entry_phase=typical_phase,
            trader_type=trader_type,
            is_sniper=behaviors['is_sniper'],
            is_holder=behaviors['is_holder'],
            is_flipper=behaviors['is_flipper'],
            fomo_tendency=behaviors['fomo_tendency'],
            panic_sell_tendency=behaviors['panic_sell_tendency'],
            dip_buy_tendency=behaviors.get('dip_buy_tendency', 0.0),
            whale_follow_tendency=behaviors.get('whale_follow_tendency', 0.0),
            last_updated=datetime.now(),
            is_valid=True,
            confidence=confidence
        )
        
        # When we have enough trades but all tendency fields are 0, mark as "not computed" (null)
        if (profile.total_trades >= self.config.MIN_TRADES_FOR_BIAS_INFERENCE and
            (profile.fomo_tendency or 0) == 0 and (profile.panic_sell_tendency or 0) == 0 and
            (profile.dip_buy_tendency or 0) == 0 and (profile.whale_follow_tendency or 0) == 0):
            profile.fomo_tendency = None
            profile.panic_sell_tendency = None
            profile.dip_buy_tendency = None
            profile.whale_follow_tendency = None
        
        # Cache the profile
        self._add_to_cache(wallet_address, profile)
        
        logger.info(f"Profile built for {wallet_address[:8]}: "
                   f"{total_trades} trades, {overall_win_rate:.1%} win rate, "
                   f"lifetime_pnl=${lifetime_pnl:.2f}, "
                   f"{len(patterns)} patterns identified")
        
        return profile
    
    def get_profile(self, wallet_address: str, use_birdeye_fallback: bool = False, from_date: Optional[str] = None, to_date: Optional[str] = None) -> Optional[UserProfile]:
        """
        Get user profile (from cache or build new)
        
        Args:
            wallet_address: User's wallet address
            use_birdeye_fallback: Whether to use Birdeye API for historical price data
            from_date: Optional start date for trades (ISO format, defaults to today − 80 days)
            to_date: Optional end date for trades (ISO format, defaults to today)
            
        Returns:
            UserProfile or None if cannot build
        """
        # Check cache (only use cache if no custom dates provided)
        if from_date is None and to_date is None:
            cached = self._get_from_cache(wallet_address)
            if cached:
                return cached
        
        # Build new profile
        return self.build_profile(wallet_address, from_date=from_date, to_date=to_date, use_birdeye_fallback=use_birdeye_fallback)
    
    def invalidate_cache(self, wallet_address: str = None):
        """
        Invalidate cache for a specific user or all users
        
        Args:
            wallet_address: Specific wallet to invalidate, or None for all
        """
        if wallet_address:
            if wallet_address in self.profile_cache:
                del self.profile_cache[wallet_address]
                logger.info(f"Cache invalidated for {wallet_address[:8]}...")
        else:
            self.profile_cache.clear()
            logger.info("All profile cache cleared")
    
    # ==================== PRIVATE METHODS ====================
    
    def _is_winning_trade(self, trade: Dict) -> bool:
        """
        Determine if a trade was a winner based on trade data
        
        Args:
            trade: Single trade dictionary from API
            
        Returns:
            True if trade was profitable
        """
        # Check for various PnL field names in the trade data
        pnl = (
            trade.get('pnl') or 
            trade.get('realized_pnl') or 
            trade.get('profit') or 
            trade.get('profit_loss') or 
            0.0
        )
        
        try:
            return float(pnl) > self.config.PNL_WIN_THRESHOLD
        except (ValueError, TypeError):
            return False
    
    def _process_trades_for_patterns(self, trades: List[Dict], token_metadata_cache: Optional[Dict[str, Dict]] = None, use_birdeye_fallback: bool = False) -> List[MintPnL]:
        """
        Process trades from new API format into MintPnL format for pattern detection
        
        Args:
            trades: List of trades from /user-profiling/trades API
            token_metadata_cache: Optional cache of token metadata (creation dates) keyed by mint address
            use_birdeye_fallback: Whether to use Birdeye API for historical price data
            
        Returns:
            List of MintPnL objects for pattern analysis
        """
        # Sort all trades by timestamp first (for price history lookback)
        all_trades_sorted = sorted(
            trades,
            key=lambda x: x.get('executed_at') or x.get('timestamp') or x.get('time') or x.get('created_at') or ''
        )
        
        # Build a price history map: mint -> list of (timestamp, price) tuples
        # This allows looking up prices before a specific trade for FOMO/DIP detection
        price_history_by_mint: Dict[str, List[Tuple[datetime, float]]] = {}
        for trade in all_trades_sorted:
            mint = (
                trade.get('mint') or 
                trade.get('token') or 
                trade.get('token_address') or 
                trade.get('address')
            )
            if not mint:
                continue
            
            timestamp_str = trade.get('executed_at') or trade.get('timestamp') or trade.get('time') or trade.get('created_at')
            timestamp = self._parse_timestamp(timestamp_str)
            price = float(trade.get('price', 0) or 0)
            
            if mint not in price_history_by_mint:
                price_history_by_mint[mint] = []
            
            if timestamp and price > 0:
                price_history_by_mint[mint].append((timestamp, price))
        
        # Group trades by mint/token
        mint_trades: Dict[str, List[Dict]] = {}
        
        for trade in trades:
            mint = (
                trade.get('mint') or 
                trade.get('token') or 
                trade.get('token_address') or 
                trade.get('address')
            )
            if not mint:
                continue
            
            if mint not in mint_trades:
                mint_trades[mint] = []
            mint_trades[mint].append(trade)
        
        # Convert grouped trades to MintPnL objects
        mint_pnls = []
        for mint, trade_list in mint_trades.items():
            # Get token metadata from cache if available
            token_metadata = token_metadata_cache.get(mint) if token_metadata_cache else None
            # Get price history for this mint
            price_history = price_history_by_mint.get(mint, [])
            pnl = self._create_mint_pnl_from_trades(mint, trade_list, token_metadata, price_history, use_birdeye_fallback=use_birdeye_fallback)
            if pnl:
                mint_pnls.append(pnl)
        
        return mint_pnls
    
    def _create_mint_pnl_from_trades(
        self, 
        mint: str, 
        trades: List[Dict], 
        token_metadata: Optional[Dict] = None,
        price_history: Optional[List[Tuple[datetime, float]]] = None,
        use_birdeye_fallback: bool = False
    ) -> Optional[MintPnL]:
        """
        Create a MintPnL object from a list of trades for a single mint
        Uses trade-level data from the new API format
        
        Args:
            mint: Token mint address
            trades: List of trades for this mint
            token_metadata: Optional token metadata with creation date
            price_history: Optional list of (timestamp, price) tuples from all trades for this mint
            
        Returns:
            MintPnL object or None
        """
        if not trades:
            return None
        
        try:
            # Sort by timestamp - use 'executed_at' as primary field from new API
            trades_sorted = sorted(
                trades, 
                key=lambda x: x.get('executed_at') or x.get('timestamp') or x.get('time') or x.get('created_at') or ''
            )
            
            # Calculate totals from trades
            total_bought_value = 0.0
            total_sold_value = 0.0
            realized_pnl = 0.0
            hold_times = []
            
            entry_timestamp = None
            first_entry_timestamp = None
            exit_timestamp = None
            entry_price = None
            
            for trade in trades_sorted:
                try:
                    # Handle different field names from API
                    is_buy = (
                        trade.get('is_buy') or 
                        trade.get('isBuy') or 
                        trade.get('type', '').upper() == 'BUY' or
                        trade.get('side', '').upper() == 'BUY'
                    )
                    
                    # Use 'qty' from new API, fallback to other names
                    amount = abs(float(trade.get('qty', 0) or trade.get('amount', 0) or trade.get('quantity', 0) or 0))
                    # Price is provided directly by the API at trade execution time
                    price = float(trade.get('price', 0) or trade.get('entry_price', 0) or 0)
                    value = float(trade.get('value', 0) or trade.get('value_usd', 0) or (amount * price))
                    
                    # Get trade PnL if available - use 'realized_pnl' from new API
                    trade_pnl = float(trade.get('realized_pnl', 0) or trade.get('pnl', 0) or trade.get('profit', 0) or 0)
                    
                    # Use 'executed_at' as primary timestamp field from new API
                    timestamp_str = trade.get('executed_at') or trade.get('timestamp') or trade.get('time') or trade.get('created_at')
                    timestamp = self._parse_timestamp(timestamp_str) if timestamp_str else None
                    
                    if is_buy:
                        total_bought_value += value
                        if entry_timestamp is None:
                            entry_timestamp = timestamp
                            entry_price = price
                        if first_entry_timestamp is None:
                            first_entry_timestamp = timestamp
                    else:
                        total_sold_value += value
                        exit_timestamp = timestamp
                        
                        # Calculate hold time
                        if entry_timestamp and timestamp:
                            hold_time = (timestamp - entry_timestamp).total_seconds() / 3600
                            hold_times.append(hold_time)
                            entry_timestamp = None
                    
                    # Accumulate PnL
                    realized_pnl += trade_pnl
                    
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error processing trade: {e}")
                    continue
            
            # If API doesn't provide per-trade PnL, calculate from buy/sell difference
            if realized_pnl == 0 and total_bought_value > 0:
                realized_pnl = total_sold_value - total_bought_value
            
            # Calculate average hold time
            avg_hold_time = np.mean(hold_times) if hold_times else 0.0
            
            # Calculate PnL percentage
            pnl_percent = (realized_pnl / total_bought_value * 100) if total_bought_value > 0 else 0.0
            
            # Determine if winner
            is_winner = realized_pnl > self.config.PNL_WIN_THRESHOLD
            
            # Extract creation_timestamp from first trade (Unix timestamp in seconds)
            # This is provided by the /user-profiling/trades API
            creation_timestamp = None
            for trade in trades_sorted:
                ct = trade.get('creation_timestamp')
                if ct and ct > 0:
                    try:
                        creation_timestamp = int(ct)
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Determine entry phase based on first trade timestamp and token metadata
            entry_phase = self._determine_entry_phase(first_entry_timestamp, token_metadata, creation_timestamp)
            
            # Calculate token age at entry (for pattern detection)
            token_age_at_entry = self._calculate_token_age(first_entry_timestamp, token_metadata, mint, creation_timestamp)
            
            # Calculate price change before entry for FOMO/DIP detection
            # Uses trade price history from the API instead of external Birdeye calls
            price_change_before_entry = self._calculate_price_change_before_entry(
                mint,
                first_entry_timestamp,
                entry_price,
                price_history,  # Price history from trades API
                use_birdeye_fallback=use_birdeye_fallback
            )
            
            # Detect patterns from calculated price change and trade metadata
            # Use calculated price change if available, otherwise check metadata
            is_fomo_entry = (
                price_change_before_entry >= self.config.FOMO_PUMP_THRESHOLD or
                any(trade.get('is_fomo') or trade.get('pattern') == 'FOMO' for trade in trades)
            )
            is_dip_buy = (
                price_change_before_entry <= -self.config.DIP_DROP_THRESHOLD or
                any(trade.get('is_dip_buy') or trade.get('pattern') == 'DIP_BUY' for trade in trades)
            )
            is_whale_follow = any(
                trade.get('is_whale_follow') or 
                trade.get('whale_state') == 'Accumulation' 
                for trade in trades
            )
            
            # Get whale state from first trade if available
            whale_state_at_entry = trades[0].get('whale_state', 'UNKNOWN') if trades else 'UNKNOWN'
            
            # Log key metrics for first few trades for debugging
            if len(trades) <= 3:
                logger.debug(
                    f"MintPnL for {mint[:8]}...: entry_phase={entry_phase}, "
                    f"entry_ts={'set' if first_entry_timestamp else 'None'}, "
                    f"creation_ts={creation_timestamp}, token_age={token_age_at_entry:.1f}d, "
                    f"hold_time={avg_hold_time:.1f}h, pnl_pct={pnl_percent:.1f}%"
                )
            
            return MintPnL(
                mint=mint,
                total_trades=len(trades),
                total_bought_value=total_bought_value,
                total_sold_value=total_sold_value,
                realized_pnl=realized_pnl,
                unrealized_holding=0.0,  # Not available from trades API
                is_winner=is_winner,
                entry_timestamp=first_entry_timestamp,
                exit_timestamp=exit_timestamp,
                avg_hold_time_hours=avg_hold_time,
                entry_phase=entry_phase,
                pnl_percent=pnl_percent,
                price_change_before_entry=price_change_before_entry,
                whale_state_at_entry=whale_state_at_entry,
                is_fomo_entry=is_fomo_entry,
                is_dip_buy=is_dip_buy,
                is_whale_follow=is_whale_follow,
                token_age_at_entry_days=token_age_at_entry
            )
            
        except Exception as e:
            logger.error(f"Error creating MintPnL from trades: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _calculate_behaviors_from_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate behavioral metrics from trades list (new API format)
        
        Args:
            trades: List of trades from API
            
        Returns:
            Dictionary of behavioral metrics
        """
        if not trades:
            return {
                'avg_hold_time': 0.0,
                'is_sniper': False,
                'is_holder': False,
                'is_flipper': False,
                'fomo_tendency': 0.0,
                'panic_sell_tendency': 0.0,
                'dip_buy_tendency': 0.0,
                'whale_follow_tendency': 0.0
            }
        
        # Calculate average hold time from trades if available
        hold_times = []
        for trade in trades:
            hold_time = (
                trade.get('hold_time_hours') or 
                trade.get('holding_time_hours') or 
                trade.get('hold_duration_hours')
            )
            if hold_time is not None:
                try:
                    hold_times.append(float(hold_time))
                except (ValueError, TypeError):
                    pass
        
        avg_hold_time = np.mean(hold_times) if hold_times else 0.0
        
        # Trader type flags based on hold time
        is_flipper = avg_hold_time < self.config.FLIPPER_MAX_HOLD_HOURS if avg_hold_time > 0 else False
        is_holder = avg_hold_time > self.config.HOLDER_MIN_HOLD_HOURS if avg_hold_time > 0 else False
        
        # Sniper detection - check for early entries
        early_entries = sum(1 for t in trades if self._is_early_entry(t))
        is_sniper = (early_entries / len(trades)) >= self.config.SNIPER_THRESHOLD_PERCENT if trades else False
        
        # Count buy trades
        buy_trades = [t for t in trades if self._is_buy_trade(t)]
        total_buy_trades = len(buy_trades)
        
        # FOMO tendency
        fomo_trades = sum(1 for t in trades if t.get('is_fomo') or t.get('pattern') == 'FOMO')
        fomo_tendency = fomo_trades / total_buy_trades if total_buy_trades > 0 else 0.0
        
        # DIP_BUY tendency
        dip_buy_trades = sum(1 for t in trades if t.get('is_dip_buy') or t.get('pattern') == 'DIP_BUY')
        dip_buy_tendency = dip_buy_trades / total_buy_trades if total_buy_trades > 0 else 0.0
        
        # WHALE_FOLLOW tendency
        whale_follow_trades = sum(1 for t in trades if t.get('is_whale_follow') or t.get('whale_state') == 'Accumulation')
        whale_follow_tendency = whale_follow_trades / total_buy_trades if total_buy_trades > 0 else 0.0
        
        # Panic sell tendency
        panic_sells = sum(1 for t in trades if self._is_panic_sell(t))
        panic_sell_tendency = panic_sells / len(trades) if trades else 0.0
        
        return {
            'avg_hold_time': avg_hold_time,
            'is_sniper': is_sniper,
            'is_holder': is_holder,
            'is_flipper': is_flipper,
            'fomo_tendency': fomo_tendency,
            'panic_sell_tendency': panic_sell_tendency,
            'dip_buy_tendency': dip_buy_tendency,
            'whale_follow_tendency': whale_follow_tendency
        }
    
    def _is_buy_trade(self, trade: Dict) -> bool:
        """Check if trade is a buy"""
        return (
            trade.get('is_buy') or 
            trade.get('isBuy') or 
            trade.get('type', '').upper() == 'BUY' or
            trade.get('side', '').upper() == 'BUY'
        )
    
    def _is_early_entry(self, trade: Dict) -> bool:
        """Check if trade was an early/sniper entry"""
        # Check for explicit flag
        if trade.get('is_sniper') or trade.get('is_early_entry'):
            return True
        
        # Check token age at entry
        token_age = trade.get('token_age_at_entry') or trade.get('token_age_hours', 999)
        try:
            return float(token_age) <= (self.config.SNIPER_ENTRY_SECONDS / 3600)  # Convert seconds to hours
        except (ValueError, TypeError):
            return False
    
    def _is_panic_sell(self, trade: Dict) -> bool:
        """Check if trade was a panic sell"""
        # Check for explicit flag
        if trade.get('is_panic_sell') or trade.get('pattern') == 'PANIC_SELL':
            return True
        
        # Check for significant loss with short hold time
        try:
            pnl_percent = float(trade.get('pnl_percent', 0) or trade.get('profit_percent', 0) or 0)
            hold_hours = float(trade.get('hold_time_hours', 999) or trade.get('holding_time_hours', 999) or 999)
            
            return (pnl_percent <= self.config.PANIC_SELL_LOSS_THRESHOLD and 
                    hold_hours < self.config.PANIC_SELL_MAX_HOLD_HOURS)
        except (ValueError, TypeError):
            return False
    
    def _calculate_all_pnls(self, mints_data: List[Dict]) -> List[MintPnL]:
        """Calculate PnL for all mints traded by user"""
        mint_pnls = []
        
        for mint_data in mints_data:
            pnl = self._calculate_mint_pnl(mint_data)
            if pnl:
                mint_pnls.append(pnl)
        
        return mint_pnls
    
    def _calculate_mint_pnl(self, mint_data: Dict, token_metadata: Optional[Dict] = None) -> Optional[MintPnL]:
        """
        Calculate realized PnL for a single mint using FIFO method
        Enhanced with price context and entry phase detection
        
        Note: This method is for backward compatibility with old /mint/user-timeline API.
        The new /user-profiling/trades API uses _create_mint_pnl_from_trades() instead.
        
        Args:
            mint_data: Mint data from user timeline
            token_metadata: Optional token metadata with creation date
            
        Returns:
            MintPnL object or None
        """
        try:
            mint = mint_data.get('mint')
            history = mint_data.get('history', [])
            
            if not mint or not history:
                return None
            
            # Sort by timestamp
            history_sorted = sorted(history, key=lambda x: x.get('timestamp', ''))
            
            # Build price history as list of (timestamp, price) tuples
            price_history: List[Tuple[datetime, float]] = []
            for trade in history_sorted:
                timestamp_str = trade.get('timestamp')
                timestamp = self._parse_timestamp(timestamp_str)
                price = float(trade.get('price', 0) or 0)
                if timestamp and price > 0:
                    price_history.append((timestamp, price))
            
            # Track cost basis using FIFO
            cost_basis = 0.0
            tokens_held = 0.0
            total_bought_value = 0.0
            total_sold_value = 0.0
            realized_pnl = 0.0
            
            entry_timestamp = None
            first_entry_timestamp = None
            exit_timestamp = None
            hold_times = []
            entry_price = None
            
            for trade in history_sorted:
                try:
                    amount = float(trade.get('amount', 0))
                    price = float(trade.get('price', 0))
                    is_buy = trade.get('isBuy', False)
                    timestamp_str = trade.get('timestamp')
                    
                    timestamp = self._parse_timestamp(timestamp_str)
                    
                    if is_buy:
                        if entry_timestamp is None:
                            entry_timestamp = timestamp
                            entry_price = price
                        if first_entry_timestamp is None:
                            first_entry_timestamp = timestamp
                        
                        value = amount * price
                        cost_basis += value
                        tokens_held += amount
                        total_bought_value += value
                    else:
                        # SELL - Calculate PnL
                        value = amount * price
                        total_sold_value += value
                        
                        if tokens_held > 0:
                            avg_cost = cost_basis / tokens_held
                            pnl = (price - avg_cost) * amount
                            realized_pnl += pnl
                            
                            # Update cost basis
                            tokens_held -= amount
                            cost_basis = avg_cost * tokens_held if tokens_held > 0 else 0
                        
                        exit_timestamp = timestamp
                        
                        # Calculate hold time
                        if entry_timestamp and timestamp:
                            hold_time = (timestamp - entry_timestamp).total_seconds() / 3600  # hours
                            hold_times.append(hold_time)
                            entry_timestamp = None  # Reset for next position
                
                except Exception as e:
                    logger.debug(f"Error processing trade: {e}")
                    continue
            
            # Calculate average hold time
            avg_hold_time = np.mean(hold_times) if hold_times else 0.0
            
            # Calculate PnL percentage
            pnl_percent = (realized_pnl / total_bought_value * 100) if total_bought_value > 0 else 0.0
            
            # Determine if winner
            is_winner = realized_pnl > self.config.PNL_WIN_THRESHOLD
            
            # Get final holding
            final_holding = float(mint_data.get('finalHolding', 0))
            
            # Calculate entry phase based on token age
            entry_phase = self._determine_entry_phase(first_entry_timestamp, token_metadata)
            token_age_at_entry = self._calculate_token_age(first_entry_timestamp, token_metadata, mint)
            
            # Calculate price change before entry (for FOMO/DIP detection)
            # Uses price history from trade data
            price_change_before_entry = self._calculate_price_change_before_entry(
                mint,
                first_entry_timestamp,
                entry_price,
                price_history
            )
            
            # Determine pattern flags based on price context
            is_fomo_entry = price_change_before_entry >= self.config.FOMO_PUMP_THRESHOLD
            is_dip_buy = price_change_before_entry <= -self.config.DIP_DROP_THRESHOLD
            
            # Whale state at entry - get from entry_context if available
            whale_state_at_entry = mint_data.get('whale_state_at_entry', 'UNKNOWN')
            is_whale_follow = whale_state_at_entry == "Accumulation"
            
            return MintPnL(
                mint=mint,
                total_trades=mint_data.get('totalTrades', len(history)),
                total_bought_value=total_bought_value,
                total_sold_value=total_sold_value,
                realized_pnl=realized_pnl,
                unrealized_holding=final_holding,
                is_winner=is_winner,
                entry_timestamp=first_entry_timestamp,
                exit_timestamp=exit_timestamp,
                avg_hold_time_hours=avg_hold_time,
                entry_phase=entry_phase,
                pnl_percent=pnl_percent,
                price_change_before_entry=price_change_before_entry,
                whale_state_at_entry=whale_state_at_entry,
                is_fomo_entry=is_fomo_entry,
                is_dip_buy=is_dip_buy,
                is_whale_follow=is_whale_follow,
                token_age_at_entry_days=token_age_at_entry
            )
            
        except Exception as e:
            logger.error(f"Error calculating PnL for mint: {e}")
            return None
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime object"""
        if not timestamp_str:
            return None
        
        try:
            # Handle ISO format with 'Z' suffix (e.g., "2025-11-29T22:04:12.000Z")
            # This is the most common format from API
            if 'T' in str(timestamp_str):
                ts = str(timestamp_str)
                # Replace 'Z' with '+00:00' for fromisoformat compatibility
                if ts.endswith('Z'):
                    ts = ts[:-1] + '+00:00'
                try:
                    return datetime.fromisoformat(ts).replace(tzinfo=None)
                except ValueError:
                    pass
            
            # Try parsing with microseconds first
            try:
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                pass
            
            # Try without microseconds
            try:
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
            
            # Try with pandas for flexible parsing
            parsed = pd.to_datetime(timestamp_str, format='mixed', errors='coerce')
            if pd.isna(parsed):
                return None
            return parsed.to_pydatetime()
            
        except Exception:
            return None
    
    def _determine_entry_phase(
        self, 
        entry_timestamp: Optional[datetime], 
        token_metadata: Optional[Dict],
        creation_timestamp: Optional[int] = None
    ) -> str:
        """
        Determine entry phase based on token age when user entered
        
        Phase definitions:
        - P1: Token age 0-3 days (Launch/Snipers)
        - P2: Token age 4-14 days (Holder Expansion)
        - P3: Token age 15-45 days (Trend Stability)
        - P4: Token age 45+ days (Mature)
        
        Args:
            entry_timestamp: When user first bought
            token_metadata: Token metadata with creation date
            creation_timestamp: Optional Unix timestamp (seconds) from trade API
            
        Returns:
            Phase string (P1, P2, P3, P4, or UNKNOWN)
        """
        if not entry_timestamp:
            return "UNKNOWN"
        
        token_age_days = self._calculate_token_age(
            entry_timestamp, 
            token_metadata, 
            None, 
            creation_timestamp
        )
        
        if token_age_days < 0:
            return "UNKNOWN"
        elif token_age_days <= 3:
            return "P1"  # Launch/Snipers phase
        elif token_age_days <= 14:
            return "P2"  # Holder Expansion phase
        elif token_age_days <= 45:
            return "P3"  # Trend Stability phase
        else:
            return "P4"  # Mature phase
    
    def _determine_trade_phase(self, trade: Dict) -> str:
        """
        Determine phase for a single trade using its executed_at and creation_timestamp.
        
        Token age = executed_at - creation_timestamp  →  P1/P2/P3/P4/UNKNOWN
        """
        creation_ts = trade.get('creation_timestamp')
        if not creation_ts or creation_ts <= 0:
            return "UNKNOWN"
        
        executed_at_str = (
            trade.get('executed_at') or trade.get('timestamp') or
            trade.get('time') or trade.get('created_at')
        )
        if not executed_at_str:
            return "UNKNOWN"
        
        executed_at = self._parse_timestamp(executed_at_str)
        if not executed_at:
            return "UNKNOWN"
        
        try:
            token_created_at = datetime.fromtimestamp(int(creation_ts))
        except (ValueError, OSError, OverflowError):
            return "UNKNOWN"
        
        token_age_days = (executed_at - token_created_at).total_seconds() / 86400.0
        if token_age_days < 0:
            return "UNKNOWN"
        elif token_age_days <= 3:
            return "P1"
        elif token_age_days <= 14:
            return "P2"
        elif token_age_days <= 45:
            return "P3"
        else:
            return "P4"
    
    def _compute_typical_entry_phase(self, trades: List[Dict]) -> str:
        """
        Compute typical_entry_phase from per-trade phase counts (buy trades only).
        
        Each buy trade's phase is derived from (executed_at - creation_timestamp).
        Returns the phase with the most buy trades, reflecting "in which phase the
        wallet entered mostly."
        """
        phase_counts: Dict[str, int] = {}
        for trade in trades:
            is_buy = (
                trade.get('is_buy') or
                trade.get('isBuy') or
                trade.get('type', '').upper() == 'BUY' or
                trade.get('side', '').upper() == 'BUY'
            )
            if not is_buy:
                continue
            phase = self._determine_trade_phase(trade)
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        if not phase_counts:
            return "UNKNOWN"
        
        logger.info(f"Per-trade (buy) phase distribution: {phase_counts}")
        return max(phase_counts, key=phase_counts.get)
    
    def _calculate_token_age(
        self, 
        entry_timestamp: Optional[datetime], 
        token_metadata: Optional[Dict],
        mint: Optional[str] = None,
        creation_timestamp: Optional[int] = None
    ) -> float:
        """
        Calculate token age in days at the time of user's entry
        
        Priority for token creation date:
        1. creation_timestamp from trade API (Unix timestamp in seconds)
        2. token_metadata from cache
        3. Default fallback (assumes token created X days before entry)
        
        Args:
            entry_timestamp: When user first bought
            token_metadata: Token metadata with creation date (from cache)
            mint: Optional mint address (for logging only)
            creation_timestamp: Optional Unix timestamp (seconds) from trade API
            
        Returns:
            Token age in days
        """
        if not entry_timestamp:
            return self.config.DEFAULT_TOKEN_AGE_DAYS
        
        token_created_at = None
        source = "default"
        
        # Priority 1: Use creation_timestamp from trade API (Unix timestamp in seconds)
        if creation_timestamp and creation_timestamp > 0:
            try:
                token_created_at = datetime.fromtimestamp(creation_timestamp)
                source = "trade_api"
            except (ValueError, OSError, OverflowError) as e:
                logger.debug(f"Invalid creation_timestamp {creation_timestamp}: {e}")
        
        # Priority 2: Try to get token creation date from metadata (from cache only - no API calls)
        if not token_created_at and token_metadata:
            # Birdeye blockUnixTime is an int (Unix seconds) — handle before string keys
            block_unix = token_metadata.get('blockUnixTime')
            if block_unix and isinstance(block_unix, (int, float)) and block_unix > 0:
                try:
                    token_created_at = datetime.fromtimestamp(int(block_unix))
                    source = "birdeye_creation_info"
                except (ValueError, OSError, OverflowError):
                    pass

            if not token_created_at:
                created_str = (
                    token_metadata.get('CreatedAt') or
                    token_metadata.get('created_at') or
                    token_metadata.get('createdAt') or
                    token_metadata.get('blockHumanTime') or
                    token_metadata.get('launch_date') or
                    token_metadata.get('mint_time')
                )

                if created_str:
                    token_created_at = self._parse_timestamp(str(created_str))
                    if token_created_at:
                        source = "metadata_cache"
        
        if token_created_at:
            # Calculate age in days from actual creation date
            age_delta = entry_timestamp - token_created_at
            calculated_age = max(0.0, age_delta.total_seconds() / (24 * 3600))
            logger.debug(f"Token age for {mint[:8] if mint else 'unknown'}...: {calculated_age:.1f} days (source: {source})")
            return calculated_age
        
        # Priority 3: No creation date available - use smart default
        # Assume token was created a few days before user's first trade
        default_days_before = self.config.DEFAULT_TOKEN_CREATION_DAYS_BEFORE_ENTRY
        estimated_creation_date = entry_timestamp - timedelta(days=default_days_before)
        age_delta = entry_timestamp - estimated_creation_date
        calculated_age = max(0.0, age_delta.total_seconds() / (24 * 3600))
        
        logger.debug(f"Using default token age for {mint[:8] if mint else 'unknown'}...: {calculated_age:.1f} days (assumed created {default_days_before} days before entry)")
        return calculated_age
    
    def _calculate_price_change_before_entry(
        self,
        mint: str,
        entry_timestamp: Optional[datetime],
        entry_price: Optional[float],
        price_history: Optional[List[Tuple[datetime, float]]] = None,
        use_birdeye_fallback: bool = False
    ) -> float:
        """
        Calculate price change in the lookback period before user's entry
        Used for FOMO and DIP_BUY detection
        
        Strategy:
        1. Use price data from the user's prior trades (from /user-profiling/trades API)
        2. If use_birdeye_fallback=True and no prior trades, try Birdeye API
        
        Note: Birdeye fallback is disabled by default during bulk profile building
        to avoid thousands of API calls. Enable it for single-trade analysis.
        
        Args:
            mint: Token mint address
            entry_timestamp: When user first bought
            entry_price: Price at entry (from trades API)
            price_history: List of (timestamp, price) tuples from trades API for this mint
            use_birdeye_fallback: Whether to call Birdeye if no trade history (default: False)
            
        Returns:
            Price change as decimal (e.g., 0.20 = 20% increase, -0.10 = 10% decrease)
        """
        if not entry_timestamp or not entry_price or entry_price <= 0:
            return 0.0
        
        if not mint:
            return 0.0
        
        lookback_hours = self.config.PRICE_LOOKBACK_HOURS
        lookback_time = entry_timestamp - timedelta(hours=lookback_hours)
        
        price_before = None
        price_before_timestamp = None
        source = "none"
        
        # PRIMARY: Use price history from trades API (already fetched, no API calls)
        if price_history:
            for timestamp, price in price_history:
                if timestamp and timestamp < entry_timestamp and price > 0:
                    # We want the most recent price that's at least lookback_hours old
                    if timestamp <= lookback_time:
                        # This price is from before/at the lookback window start
                        if price_before is None or timestamp > price_before_timestamp:
                            price_before = price
                            price_before_timestamp = timestamp
                            source = "trade_history"
                    elif price_before is None:
                        # If we don't have any price from before lookback, 
                        # use the earliest available price before entry
                        price_before = price
                        price_before_timestamp = timestamp
                        source = "trade_history"
        
        # OPTIONAL FALLBACK: Birdeye API (only when explicitly enabled)
        # Disabled by default to prevent thousands of API calls during bulk profile building
        if price_before is None and use_birdeye_fallback and self.birdeye_fetcher:
            try:
                if self.birdeye_fetcher.is_configured():
                    birdeye_price = self.birdeye_fetcher.fetch_historical_price(mint, lookback_time)
                    if birdeye_price and birdeye_price > 0:
                        price_before = birdeye_price
                        price_before_timestamp = lookback_time
                        source = "birdeye_api"
                        sentry_fallback_warning(
                            "birdeye",
                            f"No trade-history price available — using Birdeye historical price for {mint[:8]}...",
                            {"mint": mint[:8], "birdeye_price": birdeye_price},
                        )
                        logger.debug(f"Got historical price from Birdeye for {mint[:8]}...: ${birdeye_price:.8f}")
            except Exception as e:
                logger.debug(f"Birdeye historical price fetch failed for {mint[:8]}...: {e}")
        
        # Calculate price change if we have a reference price
        if price_before and price_before > 0:
            price_change = (entry_price - price_before) / price_before
            time_diff_hours = (entry_timestamp - price_before_timestamp).total_seconds() / 3600 if price_before_timestamp else lookback_hours
            logger.debug(f"Price change before entry for {mint[:8]}...: {price_change:.2%} "
                        f"(entry: ${entry_price:.8f}, {time_diff_hours:.1f}h ago: ${price_before:.8f}, source: {source})")
            return price_change
        
        # No price data available
        return 0.0
    
    def _identify_patterns(self, mint_pnls: List[MintPnL]) -> List[TradePattern]:
        """
        Identify recurring patterns in user's trading behavior
        
        Args:
            mint_pnls: List of PnL calculations per mint
            
        Returns:
            List of identified patterns
        """
        patterns = []
        pattern_trades: Dict[PatternType, List[MintPnL]] = {}
        
        # Track entry phase distribution for debugging
        entry_phase_counts = {}
        
        # Classify each trade into pattern(s)
        for pnl in mint_pnls:
            # Track entry phases for debugging
            entry_phase_counts[pnl.entry_phase] = entry_phase_counts.get(pnl.entry_phase, 0) + 1
            
            trade_patterns = self._classify_trade_patterns(pnl)
            
            for pattern_type in trade_patterns:
                if pattern_type not in pattern_trades:
                    pattern_trades[pattern_type] = []
                pattern_trades[pattern_type].append(pnl)
        
        # Log pattern distribution before filtering
        logger.info(f"Entry phase distribution: {entry_phase_counts}")
        logger.info(f"Pattern counts before filtering (min {self.config.MIN_TRADES_FOR_PATTERN}): "
                   f"{ {pt.value: len(trades) for pt, trades in pattern_trades.items()} }")
        
        # Create pattern objects for patterns with enough occurrences
        for pattern_type, trades in pattern_trades.items():
            if len(trades) >= self.config.MIN_TRADES_FOR_PATTERN:
                wins = sum(1 for t in trades if t.is_winner)
                losses = len(trades) - wins
                win_rate = wins / len(trades)
                total_pnl = sum(t.realized_pnl for t in trades)
                avg_pnl_percent = np.mean([t.pnl_percent for t in trades])
                
                pattern = TradePattern(
                    pattern_id=f"{pattern_type.value}_{len(patterns)}",
                    pattern_type=pattern_type,
                    occurrences=len(trades),
                    wins=wins,
                    losses=losses,
                    win_rate=win_rate,
                    avg_pnl_percent=avg_pnl_percent,
                    total_pnl=total_pnl,
                    typical_context=self._calculate_typical_context(trades),
                    sample_mints=[t.mint[:8] for t in trades[:5]]
                )
                patterns.append(pattern)
        
        # Sort by occurrences
        patterns.sort(key=lambda p: p.occurrences, reverse=True)
        
        return patterns
    
    def _classify_trade_patterns(self, pnl: MintPnL) -> List[PatternType]:
        """
        Classify a single trade into pattern type(s)
        
        Args:
            pnl: MintPnL for a single mint
            
        Returns:
            List of pattern types this trade matches
        """
        patterns = []
        
        # Entry phase patterns
        if pnl.entry_phase in ["P1", "P2"]:
            patterns.append(PatternType.EARLY_ENTRY)
        elif pnl.entry_phase == "P4":
            patterns.append(PatternType.LATE_ENTRY)
        
        # Hold time patterns
        if pnl.avg_hold_time_hours > 0:
            if pnl.avg_hold_time_hours < self.config.FLIPPER_MAX_HOLD_HOURS:
                patterns.append(PatternType.QUICK_FLIP)
            elif pnl.avg_hold_time_hours > self.config.HOLDER_MIN_HOLD_HOURS:
                patterns.append(PatternType.LONG_HOLD)
            else:
                # Between FLIPPER_MAX_HOLD_HOURS (24h) and HOLDER_MIN_HOLD_HOURS (168h)
                patterns.append(PatternType.MEDIUM_HOLD)
        
        # PnL-based patterns
        if (pnl.pnl_percent <= self.config.PANIC_SELL_LOSS_THRESHOLD and 
            pnl.avg_hold_time_hours < self.config.PANIC_SELL_MAX_HOLD_HOURS):
            patterns.append(PatternType.PANIC_SELL)
        
        # FOMO Detection - User bought after significant price pump
        # price_change_before_entry > FOMO_PUMP_THRESHOLD (default 20%)
        if pnl.is_fomo_entry or pnl.price_change_before_entry >= self.config.FOMO_PUMP_THRESHOLD:
            patterns.append(PatternType.FOMO)
            logger.debug(f"FOMO pattern detected for {pnl.mint[:8]}... (price change: {pnl.price_change_before_entry:.2%})")
        
        # DIP_BUY Detection - User bought after significant price drop
        # price_change_before_entry < -DIP_DROP_THRESHOLD (default -10%)
        if pnl.is_dip_buy or pnl.price_change_before_entry <= -self.config.DIP_DROP_THRESHOLD:
            patterns.append(PatternType.DIP_BUY)
            logger.debug(f"DIP_BUY pattern detected for {pnl.mint[:8]}... (price change: {pnl.price_change_before_entry:.2%})")
        
        # WHALE_FOLLOW Detection - User bought when whales were accumulating
        if pnl.is_whale_follow or pnl.whale_state_at_entry == "Accumulation":
            patterns.append(PatternType.WHALE_FOLLOW)
            logger.debug(f"WHALE_FOLLOW pattern detected for {pnl.mint[:8]}... (whale state: {pnl.whale_state_at_entry})")
        
        # Concentration-based patterns (from entry_context if available)
        top10_concentration = pnl.entry_context.get('top10_concentration', 50.0)
        if top10_concentration > 60:
            patterns.append(PatternType.HIGH_CONCENTRATION)
        elif top10_concentration < 30:
            patterns.append(PatternType.LOW_CONCENTRATION)
        
        # If no patterns identified, mark as unknown
        if not patterns:
            patterns.append(PatternType.UNKNOWN)
        
        return patterns
    
    def _calculate_typical_context(self, trades: List[MintPnL]) -> Dict[str, Any]:
        """Calculate the typical market context for a set of trades"""
        # Aggregate context from trades
        phases = [t.entry_phase for t in trades if t.entry_phase != "UNKNOWN"]
        hold_times = [t.avg_hold_time_hours for t in trades if t.avg_hold_time_hours > 0]
        
        return {
            'typical_phase': max(set(phases), key=phases.count) if phases else "UNKNOWN",
            'avg_hold_time': np.mean(hold_times) if hold_times else 0.0,
            'concentration': 50.0,  # Default, would need actual context
            'whale_state': "UNKNOWN",
            'trend': "UNKNOWN",
            'rsi_signal': "neutral"
        }
    
    def _calculate_behaviors(self, mint_pnls: List[MintPnL]) -> Dict[str, Any]:
        """
        Calculate behavioral metrics from trading history
        
        Args:
            mint_pnls: List of PnL calculations
            
        Returns:
            Dictionary of behavioral metrics
        """
        if not mint_pnls:
            return {
                'avg_hold_time': 0.0,
                'is_sniper': False,
                'is_holder': False,
                'is_flipper': False,
                'fomo_tendency': 0.0,
                'panic_sell_tendency': 0.0,
                'dip_buy_tendency': 0.0,
                'whale_follow_tendency': 0.0
            }
        
        # Average hold time
        hold_times = [p.avg_hold_time_hours for p in mint_pnls if p.avg_hold_time_hours > 0]
        avg_hold_time = np.mean(hold_times) if hold_times else 0.0
        
        # Trader type flags
        is_flipper = avg_hold_time < self.config.FLIPPER_MAX_HOLD_HOURS if avg_hold_time > 0 else False
        is_holder = avg_hold_time > self.config.HOLDER_MIN_HOLD_HOURS if avg_hold_time > 0 else False
        
        # Sniper detection - based on token age at entry
        sniper_entries = sum(1 for p in mint_pnls if p.token_age_at_entry_days <= 0.01)  # ~15 minutes
        is_sniper = (sniper_entries / len(mint_pnls)) >= self.config.SNIPER_THRESHOLD_PERCENT if mint_pnls else False
        
        # Count buy trades (entries)
        total_buy_trades = len(mint_pnls)  # Each MintPnL represents at least one entry
        
        # FOMO tendency - count trades where user bought after price pump
        fomo_trades = sum(1 for p in mint_pnls if p.is_fomo_entry or p.price_change_before_entry >= self.config.FOMO_PUMP_THRESHOLD)
        fomo_tendency = fomo_trades / total_buy_trades if total_buy_trades > 0 else 0.0
        
        # DIP_BUY tendency - count trades where user bought after price drop
        dip_buy_trades = sum(1 for p in mint_pnls if p.is_dip_buy or p.price_change_before_entry <= -self.config.DIP_DROP_THRESHOLD)
        dip_buy_tendency = dip_buy_trades / total_buy_trades if total_buy_trades > 0 else 0.0
        
        # WHALE_FOLLOW tendency - count trades where user bought during whale accumulation
        whale_follow_trades = sum(1 for p in mint_pnls if p.is_whale_follow or p.whale_state_at_entry == "Accumulation")
        whale_follow_tendency = whale_follow_trades / total_buy_trades if total_buy_trades > 0 else 0.0
        
        # Panic sell tendency
        panic_sells = sum(1 for p in mint_pnls 
                         if (p.pnl_percent <= self.config.PANIC_SELL_LOSS_THRESHOLD and 
                             p.avg_hold_time_hours < self.config.PANIC_SELL_MAX_HOLD_HOURS))
        panic_sell_tendency = panic_sells / len(mint_pnls) if mint_pnls else 0.0
        
        logger.debug(f"Behavioral metrics: fomo={fomo_tendency:.2f}, dip_buy={dip_buy_tendency:.2f}, "
                    f"whale_follow={whale_follow_tendency:.2f}, panic_sell={panic_sell_tendency:.2f}")
        
        return {
            'avg_hold_time': avg_hold_time,
            'is_sniper': is_sniper,
            'is_holder': is_holder,
            'is_flipper': is_flipper,
            'fomo_tendency': fomo_tendency,
            'panic_sell_tendency': panic_sell_tendency,
            'dip_buy_tendency': dip_buy_tendency,
            'whale_follow_tendency': whale_follow_tendency
        }
    
    def _determine_trader_type(self, behaviors: Dict[str, Any]) -> TraderType:
        """Determine the primary trader type from behaviors"""
        if behaviors['is_sniper']:
            return TraderType.SNIPER
        elif behaviors['is_flipper']:
            return TraderType.FLIPPER
        elif behaviors['is_holder']:
            return TraderType.HOLDER
        else:
            return TraderType.MIXED
    
    def _calculate_confidence(self, total_trades: int, total_mints: int, 
                             num_patterns: int) -> float:
        """
        Calculate confidence level in the profile based on data quantity
        
        Returns value between 0 and 1
        """
        # Base confidence on trade count
        trade_confidence = min(1.0, total_trades / 50)  # Max confidence at 50+ trades
        
        # Bonus for mint diversity
        mint_confidence = min(1.0, total_mints / 20)  # Max at 20+ mints
        
        # Bonus for pattern identification
        pattern_confidence = min(1.0, num_patterns / self.config.CONFIDENCE_MAX_PATTERNS)
        
        # Weighted combination using configurable weights
        confidence = (
            trade_confidence * self.config.CONFIDENCE_TRADE_WEIGHT +
            mint_confidence * self.config.CONFIDENCE_MINT_WEIGHT +
            pattern_confidence * self.config.CONFIDENCE_PATTERN_WEIGHT
        )
        confidence = min(self.config.CONFIDENCE_MAX, max(self.config.CONFIDENCE_MIN, confidence))
        # Low-confidence band: below ~50 trades, cap so we treat as low confidence
        low_cap = getattr(self.config, 'BEHAVIOUR_LOW_CONFIDENCE_MAX_TRADES', 50)
        if total_trades < low_cap:
            confidence = min(confidence, 0.5)
        return confidence
    
    def _create_empty_profile(self, wallet_address: str) -> UserProfile:
        """Create an empty profile for users with insufficient data"""
        return UserProfile(
            wallet_address=wallet_address,
            total_trades=0,
            total_mints_traded=0,
            overall_win_rate=0.0,
            avg_pnl_per_trade=0.0,
            total_realized_pnl=0.0,
            # New API-provided fields
            lifetime_pnl=0.0,
            total_volume_usd=0.0,
            max_drawdown_pct=0.0,
            avg_r_multiple=0.0,
            avg_holding_time_minutes=0.0,
            is_valid=False,
            confidence=0.0
        )
    
    def _get_from_cache(self, wallet_address: str) -> Optional[UserProfile]:
        """Get profile from cache if not expired"""
        if wallet_address not in self.profile_cache:
            return None
        
        profile, cached_at = self.profile_cache[wallet_address]
        
        # Check if cache is expired
        if cache_config.CACHE_USER_PROFILE:
            cache_age = (datetime.now() - cached_at).total_seconds()
            if cache_age > cache_config.USER_PROFILE_TTL:
                del self.profile_cache[wallet_address]
                return None
        
        return profile
    
    def _add_to_cache(self, wallet_address: str, profile: UserProfile):
        """Add profile to cache"""
        self.profile_cache[wallet_address] = (profile, datetime.now())


# ==================== HELPER FUNCTIONS ====================

PATTERN_DESCRIPTIONS: Dict[str, str] = {
    "EARLY_ENTRY": "You often enter tokens in their earliest phases (P1/P2), catching momentum early.",
    "LATE_ENTRY": "You tend to enter tokens in later phases (P4), after most early gains have passed.",
    "DIP_BUY": "You frequently buy after price dips, attempting to catch reversals.",
    "FOMO": "You tend to buy after significant price pumps, chasing momentum.",
    "QUICK_FLIP": "You typically hold positions for less than 24 hours, taking quick profits or losses.",
    "MEDIUM_HOLD": "You hold positions for 1–7 days, balancing momentum and conviction.",
    "LONG_HOLD": "You hold positions for over 7 days, showing patience and conviction.",
    "PANIC_SELL": "You tend to sell at a significant loss quickly, often triggered by sharp drops.",
    "HIGH_CONCENTRATION": "You enter tokens with high holder concentration (top 10 > 60%).",
    "LOW_CONCENTRATION": "You enter tokens with low holder concentration (top 10 < 30%).",
    "WHALE_FOLLOW": "Your entries often align with whale accumulation activity.",
    "UNKNOWN": "Unclassified trading pattern.",
}


def get_dominant_patterns(patterns: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Return the top-N patterns by occurrences with plain-English descriptions."""
    sorted_pats = sorted(patterns, key=lambda p: p.get('occurrences', 0), reverse=True)
    result = []
    for p in sorted_pats[:top_n]:
        pt = p.get('pattern_type', 'UNKNOWN')
        if pt == 'UNKNOWN':
            continue
        result.append({
            'pattern_type': pt,
            'occurrences': p.get('occurrences', 0),
            'win_rate': p.get('win_rate', 0),
            'avg_pnl_percent': p.get('avg_pnl_percent', 0),
            'description': PATTERN_DESCRIPTIONS.get(pt, PATTERN_DESCRIPTIONS['UNKNOWN']),
        })
    return result


def profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
    """Convert UserProfile to dictionary for JSON serialization"""
    return {
        'wallet_address': profile.wallet_address,
        'total_trades': profile.total_trades,
        'total_mints_traded': profile.total_mints_traded,
        'overall_win_rate': round(profile.overall_win_rate, 4),
        'avg_pnl_per_trade': round(profile.avg_pnl_per_trade, 6),
        'total_realized_pnl': round(profile.total_realized_pnl, 6),
        # New API-provided fields
        'lifetime_pnl': round(profile.lifetime_pnl, 2),
        'total_volume_usd': round(profile.total_volume_usd, 2),
        'max_drawdown_pct': round(profile.max_drawdown_pct, 4),
        'avg_r_multiple': round(profile.avg_r_multiple, 4),
        'avg_holding_time_minutes': round(profile.avg_holding_time_minutes, 2),
        # Behavioral metrics
        'avg_hold_time_hours': round(profile.avg_hold_time_hours, 2),
        'typical_entry_phase': profile.typical_entry_phase,
        'trader_type': profile.trader_type.value,
        'is_sniper': profile.is_sniper,
        'is_holder': profile.is_holder,
        'is_flipper': profile.is_flipper,
        'fomo_tendency': round(profile.fomo_tendency, 4) if profile.fomo_tendency is not None else None,
        'panic_sell_tendency': round(profile.panic_sell_tendency, 4) if profile.panic_sell_tendency is not None else None,
        'dip_buy_tendency': round(profile.dip_buy_tendency, 4) if profile.dip_buy_tendency is not None else None,
        'whale_follow_tendency': round(profile.whale_follow_tendency, 4) if profile.whale_follow_tendency is not None else None,
        'behavioral_biases_status': 'not_computed' if (
            profile.fomo_tendency is None and profile.panic_sell_tendency is None and
            profile.dip_buy_tendency is None and profile.whale_follow_tendency is None
        ) else 'computed',
        'behavioral_biases_message': 'Not enough data to infer biases' if (
            profile.fomo_tendency is None and profile.panic_sell_tendency is None and
            profile.dip_buy_tendency is None and profile.whale_follow_tendency is None
        ) else None,
        'confidence': round(profile.confidence, 4),
        'is_valid': profile.is_valid,
        'last_updated': profile.last_updated.isoformat(),
        'patterns': [
            {
                'pattern_type': p.pattern_type.value,
                'occurrences': p.occurrences,
                'win_rate': round(p.win_rate, 4),
                'avg_pnl_percent': round(p.avg_pnl_percent, 2)
            }
            for p in profile.patterns
        ],
        'best_pattern': {
            'pattern_type': profile.best_pattern.pattern_type.value,
            'win_rate': round(profile.best_pattern.win_rate, 4),
            'occurrences': profile.best_pattern.occurrences
        } if profile.best_pattern else None,
        'worst_pattern': {
            'pattern_type': profile.worst_pattern.pattern_type.value,
            'win_rate': round(profile.worst_pattern.win_rate, 4),
            'occurrences': profile.worst_pattern.occurrences
        } if profile.worst_pattern else None
    }


if __name__ == "__main__":
    print("User Profiler Engine")
    print("=" * 60)
    
    # Test with mock data
    profiler = UserProfiler()
    
    # Create a mock profile for testing
    test_profile = UserProfile(
        wallet_address="TestWallet123456789012345678901234567890",
        total_trades=25,
        total_mints_traded=10,
        overall_win_rate=0.65,
        avg_pnl_per_trade=0.05,
        total_realized_pnl=0.5,
        avg_hold_time_hours=12.5,
        typical_entry_phase="P2",
        trader_type=TraderType.FLIPPER,
        is_flipper=True,
        fomo_tendency=0.3,
        panic_sell_tendency=0.2,
        confidence=0.75,
        is_valid=True
    )
    
    print(f"\nTest Profile:")
    profile_dict = profile_to_dict(test_profile)
    for key, value in profile_dict.items():
        if key != 'patterns':
            print(f"  {key}: {value}")
    
    print("\nUser Profiler initialized successfully!")

