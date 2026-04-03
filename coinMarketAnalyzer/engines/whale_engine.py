"""
Whale Engine v2.0
Enhanced whale behavior analysis with time-series delta calculation
Implements the document requirements for accurate whale tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import whale_config, wallet_classification_config, get_phase_from_age, get_noise_threshold, get_whale_percentile
from core.circuit_breaker import sentry_fallback_warning
from engines.wallet_classifier import WalletClassifier, WalletInfo, WalletType

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class WhaleMetrics:
    """Comprehensive whale metrics for a token"""
    # Whale Buy/Sell Volume (in token units)
    whale_buy_volume: float  # Total tokens bought by whales
    whale_sell_volume: float  # Total tokens sold by whales
    whale_net_volume: float  # Net volume (buys - sells, positive = accumulation)
    
    # Concentration metrics
    gini_coefficient: float
    top10_hold_percent: float
    
    # Special wallet holdings
    dev_hold_percent: float
    sniper_hold_percent: float
    
    # Whale state
    whale_state: str  # "Accumulation", "Distribution", "Stability"
    whale_count: int
    confirmed_whale_count: int
    
    # Phase info
    phase: str
    
    # Additional context
    total_holders: int
    active_holders: int
    holder_growth_24h: float  # Percentage change in holders
    
    # Data quality flags
    is_whale_data_stale: bool = False  # True if no recent whale activity (using lifetime data as fallback)
    whale_data_source: str = "recent"  # "recent" = activity in lookback window, "lifetime" = using lifetime totals
    
    # ==================== DOMINANT WHALE INACTIVITY DETECTION ====================
    # These fields track whether the TOP holders (by holdings %) have been active recently
    # This helps detect scenarios where whale activity signals are misleading
    dominant_whale_count: int = 0  # Number of wallets classified as "dominant whales" (top holders)
    dominant_whale_inactive_count: int = 0  # How many of those dominant whales are inactive (5+ days)
    dominant_whale_aging_count: int = 0  # How many dominant whales are "aging" (3-5 days inactive)
    dominant_whale_inactive_holding_pct: float = 0.0  # % of supply held by inactive dominant whales
    dominant_whale_aging_holding_pct: float = 0.0  # % of supply held by aging dominant whales
    top_holder_last_activity_hours: Optional[float] = None  # Hours since the largest holder's last trade
    dominant_whale_status: str = "UNKNOWN"  # "ACTIVE", "AGING", "PARTIALLY_INACTIVE", "FULLY_INACTIVE", "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        """Convert WhaleMetrics to dictionary for JSON serialization"""
        return {
            'whale_buy_volume': self.whale_buy_volume,
            'whale_sell_volume': self.whale_sell_volume,
            'whale_net_volume': self.whale_net_volume,
            'gini_coefficient': self.gini_coefficient,
            'top10_hold_percent': self.top10_hold_percent,
            'dev_hold_percent': self.dev_hold_percent,
            'sniper_hold_percent': self.sniper_hold_percent,
            'whale_state': self.whale_state,
            'whale_count': self.whale_count,
            'confirmed_whale_count': self.confirmed_whale_count,
            'phase': self.phase,
            'total_holders': self.total_holders,
            'active_holders': self.active_holders,
            'holder_growth_24h': self.holder_growth_24h,
            'is_whale_data_stale': self.is_whale_data_stale,
            'whale_data_source': self.whale_data_source,
            # Dominant whale inactivity fields
            'dominant_whale_count': self.dominant_whale_count,
            'dominant_whale_inactive_count': self.dominant_whale_inactive_count,
            'dominant_whale_aging_count': self.dominant_whale_aging_count,
            'dominant_whale_inactive_holding_pct': self.dominant_whale_inactive_holding_pct,
            'dominant_whale_aging_holding_pct': self.dominant_whale_aging_holding_pct,
            'top_holder_last_activity_hours': self.top_holder_last_activity_hours,
            'dominant_whale_status': self.dominant_whale_status
        }


class WhaleEngine:
    """
    Enhanced Whale Engine with time-series tracking
    """
    
    def __init__(self):
        """Initialize Whale Engine v2"""
        self.wallet_classifier = WalletClassifier()
        logger.info("Whale Engine initialized")
    
    def create_empty_metrics(self) -> WhaleMetrics:
        """
        Create empty/default WhaleMetrics for tokens without holder data
        Used for perps tokens where on-chain holder data is not available
        
        Returns:
            WhaleMetrics with default/neutral values
        """
        return WhaleMetrics(
            whale_buy_volume=0.0,
            whale_sell_volume=0.0,
            whale_net_volume=0.0,
            gini_coefficient=0.5,  # Neutral
            top10_hold_percent=0.0,
            dev_hold_percent=0.0,
            sniper_hold_percent=0.0,
            whale_state="UNKNOWN",
            whale_count=0,
            confirmed_whale_count=0,
            phase="N/A",
            total_holders=0,
            active_holders=0,
            holder_growth_24h=0.0,
            is_whale_data_stale=True,
            whale_data_source="no_data",
            dominant_whale_count=0,
            dominant_whale_inactive_count=0,
            dominant_whale_aging_count=0,
            dominant_whale_inactive_holding_pct=0.0,
            dominant_whale_aging_holding_pct=0.0,
            top_holder_last_activity_hours=None,
            dominant_whale_status="NOT_AVAILABLE"
        )

    def create_whale_metrics_from_birdeye(
        self,
        trade_data: Dict[str, Any],
        token_holders_df: pd.DataFrame,
        total_supply: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WhaleMetrics:
        """
        Build WhaleMetrics from Birdeye aggregate data (no time-series).
        Used when meme prediction uses Birdeye as primary data source.

        trade_data: from Birdeye fetch_trade_data (buy24h, sell24h, holders, etc.)
        token_holders_df: DataFrame with lastHolding or balance column (from Birdeye fetch_token_holders)
        total_supply: Total token supply (from metadata or market_data)
        metadata: Optional token metadata for phase
        """
        buy24h = float(trade_data.get("buy24h", 0) or 0)
        sell24h = float(trade_data.get("sell24h", 0) or 0)
        holders_count = int(trade_data.get("holders", 0) or 0)

        whale_net_volume = buy24h - sell24h
        total_vol = buy24h + sell24h
        has_trade_data = total_vol > 0
        if has_trade_data:
            buy_ratio = buy24h / total_vol
            if buy_ratio >= whale_config.ACCUMULATION_THRESHOLD:
                whale_state = "Accumulation"
            elif buy_ratio <= whale_config.DISTRIBUTION_THRESHOLD:
                whale_state = "Distribution"
            else:
                whale_state = "Stability"
        else:
            whale_state = "Stability"

        gini = whale_config.GINI_NEUTRAL_DEFAULT
        top10_pct = 0.0
        if len(token_holders_df) > 0:
            holding_col = None
            for col in ["lastHolding", "current_holding", "balance"]:
                if col in token_holders_df.columns:
                    holding_col = col
                    break
            if holding_col is not None:
                holdings = pd.to_numeric(token_holders_df[holding_col], errors="coerce").fillna(0).values
                holdings = holdings[holdings > 0]
                if len(holdings) > 0:
                    gini = self._calculate_gini_coefficient(holdings)
                    supply_for_top10 = total_supply if total_supply > 0 else float(holdings.sum())
                    df_for_top10 = token_holders_df.copy()
                    if holding_col != "lastHolding":
                        df_for_top10["lastHolding"] = pd.to_numeric(df_for_top10[holding_col], errors="coerce").fillna(0)
                    top10_pct = self._calculate_top10_concentration(df_for_top10, supply_for_top10)

        phase = self._determine_phase(metadata) if metadata else "P4"
        active_holders = len(token_holders_df) if not token_holders_df.empty else holders_count

        return WhaleMetrics(
            whale_buy_volume=buy24h,
            whale_sell_volume=sell24h,
            whale_net_volume=whale_net_volume,
            gini_coefficient=gini,
            top10_hold_percent=top10_pct,
            dev_hold_percent=0.0,
            sniper_hold_percent=0.0,
            whale_state=whale_state,
            whale_count=0,
            confirmed_whale_count=0,
            phase=phase,
            total_holders=holders_count or active_holders,
            active_holders=active_holders,
            holder_growth_24h=0.0,
            is_whale_data_stale=not has_trade_data,
            whale_data_source="birdeye_24h" if has_trade_data else "no_data",
            dominant_whale_count=0,
            dominant_whale_inactive_count=0,
            dominant_whale_aging_count=0,
            dominant_whale_inactive_holding_pct=0.0,
            dominant_whale_aging_holding_pct=0.0,
            top_holder_last_activity_hours=None,
            dominant_whale_status="NOT_AVAILABLE",
        )

    def analyze_token(
        self,
        user_holdings_df: pd.DataFrame,
        token_metadata: Optional[Dict[str, Any]],
        total_supply: float,
        lookback_hours: Optional[int] = None
    ) -> WhaleMetrics:
        """
        Perform comprehensive whale analysis for a token
        
        Args:
            user_holdings_df: DataFrame with user holdings and transaction history
            token_metadata: Token metadata including creation date and developer
            total_supply: Total token supply
            lookback_hours: Hours to look back for delta calculation (default: from config, 10 days)
            
        Returns:
            WhaleMetrics object with all calculated metrics
        """
        # Use config value if not specified (default: 240 hours = 10 days)
        if lookback_hours is None:
            lookback_hours = whale_config.WHALE_DELTA_WINDOW_HOURS
        logger.info("Starting comprehensive whale analysis...")
        logger.info(f"Input: {len(user_holdings_df)} rows, columns: {list(user_holdings_df.columns)}")
        
        # Determine token phase
        phase = self._determine_phase(token_metadata)
        logger.info(f"Token phase: {phase}")
        
        # Find the holding column (try multiple names)
        holding_col = None
        possible_cols = ['current_holding', 'lastHolding', 'finalHolding', 'balance', 'holding']
        for col in possible_cols:
            if col in user_holdings_df.columns:
                holding_col = col
                break
        
        if holding_col is None:
            logger.error(f"No holding column found. Available: {list(user_holdings_df.columns)}")
            # Create a default empty holdings array
            holdings = np.array([])
            positive_holdings = np.array([])
            total_holders_in_data = len(user_holdings_df)
        else:
            # Convert holdings to numeric (in case they're strings)
            raw_values = user_holdings_df[holding_col]
            holdings = pd.to_numeric(raw_values, errors='coerce').fillna(0).values
            positive_holdings = holdings[holdings > 0]
            total_holders_in_data = len(user_holdings_df)
            
            # Log detailed info for debugging
            logger.info(f"Using column '{holding_col}': {len(holdings)} total, {len(positive_holdings)} positive")
            
            # CRITICAL: If no positive holdings, try 'highestHolding' as fallback
            # This handles cases where lastHolding shows net position (can be negative after sells)
            if len(positive_holdings) == 0 and 'highestHolding' in user_holdings_df.columns:
                highest_values = pd.to_numeric(user_holdings_df['highestHolding'], errors='coerce').fillna(0).values
                highest_positive = highest_values[highest_values > 0]
                if len(highest_positive) > 0:
                    logger.info(f"Using 'highestHolding' as fallback: {len(highest_positive)} positive values")
                    positive_holdings = highest_positive
                    holdings = highest_values
            
            # CRITICAL: Flag when we have holders but no positive holdings - likely data issue
            if total_holders_in_data > 0 and len(positive_holdings) == 0:
                logger.warning(f"SUSPICIOUS: {total_holders_in_data} holders in data but NO positive holdings! "
                             f"Sample values: {raw_values.head(3).tolist()}")
                # Check if values might be strings that couldn't convert
                non_null_values = raw_values[raw_values.notna()]
                if len(non_null_values) > 0:
                    logger.warning(f"Raw value types: {type(non_null_values.iloc[0])}, "
                                 f"sample raw: {non_null_values.head(3).tolist()}")
        
        # Classify wallets
        classifications = self.wallet_classifier.classify_wallets(
            user_holdings_df=user_holdings_df,
            token_metadata=token_metadata,
            total_supply=total_supply,
            phase=phase
        )
        logger.info(f"Classified {len(classifications)} wallets")
        
        # Calculate time-series delta (default: 10-day window from config)
        accumulation, distribution, is_whale_data_stale, whale_data_source = self._calculate_time_series_delta(
            user_holdings_df=user_holdings_df,
            classifications=classifications,
            lookback_hours=lookback_hours,
            total_supply=total_supply
        )
        
        net_flow = accumulation - distribution
        logger.info(f"Whale delta: accumulation={accumulation:.0f}, distribution={distribution:.0f}, net={net_flow:.0f} "
                   f"(source: {whale_data_source}, stale: {is_whale_data_stale})")
        
        # Calculate effective supply (sum of visible holdings)
        # This is used for more accurate concentration calculations
        effective_supply = float(positive_holdings.sum()) if len(positive_holdings) > 0 else total_supply
        logger.info(f"Effective supply: {effective_supply:.0f} (total_supply: {total_supply:.0f})")
        
        # Calculate Gini on visible holders only
        # CRITICAL: If we have holders in data but no positive holdings, treat as HIGH risk
        if len(positive_holdings) > 0:
            gini = self._calculate_gini_coefficient(positive_holdings)
        elif total_holders_in_data > 0:
            # We have holders but no positive holdings data - likely:
            # 1. All holders sold (negative lastHolding) OR
            # 2. Data issue
            # Either way, this is EXTREMELY suspicious - return HIGH gini
            gini = whale_config.GINI_EXTREME_MIN_VALUE
            logger.warning(f"No positive holdings but {total_holders_in_data} holders in data - "
                          f"returning high gini ({gini}) as safety measure (likely all sold or data issue)")
        else:
            gini = whale_config.GINI_NEUTRAL_DEFAULT
        logger.info(f"Calculated gini: {gini:.4f} (holders in data: {total_holders_in_data}, positive: {len(positive_holdings)})")
        
        # Calculate top 10 concentration using effective_supply for accuracy
        # This gives percentage among visible holders
        top10_pct = self._calculate_top10_concentration(user_holdings_df, total_supply)
        
        # CRITICAL: If top10 is 0 but we have holders, this is suspicious
        # This handles cases where holdings data is corrupted/missing or all sold
        if top10_pct == 0 and total_holders_in_data > 0:
            if total_holders_in_data <= 10:
                top10_pct = 100.0
                logger.warning(f"Top10 was 0% but only {total_holders_in_data} holders in data - "
                              f"setting to 100% as safety measure")
            elif len(positive_holdings) == 0:
                # Many historical holders but no positive holdings = everyone sold
                # Set to 100% to trigger safety overrides (token is likely dead/rugged)
                top10_pct = 100.0
                logger.warning(f"Top10 was 0% with {total_holders_in_data} historical holders but "
                              f"NO positive holdings - setting to 100% (likely all sold/rugged)")
        
        logger.info(f"Calculated top10: {top10_pct:.1f}%")

        # When top 10 hold 100% of (visible) supply, inequality is maximum
        if top10_pct >= 99.99:
            gini = 1.0
            logger.info("Top 10 hold 100% of supply — setting Gini to 1.0 (maximum inequality)")

        # Calculate type-specific holdings
        type_holdings = self.wallet_classifier.calculate_type_holdings(classifications, total_supply)
        
        # Determine whale state
        whale_state = self._determine_whale_state(net_flow, total_supply)
        
        # Count whales
        whales = self.wallet_classifier.get_wallets_by_type(classifications, WalletType.WHALE)
        confirmed_whales = self.wallet_classifier.get_confirmed_whales(classifications)
        logger.info(f"Found {len(whales)} whales ({len(confirmed_whales)} confirmed)")
        
        # Holder metrics
        total_holders = len(user_holdings_df)
        active_holders = len(positive_holdings)  # Use the already calculated positive_holdings
        holder_growth = self._calculate_holder_growth(user_holdings_df, lookback_hours)
        logger.info(f"Holders: {total_holders} total, {active_holders} active, growth: {holder_growth:.1f}%")
        
        # ==================== DOMINANT WHALE INACTIVITY DETECTION ====================
        # Analyze whether the TOP holders (dominant whales) have been active recently
        dominant_whale_metrics = self._analyze_dominant_whale_activity(
            user_holdings_df=user_holdings_df,
            total_supply=total_supply
        )
        
        # Reconcile stale with dominant-whale activity: if top holders are recently active, do not report stale
        if is_whale_data_stale:
            status = dominant_whale_metrics.get('dominant_whale_status', '')
            top_hours = dominant_whale_metrics.get('top_holder_last_activity_hours')
            threshold = getattr(whale_config, 'STALE_OVERRIDE_TOP_HOLDER_ACTIVE_HOURS', 24)
            if status in ('ACTIVE', 'AGING') and top_hours is not None and top_hours <= threshold:
                is_whale_data_stale = False
                logger.info(f"Stale overridden: dominant whales {status}, top holder active {top_hours:.1f}h ago (<= {threshold}h)")
        
        metrics = WhaleMetrics(
            whale_buy_volume=accumulation,
            whale_sell_volume=distribution,
            whale_net_volume=net_flow,
            gini_coefficient=gini,
            top10_hold_percent=top10_pct,
            dev_hold_percent=type_holdings.get('developer', 0.0),
            sniper_hold_percent=type_holdings.get('sniper', 0.0),
            whale_state=whale_state,
            whale_count=len(whales),
            confirmed_whale_count=len(confirmed_whales),
            phase=phase,
            total_holders=total_holders,
            active_holders=active_holders,
            holder_growth_24h=holder_growth,
            is_whale_data_stale=is_whale_data_stale,
            whale_data_source=whale_data_source,
            # Dominant whale inactivity fields
            dominant_whale_count=dominant_whale_metrics['dominant_whale_count'],
            dominant_whale_inactive_count=dominant_whale_metrics['dominant_whale_inactive_count'],
            dominant_whale_aging_count=dominant_whale_metrics.get('dominant_whale_aging_count', 0),
            dominant_whale_inactive_holding_pct=dominant_whale_metrics['dominant_whale_inactive_holding_pct'],
            dominant_whale_aging_holding_pct=dominant_whale_metrics.get('dominant_whale_aging_holding_pct', 0.0),
            top_holder_last_activity_hours=dominant_whale_metrics['top_holder_last_activity_hours'],
            dominant_whale_status=dominant_whale_metrics['dominant_whale_status']
        )
        
        logger.info(f"Whale analysis complete: {whale_state}, Net flow: {net_flow:.2f}, data_source: {whale_data_source}, "
                   f"dominant_whale_status: {dominant_whale_metrics['dominant_whale_status']}")
        
        return metrics
    
    def batch_process_holder_metrics(
        self,
        holders_df: pd.DataFrame,
        mints_df: pd.DataFrame,
        target_mints: Optional[List[str]] = None
    ) -> Dict[str, WhaleMetrics]:
        """
        Process whale metrics for multiple tokens in batch
        
        Args:
            holders_df: DataFrame with holder data (must have 'mint' column)
            mints_df: DataFrame with token metadata
            target_mints: Optional list of specific mints to process
            
        Returns:
            Dictionary mapping mint address to WhaleMetrics
        """
        results = {}
        
        # Get mint column - robust detection
        mint_col = None
        possible_columns = [
            'mint', 'Mint', 'MINT',
            'address', 'Address', 'ADDRESS',
            'mint_address', 'MintAddress', 'mintAddress',
            'token', 'Token', 'TOKEN',
            'token_address', 'TokenAddress', 'tokenAddress'
        ]
        
        for col in possible_columns:
            if col in holders_df.columns:
                mint_col = col
                break
        
        # Case-insensitive search as fallback
        if mint_col is None:
            for col in holders_df.columns:
                if col.lower() == 'mint':
                    mint_col = col
                    break
        
        if mint_col is None:
            raise ValueError(f"Could not find mint column in holders data. Available columns: {list(holders_df.columns)}")
        
        # Normalize to lowercase 'mint' if needed
        if mint_col != 'mint' and 'mint' not in holders_df.columns:
            holders_df['mint'] = holders_df[mint_col]
            mint_col = 'mint'
        
        # Get unique mints to process
        if target_mints:
            mints_to_process = target_mints
        else:
            mints_to_process = holders_df[mint_col].unique().tolist()
        
        logger.info(f"Processing whale metrics for {len(mints_to_process)} tokens...")
        
        # Create metadata lookup - robust column detection
        metadata_lookup = {}
        mint_meta_col = None
        possible_columns = [
            'mint', 'Mint', 'MINT',
            'address', 'Address', 'ADDRESS',
            'mint_address', 'MintAddress', 'mintAddress',
            'token', 'Token', 'TOKEN',
            'token_address', 'TokenAddress', 'tokenAddress'
        ]
        
        for col in possible_columns:
            if col in mints_df.columns:
                mint_meta_col = col
                break
        
        # Case-insensitive search as fallback
        if mint_meta_col is None:
            for col in mints_df.columns:
                if col.lower() == 'mint':
                    mint_meta_col = col
                    break
        
        if mint_meta_col is None:
            raise ValueError(f"Could not find mint column in mints data. Available columns: {list(mints_df.columns)}")
        
        for _, row in mints_df.iterrows():
            metadata_lookup[row[mint_meta_col]] = row.to_dict()
        
        for i, mint in enumerate(mints_to_process):
            try:
                # Filter holders for this mint
                mint_holders = holders_df[holders_df[mint_col] == mint].copy()
                
                if len(mint_holders) == 0:
                    continue
                
                # Get metadata
                metadata = metadata_lookup.get(mint, {})
                
                # Estimate total supply from holders
                holding_col = 'lastHolding' if 'lastHolding' in mint_holders.columns else 'current_holding'
                total_supply = mint_holders[holding_col].sum()
                if total_supply <= 0:
                    total_supply = 1_000_000_000  # Default 1B
                
                # Standardize column names for analyze_token
                standardized_df = self._standardize_holder_df(mint_holders)
                
                # Analyze token
                metrics = self.analyze_token(
                    user_holdings_df=standardized_df,
                    token_metadata=metadata,
                    total_supply=total_supply
                )
                
                results[mint] = metrics
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Processed {i + 1}/{len(mints_to_process)} tokens")
                    
            except Exception as e:
                logger.warning(f"Failed to process mint {mint[:8]}...: {e}")
                continue
        
        logger.info(f"Completed whale analysis for {len(results)} tokens")
        return results
    
    def _standardize_holder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize holder DataFrame column names for analyze_token
        
        Maps various column name formats to the expected format:
        - address/wallet/signer -> wallet
        - lastHolding/finalHolding -> current_holding
        - totalBuys -> total_buys
        - totalSells -> total_sells
        """
        result = df.copy()
        
        # Wallet column
        if 'address' in result.columns:
            result['wallet'] = result['address']
        elif 'signer' in result.columns:
            result['wallet'] = result['signer']
        
        # Current holding
        if 'lastHolding' in result.columns:
            result['current_holding'] = result['lastHolding']
        elif 'finalHolding' in result.columns:
            result['current_holding'] = result['finalHolding']
        
        # Trade columns
        if 'totalBuys' in result.columns:
            result['total_buys'] = result['totalBuys']
        if 'totalSells' in result.columns:
            result['total_sells'] = result['totalSells']
        if 'lastTradeAt' in result.columns:
            result['last_trade_at'] = result['lastTradeAt']
        
        return result
    
    def _calculate_time_series_delta(
        self,
        user_holdings_df: pd.DataFrame,
        classifications: Dict[str, WalletInfo],
        lookback_hours: int = 24,
        total_supply: float = 0.0
    ) -> Tuple[float, float, bool, str]:
        """
        Calculate whale accumulation/distribution using TIME-SERIES method
        
        This is the KEY IMPROVEMENT over the old system:
        - Old: net_delta = totalBuys - totalSells (lifetime cumulative)
        - New: delta = SUM(buys in lookback window) - SUM(sells in lookback window) (time-series)
        
        Special handling for "holding but not trading" scenario:
        - If whales hold significant positions (>5% of supply) but haven't traded recently,
          this is still a meaningful signal (not stale)
        - Only marks as "stale" if whales don't hold much or have sold out
        
        Args:
            user_holdings_df: DataFrame with user holdings
            classifications: Wallet classifications
            lookback_hours: Hours to look back (default: 10 days = 240 hours from config)
            total_supply: Total token supply (used to calculate holding percentage)
            
        Returns:
            Tuple of (accumulation_pressure, distribution_pressure, is_stale, data_source)
            - is_stale: True if using lifetime data AND whales have no significant holdings
            - data_source: "recent" (active trading), "holding" (holding but not trading), "lifetime" (no holdings), or "no_whales"
        """
        accumulation = 0.0
        distribution = 0.0
        
        current_time = datetime.now()
        lookback_time = current_time - timedelta(hours=lookback_hours)
        
        # Get confirmed whale addresses
        whale_addresses = set()
        for addr, info in classifications.items():
            if info.wallet_type == WalletType.WHALE and info.is_confirmed_whale:
                whale_addresses.add(addr)
        
        if not whale_addresses:
            logger.warning("No confirmed whales found, using all whales")
            for addr, info in classifications.items():
                if info.wallet_type == WalletType.WHALE:
                    whale_addresses.add(addr)
        
        if not whale_addresses:
            logger.info("No whale wallets found, returning zero delta")
            return 0.0, 0.0, True, "no_whales"
        
        logger.info(f"Calculating delta for {len(whale_addresses)} whale wallets (lookback: {lookback_hours} hours)")
        
        # Track how many whales have RECENT transaction data (within lookback window)
        whales_with_recent_txns = 0
        whales_with_recent_trades = 0
        
        # NEW: Track current whale holdings to detect "holding but not trading" scenario
        total_whale_holdings = 0.0
        
        for _, row in user_holdings_df.iterrows():
            wallet = row.get('wallet', row.get('address', ''))
            
            if wallet not in whale_addresses:
                continue
            
            # Calculate current whale holdings
            current_holding = float(row.get('current_holding', row.get('lastHolding', row.get('finalHolding', 0))))
            if current_holding > 0:
                total_whale_holdings += current_holding
            
            # Get the target mint for this holder (to filter transactions)
            target_mint = row.get('mint', '')
            
            # Method 1: Use transaction history if available
            transactions = row.get('transactions', [])
            
            if isinstance(transactions, list) and len(transactions) > 0:
                buys_in_window, sells_in_window = self._sum_transactions_in_window(
                    transactions=transactions,
                    lookback_time=lookback_time,
                    target_mint=target_mint
                )
                
                # Only count if there was ACTUAL activity in the window
                if buys_in_window > 0 or sells_in_window > 0:
                    whales_with_recent_txns += 1
                    accumulation += buys_in_window
                    distribution += sells_in_window
                # If whale has transactions but none in window, we'll use fallback
            else:
                # No transaction list — we cannot attribute volume to the lookback window.
                # totalBuys/totalSells are lifetime; only in-window trades count for volume.
                last_trade = self._parse_datetime(
                    row.get('last_trade_at') or row.get('lastTradeAt')
                )
                if last_trade and last_trade >= lookback_time:
                    whales_with_recent_trades += 1
                # Volume stays 0 for this whale without per-trade data
        
        # Calculate whale holdings as % of supply
        whale_holdings_pct = 0.0
        if total_supply > 0:
            whale_holdings_pct = (total_whale_holdings / total_supply) * 100
            logger.info(f"Whale holdings: {total_whale_holdings:.0f} tokens ({whale_holdings_pct:.1f}% of supply)")
        
        if whales_with_recent_txns == 0 and whales_with_recent_trades == 0:
            # No recent trading activity - check if whales are still holding significant positions
            if whale_holdings_pct >= whale_config.WHALE_HOLDING_THRESHOLD_PCT:
                # Whale is still holding significant position - this is a signal, not stale
                logger.info(f"No recent trading but whales hold {whale_holdings_pct:.1f}% of supply "
                           f"(threshold: {whale_config.WHALE_HOLDING_THRESHOLD_PCT}%) - treating as 'holding' signal")
                # Reported volume must be in-window only; we have no in-window trades here
                accumulation = 0.0
                distribution = 0.0
                logger.info("Whale holding signal: Accumulation=0, Distribution=0 (only in-window volume reported)")
                return accumulation, distribution, False, "holding"
            else:
                # No recent activity AND whales don't hold much - truly stale
                logger.warning(f"No recent ({lookback_hours}hr) activity and whales only hold {whale_holdings_pct:.1f}% "
                             f"(threshold: {whale_config.WHALE_HOLDING_THRESHOLD_PCT}%) - marking as stale")
                sentry_fallback_warning(
                    "whale_engine",
                    f"No recent whale activity ({lookback_hours}h) — using lifetime totals as fallback",
                    {"whale_holdings_pct": whale_holdings_pct, "lookback_hours": lookback_hours},
                )
                # Reported volume must be in-window only; we have none here
                accumulation = 0.0
                distribution = 0.0
                logger.info("Lifetime whale activity: Accumulation=0, Distribution=0 (only in-window volume reported)")
                return accumulation, distribution, True, "lifetime"
        else:
            logger.info(f"{lookback_hours}hr Delta: Accumulation={accumulation:.2f}, Distribution={distribution:.2f} "
                       f"(from {whales_with_recent_txns} whales with recent txns, {whales_with_recent_trades} with recent trades)")
            # Recent activity detected - data is fresh
            return accumulation, distribution, False, "recent"
    
    def _sum_transactions_in_window(
        self,
        transactions: List[Dict],
        lookback_time: datetime,
        target_mint: str = ''
    ) -> Tuple[float, float]:
        """
        Sum buy and sell transactions within the time window
        
        Args:
            transactions: List of transaction dictionaries
            lookback_time: Start of the lookback window
            target_mint: Only count transactions for this mint (safety filter)
            
        Returns:
            Tuple of (total_buys, total_sells) in window
        """
        buys = 0.0
        sells = 0.0
        
        for txn in transactions:
            # SAFETY: Only count transactions for the target mint
            # This prevents counting transactions from other tokens if API returns mixed data
            txn_mint = txn.get('mint', '')
            if target_mint and txn_mint and txn_mint != target_mint:
                continue
            
            timestamp = self._parse_datetime(txn.get('timestamp'))
            
            if timestamp and timestamp >= lookback_time:
                # Handle both API formats:
                # 1. API returns 'isBuy' (boolean)
                # 2. Legacy format uses 'type' ('BUY'/'SELL')
                is_buy = txn.get('isBuy')
                if is_buy is None:
                    # Fallback to 'type' field
                    txn_type = str(txn.get('type', '')).upper()
                    is_buy = txn_type == 'BUY'
                
                # Amount might be string or number
                try:
                    amount = float(txn.get('amount', 0))
                except (ValueError, TypeError):
                    amount = 0.0
                
                if is_buy:
                    buys += amount
                else:
                    sells += amount
        
        return buys, sells
    
    def _detect_extreme_concentration(self, holdings: np.ndarray) -> tuple[bool, bool, float]:
        """
        Auto-detect extreme concentration using statistical methods
        
        Args:
            holdings: Sorted array of holdings
            
        Returns:
            (is_extreme, is_high, median_ratio)
            - is_extreme: True if concentration is extreme (outlier detection)
            - is_high: True if concentration is high (but not extreme)
            - median_ratio: Ratio of top holder to median of others
        """
        if len(holdings) < 2:
            return False, False, 0.0
        
        top_holder = holdings[-1]
        median_others = np.median(holdings[:-1]) if len(holdings) > 1 else holdings[0]
        mean_all = np.mean(holdings)
        
        # Dynamic thresholds based on statistical ratios
        median_ratio = top_holder / median_others if median_others > 0 else float('inf')
        mean_ratio = top_holder / mean_all if mean_all > 0 else float('inf')
        
        # Extreme: top holder is outlier (>10x median or >5x mean)
        is_extreme = (median_ratio > whale_config.GINI_EXTREME_MEDIAN_RATIO or 
                     mean_ratio > whale_config.GINI_EXTREME_MEAN_RATIO)
        
        # High: top holder is significantly larger (>5x median or >3x mean) but not extreme
        is_high = ((median_ratio > whale_config.GINI_HIGH_MEDIAN_RATIO or 
                   mean_ratio > whale_config.GINI_HIGH_MEAN_RATIO) and not is_extreme)
        
        return is_extreme, is_high, median_ratio
    
    def _is_few_holders(self, n: int, holdings: np.ndarray) -> bool:
        """
        Auto-detect if sample size is too small using statistical principles
        
        Uses statistical significance: <30 is small sample (Central Limit Theorem threshold)
        For highly variable distributions, even n=20 might be "few"
        
        Args:
            n: Number of holders
            holdings: Array of holdings
            
        Returns:
            True if sample size is statistically too small
        """
        if n < 2:
            return True
        
        # Statistical significance: <30 is small sample
        if n < whale_config.STATS_SMALL_SAMPLE_THRESHOLD:
            # Very small sample
            if n < whale_config.STATS_VERY_SMALL_SAMPLE:
                return True
            # For n between 20-30, check if distribution is highly variable
            # High coefficient of variation (CV > 1.0) means few effective holders
            holdings_array = np.array(holdings)
            mean = np.mean(holdings_array)
            std = np.std(holdings_array)
            cv = std / mean if mean > 0 else float('inf')
            return cv > whale_config.STATS_HIGH_VARIABILITY_CV
        
        return False
    
    def _calculate_gini_coefficient(self, holdings: np.ndarray) -> float:
        """
        Calculate Gini coefficient for token holder distribution using a
        CONCENTRATION-FOCUSED approach.
        
        Standard Gini measures overall inequality, which can be misleadingly high
        when there are many dust holders. For token analysis, we care about
        CONCENTRATION among significant holders, not dust holder inequality.
        
        Approach:
        1. For tokens with many holders (>100), calculate Gini among TOP holders only
           (top 50 by holdings) to measure concentration among significant holders
        2. This gives LOW gini when top holders have balanced distribution
        3. This gives HIGH gini when there's extreme concentration at the top
        
        0 = Perfect equality (top holders have equal share)
        1 = Perfect inequality (one holder dominates)
        
        Args:
            holdings: Array of token holdings (non-negative values)
            
        Returns:
            Gini coefficient (0-1)
        """
        # Remove zero holdings and sort
        holdings = holdings[holdings > 0]
        if len(holdings) == 0:
            logger.warning("No positive holdings found, returning neutral gini")
            return whale_config.GINI_NEUTRAL_DEFAULT
        
        # Special case: only one holder means maximum concentration
        if len(holdings) == 1:
            logger.info("Only 1 holder - returning maximum gini (1.0)")
            return 1.0
        
        # Special case: 2 holders - calculate directly based on ratio
        if len(holdings) == 2:
            holdings = np.sort(holdings)
            smaller, larger = holdings[0], holdings[1]
            ratio = larger / (larger + smaller) if (larger + smaller) > 0 else 0.5
            # Gini for 2 holders: 0.5 = equal, approaching 1.0 for concentration
            gini = 2 * (ratio - 0.5)  # This gives 0 for equal, 1 for max concentration
            logger.info(f"Only 2 holders - ratio={ratio:.4f}, gini={gini:.4f}")
            return float(max(0, min(1, gini)))
        
        # Sort holdings in ascending order
        holdings = np.sort(holdings)
        n = len(holdings)
        total_sum = holdings.sum()
        
        # Handle edge case: all holdings are equal (perfect equality)
        if np.all(holdings == holdings[0]):
            return 0.0
        
        # For tokens with MANY holders (>100), calculate Gini among TOP holders only
        # This focuses on concentration among significant holders, not dust inequality
        top_n_for_gini = 50  # Focus on top 50 holders
        
        if n > 100:
            # Use only top 50 holders for Gini calculation
            top_holdings = holdings[-top_n_for_gini:]
            n_effective = len(top_holdings)
            total_sum_effective = top_holdings.sum()
            
            # Calculate Gini among top holders
            index_weighted_sum = np.sum((np.arange(1, n_effective + 1)) * top_holdings)
            gini = (2.0 * index_weighted_sum) / (n_effective * total_sum_effective) - (n_effective + 1.0) / n_effective
            
            # Also check top holder concentration as a sanity check
            top_holder_pct = top_holdings[-1] / total_sum if total_sum > 0 else 0
            top10_pct = top_holdings[-10:].sum() / total_sum if total_sum > 0 else 0
            
            logger.debug(f"Gini calculated among top {n_effective} of {n} holders: "
                        f"gini={gini:.4f}, top_holder={top_holder_pct*100:.1f}%, top10={top10_pct*100:.1f}%")
            
            # If top holder has very high concentration, ensure gini reflects it
            if top_holder_pct > 0.30:  # Top holder > 30%
                gini = max(gini, 0.70)
            elif top_holder_pct > 0.20:  # Top holder > 20%
                gini = max(gini, 0.55)
            elif top_holder_pct > 0.10:  # Top holder > 10%
                gini = max(gini, 0.40)
            
        else:
            # For small number of holders, use all holdings
            # Special case: few holders - check for extreme concentration
            if n <= whale_config.FEW_HOLDERS_THRESHOLD:
                top_holder = holdings[-1]
                top_holder_pct = top_holder / total_sum if total_sum > 0 else 0
                
                logger.info(f"Few holders ({n}) - top holder has {top_holder_pct*100:.1f}% of visible supply")
                
                # If top holder has > 90% of supply, this is extreme
                if top_holder_pct > 0.90:
                    return 0.95
                elif top_holder_pct > 0.80:
                    return 0.85
                elif top_holder_pct > 0.70:
                    return 0.75
                elif top_holder_pct > 0.60:
                    return 0.65
                elif top_holder_pct > 0.50:
                    return 0.55
            
            # Standard Gini calculation for small-medium holder counts
            index_weighted_sum = np.sum((np.arange(1, n + 1)) * holdings)
            gini = (2.0 * index_weighted_sum) / (n * total_sum) - (n + 1.0) / n
        
        return float(max(0, min(1, gini)))  # Clamp to [0, 1]
    
    def _calculate_top10_concentration(
        self,
        user_holdings_df: pd.DataFrame,
        total_supply: float
    ) -> float:
        """
        Calculate the percentage held by top 10 wallets.

        When total_supply is available and positive, returns % of total supply
        (aligned with Birdeye). Callers may override with Birdeye
        /holder/v1/distribution when available. Otherwise uses sum of visible
        holdings as denominator (% of visible supply).
        """
        if len(user_holdings_df) == 0:
            logger.warning("Empty user_holdings_df for top10 calculation")
            return 0.0
        
        # Try multiple column names for holdings
        holding_col = None
        possible_cols = ['current_holding', 'lastHolding', 'finalHolding', 'balance', 'holding']
        for col in possible_cols:
            if col in user_holdings_df.columns:
                holding_col = col
                break
        
        if holding_col is None:
            logger.warning(f"No holding column found. Available columns: {list(user_holdings_df.columns)}")
            return 0.0
        
        # Convert to numeric if needed
        holdings_values = pd.to_numeric(user_holdings_df[holding_col], errors='coerce').fillna(0)
        
        # Get positive holdings only
        positive_mask = holdings_values > 0
        positive_holdings = holdings_values[positive_mask]
        
        if len(positive_holdings) == 0:
            logger.warning(f"No positive holdings found in column {holding_col}")
            return 0.0
        
        logger.info(f"Top10 calculation: {len(positive_holdings)} holders with positive balance, "
                   f"using column '{holding_col}'")
        
        # Calculate top 10 from visible holders
        n_top = min(10, len(positive_holdings))
        top10_sum = positive_holdings.nlargest(n_top).sum()
        holdings_sum = positive_holdings.sum()
        
        # If holdings_sum is 0 or very small, return 0
        if holdings_sum <= 0:
            logger.warning("Holdings sum is 0 or negative")
            return 0.0
        
        # Use total_supply as denominator when available (align with Birdeye / fallback)
        if total_supply > 0:
            top10_pct = (top10_sum / total_supply) * 100
            top10_pct = min(100.0, max(0.0, top10_pct))
            logger.info(f"Top10 concentration: {top10_pct:.1f}% (top {n_top} hold {top10_sum:.0f} of total_supply {total_supply:.0f})")
        else:
            top10_pct = (top10_sum / holdings_sum) * 100
            if len(positive_holdings) <= 10:
                logger.info(f"Only {len(positive_holdings)} holders, top10 concentration is 100%")
                top10_pct = 100.0
            logger.info(f"Top10 concentration: {top10_pct:.1f}% (top {n_top} hold {top10_sum:.0f} of visible {holdings_sum:.0f})")
        
        # Flag if total_supply is significantly larger than holdings_sum
        if total_supply > 0 and holdings_sum / total_supply < whale_config.MISSING_CONTRACT_SUPPLY_RATIO:
            logger.warning(f"Only {holdings_sum/total_supply*100:.1f}% of supply in visible holders - "
                         f"likely missing contract addresses (e.g., pump.fun)")
        
        return float(top10_pct)
    
    def _determine_whale_state(self, net_flow: float, total_supply: float) -> str:
        """
        Determine whale accumulation state based on net flow
        
        Args:
            net_flow: Net whale flow (accumulation - distribution)
            total_supply: Total token supply
            
        Returns:
            Whale state: 'Accumulation', 'Stability', or 'Distribution'
        """
        if total_supply <= 0:
            return "Stability"
        
        # If net flow is exactly 0 or negligible, it's stable (not distributing)
        if abs(net_flow) < 0.01:  # Negligible flow threshold
            return "Stability"
        
        # Calculate percentage change
        delta_pct = net_flow / total_supply
        
        if delta_pct >= whale_config.ACCUMULATION_THRESHOLD:
            return "Accumulation"
        elif delta_pct <= -whale_config.ACCUMULATION_THRESHOLD:  # Negative threshold for distribution
            return "Distribution"
        else:
            return "Stability"
    
    def _determine_phase(self, metadata: Optional[Dict]) -> str:
        """Determine token phase from metadata"""
        if not metadata:
            return "P4"  # Default to mature phase
        
        created_at = None
        for key in ['CreatedAt', 'createdAt', 'created_at', 'blockUnixTime', 'blockHumanTime', 'timestamp']:
            if key in metadata:
                created_at = self._parse_datetime(metadata[key])
                if created_at:
                    break
        
        if not created_at:
            return "P4"
        
        days_since_launch = (datetime.now() - created_at).total_seconds() / 86400
        return get_phase_from_age(days_since_launch)
    
    def _calculate_holder_growth(
        self,
        user_holdings_df: pd.DataFrame,
        lookback_hours: int = 24
    ) -> float:
        """
        Calculate holder growth percentage over the lookback period
        This is an approximation based on first trade dates
        """
        if len(user_holdings_df) == 0:
            return 0.0
        
        lookback_time = datetime.now() - timedelta(hours=lookback_hours)
        
        # Count new holders (first trade within lookback period)
        new_holders = 0
        for _, row in user_holdings_df.iterrows():
            first_trade = self._parse_datetime(row.get('first_trade_at'))
            if first_trade and first_trade >= lookback_time:
                new_holders += 1
        
        total_holders = len(user_holdings_df)
        if total_holders <= new_holders:
            return 0.0
        
        # Calculate growth percentage
        previous_holders = total_holders - new_holders
        if previous_holders > 0:
            growth_pct = (new_holders / previous_holders) * 100
        else:
            growth_pct = 100.0 if new_holders > 0 else 0.0
        
        return growth_pct
    
    def _analyze_dominant_whale_activity(
        self,
        user_holdings_df: pd.DataFrame,
        total_supply: float,
        inactivity_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze whether the TOP holders (dominant whales) have been active recently.
        
        This helps detect scenarios where whale activity signals are misleading because:
        - Small whales are trading (making it look like "whale accumulation")
        - But the dominant holders (who hold most of the supply) are dormant
        
        Args:
            user_holdings_df: DataFrame with user holdings and transaction history
            total_supply: Total token supply
            inactivity_hours: Hours without trades to be considered "inactive" (default: from config)
            
        Returns:
            Dictionary with:
            - dominant_whale_count: Number of wallets classified as dominant whales
            - dominant_whale_inactive_count: How many are inactive
            - dominant_whale_inactive_holding_pct: % of supply held by inactive dominant whales
            - top_holder_last_activity_hours: Hours since the largest holder's last trade
            - dominant_whale_status: "ACTIVE", "PARTIALLY_INACTIVE", "FULLY_INACTIVE", "UNKNOWN"
        """
        if inactivity_hours is None:
            inactivity_hours = whale_config.DOMINANT_WHALE_INACTIVITY_HOURS
        
        result = {
            'dominant_whale_count': 0,
            'dominant_whale_inactive_count': 0,
            'dominant_whale_inactive_holding_pct': 0.0,
            'top_holder_last_activity_hours': None,
            'dominant_whale_status': 'UNKNOWN'
        }
        
        if len(user_holdings_df) == 0 or total_supply <= 0:
            return result
        
        current_time = datetime.now()
        inactivity_threshold = current_time - timedelta(hours=inactivity_hours)
        aging_threshold = current_time - timedelta(hours=whale_config.DOMINANT_WHALE_AGING_HOURS)
        
        # Find the holding column
        holding_col = None
        possible_cols = ['current_holding', 'lastHolding', 'finalHolding', 'balance', 'holding']
        for col in possible_cols:
            if col in user_holdings_df.columns:
                holding_col = col
                break
        
        if holding_col is None:
            logger.warning("No holding column found for dominant whale analysis")
            return result
        
        # Convert holdings to numeric
        df = user_holdings_df.copy()
        df['_holding_numeric'] = pd.to_numeric(df[holding_col], errors='coerce').fillna(0)
        
        # Filter to positive holdings and sort by holdings (descending)
        df = df[df['_holding_numeric'] > 0].sort_values('_holding_numeric', ascending=False)
        
        if len(df) == 0:
            return result
        
        # IMPORTANT: Use EFFECTIVE supply (sum of visible holdings) not total supply
        # This is crucial for pump.fun tokens where most supply is in the bonding curve contract
        # Without this, a whale holding 80% of visible supply might be <5% of total supply
        effective_supply = df['_holding_numeric'].sum()
        
        # Calculate holding percentage against EFFECTIVE supply for dominant whale detection
        df['_holding_pct'] = (df['_holding_numeric'] / effective_supply) * 100 if effective_supply > 0 else 0
        
        logger.info(f"Dominant whale analysis: effective_supply={effective_supply:.0f} "
                   f"({(effective_supply/total_supply)*100:.1f}% of total), "
                   f"analyzing {len(df)} holders with positive balance")
        
        # Identify dominant whales: top N holders with at least MIN_HOLDING_PCT of EFFECTIVE supply
        min_holding_pct = whale_config.DOMINANT_WHALE_MIN_HOLDING_PCT
        max_dominant_count = whale_config.DOMINANT_WHALE_COUNT
        
        dominant_whales = []
        for idx, row in df.head(max_dominant_count).iterrows():
            holding_pct = row['_holding_pct']
            if holding_pct >= min_holding_pct:
                # Parse last trade timestamp
                last_trade = self._parse_datetime(
                    row.get('last_trade_at') or row.get('lastTradeAt')
                )
                
                # Calculate hours since last trade
                hours_since_last_trade = None
                is_inactive = True  # Assume inactive if no timestamp
                is_aging = False  # Aging = borderline (between aging_threshold and inactivity_threshold)
                
                if last_trade:
                    hours_since_last_trade = (current_time - last_trade).total_seconds() / 3600
                    is_inactive = last_trade < inactivity_threshold  # 5+ days by default
                    is_aging = not is_inactive and last_trade < aging_threshold  # 3-5 days by default
                
                dominant_whales.append({
                    'wallet': row.get('wallet', row.get('address', row.get('signer', 'unknown'))),
                    'holding_pct': holding_pct,
                    'hours_since_last_trade': hours_since_last_trade,
                    'is_inactive': is_inactive,
                    'is_aging': is_aging,
                    'last_trade': last_trade
                })
        
        if not dominant_whales:
            logger.info(f"No dominant whales found (no holders meet the {min_holding_pct}% minimum holding threshold of effective supply)")
            return result
        
        # Calculate metrics (percentages are relative to EFFECTIVE supply)
        dominant_count = len(dominant_whales)
        inactive_count = sum(1 for w in dominant_whales if w['is_inactive'])
        aging_count = sum(1 for w in dominant_whales if w['is_aging'])
        inactive_holding_pct = sum(w['holding_pct'] for w in dominant_whales if w['is_inactive'])
        aging_holding_pct = sum(w['holding_pct'] for w in dominant_whales if w['is_aging'])
        
        # Get the largest holder's last activity
        top_holder = dominant_whales[0]
        top_holder_last_activity = top_holder['hours_since_last_trade']
        
        # Determine status (graduated: ACTIVE -> AGING -> PARTIALLY_INACTIVE -> FULLY_INACTIVE)
        # Priority: inactive > aging > active
        if inactive_count == dominant_count:
            status = "FULLY_INACTIVE"
        elif inactive_count > 0:
            status = "PARTIALLY_INACTIVE"
        elif aging_count == dominant_count:
            status = "AGING"  # All dominant whales are aging (3-5 days inactive)
        elif aging_count > 0:
            status = "AGING"  # Some dominant whales are aging
        else:
            status = "ACTIVE"
        
        result = {
            'dominant_whale_count': dominant_count,
            'dominant_whale_inactive_count': inactive_count,
            'dominant_whale_aging_count': aging_count,
            'dominant_whale_inactive_holding_pct': round(inactive_holding_pct, 2),
            'dominant_whale_aging_holding_pct': round(aging_holding_pct, 2),
            'top_holder_last_activity_hours': round(top_holder_last_activity, 1) if top_holder_last_activity is not None else None,
            'dominant_whale_status': status
        }
        
        # Log detailed information
        logger.info(f"Dominant whale analysis: {dominant_count} dominant whales found, "
                   f"{inactive_count} inactive ({inactive_holding_pct:.1f}% of supply), "
                   f"{aging_count} aging ({aging_holding_pct:.1f}% of supply), "
                   f"status: {status}")
        
        if top_holder_last_activity is not None:
            logger.info(f"  Top holder ({top_holder['holding_pct']:.1f}% of supply) "
                       f"last active {top_holder_last_activity:.1f} hours ago")
        
        for i, whale in enumerate(dominant_whales[:3], 1):  # Log top 3 for debugging
            inactive_str = "INACTIVE" if whale['is_inactive'] else "ACTIVE"
            hours_str = f"{whale['hours_since_last_trade']:.1f}h ago" if whale['hours_since_last_trade'] else "unknown"
            logger.debug(f"  Dominant whale #{i}: {whale['holding_pct']:.1f}% of supply, "
                        f"last trade: {hours_str}, status: {inactive_str}")
        
        return result
    
    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse various datetime formats including Unix timestamps (int/float).

        Always returns a naive UTC datetime so it can be safely compared with
        datetime.now() / datetime.utcnow() used elsewhere in the engine.
        """
        if value is None:
            return None

        dt: Optional[datetime] = None

        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, pd.Timestamp):
            dt = value.to_pydatetime()
        elif isinstance(value, (int, float)):
            try:
                if 1_000_000_000 <= value <= 9_999_999_999:
                    dt = datetime.utcfromtimestamp(value)
            except (ValueError, OSError, OverflowError):
                return None
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    dt = pd.to_datetime(value).to_pydatetime()
                except Exception:
                    return None

        if dt is not None and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        return dt


def convert_metrics_to_dict(metrics: WhaleMetrics) -> Dict[str, Any]:
    """Convert WhaleMetrics dataclass to dictionary"""
    return {
        'whale_buy_volume': metrics.whale_buy_volume,
        'whale_sell_volume': metrics.whale_sell_volume,
        'whale_net_volume': metrics.whale_net_volume,
        'gini_coefficient': metrics.gini_coefficient,
        'top10_hold_percent': metrics.top10_hold_percent,
        'dev_hold_percent': metrics.dev_hold_percent,
        'sniper_hold_percent': metrics.sniper_hold_percent,
        'whale_state': metrics.whale_state,
        'whale_count': metrics.whale_count,
        'confirmed_whale_count': metrics.confirmed_whale_count,
        'phase': metrics.phase,
        'total_holders': metrics.total_holders,
        'active_holders': metrics.active_holders,
        'holder_growth_24h': metrics.holder_growth_24h
    }


def analyze_whale_behavior_summary(whale_metrics: Dict[str, WhaleMetrics]) -> Dict[str, Any]:
    """
    Analyze whale behavior across multiple tokens and generate summary statistics
    
    Args:
        whale_metrics: Dictionary mapping token mint to WhaleMetrics
        
    Returns:
        Summary dictionary with aggregated statistics
    """
    if not whale_metrics:
        return {
            'total_tokens_analyzed': 0,
            'avg_gini_coefficient': whale_config.GINI_NEUTRAL_DEFAULT,
            'high_risk_tokens': 0,
            'accumulation_tokens': 0,
            'distribution_tokens': 0,
            'stable_tokens': 0
        }
    
    gini_values = []
    accumulation_count = 0
    distribution_count = 0
    stable_count = 0
    high_risk_count = 0
    
    for mint, metrics in whale_metrics.items():
        if isinstance(metrics, WhaleMetrics):
            gini_values.append(metrics.gini_coefficient)
            
            # Count whale states
            if metrics.whale_state == 'ACCUMULATION':
                accumulation_count += 1
            elif metrics.whale_state == 'DISTRIBUTION':
                distribution_count += 1
            else:
                stable_count += 1
            
            # High risk: high gini + high concentration
            if (metrics.gini_coefficient > whale_config.GINI_HIGH_RISK or 
                metrics.top10_hold_percent > wallet_classification_config.TOP10_CONCENTRATION_HIGH_RISK * 100):
                high_risk_count += 1
    
    return {
        'total_tokens_analyzed': len(whale_metrics),
        'avg_gini_coefficient': np.mean(gini_values) if gini_values else whale_config.GINI_NEUTRAL_DEFAULT,
        'high_risk_tokens': high_risk_count,
        'accumulation_tokens': accumulation_count,
        'distribution_tokens': distribution_count,
        'stable_tokens': stable_count
    }


if __name__ == "__main__":
    print("Whale Engine")
    print("=" * 60)
    
    # Create sample test data with transaction history
    sample_data = pd.DataFrame([
        {
            'wallet': 'whale1',
            'current_holding': 50000000,
            'first_trade_at': (datetime.now() - timedelta(days=5)).isoformat(),
            'last_trade_at': (datetime.now() - timedelta(hours=2)).isoformat(),
            'total_buys': 55000000,
            'total_sells': 5000000,
            'transactions': [
                {'type': 'BUY', 'amount': 30000000, 'timestamp': (datetime.now() - timedelta(hours=20)).isoformat()},
                {'type': 'BUY', 'amount': 25000000, 'timestamp': (datetime.now() - timedelta(hours=10)).isoformat()},
                {'type': 'SELL', 'amount': 5000000, 'timestamp': (datetime.now() - timedelta(hours=2)).isoformat()},
            ]
        },
        {
            'wallet': 'whale2',
            'current_holding': 30000000,
            'first_trade_at': (datetime.now() - timedelta(days=3)).isoformat(),
            'last_trade_at': (datetime.now() - timedelta(hours=5)).isoformat(),
            'total_buys': 40000000,
            'total_sells': 10000000,
            'transactions': [
                {'type': 'BUY', 'amount': 20000000, 'timestamp': (datetime.now() - timedelta(hours=15)).isoformat()},
                {'type': 'SELL', 'amount': 10000000, 'timestamp': (datetime.now() - timedelta(hours=5)).isoformat()},
            ]
        },
        {
            'wallet': 'retail1',
            'current_holding': 1000000,
            'first_trade_at': (datetime.now() - timedelta(days=1)).isoformat(),
            'last_trade_at': (datetime.now() - timedelta(hours=12)).isoformat(),
            'total_buys': 1000000,
            'total_sells': 0,
            'transactions': []
        },
    ])
    
    metadata = {
        'Developer': 'dev_wallet_123',
        'CreatedAt': (datetime.now() - timedelta(days=10)).isoformat()
    }
    
    engine = WhaleEngine()
    metrics = engine.analyze_token(
        user_holdings_df=sample_data,
        token_metadata=metadata,
        total_supply=1000000000  # 1 billion
    )
    
    print("\nWhale Metrics:")
    print(f"  Whale Buy Volume: {metrics.whale_buy_volume:,.0f}")
    print(f"  Whale Sell Volume: {metrics.whale_sell_volume:,.0f}")
    print(f"  Whale Net Volume: {metrics.whale_net_volume:,.0f}")
    print(f"  Whale State: {metrics.whale_state}")
    print(f"  Gini Coefficient: {metrics.gini_coefficient:.3f}")
    print(f"  Top 10 Hold %: {metrics.top10_hold_percent:.2f}%")
    print(f"  Phase: {metrics.phase}")
    print(f"  Whale Count: {metrics.whale_count} ({metrics.confirmed_whale_count} confirmed)")

