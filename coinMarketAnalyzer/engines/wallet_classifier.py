"""
Wallet Classifier Engine
Classifies wallets into categories: Developer, Sniper, Whale, Retail
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    wallet_classification_config, phase_thresholds,
    get_phase_from_age, get_noise_threshold, get_whale_percentile
)

# Setup logging
logger = logging.getLogger(__name__)


class WalletType(Enum):
    """Wallet classification types"""
    DEVELOPER = "developer"
    SNIPER = "sniper"
    WHALE = "whale"
    RETAIL = "retail"
    UNKNOWN = "unknown"


@dataclass
class WalletInfo:
    """Information about a classified wallet"""
    address: str
    wallet_type: WalletType
    current_holding: float
    holding_percentage: float
    first_trade_at: Optional[datetime]
    last_trade_at: Optional[datetime]
    total_buys: float
    total_sells: float
    is_confirmed_whale: bool  # Passed 24-hour persistence check
    hours_as_whale: float  # How long above whale threshold


class WalletClassifier:
    """
    Classifies wallets into categories based on behavior and holdings
    """
    
    def __init__(self):
        """Initialize Wallet Classifier"""
        self.config = wallet_classification_config
        logger.info("Wallet Classifier initialized")
    
    def classify_wallets(
        self,
        user_holdings_df: pd.DataFrame,
        token_metadata: Optional[Dict[str, Any]],
        total_supply: float,
        phase: str
    ) -> Dict[str, WalletInfo]:
        """
        Classify all wallets for a token
        
        Args:
            user_holdings_df: DataFrame with user holdings data
            token_metadata: Token metadata including developer address and creation time
            total_supply: Total token supply
            phase: Token phase (P1, P2, P3, P4)
            
        Returns:
            Dictionary mapping wallet address to WalletInfo
        """
        if len(user_holdings_df) == 0:
            logger.warning("Empty user holdings DataFrame, returning empty classification")
            return {}
        
        # Extract metadata
        developer_address = self._get_developer_address(token_metadata)
        token_creation_time = self._get_token_creation_time(token_metadata)
        
        # Get phase-specific thresholds
        noise_threshold = get_noise_threshold(phase)
        whale_percentile = get_whale_percentile(phase)
        noise_threshold_absolute = total_supply * noise_threshold
        
        # Get holdings column (handle different column names)
        holding_col = 'current_holding' if 'current_holding' in user_holdings_df.columns else 'lastHolding'
        
        # Calculate whale threshold based on percentile
        holdings = user_holdings_df[holding_col].values
        holdings = holdings[holdings > 0]
        if len(holdings) > 0:
            whale_threshold = np.percentile(holdings, whale_percentile)
        else:
            whale_threshold = float('inf')
        
        logger.info(f"Classifying wallets - Phase: {phase}, Whale threshold: {whale_threshold:.2f}")
        
        
        classifications = {}
        
        for idx, row in user_holdings_df.iterrows():
            # Get wallet address - try multiple column names
            wallet_address = None
            for addr_col in ['wallet', 'address', 'owner', 'signer']:
                if addr_col in row.index:
                    val = row[addr_col]
                    if val is not None and pd.notna(val) and str(val).strip():
                        wallet_address = str(val)
                        break
            
            if wallet_address is None:
                wallet_address = f"unknown_{idx}"
            
            # Handle different column names for holdings
            current_holding = 0.0
            for hold_col in ['current_holding', 'lastHolding', 'finalHolding', 'balance']:
                if hold_col in row.index:
                    val = row[hold_col]
                    if val is not None:
                        try:
                            current_holding = float(val)
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Skip wallets below noise threshold
            if current_holding < noise_threshold_absolute and current_holding > 0:
                continue
            
            # Calculate holding percentage
            holding_pct = (current_holding / total_supply * 100) if total_supply > 0 else 0
            
            # Get transaction times
            first_trade = self._parse_datetime(row.get('first_trade_at'))
            last_trade = self._parse_datetime(row.get('last_trade_at'))
            
            # Determine wallet type
            wallet_type = self._determine_wallet_type(
                wallet_address=wallet_address,
                current_holding=current_holding,
                first_trade_at=first_trade,
                developer_address=developer_address,
                token_creation_time=token_creation_time,
                whale_threshold=whale_threshold,
                row=row  # Pass row for API flag checking
            )
            
            # Check whale persistence
            is_confirmed_whale = False
            hours_as_whale = 0.0
            
            if wallet_type == WalletType.WHALE:
                # If we have API-provided whale classification (from /mint/details), trust it
                if 'isWhale' in row.index and row.get('isWhale'):
                    is_confirmed_whale = True
                    hours_as_whale = 48.0  # Assume confirmed if API classifies as whale
                    logger.debug(f"Whale {wallet_address[:8]}... confirmed via API classification")
                else:
                    # Verify persistence using transaction history
                    is_confirmed_whale, hours_as_whale = self._verify_whale_persistence(
                        row=row,
                        whale_threshold=whale_threshold,
                        total_supply=total_supply
                    )
                
                # Downgrade to retail if not confirmed
                if not is_confirmed_whale:
                    wallet_type = WalletType.RETAIL
                    logger.debug(f"Whale {wallet_address[:8]}... downgraded: only {hours_as_whale:.1f} hours as whale")
            
            classifications[wallet_address] = WalletInfo(
                address=wallet_address,
                wallet_type=wallet_type,
                current_holding=current_holding,
                holding_percentage=holding_pct,
                first_trade_at=first_trade,
                last_trade_at=last_trade,
                total_buys=float(row.get('total_buys', 0)),
                total_sells=float(row.get('total_sells', 0)),
                is_confirmed_whale=is_confirmed_whale,
                hours_as_whale=hours_as_whale
            )
        
        # Log classification summary
        self._log_classification_summary(classifications)
        
        return classifications
    
    def _determine_wallet_type(
        self,
        wallet_address: str,
        current_holding: float,
        first_trade_at: Optional[datetime],
        developer_address: Optional[str],
        token_creation_time: Optional[datetime],
        whale_threshold: float,
        row: Optional[pd.Series] = None
    ) -> WalletType:
        """
        Determine the type of a wallet based on its characteristics
        If row is provided with API flags (isWhale, isSniper, isDev), use those first
        """
        # Priority 1: Check API-provided classifications (from /mint/details)
        if row is not None:
            if row.get('isDev') or row.get('isZeroAddress'):
                return WalletType.DEVELOPER
            if row.get('isSniper'):
                return WalletType.SNIPER
            if row.get('isWhale'):
                return WalletType.WHALE
        
        # Priority 2: Check if developer by address
        if developer_address and wallet_address.lower() == developer_address.lower():
            return WalletType.DEVELOPER
        
        # Priority 3: Check if sniper (traded within sniper window of token creation)
        if first_trade_at and token_creation_time:
            seconds_after_creation = (first_trade_at - token_creation_time).total_seconds()
            if 0 <= seconds_after_creation <= self.config.SNIPER_WINDOW_SECONDS:
                return WalletType.SNIPER
        
        # Priority 4: Check if whale (above threshold)
        if current_holding >= whale_threshold:
            return WalletType.WHALE
        
        # Default to retail
        return WalletType.RETAIL
    
    def _verify_whale_persistence(
        self,
        row: pd.Series,
        whale_threshold: float,
        total_supply: float
    ) -> Tuple[bool, float]:
        """
        Verify if a whale has held above threshold for required duration
        
        Returns:
            Tuple of (is_confirmed, hours_as_whale)
        """
        # Check if we have transaction history to verify persistence
        transactions = row.get('transactions', [])
        
        if isinstance(transactions, list) and len(transactions) > 0:
            # Calculate balance history from transactions
            first_whale_time = self._find_first_whale_time(
                transactions=transactions,
                whale_threshold=whale_threshold
            )
            
            if first_whale_time:
                hours_as_whale = (datetime.now() - first_whale_time).total_seconds() / 3600
                is_confirmed = hours_as_whale >= self.config.WHALE_PERSISTENCE_HOURS
                return is_confirmed, hours_as_whale
        
        # Fallback: use first_trade_at or lastTradeAt if available
        first_trade = self._parse_datetime(row.get('first_trade_at'))
        if first_trade:
            # Assume they became whale at first trade (conservative estimate)
            hours_since_first = (datetime.now() - first_trade).total_seconds() / 3600
            is_confirmed = hours_since_first >= self.config.WHALE_PERSISTENCE_HOURS
            return is_confirmed, hours_since_first
        
        # Secondary fallback: use lastTradeAt (some APIs only provide this)
        last_trade = self._parse_datetime(row.get('last_trade_at') or row.get('lastTradeAt'))
        if last_trade:
            hours_since_last = (datetime.now() - last_trade).total_seconds() / 3600
            
            # Logic: If their last trade was 24+ hours ago AND they still hold whale amounts,
            # they've been holding at whale level for at least that long.
            # This is conservative: actual holding time could be longer than time since last trade.
            # 
            # For someone who bought and hasn't sold (last trade = buy 24hrs ago):
            #   - They've held for 24+ hours -> confirmed
            # For someone who just sold (last trade = sell 1hr ago):
            #   - If they still have whale amounts, they're accumulating -> confirmed if holding < 24hrs
            #   - But we can't know exactly when they became a whale
            #
            # Simpler approach: If they've traded in the past (have lastTradeAt data),
            # and currently hold whale amounts, consider them confirmed whales.
            # The persistence check is mainly to filter out very new whales (< 24hrs).
            # Since we can't get exact first-whale time, use lastTrade as proxy.
            is_confirmed = hours_since_last >= self.config.WHALE_PERSISTENCE_HOURS
            return is_confirmed, hours_since_last
        
        # If no history available, check if token is mature enough (P4 = 45+ days old)
        # For mature tokens, if someone holds whale amounts, they're likely established
        # This is a fallback when no timestamp data is available
        return False, 0.0
    
    def _find_first_whale_time(
        self,
        transactions: List[Dict],
        whale_threshold: float
    ) -> Optional[datetime]:
        """
        Find the first time a wallet exceeded whale threshold
        
        Args:
            transactions: List of transaction dictionaries with 'type'/'isBuy', 'amount', 'timestamp'
            whale_threshold: Threshold to be considered a whale
            
        Returns:
            Datetime when wallet first became a whale, or None
        """
        running_balance = 0
        
        # Sort transactions by timestamp
        sorted_txns = sorted(transactions, key=lambda x: x.get('timestamp', ''))
        
        for txn in sorted_txns:
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
            
            timestamp = self._parse_datetime(txn.get('timestamp'))
            
            if is_buy:
                running_balance += amount
            else:
                running_balance -= amount
            
            # Check if this transaction pushed them over whale threshold
            if running_balance >= whale_threshold:
                return timestamp
        
        return None
    
    def _get_developer_address(self, metadata: Optional[Dict]) -> Optional[str]:
        """Extract developer address from token metadata"""
        if not metadata:
            return None
        
        # Try different possible key names
        for key in ['Developer', 'developer', 'creator', 'Creator', 'deployer', 'owner']:
            if key in metadata:
                return str(metadata[key])
        
        return None
    
    def _get_token_creation_time(self, metadata: Optional[Dict]) -> Optional[datetime]:
        """Extract token creation time from metadata"""
        if not metadata:
            return None
        
        for key in ['CreatedAt', 'createdAt', 'created_at', 'blockUnixTime', 'blockHumanTime', 'creation_time', 'timestamp']:
            if key in metadata:
                result = self._parse_datetime(metadata[key])
                if result:
                    return result
        
        return None
    
    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse various datetime formats including Unix timestamps (int/float)."""
        if value is None:
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        
        if isinstance(value, (int, float)):
            try:
                if 1_000_000_000 <= value <= 9_999_999_999:
                    return datetime.utcfromtimestamp(value)
            except (ValueError, OSError, OverflowError):
                return None
        
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    return pd.to_datetime(value).to_pydatetime()
                except Exception:
                    return None
        
        return None
    
    def _log_classification_summary(self, classifications: Dict[str, WalletInfo]):
        """Log summary of wallet classifications"""
        type_counts = {wt: 0 for wt in WalletType}
        confirmed_whales = 0
        
        for wallet_info in classifications.values():
            type_counts[wallet_info.wallet_type] += 1
            if wallet_info.is_confirmed_whale:
                confirmed_whales += 1
        
        logger.info(f"Wallet Classification Summary:")
        logger.info(f"  Developer: {type_counts[WalletType.DEVELOPER]}")
        logger.info(f"  Snipers: {type_counts[WalletType.SNIPER]}")
        logger.info(f"  Whales: {type_counts[WalletType.WHALE]} ({confirmed_whales} confirmed)")
        logger.info(f"  Retail: {type_counts[WalletType.RETAIL]}")
    
    def get_wallets_by_type(
        self,
        classifications: Dict[str, WalletInfo],
        wallet_type: WalletType
    ) -> List[WalletInfo]:
        """Get all wallets of a specific type"""
        return [w for w in classifications.values() if w.wallet_type == wallet_type]
    
    def get_developer_wallet(self, classifications: Dict[str, WalletInfo]) -> Optional[WalletInfo]:
        """Get the developer wallet if identified"""
        devs = self.get_wallets_by_type(classifications, WalletType.DEVELOPER)
        return devs[0] if devs else None
    
    def get_confirmed_whales(self, classifications: Dict[str, WalletInfo]) -> List[WalletInfo]:
        """Get all confirmed whale wallets (passed 24hr persistence)"""
        return [w for w in classifications.values() 
                if w.wallet_type == WalletType.WHALE and w.is_confirmed_whale]
    
    def get_snipers(self, classifications: Dict[str, WalletInfo]) -> List[WalletInfo]:
        """Get all sniper wallets"""
        return self.get_wallets_by_type(classifications, WalletType.SNIPER)
    
    def calculate_type_holdings(
        self,
        classifications: Dict[str, WalletInfo],
        total_supply: float
    ) -> Dict[str, float]:
        """
        Calculate total holdings percentage by wallet type
        
        Returns:
            Dictionary with holding percentages by type
        """
        type_holdings = {wt.value: 0.0 for wt in WalletType}
        
        for wallet_info in classifications.values():
            type_holdings[wallet_info.wallet_type.value] += wallet_info.current_holding
        
        # Convert to percentages
        if total_supply > 0:
            for wt in type_holdings:
                type_holdings[wt] = (type_holdings[wt] / total_supply) * 100
        
        return type_holdings


if __name__ == "__main__":
    print("Wallet Classifier Engine")
    print("=" * 50)
    
    # Create sample test data
    sample_data = pd.DataFrame([
        {'wallet': 'dev123', 'current_holding': 5000000, 'first_trade_at': '2024-01-01T00:00:01', 
         'total_buys': 5000000, 'total_sells': 0},
        {'wallet': 'sniper1', 'current_holding': 1000000, 'first_trade_at': '2024-01-01T00:00:30', 
         'total_buys': 1000000, 'total_sells': 0},
        {'wallet': 'whale1', 'current_holding': 50000000, 'first_trade_at': '2024-01-02T00:00:00', 
         'total_buys': 50000000, 'total_sells': 0},
        {'wallet': 'retail1', 'current_holding': 100000, 'first_trade_at': '2024-01-03T00:00:00', 
         'total_buys': 100000, 'total_sells': 0},
    ])
    
    metadata = {
        'Developer': 'dev123',
        'CreatedAt': '2024-01-01T00:00:00'
    }
    
    classifier = WalletClassifier()
    classifications = classifier.classify_wallets(
        user_holdings_df=sample_data,
        token_metadata=metadata,
        total_supply=1000000000,  # 1 billion
        phase='P2'
    )
    
    print("\nClassified Wallets:")
    for addr, info in classifications.items():
        print(f"  {addr}: {info.wallet_type.value} ({info.holding_percentage:.2f}%)")
    
    type_holdings = classifier.calculate_type_holdings(classifications, 1000000000)
    print(f"\nHoldings by Type:")
    for wt, pct in type_holdings.items():
        print(f"  {wt}: {pct:.2f}%")

