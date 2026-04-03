"""
Holder Metrics Calculator
Calculates Gini coefficient, concentration metrics, and holder statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import whale_config, wallet_classification_config

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class HolderStats:
    """Comprehensive holder statistics"""
    total_holders: int
    active_holders: int
    gini_coefficient: float
    top10_concentration: float
    top20_concentration: float
    top50_concentration: float
    median_holding: float
    mean_holding: float
    std_holding: float
    holder_score: float  # 0-100 score indicating health of holder distribution


class HolderMetricsCalculator:
    """
    Calculates comprehensive holder metrics and statistics
    """
    
    def __init__(self):
        """Initialize Holder Metrics Calculator"""
        logger.info("Holder Metrics Calculator initialized")
    
    def create_empty_stats(self) -> HolderStats:
        """
        Create empty/default HolderStats for tokens without holder data
        Used for perps tokens where on-chain holder data is not available
        
        Returns:
            HolderStats with default/neutral values
        """
        return self._empty_holder_stats()
    
    def calculate_all_metrics(
        self,
        holdings_df: pd.DataFrame,
        total_supply: float
    ) -> HolderStats:
        """
        Calculate all holder metrics
        
        Args:
            holdings_df: DataFrame with holder data
            total_supply: Total token supply
            
        Returns:
            HolderStats object with all calculated metrics
        """
        if len(holdings_df) == 0:
            return self._empty_holder_stats()
        
        # Get holdings array
        if 'current_holding' in holdings_df.columns:
            holdings = holdings_df['current_holding'].values
        elif 'lastHolding' in holdings_df.columns:
            holdings = holdings_df['lastHolding'].values
        else:
            logger.warning("No holding column found")
            return self._empty_holder_stats()
        
        # Filter positive holdings
        positive_holdings = holdings[holdings > 0]
        
        if len(positive_holdings) == 0:
            return self._empty_holder_stats()
        
        # Calculate metrics
        gini = self.calculate_gini(positive_holdings)
        top10 = self.calculate_top_n_concentration(holdings, 10, total_supply)
        top20 = self.calculate_top_n_concentration(holdings, 20, total_supply)
        top50 = self.calculate_top_n_concentration(holdings, 50, total_supply)

        # When top 10 hold 100% of (visible) supply, inequality is maximum.
        # Only apply when we have >10 positive holders so the 100% is from real concentration,
        # not from the "only N holders" shortcut (which would wrongly set gini=1.0 with low top10).
        if top10 >= 99.99 and len(positive_holdings) > 10:
            gini = 1.0
            logger.info("Top 10 hold 100% of supply — setting Gini to 1.0 (maximum inequality)")

        # Statistical measures
        median = float(np.median(positive_holdings))
        mean = float(np.mean(positive_holdings))
        std = float(np.std(positive_holdings))
        
        # Calculate holder health score
        holder_score = self._calculate_holder_score(
            gini=gini,
            top10_concentration=top10,
            total_holders=len(holdings_df),
            active_holders=len(positive_holdings)
        )
        
        return HolderStats(
            total_holders=len(holdings_df),
            active_holders=len(positive_holdings),
            gini_coefficient=gini,
            top10_concentration=top10,
            top20_concentration=top20,
            top50_concentration=top50,
            median_holding=median,
            mean_holding=mean,
            std_holding=std,
            holder_score=holder_score
        )
    
    def _detect_extreme_concentration(self, holdings: np.ndarray) -> tuple[bool, bool, float]:
        """
        Auto-detect extreme concentration using statistical methods
        (Same logic as whale_engine for consistency)
        """
        if len(holdings) < 2:
            return False, False, 0.0
        
        top_holder = holdings[-1]
        median_others = np.median(holdings[:-1]) if len(holdings) > 1 else holdings[0]
        mean_all = np.mean(holdings)
        
        median_ratio = top_holder / median_others if median_others > 0 else float('inf')
        mean_ratio = top_holder / mean_all if mean_all > 0 else float('inf')
        
        is_extreme = (median_ratio > whale_config.GINI_EXTREME_MEDIAN_RATIO or 
                     mean_ratio > whale_config.GINI_EXTREME_MEAN_RATIO)
        is_high = ((median_ratio > whale_config.GINI_HIGH_MEDIAN_RATIO or 
                   mean_ratio > whale_config.GINI_HIGH_MEAN_RATIO) and not is_extreme)
        
        return is_extreme, is_high, median_ratio
    
    def _is_few_holders(self, n: int, holdings: np.ndarray) -> bool:
        """Auto-detect if sample size is too small using statistical principles"""
        if n < 2:
            return True
        
        if n < whale_config.STATS_SMALL_SAMPLE_THRESHOLD:
            if n < whale_config.STATS_VERY_SMALL_SAMPLE:
                return True
            holdings_array = np.array(holdings)
            mean = np.mean(holdings_array)
            std = np.std(holdings_array)
            cv = std / mean if mean > 0 else float('inf')
            return cv > whale_config.STATS_HIGH_VARIABILITY_CV
        
        return False
    
    def calculate_gini(self, holdings: np.ndarray) -> float:
        """
        Calculate Gini coefficient for distribution inequality using a 
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
            holdings: Array of positive holdings
            
        Returns:
            Gini coefficient (0-1)
        """
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
        
        return float(max(0, min(1, gini)))
    
    def calculate_top_n_concentration(
        self,
        holdings: np.ndarray,
        n: int,
        total_supply: float
    ) -> float:
        """
        Calculate percentage held by top N holders
        
        IMPORTANT: Uses the sum of holdings as denominator when total_supply
        is significantly larger, to accurately reflect distribution among
        visible holders (avoids issues with pump.fun contract not in data).
        
        Args:
            holdings: Array of all holdings
            n: Number of top holders
            total_supply: Total token supply
            
        Returns:
            Percentage held by top N
        """
        holdings = holdings[holdings > 0]  # Filter zeros
        if len(holdings) == 0:
            logger.warning(f"No positive holdings for top{n} calculation")
            return 0.0
        
        # Special case: if there are <= N holders, concentration is 100%
        if len(holdings) <= n:
            logger.info(f"Only {len(holdings)} holders, top{n} concentration is 100%")
            return 100.0
        
        # Get top N holdings
        top_n = np.sort(holdings)[-n:]
        top_n_sum = top_n.sum()
        holdings_sum = holdings.sum()
        
        logger.info(f"Top{n} calculation: {len(holdings)} holders, top{n} sum={top_n_sum:.0f}, total={holdings_sum:.0f}")
        
        # Use holdings_sum as denominator if total_supply is much larger
        # This handles cases where pump.fun contract holds most but isn't in holdings
        if total_supply > 0 and holdings_sum > 0:
            # If holdings_sum is less than threshold of total_supply, use holdings_sum
            # This gives accurate percentage among visible holders
            if holdings_sum / total_supply < whale_config.MISSING_CONTRACT_SUPPLY_RATIO:
                # Likely missing contract addresses - use holdings_sum for accuracy
                logger.info(f"Using holdings_sum as denominator (only {holdings_sum/total_supply*100:.1f}% of supply visible)")
                return (top_n_sum / holdings_sum) * 100
            else:
                # Use total_supply when it's close to holdings_sum
                return (top_n_sum / total_supply) * 100
        elif holdings_sum > 0:
            # Fallback to holdings_sum if total_supply is invalid
            return (top_n_sum / holdings_sum) * 100
        
        return 0.0
    
    def calculate_dev_holding_percent(
        self,
        dev_holding: float,
        total_supply: float
    ) -> float:
        """Calculate developer holding percentage"""
        if total_supply <= 0:
            return 0.0
        return (dev_holding / total_supply) * 100
    
    def calculate_sniper_holding_percent(
        self,
        sniper_holdings: List[float],
        total_supply: float
    ) -> float:
        """Calculate total sniper holdings percentage"""
        if total_supply <= 0:
            return 0.0
        return (sum(sniper_holdings) / total_supply) * 100
    
    def _calculate_holder_score(
        self,
        gini: float,
        top10_concentration: float,
        total_holders: int,
        active_holders: int
    ) -> float:
        """
        Calculate overall holder health score (0-100)
        
        Higher score = healthier distribution
        
        Components:
        - Gini score (lower is better)
        - Top 10 concentration (lower is better)
        - Active holder ratio (higher is better)
        - Total holder count bonus
        """
        # Gini score (40 points max) - lower gini is better
        # Gini < 0.3 = excellent, > 0.7 = poor
        gini_score = max(0, (1 - gini) * 40)
        
        # Top 10 concentration score (30 points max)
        # < 20% = excellent, > 70% = poor
        if top10_concentration <= 20:
            conc_score = 30
        elif top10_concentration >= 70:
            conc_score = 0
        else:
            conc_score = 30 * (1 - (top10_concentration - 20) / 50)
        
        # Active holder ratio score (20 points max)
        if total_holders > 0:
            active_ratio = active_holders / total_holders
            active_score = active_ratio * 20
        else:
            active_score = 0
        
        # Holder count bonus (10 points max)
        # More holders = more distributed = healthier
        if total_holders >= 1000:
            holder_bonus = 10
        elif total_holders >= 500:
            holder_bonus = 7
        elif total_holders >= 100:
            holder_bonus = 4
        else:
            holder_bonus = total_holders / 25  # 0-4 points
        
        total_score = gini_score + conc_score + active_score + holder_bonus
        
        return min(100, max(0, total_score))
    
    def _empty_holder_stats(self) -> HolderStats:
        """Return empty holder stats with neutral defaults"""
        return HolderStats(
            total_holders=0,
            active_holders=0,
            gini_coefficient=whale_config.GINI_NEUTRAL_DEFAULT,
            top10_concentration=0.0,
            top20_concentration=0.0,
            top50_concentration=0.0,
            median_holding=0.0,
            mean_holding=0.0,
            std_holding=0.0,
            holder_score=50.0
        )
    
    def assess_risk_level(self, stats: HolderStats) -> str:
        """
        Assess risk level based on holder metrics
        
        Returns:
            Risk level: "Low", "Medium", or "High"
        """
        risk_score = 0
        
        # Gini risk
        if stats.gini_coefficient > whale_config.GINI_HIGH_RISK:
            risk_score += 2
        elif stats.gini_coefficient > 0.4:
            risk_score += 1
        
        # Top 10 concentration risk
        if stats.top10_concentration > wallet_classification_config.TOP10_CONCENTRATION_HIGH_RISK * 100:
            risk_score += 2
        elif stats.top10_concentration > wallet_classification_config.TOP10_CONCENTRATION_MEDIUM_RISK * 100:
            risk_score += 1
        
        # Low holder count risk
        if stats.active_holders < wallet_classification_config.SAFETY_FEW_HOLDERS:
            risk_score += 2
        elif stats.active_holders < wallet_classification_config.SAFETY_MEDIUM_HOLDERS:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"


def convert_stats_to_dict(stats: HolderStats) -> Dict[str, Any]:
    """Convert HolderStats dataclass to dictionary"""
    return {
        'total_holders': stats.total_holders,
        'active_holders': stats.active_holders,
        'gini_coefficient': stats.gini_coefficient,
        'top10_concentration': stats.top10_concentration,
        'top20_concentration': stats.top20_concentration,
        'top50_concentration': stats.top50_concentration,
        'median_holding': stats.median_holding,
        'mean_holding': stats.mean_holding,
        'std_holding': stats.std_holding,
        'holder_score': stats.holder_score
    }


if __name__ == "__main__":
    print("Holder Metrics Calculator")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    
    # Simulated power-law distribution (typical for meme coins)
    holdings = np.concatenate([
        np.random.exponential(1000000, 10),      # Whales
        np.random.exponential(100000, 50),       # Large holders
        np.random.exponential(10000, 200),       # Medium holders
        np.random.exponential(1000, 500),        # Small holders
        np.random.exponential(100, 1000),        # Dust holders
    ])
    
    df = pd.DataFrame({'current_holding': holdings})
    total_supply = holdings.sum()
    
    calculator = HolderMetricsCalculator()
    stats = calculator.calculate_all_metrics(df, total_supply)
    
    print(f"\nHolder Statistics:")
    print(f"  Total Holders: {stats.total_holders:,}")
    print(f"  Active Holders: {stats.active_holders:,}")
    print(f"  Gini Coefficient: {stats.gini_coefficient:.4f}")
    print(f"  Top 10 Concentration: {stats.top10_concentration:.2f}%")
    print(f"  Top 20 Concentration: {stats.top20_concentration:.2f}%")
    print(f"  Top 50 Concentration: {stats.top50_concentration:.2f}%")
    print(f"  Median Holding: {stats.median_holding:,.2f}")
    print(f"  Mean Holding: {stats.mean_holding:,.2f}")
    print(f"  Holder Score: {stats.holder_score:.1f}/100")
    
    risk_level = calculator.assess_risk_level(stats)
    print(f"\n  Risk Level: {risk_level}")

