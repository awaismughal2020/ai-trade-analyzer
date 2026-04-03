"""
Meme Token Data Pipeline
Fetches and processes meme token data to create training datasets

USAGE:
1. Option A - Use pre-collected data (recommended):
   - First run POST /data/fetch-all to collect candles, holders, trades to CSV files
   - Then run training with use_csv_data=True to use the collected data
   
2. Option B - Fresh fetch from API:
   - Run training directly (only works for older mints with aggregated candle data)
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_config, DATA_DIR, CANDLES_DIR, HOLDERS_DIR, MINTS_DIR, TRADES_DIR
from core.data_fetcher import DataFetcher
from engines.technical_engine import TechnicalIndicatorEngine
from engines.holder_metrics import HolderMetricsCalculator
from engines.whale_engine import WhaleEngine

logger = logging.getLogger(__name__)


class MemeDataPipeline:
    """
    Data pipeline for fetching and processing meme token data for model training
    
    Supports two modes:
    1. use_csv_data=True: Load from pre-collected CSV files (recommended)
    2. use_csv_data=False: Fetch fresh from API (only for older mints with data)
    """
    
    def __init__(self):
        self.config = get_config()
        self.data_fetcher = DataFetcher()
        self.technical_engine = TechnicalIndicatorEngine()
        self.holder_metrics = HolderMetricsCalculator()
        self.whale_engine = WhaleEngine()
        
        # Output directory
        self.output_dir = DATA_DIR / "meme"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV data cache
        self._candles_cache = None
        self._holders_cache = None
        
        logger.info("MemeDataPipeline initialized")
    
    def load_csv_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV data from data directories collected by /data/fetch-all
        
        Returns:
            Dictionary with 'candles', 'holders', 'mints' DataFrames
        """
        logger.info("Loading pre-collected CSV data from data directories...")
        
        result = {
            'candles': pd.DataFrame(),
            'holders': pd.DataFrame(),
            'mints': pd.DataFrame()
        }
        
        # Load all candles CSVs
        candles_pattern = str(CANDLES_DIR / "candles_*.csv")
        candles_files = glob.glob(candles_pattern)
        if candles_files:
            all_candles = []
            for f in candles_files:
                try:
                    df = pd.read_csv(f)
                    if len(df) > 0:
                        all_candles.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
            if all_candles:
                result['candles'] = pd.concat(all_candles, ignore_index=True)
                logger.info(f"Loaded {len(result['candles'])} candle records from {len(candles_files)} files")
        else:
            logger.warning(f"No candle CSV files found in {CANDLES_DIR}")
        
        # Load all holders CSVs
        # NOTE: Holder CSVs do NOT contain a mint column. The mint identity is
        # encoded in the filename as a truncated prefix, e.g.:
        #   holders_11Dhnyrx_20260203_101757.csv -> mint starts with "11Dhnyrx"
        # We extract this prefix and add it as a column so we can match later.
        holders_pattern = str(HOLDERS_DIR / "holders_*.csv")
        holders_files = glob.glob(holders_pattern)
        if holders_files:
            all_holders = []
            for f in holders_files:
                try:
                    df = pd.read_csv(f)
                    if len(df) > 0:
                        # Extract mint prefix from filename: holders_{PREFIX}_{date}_{time}.csv
                        basename = os.path.basename(f)
                        parts = basename.replace('holders_', '').split('_')
                        mint_prefix = parts[0] if parts else ''
                        df['mint_prefix'] = mint_prefix
                        all_holders.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
            if all_holders:
                result['holders'] = pd.concat(all_holders, ignore_index=True)
                logger.info(f"Loaded {len(result['holders'])} holder records from {len(holders_files)} files")
                
                # Build a prefix -> full-mint lookup from candle data if available
                if not result['candles'].empty:
                    mint_col = None
                    for col in ['Mint', 'mint', 'MINT', 'token', 'Token']:
                        if col in result['candles'].columns:
                            mint_col = col
                            break
                    if mint_col:
                        unique_mints = result['candles'][mint_col].unique()
                        prefix_to_mint = {}
                        for m in unique_mints:
                            prefix = m[:8] if len(m) >= 8 else m
                            prefix_to_mint[prefix] = m
                        # Map prefix to full mint address
                        result['holders']['Mint'] = result['holders']['mint_prefix'].map(prefix_to_mint)
                        matched = result['holders']['Mint'].notna().sum()
                        logger.info(f"Matched {matched}/{len(result['holders'])} holder records to full mint addresses")
        else:
            logger.warning(f"No holder CSV files found in {HOLDERS_DIR}")
        
        # Load mints CSVs
        mints_pattern = str(MINTS_DIR / "mints_*.csv")
        mints_files = glob.glob(mints_pattern)
        if mints_files:
            all_mints = []
            for f in mints_files:
                try:
                    df = pd.read_csv(f)
                    if len(df) > 0:
                        all_mints.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")
            if all_mints:
                result['mints'] = pd.concat(all_mints, ignore_index=True).drop_duplicates()
                logger.info(f"Loaded {len(result['mints'])} unique mints from {len(mints_files)} files")
        
        return result
    
    def get_mint_data_from_csv(self, mint: str, csv_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific mint from pre-loaded CSV data
        
        Args:
            mint: Mint address
            csv_data: Dictionary with candles, holders DataFrames
            
        Returns:
            Dictionary with mint data or None if not found
        """
        try:
            # Find mint column (case insensitive)
            candles_df = csv_data['candles']
            holders_df = csv_data['holders']
            
            if candles_df.empty:
                return None
            
            # Find mint column in candles
            mint_col = None
            for col in ['Mint', 'mint', 'MINT', 'token', 'Token']:
                if col in candles_df.columns:
                    mint_col = col
                    break
            
            if not mint_col:
                logger.warning(f"No mint column found in candles CSV")
                return None
            
            # Filter for this mint
            mint_candles = candles_df[candles_df[mint_col] == mint].copy()
            
            if len(mint_candles) == 0:
                logger.debug(f"No candles found for {mint[:8]}... in CSV data")
                return None
            
            # Filter holders for this mint
            # Holder CSVs originally had no mint column; load_csv_data() now adds
            # a 'Mint' column by mapping the filename prefix to the full mint address.
            # Fallback: try prefix matching if the Mint column mapping failed.
            mint_holders = pd.DataFrame()
            if not holders_df.empty:
                # Primary: use the Mint column added during load
                if 'Mint' in holders_df.columns:
                    mint_holders = holders_df[holders_df['Mint'] == mint].copy()
                
                # Fallback: prefix matching via mint_prefix column
                if mint_holders.empty and 'mint_prefix' in holders_df.columns:
                    mint_prefix = mint[:8] if len(mint) >= 8 else mint
                    mint_holders = holders_df[holders_df['mint_prefix'] == mint_prefix].copy()
                
                if not mint_holders.empty:
                    logger.debug(f"Found {len(mint_holders)} holder records for {mint[:8]}...")
            
            # Standardize candle columns
            mint_candles = self.data_fetcher.standardize_candles(mint_candles)
            
            # Extract metadata from candle data
            metadata = {'mint': mint}
            if 'TotalSupply' in mint_candles.columns:
                ts_val = mint_candles['TotalSupply'].iloc[0]
                if pd.notna(ts_val) and ts_val > 0:
                    metadata['totalSupply'] = float(ts_val)
            elif 'totalSupply' in mint_candles.columns:
                ts_val = mint_candles['totalSupply'].iloc[0]
                if pd.notna(ts_val) and ts_val > 0:
                    metadata['totalSupply'] = float(ts_val)
            
            return {
                'candles': mint_candles,
                'holders': mint_holders,
                'user_holdings': pd.DataFrame(),
                'holders_historical': mint_holders,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting CSV data for {mint[:8]}...: {e}")
            return None
    
    def fetch_mint_data(
        self,
        mint: str,
        candle_days: int = 90
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch all data for a single mint
        
        Args:
            mint: Token mint address
            candle_days: Number of days of candle data
            
        Returns:
            Dictionary with all fetched data or None if failed
        """
        try:
            logger.info(f"Fetching data for mint {mint[:8]}...")
            
            # Fetch complete data using data_fetcher
            data = self.data_fetcher.fetch_complete_data_v2(
                mint=mint,
                candle_days=candle_days,
                holder_limit=1000
            )
            
            if not data:
                logger.warning(f"No data returned for {mint[:8]}")
                return None
            
            # Check if we have minimum required data
            if 'candles' not in data or data['candles'].empty:
                logger.warning(f"No candle data for {mint[:8]}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {mint}: {e}")
            return None
    
    def extract_features_at_index(
        self,
        mint: str,
        data: Dict[str, Any],
        candle_index: int = -1
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features from fetched data at a specific candle index.
        
        This allows creating multiple training samples per token by extracting
        features at different points in the candle history.
        
        Args:
            mint: Token mint address
            data: Dictionary with fetched data
            candle_index: Index of the "current" candle to extract features up to.
                          Use -1 for the last candle (legacy behavior).
            
        Returns:
            Dictionary with extracted features or None if failed
        """
        try:
            features = {'mint_address': mint}
            
            # Get dataframes
            candles_df = data.get('candles', pd.DataFrame())
            holders_df = data.get('holders', pd.DataFrame())
            holders_historical_df = data.get('holders_historical', pd.DataFrame())
            trades_df = data.get('trades', pd.DataFrame())
            metadata = data.get('metadata', {}) or {}
            
            if candles_df.empty:
                return None
            
            # Slice candles up to the requested index (only see past)
            if candle_index == -1:
                candles_slice = candles_df
            else:
                candles_slice = candles_df.iloc[:candle_index + 1]
            
            if len(candles_slice) < 20:
                return None  # Need minimum candles for indicators
            
            # Get latest candle in the slice
            latest_candle = candles_slice.iloc[-1]
            
            # ========== TECHNICAL INDICATORS ==========
            tech_signals = self.technical_engine.calculate_all_indicators(candles_slice)
            
            features['rsi'] = tech_signals.rsi if tech_signals.rsi is not None else 50
            features['rsi_signal'] = 1 if features['rsi'] > 70 else -1 if features['rsi'] < 30 else 0
            features['ema_20'] = tech_signals.ema_20 if tech_signals.ema_20 is not None else 0
            features['ema_50'] = tech_signals.ema_50 if tech_signals.ema_50 is not None else 0
            features['ema_cross'] = 1 if features['ema_20'] > features['ema_50'] else -1
            features['macd_line'] = tech_signals.macd_line if tech_signals.macd_line is not None else 0
            features['macd_signal'] = tech_signals.macd_signal if tech_signals.macd_signal is not None else 0
            features['macd_histogram'] = tech_signals.macd_histogram if tech_signals.macd_histogram is not None else 0
            features['macd_trend'] = 1 if features['macd_histogram'] > 0 else -1
            features['bb_upper'] = tech_signals.bb_upper if tech_signals.bb_upper is not None else 0
            features['bb_middle'] = tech_signals.bb_middle if tech_signals.bb_middle is not None else 0
            features['bb_lower'] = tech_signals.bb_lower if tech_signals.bb_lower is not None else 0
            features['bb_position'] = tech_signals.bb_position if tech_signals.bb_position is not None else 0.5
            
            # ========== VOLUME ANALYSIS ==========
            vol_col = 'Volume' if 'Volume' in candles_slice.columns else 'volume'
            features['volume'] = latest_candle[vol_col] if vol_col in latest_candle.index else 0
            features['volume_ma'] = candles_slice[vol_col].tail(20).mean() if vol_col in candles_slice.columns else 0
            features['volume_ratio'] = features['volume'] / features['volume_ma'] if features['volume_ma'] > 0 else 1
            
            buy_vol = latest_candle['BuyVolume'] if 'BuyVolume' in latest_candle.index else (latest_candle['buyVolume'] if 'buyVolume' in latest_candle.index else 0)
            sell_vol = latest_candle['SellVolume'] if 'SellVolume' in latest_candle.index else (latest_candle['sellVolume'] if 'sellVolume' in latest_candle.index else 1)
            features['buy_sell_ratio'] = buy_vol / (sell_vol + 1e-10)
            
            # ========== PRICE MOMENTUM ==========
            close_col = 'Close' if 'Close' in candles_slice.columns else 'close'
            closes = candles_slice[close_col].values
            
            if len(closes) >= 2:
                features['price_momentum'] = (closes[-1] - closes[-2]) / (closes[-2] + 1e-10)
            else:
                features['price_momentum'] = 0
            
            features['volatility'] = candles_slice[close_col].tail(20).std() / (candles_slice[close_col].tail(20).mean() + 1e-10)
            features['atr'] = features['volatility']
            
            # ========== HOLDER METRICS ==========
            if not holders_df.empty:
                total_supply = metadata.get('totalSupply', 0)
                if total_supply == 0 and 'totalSupply' in holders_df.columns:
                    total_supply = holders_df['totalSupply'].iloc[0]
                # If totalSupply is in candle data columns (via standardize), check that too
                if total_supply == 0 and 'TotalSupply' in candles_slice.columns:
                    ts_val = candles_slice['TotalSupply'].iloc[-1]
                    if pd.notna(ts_val) and ts_val > 0:
                        total_supply = float(ts_val)
                
                holder_stats = self.holder_metrics.calculate_all_metrics(holders_df, total_supply)
                
                features['gini_coefficient'] = holder_stats.gini_coefficient if holder_stats.gini_coefficient is not None else 0.5
                features['top10_concentration'] = holder_stats.top10_concentration if holder_stats.top10_concentration is not None else 0
                # HolderStats uses 'total_holders' and 'active_holders', not 'holder_count'/'unique_wallets'
                features['holder_count'] = holder_stats.total_holders if holder_stats.total_holders is not None else 0
                features['unique_wallets'] = holder_stats.active_holders if holder_stats.active_holders is not None else 0
                features['holder_growth_rate'] = 0
            else:
                features['gini_coefficient'] = 0.5
                features['top10_concentration'] = 0
                features['holder_count'] = 0
                features['unique_wallets'] = 0
                features['holder_growth_rate'] = 0
            
            # ========== WHALE METRICS ==========
            whale_df = holders_historical_df if not holders_historical_df.empty else holders_df
            
            if not whale_df.empty:
                total_supply = metadata.get('totalSupply', 0)
                if total_supply == 0 and 'totalSupply' in holders_df.columns:
                    total_supply = holders_df['totalSupply'].iloc[0]
                
                whale_metrics = self.whale_engine.analyze_token(whale_df, metadata, total_supply)
                
                features['whale_buy_volume'] = whale_metrics.whale_buy_volume if whale_metrics.whale_buy_volume is not None else 0
                features['whale_sell_volume'] = whale_metrics.whale_sell_volume if whale_metrics.whale_sell_volume is not None else 0
                features['whale_net_volume'] = whale_metrics.whale_net_volume if whale_metrics.whale_net_volume is not None else 0
                features['whale_state_encoded'] = {'Accumulation': 1, 'Distribution': -1, 'Stability': 0}.get(whale_metrics.whale_state, 0)
                features['dominant_whale_pct'] = 0
            else:
                features['whale_buy_volume'] = 0
                features['whale_sell_volume'] = 0
                features['whale_net_volume'] = 0
                features['whale_state_encoded'] = 0
                features['dominant_whale_pct'] = 0
            
            # ========== MARKET CONTEXT ==========
            features['market_cap'] = metadata.get('marketCap', 0)
            features['liquidity'] = metadata.get('liquidity', 0)
            features['price_change_24h'] = metadata.get('priceChange24h', 0)
            
            # ========== USER PROFILE (Placeholder) ==========
            features['user_pnl'] = 0
            features['user_win_rate'] = 0
            features['user_trade_count'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {mint}: {e}")
            return None
    
    def extract_features(
        self,
        mint: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features from fetched data (uses latest candle).
        
        This is the legacy method kept for backward compatibility with
        the prediction flow. For training, use extract_features_at_index().
        
        Args:
            mint: Token mint address
            data: Dictionary with fetched data
            
        Returns:
            Dictionary with extracted features or None if failed
        """
        return self.extract_features_at_index(mint, data, candle_index=-1)
    
    def create_target_label(
        self,
        mint: str,
        current_price: float,
        candles_df: pd.DataFrame,
        lookforward_hours: int = 24
    ) -> int:
        """
        Create target label based on future price movement (FORWARD-LOOKING).
        
        This is the legacy single-sample method. For multi-sample training,
        use create_forward_label() which takes a candle index.
        
        Args:
            mint: Token mint address
            current_price: Current token price
            candles_df: Full candle history
            lookforward_hours: Hours to look forward for price change
            
        Returns:
            0 = SELL/HOLD, 1 = BUY
        """
        try:
            if len(candles_df) < 20:
                return 0
            
            close_col = 'Close' if 'Close' in candles_df.columns else 'close'
            closes = candles_df[close_col].values
            
            # Use the midpoint of candles as our "current" point, and look forward
            mid_idx = len(closes) // 2
            if mid_idx < 10:
                mid_idx = 10
            
            current = closes[mid_idx]
            future_closes = closes[mid_idx + 1:]
            
            if len(future_closes) < 5:
                return 0
            
            max_future = np.max(future_closes)
            max_gain = (max_future - current) / (current + 1e-10)
            
            # BUY if price increases by >= 5% at any point in the future window
            if max_gain >= 0.05:
                return 1
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error creating target label for {mint}: {e}")
            return 0
    
    def create_forward_label(
        self,
        candles_df: pd.DataFrame,
        current_idx: int,
        lookforward_candles: int = 24,
        buy_threshold: float = 0.05
    ) -> Optional[int]:
        """
        Create a forward-looking target label at a specific candle index.
        
        Looks at FUTURE price movement after current_idx to determine
        whether buying at this point would have been profitable.
        
        Args:
            candles_df: Full candle DataFrame
            current_idx: Index of the "current" candle (features are computed up to here)
            lookforward_candles: Number of future candles to check
            buy_threshold: Minimum price increase to label as BUY (default 5%)
            
        Returns:
            1 = BUY (price increased >= threshold in future window)
            0 = SELL/HOLD (price did not increase enough)
            None = insufficient future data (skip this sample)
        """
        try:
            close_col = 'Close' if 'Close' in candles_df.columns else 'close'
            
            # Need future candles to create label
            if current_idx + 5 >= len(candles_df):
                return None  # Not enough future data
            
            current_price = candles_df[close_col].iloc[current_idx]
            if current_price <= 0:
                return None
            
            # Look at future candles (up to lookforward_candles ahead)
            future_end = min(current_idx + lookforward_candles + 1, len(candles_df))
            future_prices = candles_df[close_col].iloc[current_idx + 1:future_end].values
            
            if len(future_prices) == 0:
                return None
            
            max_future_price = np.max(future_prices)
            max_gain = (max_future_price - current_price) / current_price
            
            return 1 if max_gain >= buy_threshold else 0
            
        except Exception as e:
            logger.error(f"Error creating forward label: {e}")
            return None
    
    def _get_sample_indices(self, n_candles: int, min_history: int = 50, lookforward: int = 24, max_samples: int = 5) -> List[int]:
        """
        Calculate evenly-spaced candle indices for multi-sample extraction.
        
        Args:
            n_candles: Total number of candles for this token
            min_history: Minimum candles needed before a sample point (for indicators)
            lookforward: Candles needed after a sample point (for forward label)
            max_samples: Maximum samples per token
            
        Returns:
            List of candle indices to sample at
        """
        # Earliest valid index: need min_history candles before
        earliest = min_history
        # Latest valid index: need lookforward candles after
        latest = n_candles - lookforward - 1
        
        if latest <= earliest:
            # Only one sample possible at the midpoint
            mid = n_candles // 2
            if mid >= min_history and mid + 5 < n_candles:
                return [mid]
            return []
        
        usable_range = latest - earliest
        n_samples = min(max_samples, max(1, usable_range // 20))  # At least 20 candles apart
        
        if n_samples == 1:
            return [earliest + usable_range // 2]
        
        step = usable_range / (n_samples - 1)
        indices = [earliest + int(step * i) for i in range(n_samples)]
        return indices
    
    def process_mints(
        self,
        mints: List[str],
        candle_days: int = 90
    ) -> pd.DataFrame:
        """
        Process multiple mints and extract features with multi-sample extraction.
        
        For each token, creates multiple training samples at different points in
        the candle history using forward-looking labels.
        
        Args:
            mints: List of mint addresses
            candle_days: Number of days of candle data
            
        Returns:
            DataFrame with all features and labels
        """
        logger.info(f"Processing {len(mints)} mints (multi-sample, forward-looking)...")
        
        all_features = []
        
        for i, mint in enumerate(mints):
            try:
                if i > 0 and i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(mints)} mints processed, {len(all_features)} samples")
                    time.sleep(2)
                
                data = self.fetch_mint_data(mint, candle_days)
                if not data:
                    continue
                
                candles_df = data.get('candles', pd.DataFrame())
                if candles_df.empty or len(candles_df) < 30:
                    continue
                
                # Get sample points for this token
                sample_indices = self._get_sample_indices(len(candles_df))
                
                for idx in sample_indices:
                    features = self.extract_features_at_index(mint, data, candle_index=idx)
                    if not features:
                        continue
                    
                    label = self.create_forward_label(candles_df, current_idx=idx)
                    if label is None:
                        continue
                    
                    features['target'] = label
                    all_features.append(features)
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing mint {mint}: {e}")
                continue
        
        if not all_features:
            logger.error("No features extracted from any mint")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_features)
        
        logger.info(f"Extracted {len(df)} samples from {len(mints)} mints (multi-sample)")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def run_pipeline(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        max_mints: int = 100,
        candle_days: int = 90,
        output_filename: str = "train_data.csv",
        use_csv_data: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete meme data pipeline
        
        Args:
            from_date: Start date (YYYY-MM-DD) for mint discovery
            to_date: End date (YYYY-MM-DD) for mint discovery
            max_mints: Maximum number of mints to process
            candle_days: Number of days of candle data
            output_filename: Output CSV filename
            use_csv_data: If True, use pre-collected CSV data from /data/fetch-all (RECOMMENDED)
                          If False, fetch fresh from API (only works for older mints)
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        
        # Set default dates if not provided
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        logger.info(f"Running meme data pipeline from {from_date} to {to_date}...")
        logger.info(f"Mode: {'Using pre-collected CSV data' if use_csv_data else 'Fetching fresh from API'}")
        
        # Step 0: Load CSV data if using that mode
        csv_data = None
        if use_csv_data:
            csv_data = self.load_csv_data()
            if csv_data['candles'].empty:
                logger.warning("No CSV candle data found. Falling back to API fetch mode.")
                logger.warning("TIP: Run POST /data/fetch-all first to collect training data.")
                use_csv_data = False
            else:
                logger.info(f"CSV data loaded: {len(csv_data['candles'])} candles, {len(csv_data['holders'])} holders")
        
        # Step 1: Get mints to process
        logger.info("Step 1: Getting mints to process...")
        
        if use_csv_data and csv_data and not csv_data['candles'].empty:
            # Get unique mints from the CSV data
            mint_col = None
            for col in ['Mint', 'mint', 'MINT', 'token', 'Token']:
                if col in csv_data['candles'].columns:
                    mint_col = col
                    break
            
            if mint_col:
                mints = csv_data['candles'][mint_col].unique().tolist()[:max_mints]
                logger.info(f"Found {len(mints)} unique mints in CSV data")
            else:
                return {
                    'success': False,
                    'error': 'Could not find mint column in CSV data',
                    'columns': list(csv_data['candles'].columns)
                }
        else:
            # Fetch mints from API
            mints_df = self.data_fetcher.fetch_mints_range(from_date, to_date)
            
            if mints_df.empty:
                return {
                    'success': False,
                    'error': 'No mints found in date range. TIP: Run POST /data/fetch-all first to collect data.',
                    'date_range': {'start': from_date, 'end': to_date}
                }
            
            logger.info(f"Found {len(mints_df)} mints in date range")
            
            # Extract mint addresses
            mint_column = None
            for col in ['Mint', 'mint', 'address', 'mintAddress', 'mint_address']:
                if col in mints_df.columns:
                    mint_column = col
                    break
            
            if not mint_column:
                return {
                    'success': False,
                    'error': 'Could not find mint address column in response',
                    'columns': list(mints_df.columns)
                }
            
            mints = mints_df[mint_column].unique().tolist()[:max_mints]
        
        logger.info(f"Processing up to {len(mints)} mints...")
        
        # Step 2: Process mints and extract features
        logger.info("Step 2: Extracting features from mints...")
        
        if use_csv_data and csv_data:
            # Process mints using CSV data
            df = self.process_mints_from_csv(mints, csv_data)
        else:
            # Process mints by fetching from API
            df = self.process_mints(mints, candle_days)
        
        if df.empty:
            error_msg = 'No features extracted from mints'
            if use_csv_data:
                error_msg += '. CSV data may be incomplete - try running POST /data/fetch-all with a date range that has more established mints.'
            else:
                error_msg += '. API may not have candle data for recent mints. Try using use_csv_data=True after running /data/fetch-all.'
            return {
                'success': False,
                'error': error_msg,
                'mints_attempted': len(mints)
            }
        
        # Step 3: Save to CSV
        logger.info("Step 3: Saving training data...")
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Count unique mints in the output
        n_unique_mints = df['mint_address'].nunique() if 'mint_address' in df.columns else len(df)
        
        results = {
            'success': True,
            'output_path': str(output_path),
            'total_records': len(df),
            'mints_processed': n_unique_mints,
            'mints_attempted': len(mints),
            'samples_per_mint': round(len(df) / max(n_unique_mints, 1), 1),
            'features': list(df.columns),
            'target_distribution': df['target'].value_counts().to_dict(),
            'date_range': {'start': from_date, 'end': to_date},
            'data_source': 'csv' if use_csv_data else 'api',
            'elapsed_seconds': round(elapsed, 2),
            'labeling_method': 'forward-looking (5% gain in next 24 candles)'
        }
        
        logger.info(f"Pipeline complete: {len(df)} records saved to {output_path}")
        logger.info(f"Target distribution: {results['target_distribution']}")
        
        return results
    
    def process_mints_from_csv(
        self,
        mints: List[str],
        csv_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Process mints using pre-loaded CSV data with multi-sample extraction.
        
        For each token, creates multiple training samples at different points in
        the candle history using forward-looking labels. This produces a much
        larger and more balanced training dataset.
        
        Args:
            mints: List of mint addresses
            csv_data: Dictionary with candles, holders DataFrames
            
        Returns:
            DataFrame with all features and labels
        """
        logger.info(f"Processing {len(mints)} mints from CSV data (multi-sample, forward-looking)...")
        
        all_features = []
        mints_processed = 0
        
        for i, mint in enumerate(mints):
            try:
                if i > 0 and i % 100 == 0:
                    logger.info(f"Progress: {i}/{len(mints)} mints scanned, {mints_processed} with data, {len(all_features)} samples")
                
                data = self.get_mint_data_from_csv(mint, csv_data)
                if not data:
                    continue
                
                candles_df = data.get('candles', pd.DataFrame())
                if candles_df.empty or len(candles_df) < 30:
                    continue
                
                # Get sample points for this token
                sample_indices = self._get_sample_indices(len(candles_df))
                
                mint_samples = 0
                for idx in sample_indices:
                    features = self.extract_features_at_index(mint, data, candle_index=idx)
                    if not features:
                        continue
                    
                    label = self.create_forward_label(candles_df, current_idx=idx)
                    if label is None:
                        continue
                    
                    features['target'] = label
                    all_features.append(features)
                    mint_samples += 1
                
                if mint_samples > 0:
                    mints_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing mint {mint[:8]}...: {e}")
                continue
        
        if not all_features:
            logger.error("No features extracted from any mint in CSV data")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_features)
        
        logger.info(f"Extracted {len(df)} samples from {mints_processed} mints (multi-sample, from CSV data)")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = MemeDataPipeline()
    
    # Run pipeline with date range
    results = pipeline.run_pipeline(
        from_date='2025-01-01',
        to_date='2025-01-15',
        max_mints=50
    )
    
    print(f"\nPipeline Results:")
    print(f"  Success: {results['success']}")
    print(f"  Records: {results.get('total_records', 0)}")
    print(f"  Output: {results.get('output_path', 'N/A')}")
    print(f"  Target Distribution: {results.get('target_distribution', {})}")

