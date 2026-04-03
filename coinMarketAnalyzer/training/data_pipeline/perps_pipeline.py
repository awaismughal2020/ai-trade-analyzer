"""
Perps Data Pipeline
Fetches and processes data from the API to create training datasets
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import get_config, DATA_DIR

logger = logging.getLogger(__name__)


class PerpsDataPipeline:
    """
    Data pipeline for fetching and processing perps data for model training
    """
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.perps.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Output directory
        self.output_dir = DATA_DIR / "perps"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PerpsDataPipeline initialized with base URL: {self.base_url}")
    
    def fetch_candles(
        self,
        ticker: str,
        resolution: str = "1HOUR",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data via HyperEVM candle-snapshot"""
        perps_cfg = self.config.perps
        coin = perps_cfg.to_hyper_coin(ticker)
        interval = perps_cfg.to_hyper_resolution(resolution)
        
        if to_date:
            end_dt = pd.to_datetime(to_date)
        else:
            end_dt = datetime.utcnow()
        
        if from_date:
            start_dt = pd.to_datetime(from_date)
        else:
            resolution_hours = {
                "1MIN": 1/60, "5MINS": 5/60, "15MINS": 0.25,
                "30MINS": 0.5, "1HOUR": 1, "4HOURS": 4, "1DAY": 24,
            }
            hours_per_candle = resolution_hours.get(resolution, 1)
            start_dt = end_dt - timedelta(hours=limit * hours_per_candle)
        
        logger.info(f"Fetching candles for {ticker} ({coin}, {interval})...")
        
        params = {
            'coin': coin,
            'interval': interval,
            'startTime': start_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'endTime': end_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}{perps_cfg.ENDPOINT_CANDLES}",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            records = data.get('data', [])
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return pd.DataFrame()
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        df = df.rename(columns={
            't': 'timestamp', 'o': 'open', 'h': 'high',
            'l': 'low', 'c': 'close', 'v': 'base_volume', 'n': 'trade_count',
        })
        
        for col in ['open', 'high', 'low', 'close', 'base_volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        if 'base_volume' in df.columns and 'close' in df.columns:
            df['volume'] = df['base_volume'] * df['close']
        
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} candle records for {ticker}")
        return df
    
    def fetch_funding_rates(
        self, 
        ticker: str, 
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 500,
        max_pages: int = 100
    ) -> pd.DataFrame:
        """
        Fetch historical funding rate data via HyperEVM /hyper-evm/funding-history.
        
        Args:
            ticker: Trading pair (e.g., 'BTC-USD')
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            limit: Maximum unique funding records to fetch
            max_pages: Unused (kept for signature compatibility)
            
        Returns:
            DataFrame with funding rate data
        """
        perps_cfg = self.config.perps
        coin = perps_cfg.to_hyper_coin(ticker)
        
        if to_date:
            end_dt = pd.to_datetime(to_date)
        else:
            end_dt = datetime.utcnow()
        if from_date:
            start_dt = pd.to_datetime(from_date)
        else:
            start_dt = end_dt - timedelta(hours=limit)
        
        logger.info(f"Fetching funding rates for {coin}...")
        
        params = {
            'coin': coin,
            'startTime': start_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'endTime': end_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}{perps_cfg.ENDPOINT_FUNDING_HISTORY}",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Error fetching funding rates: {e}")
            return pd.DataFrame()
        
        records = data.get('historical', [])
        if not records:
            logger.warning(f"No funding rate data found for {coin}")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        
        if 'fundingRate' in df.columns:
            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        if 'premium' in df.columns:
            df['premium'] = pd.to_numeric(df['premium'], errors='coerce')
        
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        if 'fundingRate' in df.columns:
            df['nextFundingRate'] = df['fundingRate'].shift(-1)
            df['nextFundingRate'] = df['nextFundingRate'].fillna(df['fundingRate'])
        
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        logger.info(f"Extracted {len(df)} unique funding rate records for {coin}")
        return df
    
    def fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch current market data via HyperEVM meta-asset-ctxs"""
        perps_cfg = self.config.perps
        coin = perps_cfg.to_hyper_coin(ticker)
        
        try:
            response = self.session.get(
                f"{self.base_url}{perps_cfg.ENDPOINT_MARKETS}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            for entry in data.get('data', []):
                if entry.get('name') == coin:
                    return entry
            
            logger.warning(f"Coin {coin} not found in meta-asset-ctxs")
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def _fetch_whale_flow_range(self, from_date: Optional[str], to_date: Optional[str]) -> pd.DataFrame:
        """Fetch whale flow data for a date range and return as DataFrame."""
        perps_cfg = self.config.perps
        endpoint = perps_cfg.ENDPOINT_BLOCK_LIQUIDITY_WHALE_FLOW

        if to_date:
            end_ts = f"{to_date[:10]}T23:59:59.000Z"
        else:
            end_ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')
        if from_date:
            start_ts = f"{from_date[:10]}T00:00:00.000Z"
        else:
            start_ts = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%S.000Z')

        url = f"{self.base_url}{endpoint}"
        try:
            resp = self.session.get(url, params={'start_timestamp': start_ts, 'end_timestamp': end_ts}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Whale flow fetch failed: {e}")
            return pd.DataFrame()

        records = data.get('data', []) if isinstance(data, dict) else []
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        if 'net_flow' in df.columns:
            df = df.rename(columns={'net_flow': 'whale_flow_net'})
        if 'long_volume_usdc_perp' in df.columns and 'total_volume_usdc_perp' in df.columns:
            total = df['total_volume_usdc_perp'].replace(0, float('nan'))
            df['whale_flow_long_ratio'] = (df['long_volume_usdc_perp'] / total).fillna(0.5)
        else:
            df['whale_flow_long_ratio'] = 0.5
        if 'count_whales_perp' in df.columns:
            df = df.rename(columns={'count_whales_perp': 'whale_flow_count_whales'})
        else:
            df['whale_flow_count_whales'] = 0

        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} whale flow records for training")
        return df

    def fetch_and_merge_data(
        self,
        tickers: List[str],
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        candle_limit: int = 2000,
        funding_limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch and merge data for multiple tickers
        
        Args:
            tickers: List of ticker symbols (e.g., ['BTC-USD', 'ETH-USD'])
            from_date: Start date (YYYY-MM-DD) - optional, takes precedence over limit
            to_date: End date (YYYY-MM-DD) - optional, takes precedence over limit
            candle_limit: Max candles per ticker (used if dates not provided)
            funding_limit: Max funding records per ticker (used if dates not provided)
            
        Returns:
            Merged DataFrame ready for training
        """
        date_info = f" from {from_date} to {to_date}" if from_date and to_date else f" (limit: {candle_limit})"
        logger.info(f"Fetching data for {len(tickers)} tickers{date_info}...")
        
        all_data = []
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            
            # Fetch candles with date range
            candles = self.fetch_candles(
                ticker, 
                from_date=from_date,
                to_date=to_date,
                limit=candle_limit
            )
            if candles.empty:
                logger.warning(f"No candle data for {ticker}, skipping...")
                continue
            
            # Fetch funding rates with date range
            funding = self.fetch_funding_rates(
                ticker,
                from_date=from_date,
                to_date=to_date,
                limit=funding_limit
            )
            
            # Merge funding into candles
            if not funding.empty and 'timestamp' in candles.columns and 'timestamp' in funding.columns:
                candles = candles.sort_values('timestamp')
                funding = funding.sort_values('timestamp')
                
                candles = pd.merge_asof(
                    candles,
                    funding[['timestamp', 'fundingRate', 'premium', 'nextFundingRate']].rename(columns={
                        'fundingRate': 'funding_rate',
                        'nextFundingRate': 'next_funding_rate'
                    }),
                    on='timestamp',
                    direction='backward'
                )
            
            # Fetch market-wide whale flow for the same date range and merge onto candles
            try:
                whale_flow_data = self._fetch_whale_flow_range(from_date, to_date)
                if not whale_flow_data.empty and 'timestamp' in candles.columns:
                    candles = pd.merge_asof(
                        candles.sort_values('timestamp'),
                        whale_flow_data[['timestamp', 'whale_flow_net', 'whale_flow_long_ratio', 'whale_flow_count_whales']].sort_values('timestamp'),
                        on='timestamp',
                        direction='backward'
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch whale flow for training: {e}")
            
            # Add ticker column
            candles['ticker'] = ticker
            
            all_data.append(candles)
            
            # Rate limiting
            time.sleep(0.5)
        
        if not all_data:
            logger.error("No data fetched for any ticker")
            return pd.DataFrame()
        
        # Combine all tickers
        df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Total records: {len(df)}")
        
        return df
    
    def run_pipeline(
        self,
        tickers: Optional[List[str]] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        output_filename: str = "train_data.csv",
        candle_limit: int = 2000,
        funding_limit: int = 500
    ) -> Dict[str, Any]:
        """
        Run the complete data pipeline
        
        Args:
            tickers: List of tickers to fetch (default: major perps pairs)
            from_date: Start date (YYYY-MM-DD) - optional, takes precedence over limit
            to_date: End date (YYYY-MM-DD) - optional, takes precedence over limit
            output_filename: Output CSV filename
            candle_limit: Max candles per ticker (used if dates not provided)
            funding_limit: Max funding records per ticker (used if dates not provided)
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        
        if tickers is None:
            tickers = [
                'BTC-USD', 'ETH-USD', 'SOL-USD', 'HYPE-USD',
                'ARB-USD', 'OP-USD', 'AVAX-USD', 'MATIC-USD'
            ]
        
        date_info = f" from {from_date} to {to_date}" if from_date and to_date else ""
        logger.info(f"Running perps data pipeline for {len(tickers)} tickers{date_info}...")
        
        # Fetch and merge data
        df = self.fetch_and_merge_data(
            tickers, 
            from_date=from_date,
            to_date=to_date,
            candle_limit=candle_limit, 
            funding_limit=funding_limit
        )
        
        if df.empty:
            return {
                'success': False,
                'error': 'No data fetched',
                'tickers': tickers
            }
        
        # Save to CSV
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        results = {
            'success': True,
            'output_path': str(output_path),
            'total_records': len(df),
            'tickers': tickers,
            'columns': list(df.columns),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
            },
            'elapsed_seconds': round(elapsed, 2)
        }
        
        logger.info(f"Pipeline complete: {len(df)} records saved to {output_path}")
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = PerpsDataPipeline()
    
    # Run pipeline with date range example
    results = pipeline.run_pipeline(
        tickers=['BTC-USD', 'ETH-USD'],
        from_date='2025-01-01',
        to_date='2025-12-31'
    )
    
    print(f"\nPipeline Results:")
    print(f"  Success: {results['success']}")
    print(f"  Records: {results.get('total_records', 0)}")
    print(f"  Output: {results.get('output_path', 'N/A')}")
    print(f"  Date Range: {results.get('date_range', {})}")
