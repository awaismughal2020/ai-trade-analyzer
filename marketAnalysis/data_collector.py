"""
Data Collection Module for CoinGecko API
Handles fetching OHLC and volume data for cryptocurrency analysis
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class CoinGeckoDataCollector:
    def __init__(self):
        self.base_url = os.getenv('COINGECKO_BASE_URL')
        self.api_key = os.getenv('COINGECKO_API_KEY')
        self.coins = os.getenv('TARGET_COINS', '').split(',')
        self.rate_limit = int(os.getenv('API_RATE_LIMIT_SECONDS', 3))

        # Validate configuration
        if not self.base_url:
            raise ValueError("COINGECKO_BASE_URL not found in environment variables")

        if not self.coins or self.coins == ['']:
            raise ValueError("TARGET_COINS not found in environment variables")

        # Remove any whitespace from coin names
        self.coins = [coin.strip() for coin in self.coins if coin.strip()]

        print(f"Initialized CoinGecko collector for coins: {self.coins}")

    def _get_headers(self):
        """Get headers for API requests"""
        headers = {
            'User-Agent': 'Trading-Assistant/1.0',
            'Accept': 'application/json'
        }

        # Add API key if provided
        if self.api_key and self.api_key != 'your_api_key_here':
            headers['x-cg-demo-api-key'] = self.api_key

        return headers

    def _make_request(self, url, params=None):
        """Make API request with error handling and rate limiting"""
        try:
            headers = self._get_headers()
            response = requests.get(url, params=params, headers=headers, timeout=30)

            # Check for rate limiting
            if response.status_code == 429:
                print("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                response = requests.get(url, params=params, headers=headers, timeout=30)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def get_historical_ohlc(self, coin_id, days=180):
        """
        Fetch OHLC data from CoinGecko

        Args:
            coin_id (str): CoinGecko coin identifier
            days (int): Number of days of historical data

        Returns:
            pd.DataFrame: OHLC data with timestamp, open, high, low, close
        """
        print(f"Fetching OHLC data for {coin_id} ({days} days)...")

        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }

        data = self._make_request(url, params)

        if not data:
            print(f"Failed to fetch OHLC data for {coin_id}")
            return None

        try:
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['coin_id'] = coin_id

            # Ensure proper data types
            price_columns = ['open', 'high', 'low', 'close']
            df[price_columns] = df[price_columns].astype(float)

            return df

        except Exception as e:
            print(f"Error processing OHLC data for {coin_id}: {e}")
            return None

    def get_volume_data(self, coin_id, days=180):
        """
        Fetch volume data from market chart endpoint

        Args:
            coin_id (str): CoinGecko coin identifier
            days (int): Number of days of historical data

        Returns:
            pd.DataFrame: Volume data with timestamp and volume
        """
        print(f"Fetching volume data for {coin_id}...")

        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily" if days > 90 else "hourly"
        }

        data = self._make_request(url, params)

        if not data or 'total_volumes' not in data:
            print(f"Failed to fetch volume data for {coin_id}")
            return pd.DataFrame(columns=['timestamp', 'volume'])

        try:
            volumes = data['total_volumes']
            df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['volume'] = df['volume'].astype(float)

            return df

        except Exception as e:
            print(f"Error processing volume data for {coin_id}: {e}")
            return pd.DataFrame(columns=['timestamp', 'volume'])

    def get_coin_data(self, coin_id, days=None):
        """
        Get complete OHLCV data for a single coin

        Args:
            coin_id (str): CoinGecko coin identifier
            days (int): Number of days (defaults to env variable)

        Returns:
            pd.DataFrame: Complete OHLCV data
        """
        if days is None:
            days = int(os.getenv('HISTORICAL_DAYS', 180))

        # Get OHLC data
        ohlc_data = self.get_historical_ohlc(coin_id, days)
        if ohlc_data is None or ohlc_data.empty:
            return None

        # Get volume data
        volume_data = self.get_volume_data(coin_id, days)

        # Merge OHLC and volume data
        if not volume_data.empty:
            # Round timestamps to nearest hour for better matching
            ohlc_data['timestamp_rounded'] = ohlc_data['timestamp'].dt.round('H')
            volume_data['timestamp_rounded'] = volume_data['timestamp'].dt.round('H')

            merged_data = pd.merge(
                ohlc_data,
                volume_data[['timestamp_rounded', 'volume']],
                on='timestamp_rounded',
                how='left'
            )

            # Drop the rounded timestamp column
            merged_data = merged_data.drop('timestamp_rounded', axis=1)
        else:
            merged_data = ohlc_data.copy()
            merged_data['volume'] = 0  # Fill with zeros if no volume data

        # Fill any remaining NaN volumes with forward fill
        merged_data['volume'] = merged_data['volume'].fillna(method='ffill').fillna(0)

        print(f"✓ Collected {len(merged_data)} records for {coin_id}")
        return merged_data

    def collect_all_coins(self, days=None):
        """
        Collect data for all configured coins

        Args:
            days (int): Number of days (defaults to env variable)

        Returns:
            pd.DataFrame: Combined data for all coins
        """
        if days is None:
            days = int(os.getenv('HISTORICAL_DAYS', 180))

        print(f"Starting data collection for {len(self.coins)} coins...")
        print(f"Historical period: {days} days")
        print("-" * 50)

        all_data = []
        successful_coins = 0

        for i, coin in enumerate(self.coins, 1):
            print(f"[{i}/{len(self.coins)}] Processing {coin}...")

            data = self.get_coin_data(coin, days)

            if data is not None and not data.empty:
                all_data.append(data)
                successful_coins += 1
                print(f"✓ Success: {len(data)} records")
            else:
                print(f"✗ Failed to collect data for {coin}")

            # Rate limiting between coins
            if i < len(self.coins):  # Don't wait after the last coin
                print(f"Waiting {self.rate_limit} seconds (rate limiting)...")
                time.sleep(self.rate_limit)

            print()  # Empty line for readability

        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['coin_id', 'timestamp']).reset_index(drop=True)

            print("=" * 50)
            print(f"DATA COLLECTION SUMMARY")
            print(f"Successful coins: {successful_coins}/{len(self.coins)}")
            print(f"Total records: {len(combined_df):,}")
            print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            print(f"Coins collected: {combined_df['coin_id'].unique().tolist()}")
            print("=" * 50)

            return combined_df
        else:
            print("❌ No data collected for any coins!")
            return pd.DataFrame()

    def save_data(self, df, filename=None):
        """
        Save collected data to CSV file

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Output filename (optional)
        """
        if df.empty:
            print("No data to save!")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_data_{timestamp}.csv"

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)

        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

        return filepath
