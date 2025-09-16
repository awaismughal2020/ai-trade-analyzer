"""
Data Collection Module with Dual Provider Support
Handles fetching OHLC and volume data from CoinGecko or Cooking.gg APIs
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DataCollector:
    def __init__(self):
        # Provider selection
        self.use_cooking = os.getenv('USE_COOKING', 'false').lower() == 'true'

        if self.use_cooking:
            # Cooking.gg configuration
            self.cooking_token = os.getenv('COOKING_API_TOKEN')
            self.cooking_base_url = os.getenv('COOKING_BASE_URL', 'https://api.dev.cooking.gg')
            self.rate_limit = float(os.getenv('API_RATE_LIMIT_SECONDS', 1))

            if not self.cooking_token:
                raise ValueError("COOKING_API_TOKEN not found in environment variables")

            print(f"Initialized Cooking.gg data collector")
            print(f"Base URL: {self.cooking_base_url}")
        else:
            # CoinGecko configuration
            self.base_url = os.getenv('COINGECKO_BASE_URL')
            self.api_key = os.getenv('COINGECKO_API_KEY')
            self.rate_limit = int(os.getenv('API_RATE_LIMIT_SECONDS'))

            if not self.base_url:
                raise ValueError("COINGECKO_BASE_URL not found in environment variables")

            print(f"Initialized CoinGecko data collector")
            print(f"Base URL: {self.base_url}")

        # Common configuration
        self.coins = os.getenv('TARGET_COINS', '').split(',')
        if not self.coins or self.coins == ['']:
            raise ValueError("TARGET_COINS not found in environment variables")

        # Remove any whitespace from coin names
        self.coins = [coin.strip() for coin in self.coins if coin.strip()]
        print(f"Target coins: {self.coins}")

    def _get_headers(self):
        """Get headers for API requests based on provider"""
        if self.use_cooking:
            return {
                'accept': '*/*',
                'Authorization': f'Bearer {self.cooking_token}'
            }
        else:
            headers = {
                'User-Agent': 'Trading-Assistant/1.0',
                'Accept': 'application/json'
            }

            # Add API key if provided for CoinGecko
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
        Fetch OHLC data from selected provider

        Args:
            coin_id (str): Coin identifier
            days (int): Number of days of historical data

        Returns:
            pd.DataFrame: OHLC data with timestamp, open, high, low, close
        """
        if self.use_cooking:
            return self._get_cooking_ohlc(coin_id, days)
        else:
            return self._get_coingecko_ohlc(coin_id, days)

    def _get_coingecko_ohlc(self, coin_id, days):
        """Fetch OHLC data from CoinGecko"""
        print(f"Fetching OHLC data for {coin_id} ({days} days) from CoinGecko...")

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

    def _get_cooking_ohlc(self, token_id, days):
        """Fetch OHLC data from Cooking.gg"""
        print(f"Fetching OHLC data for {token_id} ({days} days) from Cooking.gg...")

        url = f"{self.cooking_base_url}/token/{token_id}/bars"

        # Cooking.gg API doesn't accept from/to parameters - only timeframe and limit
        params = {
            'timeframe': 1,  # 1-minute bars
            'limit': min(days * 1440, 1000)  # Reduced limit to match working curl command
        }

        data = self._make_request(url, params)

        if not data:
            print(f"Failed to fetch OHLC data for {token_id}")
            return None

        try:
            # Extract bars data from the nested response
            if 'bars' not in data:
                print(f"No bars data in response for {token_id}")
                return None

            # Convert bars array to DataFrame
            df = pd.DataFrame(data['bars'])

            if df.empty:
                print(f"No data returned for {token_id}")
                return None

            # Convert timestamp from Unix timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # The response already has the correct column names: open, high, low, close, volume, timestamp
            # No column mapping needed

            # Add coin_id column
            df['coin_id'] = token_id

            # Ensure required OHLC columns exist
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Missing {col} column for {token_id}")
                    return None

            # Convert price columns to float
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with NaN values
            df = df.dropna(subset=required_columns)

            if df.empty:
                print(f"No valid OHLC data for {token_id}")
                return None

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error processing OHLC data for {token_id}: {e}")
            return None

    def get_volume_data(self, coin_id, days=180):
        """
        Fetch volume data from selected provider

        Args:
            coin_id (str): Coin identifier
            days (int): Number of days of historical data

        Returns:
            pd.DataFrame: Volume data with timestamp and volume
        """
        if self.use_cooking:
            return self._get_cooking_volume(coin_id, days)
        else:
            return self._get_coingecko_volume(coin_id, days)

    def _get_coingecko_volume(self, coin_id, days):
        """Fetch volume data from CoinGecko market chart endpoint"""
        print(f"Fetching volume data for {coin_id} from CoinGecko...")

        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days
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

    def _get_cooking_volume(self, token_id, days):
        """Extract volume data from Cooking.gg OHLC response"""
        # For Cooking.gg, volume is typically included in the bars response
        # We'll extract it from the OHLC data we already fetch
        ohlc_data = self._get_cooking_ohlc(token_id, days)

        if ohlc_data is None or ohlc_data.empty:
            return pd.DataFrame(columns=['timestamp', 'volume'])

        if 'volume' in ohlc_data.columns:
            return ohlc_data[['timestamp', 'volume']].copy()
        else:
            # If no volume data, return empty DataFrame
            return pd.DataFrame(columns=['timestamp', 'volume'])

    def get_coin_data(self, coin_id, days=None):
        """
        Get complete OHLCV data for a single coin

        Args:
            coin_id (str): Coin identifier
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

        # Handle volume data differently based on provider
        if self.use_cooking:
            # For Cooking.gg, volume is already included in OHLC data
            if 'volume' not in ohlc_data.columns:
                ohlc_data['volume'] = 0  # Add zero volume if not present
            merged_data = ohlc_data
        else:
            # For CoinGecko, get separate volume data and merge
            volume_data = self.get_volume_data(coin_id, days)

            if not volume_data.empty:
                # Round timestamps to nearest hour for better matching
                ohlc_data['timestamp_rounded'] = ohlc_data['timestamp'].dt.round('h')
                volume_data['timestamp_rounded'] = volume_data['timestamp'].dt.round('h')

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
        merged_data['volume'] = merged_data['volume'].ffill().fillna(0)

        print(f"Collected {len(merged_data)} records for {coin_id}")
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

        provider_name = "Cooking.gg" if self.use_cooking else "CoinGecko"
        print(f"Starting data collection for {len(self.coins)} coins using {provider_name}...")
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
                print(f"Success: {len(data)} records")
            else:
                print(f"Failed to collect data for {coin}")

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
            print(f"Provider: {provider_name}")
            print(f"Successful coins: {successful_coins}/{len(self.coins)}")
            print(f"Total records: {len(combined_df):,}")
            print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            print(f"Coins collected: {combined_df['coin_id'].unique().tolist()}")
            print("=" * 50)

            return combined_df
        else:
            print("No data collected for any coins!")
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
            provider_suffix = "cooking" if self.use_cooking else "coingecko"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_data_{provider_suffix}_{timestamp}.csv"

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)

        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

        return filepath

    def test_connection(self):
        """
        Test connection to the selected API provider

        Returns:
            bool: True if connection successful, False otherwise
        """
        provider_name = "Cooking.gg" if self.use_cooking else "CoinGecko"
        print(f"Testing connection to {provider_name}...")

        try:
            if self.use_cooking:
                # Test with a simple request to a known endpoint
                test_token = self.coins[0] if self.coins else "test"
                url = f"{self.cooking_base_url}/token/{test_token}/bars"
                params = {
                    'timeframe': 1,
                    'from': int(time.time()) - 3600,  # Last hour
                    'to': int(time.time()),
                    'limit': 10
                }
            else:
                # Test CoinGecko with ping endpoint
                url = f"{self.base_url}/ping"
                params = None

            response = self._make_request(url, params)

            if response is not None:
                print(f"Connection to {provider_name} successful!")
                return True
            else:
                print(f"Failed to connect to {provider_name}")
                return False

        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
