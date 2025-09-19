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
            self.rate_limit = float(os.getenv('API_RATE_LIMIT_SECONDS', '1'))

            if not self.cooking_token:
                raise ValueError("COOKING_API_TOKEN not found in environment variables")

            print(f"Initialized Cooking.gg data collector")
            print(f"Base URL: {self.cooking_base_url}")
        else:
            # CoinGecko configuration
            self.base_url = os.getenv('COINGECKO_BASE_URL')
            self.api_key = os.getenv('COINGECKO_API_KEY')
            self.rate_limit = int(os.getenv('API_RATE_LIMIT_SECONDS', '1'))

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
        
        # CSV-based coin loading configuration
        self.csv_file_path = os.getenv('CSV_COINS_FILE', 'coingecko_category_coins.csv')
        self.use_csv_coins = os.getenv('USE_CSV_COINS', 'false').lower() == 'true'
        
        # Incremental saving configuration
        self.incremental_save_file = os.getenv('INCREMENTAL_SAVE_FILE', 'data/coin_data.csv')
        self.save_incrementally = os.getenv('SAVE_INCREMENTALLY', 'true').lower() == 'true'

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
                # Use Pro API header format
                headers['x-cg-pro-api-key'] = self.api_key

            return headers

    def _initialize_incremental_csv(self):
        """
        Initialize the incremental CSV file with headers if it doesn't exist
        
        Returns:
            bool: True if file was initialized or already exists, False on error
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.incremental_save_file), exist_ok=True)
            
            # Check if file exists and has content
            if os.path.exists(self.incremental_save_file):
                # Check if file has headers
                with open(self.incremental_save_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line and 'timestamp' in first_line:
                        print(f"✅ Incremental CSV file already exists: {self.incremental_save_file}")
                        return True
            
            # Create file with headers
            headers = ['timestamp', 'open', 'high', 'low', 'close', 'coin_id', 'volume']
            df_header = pd.DataFrame(columns=headers)
            df_header.to_csv(self.incremental_save_file, index=False)
            print(f"✅ Initialized incremental CSV file: {self.incremental_save_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing incremental CSV: {e}")
            return False

    def _append_to_incremental_csv(self, coin_data):
        """
        Append coin data to the incremental CSV file
        
        Args:
            coin_data (pd.DataFrame): Data for a single coin
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if coin_data is None or coin_data.empty:
                return False
                
            # Append to CSV file
            coin_data.to_csv(self.incremental_save_file, mode='a', header=False, index=False)
            print(f"💾 Saved {len(coin_data)} records for {coin_data['coin_id'].iloc[0]} to incremental CSV")
            return True
            
        except Exception as e:
            print(f"❌ Error appending to incremental CSV: {e}")
            return False

    def get_existing_coins_from_incremental_csv(self):
        """
        Get list of coins that already have data in the incremental CSV file
        
        Returns:
            set: Set of coin IDs that already have data
        """
        try:
            if not os.path.exists(self.incremental_save_file):
                return set()
                
            # Read the CSV file and get unique coin IDs
            df = pd.read_csv(self.incremental_save_file)
            if df.empty or 'coin_id' not in df.columns:
                return set()
                
            existing_coins = set(df['coin_id'].unique())
            print(f"📊 Found {len(existing_coins)} coins already in incremental CSV")
            return existing_coins
            
        except Exception as e:
            print(f"❌ Error reading existing incremental CSV: {e}")
            return set()

    def filter_coins_to_process(self, coins_to_process):
        """
        Filter out coins that already have data in incremental CSV
        
        Args:
            coins_to_process (list): List of coins to process
            
        Returns:
            list: Filtered list of coins that need processing
        """
        if not self.save_incrementally:
            return coins_to_process
            
        existing_coins = self.get_existing_coins_from_incremental_csv()
        if not existing_coins:
            return coins_to_process
            
        # Filter out existing coins
        filtered_coins = [coin for coin in coins_to_process if coin not in existing_coins]
        skipped_count = len(coins_to_process) - len(filtered_coins)
        
        if skipped_count > 0:
            print(f"⏭️  Skipping {skipped_count} coins that already have data")
            print(f"🔄 Processing {len(filtered_coins)} remaining coins")
        
        return filtered_coins

    def load_coins_from_csv(self, csv_file_path=None):
        """
        Load coin information from CSV file
        
        Args:
            csv_file_path (str): Path to CSV file (defaults to self.csv_file_path)
            
        Returns:
            pd.DataFrame: DataFrame with coin information
        """
        if csv_file_path is None:
            csv_file_path = self.csv_file_path
            
        try:
            if not os.path.exists(csv_file_path):
                print(f"CSV file not found: {csv_file_path}")
                return pd.DataFrame()
                
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                print(f"CSV file is empty: {csv_file_path}")
                return pd.DataFrame()
                
            # Validate required columns
            required_columns = ['name', 'symbol']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing required columns in CSV: {missing_columns}")
                return pd.DataFrame()
                
            print(f"Loaded {len(df)} coins from CSV: {csv_file_path}")
            print(f"Columns available: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV file {csv_file_path}: {e}")
            return pd.DataFrame()

    def get_coins_to_process(self):
        """
        Get list of coins to process based on configuration
        
        Returns:
            list: List of coin identifiers to process
        """
        if self.use_csv_coins:
            csv_data = self.load_coins_from_csv()
            if csv_data.empty:
                print("Falling back to TARGET_COINS from environment")
                return self.coins
            
            # For CoinGecko, we need coin IDs, not symbols
            # We'll use the symbol as coin_id for now, but this might need adjustment
            coins_to_process = csv_data['symbol'].str.lower().tolist()
            print(f"Using {len(coins_to_process)} coins from CSV file")
            return coins_to_process
        else:
            return self.coins

    def get_coins_to_process_with_mapping(self):
        """
        Get list of coins to process with proper CoinGecko ID mapping
        
        Returns:
            list: List of coin identifiers to process
        """
        if self.use_csv_coins:
            csv_data = self.load_coins_from_csv()
            if csv_data.empty:
                print("Falling back to TARGET_COINS from environment")
                return self.coins
            
            # Try to get CoinGecko coin list to map symbols to IDs
            try:
                coin_list_url = f"{self.base_url}/coins/list"
                coin_list = self._make_request(coin_list_url)
                
                if coin_list:
                    # Create mapping from symbol to id
                    symbol_to_id = {coin['symbol'].lower(): coin['id'] for coin in coin_list}
                    
                    # Map CSV symbols to CoinGecko IDs
                    coins_to_process = []
                    unmapped_symbols = []
                    
                    for symbol in csv_data['symbol'].str.lower():
                        if symbol in symbol_to_id:
                            coins_to_process.append(symbol_to_id[symbol])
                        else:
                            unmapped_symbols.append(symbol)
                    
                    print(f"Mapped {len(coins_to_process)} symbols to CoinGecko IDs")
                    if unmapped_symbols:
                        print(f"Could not map {len(unmapped_symbols)} symbols: {unmapped_symbols[:10]}{'...' if len(unmapped_symbols) > 10 else ''}")
                    
                    return coins_to_process
                else:
                    print("Failed to get CoinGecko coin list, using symbols directly")
                    return csv_data['symbol'].str.lower().tolist()
                    
            except Exception as e:
                print(f"Error mapping symbols to IDs: {e}")
                print("Using symbols directly")
                return csv_data['symbol'].str.lower().tolist()
        else:
            return self.coins

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

    def collect_all_coins(self, days=None, use_csv=None):
        """
        Collect data for all configured coins

        Args:
            days (int): Number of days (defaults to env variable)
            use_csv (bool): Override CSV usage setting (optional)

        Returns:
            pd.DataFrame: Combined data for all coins
        """
        if days is None:
            days = int(os.getenv('HISTORICAL_DAYS', 180))
            
        # Override CSV usage if specified
        if use_csv is not None:
            original_csv_setting = self.use_csv_coins
            self.use_csv_coins = use_csv

        # Get coins to process (use mapping for CSV coins)
        if self.use_csv_coins:
            coins_to_process = self.get_coins_to_process_with_mapping()
        else:
            coins_to_process = self.get_coins_to_process()
        
        # Filter out coins that already have data (if incremental saving is enabled)
        coins_to_process = self.filter_coins_to_process(coins_to_process)
        
        if not coins_to_process:
            print("No coins to process!")
            return pd.DataFrame()

        provider_name = "Cooking.gg" if self.use_cooking else "CoinGecko"
        data_source = "CSV file" if self.use_csv_coins else "environment variables"
        
        print(f"Starting data collection for {len(coins_to_process)} coins using {provider_name}...")
        print(f"Data source: {data_source}")
        print(f"Historical period: {days} days")
        
        # Initialize incremental saving if enabled
        if self.save_incrementally:
            if not self._initialize_incremental_csv():
                print("⚠️  Warning: Could not initialize incremental CSV, continuing without incremental saving")
                self.save_incrementally = False
            else:
                print(f"💾 Incremental saving enabled: {self.incremental_save_file}")
        
        print("-" * 50)

        all_data = []
        successful_coins = 0
        failed_coins = []

        for i, coin in enumerate(coins_to_process, 1):
            print(f"[{i}/{len(coins_to_process)}] Processing {coin}...")

            data = self.get_coin_data(coin, days)

            if data is not None and not data.empty:
                all_data.append(data)
                successful_coins += 1
                print(f"Success: {len(data)} records")
                
                # Save incrementally if enabled
                if self.save_incrementally:
                    self._append_to_incremental_csv(data)
            else:
                failed_coins.append(coin)
                print(f"Failed to collect data for {coin}")

            # Rate limiting between coins
            if i < len(coins_to_process):  # Don't wait after the last coin
                print(f"Waiting {self.rate_limit} seconds (rate limiting)...")
                time.sleep(self.rate_limit)

            print()  # Empty line for readability

        # Restore original CSV setting if it was overridden
        if use_csv is not None:
            self.use_csv_coins = original_csv_setting

        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['coin_id', 'timestamp']).reset_index(drop=True)

            print("=" * 50)
            print(f"DATA COLLECTION SUMMARY")
            print(f"Provider: {provider_name}")
            print(f"Data source: {data_source}")
            print(f"Successful coins: {successful_coins}/{len(coins_to_process)}")
            print(f"Total records: {len(combined_df):,}")
            print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            print(f"Coins collected: {combined_df['coin_id'].unique().tolist()}")
            
            if self.save_incrementally:
                print(f"💾 Incremental data saved to: {self.incremental_save_file}")
            
            if failed_coins:
                print(f"Failed coins: {failed_coins[:10]}{'...' if len(failed_coins) > 10 else ''}")
            print("=" * 50)

            return combined_df
        else:
            print("No data collected for any coins!")
            return pd.DataFrame()

    def collect_csv_coins(self, days=None, csv_file_path=None):
        """
        Convenience method to collect data from all coins in CSV file
        
        Args:
            days (int): Number of days (defaults to env variable)
            csv_file_path (str): Path to CSV file (optional)
            
        Returns:
            pd.DataFrame: Combined data for all coins
        """
        if csv_file_path:
            self.csv_file_path = csv_file_path
            
        return self.collect_all_coins(days=days, use_csv=True)

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
            data_source_suffix = "csv" if self.use_csv_coins else "env"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_data_{timestamp}.csv"

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
                # Test CoinGecko Pro API with ping endpoint
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


def main():
    """
    Example usage of the DataCollector with CSV-based coin loading
    """
    try:
        # Initialize data collector
        collector = DataCollector()
        
        # Test connection first
        if not collector.test_connection():
            print("Connection test failed. Please check your configuration.")
            return
        
        print("\n" + "="*60)
        print("CSV-BASED DATA COLLECTION DEMO")
        print("="*60)
        
        # Load and display CSV coins info
        csv_data = collector.load_coins_from_csv()
        if not csv_data.empty:
            print(f"\nCSV Summary:")
            print(f"Total coins in CSV: {len(csv_data)}")
            print(f"Categories: {csv_data['category'].unique().tolist()}")
            print(f"Sample coins: {csv_data['name'].head(5).tolist()}")
        
        # Ask user for confirmation before proceeding with large collection
        print(f"\n⚠️  WARNING: This will collect OHLC data for ALL {len(csv_data)} coins!")
        print("This may take a very long time and use significant API quota.")
        
        response = input("\nDo you want to proceed? (y/N): ").strip().lower()
        if response != 'y':
            print("Collection cancelled.")
            return
        
        # Collect data from CSV coins
        print(f"\nStarting collection from CSV file...")
        data = collector.collect_csv_coins(days=30)  # Start with 30 days for testing
        
        if not data.empty:
            # Save the data
            filepath = collector.save_data(data)
            print(f"\n✅ Data collection completed successfully!")
            print(f"📁 Data saved to: {filepath}")
            
            # Display summary statistics
            print(f"\n📊 Collection Summary:")
            print(f"   Total records: {len(data):,}")
            print(f"   Unique coins: {data['coin_id'].nunique()}")
            print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            print(f"   Average records per coin: {len(data) / data['coin_id'].nunique():.1f}")
        else:
            print("❌ No data was collected.")
            
    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
