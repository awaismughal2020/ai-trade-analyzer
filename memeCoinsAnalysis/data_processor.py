"""
Data Processing Module
Handles data cleaning, technical indicator calculation, and sequence creation
"""

import pandas as pd
import numpy as np
import ta
import os
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.sequence_length = int(os.getenv('SEQUENCE_LENGTH', 20))
        self.prediction_horizon = int(os.getenv('PREDICTION_HORIZON', 1))
        self.trending_threshold = float(os.getenv('TRENDING_THRESHOLD', 0.001))

        # Feature columns that will be used for model training
        self.feature_columns = [
            'rsi_14', 'ema_20', 'volume_ratio', 'price_change_pct',
            'high_low_ratio', 'close_ema_ratio', 'volatility_20',
            'price_above_ema', 'volume_above_avg'
        ]

        print(f"Initialized DataProcessor:")
        print(f"- Sequence length: {self.sequence_length}")
        print(f"- Prediction horizon: {self.prediction_horizon}")
        print(f"- Feature columns: {len(self.feature_columns)}")

    def clean_data(self, df):
        """
        Clean and validate market data

        Args:
            df (pd.DataFrame): Raw market data

        Returns:
            pd.DataFrame: Cleaned data
        """
        print("Starting data cleaning...")
        initial_count = len(df)

        if df.empty:
            print("Warning: Empty dataframe provided!")
            return df

        # Make a copy to avoid modifying original
        df_clean = df.copy()

        # Remove rows with missing critical values
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'coin_id']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove rows with NaN in critical columns
        df_clean = df_clean.dropna(subset=required_columns)

        # Ensure proper data types
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        df_clean[price_columns] = df_clean[price_columns].astype(float)

        # Remove rows with zero or negative prices
        for col in ['open', 'high', 'low', 'close']:
            df_clean = df_clean[df_clean[col] > 0]

        # Remove rows with negative volume
        df_clean = df_clean[df_clean['volume'] >= 0]

        # Sort by coin and timestamp
        df_clean = df_clean.sort_values(['coin_id', 'timestamp'])

        # Detect and remove obvious price errors
        print("Detecting price anomalies...")
        df_clean = self._remove_price_anomalies(df_clean)

        # Reset index
        df_clean = df_clean.reset_index(drop=True)

        cleaned_count = len(df_clean)
        removed_count = initial_count - cleaned_count

        print(f"Data cleaning complete:")
        print(f"- Initial records: {initial_count:,}")
        print(f"- Cleaned records: {cleaned_count:,}")
        print(f"- Removed records: {removed_count:,} ({removed_count / initial_count * 100:.1f}%)")

        return df_clean

    def _remove_price_anomalies(self, df):
        """Remove obvious price errors and anomalies"""
        clean_dfs = []

        for coin in df['coin_id'].unique():
            coin_df = df[df['coin_id'] == coin].copy()

            # Calculate price changes
            coin_df['price_change'] = coin_df['close'].pct_change()

            # Remove extreme single-period changes (>50%)
            mask_extreme = abs(coin_df['price_change']) < 0.5
            coin_df = coin_df[mask_extreme]

            # Remove impossible OHLC relationships
            mask_ohlc = (
                    (coin_df['high'] >= coin_df['low']) &
                    (coin_df['high'] >= coin_df['open']) &
                    (coin_df['high'] >= coin_df['close']) &
                    (coin_df['low'] <= coin_df['open']) &
                    (coin_df['low'] <= coin_df['close'])
            )
            coin_df = coin_df[mask_ohlc]

            # Remove outliers in high-low ratio (>100% range)
            coin_df['high_low_range'] = (coin_df['high'] - coin_df['low']) / coin_df['close']
            mask_range = coin_df['high_low_range'] < 1.0
            coin_df = coin_df[mask_range]

            # Drop temporary columns
            coin_df = coin_df.drop(['price_change', 'high_low_range'], axis=1)

            clean_dfs.append(coin_df)

        return pd.concat(clean_dfs, ignore_index=True)

    def calculate_technical_indicators(self, df):
        print("Calculating technical indicators (volume-safe version)...")

        processed_dfs = []

        for coin in df['coin_id'].unique():
            coin_df = df[df['coin_id'] == coin].copy()

            if len(coin_df) < 30:  # Reduced requirement
                print(f"Skipping {coin} - insufficient data ({len(coin_df)} < 30 records)")
                continue

            print(f"Processing indicators for {coin} ({len(coin_df)} records)...")

            try:
                coin_df = coin_df.sort_values('timestamp')

                # Price indicators (these work fine)
                coin_df['price_change_pct'] = coin_df['close'].pct_change()
                coin_df['high_low_ratio'] = (coin_df['high'] - coin_df['low']) / coin_df['close']

                # RSI-14
                coin_df['rsi_14'] = ta.momentum.RSIIndicator(coin_df['close'], window=14).rsi()

                # EMA-20
                coin_df['ema_20'] = ta.trend.EMAIndicator(coin_df['close'], window=20).ema_indicator()
                coin_df['close_ema_ratio'] = coin_df['close'] / coin_df['ema_20']

                # Volatility
                coin_df['volatility_20'] = coin_df['close'].rolling(window=20).std()

                # Fixed volume indicators (handle zero volume)
                if coin_df['volume'].sum() == 0:
                    # If all volumes are zero, create dummy indicators
                    coin_df['volume_ratio'] = 1.0  # Neutral value
                    coin_df['volume_above_avg'] = 0  # Conservative
                else:
                    # Normal volume calculation
                    coin_df['volume_sma_20'] = coin_df['volume'].rolling(window=20).mean()
                    coin_df['volume_ratio'] = coin_df['volume'] / coin_df['volume_sma_20'].replace(0,
                                                                                                   1)  # Avoid division by zero
                    coin_df['volume_above_avg'] = (coin_df['volume_ratio'] > 1.0).astype(int)

                # Boolean indicators
                coin_df['price_above_ema'] = (coin_df['close'] > coin_df['ema_20']).astype(int)
                coin_df['rsi_oversold'] = (coin_df['rsi_14'] < 30).astype(int)
                coin_df['rsi_overbought'] = (coin_df['rsi_14'] > 70).astype(int)

                processed_dfs.append(coin_df)

            except Exception as e:
                print(f"Error calculating indicators for {coin}: {e}")
                continue

        if not processed_dfs:
            return pd.DataFrame()

        result = pd.concat(processed_dfs, ignore_index=True)
        result = result.dropna()

        print(f"Technical indicators calculated:")
        print(f"- Final dataset: {len(result):,} records across {result['coin_id'].nunique()} coins")

        return result

    def create_labels(self, df):
        """Create trading labels based on future returns"""
        print("Creating trading labels...")

        labeled_dfs = []
        trending_coins = 0

        for coin in df['coin_id'].unique():
            coin_df = df[df['coin_id'] == coin].copy()
            coin_df = coin_df.sort_values('timestamp')

            # Check if this coin is in a trending market
            if not self.is_trending_market(coin_df):
                print(f"Skipping {coin} - not in trending market")
                continue

            trending_coins += 1
            print(f"Processing {coin} - trending market detected")

            # Calculate future returns
            coin_df['future_price'] = coin_df['close'].shift(-self.prediction_horizon)
            coin_df['future_return'] = (coin_df['future_price'] - coin_df['close']) / coin_df['close']

            # Create discrete labels
            coin_df['label'] = 1  # Default to HOLD
            coin_df.loc[coin_df['future_return'] > 0.02, 'label'] = 2  # BUY
            coin_df.loc[coin_df['future_return'] < -0.02, 'label'] = 0  # SELL

            # Remove rows where we can't calculate future returns
            coin_df = coin_df.dropna(subset=['future_return'])
            labeled_dfs.append(coin_df)

        if not labeled_dfs:
            print("Warning: No coins in trending markets found!")
            return pd.DataFrame()

        print(f"Found {trending_coins} coins in trending markets out of {df['coin_id'].nunique()}")

        result = pd.concat(labeled_dfs, ignore_index=True)

        # Print label distribution
        label_counts = result['label'].value_counts().sort_index()
        total = len(result)

        print(f"Label distribution:")
        print(f"- SELL (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0) / total * 100:.1f}%)")
        print(f"- HOLD (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0) / total * 100:.1f}%)")
        print(f"- BUY (2):  {label_counts.get(2, 0):,} ({label_counts.get(2, 0) / total * 100:.1f}%)")

        return result

    def create_sequences(self, df):
        """
        Create sequences for LSTM training

        Args:
            df (pd.DataFrame): Data with indicators and labels

        Returns:
            tuple: (X, y) numpy arrays for training
        """
        print(f"Creating sequences (length={self.sequence_length})...")

        # Ensure we have all required feature columns
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        sequences = []
        labels = []
        coin_sequence_counts = {}

        for coin in df['coin_id'].unique():
            coin_df = df[df['coin_id'] == coin].sort_values('timestamp')

            # Need enough data for sequences
            min_required = self.sequence_length + self.prediction_horizon
            if len(coin_df) < min_required:
                print(f"Skipping {coin} - insufficient data for sequences ({len(coin_df)} < {min_required})")
                continue

            coin_sequences = 0

            # Create overlapping sequences
            for i in range(len(coin_df) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence (features only)
                sequence = coin_df.iloc[i:i + self.sequence_length][self.feature_columns].values

                # Label from the end of the sequence
                label = coin_df.iloc[i + self.sequence_length - 1]['label']

                # Validate sequence (no NaN values)
                if not np.isnan(sequence).any() and not np.isnan(label):
                    sequences.append(sequence)
                    labels.append(label)
                    coin_sequences += 1

            coin_sequence_counts[coin] = coin_sequences
            print(f"- {coin}: {coin_sequences:,} sequences")

        X = np.array(sequences)
        y = np.array(labels)

        print(f"\nSequence creation complete:")
        print(f"- Total sequences: {len(X):,}")
        print(f"- Sequence shape: {X.shape}")
        print(f"- Features per timestep: {X.shape[2]}")

        # Final label distribution
        unique, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique, counts))
        total = len(y)

        print(f"\nFinal label distribution:")
        for label_idx, label_name in enumerate(['SELL', 'HOLD', 'BUY']):
            count = label_dist.get(label_idx, 0)
            print(f"- {label_name}: {count:,} ({count / total * 100:.1f}%)")

        return X, y

    def save_processed_data(self, df, filename=None):
        """
        Save processed data to CSV

        Args:
            df (pd.DataFrame): Processed data
            filename (str): Output filename (optional)
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_data_{timestamp}.csv"

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)

        df.to_csv(filepath, index=False)
        print(f"Processed data saved to: {filepath}")

        return filepath

    def process_crypto_data_file(self, input_file=None, output_file=None):
        """
        Complete processing pipeline for crypto data file
        
        Args:
            input_file (str): Path to input CSV file (if None, will auto-detect)
            output_file (str): Path to output CSV file (if None, will auto-generate)
            
        Returns:
            pd.DataFrame: Processed data
        """
        # Auto-detect input file if not provided
        if input_file is None:
            input_file = self._find_latest_crypto_data_file()
            if input_file is None:
                raise FileNotFoundError("No crypto_data_*.csv files found in data directory")
        
        # Auto-generate output file if not provided
        if output_file is None:
            input_basename = os.path.basename(input_file)
            if input_basename.startswith('crypto_data_') and input_basename.endswith('.csv'):
                # Extract timestamp from input filename
                timestamp = input_basename.replace('crypto_data_', '').replace('.csv', '')
                output_file = f"data/processed_crypto_data_{timestamp}.csv"
            else:
                # Fallback to timestamp-based naming
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"data/processed_crypto_data_{timestamp}.csv"
        
        print(f"Starting processing pipeline for {input_file}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read the CSV data
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df):,} records from {df['coin_id'].nunique()} coins")
        
        # Clean the data
        df_clean = self.clean_data(df)
        
        if df_clean.empty:
            print("Warning: No data remaining after cleaning!")
            return pd.DataFrame()
        
        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df_clean)
        
        if df_with_indicators.empty:
            print("Warning: No data remaining after indicator calculation!")
            return pd.DataFrame()
        
        # Create labels
        df_with_labels = self.create_labels(df_with_indicators)
        
        if df_with_labels.empty:
            print("Warning: No data remaining after label creation!")
            return pd.DataFrame()
        
        # Save processed data
        self.save_processed_data(df_with_labels, os.path.basename(output_file))
        
        print(f"\nProcessing complete!")
        print(f"- Final dataset: {len(df_with_labels):,} records")
        print(f"- Coins processed: {df_with_labels['coin_id'].nunique()}")
        print(f"- Output saved to: {output_file}")
        
        return df_with_labels

    def _find_latest_crypto_data_file(self, data_dir="data"):
        """
        Find the latest crypto_data_*.csv file in the data directory
        
        Args:
            data_dir (str): Directory to search for CSV files
            
        Returns:
            str: Path to the latest crypto data file, or None if not found
        """
        import glob
        import os
        
        if not os.path.exists(data_dir):
            print(f"Data directory '{data_dir}' does not exist")
            return None
        
        # Find all crypto_data_*.csv files
        pattern = os.path.join(data_dir, "crypto_data_*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            print(f"No crypto_data_*.csv files found in {data_dir}")
            return None
        
        # Sort by modification time (newest first)
        csv_files.sort(key=os.path.getmtime, reverse=True)
        
        latest_file = csv_files[0]
        print(f"Found {len(csv_files)} crypto data files, using latest: {latest_file}")
        
        return latest_file

    def process_crypto_data_by_timestamp(self, timestamp):
        """
        Process crypto data file by specific timestamp
        
        Args:
            timestamp (str): Timestamp in format YYYYMMDD_HHMMSS
            
        Returns:
            pd.DataFrame: Processed data
        """
        input_file = f"data/crypto_data_{timestamp}.csv"
        output_file = f"data/processed_crypto_data_{timestamp}.csv"
        
        return self.process_crypto_data_file(input_file, output_file)

    def list_available_crypto_data_files(self, data_dir="data"):
        """
        List all available crypto_data_*.csv files in the data directory
        
        Args:
            data_dir (str): Directory to search for CSV files
            
        Returns:
            list: List of available crypto data files
        """
        import glob
        import os
        
        if not os.path.exists(data_dir):
            print(f"Data directory '{data_dir}' does not exist")
            return []
        
        # Find all crypto_data_*.csv files
        pattern = os.path.join(data_dir, "crypto_data_*.csv")
        csv_files = glob.glob(pattern)
        
        # Sort by modification time (newest first)
        csv_files.sort(key=os.path.getmtime, reverse=True)
        
        print(f"Found {len(csv_files)} crypto data files:")
        for i, file in enumerate(csv_files, 1):
            basename = os.path.basename(file)
            mod_time = os.path.getmtime(file)
            from datetime import datetime
            mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i}. {basename} (modified: {mod_time_str})")
        
        return csv_files

    def get_feature_info(self):
        """
        Get information about features used in the model

        Returns:
            dict: Feature information
        """
        feature_info = {
            'rsi_14': 'Relative Strength Index (14 periods) - momentum oscillator (0-100)',
            'ema_20': 'Exponential Moving Average (20 periods) - trend indicator',
            'volume_ratio': 'Current volume / 20-period average volume',
            'price_change_pct': 'Percentage change from previous period',
            'high_low_ratio': '(High - Low) / Close - measure of volatility',
            'close_ema_ratio': 'Close price / EMA-20 - price vs trend strength',
            'volatility_20': '20-period rolling standard deviation of close price',
            'price_above_ema': 'Binary: 1 if price > EMA-20, 0 otherwise',
            'volume_above_avg': 'Binary: 1 if volume > average, 0 otherwise'
        }

        return feature_info

    def is_trending_market(self, df, threshold=0.001):
        """
        Determine if market is in a trending state

        Args:
            df (pd.DataFrame): Data with EMA-20 calculated
            threshold (float): Minimum EMA change rate to consider trending

        Returns:
            bool: True if market is trending
        """
        if len(df) < 20 or 'ema_20' not in df.columns:
            return False

        # Calculate EMA change rate
        ema_change_rate = df['ema_20'].diff().abs().mean() / df['ema_20'].mean()

        return ema_change_rate > threshold


def main():
    """
    Main function to process crypto data file
    Automatically detects and processes the latest crypto_data_*.csv file
    """
    import sys
    
    print("=" * 60)
    print("CRYPTO DATA PROCESSOR")
    print("=" * 60)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            # List available files
            processor.list_available_crypto_data_files()
            return
        elif sys.argv[1] == "--timestamp" and len(sys.argv) > 2:
            # Process specific timestamp
            timestamp = sys.argv[2]
            try:
                processed_data = processor.process_crypto_data_by_timestamp(timestamp)
                print(f"\nProcessed data for timestamp: {timestamp}")
            except Exception as e:
                print(f"Error processing timestamp {timestamp}: {e}")
                return
        else:
            print("Usage:")
            print("  python data_processor.py                    # Process latest file")
            print("  python data_processor.py --list             # List available files")
            print("  python data_processor.py --timestamp YYYYMMDD_HHMMSS  # Process specific file")
            return
    else:
        # Process the crypto data file (auto-detect input and output files)
        try:
            processed_data = processor.process_crypto_data_file()
        except Exception as e:
            print(f"Error during processing: {e}")
            raise
    
    if not processed_data.empty:
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Successfully processed {len(processed_data):,} records")
        print(f"Coins processed: {processed_data['coin_id'].nunique()}")
        print(f"Features calculated: {len(processor.feature_columns)}")
        
        # Show feature info
        print("\nTechnical Indicators Calculated:")
        feature_info = processor.get_feature_info()
        for feature, description in feature_info.items():
            print(f"- {feature}: {description}")
            
    else:
        print("No data was processed successfully!")


if __name__ == "__main__":
    main()


