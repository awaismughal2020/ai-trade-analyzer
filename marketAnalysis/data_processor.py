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


