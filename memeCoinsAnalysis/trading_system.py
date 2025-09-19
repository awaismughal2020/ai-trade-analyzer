"""
COMPLETE PRODUCTION MEME COIN TRADING ANALYSIS SYSTEM
======================================================
Professional-grade implementation with all components
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import os
import sys
import json
import time
import logging
import hashlib
import pickle
import warnings
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import ta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

import joblib
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables
load_dotenv()


# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================

@dataclass
class SystemConfig:
    """Centralized system configuration"""
    # API Configuration
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    coingecko_api_key: Optional[str] = None
    api_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_calls: int = 30
    rate_limit_period: int = 60

    # Data Configuration
    target_coins: List[str] = field(default_factory=lambda: [
        'dogecoin'
    ])
    historical_days: int = 60
    min_data_points: int = 20

    # Model Configuration
    sequence_length: int = 10
    n_features: int = 12
    n_classes: int = 3
    test_split: float = 0.2
    validation_split: float = 0.1

    # Training Configuration
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7

    # Analysis Configuration
    min_market_cap: float = 1_000_000
    min_volume_24h: float = 100_000
    trending_threshold: float = 0.002
    confidence_threshold: float = 0.65

    # System Configuration
    cache_dir: Path = Path("./data/cache")
    model_dir: Path = Path("./models")
    log_dir: Path = Path("./logs")
    report_dir: Path = Path("./reports")

    # Feature Configuration
    technical_indicators: List[str] = field(default_factory=lambda: [
        'rsi', 'ema', 'sma', 'macd', 'bb_upper', 'bb_lower',
        'volume_ratio', 'price_change', 'volatility', 'momentum'
    ])

    def __post_init__(self):
        """Create directories and load from environment"""
        for dir_path in [self.cache_dir, self.model_dir, self.log_dir, self.report_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load from environment variables
        if api_key := os.getenv('COINGECKO_API_KEY'):
            self.coingecko_api_key = api_key
        if base_url := os.getenv('COINGECKO_BASE_URL'):
            self.coingecko_base_url = base_url


# ==============================================================================
# LOGGING SYSTEM
# ==============================================================================

class LoggerManager:
    """Centralized logging management"""

    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(name)

        if logger.handlers:
            return logger

        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = Path("logs") / f"{name}_{datetime.now():%Y%m%d}.log"
        log_file.parent.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger


# ==============================================================================
# ERROR HANDLING AND RESILIENCE
# ==============================================================================

class TradingSignal(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class AnalysisResult:
    """Structured analysis result"""
    coin_id: str
    timestamp: datetime
    price: float
    signal: TradingSignal
    confidence: float
    technical_score: float
    risk_score: float
    indicators: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker for API resilience"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False

    def call(self, func, *args, **kwargs):
        """Execute with circuit breaker protection"""
        if self.is_open:
            if (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            raise e


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retry logic with exponential backoff"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        return wrapper

    return decorator


# ==============================================================================
# CACHE MANAGER
# ==============================================================================

class CacheManager:
    """Advanced caching system with TTL and compression"""

    def __init__(self, cache_dir: Path = Path("./data/cache"), ttl_hours: int = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.logger = LoggerManager.setup_logger(__name__)

    def _get_cache_key(self, key: str) -> str:
        """Generate cache filename from key"""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache if valid"""
        cache_file = self.cache_dir / f"{self._get_cache_key(key)}.pkl"

        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < self.ttl_seconds:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Cache read error: {e}")
        return None

    def set(self, key: str, value: Any) -> bool:
        """Store in cache"""
        cache_file = self.cache_dir / f"{self._get_cache_key(key)}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            self.logger.error(f"Cache write error: {e}")
            return False

    def clear(self) -> int:
        """Clear all cache files"""
        count = 0
        for file in self.cache_dir.glob("*.pkl"):
            try:
                file.unlink()
                count += 1
            except:
                pass
        return count


# ==============================================================================
# DATA COLLECTOR
# ==============================================================================

class DataCollector:
    """Advanced data collection with multiple source support"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = LoggerManager.setup_logger(self.__class__.__name__)
        self.cache = CacheManager()
        self.circuit_breaker = CircuitBreaker()
        self.session = self._create_session()
        self._api_call_times = []

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers for paid tier"""
        headers = {'Accept': 'application/json'}
        if self.config.coingecko_api_key:
            # For paid tier, use x-cg-pro-api-key instead of x-cg-demo-api-key
            headers['x-cg-pro-api-key'] = self.config.coingecko_api_key
        return headers

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        self._api_call_times = [
            t for t in self._api_call_times
            if current_time - t < self.config.rate_limit_period
        ]

        if len(self._api_call_times) >= self.config.rate_limit_calls:
            sleep_time = self.config.rate_limit_period - (current_time - self._api_call_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._api_call_times = []

        self._api_call_times.append(current_time)

    @retry_on_failure(max_retries=3)
    def fetch_ohlc_data(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """Fetch OHLC data for a coin"""
        cache_key = f"ohlc_{coin_id}_{days}_{datetime.now():%Y%m%d}"

        if cached := self.cache.get(cache_key):
            self.logger.debug(f"Using cached OHLC data for {coin_id}")
            return pd.DataFrame(cached)

        self._rate_limit()

        url = f"{self.config.coingecko_base_url}/coins/{coin_id}/ohlc"
        params = {"vs_currency": "usd", "days": days}

        try:
            response = self.circuit_breaker.call(
                self.session.get, url,
                headers=self._get_headers(),
                params=params,
                timeout=self.config.api_timeout
            )

            self.logger.debug(f"API Response status: {response.status_code}")
            self.logger.debug(f"API Response headers: {dict(response.headers)}")

            response.raise_for_status()

            data = response.json()

            # Check for API error responses
            if isinstance(data, dict):
                if 'error' in data:
                    raise Exception(f"CoinGecko API Error: {data['error']}")
                if 'status' in data and 'error_code' in data['status']:
                    raise Exception(f"CoinGecko API Error: {data['status']}")

            # Validate data format
            if not isinstance(data, list) or not data:
                raise Exception(f"Invalid data format from API: {type(data)}")

            # Check if each row has expected format
            if not all(isinstance(row, list) and len(row) >= 5 for row in data[:3]):
                raise Exception(f"Unexpected OHLC data format: {data[:2] if data else 'empty'}")

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['coin_id'] = coin_id

            # Fetch volume separately
            volume_df = self._fetch_volume_data(coin_id, days)
            if not volume_df.empty:
                # Align timestamps for merging
                df['timestamp_key'] = df['timestamp'].dt.floor('H')  # Round to nearest hour
                volume_df['timestamp_key'] = volume_df['timestamp'].dt.floor('H')

                df = df.merge(volume_df[['timestamp_key', 'volume']], on='timestamp_key', how='left')
                df = df.drop('timestamp_key', axis=1)
                df['volume'] = df['volume'].fillna(method='ffill').fillna(0)
            else:
                df['volume'] = 0

            if df.empty:
                raise Exception("No data returned after processing")

            self.cache.set(cache_key, df.to_dict('records'))
            self.logger.info(f"Fetched {len(df)} OHLC records for {coin_id}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch OHLC for {coin_id}: {e}")
            # Log more details for debugging
            if 'response' in locals():
                try:
                    self.logger.error(f"Response content: {response.text[:500]}")
                except:
                    pass
            return pd.DataFrame()

    def _fetch_volume_data(self, coin_id: str, days: int) -> pd.DataFrame:
        """Fetch volume data"""
        url = f"{self.config.coingecko_base_url}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}

        try:
            response = self.session.get(
                url, headers=self._get_headers(),
                params=params, timeout=self.config.api_timeout
            )
            response.raise_for_status()

            data = response.json()
            if 'total_volumes' in data:
                df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df

        except Exception as e:
            self.logger.warning(f"Failed to fetch volume for {coin_id}: {e}")

        return pd.DataFrame()

    def get_top_meme_coins(self, limit: int = 20) -> List[Dict]:
        """Get top meme coins by market cap"""
        cache_key = f"meme_coins_{limit}_{datetime.now():%Y%m%d%H}"

        if cached := self.cache.get(cache_key):
            return cached

        self._rate_limit()

        # Use predefined meme coin IDs for free tier
        meme_coin_ids = ','.join(self.config.target_coins[:limit])

        url = f"{self.config.coingecko_base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': meme_coin_ids,
            'order': 'market_cap_desc',
            'sparkline': False,
            'price_change_percentage': '24h,7d'
        }

        try:
            response = self.session.get(
                url, headers=self._get_headers(),
                params=params, timeout=self.config.api_timeout
            )
            response.raise_for_status()

            coins = response.json()

            # Filter by minimum requirements
            filtered = [
                {
                    'id': coin['id'],
                    'symbol': coin['symbol'].upper(),
                    'name': coin['name'],
                    'price': coin['current_price'],
                    'market_cap': coin['market_cap'],
                    'volume_24h': coin['total_volume'],
                    'price_change_24h': coin.get('price_change_percentage_24h', 0),
                    'price_change_7d': coin.get('price_change_percentage_7d_in_currency', 0)
                }
                for coin in coins
                if coin['market_cap'] >= self.config.min_market_cap
                   and coin['total_volume'] >= self.config.min_volume_24h
            ]

            self.cache.set(cache_key, filtered)
            self.logger.info(f"Fetched {len(filtered)} meme coins")
            return filtered

        except Exception as e:
            self.logger.error(f"Failed to fetch meme coins: {e}")
            return []


# ==============================================================================
# DATA PROCESSOR
# ==============================================================================

class DataProcessor:
    """Advanced data processing with technical indicators"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = LoggerManager.setup_logger(self.__class__.__name__)
        self.scaler = RobustScaler()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        if df.empty:
            return df

        initial_len = len(df)

        # Remove invalid prices
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]

        # Fix OHLC relationships
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
            ]

        # Remove extreme price changes (>50% in single period)
        df['price_change'] = df['close'].pct_change()
        df = df[df['price_change'].abs() < 0.5]
        df = df.drop('price_change', axis=1)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.logger.info(f"Cleaned data: {initial_len} → {len(df)} records")
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if len(df) < 20:  # Reduced from 30
            self.logger.warning(f"Insufficient data for indicators: {len(df)} records")
            return pd.DataFrame()

        df = df.copy()

        try:
            # Price-based indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Moving averages (reduce window sizes for small datasets)
            window_short = min(14, len(df) // 2)  # Adaptive window
            window_long = min(20, len(df) - 5)  # Adaptive window

            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=window_short)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=window_short)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=window_long)

            # RSI with smaller window
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=min(14, len(df) // 3)).rsi()

            # MACD with smaller windows
            macd = ta.trend.MACD(df['close'], window_slow=min(26, len(df) // 2),
                                 window_fast=min(12, len(df) // 4))
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Bollinger Bands
            bb_window = min(20, len(df) // 2)
            bb = ta.volatility.BollingerBands(df['close'], window=bb_window)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Volume indicators (handle zero volume)
            df['volume'] = df['volume'].fillna(0)
            df['volume'] = df['volume'].replace(0, df['volume'].mean())  # Replace 0 with mean

            vol_window = min(20, len(df) // 2)
            df['volume_sma'] = df['volume'].rolling(window=vol_window).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)  # Avoid division by zero

            short_vol = min(5, len(df) // 4)
            df['volume_trend'] = (df['volume'].rolling(window=short_vol).mean() /
                                  (df['volume'].rolling(window=vol_window).mean() + 1e-8))

            # Volatility
            vol_window = min(20, len(df) // 2)
            df['volatility'] = df['returns'].rolling(window=vol_window).std()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'],
                                                       window=min(14, len(df) // 3)).average_true_range()

            # Momentum
            mom_window = min(10, len(df) // 4)
            df['momentum'] = ta.momentum.ROCIndicator(df['close'], window=mom_window).roc()
            df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'],
                                                           window=min(14, len(df) // 3)).stoch()

            # Custom indicators
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

            # Price relative to moving averages
            df['price_to_sma20'] = df['close'] / (df['sma_20'] + 1e-8)
            df['price_to_ema20'] = df['close'] / (df['ema_20'] + 1e-8)

            # Trend strength
            df['trend_strength'] = abs(df['ema_20'] - df['sma_50']) / df['close']

            # Support/Resistance levels
            support_window = min(20, len(df) // 2)
            df['resistance'] = df['high'].rolling(window=support_window).max()
            df['support'] = df['low'].rolling(window=support_window).min()
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']

            # Fill infinite values
            df = df.replace([np.inf, -np.inf], np.nan)

            # Forward fill then backward fill NaN values instead of dropping
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Only drop rows where essential columns are still NaN
            essential_cols = ['close', 'rsi', 'macd_diff']
            df = df.dropna(subset=essential_cols)

            self.logger.info(f"Calculated {len(df.columns)} indicators for {len(df)} records")
            return df

        except Exception as e:
            self.logger.error(f"Failed to calculate indicators: {e}")
            self.logger.error(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            return pd.DataFrame()

    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create feature matrix for model input"""
        feature_columns = [
            'rsi', 'macd_diff', 'bb_position', 'volume_ratio',
            'momentum', 'stoch', 'price_to_sma20', 'price_to_ema20',
            'volatility', 'trend_strength', 'support_distance', 'resistance_distance'
        ]

        available_features = [col for col in feature_columns if col in df.columns]

        if len(available_features) < 8:
            self.logger.warning(f"Insufficient features: {len(available_features)}")
            return np.array([])

        features = df[available_features].values

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        return features_scaled

    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        if len(features) < self.config.sequence_length:
            return np.array([]), np.array([])

        X, y = [], []

        for i in range(len(features) - self.config.sequence_length):
            X.append(features[i:i + self.config.sequence_length])
            y.append(labels[i + self.config.sequence_length])

        return np.array(X), np.array(y)

    def create_labels(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Create trading labels based on future returns"""
        if 'close' not in df.columns:
            return np.array([])

        future_returns = df['close'].shift(-horizon).pct_change(horizon)

        # Dynamic thresholds based on volatility
        volatility = df['returns'].std() if 'returns' in df.columns else 0.02
        buy_threshold = volatility * 1.5
        sell_threshold = -volatility * 1.5

        labels = np.ones(len(df))  # Default HOLD
        labels[future_returns > buy_threshold] = 2  # BUY
        labels[future_returns < sell_threshold] = 0  # SELL

        # Handle NaN values
        labels = labels[:-horizon]

        return labels.astype(int)


# ==============================================================================
# LSTM MODEL
# ==============================================================================

class TradingModel:
    """Advanced LSTM model for trading predictions"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = LoggerManager.setup_logger(self.__class__.__name__)
        self.model = None
        self.scaler = None
        self.history = None
        self.is_trained = False

    def build_model(self) -> Sequential:
        """Build advanced LSTM architecture"""
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(
                128, return_sequences=True,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            ), input_shape=(self.config.sequence_length, self.config.n_features)),
            Dropout(0.3),
            BatchNormalization(),

            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(
                64, return_sequences=True,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            )),
            Dropout(0.3),
            BatchNormalization(),

            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),

            # Output layer
            Dense(self.config.n_classes, activation='softmax')
        ])

        # Compile with custom optimizer
        optimizer = Adam(learning_rate=self.config.learning_rate, clipnorm=1.0)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        self.logger.info(f"Model built with {model.count_params():,} parameters")

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train model with advanced callbacks"""

        if self.model is None:
            self.build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                self.config.model_dir / 'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]

        # Train model
        self.logger.info(f"Training on {len(X_train)} samples...")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        self.history = history
        self.is_trained = True

        # Evaluate performance
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        self.logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        return {
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'history': history.history
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        return predicted_classes, confidences

    def save_model(self, path: Optional[Path] = None):
        """Save model and scaler"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        save_path = path or self.config.model_dir / 'trading_model'

        # Save model
        self.model.save(f"{save_path}.h5")

        # Save scaler if exists
        if self.scaler:
            joblib.dump(self.scaler, f"{save_path}_scaler.pkl")

        # Save config
        with open(f"{save_path}_config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, path: Optional[Path] = None) -> bool:
        """Load model and scaler"""
        load_path = path or self.config.model_dir / 'trading_model'

        try:
            # Load model
            self.model = load_model(f"{load_path}.h5")

            # Load scaler if exists
            scaler_path = f"{load_path}_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            self.is_trained = True
            self.logger.info("Model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False


# ==============================================================================
# MAIN ANALYZER
# ==============================================================================

class MemeCoinAnalyzer:
    """Main analyzer orchestrating all components"""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.logger = LoggerManager.setup_logger(self.__class__.__name__)

        # Initialize components
        self.collector = DataCollector(self.config)
        self.processor = DataProcessor(self.config)
        self.model = TradingModel(self.config)
        self.cache = CacheManager()

        # Results storage
        self.analysis_results: List[AnalysisResult] = []

        self.logger.info("Meme Coin Analyzer initialized")

    def analyze_coin(self, coin_id: str) -> AnalysisResult:
        """Analyze single coin comprehensively"""
        self.logger.info(f"Analyzing {coin_id}...")

        try:
            # Collect data
            df = self.collector.fetch_ohlc_data(coin_id, self.config.historical_days)
            if df.empty:
                raise ValueError(f"No data for {coin_id}")

            # Process data
            df_clean = self.processor.clean_data(df)
            df_indicators = self.processor.calculate_indicators(df_clean)

            if df_indicators.empty:
                raise ValueError(f"Failed to process data for {coin_id}")

            # Create features
            features = self.processor.create_features(df_indicators)
            if len(features) < self.config.sequence_length:
                raise ValueError(f"Insufficient data for {coin_id}")

            # Get latest values
            latest = df_indicators.iloc[-1]
            current_price = float(latest['close'])

            # Calculate scores
            technical_score = self._calculate_technical_score(df_indicators)
            risk_score = self._calculate_risk_score(df_indicators)

            # Get model prediction if available
            signal = TradingSignal.HOLD
            confidence = 0.5

            if self.model.is_trained:
                # Create sequence for prediction
                labels = self.processor.create_labels(df_indicators)
                X, _ = self.processor.create_sequences(features, labels)

                if len(X) > 0:
                    # Predict on latest sequence
                    pred_class, pred_conf = self.model.predict(X[-1:])
                    confidence = float(pred_conf[0])

                    # Map to signal
                    if pred_class[0] == 2 and confidence > self.config.confidence_threshold:
                        signal = TradingSignal.BUY if confidence > 0.8 else TradingSignal.STRONG_BUY
                    elif pred_class[0] == 0 and confidence > self.config.confidence_threshold:
                        signal = TradingSignal.SELL if confidence > 0.8 else TradingSignal.STRONG_SELL
                    else:
                        signal = TradingSignal.HOLD
            else:
                # Use technical analysis only
                signal = self._determine_signal_from_technicals(df_indicators)

            # Create result
            result = AnalysisResult(
                coin_id=coin_id,
                timestamp=datetime.now(),
                price=current_price,
                signal=signal,
                confidence=confidence,
                technical_score=technical_score,
                risk_score=risk_score,
                indicators={
                    'rsi': float(latest.get('rsi', 50)),
                    'macd': float(latest.get('macd_diff', 0)),
                    'volume_ratio': float(latest.get('volume_ratio', 1)),
                    'volatility': float(latest.get('volatility', 0)),
                    'momentum': float(latest.get('momentum', 0))
                },
                metadata={
                    'data_points': len(df_indicators),
                    'market_cap_rank': 0  # Would need separate API call
                }
            )

            self.analysis_results.append(result)
            self.logger.info(f"Analysis complete: {signal.value} ({confidence:.2%} confidence)")

            return result

        except Exception as e:
            self.logger.error(f"Analysis failed for {coin_id}: {e}")
            return AnalysisResult(
                coin_id=coin_id,
                timestamp=datetime.now(),
                price=0,
                signal=TradingSignal.NO_SIGNAL,
                confidence=0,
                technical_score=0,
                risk_score=1,
                indicators={},
                error=str(e)
            )

    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Calculate technical analysis score (0-1)"""
        if df.empty:
            return 0.5

        latest = df.iloc[-1]
        score = 0
        weights = 0

        # RSI score
        if 'rsi' in latest:
            rsi = latest['rsi']
            if 30 <= rsi <= 70:
                score += 0.5
            elif rsi < 30:
                score += 1  # Oversold
            else:
                score += 0  # Overbought
            weights += 1

        # MACD score
        if 'macd_diff' in latest:
            score += 1 if latest['macd_diff'] > 0 else 0
            weights += 1

        # Price vs MA score
        if 'price_to_sma20' in latest:
            score += 1 if latest['price_to_sma20'] > 1 else 0
            weights += 1

        # Volume score
        if 'volume_ratio' in latest:
            score += min(latest['volume_ratio'] / 2, 1)
            weights += 1

        # Momentum score
        if 'momentum' in latest:
            score += 1 if latest['momentum'] > 0 else 0
            weights += 1

        return score / weights if weights > 0 else 0.5

    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """Calculate risk score (0-1, higher is riskier)"""
        if df.empty:
            return 0.5

        risk = 0

        # Volatility risk
        if 'volatility' in df.columns:
            vol = df['volatility'].iloc[-1]
            risk += min(vol * 10, 0.3)

        # Drawdown risk
        if 'close' in df.columns:
            prices = df['close'].values
            running_max = np.maximum.accumulate(prices)
            drawdown = (prices - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            risk += min(max_drawdown, 0.3)

        # Volume stability risk
        if 'volume_ratio' in df.columns:
            vol_std = df['volume_ratio'].std()
            risk += min(vol_std * 0.2, 0.2)

        # Price range risk
        if all(col in df.columns for col in ['high', 'low', 'close']):
            recent = df.tail(20)
            price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
            risk += min(price_range * 0.5, 0.2)

        return min(risk, 1.0)

    def _determine_signal_from_technicals(self, df: pd.DataFrame) -> TradingSignal:
        """Determine signal from technical indicators only"""
        if df.empty:
            return TradingSignal.NO_SIGNAL

        latest = df.iloc[-1]
        buy_signals = 0
        sell_signals = 0

        # Check various indicators
        if 'rsi' in latest:
            if latest['rsi'] < 30:
                buy_signals += 2
            elif latest['rsi'] > 70:
                sell_signals += 2

        if 'macd_diff' in latest:
            if latest['macd_diff'] > 0:
                buy_signals += 1
            else:
                sell_signals += 1

        if 'bb_position' in latest:
            if latest['bb_position'] < 0.2:
                buy_signals += 1
            elif latest['bb_position'] > 0.8:
                sell_signals += 1

        if 'momentum' in latest:
            if latest['momentum'] > 5:
                buy_signals += 1
            elif latest['momentum'] < -5:
                sell_signals += 1

        # Determine signal
        if buy_signals >= 4:
            return TradingSignal.STRONG_BUY
        elif buy_signals >= 3:
            return TradingSignal.BUY
        elif sell_signals >= 4:
            return TradingSignal.STRONG_SELL
        elif sell_signals >= 3:
            return TradingSignal.SELL
        else:
            return TradingSignal.HOLD

    def batch_analyze(self, coin_ids: List[str], parallel: bool = True) -> List[AnalysisResult]:
        """Analyze multiple coins"""
        self.logger.info(f"Starting batch analysis for {len(coin_ids)} coins...")

        if parallel:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self.analyze_coin, coin_id): coin_id
                           for coin_id in coin_ids}

                results = []
                for future in as_completed(futures):
                    coin_id = futures[future]
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed to analyze {coin_id}: {e}")
                        results.append(AnalysisResult(
                            coin_id=coin_id,
                            timestamp=datetime.now(),
                            price=0,
                            signal=TradingSignal.NO_SIGNAL,
                            confidence=0,
                            technical_score=0,
                            risk_score=1,
                            indicators={},
                            error=str(e)
                        ))
        else:
            results = [self.analyze_coin(coin_id) for coin_id in coin_ids]

        return results

    def train_model(self, coin_ids: Optional[List[str]] = None) -> Optional[Dict]:
        """Train model on historical data"""
        coin_ids = coin_ids or self.config.target_coins

        self.logger.info(f"Training model on {len(coin_ids)} coins...")

        all_features = []
        all_labels = []
        successful_coins = []

        # Collect training data
        for i, coin_id in enumerate(coin_ids):
            try:
                self.logger.info(f"Processing coin {i + 1}/{len(coin_ids)}: {coin_id}")

                # Get data with more details
                df = self.collector.fetch_ohlc_data(coin_id, self.config.historical_days)
                if df.empty:
                    self.logger.warning(f"No data received for {coin_id}")
                    continue

                self.logger.info(f"Raw data: {len(df)} records for {coin_id}")

                # Validate data quality
                if len(df) < 30:
                    self.logger.warning(f"Insufficient data for {coin_id}: {len(df)} records")
                    continue

                # Check for valid price data
                if df[['open', 'high', 'low', 'close']].isnull().all().any():
                    self.logger.warning(f"Missing price data for {coin_id}")
                    continue

                # Process data
                df_clean = self.processor.clean_data(df)
                self.logger.info(f"After cleaning: {len(df_clean)} records for {coin_id}")

                if len(df_clean) < 25:
                    self.logger.warning(f"Insufficient clean data for {coin_id}: {len(df_clean)}")
                    continue

                df_indicators = self.processor.calculate_indicators(df_clean)
                self.logger.info(f"After indicators: {len(df_indicators)} records for {coin_id}")

                if len(df_indicators) < 15:
                    self.logger.warning(f"Insufficient indicator data for {coin_id}: {len(df_indicators)}")
                    continue

                # Create features and labels
                features = self.processor.create_features(df_indicators)
                if len(features) < 10:
                    self.logger.warning(f"Insufficient features for {coin_id}: {len(features)}")
                    continue

                labels = self.processor.create_labels(df_indicators)
                self.logger.info(f"Features shape: {features.shape}, Labels: {len(labels)} for {coin_id}")

                # Create sequences
                X, y = self.processor.create_sequences(features, labels)
                self.logger.info(f"Sequences: {X.shape if len(X) > 0 else 'empty'} for {coin_id}")

                if len(X) >= 5:  # Minimum sequences needed
                    all_features.append(X)
                    all_labels.append(y)
                    successful_coins.append(coin_id)
                    self.logger.info(f"✓ Added {len(X)} samples from {coin_id}")
                else:
                    self.logger.warning(f"Insufficient sequences for {coin_id}: {len(X)}")

            except Exception as e:
                self.logger.error(f"Failed to process {coin_id}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue

        if not all_features:
            self.logger.error(f"No training data collected from {len(coin_ids)} coins")
            self.logger.error("Check your API key, network connection, and coin IDs")
            return None

        self.logger.info(f"Successfully processed {len(successful_coins)} coins: {successful_coins}")

        try:
            # Combine all data
            X = np.vstack(all_features)
            y = np.hstack(all_labels)

            self.logger.info(f"Combined data - X: {X.shape}, y: {y.shape}")

            # Update config
            self.config.n_features = X.shape[2]

            # Split data
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.config.test_split, random_state=42, stratify=y
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=self.config.validation_split, random_state=42, stratify=y_temp
            )

            self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Train model
            results = self.model.train(X_train, y_train, X_val, y_val)

            # Test model
            test_pred, test_conf = self.model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)

            self.logger.info(f"Test Accuracy: {test_acc:.4f}")

            # Save model
            self.model.save_model()
            self.model.scaler = self.processor.scaler

            results['test_accuracy'] = test_acc
            results['samples'] = len(X)
            results['coins_used'] = successful_coins

            return results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        if not self.analysis_results:
            return {"error": "No analysis results available"}

        # Filter valid results
        valid_results = [r for r in self.analysis_results if r.error is None]

        # Group by signal
        signals = {}
        for signal in TradingSignal:
            signals[signal.value] = [
                r for r in valid_results if r.signal == signal
            ]

        # Find top opportunities
        opportunities = sorted(
            [r for r in valid_results if r.signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY]],
            key=lambda x: (x.confidence, x.technical_score),
            reverse=True
        )[:5]

        # Calculate statistics
        avg_confidence = np.mean([r.confidence for r in valid_results]) if valid_results else 0
        avg_risk = np.mean([r.risk_score for r in valid_results]) if valid_results else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_analyzed": len(self.analysis_results),
                "successful": len(valid_results),
                "failed": len(self.analysis_results) - len(valid_results),
                "average_confidence": float(avg_confidence),
                "average_risk": float(avg_risk)
            },
            "signals": {
                signal: len(coins) for signal, coins in signals.items()
            },
            "top_opportunities": [
                {
                    "coin_id": r.coin_id,
                    "price": r.price,
                    "signal": r.signal.value,
                    "confidence": float(r.confidence),
                    "technical_score": float(r.technical_score),
                    "risk_score": float(r.risk_score),
                    "indicators": r.indicators
                }
                for r in opportunities
            ],
            "detailed_results": [
                {
                    "coin_id": r.coin_id,
                    "timestamp": r.timestamp.isoformat(),
                    "price": r.price,
                    "signal": r.signal.value,
                    "confidence": float(r.confidence),
                    "technical_score": float(r.technical_score),
                    "risk_score": float(r.risk_score),
                    "indicators": r.indicators,
                    "error": r.error
                }
                for r in self.analysis_results
            ]
        }

        # Save report
        report_file = self.config.report_dir / f"analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to {report_file}")

        return report


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Meme Coin Trading Analysis System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--analyze', type=str, nargs='+', help='Coins to analyze')
    parser.add_argument('--top', type=int, default=10, help='Analyze top N meme coins')
    parser.add_argument('--config', type=str, help='Config file path')

    args = parser.parse_args()

    # Initialize system
    config = SystemConfig()
    analyzer = MemeCoinAnalyzer(config)

    try:
        # Train model if requested
        if args.train:
            print("\n🚀 Training Model...")
            results = analyzer.train_model()
            print(f"✅ Training complete! Test accuracy: {results['test_accuracy']:.2%}")

        # Load model if exists
        if analyzer.model.load_model():
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No trained model found - using technical analysis only")

        # Analyze specific coins
        if args.analyze:
            print(f"\n📊 Analyzing {len(args.analyze)} coins...")
            results = analyzer.batch_analyze(args.analyze)
        else:
            # Analyze top meme coins
            print(f"\n🔍 Fetching top {args.top} meme coins...")
            coins = analyzer.collector.get_top_meme_coins(args.top)

            if coins:
                coin_ids = [c['id'] for c in coins]
                print(f"📊 Analyzing {len(coin_ids)} coins...")
                results = analyzer.batch_analyze(coin_ids)
            else:
                print("❌ Failed to fetch meme coins")
                return

        # Generate report
        print("\n📝 Generating report...")
        report = analyzer.generate_report()

        # Display summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"✅ Analyzed: {report['summary']['successful']} coins")
        print(f"📊 Average Confidence: {report['summary']['average_confidence']:.1%}")
        print(f"⚠️  Average Risk: {report['summary']['average_risk']:.1%}")

        print("\n📈 Signal Distribution:")
        for signal, count in report['signals'].items():
            if count > 0:
                print(f"  {signal}: {count}")

        if report['top_opportunities']:
            print("\n🌟 Top Opportunities:")
            for i, opp in enumerate(report['top_opportunities'], 1):
                print(f"{i}. {opp['coin_id']}: {opp['signal']} "
                      f"(Confidence: {opp['confidence']:.1%}, Risk: {opp['risk_score']:.1%})")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
