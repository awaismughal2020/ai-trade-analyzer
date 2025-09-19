"""
Production-Ready Meme Coin Market Analysis System
Enhanced with logging, error handling, monitoring, and safety features
"""

import os
import sys
import json
import time
import logging
import traceback
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps
from dotenv import load_dotenv
import hashlib
import pickle

# Import existing modules
from data_collector import DataCollector
from data_processor import DataProcessor
from market_model import MarketAnalysisLSTM

# Load environment variables
load_dotenv()


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class LoggerSetup:
    @staticmethod
    def setup_logger(name: str, log_file: str = None) -> logging.Logger:
        """Setup production-grade logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        return logger


# ============================================================================
# ERROR HANDLING AND MONITORING
# ============================================================================

class SignalType(Enum):
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
    signal: SignalType
    confidence: float
    rsi: Optional[float] = None
    market_strength: Optional[str] = None
    risk_score: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict = None


@dataclass
class SystemHealth:
    """System health monitoring"""
    api_status: bool
    model_status: bool
    cache_status: bool
    last_check: datetime
    errors: List[str]
    warnings: List[str]


class CircuitBreaker:
    """Circuit breaker pattern for API calls"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.is_open:
            if (datetime.now() - self.last_failure_time).seconds > self.timeout:
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
    """Decorator for retrying failed operations"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_delay = delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= backoff
                    else:
                        raise last_exception

            raise last_exception

        return wrapper

    return decorator


def timeout_handler(timeout_seconds: int = 30):
    """Decorator for handling timeouts"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")

        return wrapper

    return decorator


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

class CacheManager:
    """Production-grade cache management"""

    def __init__(self, cache_dir: str = "./data/cache", expiry_hours: int = 1):
        self.cache_dir = cache_dir
        self.expiry_seconds = expiry_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)
        self.logger = LoggerSetup.setup_logger(__name__)

    def _get_cache_path(self, key: str) -> str:
        """Generate cache file path from key"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.cache")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached data if valid"""
        cache_path = self._get_cache_path(key)

        try:
            if os.path.exists(cache_path):
                # Check expiry
                cache_age = time.time() - os.path.getmtime(cache_path)
                if cache_age < self.expiry_seconds:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                        self.logger.debug(f"Cache hit for key: {key}")
                        return data
                else:
                    self.logger.debug(f"Cache expired for key: {key}")
                    os.remove(cache_path)
        except Exception as e:
            self.logger.error(f"Cache read error for {key}: {e}")

        return None

    def set(self, key: str, data: Any) -> bool:
        """Store data in cache"""
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                self.logger.debug(f"Cached data for key: {key}")
                return True
        except Exception as e:
            self.logger.error(f"Cache write error for {key}: {e}")
            return False

    def clear(self) -> int:
        """Clear all cache files"""
        count = 0
        for file in os.listdir(self.cache_dir):
            if file.endswith('.cache'):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                    count += 1
                except:
                    pass
        self.logger.info(f"Cleared {count} cache files")
        return count


# ============================================================================
# PRODUCTION MEME COIN ANALYZER
# ============================================================================

class ProductionMemeCoinAnalyzer:
    """Production-ready meme coin analyzer with enhanced safety and monitoring"""

    def __init__(self, config: Dict = None):
        """
        Initialize production analyzer

        Args:
            config: Configuration dictionary (uses env vars if not provided)
        """
        # Setup logging
        log_file = os.getenv('LOG_FILE', 'logs/meme_analysis.log')
        self.logger = LoggerSetup.setup_logger(__name__, log_file)
        self.logger.info("Initializing Production Meme Coin Analyzer")

        # Configuration
        self.config = config or self._load_config()

        # Components
        self.cache = CacheManager(
            cache_dir=self.config.get('cache_dir', './data/cache'),
            expiry_hours=self.config.get('cache_expiry_hours', 1)
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get('circuit_breaker_threshold', 5),
            timeout=self.config.get('circuit_breaker_timeout', 60)
        )

        # Initialize data components
        self.collector = None
        self.processor = DataProcessor()
        self.model = None

        # Monitoring
        self.health = SystemHealth(
            api_status=False,
            model_status=False,
            cache_status=True,
            last_check=datetime.now(),
            errors=[],
            warnings=[]
        )

        # Results storage
        self.analysis_history = []
        self.performance_metrics = {}

        # Safety limits
        self.max_coins_per_run = self.config.get('max_coins_per_run', 50)
        self.max_api_calls_per_minute = self.config.get('max_api_calls_per_minute', 30)
        self.api_call_times = []

        self.logger.info("Analyzer initialized successfully")

    def _load_config(self) -> Dict:
        """Load configuration from environment variables"""
        return {
            'api_key': os.getenv('COINGECKO_API_KEY'),
            'base_url': os.getenv('COINGECKO_BASE_URL', 'https://api.coingecko.com/api/v3'),
            'cache_dir': os.getenv('CACHE_DIRECTORY', './data/cache'),
            'cache_expiry_hours': int(os.getenv('CACHE_EXPIRY_HOURS', 1)),
            'max_coins_per_run': int(os.getenv('MAX_COINS_PER_RUN', 50)),
            'max_api_calls_per_minute': int(os.getenv('MAX_API_CALLS_PER_MINUTE', 30)),
            'circuit_breaker_threshold': int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', 5)),
            'circuit_breaker_timeout': int(os.getenv('CIRCUIT_BREAKER_TIMEOUT', 60)),
            'min_market_cap': float(os.getenv('MIN_MARKET_CAP', 1000000)),
            'min_volume': float(os.getenv('MIN_VOLUME_24H', 100000)),
            'risk_tolerance': float(os.getenv('RISK_TOLERANCE', 0.5)),
            'enable_notifications': os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'
        }

    def check_system_health(self) -> SystemHealth:
        """Comprehensive system health check"""
        self.logger.info("Performing system health check")

        # Clear old errors/warnings
        self.health.errors = []
        self.health.warnings = []

        # Check API connectivity
        try:
            self._test_api_connection()
            self.health.api_status = True
        except Exception as e:
            self.health.api_status = False
            self.health.errors.append(f"API connection failed: {str(e)}")

        # Check model availability
        try:
            if self.model and self.model.is_trained:
                self.health.model_status = True
            else:
                self.health.model_status = False
                self.health.warnings.append("Model not loaded or trained")
        except:
            self.health.model_status = False
            self.health.errors.append("Model check failed")

        # Check cache
        try:
            test_key = "health_check_test"
            self.cache.set(test_key, {"test": "data"})
            if self.cache.get(test_key):
                self.health.cache_status = True
            else:
                self.health.cache_status = False
                self.health.warnings.append("Cache write/read failed")
        except Exception as e:
            self.health.cache_status = False
            self.health.errors.append(f"Cache system error: {str(e)}")

        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024 ** 3)
            if free_gb < 1:
                self.health.errors.append(f"Low disk space: {free_gb:.2f}GB")
            elif free_gb < 5:
                self.health.warnings.append(f"Disk space warning: {free_gb:.2f}GB")
        except:
            pass

        self.health.last_check = datetime.now()

        # Log health status
        if self.health.errors:
            self.logger.error(f"Health check errors: {self.health.errors}")
        if self.health.warnings:
            self.logger.warning(f"Health check warnings: {self.health.warnings}")

        return self.health

    @retry_on_failure(max_retries=3, delay=2.0)
    def _test_api_connection(self):
        """Test API connectivity with retry logic"""
        headers = {'accept': 'application/json'}
        if self.config.get('api_key'):
            headers['x-cg-demo-api-key'] = self.config['api_key']

        response = requests.get(
            f"{self.config['base_url']}/ping",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return True

    def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        current_time = time.time()

        # Clean old timestamps
        self.api_call_times = [
            t for t in self.api_call_times
            if current_time - t < 60
        ]

        # Check rate limit
        if len(self.api_call_times) >= self.max_api_calls_per_minute:
            wait_time = 60 - (current_time - self.api_call_times[0])
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.api_call_times = []

        self.api_call_times.append(current_time)

    @timeout_handler(timeout_seconds=30)
    @retry_on_failure(max_retries=3)
    def fetch_top_meme_coins(self, limit: int = 20) -> List[Dict]:
        """
        Fetch top meme coins with production safety

        Args:
            limit: Number of coins to fetch

        Returns:
            List of meme coin data
        """
        self.logger.info(f"Fetching top {limit} meme coins")

        # Check rate limit
        self._rate_limit_check()

        # Check cache first
        cache_key = f"meme_coins_{limit}_{self.config['min_market_cap']}_{self.config['min_volume']}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            self.logger.info(f"Using cached meme coins data ({len(cached_data)} coins)")
            return cached_data

        # Fetch from API with circuit breaker
        try:
            headers = {'accept': 'application/json'}
            if self.config.get('api_key'):
                headers['x-cg-demo-api-key'] = self.config['api_key']

            params = {
                'vs_currency': 'usd',
                'category': 'meme-token',
                'order': 'market_cap_desc',
                'per_page': min(limit * 2, 250),  # Safety limit
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h,7d,30d'
            }

            response = self.circuit_breaker.call(
                requests.get,
                f"{self.config['base_url']}/coins/markets",
                params=params,
                headers=headers,
                timeout=15
            )

            response.raise_for_status()
            coins_data = response.json()

            # Filter and validate
            filtered_coins = []
            for coin in coins_data:
                try:
                    # Validate data
                    if not coin.get('id') or not coin.get('symbol'):
                        continue

                    market_cap = float(coin.get('market_cap', 0))
                    volume = float(coin.get('total_volume', 0))

                    if (market_cap >= self.config['min_market_cap'] and
                            volume >= self.config['min_volume']):
                        filtered_coins.append({
                            'id': coin['id'],
                            'symbol': coin['symbol'].upper(),
                            'name': coin['name'],
                            'market_cap': market_cap,
                            'volume_24h': volume,
                            'price': float(coin.get('current_price', 0)),
                            'price_change_24h': float(coin.get('price_change_percentage_24h', 0)),
                            'price_change_7d': float(coin.get('price_change_percentage_7d', 0)),
                            'price_change_30d': float(coin.get('price_change_percentage_30d', 0)),
                            'market_cap_rank': int(coin.get('market_cap_rank', 999))
                        })

                    if len(filtered_coins) >= limit:
                        break

                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid coin data: {e}")
                    continue

            # Cache the results
            self.cache.set(cache_key, filtered_coins)

            self.logger.info(f"Fetched {len(filtered_coins)} meme coins")
            return filtered_coins

        except Exception as e:
            self.logger.error(f"Failed to fetch meme coins: {e}")
            raise

    def analyze_coin_safe(self, coin_id: str, days: int = 30) -> AnalysisResult:
        """
        Analyze a coin with comprehensive error handling

        Args:
            coin_id: Coin identifier
            days: Days of historical data

        Returns:
            AnalysisResult object
        """
        self.logger.info(f"Analyzing coin: {coin_id}")

        try:
            # Check rate limit
            self._rate_limit_check()

            # Initialize collector if needed
            if self.collector is None:
                original_coins = os.getenv('TARGET_COINS', '')
                os.environ['TARGET_COINS'] = coin_id
                self.collector = DataCollector()
                os.environ['TARGET_COINS'] = original_coins

            # Fetch data with caching
            cache_key = f"coin_data_{coin_id}_{days}"
            coin_data = self.cache.get(cache_key)

            if coin_data is None:
                coin_data = self.collector.get_coin_data(coin_id, days)
                if coin_data is not None and not coin_data.empty:
                    self.cache.set(cache_key, coin_data)
            else:
                # Convert cached data back to DataFrame if needed
                if isinstance(coin_data, dict):
                    coin_data = pd.DataFrame(coin_data)

            if coin_data is None or coin_data.empty:
                raise ValueError(f"No data available for {coin_id}")

            # Process data
            clean_data = self.processor.clean_data(coin_data)
            if clean_data.empty:
                raise ValueError("Data cleaning failed")

            processed_data = self.processor.calculate_technical_indicators(clean_data)
            if processed_data.empty:
                raise ValueError("Indicator calculation failed")

            # Extract latest values
            latest = processed_data.iloc[-1]

            # Determine signal with safety checks
            signal = self._determine_signal_safe(processed_data)

            # Calculate risk score
            risk_score = self._calculate_risk_score(processed_data)

            # Get model prediction if available
            confidence = 0.5  # Default confidence
            if self.model and self.model.is_trained:
                try:
                    # Create sequences for prediction
                    labeled_data = self.processor.create_labels(processed_data)
                    if not labeled_data.empty:
                        X, _ = self.processor.create_sequences(labeled_data)
                        if len(X) > 0:
                            prediction = self.model.predict(X[-1])
                            confidence = prediction['confidence']
                            # Update signal based on model
                            if prediction['signal'] == 'BUY' and confidence > 0.7:
                                signal = SignalType.STRONG_BUY if confidence > 0.85 else SignalType.BUY
                            elif prediction['signal'] == 'SELL' and confidence > 0.7:
                                signal = SignalType.STRONG_SELL if confidence > 0.85 else SignalType.SELL
                except Exception as e:
                    self.logger.warning(f"Model prediction failed for {coin_id}: {e}")

            # Create result
            result = AnalysisResult(
                coin_id=coin_id,
                timestamp=datetime.now(),
                price=float(latest['close']),
                signal=signal,
                confidence=confidence,
                rsi=float(latest.get('rsi_14', 50)),
                market_strength=self._calculate_market_strength(processed_data),
                risk_score=risk_score,
                metadata={
                    'data_points': len(processed_data),
                    'volatility': float(processed_data['close'].pct_change().std()),
                    'volume_ratio': float(latest.get('volume_ratio', 1))
                }
            )

            # Store in history
            self.analysis_history.append(result)

            self.logger.info(f"Analysis complete for {coin_id}: {signal.value} (confidence: {confidence:.2%})")
            return result

        except Exception as e:
            self.logger.error(f"Analysis failed for {coin_id}: {e}")
            return AnalysisResult(
                coin_id=coin_id,
                timestamp=datetime.now(),
                price=0,
                signal=SignalType.NO_SIGNAL,
                confidence=0,
                error=str(e)
            )

    def _determine_signal_safe(self, df: pd.DataFrame) -> SignalType:
        """Determine trading signal with safety checks"""
        try:
            latest = df.iloc[-1]
            score = 0

            # RSI analysis
            rsi = latest.get('rsi_14', 50)
            if rsi < 20:
                score -= 2  # Very oversold
            elif rsi < 30:
                score -= 1  # Oversold
            elif rsi > 80:
                score += 2  # Very overbought
            elif rsi > 70:
                score += 1  # Overbought

            # Trend analysis
            if len(df) > 5:
                recent_change = (df.iloc[-1]['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close']
                if recent_change > 0.1:
                    score += 2
                elif recent_change > 0.05:
                    score += 1
                elif recent_change < -0.1:
                    score -= 2
                elif recent_change < -0.05:
                    score -= 1

            # Volume analysis
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 2:
                score += 1 if score > 0 else -1  # Strengthen existing signal

            # Convert score to signal
            if score <= -2:
                return SignalType.STRONG_BUY
            elif score == -1:
                return SignalType.BUY
            elif score >= 2:
                return SignalType.STRONG_SELL
            elif score == 1:
                return SignalType.SELL
            else:
                return SignalType.HOLD

        except Exception as e:
            self.logger.error(f"Signal determination failed: {e}")
            return SignalType.NO_SIGNAL

    def _calculate_risk_score(self, df: pd.DataFrame) -> float:
        """Calculate risk score (0-1, higher is riskier)"""
        try:
            risk_score = 0.0

            # Volatility component (40%)
            volatility = df['close'].pct_change().std()
            risk_score += min(volatility * 10, 0.4)  # Cap at 0.4

            # Drawdown component (30%)
            cumulative_returns = (1 + df['close'].pct_change()).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            risk_score += min(max_drawdown, 0.3)  # Cap at 0.3

            # Volume stability component (20%)
            if 'volume' in df.columns:
                volume_cv = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 1
                risk_score += min(volume_cv * 0.1, 0.2)  # Cap at 0.2

            # Price range component (10%)
            price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
            risk_score += min(price_range * 0.05, 0.1)  # Cap at 0.1

            return min(risk_score, 1.0)  # Ensure between 0 and 1

        except Exception as e:
            self.logger.error(f"Risk calculation failed: {e}")
            return 0.5  # Default medium risk

    def _calculate_market_strength(self, df: pd.DataFrame) -> str:
        """Calculate market strength with error handling"""
        try:
            score = 0
            latest = df.iloc[-1]

            # Price momentum
            if len(df) > 20:
                momentum = (df.iloc[-1]['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close']
                if momentum > 0.2:
                    score += 3
                elif momentum > 0.1:
                    score += 2
                elif momentum > 0:
                    score += 1
                elif momentum < -0.2:
                    score -= 3
                elif momentum < -0.1:
                    score -= 2
                else:
                    score -= 1

            # RSI
            rsi = latest.get('rsi_14', 50)
            if 40 < rsi < 60:
                score += 1  # Neutral zone
            elif rsi <= 30:
                score += 2  # Oversold
            elif rsi >= 70:
                score -= 2  # Overbought

            # Volume
            if latest.get('volume_ratio', 1) > 1.5:
                score += 1

            # Determine strength
            if score >= 4:
                return 'VERY_STRONG'
            elif score >= 2:
                return 'STRONG'
            elif score >= 0:
                return 'NEUTRAL'
            elif score >= -2:
                return 'WEAK'
            else:
                return 'VERY_WEAK'

        except Exception as e:
            self.logger.error(f"Market strength calculation failed: {e}")
            return 'UNKNOWN'

    def batch_analyze(self,
                      coin_ids: List[str],
                      days: int = 30,
                      max_workers: int = 3) -> List[AnalysisResult]:
        """
        Analyze multiple coins with parallel processing

        Args:
            coin_ids: List of coin IDs
            days: Historical days for analysis
            max_workers: Maximum parallel workers

        Returns:
            List of analysis results
        """
        self.logger.info(f"Starting batch analysis for {len(coin_ids)} coins")

        # Limit coins per run
        if len(coin_ids) > self.max_coins_per_run:
            self.logger.warning(f"Limiting analysis to {self.max_coins_per_run} coins")
            coin_ids = coin_ids[:self.max_coins_per_run]

        results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.analyze_coin_safe, coin_id, days): coin_id
                for coin_id in coin_ids
            }

            # Process completed tasks
            for future in futures:
                coin_id = futures[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per coin
                    results.append(result)

                    # Log progress
                    if len(results) % 5 == 0:
                        self.logger.info(f"Progress: {len(results)}/{len(coin_ids)} coins analyzed")

                except TimeoutError:
                    self.logger.error(f"Analysis timeout for {coin_id}")
                    results.append(AnalysisResult(
                        coin_id=coin_id,
                        timestamp=datetime.now(),
                        price=0,
                        signal=SignalType.NO_SIGNAL,
                        confidence=0,
                        error="Analysis timeout"
                    ))
                except Exception as e:
                    self.logger.error(f"Analysis failed for {coin_id}: {e}")
                    results.append(AnalysisResult(
                        coin_id=coin_id,
                        timestamp=datetime.now(),
                        price=0,
                        signal=SignalType.NO_SIGNAL,
                        confidence=0,
                        error=str(e)
                    ))

        self.logger.info(f"Batch analysis complete: {len(results)} results")
        return results

    def generate_report(self, results: List[AnalysisResult]) -> Dict:
        """Generate comprehensive analysis report"""
        self.logger.info("Generating analysis report")

        # Filter valid results
        valid_results = [r for r in results if r.error is None]

        # Calculate statistics
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_analyzed': len(results),
                'successful': len(valid_results),
                'failed': len(results) - len(valid_results),
                'strong_buy_signals': sum(1 for r in valid_results if r.signal == SignalType.STRONG_BUY),
                'buy_signals': sum(1 for r in valid_results if r.signal == SignalType.BUY),
                'hold_signals': sum(1 for r in valid_results if r.signal == SignalType.HOLD),
                'sell_signals': sum(1 for r in valid_results if r.signal == SignalType.SELL),
                'strong_sell_signals': sum(1 for r in valid_results if r.signal == SignalType.STRONG_SELL),
            },
            'top_opportunities': [],
            'high_risk_coins': [],
            'system_health': asdict(self.health),
            'detailed_results': []
        }

        # Sort by confidence and signal strength
        sorted_results = sorted(
            valid_results,
            key=lambda x: (
                x.signal == SignalType.STRONG_BUY,
                x.signal == SignalType.BUY,
                x.confidence
            ),
            reverse=True
        )

        # Top opportunities
        for result in sorted_results[:5]:
            if result.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                report['top_opportunities'].append({
                    'coin_id': result.coin_id,
                    'price': result.price,
                    'signal': result.signal.value,
                    'confidence': result.confidence,
                    'rsi': result.rsi,
                    'market_strength': result.market_strength,
                    'risk_score': result.risk_score
                })

        # High risk coins
        high_risk = [r for r in valid_results if r.risk_score and r.risk_score > 0.7]
        for result in sorted(high_risk, key=lambda x: x.risk_score, reverse=True)[:5]:
            report['high_risk_coins'].append({
                'coin_id': result.coin_id,
                'risk_score': result.risk_score,
                'signal': result.signal.value
            })

        # All results
        for result in results:
            report['detailed_results'].append({
                'coin_id': result.coin_id,
                'timestamp': result.timestamp.isoformat(),
                'price': result.price,
                'signal': result.signal.value,
                'confidence': result.confidence,
                'rsi': result.rsi,
                'market_strength': result.market_strength,
                'risk_score': result.risk_score,
                'error': result.error,
                'metadata': result.metadata
            })

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/production_report_{timestamp}.json"
        os.makedirs("data", exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to {report_file}")

        return report

    def run_production_analysis(self,
                                top_n: int = 20,
                                days: int = 30,
                                save_report: bool = True) -> Dict:
        """
        Run complete production analysis pipeline

        Args:
            top_n: Number of top coins to analyze
            days: Historical days for analysis
            save_report: Whether to save report to file

        Returns:
            Analysis report dictionary
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Production Analysis Pipeline")
        self.logger.info("=" * 60)

        try:
            # Step 1: System health check
            self.logger.info("Step 1: System health check")
            health = self.check_system_health()

            if health.errors:
                self.logger.error(f"System health issues detected: {health.errors}")
                if not health.api_status:
                    raise Exception("API connection failed - cannot proceed")

            # Step 2: Load model if available
            self.logger.info("Step 2: Loading ML model")
            try:
                model_path = os.getenv('MODEL_SAVE_PATH', './models/meme_coin_market_model')
                self.model = MarketAnalysisLSTM()
                if self.model.load_model(model_path):
                    self.logger.info("Model loaded successfully")
                else:
                    self.logger.warning("Model not found - proceeding without ML predictions")
                    self.model = None
            except Exception as e:
                self.logger.warning(f"Model loading failed: {e}")
                self.model = None

            # Step 3: Fetch top meme coins
            self.logger.info(f"Step 3: Fetching top {top_n} meme coins")
            meme_coins = self.fetch_top_meme_coins(limit=top_n)

            if not meme_coins:
                raise Exception("No meme coins fetched")

            self.logger.info(f"Fetched {len(meme_coins)} meme coins")

            # Step 4: Analyze coins
            self.logger.info(f"Step 4: Analyzing {len(meme_coins)} coins")
            coin_ids = [coin['id'] for coin in meme_coins]
            results = self.batch_analyze(coin_ids, days=days, max_workers=3)

            # Step 5: Generate report
            self.logger.info("Step 5: Generating report")
            report = self.generate_report(results)

            # Step 6: Send notifications if enabled
            if self.config.get('enable_notifications'):
                self._send_notifications(report)

            self.logger.info("=" * 60)
            self.logger.info("Production Analysis Complete!")
            self.logger.info(
                f"Successfully analyzed {report['summary']['successful']}/{report['summary']['total_analyzed']} coins")
            self.logger.info(f"Found {len(report['top_opportunities'])} trading opportunities")
            self.logger.info("=" * 60)

            return report

        except Exception as e:
            self.logger.error(f"Production analysis failed: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _send_notifications(self, report: Dict):
        """Send notifications for important signals"""
        try:
            # This is a placeholder for notification logic
            # Implement webhook, email, or other notification methods
            strong_buys = [
                opp for opp in report['top_opportunities']
                if opp['signal'] == 'STRONG_BUY'
            ]

            if strong_buys:
                self.logger.info(f"Would send notification for {len(strong_buys)} strong buy signals")
                # Implement actual notification logic here

        except Exception as e:
            self.logger.error(f"Notification failed: {e}")


def main():
    """Main entry point for production system"""
    import argparse

    parser = argparse.ArgumentParser(description='Production Meme Coin Analyzer')
    parser.add_argument('--coins', type=int, default=20, help='Number of coins to analyze')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before running')
    parser.add_argument('--health-check', action='store_true', help='Only run health check')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ProductionMemeCoinAnalyzer()

    # Clear cache if requested
    if args.clear_cache:
        analyzer.cache.clear()

    # Health check only
    if args.health_check:
        health = analyzer.check_system_health()
        print(f"\nSystem Health Status:")
        print(f"- API Status: {'✅' if health.api_status else '❌'}")
        print(f"- Model Status: {'✅' if health.model_status else '❌'}")
        print(f"- Cache Status: {'✅' if health.cache_status else '❌'}")
        if health.errors:
            print(f"- Errors: {health.errors}")
        if health.warnings:
            print(f"- Warnings: {health.warnings}")
        return

    # Run full analysis
    try:
        report = analyzer.run_production_analysis(
            top_n=args.coins,
            days=args.days
        )

        print("\n✅ Analysis completed successfully!")
        print(f"📊 Analyzed: {report['summary']['successful']} coins")
        print(f"🎯 Strong Buy: {report['summary']['strong_buy_signals']}")
        print(f"💰 Buy: {report['summary']['buy_signals']}")
        print(f"⏸️  Hold: {report['summary']['hold_signals']}")
        print(f"📉 Sell: {report['summary']['sell_signals']}")

        if report['top_opportunities']:
            print("\n🌟 Top Opportunities:")
            for i, opp in enumerate(report['top_opportunities'][:3], 1):
                print(f"{i}. {opp['coin_id']}: {opp['signal']} "
                      f"(Confidence: {opp['confidence']:.1%}, Risk: {opp['risk_score']:.2f})")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
