"""
Coin Market Analyzer - Unified Configuration
Supports both Meme Tokens and Perpetual Futures (Perps)
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent

# Load .env from project root so SENTRY_DSN and other vars are available (e.g. when running uvicorn locally)
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Data subdirectories (for training data collection)
CANDLES_DIR = DATA_DIR / "candles"
HOLDERS_DIR = DATA_DIR / "holders"
MINTS_DIR = DATA_DIR / "mints"
TRADES_DIR = DATA_DIR / "trades"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"


class TokenType(str, Enum):
    """Token types supported by the system"""
    MEME = "meme"
    PERPS = "perps"
    
    @classmethod
    def from_string(cls, value: str) -> 'TokenType':
        """Convert string to TokenType enum"""
        value_lower = value.lower().strip()
        if value_lower == "perps":
            return cls.PERPS
        return cls.MEME


# =============================================================================
# API Configuration
# =============================================================================

@dataclass
class APIConfig:
    """API endpoint configuration"""
    # Internal API (for both meme and perps data)
    INTERNAL_BASE_URL: str = os.getenv("INTERNAL_API_BASE_URL", "http://52.3.148.51:3000")
    
    # Birdeye API (for meme tokens on Solana)
    BIRDEYE_API_KEY: str = os.getenv("BIRDEYE_API_KEY", "")
    BIRDEYE_BASE_URL: str = "https://public-api.birdeye.so"
    BIRDEYE_CHAIN: str = os.getenv("BIRDEYE_CHAIN", "solana")
    BIRDEYE_RATE_LIMIT: int = int(os.getenv("BIRDEYE_RATE_LIMIT", "15"))
    
    # Request settings (API server: 3-8s timeout, 2-3 retries)
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "8"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))
    
    # Rate Limiting (to prevent overwhelming external APIs)
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "15"))  # Max parallel connections
    REQUESTS_PER_SECOND: int = int(os.getenv("REQUESTS_PER_SECOND", "50"))  # Target requests/sec
    MIN_REQUEST_INTERVAL: float = float(os.getenv("MIN_REQUEST_INTERVAL", "0.05"))  # 50ms between requests
    BATCH_DELAY: float = float(os.getenv("BATCH_DELAY", "0.1"))  # 100ms delay between batches
    
    # Retry with exponential backoff
    BACKOFF_FACTOR: float = float(os.getenv("BACKOFF_FACTOR", "2.0"))  # Exponential backoff multiplier
    MAX_BACKOFF: float = float(os.getenv("MAX_BACKOFF", "30.0"))  # Maximum backoff time in seconds
    JITTER_MAX: float = float(os.getenv("JITTER_MAX", "0.5"))  # Random jitter to prevent thundering herd
    
    # Connection recovery
    CONNECTION_COOLDOWN: float = float(os.getenv("CONNECTION_COOLDOWN", "2.0"))  # Cooldown after connection errors
    CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))  # Consecutive failures before circuit break
    CIRCUIT_BREAKER_RESET: float = float(os.getenv("CIRCUIT_BREAKER_RESET", "30.0"))  # Seconds to wait before retrying after circuit break
    CIRCUIT_BREAKER_COOLDOWN: float = float(os.getenv("CIRCUIT_BREAKER_COOLDOWN", "20.0"))  # Cooldown period for new circuit breaker (seconds)
    
    # Sentry error tracking
    SENTRY_DSN: str = os.getenv("SENTRY_DSN", "")
    SENTRY_ENVIRONMENT: str = os.getenv("SENTRY_ENVIRONMENT", "development")
    SENTRY_SERVER_TAG: str = os.getenv("SENTRY_SERVER_TAG", "AI-Server")
    
    # Pagination
    RECORDS_PER_PAGE: int = 100
    DELAY_BETWEEN_REQUESTS: float = 0.1
    DEFAULT_USER_PAGE_SIZE: int = 20
    
    # Common endpoints (used across data fetchers)
    ENDPOINT_USERS: str = "/users"
    ENDPOINT_USER_TIMELINE: str = "/user-profiling/timeline"
    ENDPOINT_USER_PROFILING: str = "/user-profiling"
    ENDPOINT_USER_PROFILING_TRADES: str = "/user-profiling/trades"
    ENDPOINT_MINT_TIMELINE: str = "/mint/timeline"
    USER_PROFILING_TYPE: str = "spot"
    USER_PROFILING_DEFAULT_DAYS: int = int(os.getenv('USER_PROFILING_DEFAULT_DAYS', '80'))  # default observation window length (days) when no date range provided
    BEHAVIOUR_TRADE_CAP: int = int(os.getenv('BEHAVIOUR_TRADE_CAP', '500'))
    USER_PROFILING_TIMEOUT: int = int(os.getenv('USER_PROFILING_TIMEOUT', '90'))  # seconds (longer for user profiling - processes more data)
    USER_PROFILING_RETRY_DELAY: float = float(os.getenv('USER_PROFILING_RETRY_DELAY', '2.0'))  # seconds to wait before one retry on timeout/5xx
    
    # CryptoRank global market data
    CRYPTORANK_ENDPOINT_LATEST: str = "/cryptorank/global/latest"
    CRYPTORANK_ENDPOINT_AT: str = "/cryptorank/global/at"
    CRYPTORANK_CACHE_TTL_SECONDS: int = int(os.getenv("CRYPTORANK_CACHE_TTL", "1800"))
    CRYPTORANK_TIMEOUT: int = int(os.getenv("CRYPTORANK_TIMEOUT", "5"))
    
    # Market regime thresholds (for safety overrides)
    MARKET_EXTREME_FEAR_THRESHOLD: int = int(os.getenv("MARKET_EXTREME_FEAR_THRESHOLD", "15"))
    MARKET_EXTREME_GREED_THRESHOLD: int = int(os.getenv("MARKET_EXTREME_GREED_THRESHOLD", "80"))
    MARKET_FEAR_CONFIDENCE_PENALTY: float = float(os.getenv("MARKET_FEAR_CONFIDENCE_PENALTY", "0.15"))
    MARKET_FEAR_ALONE_CONFIDENCE_PENALTY: float = float(os.getenv("MARKET_FEAR_ALONE_CONFIDENCE_PENALTY", "0.10"))
    MARKET_GREED_CONFIDENCE_PENALTY: float = float(os.getenv("MARKET_GREED_CONFIDENCE_PENALTY", "0.10"))
    MARKET_FEAR_CRASH_MCAP_THRESHOLD: float = float(os.getenv("MARKET_FEAR_CRASH_MCAP_THRESHOLD", "-2.0"))
    BEAR_MARKET_CONFIDENCE_PENALTY: float = float(os.getenv("BEAR_MARKET_CONFIDENCE_PENALTY", "0.15"))
    BEAR_MARKET_FAVORABILITY_THRESHOLD: float = float(os.getenv("BEAR_MARKET_FAVORABILITY_THRESHOLD", "0.2"))
    SELL_FEAR_CONFIDENCE_BOOST: float = float(os.getenv("SELL_FEAR_CONFIDENCE_BOOST", "0.10"))
    SELL_BEAR_CONFIDENCE_BOOST: float = float(os.getenv("SELL_BEAR_CONFIDENCE_BOOST", "0.10"))


@dataclass
class MemeConfig:
    """Configuration specific to meme token analysis"""
    # API endpoints
    BASE_URL: str = os.getenv("MEME_API_BASE_URL", "http://52.3.148.51:3000")
    
    # Endpoints
    ENDPOINT_CANDLES: str = "/mint/range-candles"
    ENDPOINT_HOLDERS: str = "/mint/holders"
    ENDPOINT_METADATA: str = "/mint/mint-metadata"
    ENDPOINT_USER_HOLDINGS: str = "/mint/mint-timeline"
    ENDPOINT_USER_PROFILING: str = "/user-profiling/summary"
    ENDPOINT_USER_TRADES: str = "/user-profiling/trades"
    
    # Model paths
    MODEL_PATH: str = str(MODELS_DIR / "meme" / "xgboost_hybrid_model.pkl")
    SCALER_PATH: str = str(MODELS_DIR / "meme" / "feature_scaler.pkl")
    FEATURES_PATH: str = str(MODELS_DIR / "meme" / "feature_columns.pkl")
    
    # Training data path
    TRAIN_DATA_PATH: str = str(DATA_DIR / "meme" / "train_data.csv")
    
    # Default parameters
    DEFAULT_CANDLE_DAYS: int = 10
    DEFAULT_HOLDER_LIMIT: int = 200
    
    # Signal effectiveness window (hours)
    # The final signal/analysis is calibrated for this window:
    # - Analyzes the LAST N hours of activity for signal generation
    # - Signal is predicted to be effective for the NEXT N hours
    SIGNAL_EFFECTIVENESS_HOURS: int = 24
    # Number of recent candles to use for 24h-focused analysis (24h * 12 candles/hour for 5min candles)
    RECENT_ANALYSIS_CANDLES: int = 288
    
    # Staleness threshold: only fallback to Birdeye if primary candle data is older than this
    CANDLE_STALE_THRESHOLD_HOURS: int = int(os.getenv("MEME_CANDLE_STALE_THRESHOLD_HOURS", "24"))

    # Meme prediction data source: "internal" = mint API first (current), "birdeye" = Birdeye first with internal fallback
    MEME_PRIMARY_DATA_SOURCE: str = os.getenv("MEME_PRIMARY_DATA_SOURCE", "birdeye")
    # When Birdeye is primary: total time budget for Birdeye fetch before falling back to internal (seconds)
    MEME_BIRDEYE_PRIMARY_TIMEOUT_SECONDS: float = float(os.getenv("MEME_BIRDEYE_PRIMARY_TIMEOUT_SECONDS", "28"))
    # When Birdeye is primary: per-call timeout for each Birdeye request (seconds)
    MEME_BIRDEYE_PER_CALL_TIMEOUT_SECONDS: float = float(os.getenv("MEME_BIRDEYE_PER_CALL_TIMEOUT_SECONDS", "12"))


def _get_perps_training_tickers() -> list:
    """Get perps training tickers from env or use defaults"""
    tickers_env = os.getenv("PERPS_TRAINING_TICKERS", "")
    if tickers_env:
        return [t.strip() for t in tickers_env.split(",") if t.strip()]
    return [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'HYPE-USD', 'ZRO-USD',
        'XRP-USD', 'ZEC-USD', 'MON-USD', 'BERA-USD', 'ASTER-USD',
        'PAXG-USD', 'FARTCOIN-USD', 'SUI-USD', 'PUMP-USD',
        'BNB-USD', 'XPL-USD', 'DOGE-USD', 'AXS-USD', 'LIT-USD',
        'XMR-USD', 'LTC-USD', 'KAITO-USD', 'LINK-USD', 'ORDI-USD',
        'TRB-USD', 'GOAT-USD', 'SUPER-USD', 'SAGA-USD',
        'MAVIA-USD', 'PEOPLE-USD', 'CHILLGUY-USD', 'ENA-USD', 'BLAST-USD',
        'TNSR-USD', 'UMA-USD', 'XAI-USD', 'RSR-USD', 'TST-USD',
        'IOTA-USD', 'BOME-USD', 'DOOD-USD', 'NOT-USD',
        'MEW-USD', 'IP-USD', 'APE-USD', 'ATOM-USD', 'AVAX-USD',
        'OP-USD', 'DYDX-USD', 'kPEPE-USD'
    ]


@dataclass
class PerpsConfig:
    """Configuration specific to perpetual futures analysis"""
    # API endpoints
    BASE_URL: str = os.getenv("PERPS_API_BASE_URL", "http://52.3.148.51:3000")
    
    # HyperEVM endpoints (primary — replaced DYDX endpoints)
    ENDPOINT_CANDLES: str = "/hyper-evm/candle-snapshot"
    ENDPOINT_FUNDING_HISTORY: str = "/hyper-evm/funding-history"
    ENDPOINT_MARKETS: str = "/hyper-evm/meta-asset-ctxs"
    
    # Block-liquidity trades (replaces user-profiling/perps-trades-by-coin)
    # Path: /block-liquidity/trades/{coin}?start_timestamp=ISO&end_timestamp=ISO
    ENDPOINT_BLOCK_LIQUIDITY_TRADES: str = "/block-liquidity/trades"
    ENDPOINT_FUNDING: str = "/block-liquidity/trades"  # alias for token_type_router / funding source
    # Block-liquidity historical positions (replaces user-profiling/perps-trade for user's perps history)
    # GET /block-liquidity/historical-positions?address=0x...&start_timestamp=ISO&end_timestamp=ISO
    ENDPOINT_BLOCK_LIQUIDITY_HISTORICAL_POSITIONS: str = "/block-liquidity/historical-positions"
    # Block-liquidity whale flow (market-wide whale positioning across all perps)
    # GET /block-liquidity/whale/flow?start_timestamp=ISO&end_timestamp=ISO
    ENDPOINT_BLOCK_LIQUIDITY_WHALE_FLOW: str = "/block-liquidity/whale/flow"
    ENDPOINT_USER_PERPS_TRADES: str = "/user-profiling/perps-trade"  # user-profiling API for user's perps trades
    # When True, fetch_user_perps_trades uses INTERNAL_BASE_URL + ENDPOINT_USER_PERPS_TRADES (from, to, address, page)
    PERPS_USER_TRADES_USE_USER_PROFILING: bool = os.getenv("PERPS_USER_TRADES_USE_USER_PROFILING", "true").lower() == "true"
    ENDPOINT_HYPER_USERS: str = "/hyper-evm/hyper-liquid-users"
    
    # Resolution mapping: DYDX format -> HyperLiquid format
    RESOLUTION_MAP: Dict[str, str] = field(default_factory=lambda: {
        "1MIN": "1m",
        "5MINS": "5m",
        "15MINS": "15m",
        "30MINS": "30m",
        "1HOUR": "1h",
        "4HOURS": "4h",
        "1DAY": "1d",
    })
    
    # Model paths
    MODEL_PATH: str = str(MODELS_DIR / "perps" / "perps_model.pkl")
    SCALER_PATH: str = str(MODELS_DIR / "perps" / "perps_scaler.pkl")
    FEATURES_PATH: str = str(MODELS_DIR / "perps" / "perps_features.pkl")
    TICKER_ENCODER_PATH: str = str(MODELS_DIR / "perps" / "perps_ticker_encoder.pkl")
    
    # Training data path
    TRAIN_DATA_PATH: str = str(DATA_DIR / "perps" / "train_data.csv")
    
    # Tickers used for training and data pipeline
    # Override via env: PERPS_TRAINING_TICKERS="BTC-USDC,ETH-USDC,SOL-USDC"
    TRAINING_TICKERS: list = field(default_factory=lambda: _get_perps_training_tickers())
    
    # Default parameters
    DEFAULT_RESOLUTION: str = "1HOUR"
    MAX_CANDLES: int = 500
    MAX_FUNDING: int = 200
    
    # Signal effectiveness window (hours)
    # The final signal/analysis is calibrated for this window:
    # - Analyzes the LAST N hours of activity for signal generation
    # - Signal is predicted to be effective for the NEXT N hours
    SIGNAL_EFFECTIVENESS_HOURS: int = 24
    # Number of recent candles to use for 24h-focused analysis (24h * 1 candle/hour)
    RECENT_ANALYSIS_CANDLES: int = 24
    # Large trader analysis lookback (aligned with signal effectiveness)
    LARGE_TRADER_LOOKBACK_DAYS: int = 1  # 1 day = 24 hours
    
    # Staleness threshold: log warning if perps candle data is older than this
    CANDLE_STALE_THRESHOLD_HOURS: int = int(os.getenv("PERPS_CANDLE_STALE_THRESHOLD_HOURS", "24"))
    
    # Data availability
    DATA_START_DATE: str = "2026-01-22"
    
    # Request settings
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # Pagination
    RECORDS_PER_PAGE: int = 100
    DELAY_BETWEEN_REQUESTS: float = 0.1
    
    # Ticker mapping for perps symbols
    TICKER_MAPPING: dict = None
    
    def __post_init__(self):
        if self.TICKER_MAPPING is None:
            self.TICKER_MAPPING = {
                "BTC": "BTC-USD",
                "ETH": "ETH-USD",
                "SOL": "SOL-USD",
                "DOGE": "DOGE-USD",
                "AVAX": "AVAX-USD",
                "LINK": "LINK-USD",
                "ARB": "ARB-USD",
                "OP": "OP-USD",
            }
    
    def to_hyper_coin(self, ticker: str) -> str:
        """Convert DYDX ticker format to HyperLiquid coin format. BTC-USD -> BTC"""
        return ticker.split('-')[0] if '-' in ticker else ticker
    
    def to_hyper_resolution(self, resolution: str) -> str:
        """Convert DYDX resolution to HyperLiquid interval. 1HOUR -> 1h"""
        return self.RESOLUTION_MAP.get(resolution, "1h")


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """ML Model configuration"""
    # Meme model (binary classification)
    MEME_N_ESTIMATORS: int = 100
    MEME_MAX_DEPTH: int = 6
    MEME_LEARNING_RATE: float = 0.1
    MEME_N_FEATURES: int = 36
    
    # Perps model (binary classification: LONG vs NOT_LONG)
    PERPS_N_ESTIMATORS: int = 150
    PERPS_MAX_DEPTH: int = 5
    PERPS_LEARNING_RATE: float = 0.08
    PERPS_N_FEATURES: int = 56
    
    # Signal thresholds
    PERPS_LONG_THRESHOLD: float = 0.55  # P(LONG) >= 0.55 => BUY
    PERPS_SHORT_THRESHOLD: float = 0.35  # P(LONG) < 0.35 => SELL


# =============================================================================
# Analysis Engine Configuration
# =============================================================================

@dataclass
class PhaseThresholds:
    """Phase-based thresholds for whale identification"""
    # Phase 1: Launch/Snipers (Days 0-3)
    P1_START_DAY: int = 0
    P1_END_DAY: int = 3
    P1_NOISE_THRESHOLD: float = 0.00002  # 0.002%
    P1_WHALE_PERCENTILE: int = 99
    
    # Phase 2: Holder Expansion (Days 4-14)
    P2_START_DAY: int = 4
    P2_END_DAY: int = 14
    P2_NOISE_THRESHOLD: float = 0.00001  # 0.001%
    P2_WHALE_PERCENTILE: int = 99
    
    # Phase 3: Trend Stability (Days 15-45)
    P3_START_DAY: int = 15
    P3_END_DAY: int = 45
    P3_NOISE_THRESHOLD: float = 0.000005  # 0.0005%
    P3_WHALE_PERCENTILE: int = 99
    
    # Phase 4: Mature (Days 45+)
    P4_START_DAY: int = 46
    P4_END_DAY: int = 90
    P4_NOISE_THRESHOLD: float = 0.000001  # 0.0001%
    P4_WHALE_PERCENTILE: int = 98


class WhaleEngineConfig:
    """Whale analysis engine configuration"""
    # Holder thresholds
    TOP_HOLDER_COUNT: int = 10
    WHALE_THRESHOLD_PERCENT: float = 5.0
    WHALE_HOLDING_THRESHOLD_PCT: float = 5.0
    DOMINANT_WHALE_THRESHOLD: float = 10.0
    DOMINANT_WHALE_COUNT: int = 5
    DOMINANT_WHALE_MIN_HOLDING_PCT: float = 10.0
    DOMINANT_WHALE_INACTIVE_HOLDING_THRESHOLD: float = 5.0
    FEW_HOLDERS_THRESHOLD: int = 50
    
    # Activity thresholds
    INACTIVE_HOURS_THRESHOLD: int = 72
    STALE_DATA_HOURS: int = 24
    WHALE_DELTA_WINDOW_HOURS: int = 24
    DOMINANT_WHALE_INACTIVITY_HOURS: int = 72
    DOMINANT_WHALE_AGING_HOURS: int = 168
    STALE_OVERRIDE_TOP_HOLDER_ACTIVE_HOURS: int = 24  # If top holder active within this many hours, do not report stale
    
    # Penalty multipliers
    DOMINANT_WHALE_CONFIDENCE_PENALTY: float = 0.3
    DOMINANT_WHALE_AGING_PENALTY: float = 0.2
    DOMINANT_WHALE_SEVERE_PENALTY: float = 0.5
    
    # Gini coefficient thresholds
    GINI_HIGH_CONCENTRATION: float = 0.8
    GINI_MEDIUM_CONCENTRATION: float = 0.6
    GINI_HIGH_RISK: float = 0.85
    GINI_SAFE: float = 0.6
    GINI_EXTREME_THRESHOLD: float = 0.90
    GINI_NEUTRAL_DEFAULT: float = 0.5
    GINI_HIGH_MEAN_RATIO: float = 1.5
    GINI_HIGH_MEDIAN_RATIO: float = 2.0
    GINI_EXTREME_MIN_VALUE: float = 0.9
    GINI_EXTREME_MEAN_RATIO: float = 2.0
    GINI_EXTREME_MEDIAN_RATIO: float = 3.0
    
    # Concentration percentages
    CONCENTRATION_MEDIUM_HIGH_PCT: float = 65.0
    CONCENTRATION_HIGH_PCT: float = 75.0
    
    # State determination thresholds
    ACCUMULATION_THRESHOLD: float = 0.6
    DISTRIBUTION_THRESHOLD: float = 0.4
    
    # Statistical thresholds
    STATS_SMALL_SAMPLE_THRESHOLD: int = 10
    STATS_VERY_SMALL_SAMPLE: int = 5
    STATS_HIGH_VARIABILITY_CV: float = 1.5
    
    # Recency and weighting
    RECENCY_WEIGHT_DEFAULT: float = 0.7
    RECENCY_WEIGHT_MIN: float = 0.3
    
    # Supply ratio
    MISSING_CONTRACT_SUPPLY_RATIO: float = 0.95


@dataclass
class TechnicalEngineConfig:
    """Technical analysis engine configuration"""
    # RSI
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    
    # EMA
    EMA_SHORT: int = 20
    EMA_MEDIUM: int = 50
    EMA_LONG: int = 200
    
    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD: float = 2.0


@dataclass
class TimingConfig:
    """Entry/Exit timing configuration"""
    # RSI thresholds
    RSI_EXTREME_HIGH: float = 80.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_EXTREME_LOW: float = 20.0
    RSI_OVERSOLD: float = 30.0
    
    # Momentum thresholds
    MOMENTUM_STRONG_UP: float = 0.05
    MOMENTUM_MILD_UP: float = 0.02
    MOMENTUM_STRONG_DOWN: float = -0.05
    MOMENTUM_MILD_DOWN: float = -0.02
    # Short-window (15m) momentum for spike/dip rules
    MOMENTUM_SHORT_WINDOW_MINUTES: int = 15
    MOMENTUM_SPIKE_UP_THRESHOLD: float = 0.10   # 10% up in last 15m -> WAIT
    MOMENTUM_DIP_DOWN_THRESHOLD: float = -0.05  # 5% down in last 15m -> ENTER

    # Volume thresholds
    VOLUME_BUY_DOMINANT: float = 0.55
    VOLUME_SELL_DOMINANT: float = 0.45
    VOLUME_SURGE_THRESHOLD: float = 2.0
    
    # Liquidity thresholds
    LIQUIDITY_HIGH: float = 100000
    LIQUIDITY_LOW: float = 10000
    LIQUIDITY_MCAP_HEALTHY: float = 0.1
    LIQUIDITY_MCAP_THIN: float = 0.03
    
    # Smart money thresholds
    SMART_MONEY_BUY_DOMINANT: float = 0.55
    SMART_MONEY_SELL_DOMINANT: float = 0.45
    SMART_MONEY_ACCELERATION: float = 1.1
    SMART_MONEY_DECELERATION: float = 0.9
    
    # Whale thresholds
    WHALE_STRONG_ACCUMULATION: float = 0.65
    WHALE_STRONG_DISTRIBUTION: float = 0.35
    
    # Decision thresholds
    WAIT_THRESHOLD: float = -0.2
    URGENT_THRESHOLD: float = 0.3
    
    # Wait time configuration
    WAIT_SCORE_SHORT: float = -0.1
    WAIT_SCORE_MEDIUM: float = -0.3
    WAIT_TIME_SHORT: int = 5
    WAIT_TIME_MEDIUM: int = 15
    WAIT_TIME_LONG: int = 30
    
    # Exit timing thresholds
    EXIT_URGENT_THRESHOLD: float = -0.3
    EXIT_WAIT_THRESHOLD: float = 0.2
    EXIT_SCORE_MEDIUM: float = 0.0
    EXIT_SCORE_SHORT: float = -0.15
    EXIT_WAIT_TIME_SHORT: int = 5
    EXIT_WAIT_TIME_MEDIUM: int = 15
    EXIT_WAIT_TIME_LONG: int = 30
    
    # Safety-override exit threshold: when the main signal is SELL due to a safety
    # override (e.g., extreme concentration / rug pull risk), use this much lower
    # threshold instead of EXIT_WAIT_THRESHOLD. Any score above this triggers
    # EXIT_NOW, since the safety override already established urgency.
    EXIT_SAFETY_OVERRIDE_THRESHOLD: float = -0.1
    
    # Coerced-WAIT settings: when predict.py overrides an immediate recommendation
    # (e.g. HOLD+ENTER_NOW → WAIT), these control the dynamic wait estimate.
    COERCED_WAIT_MAX_MINUTES: int = 45
    COERCED_WAIT_VOLATILITY_BONUS: int = 10
    
    # Default weights (graduated tokens - OFF bonding curve)
    WEIGHT_MOMENTUM: float = 0.25
    WEIGHT_VOLUME: float = 0.20
    WEIGHT_LIQUIDITY: float = 0.15
    WEIGHT_SMART_MONEY: float = 0.15
    WEIGHT_WHALE: float = 0.15
    
    # Bonding curve weights (tokens ON bonding curve)
    BC_WEIGHT_MOMENTUM: float = 0.25
    BC_WEIGHT_VOLUME: float = 0.20
    BC_WEIGHT_LIQUIDITY: float = 0.15
    BC_WEIGHT_SMART_MONEY: float = 0.10
    BC_WEIGHT_WHALE: float = 0.20
    
    def get_weights(self, is_on_bonding_curve: bool = False) -> Dict[str, float]:
        """Get appropriate weights based on bonding curve status"""
        if is_on_bonding_curve:
            weights = {
                'momentum': self.BC_WEIGHT_MOMENTUM,
                'volume': self.BC_WEIGHT_VOLUME,
                'liquidity': self.BC_WEIGHT_LIQUIDITY,
                'smart_money': self.BC_WEIGHT_SMART_MONEY,
                'whale': self.BC_WEIGHT_WHALE
            }
        else:
            weights = {
                'momentum': self.WEIGHT_MOMENTUM,
                'volume': self.WEIGHT_VOLUME,
                'liquidity': self.WEIGHT_LIQUIDITY,
                'smart_money': self.WEIGHT_SMART_MONEY,
                'whale': self.WEIGHT_WHALE
            }
        
        # Normalize weights
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


# =============================================================================
# Layer Aggregation Configuration
# =============================================================================

@dataclass
class LayerConfig:
    """Multi-layer signal aggregation configuration"""
    # Base weights for layers
    WEIGHT_ML_MODEL: float = float(os.getenv("LAYER_WEIGHT_ML", "0.30"))
    WEIGHT_WHALE_ENGINE: float = float(os.getenv("LAYER_WEIGHT_WHALE", "0.25"))
    WEIGHT_TECHNICAL: float = float(os.getenv("LAYER_WEIGHT_TECHNICAL", "0.10"))
    WEIGHT_HOLDER_METRICS: float = float(os.getenv("LAYER_WEIGHT_HOLDER", "0.15"))
    WEIGHT_USER_PROFILE: float = float(os.getenv("LAYER_WEIGHT_USER", "0.20"))
    
    # Thresholds
    BUY_THRESHOLD: float = 0.15
    SELL_THRESHOLD: float = -0.15
    STRONG_CONFIDENCE_THRESHOLD: float = 0.75
    
    def get_base_weights(self) -> Dict[str, float]:
        """Get base layer weights"""
        return {
            'ml_model': self.WEIGHT_ML_MODEL,
            'whale_engine': self.WEIGHT_WHALE_ENGINE,
            'technical': self.WEIGHT_TECHNICAL,
            'holder_metrics': self.WEIGHT_HOLDER_METRICS,
            'user_profile': self.WEIGHT_USER_PROFILE
        }


# =============================================================================
# Safety Configuration
# =============================================================================

@dataclass
class SafetyConfig:
    """Safety override configuration"""
    # Concentration thresholds
    EXTREME_CONCENTRATION_THRESHOLD: float = 80.0
    HIGH_CONCENTRATION_THRESHOLD: float = 60.0
    SAFETY_EXCELLENT_CONCENTRATION_PCT: float = 30.0
    SAFETY_GOOD_CONCENTRATION_PCT: float = 45.0
    SAFETY_MEDIUM_CONCENTRATION_PCT: float = 60.0
    SAFETY_HIGH_CONCENTRATION_PCT: float = 75.0
    SAFETY_EXTREME_CONCENTRATION_PCT: float = 85.0
    TOP10_CONCENTRATION_MEDIUM_RISK: float = 0.60
    TOP10_CONCENTRATION_HIGH_RISK: float = 0.75
    TOP10_CONCENTRATION_CRITICAL: float = 0.85
    
    # Gini thresholds
    EXTREME_GINI_THRESHOLD: float = 0.95
    HIGH_GINI_THRESHOLD: float = 0.85
    SAFETY_SUSPICIOUS_GINI_TOLERANCE: float = 0.05
    
    # Holder count thresholds
    MIN_HOLDER_COUNT: int = 10
    CRITICAL_HOLDER_COUNT: int = 5
    SAFETY_EXTREMELY_FEW_HOLDERS: int = 10
    SAFETY_FEW_HOLDERS: int = 50
    SAFETY_MEDIUM_HOLDERS: int = 200
    SAFETY_SUSPICIOUS_CONCENTRATION_HOLDERS: int = 20
    SAFETY_SUSPICIOUS_GINI_HOLDERS: int = 20
    DEV_HOLDING_HIGH_RISK: float = 0.20
    
    # Confidence multipliers
    SAFETY_CONFIDENCE_MIN: float = 0.3
    SAFETY_CONFIDENCE_MEDIUM_MULTIPLIER: float = 0.6
    SAFETY_CONFIDENCE_HIGH_MULTIPLIER: float = 0.8
    SAFETY_CONFIDENCE_EXTREME_MULTIPLIER: float = 0.5
    
    # Whale persistence
    WHALE_PERSISTENCE_HOURS: int = int(os.getenv("WHALE_PERSISTENCE_HOURS", "48"))
    
    # Sniper detection
    SNIPER_WINDOW_SECONDS: int = int(os.getenv("SNIPER_WINDOW_SECONDS", "60"))

    # --- Volatility safety override (token-type-specific) ---
    EXTREME_VOLATILITY_THRESHOLD_MEME: float = 0.20
    EXTREME_VOLATILITY_THRESHOLD_PERPS: float = 0.10
    EXTREME_PRICE_SWING_PCT_MEME: float = 50.0
    EXTREME_PRICE_SWING_PCT_PERPS: float = 25.0
    VOLATILITY_CONFIDENCE_PENALTY: float = 0.30

    # --- Low liquidity safety override (meme only) ---
    MIN_LIQUIDITY_USD: float = 5000.0
    LOW_LIQUIDITY_WARNING_USD: float = 25000.0

    # --- Token age safety override (meme only) ---
    TOKEN_AGE_P1_CONFIDENCE_PENALTY: float = 0.35
    TOKEN_AGE_P1_MAX_HOLDERS_FOR_HOLD: int = 20

    # --- Extreme funding rate override (perps only) ---
    EXTREME_FUNDING_THRESHOLD: float = 0.0005
    FUNDING_RATE_CONFIDENCE_PENALTY: float = 0.25

    # --- Signal-timing contradiction ---
    SIGNAL_TIMING_CONTRADICTION_THRESHOLD: float = -0.2
    SIGNAL_TIMING_CONTRADICTION_PENALTY: float = 0.20

    # --- Leverage / liquidation risk (perps only) ---
    LIQUIDATION_CRITICAL_DISTANCE_PCT: float = 5.0
    HIGH_LEVERAGE_THRESHOLD: int = 10

    # --- Minimum stop-loss distance (prevents 0% stop distance edge case) ---
    MIN_STOP_DISTANCE_PCT: float = 1.5

    # --- Minimum risk/reward ratio (suppress meaningless risk management) ---
    MIN_RISK_REWARD_RATIO: float = 0.5

    # --- Confidence tapering near thresholds ---
    THRESHOLD_TAPER_ZONE: float = 0.10
    TAPER_MIN_FACTOR: float = 0.60


# =============================================================================
# OpenAI Configuration
# =============================================================================

@dataclass
class OpenAIConfig:
    """OpenAI service configuration"""
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
    TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    TIMEOUT: int = int(os.getenv("OPENAI_TIMEOUT", "30"))
    MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
    
    def is_configured(self) -> bool:
        """Check if OpenAI API is properly configured"""
        return bool(self.API_KEY)


@dataclass
class LLMAnalyzerConfig:
    """Configuration for AI-powered profile analysis (Anthropic primary, OpenAI fallback)"""
    PROVIDER: str = os.getenv("LLM_ANALYZER_PROVIDER", "anthropic")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
    ANTHROPIC_MAX_TOKENS: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
    ANTHROPIC_TIMEOUT: int = int(os.getenv("ANTHROPIC_TIMEOUT", "60"))
    OPENAI_FALLBACK_MODEL: str = os.getenv("LLM_ANALYZER_OPENAI_MODEL", "gpt-4o")
    OPENAI_FALLBACK_MAX_TOKENS: int = int(os.getenv("LLM_ANALYZER_OPENAI_MAX_TOKENS", "4096"))
    MAX_RETRIES: int = int(os.getenv("LLM_ANALYZER_MAX_RETRIES", "1"))
    ENABLED: bool = os.getenv("LLM_ANALYZER_ENABLED", "true").lower() == "true"

    def is_anthropic_configured(self) -> bool:
        return bool(self.ANTHROPIC_API_KEY)

    def is_openai_configured(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY", ""))


class BirdeyeConfig:
    """Birdeye API configuration"""
    API_KEY: str = os.getenv("BIRDEYE_API_KEY", "")
    BASE_URL: str = os.getenv("BIRDEYE_BASE_URL", "https://public-api.birdeye.so")
    CHAIN: str = os.getenv("BIRDEYE_CHAIN", "solana")
    TIMEOUT: int = int(os.getenv("BIRDEYE_TIMEOUT", "5"))
    MAX_RETRIES: int = int(os.getenv("BIRDEYE_MAX_RETRIES", "2"))
    TOKEN_METADATA_ENDPOINT: str = os.getenv("BIRDEYE_TOKEN_METADATA_ENDPOINT", "/defi/token_overview")
    
    @staticmethod
    def is_configured() -> bool:
        """Check if Birdeye API is properly configured"""
        return bool(os.getenv("BIRDEYE_API_KEY", ""))


class BondingCurveConfig:
    """Bonding curve detection configuration"""
    SUPPORTED_CATEGORIES: List[str] = os.getenv(
        "BONDING_SUPPORTED_CATEGORIES", 
        "letsbonk-fun-ecosystem,pump-fun,moonshot-ecosystem"
    ).split(",")
    PUMPFUN_ADDRESS_SUFFIX: str = os.getenv("BONDING_PUMPFUN_SUFFIX", "pump")
    LIQUIDITY_THRESHOLD: float = float(os.getenv("BONDING_LIQUIDITY_THRESHOLD", "5000"))
    MARKET_CAP_THRESHOLD: float = float(os.getenv("BONDING_MARKET_CAP_THRESHOLD", "50000"))
    GRADUATED_LIQUIDITY_MIN: float = float(os.getenv("BONDING_GRADUATED_LIQUIDITY", "10000"))
    
    @staticmethod
    def is_supported_ecosystem_token(category_id: Optional[str] = None, token_address: str = "") -> bool:
        """Check if token belongs to supported ecosystems"""
        config = BondingCurveConfig()
        if category_id and category_id in config.SUPPORTED_CATEGORIES:
            return True
        if token_address and token_address.endswith(config.PUMPFUN_ADDRESS_SUFFIX):
            return True
        return False
    
    @staticmethod
    def is_pumpfun_token(token_address: str) -> bool:
        """Check if token is from pump.fun"""
        return token_address.endswith(BondingCurveConfig.PUMPFUN_ADDRESS_SUFFIX)
    
    @staticmethod
    def is_on_bonding_curve(token_address: str, liquidity: float = 0, market_cap: float = 0) -> bool:
        """Check if token is still on bonding curve"""
        config = BondingCurveConfig()
        if config.is_pumpfun_token(token_address):
            if liquidity > 0 and liquidity < config.LIQUIDITY_THRESHOLD:
                return True
            if market_cap > 0 and market_cap < config.MARKET_CAP_THRESHOLD:
                return True
        return False


@dataclass
class PostTradeReviewConfig:
    """
    Configuration for Post-Trade AI Review (Feature 4)
    Analyzes completed trades to identify mistakes and provide coaching
    """
    # Feature toggle
    ENABLED: bool = field(default_factory=lambda: os.getenv("POST_TRADE_REVIEW_ENABLED", "true").lower() == "true")
    OPENAI_ENABLED: bool = field(default_factory=lambda: os.getenv("POST_TRADE_OPENAI_ENABLED", "false").lower() == "true")
    MIN_HOLD_TIME_MINUTES: int = field(default_factory=lambda: int(os.getenv("POST_TRADE_MIN_HOLD_MINUTES", "5")))
    
    # Time windows for outcome analysis (in minutes)
    OUTCOME_WINDOW_1H: int = field(default_factory=lambda: int(os.getenv("PTR_OUTCOME_1H", "60")))
    OUTCOME_WINDOW_4H: int = field(default_factory=lambda: int(os.getenv("PTR_OUTCOME_4H", "240")))
    OUTCOME_WINDOW_24H: int = field(default_factory=lambda: int(os.getenv("PTR_OUTCOME_24H", "1440")))
    
    # Early exit detection thresholds
    EARLY_EXIT_GAIN_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_EARLY_EXIT_GAIN", "0.05")))  # 5% more gain = early exit
    EARLY_EXIT_WINDOW_HOURS: int = field(default_factory=lambda: int(os.getenv("PTR_EARLY_EXIT_WINDOW", "24")))
    EARLY_EXIT_SEVERITY_HIGH: float = field(default_factory=lambda: float(os.getenv("PTR_EARLY_EXIT_HIGH", "0.15")))  # >15% = HIGH
    EARLY_EXIT_SEVERITY_MEDIUM: float = field(default_factory=lambda: float(os.getenv("PTR_EARLY_EXIT_MEDIUM", "0.08")))  # >8% = MEDIUM
    
    # Late entry detection thresholds
    LATE_ENTRY_DIP_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_LATE_ENTRY_DIP", "0.03")))  # 3% dip after entry
    LATE_ENTRY_WINDOW_HOURS: int = field(default_factory=lambda: int(os.getenv("PTR_LATE_ENTRY_WINDOW", "4")))
    LATE_ENTRY_BETTER_PRICE_PCT: float = field(default_factory=lambda: float(os.getenv("PTR_LATE_ENTRY_BETTER", "0.05")))  # 5% better available
    
    # Over-trading detection
    OVERTRADE_WINDOW_HOURS: int = field(default_factory=lambda: int(os.getenv("PTR_OVERTRADE_WINDOW", "24")))
    OVERTRADE_MIN_TRADES: int = field(default_factory=lambda: int(os.getenv("PTR_OVERTRADE_MIN", "3")))
    
    # Revenge trade detection
    REVENGE_TRADE_WINDOW_HOURS: int = field(default_factory=lambda: int(os.getenv("PTR_REVENGE_WINDOW", "4")))
    REVENGE_LOSS_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_REVENGE_LOSS", "-0.10")))  # Previous loss > 10%
    
    # Bad Risk-Reward detection
    BAD_RR_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_BAD_RR", "0.5")))  # RR < 0.5 is bad
    
    # FOMO entry detection
    FOMO_PUMP_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_FOMO_PUMP", "0.20")))  # 20% pump before entry
    FOMO_LOOKBACK_HOURS: int = field(default_factory=lambda: int(os.getenv("PTR_FOMO_LOOKBACK", "4")))
    FOMO_SEVERITY_HIGH: float = field(default_factory=lambda: float(os.getenv("PTR_FOMO_HIGH", "0.50")))  # >50% pump = HIGH
    FOMO_SEVERITY_MEDIUM: float = field(default_factory=lambda: float(os.getenv("PTR_FOMO_MEDIUM", "0.30")))  # >30% pump = MEDIUM
    
    # Panic sell detection
    PANIC_SELL_LOSS_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_PANIC_LOSS", "-0.15")))  # -15% loss
    PANIC_SELL_MAX_HOLD_HOURS: float = field(default_factory=lambda: float(os.getenv("PTR_PANIC_HOLD", "24.0")))
    PANIC_SELL_SEVERITY_HIGH: float = field(default_factory=lambda: float(os.getenv("PTR_PANIC_HIGH", "0.30")))  # >30% loss = HIGH
    PANIC_SELL_SEVERITY_MEDIUM: float = field(default_factory=lambda: float(os.getenv("PTR_PANIC_MEDIUM", "0.20")))  # >20% loss = MEDIUM
    
    # Revenge trade severity
    REVENGE_SEVERITY_HIGH_LOSS: float = field(default_factory=lambda: float(os.getenv("PTR_REVENGE_HIGH_LOSS", "0.20")))  # >20% loss = HIGH
    
    # Over-trading severity
    OVERTRADE_SEVERITY_HIGH: int = field(default_factory=lambda: int(os.getenv("PTR_OVERTRADE_HIGH", "5")))  # >=5 trades = HIGH
    
    # Bad Risk-Reward detection
    BAD_RR_MULTIPLIER: float = field(default_factory=lambda: float(os.getenv("PTR_BAD_RR_MULT", "2.0")))  # 2x worse than avg = flag
    
    # Monthly impact extrapolation
    TRADES_PER_MONTH_ESTIMATE: int = field(default_factory=lambda: int(os.getenv("PTR_TRADES_PER_MONTH", "30")))
    MISTAKE_FREQUENCY_ESTIMATE: float = field(default_factory=lambda: float(os.getenv("PTR_MISTAKE_FREQ", "0.30")))  # Assume 30% of trades have this pattern
    
    # Confidence calculation parameters
    CONFIDENCE_BASE: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_BASE", "0.5")))
    CONFIDENCE_PRICE_AFTER_MANY: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_AFTER_MANY", "0.2")))  # >10 candles after
    CONFIDENCE_PRICE_AFTER_SOME: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_AFTER_SOME", "0.1")))  # any candles after
    CONFIDENCE_PRICE_BEFORE_MANY: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_BEFORE_MANY", "0.15")))  # >10 candles before
    CONFIDENCE_PRICE_BEFORE_SOME: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_BEFORE_SOME", "0.08")))  # any candles before
    CONFIDENCE_USER_TRADES_MANY: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_USER_MANY", "0.15")))  # >50 user trades
    CONFIDENCE_USER_TRADES_SOME: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_USER_SOME", "0.08")))  # >10 user trades
    CONFIDENCE_USER_TRADES_MANY_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("PTR_USER_TRADES_MANY", "50")))
    CONFIDENCE_USER_TRADES_SOME_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("PTR_USER_TRADES_SOME", "10")))
    CONFIDENCE_PRICE_CANDLES_MANY: int = field(default_factory=lambda: int(os.getenv("PTR_CANDLES_MANY", "10")))  # threshold for "many" candles
    CONFIDENCE_MAX: float = field(default_factory=lambda: float(os.getenv("PTR_CONF_MAX", "0.95")))
    
    # Price matching tolerance (minutes)
    PRICE_MATCH_TOLERANCE_MINUTES: int = field(default_factory=lambda: int(os.getenv("PTR_PRICE_MATCH_TOL", "30")))
    
    # OpenAI settings for coaching
    REVIEW_MAX_TOKENS: int = field(default_factory=lambda: int(os.getenv("PTR_MAX_TOKENS", "400")))
    REVIEW_TEMPERATURE: float = field(default_factory=lambda: float(os.getenv("PTR_TEMPERATURE", "0.5")))
    COACHING_ENABLED: bool = field(default_factory=lambda: os.getenv("PTR_COACHING_ENABLED", "true").lower() == "true")
    
    # Birdeye OHLCV settings
    OHLCV_INTERVAL: str = field(default_factory=lambda: os.getenv("PTR_OHLCV_INTERVAL", "15m"))  # 15 minute candles
    OHLCV_LOOKBACK_HOURS: int = field(default_factory=lambda: int(os.getenv("PTR_OHLCV_LOOKBACK", "48")))  # 48 hours of data
    
    # Severity thresholds (percentage impact)
    SEVERITY_HIGH_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_SEVERITY_HIGH", "0.10")))  # >10% = HIGH
    SEVERITY_MEDIUM_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PTR_SEVERITY_MEDIUM", "0.05")))  # >5% = MEDIUM


class UserProfileConfig:
    """User trading pattern analysis configuration"""
    # Minimum data requirements
    MIN_TRADES_FOR_PROFILE: int = 5
    MIN_TRADES_FOR_PATTERN: int = 3
    MIN_TRADES_FOR_ANALYSIS: int = int(os.getenv('USER_MIN_TRADES_FOR_ANALYSIS', '5'))
    # When total_trades >= this and all four tendency fields are 0, mark them as "not computed" (null)
    MIN_TRADES_FOR_BIAS_INFERENCE: int = int(os.getenv('MIN_TRADES_FOR_BIAS_INFERENCE', '100'))
    # Perps: trades/day above this triggers overtrading message and recommendation
    PERPS_OVERTRADING_TRADES_PER_DAY: int = int(os.getenv('PERPS_OVERTRADING_TRADES_PER_DAY', '100'))
    
    # Behaviour stats: trade cap and confidence bands (180d window, cap 500 trades)
    BEHAVIOUR_LOW_CONFIDENCE_MAX_TRADES: int = int(os.getenv('BEHAVIOUR_LOW_CONFIDENCE_MAX_TRADES', '50'))
    BEHAVIOUR_SWEET_SPOT_MIN: int = 51
    BEHAVIOUR_SWEET_SPOT_MAX: int = 300
    BEHAVIOUR_DIMINISHING_RETURNS_MIN: int = 400
    
    # Pattern matching
    PATTERN_SIMILARITY_THRESHOLD: float = 0.6
    
    # Rating thresholds
    WIN_RATE_GREEN_THRESHOLD: float = float(os.getenv('USER_GREEN_THRESHOLD', '0.60'))
    WIN_RATE_RED_THRESHOLD: float = float(os.getenv('USER_RED_THRESHOLD', '0.40'))
    GOOD_WIN_RATE: float = float(os.getenv('USER_GOOD_WIN_RATE', '0.65'))
    BAD_WIN_RATE: float = float(os.getenv('USER_BAD_WIN_RATE', '0.35'))
    
    # Behavioral detection thresholds
    FOMO_DETECTION_THRESHOLD: float = 0.7
    PANIC_SELL_THRESHOLD: float = 0.7
    FOMO_PUMP_THRESHOLD: float = float(os.getenv('USER_FOMO_PUMP_THRESHOLD', '0.20'))
    DIP_DROP_THRESHOLD: float = float(os.getenv('USER_DIP_DROP_THRESHOLD', '0.10'))
    PRICE_LOOKBACK_HOURS: int = int(os.getenv('USER_LOOKBACK_HOURS', '4'))
    
    # Trader type classification
    FLIPPER_MAX_HOLD_HOURS: float = 24.0
    HOLDER_MIN_HOLD_HOURS: float = 168.0
    SNIPER_ENTRY_SECONDS: int = 60
    SNIPER_THRESHOLD_PERCENT: float = 0.30
    
    # Confidence calculation
    CONFIDENCE_MIN: float = float(os.getenv('USER_CONFIDENCE_MIN', '0.1'))
    CONFIDENCE_MAX: float = float(os.getenv('USER_CONFIDENCE_MAX', '1.0'))
    GREEN_CONFIDENCE_BOOST: float = 0.1
    RED_CONFIDENCE_PENALTY: float = 0.15
    
    # PnL calculation
    PNL_WIN_THRESHOLD: float = 0.0  # Profit > 0 is a win
    
    # PANIC_SELL detection thresholds
    PANIC_SELL_LOSS_THRESHOLD: float = float(os.getenv('USER_PANIC_SELL_LOSS_THRESHOLD', '-20.0'))
    PANIC_SELL_MAX_HOLD_HOURS: float = float(os.getenv('USER_PANIC_SELL_MAX_HOLD_HOURS', '24.0'))
    
    # Confidence calculation weights
    CONFIDENCE_TRADE_WEIGHT: float = float(os.getenv('USER_CONFIDENCE_TRADE_WEIGHT', '0.5'))
    CONFIDENCE_MINT_WEIGHT: float = float(os.getenv('USER_CONFIDENCE_MINT_WEIGHT', '0.3'))
    CONFIDENCE_PATTERN_WEIGHT: float = float(os.getenv('USER_CONFIDENCE_PATTERN_WEIGHT', '0.2'))
    CONFIDENCE_MAX_PATTERNS: int = int(os.getenv('USER_CONFIDENCE_MAX_PATTERNS', '5'))
    
    # Best/Worst pattern selection threshold
    PATTERN_WIN_RATE_THRESHOLD: float = float(os.getenv('USER_PATTERN_WIN_RATE_THRESHOLD', '0.5'))
    
    # Token age defaults (when creation date not available from API)
    DEFAULT_TOKEN_AGE_DAYS: float = float(os.getenv('USER_DEFAULT_TOKEN_AGE_DAYS', '60.0'))
    DEFAULT_TOKEN_CREATION_DAYS_BEFORE_ENTRY: float = float(os.getenv('USER_DEFAULT_TOKEN_CREATION_DAYS_BEFORE_ENTRY', '3.0'))
    
    # Context matching weights (for pattern similarity in risk_assessor)
    PHASE_MATCH_WEIGHT: float = 0.30
    CONCENTRATION_MATCH_WEIGHT: float = 0.25
    WHALE_STATE_MATCH_WEIGHT: float = 0.20
    TREND_MATCH_WEIGHT: float = 0.15
    RSI_MATCH_WEIGHT: float = 0.10
    
    # Messages
    MESSAGE_NEW_USER: str = "Insufficient trading history for profile analysis"
    MESSAGE_FOMO_WARNING: str = "Warning: Your FOMO pattern has {loss_rate:.1%} loss rate"
    MESSAGE_GREEN: str = "This aligns with your most profitable entry conditions ({win_rate:.0%} win rate over {occurrences} trades)."
    MESSAGE_YELLOW: str = "This setup is neutral based on your history. Consider position sizing carefully."
    MESSAGE_RED: str = "This trade matches your lowest-performing pattern. {losses} of your last {total} similar trades were losses."
    MESSAGE_NO_PATTERN: str = "No matching historical pattern found. Proceed with standard risk management."
    MESSAGE_DIP_BUY_SUCCESS: str = "DIP_BUY pattern detected - your historical win rate: {win_rate:.0%}"


class CacheConfig:
    """Cache configuration"""
    ENABLED: bool = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))


class SignalConfig:
    """Signal generation configuration"""
    # Target definition
    TARGET_PRICE_INCREASE: float = 0.15  # 15% price increase
    TARGET_TIME_WINDOW_HOURS: int = 4
    
    # XGBoost confidence thresholds
    XGBOOST_LONG_THRESHOLD: float = 0.70
    XGBOOST_SHORT_THRESHOLD: float = 0.65
    XGBOOST_HIGH_RISK_FALLBACK: float = 0.85
    
    # Bonding curve signals
    BONDING_CURVE_BREAKOUT: float = 0.80
    BONDING_CURVE_EARLY: float = 0.50
    
    # Liquidity risk
    LIQUIDITY_REMOVAL_CRITICAL: bool = True


class PerpsModelConfig:
    """Perps ML Model configuration"""
    # Model filenames
    MODEL_FILENAME: str = os.getenv("PERPS_MODEL_FILENAME", "perps_model.pkl")
    SCALER_FILENAME: str = os.getenv("PERPS_SCALER_FILENAME", "perps_scaler.pkl")
    FEATURES_FILENAME: str = os.getenv("PERPS_FEATURES_FILENAME", "perps_features.pkl")
    TICKER_ENCODER_FILENAME: str = os.getenv("PERPS_TICKER_ENCODER_FILENAME", "perps_ticker_encoder.pkl")
    
    # Feature lists
    CORE_FEATURES: list = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'log_returns', 'volatility'
    ]
    PERPS_FEATURES: list = [
        'funding_rate', 'funding_premium', 'open_interest',
        'long_short_ratio', 'liquidation_volume'
    ]
    
    @classmethod
    def get_model_path(cls) -> str:
        """Get full path to perps model file"""
        return str(MODELS_DIR / "perps" / cls.MODEL_FILENAME)
    
    @classmethod
    def get_scaler_path(cls) -> str:
        """Get full path to perps scaler file"""
        return str(MODELS_DIR / "perps" / cls.SCALER_FILENAME)
    
    @classmethod
    def get_features_path(cls) -> str:
        """Get full path to perps features file"""
        return str(MODELS_DIR / "perps" / cls.FEATURES_FILENAME)
    
    @classmethod
    def get_ticker_encoder_path(cls) -> str:
        """Get full path to perps ticker encoder file"""
        return str(MODELS_DIR / "perps" / cls.TICKER_ENCODER_FILENAME)


class LayerWeightConfig:
    """Multi-layer signal aggregation configuration"""
    # Base layer weights (should sum to 1.0)
    WEIGHT_ML_MODEL: float = 0.30
    WEIGHT_WHALE_ENGINE: float = 0.25
    WEIGHT_TECHNICAL: float = 0.10
    WEIGHT_HOLDER_METRICS: float = 0.15
    WEIGHT_USER_PROFILE: float = 0.20
    
    # Confidence-based adjustment
    MIN_CONFIDENCE_MULTIPLIER: float = 0.5
    MAX_CONFIDENCE_MULTIPLIER: float = 1.0
    
    # Signal conversion thresholds
    SIGNAL_BUY_THRESHOLD: float = 0.30
    SIGNAL_SELL_THRESHOLD: float = -0.30
    CONFIDENCE_BASE: float = 0.35
    
    # Agreement bonus/penalty
    HIGH_AGREEMENT_THRESHOLD: float = 0.80
    LOW_AGREEMENT_THRESHOLD: float = 0.50
    AGREEMENT_CONFIDENCE_BONUS: float = 0.10
    DISAGREEMENT_CONFIDENCE_PENALTY: float = 0.15

    # Coverage penalty: penalize confidence when fewer than this ratio of layers are valid
    MIN_COVERAGE_RATIO: float = 0.60
    
    # Layer names (for reference)
    LAYER_NAMES: Dict[str, str] = {
        'ml_model': 'ML Model (XGBoost)',
        'whale_engine': 'Whale Behavior',
        'technical': 'Technical Indicators',
        'holder_metrics': 'Holder Metrics',
        'user_profile': 'User Profile'
    }


# =============================================================================
# Main Configuration Class
# =============================================================================

ALGO_VERSION: str = "2.1.0"
PROMPT_VERSION: str = "1.0.0"

_REQUIRED_ENV_VARS = {
    "INTERNAL_API_BASE_URL": "Internal API base URL",
    "BIRDEYE_API_KEY": "Birdeye API key",
    "OPENAI_API_KEY": "OpenAI API key",
}


@dataclass
class Config:
    """Main configuration container"""
    api: APIConfig = field(default_factory=APIConfig)
    meme: MemeConfig = field(default_factory=MemeConfig)
    perps: PerpsConfig = field(default_factory=PerpsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    whale: WhaleEngineConfig = field(default_factory=WhaleEngineConfig)
    technical: TechnicalEngineConfig = field(default_factory=TechnicalEngineConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    layer: LayerConfig = field(default_factory=LayerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    
    def get_config_for_token_type(self, token_type: TokenType):
        """Get token-type specific configuration"""
        if token_type == TokenType.PERPS:
            return self.perps
        return self.meme

    def validate(self) -> None:
        """Refuse to start if critical env vars are missing."""
        missing = [
            desc for env_var, desc in _REQUIRED_ENV_VARS.items()
            if not os.getenv(env_var)
        ]
        if missing:
            raise SystemExit(
                f"FATAL: Missing required config: {', '.join(missing)}. "
                "Set these environment variables and restart."
            )


# Global instances (for compatibility with existing code)
api_config = APIConfig()
meme_config = MemeConfig()
perps_config = PerpsConfig()
model_config = ModelConfig()
whale_config = WhaleEngineConfig()
technical_config = TechnicalEngineConfig()
timing_config = TimingConfig()
layer_config = LayerConfig()
safety_config = SafetyConfig()
openai_config = OpenAIConfig()
llm_analyzer_config = LLMAnalyzerConfig()
birdeye_config = BirdeyeConfig()
bonding_curve_config = BondingCurveConfig()
post_trade_review_config = PostTradeReviewConfig()
user_profile_config = UserProfileConfig()
cache_config = CacheConfig()
signal_config = SignalConfig()
perps_model_config = PerpsModelConfig()
layer_weight_config = LayerWeightConfig()
phase_thresholds = PhaseThresholds()

# Aliases for compatibility
perps_api_config = perps_config
holder_config = layer_config  # holder settings are in layer_config
whale_engine_config = whale_config
wallet_classification_config = safety_config  # wallet classification is part of safety
token_type_config = meme_config  # token type defaults in meme_config

_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


# =============================================================================
# Helper Functions
# =============================================================================

def get_phase_from_age(days_since_launch: float) -> str:
    """
    Determine the phase of a token based on its age
    
    Args:
        days_since_launch: Days since token launch
        
    Returns:
        Phase identifier (P1, P2, P3, P4)
    """
    if days_since_launch <= phase_thresholds.P1_END_DAY:
        return "P1"
    elif days_since_launch <= phase_thresholds.P2_END_DAY:
        return "P2"
    elif days_since_launch <= phase_thresholds.P3_END_DAY:
        return "P3"
    else:
        return "P4"


def get_noise_threshold(phase: str) -> float:
    """Get the noise threshold for a given phase"""
    thresholds = {
        "P1": phase_thresholds.P1_NOISE_THRESHOLD,
        "P2": phase_thresholds.P2_NOISE_THRESHOLD,
        "P3": phase_thresholds.P3_NOISE_THRESHOLD,
        "P4": phase_thresholds.P4_NOISE_THRESHOLD
    }
    return thresholds.get(phase, phase_thresholds.P4_NOISE_THRESHOLD)


def get_whale_percentile(phase: str) -> int:
    """Get the whale percentile threshold for a given phase"""
    percentiles = {
        "P1": phase_thresholds.P1_WHALE_PERCENTILE,
        "P2": phase_thresholds.P2_WHALE_PERCENTILE,
        "P3": phase_thresholds.P3_WHALE_PERCENTILE,
        "P4": phase_thresholds.P4_WHALE_PERCENTILE
    }
    return percentiles.get(phase, phase_thresholds.P4_WHALE_PERCENTILE)
