"""
Token Type Router
Centralized routing logic for token type based API and model selection

This module provides a clean abstraction for routing requests based on token type
(meme vs perps), handling API endpoint selection, feature building, and ML model routing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TokenType, token_type_config, api_config, meme_config, perps_api_config,
    perps_model_config, model_config, MODELS_DIR
)

logger = logging.getLogger(__name__)


# ==================== DATA CLASSES ====================

@dataclass
class APIEndpoints:
    """Container for API endpoints based on token type"""
    base_url: str
    candles: str
    holders: str
    trades: str
    funding: Optional[str] = None  # Perps only
    markets: Optional[str] = None  # Perps only
    user_holdings: Optional[str] = None  # Meme only
    mint_metadata: Optional[str] = None  # Meme only


@dataclass
class ModelPaths:
    """Container for ML model file paths"""
    model_path: str
    scaler_path: Optional[str] = None
    features_path: Optional[str] = None
    ticker_encoder_path: Optional[str] = None
    
    def exists(self) -> bool:
        """Check if model file exists"""
        return os.path.exists(self.model_path)


# ==================== ABSTRACT BASE CLASS ====================

class BaseTokenTypeStrategy(ABC):
    """
    Abstract base class for token type strategies
    Implements Strategy Pattern for clean separation of meme vs perps logic
    """
    
    @abstractmethod
    def get_endpoints(self) -> APIEndpoints:
        """Get API endpoints for this token type"""
        pass
    
    @abstractmethod
    def get_model_paths(self) -> ModelPaths:
        """Get ML model paths for this token type"""
        pass
    
    @abstractmethod
    def get_feature_list(self) -> List[str]:
        """Get list of features used by this token type's model"""
        pass
    
    @abstractmethod
    def build_feature_vector(
        self,
        technical_signals: Any,
        whale_metrics: Any,
        holder_stats: Any,
        candle_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> np.ndarray:
        """Build feature vector for ML model"""
        pass
    
    @abstractmethod
    def map_prediction_to_signal(self, prediction: Any, probabilities: np.ndarray) -> Tuple[str, float]:
        """Map model prediction to trading signal and confidence"""
        pass


# ==================== MEME TOKEN STRATEGY ====================

class MemeTokenStrategy(BaseTokenTypeStrategy):
    """
    Strategy for meme coin token type
    Uses internal API endpoints and XGBoost binary model
    """
    
    def __init__(self):
        self.token_type = TokenType.MEME
        logger.info("MemeTokenStrategy initialized")
    
    def get_endpoints(self) -> APIEndpoints:
        """Get meme coin API endpoints"""
        return APIEndpoints(
            base_url=meme_config.BASE_URL,
            candles=meme_config.ENDPOINT_CANDLES,
            holders=meme_config.ENDPOINT_HOLDERS,
            trades="/trade/trades",  # Trade endpoint from API
            user_holdings=meme_config.ENDPOINT_USER_HOLDINGS,
            mint_metadata=meme_config.ENDPOINT_METADATA
        )
    
    def get_model_paths(self) -> ModelPaths:
        """Get meme coin ML model paths"""
        return ModelPaths(
            model_path=os.path.join(MODELS_DIR, model_config.MODEL_FILENAME),
            scaler_path=os.path.join(MODELS_DIR, model_config.SCALER_FILENAME),
            features_path=os.path.join(MODELS_DIR, model_config.FEATURE_COLS_FILENAME)
        )
    
    def get_feature_list(self) -> List[str]:
        """Get meme coin feature list (36 features)"""
        return [
            'BuyCount', 'SellCount', 'TotalSupply', 'SolUSDPrice', 'returns',
            'log_returns', 'rsi_14', 'ema_20', 'ema_50', 'ema_20_50_cross',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'macd_line',
            'macd_signal', 'macd_histogram', 'volume_ma', 'volume_ratio',
            'buy_sell_ratio', 'net_volume', 'net_volume_ma', 'buy_pressure',
            'price_momentum_5', 'price_momentum_10', 'volatility', 'high_low_range',
            'gini_coefficient', 'whale_buy_volume', 'whale_sell_volume',
            'top_10_concentration', 'holder_count', 'sol_correlation',
            'market_favorability', 'whale_state_encoded', 'market_regime_encoded'
        ]
    
    def build_feature_vector(
        self,
        technical_signals: Any,
        whale_metrics: Any,
        holder_stats: Any,
        candle_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Build feature vector for meme coin model (36 features)
        This delegates to the existing _build_feature_vector in signal_generator
        """
        # This is implemented in signal_generator.py for meme tokens
        # We return None here to indicate delegation
        raise NotImplementedError("Use SignalGenerator._build_feature_vector for meme tokens")
    
    def map_prediction_to_signal(self, prediction: Any, probabilities: np.ndarray) -> Tuple[str, float]:
        """
        Map binary model prediction to signal
        Model outputs probability for positive class (BUY)
        """
        from config import signal_config
        
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        
        if probability >= signal_config.XGBOOST_LONG_THRESHOLD:
            return "BUY", float(probability)
        elif probability <= (1 - signal_config.XGBOOST_SHORT_THRESHOLD):
            return "SELL", float(1 - probability)
        else:
            return "HOLD", float(1 - abs(probability - 0.5) * 2)


# ==================== PERPS TOKEN STRATEGY ====================

class PerpsTokenStrategy(BaseTokenTypeStrategy):
    """
    Strategy for perpetual futures token type
    Uses perps API endpoints and 3-class model
    """
    
    def __init__(self):
        self.token_type = TokenType.PERPS
        logger.info("PerpsTokenStrategy initialized")
    
    def get_endpoints(self) -> APIEndpoints:
        """Get perps API endpoints"""
        return APIEndpoints(
            base_url=perps_api_config.BASE_URL,
            candles=perps_api_config.ENDPOINT_CANDLES,
            holders="",  # Not used for perps
            trades=perps_api_config.ENDPOINT_TRADES,
            funding=perps_api_config.ENDPOINT_FUNDING,
            markets=perps_api_config.ENDPOINT_MARKETS
        )
    
    def get_model_paths(self) -> ModelPaths:
        """Get perps ML model paths"""
        return ModelPaths(
            model_path=perps_model_config.get_model_path(),
            scaler_path=perps_model_config.get_scaler_path(),
            features_path=perps_model_config.get_features_path(),
            ticker_encoder_path=perps_model_config.get_ticker_encoder_path()
        )
    
    def get_feature_list(self) -> List[str]:
        """Get perps feature list"""
        return perps_model_config.CORE_FEATURES + perps_model_config.PERPS_FEATURES
    
    def build_feature_vector(
        self,
        technical_signals: Any,
        whale_metrics: Any,
        holder_stats: Any,
        candle_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Build feature vector for perps model
        
        Features:
        - ticker_encoded: 0 (default for single token)
        - rsi_14: From technical signals
        - adx: Not in current TechnicalSignals, default to 50
        - macd: From technical signals (macd_line)
        - ema_20: From technical signals
        - sma_50: Use ema_50 as proxy
        - volume: From candle data
        - obv: Not in current system, default to 0
        - return_24h: From candle data returns
        - hour_of_day: Current hour
        - day_of_week: Current day
        - funding_rate: From market data (optional)
        - open_interest: From market data (optional)
        - oi_momentum: From market data (optional)
        """
        if candle_data is None:
            candle_data = {}
        if market_data is None:
            market_data = {}
        
        features = []
        
        # ticker_encoded (default to 0 for single token prediction)
        features.append(0.0)
        
        # rsi_14
        features.append(float(getattr(technical_signals, 'rsi', 50.0)))
        
        # adx (not available in TechnicalSignals, use signal_strength as proxy)
        features.append(float(getattr(technical_signals, 'signal_strength', 0.5) * 100))
        
        # macd
        features.append(float(getattr(technical_signals, 'macd_line', 0.0)))
        
        # ema_20
        features.append(float(getattr(technical_signals, 'ema_20', 0.0)))
        
        # sma_50 (use ema_50 as proxy)
        features.append(float(getattr(technical_signals, 'ema_50', 0.0)))
        
        # volume
        features.append(float(candle_data.get('volume', candle_data.get('Volume', 0.0))))
        
        # obv (not available, default to 0)
        features.append(0.0)
        
        # return_24h
        features.append(float(candle_data.get('returns', candle_data.get('return_24h', 0.0))))
        
        # hour_of_day
        features.append(float(datetime.now().hour))
        
        # day_of_week
        features.append(float(datetime.now().weekday()))
        
        # Optional perps features
        # funding_rate
        funding_rate = market_data.get('funding_rate', market_data.get('fundingRate', 0.0))
        if funding_rate is not None:
            features.append(float(funding_rate))
        
        # open_interest
        open_interest = market_data.get('open_interest', market_data.get('openInterest', 0.0))
        if open_interest is not None:
            features.append(float(open_interest))
        
        # oi_momentum
        oi_momentum = market_data.get('oi_momentum', 0.0)
        if oi_momentum is not None:
            features.append(float(oi_momentum))
        
        return np.array(features)
    
    def map_prediction_to_signal(self, prediction: Any, probabilities: np.ndarray) -> Tuple[str, float]:
        """
        Map 3-class model prediction to signal
        Classes: 0=SHORT, 1=HOLD, 2=LONG
        Maps to: SELL, HOLD, BUY
        """
        # Get class prediction (0, 1, or 2)
        class_idx = int(prediction) if isinstance(prediction, (int, np.integer)) else int(prediction[0])
        
        # Get confidence (max probability)
        confidence = float(np.max(probabilities))
        
        # Map to signal
        signal = perps_model_config.OUTPUT_MAPPING.get(class_idx, "HOLD")
        
        return signal, confidence


# ==================== TOKEN TYPE ROUTER ====================

class TokenTypeRouter:
    """
    Main router class for token type based operations
    Implements Factory Pattern to create appropriate strategy
    """
    
    _strategies: Dict[TokenType, BaseTokenTypeStrategy] = {}
    
    def __init__(self):
        """Initialize router with strategies"""
        self._strategies = {
            TokenType.MEME: MemeTokenStrategy(),
            TokenType.PERPS: PerpsTokenStrategy()
        }
        logger.info(f"TokenTypeRouter initialized with {len(self._strategies)} strategies")
    
    def get_strategy(self, token_type: str) -> BaseTokenTypeStrategy:
        """
        Get strategy for token type
        
        Args:
            token_type: Token type string ('meme' or 'perps')
            
        Returns:
            Strategy instance for the token type
        """
        token_type_enum = TokenType.from_string(token_type)
        strategy = self._strategies.get(token_type_enum)
        
        if strategy is None:
            logger.warning(f"No strategy for token_type '{token_type}', defaulting to meme")
            strategy = self._strategies[TokenType.MEME]
        
        return strategy
    
    def get_endpoints(self, token_type: str) -> APIEndpoints:
        """Get API endpoints for token type"""
        return self.get_strategy(token_type).get_endpoints()
    
    def get_model_paths(self, token_type: str) -> ModelPaths:
        """Get ML model paths for token type"""
        return self.get_strategy(token_type).get_model_paths()
    
    def get_feature_list(self, token_type: str) -> List[str]:
        """Get feature list for token type"""
        return self.get_strategy(token_type).get_feature_list()
    
    def is_perps(self, token_type: str) -> bool:
        """Check if token type is perps"""
        return TokenType.from_string(token_type) == TokenType.PERPS
    
    def is_meme(self, token_type: str) -> bool:
        """Check if token type is meme"""
        return TokenType.from_string(token_type) == TokenType.MEME
    
    def validate_token_type(self, token_type: str) -> Tuple[bool, str]:
        """
        Validate token type
        
        Returns:
            Tuple of (is_valid, normalized_token_type)
        """
        if token_type_config.is_valid(token_type):
            return True, token_type.lower().strip()
        else:
            return False, token_type_config.DEFAULT_TOKEN_TYPE


# ==================== SINGLETON INSTANCE ====================

# Create singleton router instance
_router_instance: Optional[TokenTypeRouter] = None

def get_router() -> TokenTypeRouter:
    """Get or create singleton router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = TokenTypeRouter()
    return _router_instance


# ==================== CONVENIENCE FUNCTIONS ====================

def get_endpoints_for_token_type(token_type: str) -> APIEndpoints:
    """Convenience function to get endpoints"""
    return get_router().get_endpoints(token_type)


def get_model_paths_for_token_type(token_type: str) -> ModelPaths:
    """Convenience function to get model paths"""
    return get_router().get_model_paths(token_type)


def is_perps_token(token_type: str) -> bool:
    """Convenience function to check if perps"""
    return get_router().is_perps(token_type)


def is_meme_token(token_type: str) -> bool:
    """Convenience function to check if meme"""
    return get_router().is_meme(token_type)


# ==================== TEST ====================

if __name__ == "__main__":
    # Test the router
    print("=" * 60)
    print("Token Type Router Test")
    print("=" * 60)
    
    router = get_router()
    
    # Test meme endpoints
    print("\nMeme Token Endpoints:")
    meme_endpoints = router.get_endpoints("meme")
    print(f"  Base URL: {meme_endpoints.base_url}")
    print(f"  Candles: {meme_endpoints.candles}")
    print(f"  Holders: {meme_endpoints.holders}")
    print(f"  User Holdings: {meme_endpoints.user_holdings}")
    
    # Test perps endpoints
    print("\nPerps Token Endpoints:")
    perps_endpoints = router.get_endpoints("perps")
    print(f"  Base URL: {perps_endpoints.base_url}")
    print(f"  Candles: {perps_endpoints.candles}")
    print(f"  Funding: {perps_endpoints.funding}")
    print(f"  Markets: {perps_endpoints.markets}")
    
    # Test model paths
    print("\nMeme Model Paths:")
    meme_paths = router.get_model_paths("meme")
    print(f"  Model: {meme_paths.model_path}")
    print(f"  Exists: {meme_paths.exists()}")
    
    print("\nPerps Model Paths:")
    perps_paths = router.get_model_paths("perps")
    print(f"  Model: {perps_paths.model_path}")
    print(f"  Exists: {perps_paths.exists()}")
    
    # Test validation
    print("\nToken Type Validation:")
    print(f"  'meme': {router.validate_token_type('meme')}")
    print(f"  'perps': {router.validate_token_type('perps')}")
    print(f"  'invalid': {router.validate_token_type('invalid')}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
