"""
Technical Indicator Engine
Calculates technical indicators for price analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import technical_config

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignals:
    """Technical indicator signals"""
    rsi: float
    rsi_signal: str  # "oversold", "overbought", "neutral"
    ema_20: float
    ema_50: float
    ema_cross_signal: str  # "bullish", "bearish", "neutral"
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_trend: str  # "bullish", "bearish", "neutral"
    bb_upper: float
    bb_middle: float  # Middle Bollinger Band (SMA)
    bb_lower: float
    bb_position: float  # 0-1, where in the band
    volume_ratio: float  # Current vs average
    buy_pressure: float  # 0-1
    price_momentum_5: float
    price_momentum_10: float
    volatility: float
    overall_signal: str  # "bullish", "bearish", "neutral"
    signal_strength: float  # 0-1

    def to_dict(self) -> Dict:
        """Convert TechnicalSignals to dictionary for JSON serialization"""
        return {
            'rsi': self.rsi,
            'rsi_signal': self.rsi_signal,
            'ema_20': self.ema_20,
            'ema_50': self.ema_50,
            'ema_cross_signal': self.ema_cross_signal,
            'macd_line': self.macd_line,
            'macd_signal': self.macd_signal,
            'macd_histogram': self.macd_histogram,
            'macd_trend': self.macd_trend,
            'bb_upper': self.bb_upper,
            'bb_middle': self.bb_middle,
            'bb_lower': self.bb_lower,
            'bb_position': self.bb_position,
            'volume_ratio': self.volume_ratio,
            'buy_pressure': self.buy_pressure,
            'price_momentum_5': self.price_momentum_5,
            'price_momentum_10': self.price_momentum_10,
            'volatility': self.volatility,
            'overall_signal': self.overall_signal,
            'signal_strength': self.signal_strength
        }


class TechnicalIndicatorEngine:
    """
    Calculates technical indicators for trading signal generation
    """
    
    def __init__(self):
        """Initialize Technical Indicator Engine"""
        self.config = technical_config
        logger.info("Technical Indicator Engine initialized")
    
    def calculate_all_indicators(
        self,
        candles_df: pd.DataFrame
    ) -> TechnicalSignals:
        """
        Calculate all technical indicators from candle data
        
        Args:
            candles_df: DataFrame with OHLCV data
            
        Returns:
            TechnicalSignals object with all indicators
        """
        if len(candles_df) < 50:
            logger.warning("Insufficient candle data for full technical analysis")
            return self._empty_signals()
        
        # Ensure data is sorted (handle both 'Timestamp' and 'timestamp')
        timestamp_col = 'Timestamp' if 'Timestamp' in candles_df.columns else 'timestamp'
        df = candles_df.sort_values(timestamp_col).copy()
        
        # Handle column name case-insensitivity (perps uses lowercase)
        def get_col(df, *names):
            for name in names:
                if name in df.columns:
                    return df[name].values
            return None
        
        # Get latest values (support both meme and perps column names)
        close = get_col(df, 'Close', 'close')
        if close is None:
            logger.warning("No close price column found")
            return self._empty_signals()
        # Ensure close is numeric (perps data may have strings)
        close = np.array(close, dtype=float)
        
        volume = get_col(df, 'Volume', 'volume')
        if volume is None:
            volume = np.ones(len(df))
        else:
            volume = np.array(volume, dtype=float)
        buy_volume = get_col(df, 'BuyVolume', 'buy_volume')
        if buy_volume is None:
            buy_volume = volume * 0.5
        else:
            buy_volume = np.array(buy_volume, dtype=float)
        sell_volume = get_col(df, 'SellVolume', 'sell_volume')
        if sell_volume is None:
            sell_volume = volume * 0.5
        else:
            sell_volume = np.array(sell_volume, dtype=float)
        
        # Calculate RSI
        rsi = self.calculate_rsi(close)
        rsi_signal = self._classify_rsi(rsi)
        
        # Calculate EMAs
        ema_20 = self.calculate_ema(close, self.config.EMA_SHORT)
        ema_50 = self.calculate_ema(close, self.config.EMA_MEDIUM)
        ema_cross = self._classify_ema_cross(ema_20, ema_50)
        
        # Calculate MACD
        macd_line, macd_sig, macd_hist = self.calculate_macd(close)
        macd_trend = self._classify_macd(macd_hist)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
        bb_position = self._calculate_bb_position(close[-1], bb_upper, bb_lower)
        
        # Volume analysis
        volume_ma = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        volume_ratio = volume[-1] / volume_ma if volume_ma > 0 else 1.0
        
        # Buy pressure
        total_vol = buy_volume[-1] + sell_volume[-1]
        buy_pressure = buy_volume[-1] / total_vol if total_vol > 0 else 0.5
        
        # Price momentum
        momentum_5 = self._calculate_momentum(close, 5)
        momentum_10 = self._calculate_momentum(close, 10)
        
        # Volatility
        volatility = self._calculate_volatility(close)
        
        # Overall signal
        overall, strength = self._calculate_overall_signal(
            rsi_signal=rsi_signal,
            ema_cross=ema_cross,
            macd_trend=macd_trend,
            momentum_5=momentum_5,
            buy_pressure=buy_pressure
        )
        
        return TechnicalSignals(
            rsi=rsi,
            rsi_signal=rsi_signal,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_cross_signal=ema_cross,
            macd_line=macd_line,
            macd_signal=macd_sig,
            macd_histogram=macd_hist,
            macd_trend=macd_trend,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_position=bb_position,
            volume_ratio=volume_ratio,
            buy_pressure=buy_pressure,
            price_momentum_5=momentum_5,
            price_momentum_10=momentum_10,
            volatility=volatility,
            overall_signal=overall,
            signal_strength=strength
        )
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral default
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 99.0  # Cap below 100 so "all gains" is distinguishable from a sentinel
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return float(prices[-1]) if len(prices) > 0 else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        
        return float(ema)
    
    def calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return float(np.mean(prices))
        return float(np.mean(prices[-period:]))
    
    def calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD (line, signal, histogram)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = self._calculate_ema_series(prices, fast)
        ema_slow = self._calculate_ema_series(prices, slow)
        
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # Calculate signal line (EMA of MACD)
        macd_values = ema_fast - ema_slow
        signal_line = self._calculate_ema_series(macd_values, signal)[-1]
        
        histogram = macd_line - signal_line
        
        return float(macd_line), float(signal_line), float(histogram)
    
    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        if len(prices) < period:
            price = prices[-1] if len(prices) > 0 else 0
            return price, price, price
        
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return float(upper), float(middle), float(lower)
    
    def _calculate_ema_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA series"""
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices, dtype=float)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calculate price momentum (percentage change)"""
        if len(prices) <= period:
            return 0.0
        
        current = prices[-1]
        past = prices[-period-1]
        
        if past == 0:
            return 0.0
        
        return float((current - past) / past)
    
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        if len(prices) < period:
            return 0.0
        
        returns = np.diff(prices[-period:]) / prices[-period:-1]
        return float(np.std(returns))
    
    def _calculate_bb_position(
        self,
        price: float,
        upper: float,
        lower: float
    ) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        if upper == lower:
            return 0.5
        
        position = (price - lower) / (upper - lower)
        return float(max(0, min(1, position)))
    
    def _classify_rsi(self, rsi: float) -> str:
        """Classify RSI signal"""
        if rsi <= self.config.RSI_OVERSOLD:
            return "oversold"
        elif rsi >= self.config.RSI_OVERBOUGHT:
            return "overbought"
        else:
            return "neutral"
    
    def _classify_ema_cross(self, ema_short: float, ema_long: float) -> str:
        """Classify EMA crossover signal"""
        diff_pct = (ema_short - ema_long) / ema_long if ema_long != 0 else 0
        
        if diff_pct > 0.02:  # 2% above
            return "bullish"
        elif diff_pct < -0.02:  # 2% below
            return "bearish"
        else:
            return "neutral"
    
    def _classify_macd(self, histogram: float) -> str:
        """Classify MACD trend"""
        if histogram > 0:
            return "bullish"
        elif histogram < 0:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_overall_signal(
        self,
        rsi_signal: str,
        ema_cross: str,
        macd_trend: str,  # Kept for backward compatibility but not used
        momentum_5: float,
        buy_pressure: float
    ) -> Tuple[str, float]:
        """
        Calculate overall technical signal
        
        Note: MACD has been removed from signal calculation but values are still 
        calculated for ML model feature compatibility.
        
        Indicators used:
        - RSI (oversold/overbought)
        - EMA cross (bullish/bearish)
        - Momentum (5-period)
        - Buy pressure
        
        Returns:
            Tuple of (signal, strength)
        """
        bullish_points = 0
        bearish_points = 0
        total_points = 0
        
        # RSI contribution
        total_points += 1
        if rsi_signal == "oversold":
            bullish_points += 1
        elif rsi_signal == "overbought":
            bearish_points += 1
        
        # EMA cross contribution
        total_points += 1
        if ema_cross == "bullish":
            bullish_points += 1
        elif ema_cross == "bearish":
            bearish_points += 1
        
        # NOTE: MACD removed from signal calculation (values still computed for ML model)
        
        # Momentum contribution
        total_points += 1
        if momentum_5 > 0.05:  # 5% positive momentum
            bullish_points += 1
        elif momentum_5 < -0.05:  # 5% negative momentum
            bearish_points += 1
        
        # Buy pressure contribution
        total_points += 1
        if buy_pressure > 0.6:
            bullish_points += 1
        elif buy_pressure < 0.4:
            bearish_points += 1
        
        # Determine overall signal (now based on 4 indicators instead of 5)
        if bullish_points > bearish_points + 1:
            signal = "bullish"
            strength = bullish_points / total_points
        elif bearish_points > bullish_points + 1:
            signal = "bearish"
            strength = bearish_points / total_points
        else:
            signal = "neutral"
            strength = 0.5
        
        return signal, strength
    
    def _empty_signals(self) -> TechnicalSignals:
        """Return empty/neutral signals"""
        return TechnicalSignals(
            rsi=50.0,
            rsi_signal="neutral",
            ema_20=0.0,
            ema_50=0.0,
            ema_cross_signal="neutral",
            macd_line=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            macd_trend="neutral",
            bb_upper=0.0,
            bb_middle=0.0,
            bb_lower=0.0,
            bb_position=0.5,
            volume_ratio=1.0,
            buy_pressure=0.5,
            price_momentum_5=0.0,
            price_momentum_10=0.0,
            volatility=0.0,
            overall_signal="neutral",
            signal_strength=0.5
        )


def convert_signals_to_dict(signals: TechnicalSignals) -> Dict:
    """Convert TechnicalSignals to dictionary"""
    return {
        'rsi': signals.rsi,
        'rsi_signal': signals.rsi_signal,
        'ema_20': signals.ema_20,
        'ema_50': signals.ema_50,
        'ema_cross_signal': signals.ema_cross_signal,
        'macd_line': signals.macd_line,
        'macd_signal': signals.macd_signal,
        'macd_histogram': signals.macd_histogram,
        'macd_trend': signals.macd_trend,
        'bb_upper': signals.bb_upper,
        'bb_lower': signals.bb_lower,
        'bb_position': signals.bb_position,
        'volume_ratio': signals.volume_ratio,
        'buy_pressure': signals.buy_pressure,
        'price_momentum_5': signals.price_momentum_5,
        'price_momentum_10': signals.price_momentum_10,
        'volatility': signals.volatility,
        'overall_signal': signals.overall_signal,
        'signal_strength': signals.signal_strength
    }


if __name__ == "__main__":
    print("Technical Indicator Engine")
    print("=" * 60)
    
    # Generate sample OHLCV data
    np.random.seed(42)
    n = 100
    
    # Simulated price with trend
    base_price = 0.0001
    trend = np.cumsum(np.random.randn(n) * 0.00001)
    close = base_price + trend + np.abs(np.random.randn(n) * 0.00001)
    
    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=n, freq='5min'),
        'Open': close * 0.999,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.exponential(1000000, n),
        'BuyVolume': np.random.exponential(500000, n),
        'SellVolume': np.random.exponential(500000, n)
    })
    
    engine = TechnicalIndicatorEngine()
    signals = engine.calculate_all_indicators(df)
    
    print(f"\nTechnical Signals:")
    print(f"  RSI: {signals.rsi:.2f} ({signals.rsi_signal})")
    print(f"  EMA 20/50: {signals.ema_20:.8f} / {signals.ema_50:.8f} ({signals.ema_cross_signal})")
    print(f"  MACD: {signals.macd_histogram:.8f} ({signals.macd_trend})")
    print(f"  BB Position: {signals.bb_position:.2f}")
    print(f"  Volume Ratio: {signals.volume_ratio:.2f}")
    print(f"  Buy Pressure: {signals.buy_pressure:.2f}")
    print(f"  Momentum 5/10: {signals.price_momentum_5:.4f} / {signals.price_momentum_10:.4f}")
    print(f"  Volatility: {signals.volatility:.6f}")
    print(f"\n  Overall: {signals.overall_signal} (strength: {signals.signal_strength:.2f})")

