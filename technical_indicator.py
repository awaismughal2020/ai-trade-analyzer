import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
import time
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical analysis indicators for crypto trading strategy
    Implements RSI-14, EMA-20, and Volume Delta calculations
    """

    def __init__(self):
        self.rsi_period = 14
        self.ema_period = 20

    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices: List of closing prices
            period: Period for RSI calculation (default 14)

        Returns:
            List of RSI values
        """
        if len(prices) < period + 1:
            return [50.0] * len(prices)  # Return neutral RSI if insufficient data

        prices = np.array(prices)
        deltas = np.diff(prices)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Calculate average gains and losses
        avg_gains = []
        avg_losses = []
        rsi_values = []

        # First RSI calculation using simple average
        first_avg_gain = np.mean(gains[:period]) if len(gains) >= period else 0
        first_avg_loss = np.mean(losses[:period]) if len(losses) >= period else 1

        if first_avg_loss == 0:
            first_avg_loss = 0.0001  # Prevent division by zero

        rs = first_avg_gain / first_avg_loss
        first_rsi = 100 - (100 / (1 + rs))
        rsi_values.append(first_rsi)

        avg_gains.append(first_avg_gain)
        avg_losses.append(first_avg_loss)

        # Subsequent RSI calculations using Wilder's smoothing
        for i in range(period, len(gains)):
            current_gain = gains[i]
            current_loss = losses[i]

            # Wilder's smoothing formula
            avg_gain = ((avg_gains[-1] * (period - 1)) + current_gain) / period
            avg_loss = ((avg_losses[-1] * (period - 1)) + current_loss) / period

            avg_gains.append(avg_gain)
            avg_losses.append(avg_loss)

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            rsi_values.append(rsi)

        # Pad beginning with neutral values
        full_rsi = [50.0] * period + rsi_values
        return full_rsi[:len(prices)]

    def calculate_ema(self, prices: List[float], period: int = 20) -> List[float]:
        """
        Calculate Exponential Moving Average (EMA)

        Args:
            prices: List of closing prices
            period: Period for EMA calculation (default 20)

        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return prices.copy()  # Return original prices if insufficient data

        prices = np.array(prices)
        ema_values = []

        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # First EMA is simple moving average
        first_ema = np.mean(prices[:period])
        ema_values = [first_ema] * period

        # Calculate subsequent EMAs
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)

        return ema_values

    def check_ema_uptrend(self, ema_values: List[float], lookback: int = 20) -> bool:
        """
        Check if EMA is in uptrend by comparing current vs previous values

        Args:
            ema_values: List of EMA values
            lookback: Number of periods to compare (default 20)

        Returns:
            True if uptrend, False otherwise
        """
        if len(ema_values) < lookback + 1:
            return False

        current_ema = ema_values[-1]
        past_ema = ema_values[-lookback - 1]

        return current_ema > past_ema

    def calculate_volume_delta(self, buy_volumes: List[float], sell_volumes: List[float]) -> List[float]:
        """
        Calculate Volume Delta (Buy Volume - Sell Volume)

        Args:
            buy_volumes: List of buy volumes
            sell_volumes: List of sell volumes

        Returns:
            List of volume delta values
        """
        if len(buy_volumes) != len(sell_volumes):
            raise ValueError("Buy and sell volume lists must be same length")

        return [buy - sell for buy, sell in zip(buy_volumes, sell_volumes)]

    def get_rsi_signal(self, rsi: float) -> str:
        """
        Get RSI trading signal

        Args:
            rsi: RSI value

        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if rsi < 30:
            return 'BUY'  # Oversold
        elif rsi > 70:
            return 'SELL'  # Overbought
        else:
            return 'HOLD'  # Neutral

    def apply_3_step_strategy(self, rsi: float, ema_uptrend: bool, volume_delta: float) -> Dict:
        """
        Apply your 3-step trading strategy

        Args:
            rsi: Current RSI value
            ema_uptrend: Whether EMA is in uptrend
            volume_delta: Current volume delta

        Returns:
            Dictionary with signal and reasoning
        """
        # Step 1: Check EMA uptrend
        if not ema_uptrend:
            return {
                'signal': 'NO_SIGNAL',
                'reason': 'EMA not in uptrend - wait for better trend',
                'strength': 0
            }

        # Step 2: Check RSI levels
        rsi_signal = self.get_rsi_signal(rsi)

        if rsi_signal == 'HOLD':
            return {
                'signal': 'HOLD',
                'reason': f'RSI {rsi:.1f} in neutral zone (30-70)',
                'strength': 1
            }

        # Step 3: Volume confirmation
        volume_positive = volume_delta > 0

        if rsi_signal == 'BUY':
            if volume_positive:
                return {
                    'signal': 'STRONG_BUY',
                    'reason': f'EMA uptrend + RSI oversold ({rsi:.1f}) + positive volume',
                    'strength': 3
                }
            else:
                return {
                    'signal': 'WEAK_BUY',
                    'reason': f'EMA uptrend + RSI oversold ({rsi:.1f}) but negative volume',
                    'strength': 2
                }

        elif rsi_signal == 'SELL':
            if not volume_positive:  # Negative volume confirms sell
                return {
                    'signal': 'STRONG_SELL',
                    'reason': f'EMA uptrend but RSI overbought ({rsi:.1f}) + negative volume',
                    'strength': 3
                }
            else:
                return {
                    'signal': 'WEAK_SELL',
                    'reason': f'RSI overbought ({rsi:.1f}) but positive volume conflicts',
                    'strength': 2
                }

        return {
            'signal': 'HOLD',
            'reason': 'Mixed signals - wait for clearer setup',
            'strength': 1
        }


class CoinGeckoDataFetcher:
    """
    Fetches price and volume data from CoinGecko for technical analysis
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"

    # In your CoinGeckoDataFetcher class, modify get_hourly_data:
    def get_hourly_data(self, coin_id: str, days: int = 30) -> Dict:
        """Get hourly OHLCV data for technical analysis"""
        try:
            # Ensure days is within the range that gives hourly data automatically
            days = max(2, min(days, 90))

            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
            }
            headers = {'x-cg-pro-api-key': self.api_key}

            response = requests.get(url, params=params, headers=headers)
            time.sleep(0.1)

            if response.status_code == 200:
                data = response.json()

                timestamps = []
                prices = []
                volumes = []

                for i, (timestamp, price) in enumerate(data['prices']):
                    # Create timezone-naive datetime to match trade dates
                    dt = datetime.fromtimestamp(timestamp / 1000)
                    timestamps.append(dt)
                    prices.append(price)

                    if i < len(data['total_volumes']):
                        volumes.append(data['total_volumes'][i][1])
                    else:
                        volumes.append(0)

                print(f"DEBUG API: Successfully fetched {len(prices)} price points")
                return {
                    'timestamps': timestamps,
                    'prices': prices,
                    'volumes': volumes
                }
            else:
                print(f"API Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error fetching data for {coin_id}: {e}")
            import traceback
            traceback.print_exc()

        return {'timestamps': [], 'prices': [], 'volumes': []}


class TradingStrategyAnalyzer:
    """
    Main class that combines technical indicators with trade analysis
    """

    def __init__(self, coingecko_api_key: str):
        self.indicators = TechnicalIndicators()
        self.data_fetcher = CoinGeckoDataFetcher(coingecko_api_key)

        # Token mapping for CoinGecko IDs
        self.token_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOGE': 'dogecoin',
            'SHIB': 'shiba-inu',
            'MATIC': 'matic-network',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'AVAX': 'avalanche-2'
        }

    def analyze_trade_with_indicators(self, token: str, buy_date: datetime,
                                      sell_date: datetime) -> Dict:
        """
        Analyze a specific trade using technical indicators

        Args:
            token: Token symbol (e.g., 'BTC', 'ETH')
            buy_date: When the trade was entered
            sell_date: When the trade was exited

        Returns:
            Dictionary with technical analysis results
        """
        if token not in self.token_mapping:
            return {'error': f'Token {token} not supported'}

        coin_id = self.token_mapping[token]

        # Get data covering the trade period plus buffer for indicators
        days_needed = max(30, (sell_date - buy_date).days + 10)
        market_data = self.data_fetcher.get_hourly_data(coin_id, days_needed)

        if not market_data['prices']:
            return {'error': 'Could not fetch market data'}

        # Calculate technical indicators
        prices = market_data['prices']
        volumes = market_data['volumes']
        timestamps = market_data['timestamps']

        rsi_values = self.indicators.calculate_rsi(prices)
        ema_values = self.indicators.calculate_ema(prices)

        # Find closest data points to buy/sell dates
        buy_index = self._find_closest_timestamp_index(timestamps, buy_date)
        sell_index = self._find_closest_timestamp_index(timestamps, sell_date)

        if buy_index == -1 or sell_index == -1:
            return {'error': 'Could not find matching timestamps'}

        # Get indicator values at trade dates
        buy_rsi = rsi_values[buy_index] if buy_index < len(rsi_values) else 50
        buy_ema_uptrend = self.indicators.check_ema_uptrend(ema_values[:buy_index + 1])
        buy_volume_delta = volumes[buy_index] if buy_index < len(volumes) else 0

        sell_rsi = rsi_values[sell_index] if sell_index < len(rsi_values) else 50
        sell_ema_uptrend = self.indicators.check_ema_uptrend(ema_values[:sell_index + 1])
        sell_volume_delta = volumes[sell_index] if sell_index < len(volumes) else 0

        # Apply trading strategy
        buy_signal = self.indicators.apply_3_step_strategy(
            buy_rsi, buy_ema_uptrend, buy_volume_delta
        )

        sell_signal = self.indicators.apply_3_step_strategy(
            sell_rsi, sell_ema_uptrend, sell_volume_delta
        )

        return {
            'token': token,
            'buy_date': buy_date.strftime('%Y-%m-%d %H:%M'),
            'sell_date': sell_date.strftime('%Y-%m-%d %H:%M'),
            'buy_analysis': {
                'rsi': buy_rsi,
                'ema_uptrend': buy_ema_uptrend,
                'volume_delta': buy_volume_delta,
                'strategy_signal': buy_signal
            },
            'sell_analysis': {
                'rsi': sell_rsi,
                'ema_uptrend': sell_ema_uptrend,
                'volume_delta': sell_volume_delta,
                'strategy_signal': sell_signal
            },
            'trade_quality': self._evaluate_trade_quality(buy_signal, sell_signal)
        }

    def _find_closest_timestamp_index(self, timestamps: List[datetime],
                                      target_date: datetime) -> int:
        """Find the index of closest timestamp to target date"""
        if not timestamps:
            return -1

        min_diff = float('inf')
        closest_index = -1

        for i, timestamp in enumerate(timestamps):
            diff = abs((timestamp - target_date).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_index = i

        return closest_index

    def _evaluate_trade_quality(self, buy_signal: Dict, sell_signal: Dict) -> Dict:
        """
        Evaluate the quality of the trade based on strategy signals

        Args:
            buy_signal: Strategy signal at buy time
            sell_signal: Strategy signal at sell time

        Returns:
            Trade quality assessment
        """
        buy_strength = buy_signal.get('strength', 0)
        sell_strength = sell_signal.get('strength', 0)

        # Evaluate buy timing
        buy_quality = "Poor"
        if buy_signal['signal'] in ['STRONG_BUY', 'WEAK_BUY']:
            buy_quality = "Good" if buy_strength >= 3 else "Fair"

        # Evaluate sell timing
        sell_quality = "Poor"
        if sell_signal['signal'] in ['STRONG_SELL', 'WEAK_SELL']:
            sell_quality = "Good" if sell_strength >= 3 else "Fair"
        elif sell_signal['signal'] == 'NO_SIGNAL':  # Sold during downtrend
            sell_quality = "Fair"

        # Overall assessment
        total_score = buy_strength + sell_strength
        if total_score >= 5:
            overall = "Excellent"
        elif total_score >= 3:
            overall = "Good"
        elif total_score >= 1:
            overall = "Fair"
        else:
            overall = "Poor"

        return {
            'buy_timing': buy_quality,
            'sell_timing': sell_quality,
            'overall_quality': overall,
            'score': total_score,
            'recommendations': self._generate_recommendations(buy_signal, sell_signal)
        }

    def _generate_recommendations(self, buy_signal: Dict, sell_signal: Dict) -> List[str]:
        """Generate specific recommendations based on the trade analysis"""
        recommendations = []

        # Buy recommendations
        if buy_signal['signal'] == 'NO_SIGNAL':
            recommendations.append("Wait for EMA uptrend before buying")
        elif buy_signal['signal'] in ['WEAK_BUY']:
            recommendations.append("Look for positive volume confirmation on buy signals")

        # Sell recommendations
        if sell_signal['signal'] in ['HOLD', 'WEAK_BUY']:
            recommendations.append("Consider holding longer when RSI is not overbought")

        # General recommendations
        if buy_signal.get('strength', 0) <= 1:
            recommendations.append("Use stricter entry criteria - wait for all 3 strategy conditions")

        if not recommendations:
            recommendations.append("Good trade execution following the strategy")

        return recommendations


# Example usage
def example_usage():
    """Example of how to use the technical indicators"""

    # Initialize the analyzer
    analyzer = TradingStrategyAnalyzer(coingecko_api_key="your-api-key")

    # Sample price data for demonstration
    sample_prices = [100, 102, 98, 95, 93, 96, 99, 103, 107, 105, 102, 98, 94, 92, 95, 98, 101, 104, 108, 112]

    # Calculate indicators
    indicators = TechnicalIndicators()
    rsi_values = indicators.calculate_rsi(sample_prices)
    ema_values = indicators.calculate_ema(sample_prices)

    print("Sample RSI values:", rsi_values[-5:])  # Last 5 values
    print("Sample EMA values:", ema_values[-5:])  # Last 5 values

    # Test strategy
    current_rsi = rsi_values[-1]
    ema_uptrend = indicators.check_ema_uptrend(ema_values)
    volume_delta = 1000  # Positive volume

    strategy_result = indicators.apply_3_step_strategy(current_rsi, ema_uptrend, volume_delta)
    print("Strategy result:", strategy_result)


if __name__ == "__main__":
    example_usage()