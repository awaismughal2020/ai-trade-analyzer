"""
Meme Coin Market Analyzer
Automated wrapper for fetching and analyzing meme coins from CoinGecko
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Import your existing modules
from data_collector import DataCollector
from data_processor import DataProcessor
from market_model import MarketAnalysisLSTM

# Load environment variables
load_dotenv()


class MemeCoinAnalyzer:
    """
    Wrapper class for automated meme coin analysis
    Fetches top meme coins from CoinGecko and performs market analysis
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Meme Coin Analyzer

        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = os.getenv('COINGECKO_API_KEY')
        self.model_path = model_path or os.getenv('MODEL_SAVE_PATH', './models/meme_coin_market_model')

        # Initialize components
        self.collector = None
        self.processor = DataProcessor()
        self.model = None

        # Analysis results storage
        self.analysis_results = {}
        self.top_opportunities = []

        print("=" * 60)
        print("MEME COIN ANALYZER INITIALIZED")
        print("=" * 60)

    def get_top_meme_coins(self,
                           limit: int = 20,
                           min_volume: float = 100000,
                           min_market_cap: float = 1000000) -> List[Dict]:
        """
        Fetch top meme coins from CoinGecko based on market cap and volume

        Args:
            limit: Number of coins to fetch
            min_volume: Minimum 24h volume in USD
            min_market_cap: Minimum market cap in USD

        Returns:
            List of meme coin dictionaries
        """
        print(f"\nFetching top {limit} meme coins from CoinGecko...")

        headers = {'accept': 'application/json'}
        if self.api_key and self.api_key != 'your_api_key_here':
            headers['x-cg-demo-api-key'] = self.api_key

        try:
            # Fetch coins with meme category
            url = f"{self.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'category': 'meme-token',  # CoinGecko category for meme coins
                'order': 'market_cap_desc',
                'per_page': limit * 2,  # Get extra to filter
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h,7d,30d'
            }

            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            coins_data = response.json()

            # Filter coins based on criteria
            filtered_coins = []
            for coin in coins_data:
                if (coin.get('total_volume', 0) >= min_volume and
                        coin.get('market_cap', 0) >= min_market_cap):
                    filtered_coins.append({
                        'id': coin['id'],
                        'symbol': coin['symbol'].upper(),
                        'name': coin['name'],
                        'market_cap': coin['market_cap'],
                        'volume_24h': coin['total_volume'],
                        'price': coin['current_price'],
                        'price_change_24h': coin.get('price_change_percentage_24h', 0),
                        'price_change_7d': coin.get('price_change_percentage_7d', 0),
                        'price_change_30d': coin.get('price_change_percentage_30d', 0),
                        'market_cap_rank': coin.get('market_cap_rank', 999)
                    })

                if len(filtered_coins) >= limit:
                    break

            print(f"Found {len(filtered_coins)} meme coins meeting criteria:")
            for i, coin in enumerate(filtered_coins[:10], 1):
                print(f"{i:2}. {coin['symbol']:6} - {coin['name'][:20]:20} "
                      f"MCap: ${coin['market_cap'] / 1e6:.1f}M, "
                      f"24h: {coin['price_change_24h']:+.1f}%")

            return filtered_coins

        except requests.exceptions.RequestException as e:
            print(f"Error fetching meme coins: {e}")
            return []

    def analyze_coin(self,
                     coin_id: str,
                     days: int = 30,
                     use_cached: bool = True) -> Dict:
        """
        Perform comprehensive analysis on a single meme coin

        Args:
            coin_id: CoinGecko coin ID
            days: Days of historical data to analyze
            use_cached: Whether to use cached data if available

        Returns:
            Analysis results dictionary
        """
        print(f"\n{'=' * 40}")
        print(f"Analyzing: {coin_id}")
        print(f"{'=' * 40}")

        # Check for cached data
        cache_file = f"data/cache_{coin_id}_{days}d.csv"

        if use_cached and os.path.exists(cache_file):
            # Check if cache is recent (less than 1 hour old)
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 3600:  # 1 hour
                print(f"Using cached data ({cache_age / 60:.1f} minutes old)")
                coin_data = pd.read_csv(cache_file)
                coin_data['timestamp'] = pd.to_datetime(coin_data['timestamp'])
            else:
                coin_data = self._fetch_coin_data(coin_id, days)
                if coin_data is not None:
                    coin_data.to_csv(cache_file, index=False)
        else:
            coin_data = self._fetch_coin_data(coin_id, days)
            if coin_data is not None and not coin_data.empty:
                os.makedirs("data", exist_ok=True)
                coin_data.to_csv(cache_file, index=False)

        if coin_data is None or coin_data.empty:
            return {'error': f'No data available for {coin_id}'}

        # Process data
        analysis_result = self._perform_technical_analysis(coin_data, coin_id)

        # Get prediction if model is loaded
        if self.model and analysis_result.get('sequences') is not None:
            prediction = self._get_model_prediction(analysis_result['sequences'])
            analysis_result['prediction'] = prediction

        return analysis_result

    def _fetch_coin_data(self, coin_id: str, days: int) -> pd.DataFrame:
        """Fetch historical data for a coin"""
        if self.collector is None:
            # Temporarily modify environment to fetch single coin
            original_coins = os.getenv('TARGET_COINS', '')
            os.environ['TARGET_COINS'] = coin_id
            self.collector = DataCollector()
            os.environ['TARGET_COINS'] = original_coins

        return self.collector.get_coin_data(coin_id, days)

    def _perform_technical_analysis(self,
                                    df: pd.DataFrame,
                                    coin_id: str) -> Dict:
        """
        Perform technical analysis on coin data

        Args:
            df: Historical price data
            coin_id: Coin identifier

        Returns:
            Analysis results
        """
        # Clean and process data
        clean_data = self.processor.clean_data(df)
        if clean_data.empty:
            return {'error': 'Data cleaning failed'}

        # Calculate indicators
        processed_data = self.processor.calculate_technical_indicators(clean_data)
        if processed_data.empty:
            return {'error': 'Indicator calculation failed'}

        # Get latest values
        latest = processed_data.iloc[-1]

        # Calculate additional metrics
        recent_data = processed_data.tail(20)

        # Trend analysis
        price_trend = self._calculate_trend(recent_data['close'].values)
        volume_trend = self._calculate_trend(recent_data['volume'].values)

        # Support and resistance levels
        support, resistance = self._find_support_resistance(recent_data)

        # Volatility metrics
        volatility = recent_data['close'].pct_change().std() * np.sqrt(252)  # Annualized

        # Create sequences for prediction if we have enough data
        sequences = None
        if len(processed_data) >= self.processor.sequence_length:
            try:
                labeled_data = self.processor.create_labels(processed_data)
                if not labeled_data.empty:
                    X, y = self.processor.create_sequences(labeled_data)
                    if len(X) > 0:
                        sequences = X[-1:]  # Get last sequence for prediction
            except Exception as e:
                print(f"Warning: Could not create sequences: {e}")

        analysis = {
            'coin_id': coin_id,
            'timestamp': datetime.now().isoformat(),
            'latest_price': float(latest['close']),
            'technical_indicators': {
                'rsi': float(latest['rsi_14']) if 'rsi_14' in latest else None,
                'ema_20': float(latest['ema_20']) if 'ema_20' in latest else None,
                'volume_ratio': float(latest['volume_ratio']) if 'volume_ratio' in latest else None,
                'price_above_ema': bool(latest.get('price_above_ema', False))
            },
            'trend_analysis': {
                'price_trend': price_trend,
                'volume_trend': volume_trend,
                'support_level': support,
                'resistance_level': resistance,
                'distance_to_support': (float(latest['close']) - support) / support * 100,
                'distance_to_resistance': (resistance - float(latest['close'])) / float(latest['close']) * 100
            },
            'risk_metrics': {
                'volatility_annual': volatility,
                'volatility_daily': volatility / np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(recent_data['close'].values)
            },
            'market_strength': self._calculate_market_strength(processed_data),
            'sequences': sequences
        }

        return analysis

    def _calculate_trend(self, prices: np.array) -> str:
        """Calculate price trend direction"""
        if len(prices) < 2:
            return 'NEUTRAL'

        # Simple linear regression slope
        x = np.arange(len(prices))
        coefficients = np.polyfit(x, prices, 1)
        slope = coefficients[0]

        # Normalize slope by average price
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price != 0 else 0

        if normalized_slope > 0.01:
            return 'STRONG_UP'
        elif normalized_slope > 0.001:
            return 'UP'
        elif normalized_slope < -0.01:
            return 'STRONG_DOWN'
        elif normalized_slope < -0.001:
            return 'DOWN'
        else:
            return 'NEUTRAL'

    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Find support and resistance levels"""
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # Simple approach: use recent min/max
        support = float(np.min(lows))
        resistance = float(np.max(highs))

        # Refine using closing prices
        recent_closes = closes[-10:]
        if len(recent_closes) > 0:
            support = max(support, float(np.percentile(recent_closes, 20)))
            resistance = min(resistance, float(np.percentile(recent_closes, 80)))

        return support, resistance

    def _calculate_max_drawdown(self, prices: np.array) -> float:
        """Calculate maximum drawdown percentage"""
        cumulative = np.maximum.accumulate(prices)
        drawdown = (prices - cumulative) / cumulative
        return float(np.min(drawdown) * 100)

    def _calculate_market_strength(self, df: pd.DataFrame) -> str:
        """Calculate overall market strength score"""
        score = 0
        latest = df.iloc[-1]

        # RSI scoring
        if 'rsi_14' in latest:
            rsi = latest['rsi_14']
            if 30 < rsi < 70:
                score += 1  # Neutral zone
            elif rsi <= 30:
                score += 2  # Oversold (potential buy)
            elif rsi >= 70:
                score -= 1  # Overbought (potential sell)

        # Price vs EMA
        if latest.get('price_above_ema', False):
            score += 1

        # Volume analysis
        if latest.get('volume_ratio', 0) > 1.5:
            score += 2  # High volume
        elif latest.get('volume_ratio', 0) > 1.0:
            score += 1  # Above average volume

        # Recent price change
        if len(df) > 5:
            recent_change = (df.iloc[-1]['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close']
            if recent_change > 0.05:
                score += 2
            elif recent_change > 0:
                score += 1
            elif recent_change < -0.05:
                score -= 2
            else:
                score -= 1

        # Convert score to strength rating
        if score >= 5:
            return 'VERY_STRONG'
        elif score >= 3:
            return 'STRONG'
        elif score >= 1:
            return 'NEUTRAL'
        elif score >= -1:
            return 'WEAK'
        else:
            return 'VERY_WEAK'

    def _get_model_prediction(self, sequences: np.array) -> Dict:
        """Get model prediction for sequences"""
        if self.model is None or sequences is None:
            return None

        try:
            prediction = self.model.predict(sequences[0])
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def analyze_multiple_coins(self,
                               coin_ids: List[str],
                               days: int = 30,
                               save_report: bool = True) -> pd.DataFrame:
        """
        Analyze multiple meme coins and generate comparison report

        Args:
            coin_ids: List of coin IDs to analyze
            days: Historical days for analysis
            save_report: Whether to save report to file

        Returns:
            DataFrame with analysis results
        """
        print(f"\n{'=' * 60}")
        print(f"ANALYZING {len(coin_ids)} MEME COINS")
        print(f"{'=' * 60}")

        results = []

        for i, coin_id in enumerate(coin_ids, 1):
            print(f"\n[{i}/{len(coin_ids)}] Processing {coin_id}...")

            analysis = self.analyze_coin(coin_id, days)

            if 'error' not in analysis:
                # Flatten results for DataFrame
                flat_result = {
                    'coin_id': coin_id,
                    'price': analysis['latest_price'],
                    'rsi': analysis['technical_indicators']['rsi'],
                    'price_trend': analysis['trend_analysis']['price_trend'],
                    'volume_trend': analysis['trend_analysis']['volume_trend'],
                    'support': analysis['trend_analysis']['support_level'],
                    'resistance': analysis['trend_analysis']['resistance_level'],
                    'dist_to_support_%': analysis['trend_analysis']['distance_to_support'],
                    'dist_to_resistance_%': analysis['trend_analysis']['distance_to_resistance'],
                    'volatility_daily_%': analysis['risk_metrics']['volatility_daily'] * 100,
                    'max_drawdown_%': analysis['risk_metrics']['max_drawdown'],
                    'market_strength': analysis['market_strength']
                }

                # Add prediction if available
                if analysis.get('prediction'):
                    flat_result['signal'] = analysis['prediction']['signal']
                    flat_result['confidence'] = analysis['prediction']['confidence']

                results.append(flat_result)
                self.analysis_results[coin_id] = analysis
            else:
                print(f"  Error: {analysis['error']}")

            # Rate limiting
            if i < len(coin_ids):
                time.sleep(1)

        # Create DataFrame
        df_results = pd.DataFrame(results)

        if not df_results.empty:
            # Sort by market strength and RSI
            strength_order = {'VERY_STRONG': 5, 'STRONG': 4, 'NEUTRAL': 3, 'WEAK': 2, 'VERY_WEAK': 1}
            df_results['strength_score'] = df_results['market_strength'].map(strength_order)
            df_results = df_results.sort_values(['strength_score', 'rsi'], ascending=[False, True])
            df_results = df_results.drop('strength_score', axis=1)

            # Save report
            if save_report:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"data/meme_analysis_report_{timestamp}.csv"
                df_results.to_csv(report_file, index=False)
                print(f"\nReport saved to: {report_file}")

            # Display summary
            self._display_analysis_summary(df_results)

            # Find top opportunities
            self._identify_top_opportunities(df_results)

        return df_results

    def _display_analysis_summary(self, df: pd.DataFrame):
        """Display summary of analysis results"""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        # Market strength distribution
        print("\nMarket Strength Distribution:")
        strength_counts = df['market_strength'].value_counts()
        for strength, count in strength_counts.items():
            print(f"  {strength:12}: {count:2} coins")

        # Trend distribution
        print("\nPrice Trend Distribution:")
        trend_counts = df['price_trend'].value_counts()
        for trend, count in trend_counts.items():
            print(f"  {trend:12}: {count:2} coins")

        # Risk metrics summary
        print("\nRisk Metrics Summary:")
        print(f"  Avg Daily Volatility: {df['volatility_daily_%'].mean():.2f}%")
        print(f"  Avg Max Drawdown: {df['max_drawdown_%'].mean():.2f}%")

        # Top 5 strongest coins
        print("\nTop 5 Strongest Coins:")
        top_5 = df.head(5)
        for i, row in enumerate(top_5.iterrows(), 1):
            _, coin = row
            signal_str = f", Signal: {coin.get('signal', 'N/A')}" if 'signal' in coin else ""
            print(f"  {i}. {coin['coin_id']:15} - {coin['market_strength']:12} "
                  f"RSI: {coin['rsi']:.1f}{signal_str}")

    def _identify_top_opportunities(self, df: pd.DataFrame):
        """Identify top trading opportunities"""
        opportunities = []

        for _, coin in df.iterrows():
            score = 0
            reasons = []

            # Check for oversold conditions
            if coin['rsi'] < 35:
                score += 3
                reasons.append(f"Oversold (RSI: {coin['rsi']:.1f})")

            # Check for strong market strength
            if coin['market_strength'] in ['STRONG', 'VERY_STRONG']:
                score += 2
                reasons.append(f"Strong market ({coin['market_strength']})")

            # Check for upward trend
            if 'UP' in coin['price_trend']:
                score += 2
                reasons.append(f"Upward trend ({coin['price_trend']})")

            # Check proximity to support
            if coin['dist_to_support_%'] < 5:
                score += 2
                reasons.append(f"Near support ({coin['dist_to_support_%']:.1f}% away)")

            # Check for buy signal from model
            if coin.get('signal') == 'BUY' and coin.get('confidence', 0) > 0.6:
                score += 3
                reasons.append(f"Model BUY signal ({coin['confidence']:.1%} confidence)")

            if score >= 4:  # Threshold for opportunity
                opportunities.append({
                    'coin': coin['coin_id'],
                    'score': score,
                    'reasons': reasons,
                    'data': coin
                })

        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        self.top_opportunities = opportunities[:5]

        # Display top opportunities
        if opportunities:
            print("\n" + "=" * 80)
            print("TOP TRADING OPPORTUNITIES")
            print("=" * 80)

            for i, opp in enumerate(self.top_opportunities, 1):
                print(f"\n{i}. {opp['coin']} (Score: {opp['score']}/10)")
                print(f"   Price: ${opp['data']['price']:.6f}")
                print(f"   Reasons:")
                for reason in opp['reasons']:
                    print(f"   - {reason}")

    def load_trained_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a pre-trained LSTM model

        Args:
            model_path: Path to model files

        Returns:
            Success status
        """
        path = model_path or self.model_path

        try:
            self.model = MarketAnalysisLSTM()
            success = self.model.load_model(path)

            if success:
                print(f"Model loaded successfully from: {path}")
                return True
            else:
                print("Failed to load model")
                return False

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def run_full_analysis(self,
                          top_n: int = 20,
                          min_volume: float = 100000,
                          min_market_cap: float = 1000000,
                          analysis_days: int = 30) -> pd.DataFrame:
        """
        Run complete meme coin analysis pipeline

        Args:
            top_n: Number of top coins to analyze
            min_volume: Minimum 24h volume filter
            min_market_cap: Minimum market cap filter
            analysis_days: Days of historical data for analysis

        Returns:
            Analysis results DataFrame
        """
        print("\n" + "=" * 80)
        print("MEME COIN MARKET ANALYSIS - FULL PIPELINE")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Get top meme coins
        meme_coins = self.get_top_meme_coins(
            limit=top_n,
            min_volume=min_volume,
            min_market_cap=min_market_cap
        )

        if not meme_coins:
            print("No meme coins found!")
            return pd.DataFrame()

        # Step 2: Load model if available
        model_loaded = self.load_trained_model()
        if not model_loaded:
            print("Warning: Running without ML predictions (model not loaded)")

        # Step 3: Analyze coins
        coin_ids = [coin['id'] for coin in meme_coins[:top_n]]
        results = self.analyze_multiple_coins(coin_ids, analysis_days)

        # Step 4: Generate final report
        if not results.empty:
            self.generate_final_report(results, meme_coins)

        return results

    def generate_final_report(self,
                              analysis_df: pd.DataFrame,
                              coins_info: List[Dict]):
        """Generate comprehensive final report"""

        # Merge with coin info
        info_df = pd.DataFrame(coins_info)
        merged_df = analysis_df.merge(
            info_df[['id', 'symbol', 'name', 'market_cap', 'volume_24h',
                     'price_change_24h', 'price_change_7d']],
            left_on='coin_id', right_on='id', how='left'
        )

        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/meme_comprehensive_report_{timestamp}.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_coins_analyzed': len(analysis_df),
                'strong_opportunities': len(self.top_opportunities),
                'average_rsi': float(analysis_df['rsi'].mean()),
                'bullish_coins': len(analysis_df[analysis_df['price_trend'].str.contains('UP')]),
                'bearish_coins': len(analysis_df[analysis_df['price_trend'].str.contains('DOWN')])
            },
            'top_opportunities': [
                {
                    'coin': opp['coin'],
                    'score': opp['score'],
                    'reasons': opp['reasons'],
                    'current_price': float(opp['data']['price']),
                    'rsi': float(opp['data']['rsi']),
                    'market_strength': opp['data']['market_strength']
                }
                for opp in self.top_opportunities
            ],
            'detailed_analysis': self.analysis_results
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'=' * 80}")
        print(f"Comprehensive report saved to: {report_file}")
        print(f"\nKey Findings:")
        print(f"- Analyzed {len(analysis_df)} meme coins")
        print(f"- Found {len(self.top_opportunities)} strong opportunities")
        print(f"- {report['summary']['bullish_coins']} bullish coins")
        print(f"- {report['summary']['bearish_coins']} bearish coins")

        return report


def main():
    """Main execution function for meme coin analysis"""
    import argparse

    parser = argparse.ArgumentParser(description='Meme Coin Market Analyzer')
    parser.add_argument('--coins', type=int, default=20,
                        help='Number of top meme coins to analyze')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of historical data for analysis')
    parser.add_argument('--min-volume', type=float, default=100000,
                        help='Minimum 24h volume in USD')
    parser.add_argument('--min-mcap', type=float, default=1000000,
                        help='Minimum market cap in USD')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pre-trained model')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = MemeCoinAnalyzer(model_path=args.model_path)

    # Run full analysis
    results = analyzer.run_full_analysis(
        top_n=args.coins,
        min_volume=args.min_volume,
        min_market_cap=args.min_mcap,
        analysis_days=args.days
    )

    if not results.empty:
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Analyzed {len(results)} meme coins")
        print(f"🎯 Found {len(analyzer.top_opportunities)} trading opportunities")
    else:
        print("\n❌ Analysis failed or no data available")

    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
