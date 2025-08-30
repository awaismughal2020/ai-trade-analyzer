import os
import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import openai
from dataclasses import dataclass
from technical_indicator import TechnicalIndicators, CoinGeckoDataFetcher, TradingStrategyAnalyzer
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class TradeAnalysis:
    token: str
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    actual_profit: float
    price_24h_later: float
    price_48h_later: float
    missed_profit_24h: float
    missed_profit_48h: float
    holding_days: int
    market_context: str
    ai_recommendation: str
    confidence_score: float

    # NEW: Technical Indicators
    rsi_at_buy: float = 50.0
    rsi_at_sell: float = 50.0
    ema_at_buy: float = 0.0
    ema_at_sell: float = 0.0
    ema_uptrend_at_buy: bool = False
    ema_uptrend_at_sell: bool = False
    volume_delta_at_buy: float = 0.0
    volume_delta_at_sell: float = 0.0
    strategy_signal_buy: str = 'NO_DATA'
    strategy_signal_sell: str = 'NO_DATA'
    strategy_compliance_score: int = 0


class DirectTradeAnalyzer:
    def __init__(self, coingecko_api_key: str, openai_api_key: str):
        self.coingecko_key = coingecko_api_key
        self.openai_key = openai_api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"

        openai.api_key = openai_api_key

        # Initialize technical indicators
        self.indicators = TechnicalIndicators()
        self.data_fetcher = CoinGeckoDataFetcher(coingecko_api_key)

        # Token mapping
        self.token_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'MATIC': 'matic-network',
            'SOL': 'solana',
            'AVAX': 'avalanche-2',
            'DOGE': 'dogecoin',
            'SHIB': 'shiba-inu',
        }

    def read_transactions_direct(self, csv_file_path: str) -> List[Dict]:
        """Read transactions directly from CSV and prepare for analysis"""
        df = pd.read_csv(csv_file_path)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Unique tokens: {df['transaction_coin'].unique()}")
        logger.info(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")

        # Match buy/sell pairs
        trade_pairs = []

        for token in df['transaction_coin'].unique():
            if token not in self.token_mapping:
                logger.warning(f"Token {token} not in mapping, skipping")
                continue

            token_data = df[df['transaction_coin'] == token].sort_values('transaction_date')

            purchases = token_data[
                token_data['transaction_type'].str.lower().isin(['purchase', 'buy'])
            ].sort_values('transaction_date').to_dict('records')

            sales = token_data[
                token_data['transaction_type'].str.lower().isin(['sale', 'sell'])
            ].sort_values('transaction_date').to_dict('records')

            logger.info(f"Token {token}: {len(purchases)} buys, {len(sales)} sells")

            # Debug: Show actual transactions
            for i, purchase in enumerate(purchases):
                date_str = pd.to_datetime(purchase['transaction_date']).strftime('%Y-%m-%d %H:%M')
                amount = purchase.get('transaction_amount', 'N/A')
                logger.info(f"  Buy {i + 1}: {date_str} - Amount: {amount}")
            for i, sale in enumerate(sales):
                date_str = pd.to_datetime(sale['transaction_date']).strftime('%Y-%m-%d %H:%M')
                amount = sale.get('transaction_amount', 'N/A')
                logger.info(f"  Sell {i + 1}: {date_str} - Amount: {amount}")

            # Improved FIFO matching
            used_purchases = set()
            used_sales = set()

            for i, sale in enumerate(sales):
                if i in used_sales:
                    continue

                sale_date = pd.to_datetime(sale['transaction_date'])
                best_match = None
                best_match_idx = None

                for j, purchase in enumerate(purchases):
                    if j in used_purchases:
                        continue

                    purchase_date = pd.to_datetime(purchase['transaction_date'])
                    if purchase_date < sale_date:
                        if best_match is None or purchase_date < pd.to_datetime(best_match['transaction_date']):
                            best_match = purchase
                            best_match_idx = j

                if best_match is not None:
                    used_purchases.add(best_match_idx)
                    used_sales.add(i)

                    buy_date = pd.to_datetime(best_match['transaction_date'])
                    sell_date = pd.to_datetime(sale['transaction_date'])
                    holding_days = (sell_date - buy_date).days

                    logger.info(
                        f"  Matched: Buy {buy_date.strftime('%Y-%m-%d %H:%M')} -> Sell {sell_date.strftime('%Y-%m-%d %H:%M')} ({holding_days} days)")

                    if holding_days >= 0:
                        trade_id = f"{token}_{buy_date.strftime('%Y%m%d%H%M')}_{sell_date.strftime('%Y%m%d%H%M')}"

                        trade_pair = {
                            'buy': best_match,
                            'sell': sale,
                            'token': token,
                            'coingecko_id': self.token_mapping[token],
                            'trade_id': trade_id
                        }

                        # Check for duplicates
                        duplicate = False
                        for existing in trade_pairs:
                            if existing.get('trade_id') == trade_id:
                                duplicate = True
                                break

                        if not duplicate:
                            trade_pairs.append(trade_pair)
                        else:
                            logger.warning(f"  Skipping duplicate trade: {trade_id}")
                    else:
                        logger.warning(f"  Invalid trade: negative holding period ({holding_days} days)")
                else:
                    logger.warning(f"  No valid purchase found for sell on {sale_date.strftime('%Y-%m-%d %H:%M')}")

        logger.info(f"Created {len(trade_pairs)} valid unique trade pairs")
        return trade_pairs

    def calculate_technical_indicators(self, coin_id: str, buy_date: datetime, sell_date: datetime) -> Dict:
        """Calculate technical indicators for the trade period"""
        try:
            # Get market data covering the trade period plus buffer for indicators
            buffer_start = buy_date - timedelta(days=30)
            end_with_buffer = sell_date + timedelta(days=3)
            days_needed = (end_with_buffer - buffer_start).days
            days_needed = min(days_needed, 365)  # CoinGecko limit

            market_data = self.data_fetcher.get_hourly_data(coin_id, days_needed)

            if not market_data['prices'] or len(market_data['prices']) < 20:
                return self._default_indicators()

            timestamps = market_data['timestamps']
            prices = market_data['prices']
            volumes = market_data['volumes']

            # Calculate indicators
            rsi_values = self.indicators.calculate_rsi(prices)
            ema_values = self.indicators.calculate_ema(prices)

            # Find closest indices to trade dates
            buy_index = self._find_closest_index(timestamps, buy_date)
            sell_index = self._find_closest_index(timestamps, sell_date)

            if buy_index == -1 or sell_index == -1:
                return self._default_indicators()

            # Get indicator values at trade times
            rsi_at_buy = rsi_values[buy_index] if buy_index < len(rsi_values) else 50
            rsi_at_sell = rsi_values[sell_index] if sell_index < len(rsi_values) else 50

            ema_at_buy = ema_values[buy_index] if buy_index < len(ema_values) else prices[buy_index]
            ema_at_sell = ema_values[sell_index] if sell_index < len(ema_values) else prices[sell_index]

            ema_uptrend_at_buy = self.indicators.check_ema_uptrend(ema_values[:buy_index + 1])
            ema_uptrend_at_sell = self.indicators.check_ema_uptrend(ema_values[:sell_index + 1])

            # Volume delta (simplified as total volume)
            volume_delta_at_buy = volumes[buy_index] if buy_index < len(volumes) else 0
            volume_delta_at_sell = volumes[sell_index] if sell_index < len(volumes) else 0

            # Apply 3-step strategy
            buy_signal = self.indicators.apply_3_step_strategy(
                rsi_at_buy, ema_uptrend_at_buy, volume_delta_at_buy
            )
            sell_signal = self.indicators.apply_3_step_strategy(
                rsi_at_sell, ema_uptrend_at_sell, volume_delta_at_sell
            )

            return {
                'rsi_at_buy': rsi_at_buy,
                'rsi_at_sell': rsi_at_sell,
                'ema_at_buy': ema_at_buy,
                'ema_at_sell': ema_at_sell,
                'ema_uptrend_at_buy': ema_uptrend_at_buy,
                'ema_uptrend_at_sell': ema_uptrend_at_sell,
                'volume_delta_at_buy': volume_delta_at_buy,
                'volume_delta_at_sell': volume_delta_at_sell,
                'strategy_signal_buy': buy_signal['signal'],
                'strategy_signal_sell': sell_signal['signal'],
                'strategy_compliance_score': buy_signal['strength'] + sell_signal['strength']
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators for {coin_id}: {e}")
            return self._default_indicators()

    def _find_closest_index(self, timestamps: List[datetime], target_date: datetime) -> int:
        """Find closest timestamp index with timezone handling"""
        if not timestamps:
            return -1

        # Convert target_date to naive datetime if it's timezone-aware
        if target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)

        min_diff = float('inf')
        closest_index = -1

        for i, timestamp in enumerate(timestamps):
            # Ensure timestamp is also naive
            if timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            diff = abs((timestamp - target_date).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_index = i

        return closest_index

    def _default_indicators(self) -> Dict:
        """Return default values when indicators can't be calculated"""
        return {
            'rsi_at_buy': 50.0,
            'rsi_at_sell': 50.0,
            'ema_at_buy': 0.0,
            'ema_at_sell': 0.0,
            'ema_uptrend_at_buy': False,
            'ema_uptrend_at_sell': False,
            'volume_delta_at_buy': 0.0,
            'volume_delta_at_sell': 0.0,
            'strategy_signal_buy': 'NO_DATA',
            'strategy_signal_sell': 'NO_DATA',
            'strategy_compliance_score': 0
        }

    def get_price_on_date(self, token_id: str, target_date: datetime.date) -> Optional[float]:
        """Get price for specific date using market chart range API"""
        try:
            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = datetime.combine(target_date, datetime.max.time())

            start_timestamp = int(start_dt.timestamp())
            end_timestamp = int(end_dt.timestamp())

            url = f"{self.base_url}/coins/{token_id}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_timestamp,
                'to': end_timestamp
            }
            headers = {'x-cg-pro-api-key': self.coingecko_key}

            response = requests.get(url, params=params, headers=headers)
            time.sleep(0.2)

            if response.status_code == 200:
                data = response.json()
                if data.get('prices') and len(data['prices']) > 0:
                    timestamp, price = data['prices'][0]
                    logger.info(f"Got price for {token_id} on {target_date}: ${price:.4f}")
                    return price
                else:
                    logger.warning(f"No price data for {token_id} on {target_date}")
            else:
                logger.error(f"API error for {token_id} on {target_date}: {response.status_code}")

        except Exception as e:
            logger.error(f"Error getting price for {token_id} on {target_date}: {e}")

        return None

    def create_trade_analysis_direct(self, trade_pair: Dict) -> Optional[TradeAnalysis]:
        """Create enhanced trade analysis with technical indicators"""
        try:
            buy_tx = trade_pair['buy']
            sell_tx = trade_pair['sell']
            token = trade_pair['token']
            coingecko_id = trade_pair['coingecko_id']

            # Extract dates
            buy_date = pd.to_datetime(buy_tx['transaction_date']).to_pydatetime()
            sell_date = pd.to_datetime(sell_tx['transaction_date']).to_pydatetime()

            # Extract amounts
            buy_amount = None
            sell_amount = None

            amount_columns = ['transaction_amount', 'amount', 'quantity', 'qty']
            for col in amount_columns:
                if col in buy_tx and buy_tx[col] is not None:
                    buy_amount = abs(float(buy_tx[col]))
                    break

            for col in amount_columns:
                if col in sell_tx and sell_tx[col] is not None:
                    sell_amount = abs(float(sell_tx[col]))
                    break

            if buy_amount is None or sell_amount is None:
                logger.error(f"Could not extract amounts for {token} trade")
                return None

            trade_amount = min(buy_amount, sell_amount)

            logger.info(f"Analyzing {token}: {buy_date.date()} -> {sell_date.date()}")

            # Get prices
            buy_price = self.get_price_on_date(coingecko_id, buy_date.date())
            sell_price = self.get_price_on_date(coingecko_id, sell_date.date())
            price_24h_later = self.get_price_on_date(coingecko_id, sell_date.date() + timedelta(days=1))
            price_48h_later = self.get_price_on_date(coingecko_id, sell_date.date() + timedelta(days=2))

            if not buy_price or not sell_price:
                logger.error(f"Could not get prices for {token}")
                return None

            # Calculate profit metrics
            actual_profit = (sell_price - buy_price) * trade_amount
            holding_days = (sell_date.date() - buy_date.date()).days

            missed_24h = 0
            missed_48h = 0
            if price_24h_later:
                profit_24h = (price_24h_later - buy_price) * trade_amount
                missed_24h = profit_24h - actual_profit

            if price_48h_later:
                profit_48h = (price_48h_later - buy_price) * trade_amount
                missed_48h = profit_48h - actual_profit

            # NEW: Calculate technical indicators
            tech_indicators = self.calculate_technical_indicators(coingecko_id, buy_date, sell_date)

            # Enhanced AI recommendation
            ai_recommendation = self.generate_enhanced_recommendation(
                token, buy_price, sell_price, price_24h_later, price_48h_later,
                holding_days, tech_indicators
            )

            market_context = f"Trade over {holding_days} days with technical analysis"

            return TradeAnalysis(
                token=token,
                buy_date=buy_date.strftime('%Y-%m-%d'),
                sell_date=sell_date.strftime('%Y-%m-%d'),
                buy_price=buy_price,
                sell_price=sell_price,
                actual_profit=actual_profit,
                price_24h_later=price_24h_later or 0,
                price_48h_later=price_48h_later or 0,
                missed_profit_24h=missed_24h,
                missed_profit_48h=missed_48h,
                holding_days=holding_days,
                market_context=market_context,
                ai_recommendation=ai_recommendation,
                confidence_score=1.0,

                # Technical Indicators
                rsi_at_buy=tech_indicators['rsi_at_buy'],
                rsi_at_sell=tech_indicators['rsi_at_sell'],
                ema_at_buy=tech_indicators['ema_at_buy'],
                ema_at_sell=tech_indicators['ema_at_sell'],
                ema_uptrend_at_buy=tech_indicators['ema_uptrend_at_buy'],
                ema_uptrend_at_sell=tech_indicators['ema_uptrend_at_sell'],
                volume_delta_at_buy=tech_indicators['volume_delta_at_buy'],
                volume_delta_at_sell=tech_indicators['volume_delta_at_sell'],
                strategy_signal_buy=tech_indicators['strategy_signal_buy'],
                strategy_signal_sell=tech_indicators['strategy_signal_sell'],
                strategy_compliance_score=tech_indicators['strategy_compliance_score']
            )

        except Exception as e:
            logger.error(f"Error creating trade analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_enhanced_recommendation(self, token: str, buy_price: float, sell_price: float,
                                         price_24h: float, price_48h: float, holding_days: int,
                                         indicators: Dict) -> str:
        """Generate AI recommendation including technical analysis"""

        profit_pct = ((sell_price - buy_price) / buy_price) * 100
        recommendations = []

        # Buy timing analysis
        rsi_buy = indicators['rsi_at_buy']
        if indicators['strategy_signal_buy'] == 'STRONG_BUY':
            recommendations.append(f"Excellent buy timing - RSI was {rsi_buy:.1f} (oversold) with EMA uptrend.")
        elif indicators['strategy_signal_buy'] == 'WEAK_BUY':
            recommendations.append(f"Fair buy timing - RSI oversold ({rsi_buy:.1f}) but volume concerns.")
        elif indicators['strategy_signal_buy'] == 'NO_SIGNAL':
            recommendations.append(f"Poor buy timing - no EMA uptrend when you bought.")
        elif rsi_buy > 70:
            recommendations.append(f"Poor buy timing - RSI was {rsi_buy:.1f} (overbought).")
        else:
            recommendations.append(f"Neutral buy timing - RSI was {rsi_buy:.1f}.")

        # Sell timing analysis
        rsi_sell = indicators['rsi_at_sell']
        if indicators['strategy_signal_sell'] == 'STRONG_SELL':
            recommendations.append(f"Good exit timing - RSI was {rsi_sell:.1f} (overbought).")
        elif rsi_sell < 30:
            recommendations.append(f"Early exit - RSI was only {rsi_sell:.1f}, could have held longer.")

        # Strategy compliance
        score = indicators['strategy_compliance_score']
        if score >= 5:
            recommendations.append("Excellent strategy compliance - well-executed trade.")
        elif score >= 3:
            recommendations.append("Good strategy compliance with minor improvements needed.")
        elif score >= 1:
            recommendations.append("Partial strategy compliance - focus on better entry/exit timing.")
        else:
            recommendations.append("Poor strategy compliance - wait for proper technical setups.")

        return " ".join(recommendations)

    def analyze_individual_trades(self, analyses: List[TradeAnalysis], max_trades: int = 5) -> None:
        """Enhanced display with technical indicators"""
        if not analyses:
            print("No trades available for analysis.")
            return

        trades_to_analyze = analyses[:max_trades]

        print(f"\n{'=' * 80}")
        print(f"ENHANCED TRADE ANALYSIS WITH TECHNICAL INDICATORS - TOP {len(trades_to_analyze)} TRADES")
        print(f"{'=' * 80}")

        for i, trade in enumerate(trades_to_analyze, 1):
            print(f"\n{'-' * 60}")
            print(f"TRADE #{i} - {trade.token}")
            print(f"{'-' * 60}")

            # Basic Info
            print(f"BASIC INFO:")
            print(f"   Buy Date: {trade.buy_date}")
            print(f"   Sell Date: {trade.sell_date}")
            print(f"   Holding Days: {trade.holding_days}")

            price_change_pct = ((trade.sell_price - trade.buy_price) / trade.buy_price) * 100
            profit_status = "PROFIT" if trade.actual_profit > 0 else "LOSS" if trade.actual_profit < 0 else "BREAKEVEN"
            print(f"   Result: {profit_status} ${trade.actual_profit:.2f} ({price_change_pct:+.2f}%)")

            # Technical Indicators
            print(f"\nTECHNICAL INDICATORS:")
            rsi_buy_label = "Oversold" if trade.rsi_at_buy < 30 else "Overbought" if trade.rsi_at_buy > 70 else "Neutral"
            rsi_sell_label = "Oversold" if trade.rsi_at_sell < 30 else "Overbought" if trade.rsi_at_sell > 70 else "Neutral"

            print(f"   RSI at Buy: {trade.rsi_at_buy:.1f} ({rsi_buy_label})")
            print(f"   RSI at Sell: {trade.rsi_at_sell:.1f} ({rsi_sell_label})")
            print(f"   EMA at Buy: ${trade.ema_at_buy:.4f} ({'Uptrend' if trade.ema_uptrend_at_buy else 'Downtrend'})")
            print(
                f"   EMA at Sell: ${trade.ema_at_sell:.4f} ({'Uptrend' if trade.ema_uptrend_at_sell else 'Downtrend'})")

            if trade.volume_delta_at_buy != 0:
                print(f"   Volume Delta at Buy: {trade.volume_delta_at_buy:,.0f}")
            if trade.volume_delta_at_sell != 0:
                print(f"   Volume Delta at Sell: {trade.volume_delta_at_sell:,.0f}")

            # Strategy Analysis
            print(f"\nSTRATEGY ANALYSIS:")
            print(f"   Buy Signal: {trade.strategy_signal_buy}")
            print(f"   Sell Signal: {trade.strategy_signal_sell}")
            print(f"   Strategy Score: {trade.strategy_compliance_score}/6")

            # Missed opportunities
            if trade.missed_profit_24h > 0:
                print(f"   Missed Profit (24h): ${trade.missed_profit_24h:.2f}")
            elif trade.missed_profit_24h < 0:
                print(f"   Avoided Loss (24h): ${abs(trade.missed_profit_24h):.2f}")

            # AI Recommendation
            print(f"\nAI RECOMMENDATION:")
            print(f"   {trade.ai_recommendation}")

            if i < len(trades_to_analyze):
                input(f"\nPress Enter to view Trade #{i + 1}...")

    # Keep existing methods for backward compatibility
    def generate_simple_recommendation(self, token: str, buy_price: float, sell_price: float,
                                       price_24h: float, price_48h: float, holding_days: int) -> str:
        """Generate simple recommendation (backward compatibility)"""
        profit_pct = ((sell_price - buy_price) / buy_price) * 100

        if profit_pct > 0:
            recommendation = f"Profitable trade (+{profit_pct:.1f}%). "
        else:
            recommendation = f"Loss trade ({profit_pct:.1f}%). "

        if price_24h and price_24h > sell_price:
            missed_pct = ((price_24h - sell_price) / sell_price) * 100
            recommendation += f"Could have gained {missed_pct:.1f}% more by waiting 24h. "

        if holding_days < 1:
            recommendation += "Consider longer holding periods."
        elif holding_days > 30:
            recommendation += "Long hold - monitor trends closely."
        else:
            recommendation += "Reasonable holding period."

        return recommendation

    def debug_csv_structure(self, csv_file_path: str) -> None:
        """Debug CSV structure"""
        try:
            df = pd.read_csv(csv_file_path)
            print(f"\n{'=' * 50}")
            print("CSV STRUCTURE ANALYSIS")
            print(f"{'=' * 50}")

            print(f"Total rows: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")

            if 'transaction_date' in df.columns:
                df['transaction_date'] = pd.to_datetime(df['transaction_date'])
                print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")

            if 'transaction_coin' in df.columns:
                print(f"Unique tokens: {df['transaction_coin'].unique()}")

            if 'transaction_type' in df.columns:
                print(f"Transaction types: {df['transaction_type'].value_counts().to_dict()}")

        except Exception as e:
            print(f"Error analyzing CSV: {e}")

    def run_direct_analysis(self, csv_file_path: str, max_trades: int = 5, debug: bool = True) -> List[TradeAnalysis]:
        """Main method to run enhanced analysis"""
        logger.info("Starting enhanced trade analysis with technical indicators...")

        if debug:
            self.debug_csv_structure(csv_file_path)

        trade_pairs = self.read_transactions_direct(csv_file_path)

        if not trade_pairs:
            logger.error("No valid trade pairs found")
            return []

        analyses = []
        for i, trade_pair in enumerate(trade_pairs[:max_trades]):
            logger.info(f"Analyzing trade {i + 1}/{min(len(trade_pairs), max_trades)}")
            analysis = self.create_trade_analysis_direct(trade_pair)
            if analysis:
                analyses.append(analysis)

        return analyses


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = DirectTradeAnalyzer(
        coingecko_api_key=os.getenv('COINGECKO_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

    try:
        analyses = analyzer.run_direct_analysis("data/sample_transactions.csv", max_trades=2)

        if analyses:
            print(f"Successfully analyzed {len(analyses)} trades with technical indicators")
            analyzer.analyze_individual_trades(analyses)

            # Show summary
            total_profit = sum(a.actual_profit for a in analyses)
            avg_strategy_score = sum(a.strategy_compliance_score for a in analyses) / len(analyses)
            print(f"\nSUMMARY:")
            print(f"Total Profit: ${total_profit:.2f}")
            print(f"Average Strategy Score: {avg_strategy_score:.1f}/6")
        else:
            print("No trades could be analyzed")

    except Exception as e:
        print(f"Analysis failed: {e}")

