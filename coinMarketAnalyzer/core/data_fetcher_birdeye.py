"""
Birdeye Data Fetcher Module
Fetches real-time data from Birdeye API for liquidity, smart money, and market data
Used for Entry Timing Analysis (Feature 2)
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import deque
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime, timedelta
from config import birdeye_config, bonding_curve_config, post_trade_review_config, api_config
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, sentry_fallback_warning

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API requests
    Ensures no more than max_requests per second
    """
    
    def __init__(self, max_requests: int = 15, time_window: float = 1.0):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds (default: 1.0 for per-second limiting)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_times = deque()
        self.lock = Lock()
        self.min_interval = time_window / max_requests  # Minimum time between requests
    
    def wait_if_needed(self):
        """
        Wait if necessary to respect rate limit
        Thread-safe implementation
        
        Ensures no more than max_requests are made within the time_window.
        Uses a sliding window approach to track requests.
        """
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than the time window
            while self.request_times and current_time - self.request_times[0] >= self.time_window:
                self.request_times.popleft()
            
            # Check if we're at the limit
            if len(self.request_times) >= self.max_requests:
                # Calculate how long to wait until the oldest request expires
                oldest_request = self.request_times[0]
                wait_time = self.time_window - (current_time - oldest_request) + 0.001  # Add 1ms buffer
                
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.3f}s (at {len(self.request_times)}/{self.max_requests} requests)")
                    time.sleep(wait_time)
                    current_time = time.time()
                    
                    # Clean up again after waiting
                    while self.request_times and current_time - self.request_times[0] >= self.time_window:
                        self.request_times.popleft()
            
            # Record this request timestamp
            self.request_times.append(current_time)


class BirdeyeFetcher:
    """
    Fetches data from Birdeye API for timing analysis
    Follows the same patterns as the existing DataFetcher class
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        chain: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize Birdeye Fetcher
        
        Args:
            api_key: Birdeye API key (falls back to environment variable)
            chain: Blockchain to query (default: solana)
            base_url: Birdeye API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or birdeye_config.API_KEY
        self.chain = chain or birdeye_config.CHAIN
        self.base_url = base_url or birdeye_config.BASE_URL
        # Use tuple for (connect_timeout, read_timeout)
        # Connect timeout stays tight; read timeout uses config for flexibility
        # (OHLCV range fetches for candle fallback may need more time)
        base_timeout = timeout or birdeye_config.TIMEOUT
        self.timeout = (5, max(10, base_timeout))  # 5s connect, config-based read timeout
        
        # Initialize rate limiter (15 requests per second)
        self.rate_limiter = RateLimiter(max_requests=15, time_window=1.0)
        
        # Initialize session with connection pooling and fast failure
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'x-chain': self.chain,
            'Accept': 'application/json'
        })
        
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=birdeye_config.MAX_RETRIES,
            connect=birdeye_config.MAX_RETRIES,
            read=birdeye_config.MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self.circuit_breaker = CircuitBreaker(
            service_name="birdeye",
            failure_threshold=api_config.CIRCUIT_BREAKER_THRESHOLD,
            cooldown_period=api_config.CIRCUIT_BREAKER_COOLDOWN,
        )
        
        if self.api_key:
            logger.info(f"BirdeyeFetcher initialized for chain: {self.chain} (rate limit: 15 rps)")
        else:
            logger.warning("BirdeyeFetcher initialized without API key - calls will fail")
    
    def is_configured(self) -> bool:
        """Check if Birdeye API is properly configured"""
        return bool(self.api_key)
    
    def _api_get(self, url: str, params: dict) -> requests.Response:
        """Execute a GET request through the circuit breaker and rate limiter.

        Raises ``CircuitBreakerOpen`` if the breaker is open, or propagates
        the underlying request exception after recording the failure.
        """
        self.circuit_breaker.check()
        self.rate_limiter.wait_if_needed()
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            self.circuit_breaker.record_success()
            return response
        except Exception as e:
            self.circuit_breaker.record_failure(e)
            raise
    
    def fetch_price_with_liquidity(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch token price with liquidity information
        
        Args:
            token_address: Token mint address
            
        Returns:
            Dict with price, liquidity, priceChange24h or None on error
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping price fetch for {token_address[:8]}...")
            return None
        
        url = f"{self.base_url}/defi/price"
        params = {
            'address': token_address,
            'include_liquidity': 'true'
        }
        
        try:
            response = self._api_get(url, params)
            data = response.json()
            
            if not data.get('success', True):
                logger.warning(f"Birdeye API returned error for price: {data.get('message', 'Unknown error')}")
                return None
            
            result_data = data.get('data', {})
            
            result = {
                'price': float(result_data.get('value', 0) or 0),
                'liquidity': float(result_data.get('liquidity', 0) or 0),
                'priceChange24h': float(result_data.get('priceChange24h', 0) or 0)
            }
            
            logger.debug(f"Fetched price for {token_address[:8]}...: ${result['price']:.8f}, liq: ${result['liquidity']:,.0f}")
            return result
            
        except CircuitBreakerOpen:
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching price for {token_address[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited fetching price for {token_address[:8]}...")
            else:
                logger.error(f"HTTP error fetching price for {token_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching price for {token_address[:8]}...: {e}")
            return None
    
    def fetch_historical_price(
        self, 
        token_address: str, 
        timestamp: datetime
    ) -> Optional[float]:
        """
        Fetch historical price at a specific timestamp
        
        Args:
            token_address: Token mint address
            timestamp: Datetime object for the historical price
            
        Returns:
            Price as float or None on error
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping historical price fetch for {token_address[:8]}...")
            return None
        
        url = f"{self.base_url}/defi/history_price"
        params = {
            'address': token_address,
            'timestamp': int(timestamp.timestamp())
        }
        
        try:
            response = self._api_get(url, params)
            data = response.json()
            
            if not data.get('success', True):
                logger.debug(f"Birdeye API returned error for historical price: {data.get('message', 'Unknown error')}")
                return None
            
            result_data = data.get('data', {})
            price = (
                float(result_data.get('value', 0) or 0) or
                float(result_data.get('price', 0) or 0) or
                float(result_data.get('usdPrice', 0) or 0) or
                0.0
            )
            
            if price > 0:
                logger.debug(f"Fetched historical price for {token_address[:8]}... at {timestamp}: ${price:.8f}")
                return price
            else:
                logger.debug(f"No valid price data returned for {token_address[:8]}... at {timestamp}")
                return None
            
        except CircuitBreakerOpen:
            return None
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout fetching historical price for {token_address[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited fetching historical price for {token_address[:8]}...")
            elif e.response.status_code == 404:
                logger.debug(f"Historical price not available for {token_address[:8]}... at {timestamp}")
            else:
                logger.debug(f"HTTP error fetching historical price for {token_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching historical price for {token_address[:8]}...: {e}")
            return None
    
    def fetch_ohlcv_range(
        self,
        token_address: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "15m"
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV (candlestick) data for a specific time range
        Used for Post-Trade Review analysis (Feature 4)
        
        Args:
            token_address: Token mint address
            start_time: Start datetime for the range
            end_time: End datetime for the range
            interval: Candle interval (1m, 5m, 15m, 30m, 1H, 4H, 1D)
            
        Returns:
            List of OHLCV candles with timestamp, o, h, l, c, v fields
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping OHLCV fetch for {token_address[:8]}...")
            return []
        
        url = f"{self.base_url}/defi/ohlcv"
        
        # Convert datetime to Unix timestamps (seconds)
        time_from = int(start_time.timestamp())
        time_to = int(end_time.timestamp())
        
        params = {
            'address': token_address,
            'type': interval,
            'time_from': time_from,
            'time_to': time_to
        }
        
        try:
            response = self._api_get(url, params)
            data = response.json()
            
            if not data.get('success', True):
                logger.warning(f"Birdeye API returned error for OHLCV: {data.get('message', 'Unknown error')}")
                return []
            
            items = data.get('data', {}).get('items', [])
            
            result = []
            for candle in items:
                try:
                    unix_time = candle.get('unixTime') or candle.get('timestamp') or candle.get('time')
                    if unix_time:
                        result.append({
                            'unixTime': int(unix_time),
                            'timestamp': datetime.fromtimestamp(int(unix_time)).isoformat(),
                            'o': float(candle.get('o', 0) or candle.get('open', 0) or 0),
                            'h': float(candle.get('h', 0) or candle.get('high', 0) or 0),
                            'l': float(candle.get('l', 0) or candle.get('low', 0) or 0),
                            'c': float(candle.get('c', 0) or candle.get('close', 0) or 0),
                            'v': float(candle.get('v', 0) or candle.get('volume', 0) or 0)
                        })
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing candle data: {e}")
                    continue
            
            result.sort(key=lambda x: x['unixTime'])
            
            logger.debug(f"Fetched {len(result)} OHLCV candles for {token_address[:8]}... "
                        f"({interval}, {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')})")
            return result
            
        except CircuitBreakerOpen:
            return []
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching OHLCV for {token_address[:8]}...")
            return []
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited fetching OHLCV for {token_address[:8]}...")
            else:
                logger.error(f"HTTP error fetching OHLCV for {token_address[:8]}...: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {token_address[:8]}...: {e}")
            return []
    
    def fetch_price_history_for_trade_review(
        self,
        token_address: str,
        trade_time: datetime,
        hours_before: int = 4,
        hours_after: int = 24,
        interval: str = "15m"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch price history before and after a trade for Post-Trade Review
        
        Args:
            token_address: Token mint address
            trade_time: When the trade was executed
            hours_before: Hours of data to fetch before the trade (default: 4)
            hours_after: Hours of data to fetch after the trade (default: 24)
            interval: Candle interval (default: 15m)
            
        Returns:
            Tuple of (price_history_before, price_history_after)
        """
        # Ensure trade_time is timezone-naive for comparison
        if trade_time.tzinfo is not None:
            trade_time = trade_time.replace(tzinfo=None)
        
        # Calculate time ranges
        start_before = trade_time - timedelta(hours=hours_before)
        end_before = trade_time
        
        start_after = trade_time
        end_after = trade_time + timedelta(hours=hours_after)
        
        # Cap end_after to now if it's in the future
        now = datetime.now()
        if end_after > now:
            end_after = now
        
        logger.info(f"Fetching price history for post-trade review: {token_address[:8]}... "
                   f"({hours_before}h before, {hours_after}h after trade)")
        
        # Fetch both ranges (could be parallelized, but keeping simple for reliability)
        price_before = self.fetch_ohlcv_range(
            token_address=token_address,
            start_time=start_before,
            end_time=end_before,
            interval=interval
        )
        
        price_after = self.fetch_ohlcv_range(
            token_address=token_address,
            start_time=start_after,
            end_time=end_after,
            interval=interval
        )
        
        logger.info(f"Fetched {len(price_before)} candles before, {len(price_after)} candles after trade")
        
        return price_before, price_after
    
    def fetch_market_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive market data for a token
        
        Args:
            token_address: Token mint address
            
        Returns:
            Dict with liquidity, marketCap, supply, circulatingSupply or None on error
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping market data fetch for {token_address[:8]}...")
            return None
        
        url = f"{self.base_url}/defi/v3/token/market-data"
        params = {
            'address': token_address
        }
        
        try:
            response = self._api_get(url, params)
            data = response.json()
            
            if not data.get('success', True):
                logger.warning(f"Birdeye API returned error for market data: {data.get('message', 'Unknown error')}")
                return None
            
            result_data = data.get('data', {})
            
            result = {
                'liquidity': float(result_data.get('liquidity', 0) or 0),
                'marketCap': float(result_data.get('marketCap', 0) or result_data.get('mc', 0) or 0),
                'supply': float(result_data.get('supply', 0) or 0),
                'circulatingSupply': float(result_data.get('circulatingSupply', 0) or 0)
            }
            
            logger.debug(f"Fetched market data for {token_address[:8]}...: mcap ${result['marketCap']:,.0f}")
            return result
            
        except CircuitBreakerOpen:
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching market data for {token_address[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited fetching market data for {token_address[:8]}...")
            else:
                logger.error(f"HTTP error fetching market data for {token_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching market data for {token_address[:8]}...: {e}")
            return None
    
    def fetch_top_traders(
        self, 
        token_address: str, 
        time_frame: str = "24h",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch top traders for a token
        
        Args:
            token_address: Token mint address
            time_frame: Time frame for analysis (default: 24h)
            limit: Maximum number of traders to return (default: 10)
            
        Returns:
            List of trader dicts with buy/sell volumes, or empty list on error
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping top traders fetch for {token_address[:8]}...")
            return []
        
        url = f"{self.base_url}/defi/v2/tokens/top_traders"
        params = {
            'address': token_address,
            'time_frame': time_frame,
            'sort_by': 'volume',
            'sort_type': 'desc',
            'limit': limit
        }
        
        try:
            response = self._api_get(url, params)
            data = response.json()
            
            if not data.get('success', True):
                logger.warning(f"Birdeye API returned error for top traders: {data.get('message', 'Unknown error')}")
                return []
            
            traders = data.get('data', {}).get('items', [])
            
            result = []
            for trader in traders[:limit]:
                result.append({
                    'address': trader.get('address', ''),
                    'buyVolume': float(trader.get('buyVolume', 0) or trader.get('buy_volume', 0) or 0),
                    'sellVolume': float(trader.get('sellVolume', 0) or trader.get('sell_volume', 0) or 0),
                    'totalVolume': float(trader.get('volume', 0) or trader.get('totalVolume', 0) or 0),
                    'buyCount': int(trader.get('buyTxs', 0) or trader.get('buy_count', 0) or 0),
                    'sellCount': int(trader.get('sellTxs', 0) or trader.get('sell_count', 0) or 0)
                })
            
            logger.debug(f"Fetched {len(result)} top traders for {token_address[:8]}...")
            return result
            
        except CircuitBreakerOpen:
            return []
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching top traders for {token_address[:8]}...")
            return []
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited fetching top traders for {token_address[:8]}...")
            else:
                logger.error(f"HTTP error fetching top traders for {token_address[:8]}...: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching top traders for {token_address[:8]}...: {e}")
            return []
    
    def fetch_trade_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch aggregated trade data for a token
        
        Args:
            token_address: Token mint address
            
        Returns:
            Dict with buy/sell data for 8h and 24h periods, or None on error
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping trade data fetch for {token_address[:8]}...")
            return None
        
        url = f"{self.base_url}/defi/v3/token/trade-data/single"
        params = {
            'address': token_address
        }
        
        try:
            response = self._api_get(url, params)
            data = response.json()
            
            if not data.get('success', True):
                logger.warning(f"Birdeye API returned error for trade data: {data.get('message', 'Unknown error')}")
                return None
            
            result_data = data.get('data', {})
            # Birdeye API returns volume_buy_24h / volume_sell_24h (token volumes);
            # buy_24h / sell_24h are trade counts. We need volumes for whale metrics.
            buy24h = float(
                result_data.get('volume_buy_24h') or result_data.get('buy24h') or 0
            ) or 0
            sell24h = float(
                result_data.get('volume_sell_24h') or result_data.get('sell24h') or 0
            ) or 0
            buy8h = float(
                result_data.get('volume_buy_8h') or result_data.get('buy8h') or 0
            ) or 0
            sell8h = float(
                result_data.get('volume_sell_8h') or result_data.get('sell8h') or 0
            ) or 0
            result = {
                'buy8h': buy8h,
                'sell8h': sell8h,
                'buyCount8h': int(result_data.get('buyCount8h', 0) or result_data.get('buy8hCount', 0) or 0),
                'sellCount8h': int(result_data.get('sellCount8h', 0) or result_data.get('sell8hCount', 0) or 0),
                'buy24h': buy24h,
                'sell24h': sell24h,
                'buyCount24h': int(result_data.get('buyCount24h', 0) or result_data.get('buy24hCount', 0) or result_data.get('buy_24h', 0) or 0),
                'sellCount24h': int(result_data.get('sellCount24h', 0) or result_data.get('sell24hCount', 0) or result_data.get('sell_24h', 0) or 0),
                'holders': int(result_data.get('holders', 0) or result_data.get('holder', 0) or 0)
            }
            
            logger.debug(f"Fetched trade data for {token_address[:8]}...: 24h buy/sell = {result['buy24h']:.0f}/{result['sell24h']:.0f}")
            return result
            
        except CircuitBreakerOpen:
            return None
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching trade data for {token_address[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited fetching trade data for {token_address[:8]}...")
            else:
                logger.error(f"HTTP error fetching trade data for {token_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching trade data for {token_address[:8]}...: {e}")
            return None
    
    def fetch_token_holders(self, token_address: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch token holder list from Birdeye /defi/v3/token/holder.

        Response fields per holder (from Birdeye docs):
          amount (str), decimals (int), mint (str), owner (str),
          token_account (str), ui_amount (float)

        Args:
            token_address: Token mint address (Solana)
            limit: Maximum number of holders to return (max 100 per page)

        Returns:
            List of holder dicts with wallet, balance, percentage, source
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping holders fetch for {token_address[:8]}...")
            return []

        url = f"{self.base_url}/defi/v3/token/holder"
        params = {
            'address': token_address,
            'offset': 0,
            'limit': min(limit, 100),
        }

        try:
            response = self._api_get(url, params)
            data = response.json()

            if not data.get('success', True):
                logger.warning(f"Birdeye API returned error for holders: {data.get('message', 'Unknown error')}")
                return []

            holders = data.get('data', {}).get('items', [])
            if not holders:
                logger.info(f"No holders returned from Birdeye for {token_address[:8]}...")
                return []

            # Sum up total supply from all returned holders for percentage calc
            total_ui_amount = sum(
                float(h.get('ui_amount', 0) or h.get('uiAmount', 0) or 0)
                for h in holders
            )

            result = []
            for holder in holders[:limit]:
                ui_amount = float(
                    holder.get('ui_amount', 0) or holder.get('uiAmount', 0) or 0
                )
                pct = (ui_amount / total_ui_amount * 100) if total_ui_amount > 0 else 0
                result.append({
                    'wallet': holder.get('owner', ''),
                    'balance': ui_amount,
                    'percentage': round(pct, 4),
                    'source': 'birdeye'
                })

            logger.info(f"Fetched {len(result)} holders from Birdeye for {token_address[:8]}...")
            return result

        except CircuitBreakerOpen:
            return []
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching holders for {token_address[:8]}...")
            return []
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning(f"Rate limited fetching holders for {token_address[:8]}...")
            else:
                logger.error(f"HTTP error fetching holders for {token_address[:8]}...: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching holders for {token_address[:8]}...: {e}")
            return []

    def fetch_holder_distribution(self, token_address: str, top_n: int = 10) -> Optional[float]:
        """
        Fetch top-N holder concentration as % of total supply from Birdeye /holder/v1/distribution.
        Matches Birdeye portal definition (top holders' share of total supply).

        Args:
            token_address: Token mint address (Solana)
            top_n: Number of top holders (default 10)

        Returns:
            Percent 0-100 (top N hold this % of total supply), or None on failure/empty
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping holder distribution for {token_address[:8]}...")
            return None

        url = f"{self.base_url}/holder/v1/distribution"
        params = {
            'token_address': token_address,
            'mode': 'top',
            'top_n': min(max(1, top_n), 10000),
            'include_list': False,
        }

        try:
            response = self._api_get(url, params)
            data = response.json()

            if not data.get('success', True):
                logger.warning(f"Birdeye holder distribution error: {data.get('message', 'Unknown error')}")
                return None

            payload = data.get('data') or {}
            summary = payload.get('summary') or {}
            holders = payload.get('holders') or []

            # Prefer summary.percent_of_supply (segment share); API may return as decimal (e.g. 0.78) or percent
            pct = summary.get('percent_of_supply')
            if pct is not None:
                pct = float(pct)
                if 0 <= pct <= 1:
                    pct = pct * 100
                pct = max(0.0, min(100.0, pct))
                logger.info(f"Birdeye holder distribution for {token_address[:8]}...: top {top_n} hold {pct:.2f}% of supply")
                return pct

            # Fallback: sum top N holders' percent_of_supply
            if holders:
                total_pct = sum(float(h.get('percent_of_supply', 0) or 0) for h in holders[:top_n])
                if 0 <= total_pct <= 1:
                    total_pct = total_pct * 100
                total_pct = max(0.0, min(100.0, total_pct))
                logger.info(f"Birdeye holder distribution for {token_address[:8]}...: top {len(holders[:top_n])} hold {total_pct:.2f}% (from holders list)")
                return total_pct

            logger.info(f"No holder distribution data from Birdeye for {token_address[:8]}...")
            return None

        except CircuitBreakerOpen:
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching holder distribution for {token_address[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning(f"Rate limited fetching holder distribution for {token_address[:8]}...")
            else:
                logger.error(f"HTTP error fetching holder distribution for {token_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching holder distribution for {token_address[:8]}...: {e}")
            return None

    def fetch_token_metadata(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch token metadata including category information
        
        Args:
            token_address: Token mint address
            
        Returns:
            Dict with token metadata including category_id, or None on error
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping metadata fetch for {token_address[:8]}...")
            return None
        
        url = f"{self.base_url}{birdeye_config.TOKEN_METADATA_ENDPOINT}"
        params = {
            'address': token_address
        }
        
        try:
            response = self._api_get(url, params)
            data = response.json()
            
            if not data.get('success', True):
                logger.debug(f"Birdeye token metadata not available for {token_address[:8]}...")
                return None
            
            result_data = data.get('data', {})
            
            category_id = (
                result_data.get('category_id') or 
                result_data.get('categoryId') or 
                result_data.get('category') or
                result_data.get('ecosystem')
            )
            
            result = {
                'category_id': category_id,
                'symbol': result_data.get('symbol'),
                'name': result_data.get('name'),
                'decimals': result_data.get('decimals')
            }
            
            if category_id:
                logger.debug(f"Fetched metadata for {token_address[:8]}...: category={category_id}")
            
            return result
            
        except CircuitBreakerOpen:
            return None
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout fetching token metadata for {token_address[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"Token metadata endpoint not available for {token_address[:8]}...")
            elif e.response.status_code == 429:
                logger.warning(f"Rate limited fetching token metadata for {token_address[:8]}...")
            else:
                logger.debug(f"HTTP error fetching token metadata for {token_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching token metadata for {token_address[:8]}...: {e}")
            return None
    
    def fetch_token_creation_info(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch token creation time from Birdeye.

        Endpoint: GET /defi/token_creation_info?address=<mint>
        Response (Solana): data.blockUnixTime (int), data.blockHumanTime (str ISO)

        Returns a dict with blockUnixTime, blockHumanTime and a normalised
        created_at key (ISO string) that existing phase/age consumers already
        understand, or None on any failure.
        """
        if not self.is_configured():
            logger.debug(f"Birdeye not configured, skipping creation info fetch for {token_address[:8]}...")
            return None

        url = f"{self.base_url}/defi/token_creation_info"
        params = {"address": token_address}

        try:
            response = self._api_get(url, params)
            data = response.json()

            if not data.get("success", True):
                logger.debug(f"Birdeye token_creation_info error for {token_address[:8]}...: {data.get('message')}")
                return None

            result_data = data.get("data") or {}
            block_unix = result_data.get("blockUnixTime")
            block_human = result_data.get("blockHumanTime")

            if not block_unix and not block_human:
                logger.debug(f"No creation info returned for {token_address[:8]}...")
                return None

            result: Dict[str, Any] = {}
            if block_unix is not None:
                result["blockUnixTime"] = int(block_unix)
            if block_human is not None:
                result["blockHumanTime"] = str(block_human)
                result["created_at"] = str(block_human)

            logger.info(
                f"Fetched creation info for {token_address[:8]}...: "
                f"blockUnixTime={result.get('blockUnixTime')}, blockHumanTime={result.get('blockHumanTime')}"
            )
            return result

        except CircuitBreakerOpen:
            return None
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout fetching creation info for {token_address[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                logger.warning(f"Rate limited fetching creation info for {token_address[:8]}...")
            elif status == 404:
                logger.debug(f"Creation info not available for {token_address[:8]}...")
            else:
                logger.debug(f"HTTP error fetching creation info for {token_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching creation info for {token_address[:8]}...: {e}")
            return None

    def fetch_timing_data(self, token_address: str) -> Dict[str, Any]:
        """
        Fetch all timing-related data in one call (PARALLEL execution for speed)
        This is a convenience method that calls all required endpoints
        Also includes automatic bonding curve detection
        
        Args:
            token_address: Token mint address
            
        Returns:
            Combined dict with all timing data, handles partial failures
            Includes 'is_on_bonding_curve' boolean for weight adjustment
        """
        if not self.is_configured():
            logger.info(f"Birdeye not configured, returning empty timing data for {token_address[:8]}...")
            sentry_fallback_warning("birdeye", "Birdeye not configured — empty timing data (address pattern only)", extra={"token_address": token_address[:16]})
            # Even without Birdeye, we can detect Pump.fun tokens by address pattern
            is_pumpfun = bonding_curve_config.is_pumpfun_token(token_address)
            is_supported = bonding_curve_config.is_supported_ecosystem_token(
                category_id=None,
                token_address=token_address
            )
            # Without Birdeye data, assume on bonding curve if it's a Pump.fun token
            is_on_curve = is_pumpfun if is_pumpfun else False
            return {
                'configured': False,
                'price_data': None,
                'market_data': None,
                'top_traders': [],
                'trade_data': None,
                'fetch_success': False,
                'is_pumpfun_token': is_pumpfun,
                'is_supported_ecosystem': is_supported,
                'is_on_bonding_curve': is_on_curve,
                'bonding_curve_detection': 'address_pattern_only',
                'category_id': None
            }
        
        logger.info(f"Fetching Birdeye timing data for {token_address[:8]}...")
        
        result = {
            'configured': True,
            'price_data': None,
            'market_data': None,
            'top_traders': [],
            'trade_data': None,
            'fetch_success': False,
            'fetched_at': datetime.now().isoformat()
        }
        
        # Fast-fail if the circuit breaker is open
        if self.circuit_breaker.is_open():
            logger.info(f"Skipping Birdeye API calls — circuit breaker is open")
            sentry_fallback_warning("birdeye", "Birdeye skipped — circuit breaker open", extra={"token_address": token_address[:16]})
            is_pumpfun = bonding_curve_config.is_pumpfun_token(token_address)
            result.update({
                'is_pumpfun_token': is_pumpfun,
                'is_on_bonding_curve': is_pumpfun,
                'bonding_curve_detection': 'circuit_breaker_open',
                'category_id': None
            })
            return result
        
        fetch_results = {}
        max_parallel_timeout = 8
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.fetch_price_with_liquidity, token_address): 'price_data',
                executor.submit(self.fetch_market_data, token_address): 'market_data',
                executor.submit(self.fetch_top_traders, token_address): 'top_traders',
                executor.submit(self.fetch_trade_data, token_address): 'trade_data',
                executor.submit(self.fetch_token_metadata, token_address): 'token_metadata',
            }
            
            try:
                for future in as_completed(futures, timeout=max_parallel_timeout):
                    key = futures[future]
                    try:
                        data = future.result(timeout=0.1)
                        fetch_results[key] = data
                    except Exception as e:
                        logger.warning(f"Error fetching {key} for {token_address[:8]}...: {e}")
                        fetch_results[key] = None
            except TimeoutError:
                logger.warning(f"Birdeye API calls timed out after {max_parallel_timeout}s - using partial results")
                sentry_fallback_warning("birdeye", "Birdeye API calls timed out — using partial results", extra={"token_address": token_address[:16], "timeout_seconds": max_parallel_timeout})
                for future, key in futures.items():
                    if not future.done():
                        future.cancel()
                        fetch_results[key] = None
        
        # Process results
        fetch_count = 0
        
        if fetch_results.get('price_data'):
            result['price_data'] = fetch_results['price_data']
            fetch_count += 1
        
        if fetch_results.get('market_data'):
            result['market_data'] = fetch_results['market_data']
            fetch_count += 1
        
        if fetch_results.get('top_traders'):
            result['top_traders'] = fetch_results['top_traders']
            fetch_count += 1
        
        if fetch_results.get('trade_data'):
            result['trade_data'] = fetch_results['trade_data']
            fetch_count += 1
        
        # Process token metadata for category_id
        category_id = None
        if fetch_results.get('token_metadata'):
            token_metadata = fetch_results['token_metadata']
            category_id = token_metadata.get('category_id')
            result['token_metadata'] = token_metadata
            if category_id:
                fetch_count += 1
        
        result['fetch_success'] = fetch_count > 0
        
        # Detect bonding curve status (now with category support)
        result.update(self.detect_bonding_curve_status(
            token_address, 
            result.get('price_data'), 
            result.get('market_data'),
            category_id=category_id
        ))
        
        # Add category info to result
        if category_id:
            result['category_id'] = category_id
            result['is_supported_ecosystem'] = bonding_curve_config.is_supported_ecosystem_token(
                category_id=category_id,
                token_address=token_address
            )
        
        logger.info(f"Birdeye timing data: {fetch_count}/5 endpoints successful for {token_address[:8]}... "
                   f"(bonding_curve={result.get('is_on_bonding_curve', 'unknown')}, "
                   f"category={category_id or 'unknown'})")
        
        return result
    
    def detect_bonding_curve_status(
        self,
        token_address: str,
        price_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        category_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect if token is still on bonding curve (not yet graduated to DEX)
        Now supports multiple ecosystems: letsbonk-fun, pump-fun, moonshot
        
        Detection logic:
        1. Check token category_id (if available from API)
        2. Check if token address ends with 'pump' (Pump.fun token fallback)
        3. Check liquidity/market cap thresholds
        4. Low liquidity + low market cap = still on bonding curve
        
        Args:
            token_address: Token mint address
            price_data: Price data dict (optional, contains liquidity)
            market_data: Market data dict (optional, contains marketCap)
            category_id: Token category ID (optional, for ecosystem detection)
            
        Returns:
            Dict with bonding curve detection results
        """
        # Check if it's a Pump.fun token (by address pattern)
        is_pumpfun = bonding_curve_config.is_pumpfun_token(token_address)
        
        # Check if token is from supported ecosystem
        is_supported = bonding_curve_config.is_supported_ecosystem_token(
            category_id=category_id,
            token_address=token_address
        )
        
        # Get liquidity and market cap from available data
        liquidity = 0.0
        market_cap = 0.0
        
        if price_data:
            liquidity = float(price_data.get('liquidity', 0) or 0)
        
        if market_data:
            market_cap = float(market_data.get('marketCap', 0) or 0)
            # Some endpoints return liquidity in market_data too
            if liquidity == 0:
                liquidity = float(market_data.get('liquidity', 0) or 0)
        
        # Determine bonding curve status (now with category support)
        is_on_curve = bonding_curve_config.is_on_bonding_curve(
            token_address=token_address,
            liquidity=liquidity,
            market_cap=market_cap,
            category_id=category_id
        )
        
        # Build detection details
        detection_method = []
        if category_id:
            detection_method.append(f"category={category_id}")
        if is_pumpfun:
            detection_method.append("address_suffix=pump")
        if liquidity > 0:
            detection_method.append(f"liquidity=${liquidity:,.0f}")
        if market_cap > 0:
            detection_method.append(f"mcap=${market_cap:,.0f}")
        
        return {
            'is_pumpfun_token': is_pumpfun,
            'is_supported_ecosystem': is_supported,
            'is_on_bonding_curve': is_on_curve,
            'bonding_curve_detection': ", ".join(detection_method) if detection_method else "no_data",
            'detected_liquidity': liquidity,
            'detected_market_cap': market_cap,
            'category_id': category_id
        }
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    print("Birdeye Data Fetcher Module")
    print("=" * 50)
    
    # Test with sample mint
    fetcher = BirdeyeFetcher()
    
    if fetcher.is_configured():
        test_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
        
        print(f"\nTesting data fetch for: {test_mint[:8]}...")
        
        timing_data = fetcher.fetch_timing_data(test_mint)
        
        print(f"\nResults:")
        print(f"  Configured: {timing_data['configured']}")
        print(f"  Fetch Success: {timing_data['fetch_success']}")
        
        if timing_data['price_data']:
            print(f"  Price: ${timing_data['price_data']['price']:.6f}")
            print(f"  Liquidity: ${timing_data['price_data']['liquidity']:,.0f}")
        
        if timing_data['market_data']:
            print(f"  Market Cap: ${timing_data['market_data']['marketCap']:,.0f}")
        
        if timing_data['top_traders']:
            print(f"  Top Traders: {len(timing_data['top_traders'])}")
        
        if timing_data['trade_data']:
            print(f"  24h Volume - Buy: {timing_data['trade_data']['buy24h']:,.0f}, Sell: {timing_data['trade_data']['sell24h']:,.0f}")
    else:
        print("\nBirdeye API not configured. Set BIRDEYE_API_KEY environment variable.")

