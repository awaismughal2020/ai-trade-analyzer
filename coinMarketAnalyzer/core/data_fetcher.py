"""
Data Fetcher Module v2.1
Fetches real-time data from API endpoints for live trading recommendations
Enhanced with user holdings endpoint for time-series whale tracking
Enhanced with dual holder source support (Feature 2)
Enhanced with rate limiting and exponential backoff (v2.1)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Tuple
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import api_config, meme_config, holder_config, birdeye_config
from core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, sentry_fallback_warning
from core.user_profiling_normalizer import normalize_user_profiling_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiter with Exponential Backoff
# =============================================================================

class RateLimiter:
    """
    Thread-safe rate limiter with exponential backoff.
    Prevents overwhelming external APIs with too many requests.

    Circuit-breaking responsibility has been moved to the dedicated
    ``CircuitBreaker`` class in ``core.circuit_breaker``.
    """
    
    def __init__(
        self,
        requests_per_second: int = None,
        max_concurrent: int = None,
        min_interval: float = None,
        backoff_factor: float = None,
        max_backoff: float = None,
    ):
        self.requests_per_second = requests_per_second or api_config.REQUESTS_PER_SECOND
        self.max_concurrent = max_concurrent or api_config.MAX_CONCURRENT_REQUESTS
        self.min_interval = min_interval or api_config.MIN_REQUEST_INTERVAL
        self.backoff_factor = backoff_factor or api_config.BACKOFF_FACTOR
        self.max_backoff = max_backoff or api_config.MAX_BACKOFF
        
        # Thread safety
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(self.max_concurrent)
        
        # State tracking
        self._last_request_time = 0
        self._consecutive_failures = 0
        self._request_count = 0
        self._window_start = time.time()
        
        logger.info(f"RateLimiter initialized: {self.requests_per_second} req/s, "
                   f"{self.max_concurrent} concurrent, {self.min_interval}s interval")
    
    def acquire(self) -> bool:
        """Acquire permission to make a request. Blocks if rate limit exceeded."""
        # Acquire semaphore (blocks if max concurrent reached)
        self._semaphore.acquire()
        
        with self._lock:
            now = time.time()
            
            # Reset window if needed
            if now - self._window_start >= 1.0:
                self._request_count = 0
                self._window_start = now
            
            # Check requests per second limit
            if self._request_count >= self.requests_per_second:
                sleep_time = 1.0 - (now - self._window_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._request_count = 0
                self._window_start = time.time()
            
            # Enforce minimum interval
            elapsed_since_last = now - self._last_request_time
            if elapsed_since_last < self.min_interval:
                time.sleep(self.min_interval - elapsed_since_last)
            
            self._last_request_time = time.time()
            self._request_count += 1
        
        return True
    
    def release(self):
        """Release the semaphore after request completion."""
        self._semaphore.release()
    
    def record_success(self):
        """Record a successful request, reset failure counter."""
        with self._lock:
            self._consecutive_failures = 0
    
    def record_failure(self) -> float:
        """Record a failed request and return backoff time."""
        with self._lock:
            self._consecutive_failures += 1
            
            backoff = min(
                self.backoff_factor ** self._consecutive_failures,
                self.max_backoff
            )
            jitter = random.uniform(0, api_config.JITTER_MAX)
            return backoff + jitter
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        with self._lock:
            return {
                "requests_this_window": self._request_count,
                "consecutive_failures": self._consecutive_failures,
                "requests_per_second": self.requests_per_second,
                "max_concurrent": self.max_concurrent
            }


def create_retry_session(
    retries: int = None,
    backoff_factor: float = None,
    status_forcelist: tuple = (429, 500, 502, 503, 504)
) -> requests.Session:
    """
    Create a requests session with connection pooling.
    
    Session-level retries are DISABLED to avoid double-retry stacking
    with _make_request's application-level retry loop. All retry logic
    (including backoff and circuit breaking) is handled by _make_request.
    
    Args:
        retries: Unused (kept for backward compatibility)
        backoff_factor: Unused (kept for backward compatibility)
        status_forcelist: Unused (kept for backward compatibility)
        
    Returns:
        Configured requests Session
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=0,
        backoff_factor=0,
        status_forcelist=(),
        allowed_methods=["GET", "POST"],
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=api_config.MAX_CONCURRENT_REQUESTS,
        pool_maxsize=api_config.MAX_CONCURRENT_REQUESTS
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# Per-purpose rate limiter instances (isolated circuit breakers)
_rate_limiters: Dict[str, RateLimiter] = {}

def get_rate_limiter(name: str = "default") -> RateLimiter:
    """Get or create a named rate limiter instance.
    
    Each name gets its own independent RateLimiter with its own circuit breaker.
    This prevents failures in one area (e.g. predict) from blocking others (e.g. users).
    
    Args:
        name: Identifier for the rate limiter (e.g. 'predict', 'users', 'training')
    """
    global _rate_limiters
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter()
        logger.info(f"Created rate limiter: '{name}'")
    return _rate_limiters[name]


class DataFetcher:
    """
    Fetches live data from API endpoints with rate limiting and retry logic.
    """
    
    def __init__(self, base_url: str = None, use_rate_limiter: bool = True, limiter_name: str = "default"):
        """
        Initialize Data Fetcher
        
        Args:
            base_url: Base URL for API endpoints (uses config if not provided)
            use_rate_limiter: Whether to use rate limiting (default True)
            limiter_name: Name for the rate limiter instance (isolates circuit breakers)
        """
        if base_url is None:
            from config import meme_config
            base_url = meme_config.BASE_URL
        
        self.base_url = base_url
        self.session = create_retry_session()
        self.rate_limiter = get_rate_limiter(limiter_name) if use_rate_limiter else None
        self._use_rate_limiter = use_rate_limiter
        self.circuit_breaker = CircuitBreaker(
            service_name="api_server",
            failure_threshold=api_config.CIRCUIT_BREAKER_THRESHOLD,
            cooldown_period=api_config.CIRCUIT_BREAKER_COOLDOWN,
        )
        logger.info(f"Data Fetcher initialized with base URL: {base_url}, rate_limiter={use_rate_limiter}, limiter='{limiter_name}'")
    
    def is_api_reachable(self, timeout: int = 3) -> bool:
        """Quick connectivity check to the internal API with a short timeout.
        
        Used as a fast-fail probe before launching expensive parallel fetches.
        If the API is unreachable, callers can skip directly to fallback logic
        instead of waiting for multiple requests to time out.
        
        Args:
            timeout: Maximum seconds to wait for a response (default 3s)
            
        Returns:
            True if the API responded, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=timeout)
            return response.status_code < 500
        except Exception:
            return False
    
    def _make_request(
        self, 
        url: str, 
        params: dict = None, 
        timeout: int = None,
        method: str = "GET"
    ) -> Optional[requests.Response]:
        """
        Make an HTTP request with rate limiting, circuit breaker, and retry logic.
        
        Args:
            url: Full URL to request
            params: Query parameters
            timeout: Request timeout (uses api_config default if not provided)
            method: HTTP method (GET or POST)
            
        Returns:
            Response object or None on failure
        """
        # Fast-fail if circuit breaker is open
        try:
            self.circuit_breaker.check()
        except CircuitBreakerOpen:
            logger.warning(f"Request blocked by circuit breaker (api_server): {url}")
            return None

        timeout = timeout or api_config.REQUEST_TIMEOUT
        max_retries = api_config.MAX_RETRIES
        
        for attempt in range(max_retries):
            # Acquire rate limiter permission
            if self.rate_limiter:
                self.rate_limiter.acquire()
            
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=timeout)
                else:
                    response = self.session.post(url, json=params, timeout=timeout)
                
                # Release semaphore
                if self.rate_limiter:
                    self.rate_limiter.release()
                
                # Check for rate limit response
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limited (429). Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    self.circuit_breaker.record_failure()
                    if self.rate_limiter:
                        self.rate_limiter.record_failure()
                    continue
                
                # Success
                if response.status_code == 200:
                    self.circuit_breaker.record_success()
                    if self.rate_limiter:
                        self.rate_limiter.record_success()
                    return response
                
                # Server error - retry with backoff
                if response.status_code >= 500:
                    self.circuit_breaker.record_failure()
                    if self.rate_limiter:
                        backoff = self.rate_limiter.record_failure()
                    else:
                        backoff = api_config.RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Server error {response.status_code}. Retry in {backoff:.1f}s...")
                    time.sleep(backoff)
                    continue
                
                # Client error - don't retry
                response.raise_for_status()
                return response
                
            except requests.exceptions.ConnectionError as e:
                if self.rate_limiter:
                    self.rate_limiter.release()
                    backoff = self.rate_limiter.record_failure()
                else:
                    backoff = api_config.CONNECTION_COOLDOWN
                
                self.circuit_breaker.record_failure(e)
                logger.error(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Waiting {backoff:.1f}s before retry...")
                    time.sleep(backoff)
                else:
                    logger.error(f"Max retries exceeded for {url}")
                    return None
                    
            except requests.exceptions.Timeout as e:
                if self.rate_limiter:
                    self.rate_limiter.release()
                    backoff = self.rate_limiter.record_failure()
                else:
                    backoff = api_config.RETRY_DELAY * (attempt + 1)
                
                self.circuit_breaker.record_failure(e)
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                else:
                    return None
                    
            except Exception as e:
                if self.rate_limiter:
                    self.rate_limiter.release()
                self.circuit_breaker.record_failure(e)
                logger.error(f"Unexpected error: {e}")
                return None
        
        return None
    
    def fetch_mints_range(self, from_date: str, to_date: str, limit: int = 1000, max_pages: int = 50) -> pd.DataFrame:
        """
        Fetch mints within a date range with full pagination support and rate limiting.
        
        Args:
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            limit: Number of mints per page (default: 1000, max supported by API)
            max_pages: Maximum number of pages to fetch (default: 50, set to None for all)
            
        Returns:
            DataFrame with mint data from all pages
        """
        url = f"{self.base_url}/mint/mints-range"
        all_mints = []
        page = 1
        total_pages = 1  # Will be updated from first response
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        logger.info(f"Fetching mints from {from_date} to {to_date} (limit={limit} per page)...")
        
        while page <= total_pages:
            # Check max_pages limit
            if max_pages and page > max_pages:
                logger.info(f"Reached max_pages limit ({max_pages}), stopping pagination")
                break
            
            # Check for too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping")
                break
            
            params = {
                'from': from_date,
                'to': to_date,
                'page': page,
                'limit': limit
            }
            
            try:
                response = self._make_request(url, params=params, timeout=60)
                
                if response is None:
                    consecutive_errors += 1
                    logger.warning(f"Failed to fetch page {page}, consecutive errors: {consecutive_errors}")
                    time.sleep(api_config.CONNECTION_COOLDOWN)
                    continue
                
                data = response.json()
                consecutive_errors = 0  # Reset on success
                
                # Handle paginated response
                if isinstance(data, dict) and 'mints' in data:
                    mints_data = data.get('mints', [])
                    total_pages = int(data.get('totalPages', 1))
                    total_records = int(data.get('total', 0))
                    
                    if page == 1:
                        logger.info(f"Found {total_records} total mints across {total_pages} pages")
                    
                    if isinstance(mints_data, list) and len(mints_data) > 0:
                        all_mints.extend(mints_data)
                        logger.info(f"Page {page}/{total_pages}: fetched {len(mints_data)} mints (total collected: {len(all_mints)})")
                    else:
                        logger.warning(f"Page {page}: No mints data returned")
                        break
                        
                elif isinstance(data, list):
                    # Direct list response (non-paginated)
                    all_mints.extend(data)
                    logger.info(f"Fetched {len(data)} mints (non-paginated response)")
                    break
                else:
                    # Unknown structure
                    logger.warning(f"Unknown response structure. Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
                    break
                
                page += 1
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error fetching page {page}: {e}")
                time.sleep(api_config.CONNECTION_COOLDOWN)
        
        # Convert to DataFrame
        if all_mints:
            df = pd.DataFrame(all_mints)
            logger.info(f"Successfully fetched {len(df)} mints total")
            return df
        else:
            logger.warning("No mints fetched")
            return pd.DataFrame()
    
    def fetch_trades(self, mint: str, days: int = 90, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch trades for a specific mint with rate limiting.
        
        Args:
            mint: Mint address
            days: Number of days of historical data
            limit: Maximum number of records
            
        Returns:
            DataFrame with trade data
        """
        url = f"{self.base_url}/mint/trades-range"
        params = {
            'mint': mint,
            'days': days,
            'limit': limit
        }
        
        logger.info(f"Fetching trades for {mint[:8]}... (last {days} days)")
        
        try:
            response = self._make_request(url, params=params, timeout=30)
            
            if response is None:
                logger.error(f"Failed to fetch trades for {mint[:8]}...")
                return pd.DataFrame()
            
            data = response.json()
            
            # Handle nested response - extract 'trades' array if it exists
            if isinstance(data, dict) and 'trades' in data:
                trades_data = data['trades']
                df = pd.DataFrame(trades_data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            logger.info(f"Fetched {len(df)} trade records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching trades for {mint[:8]}...: {e}")
            return pd.DataFrame()
    
    def fetch_current_holders(self, mint: str) -> pd.DataFrame:
        """
        Fetch CURRENT on-chain holders with balances from /mint/details.
        This returns actual current token holders, not historical trading activity.
        
        Args:
            mint: Mint address
            
        Returns:
            DataFrame with current holder data including wallet, balance, percentageOfTotalSupply
        """
        url = f"{self.base_url}/mint/details"
        params = {
            'mint': mint,
            'source': 'clickhouse'  # Use ClickHouse for accurate on-chain data
        }
        
        logger.info(f"Fetching current on-chain holders for {mint[:8]}...")
        
        try:
            response = self._make_request(url, params=params, timeout=30)
            
            if response is None:
                logger.error(f"Failed to fetch current holders for {mint[:8]}...")
                return pd.DataFrame()
            
            data = response.json()
            
            # Extract holders list from response
            if isinstance(data, dict) and 'holders' in data and 'list' in data['holders']:
                holders_list = data['holders']['list']
                df = pd.DataFrame(holders_list)
                
                # Rename columns to match expected format
                if 'Wallet' in df.columns:
                    df = df.rename(columns={
                        'Wallet': 'address',
                        'balance': 'lastHolding',
                        'percentageOfTotalSupply': 'percentOfSupply'
                    })
                
                # Convert numeric columns from strings to numbers
                if 'lastHolding' in df.columns:
                    df['lastHolding'] = pd.to_numeric(df['lastHolding'], errors='coerce').fillna(0)
                if 'percentOfSupply' in df.columns:
                    df['percentOfSupply'] = pd.to_numeric(df['percentOfSupply'], errors='coerce').fillna(0)
                if 'ui_amount' in df.columns:
                    df['ui_amount'] = pd.to_numeric(df['ui_amount'], errors='coerce').fillna(0)
                
                # Add metadata
                if len(df) > 0:
                    df['mint'] = mint
                    df['currentHolders'] = data.get('currentHolders', len(df))
                    df['totalSupply'] = data.get('totalSupply', 0)
                    df['whaleCount'] = data.get('whaleCount', 0)
                    df['totalWhalePercentage'] = data.get('totalWhalePercentage', 0)
                
                # CRITICAL: Filter out pool addresses and zero addresses
                # These are contracts (pump.fun bonding curve, Raydium pools, etc), not real holders
                total_holders_before = len(df)
                
                # Filter out pools, zero addresses, and burn addresses
                if 'isPool' in df.columns:
                    df = df[df['isPool'] == False].copy()
                if 'isZeroAddress' in df.columns:
                    df = df[df['isZeroAddress'] == False].copy()
                
                # Also filter known contract addresses
                contract_keywords = ['pool', 'vault', 'pump', 'raydium', '1111111', 'incinerator', 'burn']
                if 'address' in df.columns:
                    mask = df['address'].str.lower().str.contains('|'.join(contract_keywords), na=False)
                    df = df[~mask].copy()
                
                holders_filtered = total_holders_before - len(df)
                if holders_filtered > 0:
                    logger.info(f"Filtered out {holders_filtered} contract/pool addresses (pump.fun, pools, etc)")
                
                logger.info(f"Fetched {len(df)} real holders after filtering (raw={total_holders_before}, currentHolders={data.get('currentHolders')})")
                logger.info(f"Whale stats: {data.get('whaleCount')} whales holding {data.get('totalWhalePercentage', 0):.2f}%")
                return df
            else:
                logger.warning(f"Unexpected response format from /mint/details for {mint[:8]}...")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching current holders for {mint[:8]}...: {e}")
            return pd.DataFrame()
    
    def fetch_holders(self, mint: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch holder data for a specific mint.
        Primary: Birdeye /defi/v3/token/holder (on-chain top holders).
        Fallback: Internal /mint/range-holders (historical trading activity).

        Args:
            mint: Mint / token address
            days: Number of days of historical data (used only by internal fallback)

        Returns:
            DataFrame with holder data
        """
        # --- Primary: Birdeye ---
        if birdeye_config.is_configured():
            try:
                from core.data_fetcher_birdeye import BirdeyeFetcher
                birdeye = BirdeyeFetcher()
                holders_list = birdeye.fetch_token_holders(mint, limit=100)
                if holders_list:
                    df = pd.DataFrame(holders_list)
                    logger.info(f"Fetched {len(df)} holders from Birdeye for {mint[:8]}...")
                    return df
                logger.warning(f"Birdeye returned 0 holders for {mint[:8]}..., falling back to internal API")
                sentry_fallback_warning("birdeye", "Birdeye returned 0 holders, falling back to internal API", extra={"mint": mint[:16]})
            except Exception as e:
                logger.warning(f"Birdeye holder fetch failed for {mint[:8]}...: {e}, falling back to internal API")
                sentry_fallback_warning("birdeye", "Birdeye holder fetch failed, falling back to internal API", extra={"mint": mint[:16], "error": str(e)})

        # --- Fallback: Internal /mint/range-holders ---
        return self._fetch_holders_internal(mint, days)

    def _fetch_holders_internal(self, mint: str, days: int = 90) -> pd.DataFrame:
        """Fetch holder data from internal /mint/range-holders endpoint."""
        url = f"{self.base_url}/mint/range-holders"
        params = {
            'mint': mint,
            'days': days
        }

        logger.info(f"Fetching holders from internal API for {mint[:8]}... (last {days} days)")

        try:
            response = self._make_request(url, params=params, timeout=30)

            if response is None:
                logger.error(f"Failed to fetch holders from internal API for {mint[:8]}...")
                return pd.DataFrame()

            data = response.json()

            if isinstance(data, dict) and 'holders' in data:
                holders_data = data['holders']
                df = pd.DataFrame(holders_data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])

            logger.info(f"Fetched {len(df)} holder records from internal API")
            return df

        except Exception as e:
            logger.error(f"Error fetching holders from internal API for {mint[:8]}...: {e}")
            return pd.DataFrame()
    
    def fetch_candles(self, mint: str, days: int = 90, limit: int = 1000, candle_type: str = "5m") -> pd.DataFrame:
        """
        Fetch candle (OHLCV) data for a specific mint with rate limiting.
        
        Args:
            mint: Mint address
            days: Number of days of historical data
            limit: Maximum number of records
            candle_type: Candle timeframe - "5m", "15m", "1h", "4h", "1d" (default: "5m")
            
        Returns:
            DataFrame with candle data
        """
        url = f"{self.base_url}/mint/range-candles"
        params = {
            'mint': mint,
            'days': days,
            'limit': limit,
            'type': candle_type
        }
        
        logger.info(f"Fetching candles for {mint[:8]}... (last {days} days)")
        
        try:
            response = self._make_request(url, params=params, timeout=30)
            
            if response is None:
                logger.error(f"Failed to fetch candles for {mint[:8]}...")
                return pd.DataFrame()
            
            data = response.json()
            
            # Handle nested response - extract 'bars' array if it exists
            if isinstance(data, dict) and 'bars' in data:
                bars_data = data['bars']
                df = pd.DataFrame(bars_data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
            
            logger.info(f"Fetched {len(df)} candle records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching candles for {mint[:8]}...: {e}")
            return pd.DataFrame()
    
    def fetch_user_holdings(self, mint: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch complete user holdings with transaction history using mint-timeline API.
        This endpoint provides time-series data for accurate whale tracking.
        
        Args:
            mint: Mint address
            limit: Maximum number of records (used to limit processed holders)
            
        Returns:
            DataFrame with user holdings and transaction history
        """
        url = f"{self.base_url}{meme_config.ENDPOINT_USER_HOLDINGS}"
        params = {'mint': mint}
        
        logger.info(f"Fetching user holdings from mint-timeline for {mint[:8]}...")
        
        try:
            response = self._make_request(url, params=params, timeout=api_config.REQUEST_TIMEOUT)
            
            if response is None:
                logger.error(f"Failed to fetch user holdings for {mint[:8]}...")
                return pd.DataFrame()
            
            data = response.json()
            
            if not isinstance(data, dict):
                logger.warning(f"Unexpected response format for {mint[:8]}...")
                return pd.DataFrame()
            
            logger.info(f"API response keys for {mint[:8]}...: {list(data.keys())}")
            
            # Extract holders array
            holders = data.get('holders', [])
            if not holders:
                logger.warning(f"No holder data found in API response for {mint[:8]}...")
                return pd.DataFrame()
            
            # Limit number of holders if specified
            if limit and len(holders) > limit:
                holders = holders[:limit]
            
            # Transform each holder to include derived fields
            transformed_holders = []
            for holder in holders:
                transformed = self._transform_holder_data(holder)
                transformed_holders.append(transformed)
            
            df = pd.DataFrame(transformed_holders)
            
            if len(df) > 0:
                df['mint'] = mint
                logger.info(f"Holders sample for {mint[:8]}...: columns={list(df.columns)}")
                if 'finalHolding' in df.columns:
                    sample_holdings = df['finalHolding'].head(3).tolist()
                    logger.info(f"Sample finalHolding values: {sample_holdings}")
            
            # Store metadata from API response
            on_chain_holders = data.get('onChainHolders') or data.get('totalHolders')
            if on_chain_holders:
                # Handle both integer and array formats for onChainHolders
                if isinstance(on_chain_holders, list):
                    # If it's an array, store the count
                    on_chain_count = len(on_chain_holders)
                    logger.info(f"API reports {on_chain_count} on-chain holders (array) for {mint[:8]}...")
                    if len(df) > 0:
                        df['_api_on_chain_holders'] = on_chain_count
                else:
                    logger.info(f"API reports {on_chain_holders} on-chain holders for {mint[:8]}...")
                    if len(df) > 0:
                        df['_api_on_chain_holders'] = on_chain_holders
            
            # Store mintDetails for token metadata
            mint_details = data.get('mintDetails', {})
            if mint_details and len(df) > 0:
                df['_developer'] = mint_details.get('Developer', '')
                df['_created_at'] = mint_details.get('CreatedAt', '')
                df['_decimals'] = mint_details.get('Decimals', 6)
            
            logger.info(f"Fetched and transformed {len(df)} user holding records")
            return df
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching user holdings for {mint[:8]}...")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching user holdings for {mint[:8]}...: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching user holdings for {mint[:8]}...: {e}")
            return pd.DataFrame()
    
    def _transform_holder_data(self, holder: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform holder data from mint-timeline API to the expected format
        
        Args:
            holder: Raw holder data from API
            
        Returns:
            Transformed holder with derived fields (totalBuys, totalSells, firstTradeAt, lastTradeAt)
        """
        history = holder.get('history', [])
        
        # Calculate totalBuys and totalSells from history
        total_buys = 0
        total_sells = 0
        first_trade_at = None
        last_trade_at = None
        
        if history:
            for txn in history:
                amount = float(txn.get('amount', 0))
                if txn.get('isBuy', False):
                    total_buys += amount
                else:
                    total_sells += amount
            
            # History is ordered with recent first (index 0) and oldest last
            # First trade = last item in history (oldest)
            # Last trade = first item in history (most recent)
            first_trade_at = history[-1].get('timestamp') if history else None
            last_trade_at = history[0].get('timestamp') if history else None
        
        return {
            'signer': holder.get('signer', ''),
            'address': holder.get('signer', ''),  # Alias for compatibility
            'wallet': holder.get('signer', ''),   # Alias for compatibility
            'finalHolding': holder.get('finalHolding', 0),
            'lastHolding': holder.get('finalHolding', 0),  # Alias for compatibility
            'current_holding': holder.get('finalHolding', 0),  # Alias for compatibility
            'totalTrades': holder.get('totalTrades', 0),
            'totalBuys': total_buys,
            'total_buys': total_buys,  # Alias for compatibility
            'totalSells': total_sells,
            'total_sells': total_sells,  # Alias for compatibility
            'firstTradeAt': first_trade_at,
            'first_trade_at': first_trade_at,  # Alias for compatibility
            'lastTradeAt': last_trade_at,
            'last_trade_at': last_trade_at,  # Alias for compatibility
            'history': history,  # Keep full history for detailed analysis
            'transactions': history,  # Alias for compatibility
        }
    
    def fetch_mint_metadata(self, mint: str) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata for a specific mint (creation date, developer, etc.)
        Uses dedicated /mint/mint-metadata endpoint for faster, direct access
        
        Args:
            mint: Mint address
            
        Returns:
            Dictionary with mint metadata (CreatedAt, Developer, Decimals, etc.) or None if not found
        """
        url = f"{self.base_url}{meme_config.ENDPOINT_METADATA}"
        params = {'mint': mint}
        
        logger.debug(f"Fetching metadata for mint {mint[:8]}...")
        
        try:
            response = self.session.get(url, params=params, timeout=api_config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, dict):
                # Check if response has metadata directly or nested
                metadata = (
                    data.get('metadata') or 
                    data.get('mintDetails') or 
                    data.get('details') or 
                    data
                )
                
                # Ensure we have the expected fields
                if metadata and (metadata.get('CreatedAt') or metadata.get('created_at') or metadata.get('createdAt')):
                    logger.debug(f"Fetched metadata for mint {mint[:8]}...")
                    return metadata
                else:
                    logger.debug(f"Metadata response for {mint[:8]}... missing creation date")
                    return metadata if metadata else None
            else:
                logger.warning(f"Unexpected response format for mint metadata: {type(data)}")
                return None
            
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout fetching mint metadata for {mint[:8]}...")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"Mint metadata not found for {mint[:8]}...")
            else:
                logger.debug(f"HTTP error fetching mint metadata for {mint[:8]}...: {e}")
            return None
        except Exception as e:
            logger.debug(f"Error fetching mint metadata for {mint[:8]}...: {e}")
            return None
    
    def standardize_user_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize user holdings data to expected format
        
        Args:
            df: Raw user holdings DataFrame
            
        Returns:
            Standardized DataFrame with wallet, transactions, and current holdings
        """
        if len(df) == 0:
            return df
        
        df = df.copy()
        
        # Column mapping for standardization
        # API returns: signer, totalTrades, finalHolding, history
        column_mapping = {
            # Wallet/Address variations
            'signer': 'wallet',  # API uses 'signer'
            'wallet': 'wallet',
            'address': 'wallet',
            'holder': 'wallet',
            'user': 'wallet',
            'owner': 'wallet',
            # Current balance variations
            'finalHolding': 'current_holding',  # API uses 'finalHolding'
            'currentHolding': 'current_holding',
            'current_holding': 'current_holding',
            'balance': 'current_holding',
            'lastHolding': 'current_holding',
            'holding': 'current_holding',
            # Transaction history variations
            'history': 'transactions',  # API uses 'history'
            'transactions': 'transactions',
            'txns': 'transactions',
            # Total trades
            'totalTrades': 'total_trades',
            # Total buy/sell
            'totalBuys': 'total_buys',
            'total_buys': 'total_buys',
            'totalSells': 'total_sells',
            'total_sells': 'total_sells',
            # First/Last trade times
            'firstTradeAt': 'first_trade_at',
            'first_trade_at': 'first_trade_at',
            'lastTradeAt': 'last_trade_at',
            'last_trade_at': 'last_trade_at',
        }
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df = df.rename(columns={old_col: new_col})
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure required columns exist with defaults
        if 'wallet' not in df.columns:
            # Try to find wallet-like column
            for col in ['address', 'holder', 'user', 'owner']:
                if col in df.columns:
                    df['wallet'] = df[col]
                    break
            else:
                df['wallet'] = 'unknown_' + pd.Series(range(len(df))).astype(str)
        
        # Convert current_holding from string to float (API returns strings like "0")
        if 'current_holding' not in df.columns:
            df['current_holding'] = 0
            logger.warning("No current_holding column found after standardization")
        else:
            # Log sample values before conversion
            sample_before = df['current_holding'].head(3).tolist()
            logger.info(f"current_holding before conversion: {sample_before} (types: {[type(v).__name__ for v in sample_before]})")
            
            df['current_holding'] = pd.to_numeric(df['current_holding'], errors='coerce').fillna(0)
            
            # Log sample values after conversion
            sample_after = df['current_holding'].head(3).tolist()
            positive_count = (df['current_holding'] > 0).sum()
            logger.info(f"current_holding after conversion: {sample_after}, positive count: {positive_count}/{len(df)}")
        
        # Extract total_buys and total_sells from transaction history if not present
        if 'transactions' in df.columns and ('total_buys' not in df.columns or 'total_sells' not in df.columns):
            total_buys = []
            total_sells = []
            first_trades = []
            last_trades = []
            
            for _, row in df.iterrows():
                txns = row.get('transactions', [])
                if isinstance(txns, list) and len(txns) > 0:
                    buys = sum(float(t.get('amount', 0)) for t in txns if t.get('isBuy', False))
                    sells = sum(float(t.get('amount', 0)) for t in txns if not t.get('isBuy', True))
                    total_buys.append(buys)
                    total_sells.append(sells)
                    
                    # Get first and last trade times
                    timestamps = [t.get('timestamp') for t in txns if t.get('timestamp')]
                    if timestamps:
                        first_trades.append(min(timestamps))
                        last_trades.append(max(timestamps))
                    else:
                        first_trades.append(None)
                        last_trades.append(None)
                else:
                    total_buys.append(0)
                    total_sells.append(0)
                    first_trades.append(None)
                    last_trades.append(None)
            
            if 'total_buys' not in df.columns:
                df['total_buys'] = total_buys
            if 'total_sells' not in df.columns:
                df['total_sells'] = total_sells
            if 'first_trade_at' not in df.columns:
                df['first_trade_at'] = first_trades
            if 'last_trade_at' not in df.columns:
                df['last_trade_at'] = last_trades
        
        if 'total_buys' not in df.columns:
            df['total_buys'] = 0
        
        if 'total_sells' not in df.columns:
            df['total_sells'] = 0
        
        if 'first_trade_at' not in df.columns:
            df['first_trade_at'] = None
        else:
            df['first_trade_at'] = pd.to_datetime(df['first_trade_at'], errors='coerce')
        
        if 'last_trade_at' not in df.columns:
            df['last_trade_at'] = None
        else:
            df['last_trade_at'] = pd.to_datetime(df['last_trade_at'], errors='coerce')
        
        return df
    
    def fetch_complete_data_v2(self, mint: str, candle_days: int = 90, 
                                holder_limit: int = 1000) -> Dict[str, Any]:
        """
        Fetch all required data for enhanced whale tracking (v2)
        Uses PARALLEL execution for faster data fetching
        
        Args:
            mint: Mint address
            candle_days: Days of candle data
            holder_limit: Maximum holders to fetch
            
        Returns:
            Dictionary with all data including user holdings
        """
        logger.info(f"Fetching complete dataset v2 for {mint[:8]}... (parallel)")
        
        data = {
            'candles': pd.DataFrame(),
            'user_holdings': pd.DataFrame(),
            'holders': pd.DataFrame(),
            'holders_historical': pd.DataFrame(),  # For whale trading activity
            'metadata': None
        }
        
        # Calculate proper candle limit to cover the full date range
        # For 5-minute candles: 1 day = 288 candles (24h * 12)
        # Without enough limit, API returns oldest candles first, truncating recent data
        # This caused stale data where price_current was days/weeks old
        candles_per_day = 288  # 5-minute candles per day
        candle_limit = max(1000, candle_days * candles_per_day)
        logger.info(f"Candle limit calculated: {candle_limit} (for {candle_days} days of 5m candles)")
        
        # Execute all API calls in PARALLEL for speed
        # This reduces total time significantly (from ~25s to ~18s)
        # HYBRID APPROACH:
        # - fetch_current_holders: For accurate holder counts & concentration (filters out pools)
        # - fetch_holders: For historical trading activity & whale delta calculation
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.fetch_candles, mint, candle_days, candle_limit): 'candles',
                executor.submit(self.fetch_user_holdings, mint, holder_limit): 'user_holdings',
                executor.submit(self.fetch_current_holders, mint): 'holders',  # Current on-chain holders
                executor.submit(self.fetch_holders, mint, candle_days): 'holders_historical',  # Historical trades
                executor.submit(self.fetch_mint_metadata, mint): 'metadata',
            }
            
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    data[key] = result
                except Exception as e:
                    logger.warning(f"Error fetching {key} for {mint[:8]}...: {e}")
        
        # Standardize data
        if len(data['candles']) > 0:
            data['candles'] = self.standardize_candles(data['candles'])
        
        if len(data['user_holdings']) > 0:
            data['user_holdings'] = self.standardize_user_holdings(data['user_holdings'])
        
        if len(data['holders']) > 0:
            data['holders'] = self.standardize_holders(data['holders'])
        
        if len(data['holders_historical']) > 0:
            data['holders_historical'] = self.standardize_holders(data['holders_historical'])
        
        logger.info(f"Complete dataset v2 fetched: {len(data['candles'])} candles, "
                   f"{len(data['user_holdings'])} user holdings, "
                   f"{len(data['holders'])} current holders (filtered), "
                   f"{len(data['holders_historical'])} historical holders")
        
        return data
    
    def fetch_complete_data(self, mint: str, candle_days: int = 90, holder_days: int = 90, 
                           trade_days: int = 90, candle_limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Fetch all required data for a specific mint
        
        Args:
            mint: Mint address
            candle_days: Days of candle data
            holder_days: Days of holder data
            trade_days: Days of trade data
            candle_limit: Max candle records
            
        Returns:
            Dictionary with all data frames
        """
        logger.info(f"Fetching complete dataset for {mint[:8]}...")
        
        data = {
            'candles': self.fetch_candles(mint, days=candle_days, limit=candle_limit),
            'holders': self.fetch_holders(mint, days=holder_days),
            'trades': self.fetch_trades(mint, days=trade_days, limit=1000)
        }
        
        # Add mint column if not present
        for key, df in data.items():
            if len(df) > 0 and 'mint' not in df.columns:
                df['mint'] = mint
        
        logger.info("Complete dataset fetched successfully")
        return data
    
    def fetch_multiple_mints(self, mints: List[str], candle_days: int = 90, 
                            holder_days: int = 90, trade_days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple mints and combine
        
        Args:
            mints: List of mint addresses
            candle_days: Days of candle data
            holder_days: Days of holder data
            trade_days: Days of trade data
            
        Returns:
            Dictionary with combined data frames
        """
        logger.info(f"Fetching data for {len(mints)} mints...")
        
        all_candles = []
        all_holders = []
        all_trades = []
        
        for i, mint in enumerate(mints, 1):
            logger.info(f"Processing mint {i}/{len(mints)}: {mint[:8]}...")
            
            data = self.fetch_complete_data(mint, candle_days, holder_days, trade_days)
            
            if len(data['candles']) > 0:
                all_candles.append(data['candles'])
            if len(data['holders']) > 0:
                all_holders.append(data['holders'])
            if len(data['trades']) > 0:
                all_trades.append(data['trades'])
            
            # Rate limiting
            time.sleep(0.5)
        
        combined_data = {
            'candles': pd.concat(all_candles, ignore_index=True) if all_candles else pd.DataFrame(),
            'holders': pd.concat(all_holders, ignore_index=True) if all_holders else pd.DataFrame(),
            'trades': pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        }
        
        logger.info(f"Combined data: {len(combined_data['candles'])} candles, "
                   f"{len(combined_data['holders'])} holders, {len(combined_data['trades'])} trades")
        
        return combined_data
    
    def standardize_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize candle data to match expected format
        
        Args:
            df: Raw candle DataFrame
            
        Returns:
            Standardized DataFrame
        """
        if len(df) == 0:
            return df
        
        df = df.copy()
        
        # Expected columns: Mint, Timestamp, Open, High, Low, Close, Volume, BuyVolume, SellVolume
        # Try multiple possible column name variations
        column_mapping = {
            # Timestamp variations
            'timestamp': 'Timestamp',
            'time': 'Timestamp',
            'date': 'Timestamp',
            'datetime': 'Timestamp',
            'createdAt': 'Timestamp',
            # Price variations
            'open': 'Open',
            'Open': 'Open',
            'high': 'High',
            'High': 'High',
            'low': 'Low',
            'Low': 'Low',
            'close': 'Close',
            'Close': 'Close',
            'price': 'Close',
            # Volume variations
            'volume': 'Volume',
            'Volume': 'Volume',
            'vol': 'Volume',
            'buyVolume': 'BuyVolume',
            'buy_volume': 'BuyVolume',
            'sellVolume': 'SellVolume',
            'sell_volume': 'SellVolume',
            # Mint variations
            'mint': 'mint',
            'Mint': 'mint',
            'token': 'mint',
            'address': 'mint'
        }
        
        # Rename columns if they exist (case-sensitive check)
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df = df.rename(columns={old_col: new_col})
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure Timestamp column exists
        if 'Timestamp' not in df.columns:
            # Try to find any datetime-like column
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                df['Timestamp'] = df[datetime_cols[0]]
            else:
                # Create dummy timestamps
                logger.warning("No timestamp column found, creating dummy timestamps")
                df['Timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5min')
        
        # Ensure Timestamp is datetime with robust parsing
        # Handles: string timestamps, integer epoch seconds, integer epoch milliseconds
        if 'Timestamp' in df.columns:
            ts_series = df['Timestamp']
            if ts_series.dtype in ['int64', 'float64', 'int32', 'float32']:
                # Integer timestamps: detect seconds vs milliseconds
                max_ts = float(ts_series.max())
                if max_ts > 1e15:
                    # Microseconds (unlikely but handle it)
                    df['Timestamp'] = pd.to_datetime(ts_series, unit='us', errors='coerce')
                elif max_ts > 1e12:
                    # Milliseconds (common in JS-based APIs)
                    df['Timestamp'] = pd.to_datetime(ts_series, unit='ms', errors='coerce')
                else:
                    # Seconds (Unix epoch)
                    df['Timestamp'] = pd.to_datetime(ts_series, unit='s', errors='coerce')
                logger.info(f"Parsed {len(df)} integer timestamps (max raw value: {max_ts:.0f})")
            else:
                df['Timestamp'] = pd.to_datetime(ts_series, format='mixed', errors='coerce')
            
            # Validate: check for NaT and warn
            nat_count = df['Timestamp'].isna().sum()
            if nat_count > 0:
                logger.warning(f"Timestamp parsing: {nat_count}/{len(df)} timestamps could not be parsed (NaT)")
        
        # Ensure OHLC columns exist
        if 'Open' not in df.columns and 'Close' in df.columns:
            df['Open'] = df['Close']
        if 'High' not in df.columns and 'Close' in df.columns:
            df['High'] = df['Close']
        if 'Low' not in df.columns and 'Close' in df.columns:
            df['Low'] = df['Close']
        if 'Close' not in df.columns:
            df['Close'] = 0
        
        # Add missing columns with default values
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        if 'BuyVolume' not in df.columns:
            df['BuyVolume'] = df['Volume'] * 0.5
        if 'SellVolume' not in df.columns:
            df['SellVolume'] = df['Volume'] * 0.5
        if 'BuyCount' not in df.columns:
            df['BuyCount'] = 0
        if 'SellCount' not in df.columns:
            df['SellCount'] = 0
        if 'TotalSupply' not in df.columns:
            df['TotalSupply'] = 0
        if 'SolUSDPrice' not in df.columns:
            df['SolUSDPrice'] = 0
        
        return df
    
    def standardize_holders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize holder data to match expected format
        
        Args:
            df: Raw holder DataFrame
            
        Returns:
            Standardized DataFrame
        """
        if len(df) == 0:
            return df
        
        df = df.copy()
        
        # Expected columns: address, tradeCount, buyCount, sellCount, totalBuys, totalSells, 
        #                   highestHolding, lastHolding, lastTradeAt, mint
        column_mapping = {
            # Address variations
            'wallet': 'address',
            'holder': 'address',
            'owner': 'address',
            # Trade count variations
            'trades': 'tradeCount',
            'trade_count': 'tradeCount',
            'numTrades': 'tradeCount',
            # Buy/Sell count variations
            'buys': 'buyCount',
            'buy_count': 'buyCount',
            'numBuys': 'buyCount',
            'sells': 'sellCount',
            'sell_count': 'sellCount',
            'numSells': 'sellCount',
            # Total buy/sell variations
            'totalBuy': 'totalBuys',
            'total_buy': 'totalBuys',
            'totalSell': 'totalSells',
            'total_sell': 'totalSells',
            # Holding variations
            'highest': 'highestHolding',
            'highest_holding': 'highestHolding',
            'max_holding': 'highestHolding',
            'current': 'lastHolding',
            'current_holding': 'lastHolding',
            'balance': 'lastHolding',
            'amount': 'lastHolding',
            'holding': 'lastHolding',
            # Timestamp variations
            'lastTrade': 'lastTradeAt',
            'last_trade': 'lastTradeAt',
            'last_trade_at': 'lastTradeAt',
            'timestamp': 'lastTradeAt',
            # Mint variations - Note: 'address' is NOT mapped to mint because in holder data,
            # 'address' refers to the wallet address, not the token mint
            'mint': 'mint',
            'token': 'mint',
            'token_mint': 'mint'
        }
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df = df.rename(columns={old_col: new_col})
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Add missing columns with defaults
        if 'address' not in df.columns:
            df['address'] = 'unknown_' + pd.Series(range(len(df))).astype(str)
        
        if 'tradeCount' not in df.columns:
            df['tradeCount'] = df.get('buyCount', 0) + df.get('sellCount', 0)
        
        if 'buyCount' not in df.columns:
            df['buyCount'] = 0
        
        if 'sellCount' not in df.columns:
            df['sellCount'] = 0
        
        if 'totalBuys' not in df.columns:
            df['totalBuys'] = 0
        
        if 'totalSells' not in df.columns:
            df['totalSells'] = 0
        
        if 'lastHolding' not in df.columns:
            # Try to find any numeric column that might be balance
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['lastHolding'] = df[numeric_cols[0]]
            else:
                df['lastHolding'] = 0
        
        if 'highestHolding' not in df.columns:
            df['highestHolding'] = df['lastHolding']
        
        if 'lastTradeAt' not in df.columns:
            df['lastTradeAt'] = pd.Timestamp.now()
        
        return df
    
    # ==================== DUAL HOLDER SOURCE (FEATURE 2) ====================
    
    def fetch_holders_dual_source(self, mint: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch holders from both API and Birdeye, merge based on configuration
        
        Args:
            mint: Token mint address
            limit: Maximum holders to fetch per source
            
        Returns:
            Merged DataFrame with holder data from both sources
        """
        strategy = holder_config.MERGE_STRATEGY
        logger.info(f"Fetching holders (dual source, strategy: {strategy}) for {mint[:8]}...")
        
        # Fetch from primary source (API)
        api_holders = self._fetch_api_holders(mint, limit)
        logger.info(f"API holders: {len(api_holders)} records")
        
        # Fetch from Birdeye if configured
        birdeye_holders = self._fetch_birdeye_holders(mint, limit)
        logger.info(f"Birdeye holders: {len(birdeye_holders)} records")
        
        # Merge based on strategy
        if strategy == "primary_only":
            merged = self._merge_primary_only(api_holders, birdeye_holders)
        elif strategy == "intersect":
            merged = self._merge_intersect(api_holders, birdeye_holders)
        else:  # Default: union
            merged = self._merge_union(api_holders, birdeye_holders)
        
        logger.info(f"Merged holders ({strategy}): {len(merged)} records")
        
        return merged
    
    def _fetch_api_holders(self, mint: str, limit: int) -> pd.DataFrame:
        """Fetch holders from primary API"""
        # Use existing methods
        user_holdings = self.fetch_user_holdings(mint, limit)
        
        if len(user_holdings) > 0:
            user_holdings = self.standardize_user_holdings(user_holdings)
            # Add source column
            user_holdings['source'] = 'api'
            return user_holdings
        
        # Fallback to internal range-holders (skip Birdeye to avoid circular call)
        holders = self._fetch_holders_internal(mint, days=90)
        if len(holders) > 0:
            holders = self.standardize_holders(holders)
            holders['source'] = 'api'
            return holders
        
        return pd.DataFrame()
    
    def _fetch_birdeye_holders(self, mint: str, limit: int) -> pd.DataFrame:
        """Fetch holders from Birdeye API and standardize to common format."""
        if not birdeye_config.is_configured():
            logger.debug("Birdeye not configured, skipping")
            return pd.DataFrame()

        try:
            from core.data_fetcher_birdeye import BirdeyeFetcher

            birdeye = BirdeyeFetcher()
            holders_list = birdeye.fetch_token_holders(mint, limit)

            if not holders_list:
                return pd.DataFrame()

            standardized = []
            for holder in holders_list:
                standardized.append({
                    'wallet': holder.get('wallet', ''),
                    'current_holding': float(holder.get('balance', 0) or 0),
                    'total_buys': 0.0,
                    'total_sells': 0.0,
                    'first_trade_at': None,
                    'last_trade_at': None,
                    'source': 'birdeye'
                })

            return pd.DataFrame(standardized)

        except Exception as e:
            logger.warning(f"Error fetching Birdeye holders: {e}")
            return pd.DataFrame()
    
    def _merge_union(self, api_df: pd.DataFrame, birdeye_df: pd.DataFrame) -> pd.DataFrame:
        """
        Union merge: Combine holders from both sources
        If same wallet appears in both, prefer API data (more complete)
        """
        if len(api_df) == 0 and len(birdeye_df) == 0:
            return pd.DataFrame()
        
        if len(api_df) == 0:
            return birdeye_df
        
        if len(birdeye_df) == 0:
            return api_df
        
        # Get wallet column
        api_wallet_col = 'wallet' if 'wallet' in api_df.columns else 'address'
        birdeye_wallet_col = 'wallet' if 'wallet' in birdeye_df.columns else 'address'
        
        # Get set of wallets in API data
        api_wallets = set(api_df[api_wallet_col].values)
        
        # Filter Birdeye to only wallets not in API
        birdeye_new = birdeye_df[~birdeye_df[birdeye_wallet_col].isin(api_wallets)].copy()
        
        # Standardize column names for concat
        if api_wallet_col != 'wallet':
            api_df = api_df.rename(columns={api_wallet_col: 'wallet'})
        if birdeye_wallet_col != 'wallet':
            birdeye_new = birdeye_new.rename(columns={birdeye_wallet_col: 'wallet'})
        
        # Concat
        merged = pd.concat([api_df, birdeye_new], ignore_index=True)
        merged['source'] = merged['source'].fillna('merged')
        
        return merged
    
    def _merge_primary_only(self, api_df: pd.DataFrame, birdeye_df: pd.DataFrame) -> pd.DataFrame:
        """
        Primary only: Use API data, fallback to Birdeye if empty
        """
        if len(api_df) > 0:
            return api_df
        
        return birdeye_df
    
    def _merge_intersect(self, api_df: pd.DataFrame, birdeye_df: pd.DataFrame) -> pd.DataFrame:
        """
        Intersect: Only include holders that appear in both sources
        Use API data for holder details
        """
        if len(api_df) == 0 or len(birdeye_df) == 0:
            return pd.DataFrame()
        
        # Get wallet columns
        api_wallet_col = 'wallet' if 'wallet' in api_df.columns else 'address'
        birdeye_wallet_col = 'wallet' if 'wallet' in birdeye_df.columns else 'address'
        
        # Get intersection of wallets
        api_wallets = set(api_df[api_wallet_col].values)
        birdeye_wallets = set(birdeye_df[birdeye_wallet_col].values)
        common_wallets = api_wallets.intersection(birdeye_wallets)
        
        # Filter API data to only common wallets
        merged = api_df[api_df[api_wallet_col].isin(common_wallets)].copy()
        merged['source'] = 'intersect'
        
        return merged


    # ==================== USER PROFILE ENDPOINTS (LAYER 2) ====================
    
    def fetch_users_list(self, page: int = 1, limit: int = None) -> List[str]:
        """
        Fetch list of user wallet addresses
        
        Args:
            page: Page number for pagination
            limit: Number of users per page (uses config default if not provided)
            
        Returns:
            List of wallet addresses
        """
        if limit is None:
            limit = api_config.DEFAULT_USER_PAGE_SIZE
        
        url = f"{self.base_url}{api_config.ENDPOINT_USERS}"
        params = {
            'page': page,
            'limit': limit
        }
        
        logger.info(f"Fetching users list (page {page}, limit {limit})...")
        
        try:
            response = self.session.get(url, params=params, timeout=api_config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response structures
            users = []
            if isinstance(data, list):
                # Direct list of users
                for item in data:
                    if isinstance(item, str):
                        users.append(item)
                    elif isinstance(item, dict):
                        # Extract wallet address from dict
                        wallet = item.get('address') or item.get('wallet') or item.get('user') or item.get('signer')
                        if wallet:
                            users.append(wallet)
            elif isinstance(data, dict):
                # Nested response
                user_list = data.get('users') or data.get('data') or data.get('addresses') or []
                for item in user_list:
                    if isinstance(item, str):
                        users.append(item)
                    elif isinstance(item, dict):
                        wallet = item.get('address') or item.get('wallet') or item.get('user') or item.get('signer')
                        if wallet:
                            users.append(wallet)
            
            logger.info(f"Fetched {len(users)} users")
            return users
            
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching users list")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching users list: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching users list: {e}")
            return []
    
    def fetch_user_timeline(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: Use fetch_user_profiling_summary() and fetch_user_profiling_trades() instead.
        
        Fetch complete trading timeline for a specific user.
        This method is kept for backwards compatibility.
        
        Args:
            wallet_address: User's wallet address
            
        Returns:
            Dictionary with user timeline data (old format)
        """
        logger.warning(f"fetch_user_timeline is DEPRECATED. Use fetch_user_complete_profile() instead.")
        
        url = f"{self.base_url}{api_config.ENDPOINT_USER_TIMELINE}"
        params = {
            'address': wallet_address
        }
        
        logger.info(f"Fetching user timeline (deprecated) for {wallet_address[:8]}...")
        
        try:
            response = self.session.get(url, params=params, timeout=api_config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Ensure consistent structure
            if isinstance(data, dict):
                # Standardize field names
                result = {
                    'user': data.get('user') or data.get('address') or data.get('wallet') or wallet_address,
                    'totalTrades': data.get('totalTrades') or data.get('total_trades') or 0,
                    'totalMints': data.get('totalMints') or data.get('total_mints') or 0,
                    'mints': data.get('mints') or data.get('tokens') or data.get('holdings') or []
                }
                
                # Standardize each mint entry
                standardized_mints = []
                for mint_data in result['mints']:
                    if isinstance(mint_data, dict):
                        standardized_mint = {
                            'mint': mint_data.get('mint') or mint_data.get('token') or mint_data.get('address'),
                            'totalTrades': mint_data.get('totalTrades') or mint_data.get('total_trades') or 0,
                            'finalHolding': str(mint_data.get('finalHolding') or mint_data.get('final_holding') or mint_data.get('balance') or 0),
                            'history': mint_data.get('history') or mint_data.get('transactions') or mint_data.get('trades') or []
                        }
                        
                        # Standardize transaction history
                        standardized_history = []
                        for txn in standardized_mint['history']:
                            if isinstance(txn, dict):
                                standardized_txn = {
                                    'mint': txn.get('mint') or standardized_mint['mint'],
                                    'transactionId': txn.get('transactionId') or txn.get('transaction_id') or txn.get('txId'),
                                    'timestamp': txn.get('timestamp') or txn.get('time') or txn.get('createdAt'),
                                    'slot': txn.get('slot'),
                                    'price': float(txn.get('price') or 0),
                                    'isBuy': txn.get('isBuy') if 'isBuy' in txn else (txn.get('type', '').upper() == 'BUY'),
                                    'amount': str(txn.get('amount') or 0),
                                    'beforeHolding': str(txn.get('beforeHolding') or txn.get('before_holding') or 0),
                                    'afterHolding': str(txn.get('afterHolding') or txn.get('after_holding') or 0)
                                }
                                standardized_history.append(standardized_txn)
                        
                        standardized_mint['history'] = standardized_history
                        standardized_mints.append(standardized_mint)
                
                result['mints'] = standardized_mints
                
                logger.info(f"Fetched user timeline: {result['totalTrades']} trades across {result['totalMints']} mints")
                return result
            else:
                logger.warning(f"Unexpected response format for user timeline")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching user timeline for {wallet_address[:8]}...")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching user timeline for {wallet_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching user timeline for {wallet_address[:8]}...: {e}")
            return None
    
    # ==================== NEW USER PROFILING API METHODS ====================
    
    def _get_default_date_range(self) -> Tuple[str, str]:
        """
        Generate default from/to dates for user profiling API.
        Default window: last USER_PROFILING_DEFAULT_DAYS (80) ending today.
        Returns (from_date, to_date) in ISO format.
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=api_config.USER_PROFILING_DEFAULT_DAYS)
        
        return (
            from_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            to_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        )
    
    def _normalize_to_date_for_internal_api(self, to_date: str) -> str:
        """
        Normalize to_date for the internal user-profiling API.
        The internal API returns 400 Bad Request when to_date uses 23:59:59;
        use start of next day (00:00:00Z) so the range is accepted.
        """
        if not to_date or "23:59:59" not in to_date:
            return to_date
        try:
            # Parse date part (YYYY-MM-DD) and add one day, output as 00:00:00Z
            date_part = to_date.split("T")[0]
            dt = datetime.strptime(date_part, "%Y-%m-%d")
            next_day = dt + timedelta(days=1)
            return next_day.strftime("%Y-%m-%dT00:00:00Z")
        except (ValueError, IndexError):
            return to_date
    
    def fetch_user_profiling_summary(
        self,
        wallet_address: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        trading_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch pre-calculated user statistics from the /user-profiling API.

        Supports both the legacy flat response (five fields) and the newer
        shape that includes win rates, ``max_drawdown_pct_realized``, fees,
        leverage, etc.  Raw JSON is passed through
        ``normalize_user_profiling_summary`` so every caller gets a consistent
        dict.

        Args:
            wallet_address: User's wallet address
            from_date: Start date in ISO format (default: today − 80 days)
            to_date: End date in ISO format (default: today)
            trading_type: Type of trading (default: "spot")

        Returns:
            Normalized dictionary (always contains at least the five legacy
            keys) or None on error.
        """
        if from_date is None or to_date is None:
            default_from, default_to = self._get_default_date_range()
            from_date = from_date or default_from
            to_date = to_date or default_to
        
        trading_type = trading_type or api_config.USER_PROFILING_TYPE
        
        url = f"{self.base_url}{api_config.ENDPOINT_USER_PROFILING}"
        params = {
            'address': wallet_address,
            'from': from_date,
            'to': to_date,
            'type': trading_type
        }
        
        logger.info(f"Fetching user profiling summary for {wallet_address[:8]}...")
        retry_delay = getattr(api_config, 'USER_PROFILING_RETRY_DELAY', 2.0)
        
        for attempt in range(2):
            start = time.perf_counter()
            try:
                response = self.session.get(url, params=params, timeout=api_config.USER_PROFILING_TIMEOUT)
                elapsed = time.perf_counter() - start
                
                if response.status_code >= 500:
                    logger.warning(
                        f"User profiling summary for {wallet_address[:8]}... returned {response.status_code} after {elapsed:.2f}s"
                        + ("; retrying once." if attempt == 0 else "; giving up.")
                    )
                    if attempt == 0:
                        time.sleep(retry_delay)
                        continue
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, dict):
                    result = normalize_user_profiling_summary(data)
                    logger.info(
                        f"Fetched user profiling summary in {elapsed:.2f}s: PnL=${result['lifetime_pnl']:.2f}, "
                        f"Avg Hold={result['avg_holding_time_minutes']:.1f}min"
                    )
                    return result
                else:
                    logger.warning(f"Unexpected response format for user profiling summary (after {elapsed:.2f}s)")
                    return None
                    
            except requests.exceptions.Timeout:
                elapsed = time.perf_counter() - start
                logger.warning(
                    f"Timeout fetching user profiling summary for {wallet_address[:8]}... after {elapsed:.2f}s"
                    + ("; retrying once." if attempt == 0 else "; giving up.")
                )
                if attempt == 0:
                    time.sleep(retry_delay)
                    continue
                return None
            except requests.exceptions.RequestException as e:
                elapsed = time.perf_counter() - start
                logger.error(f"Error fetching user profiling summary for {wallet_address[:8]}... after {elapsed:.2f}s: {e}")
                return None
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"Unexpected error fetching user profiling summary for {wallet_address[:8]}... after {elapsed:.2f}s: {e}")
                return None
        
        return None
    
    def fetch_user_profiling_trades(
        self,
        wallet_address: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch detailed trade history from the new user-profiling API
        Handles pagination automatically with parallel API calls for performance.
        Returns at most BEHAVIOUR_TRADE_CAP (500) trades, most recent first, for behaviour stats.
        
        Args:
            wallet_address: User's wallet address
            from_date: Start date in ISO format (default: today − 80 days)
            to_date: End date in ISO format (default: today)
            
        Returns:
            List of trade dictionaries (capped at 500, most recent) or empty list on error
        """
        # Get default dates if not provided
        if from_date is None or to_date is None:
            default_from, default_to = self._get_default_date_range()
            from_date = from_date or default_from
            to_date = to_date or default_to
        
        url = f"{self.base_url}{api_config.ENDPOINT_USER_PROFILING_TRADES}"
        
        logger.info(f"Fetching user profiling trades for {wallet_address[:8]}...")
        
        # Step 1: Fetch first page to get pagination info
        first_page_trades, total_pages = self._fetch_trades_page(
            url, wallet_address, from_date, to_date, page=1
        )
        
        if first_page_trades is None:
            return []
        
        all_trades = first_page_trades
        
        # Step 2: If more pages exist, fetch them in parallel
        if total_pages > 1:
            logger.info(f"Found {total_pages} pages of trades for {wallet_address[:8]}..., fetching pages 2-{total_pages} in parallel")
            
            remaining_pages = list(range(2, total_pages + 1))
            
            with ThreadPoolExecutor(max_workers=min(5, total_pages - 1)) as executor:
                futures = {
                    executor.submit(
                        self._fetch_trades_page,
                        url, wallet_address, from_date, to_date, page
                    ): page
                    for page in remaining_pages
                }
                
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_trades, _ = future.result()
                        if page_trades:
                            all_trades.extend(page_trades)
                            logger.debug(f"Page {page_num}: fetched {len(page_trades)} trades")
                    except Exception as e:
                        logger.warning(f"Error fetching page {page_num} for {wallet_address[:8]}...: {e}")
        
        cap = getattr(api_config, 'BEHAVIOUR_TRADE_CAP', 500)
        if len(all_trades) > cap:
            # Sort by timestamp descending (most recent first); take first cap trades
            def _trade_sort_key(t):
                ts = t.get('timestamp') or t.get('time') or t.get('created_at') or t.get('block_time') or ''
                return (ts if isinstance(ts, str) else str(ts)) if ts else ''
            original_count = len(all_trades)
            all_trades = sorted(all_trades, key=_trade_sort_key, reverse=True)[:cap]
            logger.info(f"Capped to most recent {cap} trades for behaviour stats (had {original_count})")
        logger.info(f"Fetched total {len(all_trades)} trades for {wallet_address[:8]}... ({total_pages} pages)")
        return all_trades
    
    def _fetch_trades_page(
        self,
        url: str,
        wallet_address: str,
        from_date: str,
        to_date: str,
        page: int
    ) -> Tuple[Optional[List[Dict[str, Any]]], int]:
        """
        Fetch a single page of trades
        
        Args:
            url: API endpoint URL
            wallet_address: User's wallet address
            from_date: Start date in ISO format
            to_date: End date in ISO format
            page: Page number to fetch
            
        Returns:
            Tuple of (trades list, total_pages) or (None, 0) on error
        """
        params = {
            'address': wallet_address,
            'from': from_date,
            'to': to_date,
            'page': page
        }
        retry_delay = getattr(api_config, 'USER_PROFILING_RETRY_DELAY', 2.0)
        
        for attempt in range(2):
            start = time.perf_counter()
            try:
                response = self.session.get(url, params=params, timeout=api_config.USER_PROFILING_TIMEOUT)
                elapsed = time.perf_counter() - start
                
                if response.status_code >= 500:
                    logger.warning(
                        f"User profiling trades page {page} for {wallet_address[:8]}... returned {response.status_code} after {elapsed:.2f}s"
                        + ("; retrying once." if attempt == 0 else "; giving up.")
                    )
                    if attempt == 0:
                        time.sleep(retry_delay)
                        continue
                    return None, 0
                
                response.raise_for_status()
                data = response.json()
                
                # Extract trades from response
                if isinstance(data, list):
                    trades = data
                    total_pages = 1
                elif isinstance(data, dict):
                    trades = data.get('data', []) or data.get('trades', []) or []
                    meta = data.get('meta', {})
                    total_pages = int(meta.get('totalPages', 1))
                else:
                    trades = []
                    total_pages = 1
                
                if page == 1:
                    logger.info(f"Fetched user profiling trades page {page} in {elapsed:.2f}s ({len(trades)} trades, {total_pages} pages)")
                else:
                    logger.debug(f"Fetched user profiling trades page {page} in {elapsed:.2f}s ({len(trades)} trades)")
                
                return trades, total_pages
                    
            except requests.exceptions.Timeout:
                elapsed = time.perf_counter() - start
                logger.warning(
                    f"Timeout fetching trades page {page} for {wallet_address[:8]}... after {elapsed:.2f}s"
                    + ("; retrying once." if attempt == 0 else "; giving up.")
                )
                if attempt == 0:
                    time.sleep(retry_delay)
                    continue
                return None, 0
            except requests.exceptions.RequestException as e:
                elapsed = time.perf_counter() - start
                logger.error(f"Error fetching trades page {page} for {wallet_address[:8]}... after {elapsed:.2f}s: {e}")
                return None, 0
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(f"Unexpected error fetching trades page {page} for {wallet_address[:8]}... after {elapsed:.2f}s: {e}")
                return None, 0
        
        return None, 0
    
    def fetch_user_complete_profile(
        self,
        wallet_address: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch both user summary and trades in one convenience call
        
        Args:
            wallet_address: User's wallet address
            from_date: Start date in ISO format (optional)
            to_date: End date in ISO format (optional)
            
        Returns:
            Combined dictionary:
            {
                "summary": {summary response},
                "trades": [trades list]
            }
            or None if both requests fail
        """
        logger.info(f"Fetching complete user profile for {wallet_address[:8]}...")
        
        # Resolve dates once so the *same* range is sent to both summary and trades APIs.
        if from_date is None or to_date is None:
            default_from, default_to = self._get_default_date_range()
            from_date = from_date or default_from
            to_date = to_date or default_to
        
        # Internal API returns 400 if to_date has 23:59:59; normalize to start-of-next-day 00:00:00Z.
        to_date_internal = self._normalize_to_date_for_internal_api(to_date)
        logger.info(f"Using date range {from_date} → {to_date_internal} for both summary and trades APIs")
        
        # Both calls use the identical from_date / to_date_internal.
        # The summary call must complete (or timeout) before we consider the profile empty;
        # we never return an empty profile without having waited for this response.
        summary = self.fetch_user_profiling_summary(wallet_address, from_date, to_date_internal)
        trades = self.fetch_user_profiling_trades(wallet_address, from_date, to_date_internal)
        
        # Return None only if both fail — never short-circuit before waiting for both.
        if summary is None and not trades:
            logger.warning(f"Failed to fetch complete profile for {wallet_address[:8]}...")
            return None
        
        result = {
            'summary': summary or {},
            'trades': trades or [],
            'wallet_address': wallet_address,
            'from_date': from_date,
            'to_date': to_date
        }
        
        logger.info(f"Complete profile fetched: summary={'Yes' if summary else 'No'}, "
                   f"trades={len(trades)}")
        
        return result
    
    def fetch_mint_timeline(self, mint_address: str) -> Optional[Dict[str, Any]]:
        """
        Fetch complete trading timeline for a specific mint/token
        This includes all holders and their complete transaction history
        
        Args:
            mint_address: Token mint address
            
        Returns:
            Dictionary with mint timeline data:
            {
                "mint": "token_address",
                "mintDetails": { "Name": "...", "Symbol": "...", ... },
                "totalHolders": 50,
                "totalSigners": 697,
                "holders": [
                    {
                        "signer": "wallet_address",
                        "totalTrades": 2,
                        "finalHolding": "0",
                        "history": [ ... ]
                    }
                ]
            }
        """
        url = f"{self.base_url}{api_config.ENDPOINT_MINT_TIMELINE}"
        params = {
            'mint': mint_address
        }
        
        logger.info(f"Fetching mint timeline for {mint_address[:8]}...")
        
        try:
            response = self.session.get(url, params=params, timeout=api_config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, dict):
                # Standardize field names
                result = {
                    'mint': data.get('mint') or data.get('address') or mint_address,
                    'mintDetails': data.get('mintDetails') or data.get('details') or data.get('metadata') or {},
                    'totalHolders': data.get('totalHolders') or data.get('total_holders') or 0,
                    'totalSigners': data.get('totalSigners') or data.get('total_signers') or 0,
                    'holders': data.get('holders') or data.get('users') or data.get('signers') or []
                }
                
                logger.info(f"Fetched mint timeline: {result['totalHolders']} holders, {result['totalSigners']} signers")
                return result
            else:
                logger.warning(f"Unexpected response format for mint timeline")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching mint timeline for {mint_address[:8]}...")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching mint timeline for {mint_address[:8]}...: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching mint timeline for {mint_address[:8]}...: {e}")
            return None
    
    def fetch_all_users(self, max_pages: int = 100, limit: int = 100) -> List[str]:
        """
        Fetch all users by paginating through the users endpoint with proper lastPage detection
        
        Args:
            max_pages: Maximum number of pages to fetch (safety limit)
            limit: Number of users per page (default: 100)
            
        Returns:
            List of all wallet addresses
        """
        all_users = []
        page = 1
        last_page = 1  # Will be updated from first response
        
        logger.info(f"Fetching all users (paginated, limit={limit})...")
        
        url = f"{self.base_url}{api_config.ENDPOINT_USERS}"
        
        while page <= max_pages:
            params = {
                'page': page,
                'limit': limit
            }
            
            try:
                response = self.session.get(url, params=params, timeout=api_config.REQUEST_TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract lastPage from response
                if isinstance(data, dict):
                    last_page = int(data.get('lastPage', 1))
                    total_users = int(data.get('total', 0))
                    
                    if page == 1:
                        logger.info(f"Found {total_users} total users across {last_page} pages")
                    
                    # Extract users from response
                    user_list = data.get('users') or data.get('data') or data.get('addresses') or []
                    users = []
                    for item in user_list:
                        if isinstance(item, str):
                            users.append(item)
                        elif isinstance(item, dict):
                            wallet = item.get('address') or item.get('wallet') or item.get('user') or item.get('signer')
                            if wallet:
                                users.append(wallet)
                    
                    if not users:
                        break
                    
                    all_users.extend(users)
                    logger.info(f"Page {page}/{last_page}: fetched {len(users)} users (total collected: {len(all_users)})")
                    
                    # Check if we've reached the last page
                    if page >= last_page:
                        break
                        
                else:
                    # Unexpected format - try legacy parsing
                    users = self.fetch_users_list(page=page, limit=limit)
                    if not users:
                        break
                    all_users.extend(users)
                
            except Exception as e:
                logger.error(f"Error fetching users page {page}: {e}")
                break
            
            page += 1
            
            # Rate limiting
            time.sleep(0.3)
        
        logger.info(f"Fetched total of {len(all_users)} users from {page-1} pages")
        return all_users

    def inspect_api_response(self, mint: str):
        """
        Inspect actual API response format for debugging
        
        Args:
            mint: Mint address to test
        """
        print("\n" + "=" * 80)
        print("API RESPONSE INSPECTION v2.0")
        print("=" * 80)
        
        # Test candles
        print("\n1. CANDLES ENDPOINT:")
        candles = self.fetch_candles(mint, days=1, limit=5)
        if len(candles) > 0:
            print(f"   Columns: {list(candles.columns)}")
            print(f"   Sample row:")
            print(candles.iloc[0].to_dict())
        else:
            print("   ERROR: No data returned")
        
        # Test holders
        print("\n2. HOLDERS ENDPOINT:")
        holders = self.fetch_holders(mint, days=1)
        if len(holders) > 0:
            print(f"   Columns: {list(holders.columns)}")
            print(f"   Sample row:")
            print(holders.iloc[0].to_dict())
        else:
            print("   ERROR: No data returned")
        
        # Test trades
        print("\n3. TRADES ENDPOINT:")
        trades = self.fetch_trades(mint, days=1, limit=5)
        if len(trades) > 0:
            print(f"   Columns: {list(trades.columns)}")
            print(f"   Sample row:")
            print(trades.iloc[0].to_dict())
        else:
            print("   ERROR: No data returned")
        
        # Test NEW user holdings endpoint
        print("\n4. USER HOLDINGS ENDPOINT (NEW):")
        user_holdings = self.fetch_user_holdings(mint, limit=5)
        if len(user_holdings) > 0:
            print(f"   Columns: {list(user_holdings.columns)}")
            print(f"   Sample row:")
            print(user_holdings.iloc[0].to_dict())
        else:
            print("   WARNING: No data returned (endpoint may need different params)")
        
        print("\n" + "=" * 80)


# ==================== PERPS DATA FETCHER ====================

class PerpsDataFetcher:
    """
    Fetches data for perpetual futures tokens via HyperLiquid (HyperEVM) API endpoints.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize Perps Data Fetcher
        
        Args:
            base_url: Base URL for perps API endpoints (uses config if not provided)
        """
        from config import perps_api_config
        
        self.base_url = base_url or perps_api_config.BASE_URL
        self.config = perps_api_config
        self.session = requests.Session()
        logger.info(f"PerpsDataFetcher initialized with base URL: {self.base_url}")
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Dict[str, Any],
        timeout: int = None,
        base_url: str = None
    ) -> Optional[Dict]:
        """
        Make HTTP request with retry logic

        Args:
            endpoint: API endpoint path
            params: Query parameters
            timeout: Request timeout in seconds
            base_url: Optional base URL (default: self.base_url); used e.g. for user-profiling on internal API

        Returns:
            JSON response data or None on failure
        """
        base = base_url if base_url is not None else self.base_url
        url = f"{base}{endpoint}"
        timeout = timeout or self.config.TIMEOUT
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.config.MAX_RETRIES}: {endpoint}")
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e.response.status_code}")
                if e.response.status_code == 404:
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
            
            if attempt < self.config.MAX_RETRIES - 1:
                time.sleep(self.config.DELAY_BETWEEN_REQUESTS * (attempt + 1))
        
        logger.error(f"All retry attempts exhausted for {endpoint}")
        return None
    
    def _fetch_paginated(
        self, 
        endpoint: str, 
        params: Dict[str, Any],
        max_pages: int = 5,
        max_records: int = 500
    ) -> List[Dict]:
        """
        Fetch paginated data from API (with limits for optimization)
        
        Args:
            endpoint: API endpoint path
            params: Base query parameters
            max_pages: Maximum pages to fetch (default: 5)
            max_records: Maximum total records to fetch (default: 500)
            
        Returns:
            List of all records across pages
        """
        all_records = []
        page = 1
        records_per_page = min(params.get('limit', self.config.RECORDS_PER_PAGE), 100)
        params['limit'] = records_per_page
        
        while True:
            params['page'] = page
            data = self._make_request(endpoint, params)
            
            if not data or not data.get('data'):
                break
            
            records = data['data']
            if len(records) == 0:
                break
            
            all_records.extend(records)
            
            total_pages = data.get('totalPages', 0)
            logger.debug(f"Fetched page {page}/{total_pages}: {len(records)} records (total: {len(all_records)})")
            
            # Check max_records limit
            if len(all_records) >= max_records:
                logger.info(f"Reached max_records limit ({max_records}), stopping pagination")
                break
            
            if total_pages > 0 and page >= total_pages:
                break
            
            if max_pages and page >= max_pages:
                break
            
            page += 1
            time.sleep(self.config.DELAY_BETWEEN_REQUESTS)
        
        return all_records
    
    def _compute_date_range(self, resolution: str, limit: int, from_date: str = None, to_date: str = None):
        """Compute startTime/endTime for HyperEVM date-range based APIs."""
        if to_date:
            end_dt = pd.to_datetime(to_date)
        else:
            end_dt = datetime.utcnow()
        
        if from_date:
            start_dt = pd.to_datetime(from_date)
        else:
            resolution_hours = {
                "1MIN": 1/60, "5MINS": 5/60, "15MINS": 0.25,
                "30MINS": 0.5, "1HOUR": 1, "4HOURS": 4, "1DAY": 24,
            }
            hours_per_candle = resolution_hours.get(resolution, 1)
            lookback_hours = limit * hours_per_candle
            start_dt = end_dt - timedelta(hours=lookback_hours)
        
        return start_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'), end_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    def fetch_candles(
        self,
        ticker: str,
        resolution: str = "1HOUR",
        from_date: str = None,
        to_date: str = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candle data for perps token via HyperEVM candle-snapshot.
        
        Args:
            ticker: Trading pair (e.g., 'BTC-USD')
            resolution: Candle resolution (e.g., '1HOUR', '1DAY') — auto-mapped to HyperLiquid format
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
            limit: Maximum records to fetch (default: 500 = ~21 days of hourly data)
            
        Returns:
            DataFrame with candle data
        """
        coin = self.config.to_hyper_coin(ticker)
        interval = self.config.to_hyper_resolution(resolution)
        start_time, end_time = self._compute_date_range(resolution, limit, from_date, to_date)
        
        logger.info(f"Fetching perps candles for {ticker} ({coin}, {interval}, {start_time} to {end_time})...")
        
        params = {
            'coin': coin,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
        }
        
        data = self._make_request(self.config.ENDPOINT_CANDLES, params)
        
        if not data or not data.get('data'):
            logger.warning(f"No candle data for {ticker}")
            return pd.DataFrame()
        
        records = data['data']
        df = pd.DataFrame(records)
        
        # Map HyperEVM fields to standardized names matching old DYDX output
        column_mapping = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'base_volume',
            'n': 'trade_count',
        }
        df = df.rename(columns=column_mapping)
        
        for col in ['open', 'high', 'low', 'close', 'base_volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute USD volume (base_volume * close_price)
        if 'base_volume' in df.columns and 'close' in df.columns:
            df['volume'] = df['base_volume'] * df['close']
        else:
            df['volume'] = 0
        
        # Fields not available per-candle from HyperEVM; OI is overlaid from meta-asset-ctxs in fetch_complete_perps_data
        df['open_interest'] = 0
        df['orderbook_mid_open'] = 0
        df['orderbook_mid_close'] = 0
        
        df['ticker'] = ticker
        
        # Trim to requested limit
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} candle records for {ticker}")
        return df
    
    def fetch_price_history_for_trade_review(
        self,
        ticker: str,
        trade_time: datetime,
        hours_before: int = 4,
        hours_after: int = 24,
        resolution: str = "15MINS"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch price history before and after a perps trade for Post-Trade Review.
        
        Mirrors the BirdeyeFetcher.fetch_price_history_for_trade_review() interface
        so the PostTradeReviewer engine can consume data identically for both
        meme (Birdeye) and perps (internal API) flows.
        
        Args:
            ticker: Trading pair (e.g., 'BTC-USD')
            trade_time: When the trade was executed
            hours_before: Hours of data to fetch before the trade (default: 4)
            hours_after: Hours of data to fetch after the trade (default: 24)
            resolution: Candle resolution (default: 15MINS for ~same granularity as Birdeye 15m)
            
        Returns:
            Tuple of (price_history_before, price_history_after) as lists of dicts
            Each dict has: {'unixTime': timestamp, 'c': close_price, 'v': volume}
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
        
        logger.info(f"Fetching perps price history for post-trade review: {ticker} "
                    f"({hours_before}h before, {hours_after}h after trade)")
        
        price_before = []
        price_after = []
        
        try:
            # Fetch candles BEFORE the trade
            df_before = self.fetch_candles(
                ticker=ticker,
                resolution=resolution,
                from_date=start_before.strftime('%Y-%m-%dT%H:%M:%SZ'),
                to_date=end_before.strftime('%Y-%m-%dT%H:%M:%SZ'),
                limit=200
            )
            
            if not df_before.empty and 'close' in df_before.columns:
                for _, row in df_before.iterrows():
                    ts = row.get('timestamp')
                    if ts is not None:
                        if isinstance(ts, pd.Timestamp):
                            unix_ts = int(ts.timestamp())
                        else:
                            unix_ts = int(pd.Timestamp(ts).timestamp())
                        price_before.append({
                            'unixTime': unix_ts,
                            'c': float(row['close']),
                            'v': float(row.get('volume', 0))
                        })
            
            # Fetch candles AFTER the trade
            df_after = self.fetch_candles(
                ticker=ticker,
                resolution=resolution,
                from_date=start_after.strftime('%Y-%m-%dT%H:%M:%SZ'),
                to_date=end_after.strftime('%Y-%m-%dT%H:%M:%SZ'),
                limit=200
            )
            
            if not df_after.empty and 'close' in df_after.columns:
                for _, row in df_after.iterrows():
                    ts = row.get('timestamp')
                    if ts is not None:
                        if isinstance(ts, pd.Timestamp):
                            unix_ts = int(ts.timestamp())
                        else:
                            unix_ts = int(pd.Timestamp(ts).timestamp())
                        price_after.append({
                            'unixTime': unix_ts,
                            'c': float(row['close']),
                            'v': float(row.get('volume', 0))
                        })
            
            logger.info(f"Perps price history: {len(price_before)} before, {len(price_after)} after")
            
        except Exception as e:
            logger.warning(f"Failed to fetch perps price history for trade review: {e}")
        
        return price_before, price_after
    
    def fetch_funding_rates_direct(self, ticker: str, limit: int = 200) -> pd.DataFrame:
        """
        Fetch historical funding rate data via HyperEVM /hyper-evm/funding-history.
        
        This is faster than fetch_funding_rates() which goes through
        the block-liquidity/trades endpoint (trade records).
        
        Args:
            ticker: Trading pair (e.g., 'BTC-USD')
            limit: Maximum funding records to fetch (default: 200)
            
        Returns:
            DataFrame with funding rate data (columns: timestamp, funding_rate, funding_price, ticker)
        """
        coin = self.config.to_hyper_coin(ticker)
        
        # Funding records are hourly; compute lookback from limit
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(hours=limit)
        
        logger.info(f"Fetching funding rates (direct) for {ticker} ({coin}, limit={limit})...")
        
        params = {
            'coin': coin,
            'startTime': start_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'endTime': end_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        }
        
        data = self._make_request(self.config.ENDPOINT_FUNDING_HISTORY, params, timeout=15)
        
        if not data or not data.get('historical'):
            logger.warning(f"No funding rate data (direct) for {ticker}")
            return pd.DataFrame()
        
        records = data['historical']
        df = pd.DataFrame(records)
        
        # Map HyperEVM fields to standardized names matching old DYDX output
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        if 'fundingRate' in df.columns:
            df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce').fillna(0)
        
        # funding_price not available from HyperEVM; set to 0 (not consumed downstream)
        df['funding_price'] = 0
        
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['ticker'] = ticker
        
        cols_to_keep = [c for c in ['timestamp', 'funding_rate', 'funding_price', 'ticker'] if c in df.columns]
        df = df[cols_to_keep]
        
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        
        logger.info(f"Fetched {len(df)} funding records (direct) for {ticker}")
        return df
    
    def fetch_funding_rates(self, ticker: str, limit: int = 200, max_pages: int = 25) -> pd.DataFrame:
        """
        Fetch historical funding rate data from block-liquidity trades endpoint.
        Requests a time range of trades and extracts unique funding rates from records.

        NOTE: For predictions, prefer fetch_funding_rates_direct() which uses the
        dedicated /hyper-evm/funding-history endpoint when available.

        Args:
            ticker: Trading pair (e.g., 'BTC-USD')
            limit: Maximum unique funding records to return (default: 200)
            max_pages: Unused (kept for API compatibility)

        Returns:
            DataFrame with funding rate data
        """
        coin_base = ticker.split('-')[0].upper() if '-' in ticker else ticker.upper()
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)
        start_ts = start_dt.strftime('%Y-%m-%dT00:00:00.000Z')
        end_ts = end_dt.strftime('%Y-%m-%dT00:00:00.000Z')

        endpoint = f"{self.config.ENDPOINT_BLOCK_LIQUIDITY_TRADES}/{coin_base}"
        params = {'start_timestamp': start_ts, 'end_timestamp': end_ts}

        logger.info(f"Fetching funding rates for {ticker} via block-liquidity ({start_ts} to {end_ts})...")

        data = self._make_request(endpoint, params, timeout=60)
        if not data:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        records = data.get('data', data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            records = [records] if records else []

        all_funding_records = {}
        for record in records:
            funding_time = record.get('fundingTime')
            if funding_time and funding_time not in all_funding_records:
                all_funding_records[funding_time] = {
                    'fundingTime': funding_time,
                    'funding_rate': float(record.get('fundingRate', 0) or 0),
                    'funding_premium': float(record.get('fundingPremium', 0) or 0),
                    'funding_price': float(record.get('price', 0) or 0),
                    'symbol': record.get('symbol', ticker)
                }
                if len(all_funding_records) >= limit:
                    break

        logger.info(f"Found {len(all_funding_records)} unique funding records for {ticker}")

        if not all_funding_records:
            logger.warning(f"No funding rate data for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(list(all_funding_records.values()))
        
        # Convert fundingTime to timestamp
        if 'fundingTime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['ticker'] = ticker
        
        logger.info(f"Fetched {len(df)} unique funding records for {ticker}")
        return df
    
    def fetch_market_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current market data via HyperEVM /hyper-evm/meta-asset-ctxs.
        
        The endpoint returns all coins in a single call; we filter client-side
        and map fields to match the downstream-expected dict shape.
        
        Args:
            ticker: Trading pair (e.g., 'BTC-USD')
            
        Returns:
            Dictionary with market data or None
        """
        coin = self.config.to_hyper_coin(ticker)
        logger.info(f"Fetching market data for {ticker} ({coin})...")
        
        data = self._make_request(self.config.ENDPOINT_MARKETS, {})
        
        if not data or not data.get('data'):
            logger.warning(f"No market data for {ticker}")
            return None
        
        # Find the matching coin entry
        coin_entry = None
        for entry in data['data']:
            if entry.get('name') == coin:
                coin_entry = entry
                break
        
        if not coin_entry:
            logger.warning(f"Coin {coin} not found in meta-asset-ctxs response")
            return None
        
        oracle_price = float(coin_entry.get('oraclePx', 0))
        prev_day_price = float(coin_entry.get('prevDayPx', 0))
        max_leverage = float(coin_entry.get('maxLeverage', 1))
        
        market_data = {
            'ticker': ticker,
            'oracle_price': oracle_price,
            'price_change_24h': oracle_price - prev_day_price,
            'volume_24h': float(coin_entry.get('dayNtlVlm', 0)),
            'trades_24h': 0,
            'next_funding_rate': float(coin_entry.get('funding', 0)),
            'open_interest': float(coin_entry.get('openInterest', 0)),
            'initial_margin': 1.0 / max_leverage if max_leverage > 0 else 0,
            'maintenance_margin': 0,
            'status': 'ACTIVE'
        }
        
        logger.info(f"Fetched market data for {ticker}")
        return market_data
    
    def fetch_trades(self, ticker: str, limit: int = 1000) -> pd.DataFrame:
        """
        Deprecated: DYDX /trades endpoint removed. Use fetch_perps_trades_by_coin() instead.
        Returns empty DataFrame to preserve API compatibility.
        """
        logger.info(f"fetch_trades() is deprecated (DYDX endpoint removed). Returning empty DataFrame for {ticker}.")
        return pd.DataFrame()
    
    def fetch_perps_trades_by_coin(
        self,
        coin: str,
        start_time: str = None,
        end_time: str = None,
        limit: int = 100,
        max_pages: int = 10
    ) -> pd.DataFrame:
        """
        Fetch perps trades by coin from block-liquidity endpoint.
        URL: /block-liquidity/trades/{coin}?start_timestamp=ISO&end_timestamp=ISO

        Args:
            coin: Coin symbol or ticker (e.g., 'BTC-USD', 'ETH-USD', 'BTC')
            start_time: Start date (YYYY-MM-DD format)
            end_time: End date (YYYY-MM-DD format)
            limit: Unused (kept for API compatibility)
            max_pages: Unused (kept for API compatibility)

        Returns:
            DataFrame with trade data including buyer/seller addresses
        """
        # Path uses base symbol only (BTC, ETH)
        coin_base = coin.split('-')[0].upper() if '-' in coin else coin.upper()

        # Default to last 1 day (24 hours) for focused signal effectiveness
        if not start_time:
            start_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if not end_time:
            end_time = datetime.now().strftime('%Y-%m-%d')

        # ISO 8601 timestamps for API (e.g. 2026-03-01T00:00:00.000Z)
        start_ts = f"{start_time}T00:00:00.000Z"
        end_ts = f"{end_time}T00:00:00.000Z"

        endpoint = f"{self.config.ENDPOINT_BLOCK_LIQUIDITY_TRADES}/{coin_base}"
        params = {
            'start_timestamp': start_ts,
            'end_timestamp': end_ts
        }

        logger.info(f"Fetching perps trades for {coin_base} from {start_ts} to {end_ts}...")

        data = self._make_request(endpoint, params)

        if not data:
            logger.warning(f"No response for {coin_base}")
            return pd.DataFrame()

        # Accept both { data: [...] } and direct array
        records = data.get('data', data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            records = [records] if records else []

        all_records = records
        
        if not all_records:
            logger.warning(f"No perps trades found for {coin_base}")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        
        logger.info(f"Raw trade columns from block-liquidity API: {list(df.columns)}")
        
        # Standardize column names (handle both old and new API formats)
        column_mapping = {
            # New API format
            'id': 'trade_id',
            'symbol': 'coin',
            'side': 'side',  # BUY/SELL or A/B
            'price': 'price',
            'size': 'size',
            'timestamp': 'timestamp_str',
            'source': 'source',
            'fundingRate': 'funding_rate',
            'fundingPremium': 'funding_premium',
            'fundingTime': 'funding_time',
            # Block-liquidity API returns HyperLiquid field names
            'px': 'price',
            'sz': 'size',
            # Old API format fallbacks
            'tid': 'trade_id',
            'time': 'time_ms',
            'hash': 'tx_hash',
            'buyer': 'buyer_address',
            'seller': 'seller_address',
            'createdAt': 'created_at',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert types
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if 'size' in df.columns:
            df['size'] = pd.to_numeric(df['size'], errors='coerce')
        if 'funding_rate' in df.columns:
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
        
        # Normalize side values: HyperLiquid uses A (ask/sell) / B (bid/buy)
        if 'side' in df.columns:
            df['side'] = df['side'].replace({'A': 'SELL', 'B': 'BUY', 'a': 'SELL', 'b': 'BUY'})
            df['side'] = df['side'].str.upper()

        # Handle timestamp from various formats
        if 'timestamp_str' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_str'])
        elif 'created_at' in df.columns:
            df['timestamp'] = pd.to_datetime(df['created_at'])
        elif 'time_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['time_ms'], errors='coerce'), unit='ms', errors='coerce')
        
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        # Calculate trade value in USD (use absolute size for value)
        if 'price' in df.columns and 'size' in df.columns:
            df['trade_value_usd'] = df['price'] * df['size'].abs()
        
        # Ensure trade_id exists for aggregation
        if 'trade_id' not in df.columns:
            df['trade_id'] = range(len(df))

        # Create pseudo buyer/seller addresses from side if not present
        if 'buyer_address' not in df.columns:
            if 'source' in df.columns:
                df['trader_address'] = df['source'].fillna('unknown')
            else:
                df['trader_address'] = 'unknown_' + df.index.astype(str)
            df['buyer_address'] = df.apply(
                lambda x: x['trader_address'] if x.get('side') == 'BUY' else 'market', axis=1
            )
            df['seller_address'] = df.apply(
                lambda x: x['trader_address'] if x.get('side') == 'SELL' else 'market', axis=1
            )
        
        logger.info(f"Fetched {len(df)} perps trades for {coin_base}")
        return df

    def fetch_hyper_liquid_users(self, limit: int = 100, max_pages: int = 5) -> List[str]:
        """
        Fetch HyperLiquid user addresses
        
        Args:
            limit: Records per page
            max_pages: Maximum pages to fetch
            
        Returns:
            List of user wallet addresses
        """
        logger.info("Fetching HyperLiquid users...")
        
        all_users = []
        page = 1
        
        while page <= max_pages:
            params = {'page': page, 'limit': limit}
            data = self._make_request('/hyper-evm/hyper-liquid-users', params)
            
            if not data or not data.get('users'):
                break
            
            users_data = data['users']
            if isinstance(users_data, dict) and 'users' in users_data:
                users = [u.get('user') for u in users_data['users'] if u.get('user')]
            else:
                break
            
            if not users:
                break
            
            all_users.extend(users)
            
            last_page = users_data.get('lastPage', 1)
            if page >= last_page:
                break
            
            page += 1
            time.sleep(0.1)
        
        logger.info(f"Fetched {len(all_users)} HyperLiquid user addresses")
        return all_users

    def fetch_user_open_positions(self, user_address: str, timeout: int = 10) -> List[Dict]:
        """
        Fetch a user's current open perps positions from HyperLiquid clearinghouse state.

        Args:
            user_address: EVM wallet address (0x...)
            timeout: Request timeout in seconds

        Returns:
            List of asset position dicts, each containing 'type' and 'position'
            (with fields: coin, szi, entryPx, leverage, unrealizedPnl, etc.).
            Empty list if no positions or on failure.
        """
        logger.info(f"Fetching open positions for {user_address[:10]}...")
        params = {'userAddress': user_address}
        data = self._make_request('/hyper-evm/fetch-user-open-positions', params, timeout=timeout)
        if not data:
            return []
        positions = data.get('assetPositions', [])
        logger.info(f"Found {len(positions)} open position(s) for {user_address[:10]}...")
        return positions

    def _fetch_user_perps_trades_via_user_profiling(
        self, user_address: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """Fetch user perps trades from internal API /user-profiling/perps-trade; paginate, cap 500; return normalized DataFrame."""
        base_url = getattr(api_config, 'INTERNAL_BASE_URL', None) or getattr(api_config, 'BASE_URL', self.base_url)
        endpoint = self.config.ENDPOINT_USER_PERPS_TRADES
        from_iso = from_date[:10] if from_date else ''
        to_iso = to_date[:10] if to_date else ''
        logger.info(f"Fetching perps trades for user {user_address[:10]}... via user-profiling from {from_iso} to {to_iso}")

        all_records = []
        max_records = 500
        max_pages = 25
        page = 1

        while page <= max_pages:
            params = {'from': from_iso, 'to': to_iso, 'address': user_address, 'page': page}
            data = self._make_request(endpoint, params, base_url=base_url)
            if not data:
                break
            records = data.get('data', data) if isinstance(data, dict) else data
            if not isinstance(records, list):
                records = [records] if records else []
            if not records:
                break
            all_records.extend(records)
            if len(all_records) >= max_records:
                break
            total_pages = (data.get('meta') or {}).get('totalPages') or (data.get('totalPages'))
            if total_pages is not None and page >= total_pages:
                break
            page += 1

        if not all_records:
            logger.warning(f"No perps trades found for user {user_address[:10]}... (user-profiling)")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df.loc[:, ~df.columns.duplicated()]

        column_mapping = {}
        if 'token' in df.columns and 'coin' not in df.columns:
            column_mapping['token'] = 'coin'
        if 'symbol' in df.columns and 'coin' not in df.columns:
            column_mapping['symbol'] = 'coin'
        if 'px' in df.columns and 'price' not in df.columns:
            column_mapping['px'] = 'price'
        if 'entry_px' in df.columns and 'price' not in df.columns:
            column_mapping['entry_px'] = 'price'
        if 'sz' in df.columns and 'size' not in df.columns:
            column_mapping['sz'] = 'size'
        if 'szi' in df.columns and 'size' not in df.columns:
            column_mapping['szi'] = 'size'
        if 'time' in df.columns:
            column_mapping['time'] = 'time_ms'
        if 'timestamp' in df.columns and 'time_ms' not in df.columns:
            column_mapping['timestamp'] = 'time_ms'
        if 'dir' in df.columns:
            column_mapping['dir'] = 'direction'
        if 'closedPnl' in df.columns:
            column_mapping['closedPnl'] = 'closed_pnl'
        if 'realizedPnl' in df.columns and 'closed_pnl' not in df.columns:
            column_mapping['realizedPnl'] = 'closed_pnl'
        if 'realized_pnl' in df.columns and 'closed_pnl' not in df.columns:
            column_mapping['realized_pnl'] = 'closed_pnl'
        if 'hash' in df.columns:
            column_mapping['hash'] = 'tx_hash'
        if 'feeToken' in df.columns:
            column_mapping['feeToken'] = 'fee_token'
        if 'side' in df.columns and 'direction' not in df.columns:
            column_mapping['side'] = 'direction'
        if column_mapping:
            df = df.rename(columns=column_mapping)
        df = df.loc[:, ~df.columns.duplicated()]

        if 'side' in df.columns and 'direction' not in df.columns:
            df['direction'] = df['side'].map({'A': 'Short', 'B': 'Long', 'a': 'Short', 'b': 'Long', 'sell': 'Short', 'buy': 'Long'}).fillna(df['side'])
        if 'direction' not in df.columns and 'size' in df.columns:
            df['direction'] = df['size'].apply(lambda x: 'Long' if pd.notna(x) and float(x) > 0 else 'Short')
        # Normalize direction to Long/Short for analyzer (str.contains('Long'/'Short'))
        if 'direction' in df.columns:
            d = df['direction'].astype(str).str.lower()
            df.loc[d.isin(['b', 'buy']), 'direction'] = 'Long'
            df.loc[d.isin(['a', 'sell']), 'direction'] = 'Short'

        numeric_cols = ['price', 'size', 'closed_pnl', 'realized_pnl', 'fee', 'position_value', 'unrealized_pnl']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass

        if 'time_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time_ms'], unit='ms', errors='coerce')
        elif 'timestamp' in df.columns:
            ts = df['timestamp']
            if ts.dtype == 'int64' or (ts.dtype == object and str(ts.iloc[0]).isdigit()):
                df['timestamp'] = pd.to_datetime(pd.to_numeric(ts, errors='coerce'), unit='ms', errors='coerce')
            else:
                df['timestamp'] = pd.to_datetime(ts, errors='coerce')

        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        if len(df) > max_records:
            df = df.head(max_records).copy()
            logger.info(f"Capped perps trades to {max_records} for behaviour stats")

        if 'position_value' in df.columns:
            df['trade_value'] = df['position_value']
        elif 'price' in df.columns and 'size' in df.columns:
            try:
                df['trade_value'] = df['price'].values * df['size'].values
            except Exception:
                df['trade_value'] = 0
        else:
            df['trade_value'] = 0
        if 'realized_pnl' in df.columns and 'closed_pnl' not in df.columns:
            df['closed_pnl'] = df['realized_pnl']

        logger.info(f"Fetched {len(df)} perps records for user {user_address[:10]}... (user-profiling)")
        return df

    def fetch_user_perps_trades(
        self,
        user_address: str,
        from_date: str = None,
        to_date: str = None,
        coin: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch perps data for a user. When PERPS_USER_TRADES_USE_USER_PROFILING is True, uses
        internal API GET /user-profiling/perps-trade (from, to, address, page). Otherwise uses
        block-liquidity/historical-positions. Returns DataFrame with columns expected by
        analyze_user_perps_profile: coin, closed_pnl, timestamp, direction, trade_value, etc.

        When from_date/to_date are not provided, defaults to last 3 months.
        """
        # Default to last 3 months when no date range provided
        if not from_date:
            from_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')

        use_user_profiling = getattr(self.config, 'PERPS_USER_TRADES_USE_USER_PROFILING', True)
        if use_user_profiling:
            return self._fetch_user_perps_trades_via_user_profiling(user_address, from_date, to_date)

        # Block-liquidity path
        start_ts = f"{from_date[:10]}T00:00:00.000Z"
        end_ts = f"{to_date[:10]}T00:00:00.000Z"

        endpoint = self.config.ENDPOINT_BLOCK_LIQUIDITY_HISTORICAL_POSITIONS
        params = {
            'address': user_address,
            'start_timestamp': start_ts,
            'end_timestamp': end_ts
        }

        logger.info(f"Fetching perps trades for user {user_address[:10]}... from {start_ts} to {end_ts}")

        all_records = []
        request_params = dict(params)
        max_pages = 20
        max_records = 500  # Behaviour stats cap: most recent 500 trades (180d window set by caller)
        page_count = 0

        while True:
            data = self._make_request(endpoint, request_params)
            if not data:
                break
            page_count += 1
            records = data.get('data', data) if isinstance(data, dict) else data
            if not isinstance(records, list):
                records = [records] if records else []
            all_records.extend(records)

            if len(all_records) >= max_records:
                logger.info(f"Reached max_records ({max_records}), stopping pagination")
                break
            if page_count >= max_pages:
                logger.info(f"Reached max_pages ({max_pages}), stopping pagination")
                break

            pagination = data.get('pagination') if isinstance(data, dict) else None
            if not pagination or not pagination.get('has_more'):
                break
            next_cursor = pagination.get('next_cursor')
            if not next_cursor:
                break
            request_params = dict(params)
            request_params['cursor'] = next_cursor

        if not all_records:
            logger.warning(f"No perps trades found for user {user_address[:10]}...")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        
        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Standardize column names (avoid conflicts)
        # Support both trade-list APIs and block-liquidity historical-positions (position snapshots)
        column_mapping = {}
        if 'token' in df.columns and 'coin' not in df.columns:
            column_mapping['token'] = 'coin'
        if 'px' in df.columns and 'price' not in df.columns:
            column_mapping['px'] = 'price'
        if 'entry_px' in df.columns and 'price' not in df.columns:
            column_mapping['entry_px'] = 'price'
        if 'sz' in df.columns and 'size' not in df.columns:
            column_mapping['sz'] = 'size'
        if 'szi' in df.columns and 'size' not in df.columns:
            column_mapping['szi'] = 'size'
        if 'time' in df.columns:
            column_mapping['time'] = 'time_ms'
        if 'timestamp' in df.columns and 'time_ms' not in df.columns:
            # block-liquidity returns timestamp in ms; normalize to time_ms for datetime parsing
            column_mapping['timestamp'] = 'time_ms'
        if 'dir' in df.columns:
            column_mapping['dir'] = 'direction'
        if 'closedPnl' in df.columns:
            column_mapping['closedPnl'] = 'closed_pnl'
        if 'hash' in df.columns:
            column_mapping['hash'] = 'tx_hash'
        if 'feeToken' in df.columns:
            column_mapping['feeToken'] = 'fee_token'
        if 'startPosition' in df.columns:
            column_mapping['startPosition'] = 'start_position'
        if 'oid' in df.columns:
            column_mapping['oid'] = 'order_id'
        if 'tid' in df.columns:
            column_mapping['tid'] = 'trade_id'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Remove duplicate columns again after rename
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Map side field: A = sell, B = buy (trade APIs)
        if 'side' in df.columns:
            df['side'] = df['side'].map({'A': 'sell', 'B': 'buy'}).fillna(df['side'])
        
        # Infer direction from size for position snapshots (szi: positive = long, negative = short)
        if 'direction' not in df.columns and 'size' in df.columns:
            df['direction'] = df['size'].apply(lambda x: 'Long' if pd.notna(x) and float(x) > 0 else 'Short')
        
        # Convert types safely
        numeric_cols = ['price', 'size', 'closed_pnl', 'realized_pnl', 'fee', 'start_position', 'position_value', 'unrealized_pnl', 'cum_funding_since_open']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass
        
        if 'time_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time_ms'], unit='ms')
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        # Cap at 500 trades for behaviour stats (most recent already first)
        if len(df) > 500:
            df = df.head(500).copy()
            logger.info(f"Capped perps trades to 500 for behaviour stats")
        
        # Calculate trade value (from price*size or use position_value when available)
        if 'position_value' in df.columns:
            df['trade_value'] = df['position_value']
        elif 'price' in df.columns and 'size' in df.columns:
            try:
                df['trade_value'] = df['price'].values * df['size'].values
            except Exception:
                df['trade_value'] = 0
        else:
            df['trade_value'] = 0
        
        # Use realized_pnl if closed_pnl is not available (position snapshots have no closed_pnl)
        if 'realized_pnl' in df.columns and 'closed_pnl' not in df.columns:
            df['closed_pnl'] = df['realized_pnl']
        
        logger.info(f"Fetched {len(df)} perps records for user {user_address[:10]}...")
        return df
    
    def analyze_user_perps_profile(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze user's perps trading profile from their trade history
        
        Args:
            trades_df: DataFrame from fetch_user_perps_trades
            
        Returns:
            Dictionary with user profile metrics
        """
        if trades_df.empty:
            return self._empty_user_perps_profile()
        
        # Basic stats
        total_trades = len(trades_df)
        unique_coins = trades_df['coin'].nunique() if 'coin' in trades_df.columns else 0
        
        # PnL analysis
        if 'closed_pnl' in trades_df.columns:
            total_pnl = trades_df['closed_pnl'].sum()
            winning_trades = trades_df[trades_df['closed_pnl'] > 0]
            losing_trades = trades_df[trades_df['closed_pnl'] < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = winning_trades['closed_pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['closed_pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = (avg_win * win_count) / (avg_loss * loss_count) if loss_count > 0 and avg_loss > 0 else 0
        else:
            total_pnl = 0
            win_count = 0
            loss_count = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Direction analysis
        if 'direction' in trades_df.columns:
            long_trades = trades_df[trades_df['direction'].str.contains('Long', case=False, na=False)]
            short_trades = trades_df[trades_df['direction'].str.contains('Short', case=False, na=False)]
            long_pct = len(long_trades) / total_trades * 100 if total_trades > 0 else 50
            short_pct = len(short_trades) / total_trades * 100 if total_trades > 0 else 50
            
            # Determine bias
            if long_pct > 65:
                direction_bias = "LONG_BIAS"
            elif short_pct > 65:
                direction_bias = "SHORT_BIAS"
            else:
                direction_bias = "BALANCED"
        else:
            long_pct = 50
            short_pct = 50
            direction_bias = "UNKNOWN"
        
        # Volume analysis
        if 'trade_value' in trades_df.columns:
            total_volume = trades_df['trade_value'].sum()
            avg_trade_size = trades_df['trade_value'].mean()
        else:
            total_volume = 0
            avg_trade_size = 0
        
        # Fee analysis
        if 'fee' in trades_df.columns:
            total_fees = trades_df['fee'].sum()
        else:
            total_fees = 0
        
        # Time analysis
        if 'timestamp' in trades_df.columns:
            ts = trades_df['timestamp']
            # Normalize to datetime if still numeric (e.g. from block-liquidity as int ms)
            if pd.api.types.is_numeric_dtype(ts):
                ts = pd.to_datetime(ts, unit='ms', errors='coerce')
            else:
                ts = pd.to_datetime(ts, errors='coerce')
            first_trade = ts.min()
            last_trade = ts.max()
            if pd.notna(first_trade) and pd.notna(last_trade):
                delta = last_trade - first_trade
                # Handle both datetime timedelta (.days) and numpy timedelta64
                if hasattr(delta, 'days'):
                    raw_days = int(delta.days)
                else:
                    raw_days = int(pd.Timedelta(delta).days)
                # Use at least 1 day when there is activity so trades_per_day is meaningful
                trading_period_days = max(1, raw_days) if total_trades > 0 else max(0, raw_days)
            else:
                trading_period_days = 0
            trades_per_day = total_trades / max(trading_period_days, 1)
        else:
            first_trade = None
            last_trade = None
            trading_period_days = 0
            trades_per_day = 0
        
        # Coin preference (ensure top_coins values are JSON-serializable native int)
        if 'coin' in trades_df.columns:
            coin_counts = trades_df['coin'].value_counts()
            top_coins = {str(k): int(v) for k, v in coin_counts.head(3).to_dict().items()}
            favorite_coin = str(coin_counts.index[0]) if len(coin_counts) > 0 else None
        else:
            top_coins = {}
            favorite_coin = None
        
        # From latest position snapshot (block-liquidity): unrealized PnL and funding since open
        latest_unrealized_pnl = 0.0
        total_funding_since_open = 0.0
        if 'timestamp' in trades_df.columns:
            latest_ts = trades_df['timestamp'].max()
            latest_mask = trades_df['timestamp'] == latest_ts
            if 'unrealized_pnl' in trades_df.columns:
                latest_unrealized_pnl = float(trades_df.loc[latest_mask, 'unrealized_pnl'].fillna(0).sum())
            if 'cum_funding_since_open' in trades_df.columns:
                total_funding_since_open = float(trades_df.loc[latest_mask, 'cum_funding_since_open'].fillna(0).sum())
        
        # Trader type classification
        if trades_per_day > 20:
            trader_type = "SCALPER"
        elif trades_per_day > 5:
            trader_type = "DAY_TRADER"
        elif trades_per_day > 1:
            trader_type = "SWING_TRADER"
        else:
            trader_type = "POSITION_TRADER"
        
        return {
            'total_trades': total_trades,
            'unique_coins': unique_coins,
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 2),
            'win_count': win_count,
            'loss_count': loss_count,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'long_pct': round(long_pct, 2),
            'short_pct': round(short_pct, 2),
            'direction_bias': direction_bias,
            'total_volume': round(total_volume, 2),
            'avg_trade_size': round(avg_trade_size, 2),
            'total_fees': round(total_fees, 4),
            'trading_period_days': trading_period_days,
            'trades_per_day': round(trades_per_day, 2),
            'trader_type': trader_type,
            'top_coins': top_coins,
            'favorite_coin': favorite_coin,
            'first_trade': pd.Timestamp(first_trade).isoformat() if first_trade is not None and pd.notna(first_trade) else None,
            'last_trade': pd.Timestamp(last_trade).isoformat() if last_trade is not None and pd.notna(last_trade) else None,
            'is_profitable': total_pnl > 0,
            'is_active': trades_per_day >= 1,
            'latest_unrealized_pnl': round(latest_unrealized_pnl, 4),
            'total_funding_since_open': round(total_funding_since_open, 4)
        }
    
    def _empty_user_perps_profile(self) -> Dict[str, Any]:
        """Return empty user perps profile"""
        return {
            'total_trades': 0,
            'unique_coins': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'win_count': 0,
            'loss_count': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'long_pct': 50,
            'short_pct': 50,
            'direction_bias': 'UNKNOWN',
            'total_volume': 0,
            'avg_trade_size': 0,
            'total_fees': 0,
            'trading_period_days': 0,
            'trades_per_day': 0,
            'trader_type': 'UNKNOWN',
            'top_coins': {},
            'favorite_coin': None,
            'first_trade': None,
            'last_trade': None,
            'is_profitable': False,
            'is_active': False,
            'latest_unrealized_pnl': 0,
            'total_funding_since_open': 0
        }
    
    def analyze_large_traders(
        self,
        trades_df: pd.DataFrame,
        top_n: int = 20,
        min_trade_value: float = 0
    ) -> Dict[str, Any]:
        """
        Analyze large traders from perps trade data (whale equivalent for perps)
        
        Args:
            trades_df: DataFrame from fetch_perps_trades_by_coin
            top_n: Number of top traders to analyze
            min_trade_value: Minimum trade value to consider (0 = use percentile-based filter)
            
        Returns:
            Dictionary with large trader metrics
        """
        if trades_df.empty:
            return self._empty_large_trader_metrics()
        
        has_value = 'trade_value_usd' in trades_df.columns

        # Use percentile-based filter: keep top 50% of trades by value
        if has_value and min_trade_value <= 0:
            median_value = trades_df['trade_value_usd'].median()
            sig_trades = trades_df[trades_df['trade_value_usd'] >= median_value].copy()
        elif has_value and min_trade_value > 0:
            sig_trades = trades_df[trades_df['trade_value_usd'] >= min_trade_value].copy()
        else:
            sig_trades = trades_df.copy()
        
        if sig_trades.empty:
            sig_trades = trades_df.copy()
        
        # Total market volume = sum of all individual trades (each counted once)
        market_volume = sig_trades['trade_value_usd'].sum() if has_value else 0
        
        # Buy/sell volume by taker side
        if 'side' in sig_trades.columns and has_value:
            total_buy_volume = sig_trades.loc[sig_trades['side'] == 'BUY', 'trade_value_usd'].sum()
            total_sell_volume = sig_trades.loc[sig_trades['side'] == 'SELL', 'trade_value_usd'].sum()
        else:
            total_buy_volume = 0
            total_sell_volume = 0

        # If side-based split missed volume (e.g. unknown side values), use total
        if total_buy_volume + total_sell_volume == 0 and market_volume > 0:
            total_buy_volume = market_volume / 2
            total_sell_volume = market_volume / 2
        
        # Aggregate by trader if addresses are available
        trader_data = []
        has_buyer = 'buyer_address' in sig_trades.columns
        has_seller = 'seller_address' in sig_trades.columns
        has_size = 'size' in sig_trades.columns

        if has_buyer and has_seller and has_value:
            agg_dict = {'trade_value_usd': 'sum'}
            rename_buy = {'trade_value_usd': 'buy_volume'}
            rename_sell = {'trade_value_usd': 'sell_volume'}
            if has_size:
                agg_dict['size'] = 'sum'
                rename_buy['size'] = 'buy_size'
                rename_sell['size'] = 'sell_size'
            if 'trade_id' in sig_trades.columns:
                agg_dict['trade_id'] = 'count'
                rename_buy['trade_id'] = 'buy_count'
                rename_sell['trade_id'] = 'sell_count'

            buyer_agg = sig_trades.groupby('buyer_address').agg(agg_dict).rename(columns=rename_buy)
            seller_agg = sig_trades.groupby('seller_address').agg(agg_dict).rename(columns=rename_sell)
            
            all_addresses = (set(buyer_agg.index) | set(seller_agg.index)) - {'market'}
            
            for addr in all_addresses:
                buy_vol = buyer_agg.loc[addr, 'buy_volume'] if addr in buyer_agg.index else 0
                sell_vol = seller_agg.loc[addr, 'sell_volume'] if addr in seller_agg.index else 0
                buy_count = buyer_agg.loc[addr, 'buy_count'] if addr in buyer_agg.index and 'buy_count' in buyer_agg.columns else 0
                sell_count = seller_agg.loc[addr, 'sell_count'] if addr in seller_agg.index and 'sell_count' in seller_agg.columns else 0
                
                trader_data.append({
                    'address': addr,
                    'buy_volume': float(buy_vol),
                    'sell_volume': float(sell_vol),
                    'net_volume': float(buy_vol - sell_vol),
                    'total_volume': float(buy_vol + sell_vol),
                    'buy_count': int(buy_count),
                    'sell_count': int(sell_count),
                    'total_trades': int(buy_count + sell_count)
                })
        
        # Calculate metrics
        if trader_data:
            traders_df_agg = pd.DataFrame(trader_data)
            traders_df_agg = traders_df_agg.sort_values('total_volume', ascending=False).reset_index(drop=True)
            top_traders = traders_df_agg.head(top_n)
            
            top_trader_volume = top_traders['total_volume'].sum()
            # Each trade is counted in both buyer_agg and seller_agg, so total
            # participant volume = 2 × market_volume. Concentration = share of that.
            total_participant_volume = traders_df_agg['total_volume'].sum()
            top_concentration = (top_trader_volume / total_participant_volume * 100) if total_participant_volume > 0 else 0
            top_concentration = min(100.0, top_concentration)
            large_trader_net = top_traders['net_volume'].sum()
            large_trader_count = len(top_traders)
            top_traders_list = top_traders.to_dict('records')[:5]
        else:
            top_concentration = 100.0
            large_trader_net = total_buy_volume - total_sell_volume
            large_trader_count = len(sig_trades)
            top_traders_list = []
        
        # Determine large trader state based on net flow
        net_ratio = large_trader_net / market_volume if market_volume > 0 else 0
        if net_ratio > 0.1:
            large_trader_state = "Accumulation"
        elif net_ratio < -0.1:
            large_trader_state = "Distribution"
        else:
            large_trader_state = "Neutral"
        
        buy_sell_ratio = total_buy_volume / total_sell_volume if total_sell_volume > 0 else 1.0
        
        return {
            'large_trader_count': large_trader_count,
            'top_concentration_pct': round(top_concentration, 2),
            'large_trader_net_volume': round(float(large_trader_net), 2),
            'large_trader_state': large_trader_state,
            'buy_sell_ratio': round(float(buy_sell_ratio), 4),
            'total_buy_volume': round(float(total_buy_volume), 2),
            'total_sell_volume': round(float(total_sell_volume), 2),
            'total_trades_analyzed': len(trades_df),
            'significant_trades': len(sig_trades),
            'top_traders': top_traders_list
        }
    
    def _empty_large_trader_metrics(self) -> Dict[str, Any]:
        """Return empty large trader metrics"""
        return {
            'large_trader_count': 0,
            'top_concentration_pct': 0.0,
            'large_trader_net_volume': 0.0,
            'large_trader_state': 'Unknown',
            'buy_sell_ratio': 1.0,
            'total_buy_volume': 0.0,
            'total_sell_volume': 0.0,
            'total_trades_analyzed': 0,
            'significant_trades': 0,
            'top_traders': []
        }
    
    def fetch_whale_flow(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Fetch market-wide whale flow from block-liquidity.

        Returns aggregated whale positioning data: net flow direction, long/short volumes,
        and whale count. Uses the latest bucket for point-in-time values and averages the
        most recent 6 buckets (~6h) for a smoothed net_flow signal.

        Args:
            hours_back: How many hours of history to request (default 24)

        Returns:
            Dict with whale flow metrics and whale_flow_available flag
        """
        empty = {
            'whale_flow_net': 0.0,
            'whale_flow_net_avg': 0.0,
            'whale_flow_long_ratio': 0.5,
            'whale_flow_count_whales': 0,
            'whale_flow_long_volume': 0.0,
            'whale_flow_short_volume': 0.0,
            'whale_flow_total_volume': 0.0,
            'whale_flow_available': False
        }

        try:
            end_ts = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')
            start_ts = (datetime.utcnow() - timedelta(hours=hours_back)).strftime('%Y-%m-%dT%H:%M:%S.000Z')

            data = self._make_request(
                self.config.ENDPOINT_BLOCK_LIQUIDITY_WHALE_FLOW,
                {'start_timestamp': start_ts, 'end_timestamp': end_ts},
                timeout=10
            )

            if not data:
                logger.warning("No whale flow data returned")
                return empty

            records = data.get('data', []) if isinstance(data, dict) else []
            if not records:
                logger.warning("Whale flow response has no data buckets")
                return empty

            # Sort descending by timestamp so index 0 is the latest bucket
            records = sorted(records, key=lambda r: r.get('timestamp', 0), reverse=True)
            latest = records[0]

            long_vol = float(latest.get('long_volume_usdc_perp', 0))
            short_vol = float(latest.get('short_volume_usdc_perp', 0))
            total_vol = float(latest.get('total_volume_usdc_perp', 0)) or (long_vol + short_vol)

            # Rolling average of net_flow over the most recent 6 buckets (~6h)
            recent_buckets = records[:6]
            net_flows = [float(b.get('net_flow', 0)) for b in recent_buckets]
            net_flow_avg = sum(net_flows) / len(net_flows) if net_flows else 0.0

            result = {
                'whale_flow_net': float(latest.get('net_flow', 0)),
                'whale_flow_net_avg': round(net_flow_avg, 6),
                'whale_flow_long_ratio': round(long_vol / total_vol, 4) if total_vol > 0 else 0.5,
                'whale_flow_count_whales': int(latest.get('count_whales_perp', 0)),
                'whale_flow_long_volume': round(long_vol, 2),
                'whale_flow_short_volume': round(short_vol, 2),
                'whale_flow_total_volume': round(total_vol, 2),
                'whale_flow_available': True
            }

            logger.info(
                f"Whale flow fetched: net={result['whale_flow_net']:.4f}, "
                f"avg={result['whale_flow_net_avg']:.4f}, "
                f"whales={result['whale_flow_count_whales']}, "
                f"long_ratio={result['whale_flow_long_ratio']:.3f}"
            )
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch whale flow: {e}")
            return empty

    def fetch_complete_perps_data(
        self,
        ticker: str,
        resolution: str = "1HOUR",
        from_date: str = None,
        to_date: str = None,
        max_candles: int = 500,
        max_funding: int = 200,
        include_large_trader_analysis: bool = True,
        include_trades: bool = False,
        fast_mode: bool = False,
        funding_max_pages: int = 10,
        large_trader_max_pages: int = 5,
        market_data_timeout: int = None
    ) -> Dict[str, Any]:
        """
        Fetch all perps data for a ticker (optimized version)
        
        Data strategy for ±24h signal effectiveness:
        - Candles: Fetch full history (500 = ~21 days) for robust indicator calculation
        - Funding: Fetch full history (200 records) for funding rate analysis
        - Large trader analysis: Fetches last 24h only (aligned with signal window)
        - Market data: Current snapshot
        
        Args:
            ticker: Trading pair (e.g., 'BTC-USD')
            resolution: Candle resolution
            from_date: Start date for candles
            to_date: End date for candles
            max_candles: Maximum candle records to fetch (default: 500 = ~21 days of hourly data)
            max_funding: Maximum funding records to fetch (default: 200 = ~25 days)
            include_large_trader_analysis: Whether to fetch perps trades for large trader analysis
            include_trades: Whether to fetch trade data (can be slow)
            fast_mode: If True, uses fast /historical-funding endpoint and shorter timeouts
                       (ideal for predictions; training should use False for deeper data)
            funding_max_pages: Max pages for slow funding fetcher (ignored in fast_mode)
            large_trader_max_pages: Max pages for large trader trade fetching
            market_data_timeout: Custom timeout for market data request (seconds)
            
        Returns:
            Dictionary with candles, funding, market_data, trades, large_trader_metrics
        """
        mode_label = "fast" if fast_mode else "standard"
        logger.info(f"Fetching complete perps data for {ticker} ({mode_label} mode, max_candles={max_candles})...")
        
        # Fetch candles with limit
        candles = self.fetch_candles(ticker, resolution, from_date, to_date, limit=max_candles)
        
        # Limit candles to max_candles
        if len(candles) > max_candles:
            candles = candles.tail(max_candles).reset_index(drop=True)
        
        # Fetch funding rates
        if fast_mode:
            # FAST: Use dedicated /historical-funding endpoint (~1-4 seconds)
            funding = self.fetch_funding_rates_direct(ticker, limit=max_funding)
        else:
            # STANDARD: Use trades-based extraction (slow but more data)
            funding = self.fetch_funding_rates(ticker, limit=max_funding, max_pages=funding_max_pages)
        
        # Limit funding to max_funding
        if len(funding) > max_funding:
            funding = funding.tail(max_funding).reset_index(drop=True)
        
        # Market data (single request, use custom timeout if provided)
        if market_data_timeout:
            # Temporarily override timeout
            original_timeout = self.config.TIMEOUT
            self.config.TIMEOUT = market_data_timeout
            market_data = self.fetch_market_data(ticker)
            self.config.TIMEOUT = original_timeout
        else:
            market_data = self.fetch_market_data(ticker)
        
        # Trades are optional (can be slow)
        trades = pd.DataFrame()
        if include_trades:
            trades = self.fetch_trades(ticker, limit=100)
        
        # Large trader analysis is optional
        perps_trades = pd.DataFrame()
        large_trader_metrics = self._empty_large_trader_metrics()
        
        if include_large_trader_analysis:
            perps_trades = self.fetch_perps_trades_by_coin(
                ticker, limit=1000, max_pages=large_trader_max_pages
            )
            large_trader_metrics = self.analyze_large_traders(perps_trades)
        
        # Fetch market-wide whale flow (single fast request, non-blocking on failure)
        whale_flow = self.fetch_whale_flow(hours_back=24)
        
        # Merge funding rates into candles if both exist
        if not candles.empty and not funding.empty:
            # Normalize timestamps to same type and precision (datetime64[ns], timezone-naive)
            candles['timestamp'] = pd.to_datetime(candles['timestamp'], utc=True).dt.tz_localize(None).astype('datetime64[ns]')
            funding['timestamp'] = pd.to_datetime(funding['timestamp'], utc=True).dt.tz_localize(None).astype('datetime64[ns]')
            
            candles = pd.merge_asof(
                candles.sort_values('timestamp'),
                funding[['timestamp', 'funding_rate', 'funding_price']].sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
        
        # Add market data fields to candles if available
        if not candles.empty and market_data:
            for key, value in market_data.items():
                if key not in candles.columns:
                    candles[key] = value
        
        result = {
            'candles': candles,
            'funding': funding,
            'market_data': market_data,
            'trades': trades,
            'perps_trades': perps_trades,
            'large_trader_metrics': large_trader_metrics,
            'whale_flow': whale_flow,
            'ticker': ticker
        }
        
        logger.info(f"Complete perps data fetched for {ticker} ({mode_label}): "
                   f"{len(candles)} candles, {len(funding)} funding records"
                   + (f", {len(perps_trades)} perps trades" if include_large_trader_analysis else ""))
        
        return result
    
    def get_ticker_for_token(self, token_address: str) -> Optional[str]:
        """
        Map token address to ticker symbol
        
        For perps, the token_address might be a ticker symbol directly,
        or we map known tokens to their tickers.
        
        Args:
            token_address: Token address or symbol
            
        Returns:
            Ticker symbol (e.g., 'BTC-USD') or None
        """
        # Check if it's already a valid ticker format
        if '-' in token_address and len(token_address) <= 20:
            return token_address.upper()
        
        # Check reverse mapping
        for coin_id, ticker in self.config.TICKER_MAPPING.items():
            if token_address.lower() == coin_id.lower():
                return ticker
        
        # Check if token_address matches any ticker
        for ticker in self.config.TICKER_MAPPING.values():
            if token_address.upper() == ticker:
                return ticker
        
        logger.warning(f"No ticker mapping found for {token_address}")
        return None


# ==================== UNIFIED DATA FETCHER FACTORY ====================

class DataFetcherFactory:
    """
    Factory for creating appropriate data fetcher based on token type
    Implements Factory Pattern for clean instantiation
    """
    
    _meme_fetcher: Optional[DataFetcher] = None
    _perps_fetcher: Optional[PerpsDataFetcher] = None
    
    @classmethod
    def get_fetcher(cls, token_type: str = "meme") -> Any:
        """
        Get data fetcher for token type
        
        Args:
            token_type: 'meme' or 'perps'
            
        Returns:
            Appropriate data fetcher instance
        """
        token_type_lower = token_type.lower().strip()
        
        if token_type_lower == "perps":
            if cls._perps_fetcher is None:
                cls._perps_fetcher = PerpsDataFetcher()
            return cls._perps_fetcher
        else:
            if cls._meme_fetcher is None:
                cls._meme_fetcher = DataFetcher()
            return cls._meme_fetcher
    
    @classmethod
    def get_meme_fetcher(cls) -> DataFetcher:
        """Get meme token data fetcher"""
        return cls.get_fetcher("meme")
    
    @classmethod
    def get_perps_fetcher(cls) -> PerpsDataFetcher:
        """Get perps token data fetcher"""
        return cls.get_fetcher("perps")


if __name__ == "__main__":
    print("Data Fetcher Module")
    print("=" * 50)
    
    import sys
    
    # Test with sample mint
    fetcher = DataFetcher()
    
    # Test single mint
    test_mint = "GukSwR34F8ts9xdy27V5JLsn7LKHGhn739J7bbTY1giZ"
    
    if len(sys.argv) > 1 and sys.argv[1] == '--inspect':
        # Inspection mode
        fetcher.inspect_api_response(test_mint)
    elif len(sys.argv) > 1 and sys.argv[1] == '--perps':
        # Test perps fetcher
        print("\nTesting PerpsDataFetcher...")
        perps_fetcher = PerpsDataFetcher()
        ticker = "BTC-USD"
        
        print(f"\nFetching data for {ticker}...")
        data = perps_fetcher.fetch_complete_perps_data(ticker)
        
        print(f"\nResults:")
        print(f"  Candles: {len(data['candles'])} records")
        print(f"  Funding: {len(data['funding'])} records")
        print(f"  Market Data: {'Available' if data['market_data'] else 'N/A'}")
        print(f"  Trades: {len(data['trades'])} records")
        
        if len(data['candles']) > 0:
            print(f"\nCandle columns: {list(data['candles'].columns)}")
            print(f"Sample candle data:")
            print(data['candles'].head(2))
    else:
        # Normal test mode
        print(f"\nTesting data fetch for: {test_mint[:8]}...")
        
        data = fetcher.fetch_complete_data(test_mint, candle_days=90, holder_days=90, trade_days=90)
        
        print(f"\nResults:")
        print(f"  Candles: {len(data['candles'])} records")
        print(f"  Holders: {len(data['holders'])} records")
        print(f"  Trades: {len(data['trades'])} records")
        
        if len(data['candles']) > 0:
            print(f"\nCandle columns: {list(data['candles'].columns)}")
            print(f"Sample candle data:")
            print(data['candles'].head(2))
        
        if len(data['holders']) > 0:
            print(f"\nHolder columns: {list(data['holders'].columns)}")
            print(f"Sample holder data:")
            print(data['holders'].head(2))

