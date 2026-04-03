"""
Market Context Fetcher
Fetches global crypto market data from CryptoRank API for cross-cutting enrichment
of ML features, safety overrides, summaries, and post-trade reviews.
"""

import logging
import requests
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import api_config

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Global crypto market snapshot from CryptoRank"""
    fear_greed: int = 50
    fear_greed_change: int = 0
    altcoin_index: int = 50
    altcoin_index_change: int = 0
    btc_dominance: float = 50.0
    btc_dominance_change: float = 0.0
    eth_dominance: float = 10.0
    eth_dominance_change: float = 0.0
    total_market_cap: float = 0.0
    total_market_cap_change: float = 0.0
    total_volume_24h: float = 0.0
    total_volume_24h_change: float = 0.0
    investment_activity: int = 0
    active_currencies: int = 0
    fetched_at: str = ""
    is_default: bool = True

    @property
    def market_regime(self) -> str:
        if self.fear_greed >= 65 and self.fear_greed_change >= 0:
            return "BULL"
        elif self.fear_greed <= 25 and self.fear_greed_change <= 0:
            return "BEAR"
        return "SIDEWAYS"

    @property
    def market_favorability(self) -> float:
        return self.fear_greed / 100.0

    @property
    def altcoin_momentum(self) -> float:
        return self.altcoin_index_change / 100.0

    @property
    def is_extreme_fear(self) -> bool:
        return self.fear_greed <= api_config.MARKET_EXTREME_FEAR_THRESHOLD

    @property
    def is_extreme_greed(self) -> bool:
        return self.fear_greed >= api_config.MARKET_EXTREME_GREED_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fear_greed": self.fear_greed,
            "fear_greed_change": self.fear_greed_change,
            "altcoin_index": self.altcoin_index,
            "altcoin_index_change": self.altcoin_index_change,
            "btc_dominance": self.btc_dominance,
            "btc_dominance_change": self.btc_dominance_change,
            "eth_dominance": self.eth_dominance,
            "eth_dominance_change": self.eth_dominance_change,
            "total_market_cap": self.total_market_cap,
            "total_market_cap_change": self.total_market_cap_change,
            "total_volume_24h": self.total_volume_24h,
            "total_volume_24h_change": self.total_volume_24h_change,
            "investment_activity": self.investment_activity,
            "active_currencies": self.active_currencies,
            "market_regime": self.market_regime,
            "market_favorability": round(self.market_favorability, 4),
            "is_extreme_fear": self.is_extreme_fear,
            "is_extreme_greed": self.is_extreme_greed,
            "is_default": self.is_default,
            "fetched_at": self.fetched_at,
        }


def _parse_response(data: Dict[str, Any]) -> MarketContext:
    """Parse CryptoRank API response into MarketContext"""
    return MarketContext(
        fear_greed=int(data.get("fearGreed", 50)),
        fear_greed_change=int(data.get("fearGreedChange", 0)),
        altcoin_index=int(data.get("altcoinIndex", 50)),
        altcoin_index_change=int(data.get("altcoinIndexChange", 0)),
        btc_dominance=float(data.get("btcDominance", 50.0)),
        btc_dominance_change=float(data.get("btcDominanceChange", 0.0)),
        eth_dominance=float(data.get("ethDominance", 10.0)),
        eth_dominance_change=float(data.get("ethDominanceChange", 0.0)),
        total_market_cap=float(data.get("totalMarketCap", 0)),
        total_market_cap_change=float(data.get("totalMarketCapChange", 0.0)),
        total_volume_24h=float(data.get("totalVolume24h", 0)),
        total_volume_24h_change=float(data.get("totalVolume24hChange", 0.0)),
        investment_activity=int(data.get("investmentActivity", 0)),
        active_currencies=int(data.get("activeCurrencies", 0)),
        fetched_at=data.get("createdAt", datetime.utcnow().isoformat()),
        is_default=False,
    )


class MarketContextFetcher:
    """
    Fetches global crypto market context from CryptoRank API.

    Uses in-memory caching based on the nextUpdateAt field returned by the API
    (data updates every ~30 minutes). Falls back to neutral defaults on failure
    so the prediction pipeline is never blocked.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or api_config.INTERNAL_BASE_URL).rstrip("/")
        self.timeout = api_config.CRYPTORANK_TIMEOUT
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        self._cache: Optional[MarketContext] = None
        self._cache_expires_at: Optional[datetime] = None
        self._lock = threading.Lock()

        logger.info(
            f"MarketContextFetcher initialized (base_url={self.base_url}, "
            f"timeout={self.timeout}s)"
        )

    @staticmethod
    def get_empty_context() -> MarketContext:
        """Return neutral defaults when fetch fails"""
        return MarketContext(fetched_at=datetime.utcnow().isoformat())

    def fetch_latest(self) -> MarketContext:
        """
        Fetch latest global market context. Returns cached result if still fresh.
        Falls back to neutral defaults on any error.
        """
        with self._lock:
            if self._cache is not None and self._cache_expires_at is not None:
                if datetime.utcnow() < self._cache_expires_at:
                    return self._cache

        try:
            url = f"{self.base_url}{api_config.CRYPTORANK_ENDPOINT_LATEST}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            context = _parse_response(data)

            with self._lock:
                self._cache = context
                next_update = data.get("nextUpdateAt")
                if next_update:
                    try:
                        self._cache_expires_at = datetime.fromisoformat(
                            next_update.replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                    except (ValueError, TypeError):
                        self._cache_expires_at = datetime.utcnow()
                else:
                    self._cache_expires_at = datetime.utcnow()

            logger.info(
                f"Market context fetched: F&G={context.fear_greed}, "
                f"BTC dom={context.btc_dominance:.1f}%, "
                f"regime={context.market_regime}"
            )
            return context

        except requests.Timeout:
            logger.warning(
                f"CryptoRank API timeout ({self.timeout}s) — using defaults"
            )
        except requests.RequestException as e:
            logger.warning(f"CryptoRank API error: {e} — using defaults")
        except Exception as e:
            logger.warning(
                f"Unexpected error fetching market context: {e} — using defaults"
            )

        with self._lock:
            if self._cache is not None:
                logger.info("Returning stale cached market context")
                return self._cache

        return self.get_empty_context()

    def fetch_at(self, time: str) -> MarketContext:
        """
        Fetch historical market context at a specific point in time.
        Used by post-trade review to know market conditions at trade execution.

        Args:
            time: ISO timestamp (e.g. '2026-02-16T12:00:00Z')
        """
        try:
            url = f"{self.base_url}{api_config.CRYPTORANK_ENDPOINT_AT}"
            response = self.session.get(
                url, params={"time": time}, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            context = _parse_response(data)
            logger.info(
                f"Historical market context at {time}: "
                f"F&G={context.fear_greed}, regime={context.market_regime}"
            )
            return context

        except requests.Timeout:
            logger.warning(
                f"CryptoRank historical API timeout for {time} — using defaults"
            )
        except requests.RequestException as e:
            logger.warning(
                f"CryptoRank historical API error for {time}: {e} — using defaults"
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error fetching historical market context: {e}"
            )

        return self.get_empty_context()
