"""
Circuit Breaker pattern for external service dependencies.

Prevents cascading failures by short-circuiting requests to unhealthy services
and reporting state transitions to Sentry.
"""

import time
import logging
import threading
from typing import Dict, Optional

import sentry_sdk

logger = logging.getLogger(__name__)


def sentry_fallback_warning(
    service: str,
    reason: str,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """Emit a Sentry warning when a fallback path is activated.

    Consistent tagging (``fallback.service``) makes it easy to set up
    Sentry alerts that fire whenever *any* fallback fires.
    """
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("fallback.service", service)
        scope.set_extra("reason", reason)
        if extra:
            for k, v in extra.items():
                scope.set_extra(k, v)
        sentry_sdk.capture_message(
            f"Fallback activated: {service} — {reason}",
            level="warning",
        )


class CircuitBreakerOpen(Exception):
    """Raised when a circuit breaker is open and the service is unavailable."""

    def __init__(self, service_name: str, remaining_seconds: float = 0):
        self.service_name = service_name
        self.remaining_seconds = remaining_seconds
        super().__init__(f"{service_name} is temporarily unavailable")


class CircuitBreaker:
    """
    Thread-safe circuit breaker with Sentry reporting.

    After `failure_threshold` consecutive failures the breaker trips (OPEN).
    While OPEN every call to `check()` raises `CircuitBreakerOpen` until the
    `cooldown_period` elapses.  After cooldown the next request is allowed
    through; a success resets the breaker, a failure re-trips it immediately.
    """

    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        cooldown_period: float = 20.0,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.cooldown_period = cooldown_period

        self._lock = threading.Lock()
        self._state = self.STATE_CLOSED
        self._consecutive_failures = 0
        self._tripped_at: Optional[float] = None
        self._recovering = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> None:
        """Gate-check before making a request.

        Raises ``CircuitBreakerOpen`` when the breaker is OPEN and the
        cooldown has not yet elapsed.  After cooldown expires the breaker
        transitions back to CLOSED and sets ``_recovering`` so that the
        next ``record_success`` can log recovery to Sentry.
        """
        with self._lock:
            if self._state == self.STATE_CLOSED:
                return

            elapsed = time.monotonic() - self._tripped_at
            if elapsed < self.cooldown_period:
                remaining = self.cooldown_period - elapsed
                logger.warning(
                    f"Circuit breaker OPEN for {self.service_name}. "
                    f"Retry in {remaining:.1f}s"
                )
                raise CircuitBreakerOpen(self.service_name, remaining)

            self._state = self.STATE_CLOSED
            self._consecutive_failures = 0
            self._recovering = True
            logger.info(
                f"Circuit breaker cooldown expired for {self.service_name} — "
                f"allowing next request through"
            )

    def record_success(self) -> None:
        """Record a successful call and reset failure tracking."""
        with self._lock:
            was_recovering = self._recovering
            self._consecutive_failures = 0
            self._recovering = False

        if was_recovering:
            self._sentry_recover()
            logger.info(f"Circuit breaker RECOVERED for {self.service_name}")

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call.  Trips the breaker when the threshold is reached."""
        with self._lock:
            self._consecutive_failures += 1
            self._recovering = False
            count = self._consecutive_failures

            if count >= self.failure_threshold and self._state != self.STATE_OPEN:
                self._state = self.STATE_OPEN
                self._tripped_at = time.monotonic()
                logger.error(
                    f"Circuit breaker TRIPPED for {self.service_name} after "
                    f"{count} consecutive failures. Blocking requests for "
                    f"{self.cooldown_period}s."
                )
                self._sentry_trip(count, error)

    def is_open(self) -> bool:
        with self._lock:
            return self._state == self.STATE_OPEN

    # ------------------------------------------------------------------
    # Sentry helpers
    # ------------------------------------------------------------------

    def _sentry_trip(self, failure_count: int, error: Optional[Exception] = None) -> None:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("circuit_breaker.service", self.service_name)
            scope.set_extra("service_name", self.service_name)
            scope.set_extra("failure_count", failure_count)
            scope.set_extra("failure_threshold", self.failure_threshold)
            scope.set_extra("cooldown_period_seconds", self.cooldown_period)
            if error:
                scope.set_extra("last_error", str(error))
            sentry_sdk.capture_message(
                f"Circuit breaker TRIPPED for {self.service_name} "
                f"after {failure_count} consecutive failures",
                level="error",
            )

    def _sentry_recover(self) -> None:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("circuit_breaker.service", self.service_name)
            scope.set_extra("service_name", self.service_name)
            scope.set_extra("failure_threshold", self.failure_threshold)
            scope.set_extra("cooldown_period_seconds", self.cooldown_period)
            sentry_sdk.capture_message(
                f"Circuit breaker RECOVERED for {self.service_name}",
                level="info",
            )
