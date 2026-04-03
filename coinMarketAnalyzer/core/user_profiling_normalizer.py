"""
Normalizer for the upstream GET /user-profiling (type=spot) JSON response.

Handles both legacy flat responses (five-field) and the newer shape that
includes win rates, renamed drawdown, execution counts, fees, leverage, etc.

All callers should funnel raw JSON through ``normalize_user_profiling_summary``
so the rest of the codebase works with a single, stable dict shape.
"""

from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to float, returning *default* on failure or None."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _normalize_win_rate(value: Any) -> float:
    """Return a win rate on the **0–100** scale.

    The upstream API may return 0–1 (fraction) or 0–100 (percentage).
    Heuristic: if the value is <= 1.0 and > 0, treat it as a fraction and
    multiply by 100.  Values already > 1.0 are assumed to be percentages.
    """
    rate = _safe_float(value, 0.0)
    if 0.0 < rate <= 1.0:
        rate *= 100.0
    return round(rate, 4)


def normalize_user_profiling_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw ``/user-profiling`` JSON dict into a stable internal shape.

    The returned dict always contains the **five legacy keys** that the rest of
    the codebase relies on, plus optional enriched metrics.  Fill-only fields
    (nested ``summary.raw_fill_count`` etc.) are excluded unless the caller
    explicitly reads ``execution_summary``.

    Backward-compatible: if the upstream still sends the old flat keys the
    result is identical to what ``fetch_user_profiling_summary`` used to return.

    ``max_drawdown_pct`` mapping:
        The new API returns ``max_drawdown_pct_realized`` as a **positive**
        magnitude (e.g. 1.17 means ~1.17 % peak-to-trough loss).  Downstream
        risk code compares ``max_drawdown_pct <= -30`` (negative = bad).  We
        therefore **negate** ``max_drawdown_pct_realized`` so that a 1.17 %
        drawdown becomes -1.17 in the internal model.  If the legacy key
        ``max_drawdown_pct`` is present and already negative we keep it as-is.
    """

    if not isinstance(raw, dict):
        logger.warning("normalize_user_profiling_summary received non-dict input; returning defaults")
        return _defaults()

    # --- Legacy five fields (always present in result) ---

    lifetime_pnl = _safe_float(raw.get('lifetime_pnl'))
    total_volume_usd = _safe_float(raw.get('total_volume_usd'))
    avg_holding_time_minutes = _safe_float(raw.get('avg_holding_time_minutes'))
    avg_r_multiple = _safe_float(raw.get('avg_r_multiple'))

    # Drawdown: prefer legacy key; fall back to renamed key with sign flip.
    raw_dd = raw.get('max_drawdown_pct')
    raw_dd_realized = raw.get('max_drawdown_pct_realized')
    if raw_dd is not None:
        max_drawdown_pct = _safe_float(raw_dd)
    elif raw_dd_realized is not None:
        magnitude = _safe_float(raw_dd_realized)
        max_drawdown_pct = -abs(magnitude)
    else:
        max_drawdown_pct = 0.0

    result: Dict[str, Any] = {
        'lifetime_pnl': lifetime_pnl,
        'total_volume_usd': total_volume_usd,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_r_multiple': avg_r_multiple,
        'avg_holding_time_minutes': avg_holding_time_minutes,
    }

    # --- Enriched metrics (present only when upstream provides them) ---

    if 'win_rate_round_trip_based' in raw:
        result['win_rate'] = _normalize_win_rate(raw['win_rate_round_trip_based'])

    if 'win_rate_execution_based' in raw:
        result['win_rate_execution_based'] = _normalize_win_rate(raw['win_rate_execution_based'])

    if 'total_closed_pnl' in raw:
        result['total_closed_pnl'] = _safe_float(raw['total_closed_pnl'])

    if 'total_fees' in raw:
        result['total_fees'] = _safe_float(raw['total_fees'])

    if 'closed_executions' in raw:
        result['closed_executions'] = int(_safe_float(raw['closed_executions']))

    if 'completed_round_trips' in raw:
        result['completed_round_trips'] = int(_safe_float(raw['completed_round_trips']))

    if 'trades_long' in raw:
        result['trades_long'] = int(_safe_float(raw['trades_long']))

    if 'trades_short' in raw:
        result['trades_short'] = int(_safe_float(raw['trades_short']))

    if 'avg_leverage' in raw:
        result['avg_leverage'] = _safe_float(raw['avg_leverage'])

    if 'botDetected' in raw:
        result['bot_detected'] = bool(raw['botDetected'])

    if 'responseTime' in raw:
        result['response_time_ms'] = int(_safe_float(raw['responseTime']))

    return result


def _defaults() -> Dict[str, Any]:
    return {
        'lifetime_pnl': 0.0,
        'total_volume_usd': 0.0,
        'max_drawdown_pct': 0.0,
        'avg_r_multiple': 0.0,
        'avg_holding_time_minutes': 0.0,
    }
