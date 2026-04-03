"""
Canonical meme-only / perps-only field lists and response filtering.

All endpoints use filter_response_by_token_type() to strip fields that
don't belong to the requested token_type before returning to the client.
"""

from typing import Any, Dict, Optional, Set

# --- Fields that only appear in MEME responses (stripped when token_type == "perps") ---

MEME_ONLY_TOP_LEVEL: Set[str] = {
    # Behavioral flags (user profile)
    "is_sniper",
    "is_holder",
    "is_flipper",
    # Dominant-whale tracking (predict)
    "dominant_whale_status",
    "dominant_whale_inactive_holding_pct",
    "dominant_whale_aging_holding_pct",
    "top_holder_last_activity_hours",
    # Holder / concentration metrics (predict)
    "gini_coefficient",
    "top10_hold_percent",
    "dev_hold_percent",
    "sniper_hold_percent",
    "phase",
}

MEME_ONLY_NESTED: Set[str] = {
    "is_sniper",
    "is_holder",
    "is_flipper",
}

# --- Fields that only appear in PERPS responses (stripped when token_type == "meme") ---

PERPS_ONLY_TOP_LEVEL: Set[str] = {
    "perps_profile",
    "open_positions",
    "perps_metrics",
    "user_perps_profile",
    "whale_flow",
    "position_risk",
}

PERPS_ONLY_NESTED: Set[str] = {
    "max_leverage",
    "min_liquidation_distance_pct",
    "open_positions_count",
    "has_position_in_coin",
}


PERPS_REDUNDANT_TOP_LEVEL: Set[str] = {
    "total_trades",
    "total_mints_traded",
    "overall_win_rate",
    "avg_pnl_per_trade",
    "total_realized_pnl",
    "lifetime_pnl",
    "total_volume_usd",
    "max_drawdown_pct",
    "avg_r_multiple",
    "avg_holding_time_minutes",
    "avg_hold_time_hours",
    "typical_entry_phase",
    "trader_type",
}


def strip_perps_redundant_top_level(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove top-level keys that duplicate ``perps_profile`` content."""
    return {k: v for k, v in data.items() if k not in PERPS_REDUNDANT_TOP_LEVEL}


def filter_response_by_token_type(
    data: Dict[str, Any],
    token_type: str,
    *,
    recurse_keys: Optional[set] = None,
) -> Dict[str, Any]:
    """Remove keys that don't belong to *token_type*.

    Args:
        data: The response dict to filter (not mutated; a shallow copy is returned).
        token_type: ``"meme"`` or ``"perps"``.
        recurse_keys: Optional set of top-level keys whose *dict* values
            should also be filtered (e.g. ``{"profile_summary", "user_risk_assessment"}``).

    Returns:
        A new dict with irrelevant keys removed.
    """
    if recurse_keys is None:
        recurse_keys = {"profile_summary", "user_risk_assessment"}

    is_perps = token_type.lower() == "perps"

    if is_perps:
        strip_top = MEME_ONLY_TOP_LEVEL
        strip_nested = MEME_ONLY_NESTED
    else:
        strip_top = PERPS_ONLY_TOP_LEVEL
        strip_nested = PERPS_ONLY_NESTED

    filtered = {k: v for k, v in data.items() if k not in strip_top}

    for rk in recurse_keys:
        val = filtered.get(rk)
        if isinstance(val, dict):
            filtered[rk] = {k: v for k, v in val.items() if k not in strip_nested}

    return filtered
