"""
User API Routes
Endpoints for user profiling and risk assessment
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

import sentry_sdk
from fastapi import APIRouter, HTTPException, Query, Path as FastPath
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.response_schema import filter_response_by_token_type, strip_perps_redundant_top_level
from core.user_profiling_normalizer import normalize_user_profiling_summary

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

router = APIRouter(tags=["Users"])


def _behavioral_tendencies_for_response(
    total_trades: int,
    fomo: Optional[float],
    panic: Optional[float],
    dip: Optional[float],
    whale: Optional[float],
) -> tuple:
    """When total_trades >= MIN and all four tendencies are 0, return (None, None, None, None, 'not_computed', message); else return (f, p, d, w, None, None)."""
    from config import user_profile_config
    min_trades = user_profile_config.MIN_TRADES_FOR_BIAS_INFERENCE
    if total_trades >= min_trades and (fomo or 0) == 0 and (panic or 0) == 0 and (dip or 0) == 0 and (whale or 0) == 0:
        return None, None, None, None, "not_computed", "Not enough data to infer biases"
    return (
        fomo if fomo is not None else 0.0,
        panic if panic is not None else 0.0,
        dip if dip is not None else 0.0,
        whale if whale is not None else 0.0,
        None,
        None,
    )


def _normalize_to_date_for_internal_api(to_date: str) -> str:
    """
    Normalize to_date for the internal user-profiling API.
    The internal API returns 400 if to_date uses 23:59:59; use start of next day 00:00:00Z.
    """
    if not to_date or "23:59:59" not in to_date:
        return to_date
    try:
        date_part = to_date.split("T")[0]
        dt = datetime.strptime(date_part, "%Y-%m-%d")
        next_dt = dt + timedelta(days=1)
        return next_dt.strftime("%Y-%m-%dT00:00:00Z")
    except (ValueError, IndexError):
        return to_date


def _compute_fallback_entry_phase(trades: List[Dict]) -> str:
    """Compute typical_entry_phase from per-trade (buy) phase counts in the fallback path.
    
    Uses each buy trade's executed_at and creation_timestamp to derive token age -> phase.
    """
    import pandas as pd
    phase_counts: Dict[str, int] = {}
    for trade in trades:
        is_buy = (
            trade.get('is_buy') or trade.get('isBuy') or
            trade.get('type', '').upper() == 'BUY' or
            trade.get('side', '').upper() == 'BUY'
        )
        if not is_buy:
            continue
        creation_ts = trade.get('creation_timestamp')
        if not creation_ts or creation_ts <= 0:
            continue
        executed_at_str = (
            trade.get('executed_at') or trade.get('timestamp') or
            trade.get('time') or trade.get('created_at')
        )
        if not executed_at_str:
            continue
        try:
            executed_at = pd.to_datetime(str(executed_at_str), utc=True)
            token_created_at = pd.to_datetime(int(creation_ts), unit='s', utc=True)
            token_age_days = (executed_at - token_created_at).total_seconds() / 86400.0
        except Exception:
            continue
        if token_age_days < 0:
            phase = "UNKNOWN"
        elif token_age_days <= 3:
            phase = "P1"
        elif token_age_days <= 14:
            phase = "P2"
        elif token_age_days <= 45:
            phase = "P3"
        else:
            phase = "P4"
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    return max(phase_counts, key=phase_counts.get) if phase_counts else "UNKNOWN"


# =============================================================================
# Request/Response Models
# =============================================================================

class UserProfileResponse(BaseModel):
    """Response model for user profile (matches HybridTradingSystem)"""
    wallet_address: str
    total_trades: int
    total_mints_traded: int
    overall_win_rate: float
    avg_pnl_per_trade: float
    total_realized_pnl: float
    # New API-provided fields
    lifetime_pnl: float = 0.0
    total_volume_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_r_multiple: float = 0.0
    avg_holding_time_minutes: float = 0.0
    # Behavioral metrics
    avg_hold_time_hours: float
    typical_entry_phase: str
    trader_type: List[str]
    is_sniper: bool
    is_holder: bool
    is_flipper: bool
    fomo_tendency: Optional[float] = None
    panic_sell_tendency: Optional[float] = None
    dip_buy_tendency: Optional[float] = None
    whale_follow_tendency: Optional[float] = None
    is_valid: bool
    patterns: List[Dict[str, Any]]
    best_pattern: Optional[Dict[str, Any]] = None
    worst_pattern: Optional[Dict[str, Any]] = None
    # AI-generated insights
    profile_description: Optional[str] = None
    risk_rating: Optional[Dict[str, str]] = None
    recommendations: Optional[List[str]] = None
    # Global market context snapshot
    current_market_context: Optional[Dict[str, Any]] = None
    # Behavioral biases status when tendencies are not computed (e.g. enough trades but all 0)
    behavioral_biases_status: Optional[str] = None
    behavioral_biases_message: Optional[str] = None
    # Observation window (actual from_date / to_date used for this profile)
    observation_window: Optional[Dict[str, str]] = None
    # Enriched spot metrics (from newer /user-profiling response; None when unavailable)
    win_rate_round_trip: Optional[float] = None
    win_rate_execution: Optional[float] = None
    total_closed_pnl: Optional[float] = None
    total_fees: Optional[float] = None
    completed_round_trips: Optional[int] = None
    trades_long: Optional[int] = None
    trades_short: Optional[int] = None
    avg_leverage: Optional[float] = None
    bot_detected: Optional[bool] = None
    # Perps-specific fields (None for meme wallets)
    token_type: Optional[str] = None
    perps_profile: Optional[Dict[str, Any]] = None
    open_positions: Optional[List[Dict[str, Any]]] = None


class UserRiskAssessmentRequest(BaseModel):
    """User risk assessment request (matches HybridTradingSystem)"""
    wallet_address: str = Field(..., description="User wallet address", min_length=32, max_length=64)
    token_address: str = Field(..., description="Token mint address", min_length=32, max_length=64)
    phase: str = Field("P4", description="Token phase (P1/P2/P3/P4)")
    top10_concentration: float = Field(50.0, ge=0, le=100, description="Top 10 holder concentration %")
    whale_state: str = Field("Stability", description="Current whale state")
    token_type: str = Field(default="meme", description="Token type: 'meme' or 'perps'")
    coin: Optional[str] = Field(default=None, description="Coin for perps context (e.g., 'BTC')")

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, values):
        if isinstance(values, dict):
            if "token_type" in values and isinstance(values["token_type"], str):
                values["token_type"] = values["token_type"].lower().strip()
        return values


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response (matches HybridTradingSystem)"""
    rating: str  # GREEN, YELLOW, RED
    confidence: float
    signal: str  # BUY, SELL, HOLD
    signal_weight: float
    message: str
    risk_factors: List[str]
    matching_pattern: Optional[Dict[str, Any]] = None
    profile_summary: Dict[str, Any]


# =============================================================================
# Components (lazy loaded)
# =============================================================================

_components: Optional[Dict[str, Any]] = None


def get_components():
    """Initialize and return components"""
    global _components
    
    if _components is not None:
        return _components
    
    # Return empty components - user endpoints work without dependencies
    _components = {}
    
    return _components


# =============================================================================
# User Endpoints
# =============================================================================

@router.get("/users", summary="List Users")
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Users per page"),
    token_type: str = Query("meme", description="Token type: 'meme' or 'perps'")
):
    """
    List users with trading activity
    
    Returns paginated list of wallet addresses with basic stats.
    
    **Query Parameters:**
    - `token_type`: 'meme' for spot/meme traders, 'perps' for perpetual futures traders
    """
    token_type = token_type.lower().strip()
    try:
        if token_type == "perps":
            from core.data_fetcher import PerpsDataFetcher
            
            perps_fetcher = PerpsDataFetcher()
            all_users = await asyncio.to_thread(perps_fetcher.fetch_hyper_liquid_users)
            
            start = (page - 1) * limit
            end = start + limit
            paginated = all_users[start:end] if all_users else []
            
            return {
                "page": page,
                "limit": limit,
                "token_type": "perps",
                "count": len(paginated),
                "total": len(all_users) if all_users else 0,
                "users": paginated
            }
        else:
            from core.data_fetcher import DataFetcher
            
            data_fetcher = DataFetcher(limiter_name="users")
            users = await asyncio.to_thread(data_fetcher.fetch_users_list, page=page, limit=limit)
            
            return {
                "page": page,
                "limit": limit,
                "token_type": "meme",
                "count": len(users),
                "users": users
            }
        
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list users: {e}")


@router.get("/user/profile/{wallet_address}", summary="Get User Profile")
async def get_user_profile(
    wallet_address: str = FastPath(..., description="User wallet address", min_length=32, max_length=64),
    from_date: Optional[str] = Query(None, description="Start date (ISO format: 2025-01-01T00:00:00Z)"),
    to_date: Optional[str] = Query(None, description="End date (ISO format: 2025-01-05T00:00:00Z)"),
    token_type: str = Query("meme", description="Token type: 'meme' or 'perps'")
):
    """
    Get comprehensive trading profile for a user.
    
    Response shape depends on ``token_type``: meme-only fields (e.g.
    ``is_sniper``, ``is_holder``, ``is_flipper``) are omitted for perps;
    perps-only fields (e.g. ``perps_profile``, ``open_positions``) are
    omitted for meme.
    
    **Query Parameters:**
    - `token_type`: 'meme' for spot/meme traders, 'perps' for perpetual futures traders
    - `from_date`: Start date for analysis (default: today − 80 days)
    - `to_date`: End date for analysis (default: today)
    
    **Response includes:**
    - Trading stats: total_trades, win_rate, PnL, volume
    - Behavioral analysis: trader_type, patterns, tendencies
    - For perps: direction bias, open positions, leverage info
    """
    token_type = token_type.lower().strip()
    try:
        from config import get_config
        from core.market_context_fetcher import MarketContextFetcher
        
        config = get_config()
        
        # Fetch current market context (shared by both paths)
        current_market_context = None
        try:
            market_ctx_fetcher = MarketContextFetcher()
            current_market_context = await asyncio.to_thread(market_ctx_fetcher.fetch_latest)
        except Exception as e:
            logger.warning(f"Failed to fetch market context for user profile: {e}")
        
        market_ctx_dict = (
            current_market_context.to_dict()
            if current_market_context and hasattr(current_market_context, 'to_dict')
            else None
        )
        
        # Default window when dates omitted: last 80 days ending today
        if from_date is None or to_date is None:
            api_cfg = config.api
            to_dt = datetime.utcnow()
            default_days = getattr(api_cfg, 'USER_PROFILING_DEFAULT_DAYS', 80)
            from_dt = to_dt - timedelta(days=default_days)
            from_date = from_date or from_dt.strftime('%Y-%m-%dT00:00:00Z')
            to_date = to_date or to_dt.strftime('%Y-%m-%dT23:59:59Z')
        
        observation_window = {"from_date": from_date, "to_date": to_date}

        if token_type == "perps":
            profile = await _build_perps_profile(
                wallet_address, from_date, to_date,
                current_market_context, market_ctx_dict,
                observation_window=observation_window,
            )
        else:
            profile = await _build_meme_profile(
                wallet_address, from_date, to_date,
                current_market_context, market_ctx_dict, config,
                observation_window=observation_window,
            )
        
        profile_dict = profile.model_dump(exclude_none=True)
        filtered = filter_response_by_token_type(profile_dict, token_type, recurse_keys=set())
        if token_type == "perps":
            filtered = strip_perps_redundant_top_level(filtered)
            if "current_market_context" in filtered:
                ctx = filtered.pop("current_market_context")
                filtered["current_market_context"] = ctx
        return filtered
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("user_profile_request", {
                "wallet_address": wallet_address,
                "from_date": from_date,
                "to_date": to_date,
                "token_type": token_type,
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {e}")


def _market_context_recommendations(current_market_context) -> List[str]:
    """Generate recommendations from market context (shared by meme and perps paths)."""
    recs = []
    if current_market_context and not getattr(current_market_context, 'is_default', True):
        if current_market_context.is_extreme_fear:
            recs.append(
                f"Market is in extreme fear (F&G: {current_market_context.fear_greed}) — "
                f"consider reducing position sizes and being extra selective."
            )
        elif current_market_context.is_extreme_greed:
            recs.append(
                f"Market is in extreme greed (F&G: {current_market_context.fear_greed}) — "
                f"be cautious of FOMO entries and consider taking profits."
            )
    return recs


def _derive_trader_types(is_sniper: bool, is_flipper: bool, is_holder: bool) -> List[str]:
    """Build a list of all applicable trader types from boolean flags."""
    types: List[str] = []
    if is_sniper:
        types.append("SNIPER")
    if is_flipper:
        types.append("FLIPPER")
    if is_holder:
        types.append("HOLDER")
    return types or ["MIXED"]


_TENDENCY_THRESHOLD = 0.10
_MAX_RECOMMENDATIONS = 15


def _build_meme_recommendations(
    *,
    total_trades: int,
    overall_win_rate: float,
    avg_pnl_per_trade: float,
    total_realized_pnl: float,
    trader_type: str,
    patterns: List[Dict[str, Any]],
    best_pattern: Optional[Dict[str, Any]],
    worst_pattern: Optional[Dict[str, Any]],
    fomo_tendency: Optional[float],
    panic_sell_tendency: Optional[float],
    dip_buy_tendency: Optional[float],
    whale_follow_tendency: Optional[float],
    is_sniper: bool,
    is_flipper: bool,
    is_holder: bool,
    current_market_context,
    observation_window: Optional[Dict[str, str]] = None,
    max_drawdown_pct: float = 0.0,
) -> List[str]:
    """Build structured meme recommendations using full profile + market context."""
    from engines.user_profiler import get_dominant_patterns, PATTERN_DESCRIPTIONS
    recs: List[str] = []

    # --- 1. Worst pattern (always surface first when present) ---
    if worst_pattern and worst_pattern.get('pattern_type', 'UNKNOWN') != 'UNKNOWN':
        wp = worst_pattern
        wp_desc = PATTERN_DESCRIPTIONS.get(wp['pattern_type'], '')
        if wp.get('win_rate', 0) == 0:
            recs.append(
                f"Critical: your worst pattern {wp['pattern_type']} has a 0% win rate "
                f"across {wp.get('occurrences', '?')} occurrences — {wp_desc} "
                f"Strongly consider eliminating this behavior entirely."
            )
        else:
            recs.append(
                f"Your worst-performing pattern is {wp['pattern_type']} "
                f"({wp.get('occurrences', '?')} occurrences, "
                f"{wp.get('win_rate', 0):.1f}% win rate). "
                f"{wp_desc} Consider reducing exposure to situations that trigger this."
            )

    # --- 1b. Surface any other 0%-win-rate patterns not already shown as worst ---
    worst_pt = worst_pattern.get('pattern_type') if worst_pattern else None
    for p in patterns:
        pt = p.get('pattern_type', 'UNKNOWN')
        if pt == 'UNKNOWN' or pt == worst_pt:
            continue
        if p.get('win_rate', 100) == 0 and p.get('occurrences', 0) >= 3:
            desc = PATTERN_DESCRIPTIONS.get(pt, '')
            recs.append(
                f"Warning: {pt} has 0% win rate across {p['occurrences']} occurrences. "
                f"{desc} Review and avoid this pattern."
            )

    # --- 2. Tendency scores (non-zero) ---
    fomo_val = fomo_tendency or 0.0
    panic_val = panic_sell_tendency or 0.0
    dip_val = dip_buy_tendency or 0.0
    whale_val = whale_follow_tendency or 0.0

    if panic_val >= _TENDENCY_THRESHOLD:
        recs.append(
            f"You show a tendency to panic sell ({panic_val*100:.0f}% of trades). "
            f"Consider setting pre-defined exit points before entering."
        )
    if fomo_val >= _TENDENCY_THRESHOLD:
        recs.append(
            f"You show FOMO behavior ({fomo_val*100:.0f}% of buys). "
            f"Wait for pullbacks instead of chasing pumps."
        )
    if dip_val >= _TENDENCY_THRESHOLD:
        recs.append(
            f"You frequently buy dips ({dip_val*100:.0f}% of buys). "
            f"Ensure dips are supported by volume before averaging in."
        )
    if whale_val >= _TENDENCY_THRESHOLD:
        recs.append(
            f"You tend to follow whale accumulation ({whale_val*100:.0f}% of buys). "
            f"Cross-check whale moves with on-chain data to avoid traps."
        )

    # --- 2b. Positive signals for low/zero tendencies in adverse conditions ---
    _has_market = (
        current_market_context
        and not getattr(current_market_context, 'is_default', True)
    )
    if _has_market:
        _fg = getattr(current_market_context, 'fear_greed', 50)
        if fomo_val < _TENDENCY_THRESHOLD and _fg <= 40:
            recs.append(
                "You don't chase pumps — maintain this discipline in the current "
                f"fearful market (F&G: {_fg})."
            )
        if panic_val < _TENDENCY_THRESHOLD and _fg <= 30:
            recs.append(
                f"Despite a fearful market (F&G: {_fg}), you show no panic selling tendency. "
                f"This composure is a strength — keep it up."
            )

    # --- 3. Fear/greed vs weaknesses ---
    if _has_market:
        fg = getattr(current_market_context, 'fear_greed', 50)
        if getattr(current_market_context, 'is_extreme_fear', False) and panic_val >= _TENDENCY_THRESHOLD:
            recs.append(
                f"Extreme fear (F&G: {fg}) can amplify your panic selling tendency. "
                f"Consider smaller positions or waiting for stabilization."
            )
        elif fg <= 30 and panic_val >= _TENDENCY_THRESHOLD:
            recs.append(
                f"Fearful market (F&G: {fg}) combined with your panic selling tendency "
                f"increases risk of reactive exits. Set stops before entering."
            )
        if getattr(current_market_context, 'is_extreme_greed', False) and fomo_val >= _TENDENCY_THRESHOLD:
            recs.append(
                f"Extreme greed (F&G: {fg}) increases your FOMO risk. "
                f"Stick to your rules and avoid chasing."
            )
        elif fg >= 70 and fomo_val >= _TENDENCY_THRESHOLD:
            recs.append(
                f"Greedy market (F&G: {fg}) combined with your FOMO tendency "
                f"increases risk of chasing entries. Wait for pullbacks."
            )

    # --- 4. Market favorability vs trader style ---
    # Use boolean flags instead of trader_type string so dual-classified
    # wallets (e.g. is_sniper=True AND is_flipper=True) get both sets of rules.
    if _has_market:
        regime = getattr(current_market_context, 'market_regime', 'SIDEWAYS')
        fg = getattr(current_market_context, 'fear_greed', 50)
        if is_sniper and regime == "SIDEWAYS":
            recs.append(
                f"Sideways market (F&G: {fg}) — fewer clean P1 entries for snipers. "
                f"Be highly selective and avoid chasing."
            )
        if is_holder and regime == "BULL":
            recs.append(
                "Bull regime aligns well with a holding style. "
                "Consider trailing stops to lock in gains."
            )
        if is_flipper and regime == "BEAR":
            recs.append(
                f"Choppy bear market (F&G: {fg}) can whipsaw short holds. "
                f"Consider smaller position sizes or fewer flips."
            )
        if is_flipper and regime == "SIDEWAYS":
            recs.append(
                f"Sideways market (F&G: {fg}) creates choppy conditions for flippers. "
                f"Be selective, size down, and avoid forcing trades."
            )
        if is_sniper and regime == "BEAR":
            recs.append(
                f"Bear market (F&G: {fg}) — early entries carry higher risk. "
                f"Reduce size and wait for clearer setups."
            )
        if is_holder and regime == "BEAR":
            recs.append(
                f"Bear market (F&G: {fg}) — holding through extended drawdowns "
                f"can erode capital. Consider tighter stop-losses or reduced exposure."
            )

    # --- 5. Dual classifications ---
    if is_sniper and is_flipper:
        recs.append(
            "You show both sniper and flipper behavior — "
            "mixing very fast entries with short holds can amplify volatility risk."
        )
    elif is_sniper and is_holder:
        recs.append(
            "You combine sniper entries with long holds — "
            "strong when entries are precise, but consider exits if early conviction fades."
        )

    # --- 6. PnL / avg_pnl_per_trade ---
    if total_realized_pnl > 0:
        recs.append(
            f"Account is net profitable (${total_realized_pnl:,.2f}) over the observation window."
        )
    elif total_realized_pnl < 0:
        recs.append(
            f"Account is net negative (${total_realized_pnl:,.2f}). "
            f"Review strategy and tighten risk management."
        )

    if avg_pnl_per_trade < 0:
        recs.append(
            f"Average PnL per trade is negative (${avg_pnl_per_trade:,.4f}). "
            f"Focus on cutting losers early and improving position sizing."
        )

    # High PnL + low win rate → few big wins carrying everything
    if total_realized_pnl > 0 and overall_win_rate < 50:
        recs.append(
            "Positive PnL despite a sub-50% win rate — a few big wins are carrying results. "
            "Tighten stops on losers to improve consistency."
        )

    # --- 6b. Max drawdown ---
    if max_drawdown_pct <= -50:
        recs.append(
            f"Maximum drawdown of {max_drawdown_pct:.1f}% is severe. "
            f"Implement strict stop-losses and position sizing rules to protect capital."
        )
    elif max_drawdown_pct <= -30:
        recs.append(
            f"Maximum drawdown of {max_drawdown_pct:.1f}% is significant. "
            f"Review risk management and consider tighter stop-losses."
        )

    # --- 7. Win rate ---
    if overall_win_rate < 30:
        recs.append(
            f"Win rate of {overall_win_rate:.1f}% is critically low. "
            f"Re-evaluate entry criteria and consider paper trading to rebuild edge."
        )
    elif overall_win_rate < 50:
        recs.append(
            f"Win rate of {overall_win_rate:.1f}% is below average. "
            f"Focus on quality over quantity — wait for higher-probability setups."
        )
    elif overall_win_rate > 60:
        recs.append(
            f"Strong win rate of {overall_win_rate:.1f}%. "
            f"Consider increasing position sizes on high-confidence setups."
        )

    # --- 8. Dominant patterns in plain English ---
    dominant = get_dominant_patterns(patterns, top_n=3)
    for dp in dominant:
        recs.append(
            f"Dominant pattern: {dp['pattern_type']} ({dp['occurrences']} trades, "
            f"{dp['win_rate']:.1f}% win rate) — {dp['description']}"
        )

    # --- 8b. Win rate discrepancy note ---
    max_pattern_wr = max((p.get('win_rate', 0) for p in patterns), default=0)
    if patterns and max_pattern_wr > 0 and abs(max_pattern_wr - overall_win_rate) > 20:
        recs.append(
            f"Note: overall win rate ({overall_win_rate:.1f}%) is calculated per trade, "
            f"while pattern win rates are per token position — "
            f"a single token can include multiple trades."
        )

    # --- 9. Observation window + total trades ---
    if observation_window:
        fd = observation_window.get('from_date', '?')[:10]
        td = observation_window.get('to_date', '?')[:10]
        recs.append(
            f"Based on {total_trades} trades from {fd} to {td}."
        )
    else:
        recs.append(f"Based on {total_trades} trades in the observation window.")

    # --- 10. Generic market context recs ---
    recs.extend(_market_context_recommendations(current_market_context))

    # Cap and return
    return recs[:_MAX_RECOMMENDATIONS]


async def _build_meme_profile(
    wallet_address: str,
    from_date: str,
    to_date: str,
    current_market_context,
    market_ctx_dict: Optional[Dict],
    config,
    observation_window: Optional[Dict[str, str]] = None,
) -> UserProfileResponse:
    """Build profile from meme/spot data. Uses UserProfiler for pattern detection when possible."""
    import requests
    from core.data_fetcher import DataFetcher
    from engines.user_profiler import UserProfiler, profile_to_dict

    # Try full profile with pattern detection via UserProfiler (same internal API, but runs pattern logic)
    try:
        data_fetcher = DataFetcher(
            base_url=config.api.INTERNAL_BASE_URL.rstrip('/'),
            limiter_name="users"
        )
        profiler = UserProfiler(data_fetcher=data_fetcher)
        # Ensure we build for the requested date range (skip cache so dates are respected)
        profiler.invalidate_cache(wallet_address)
        profile = await asyncio.to_thread(
            profiler.get_profile,
            wallet_address,
            use_birdeye_fallback=False,
            from_date=from_date,
            to_date=to_date
        )
        if profile is not None and profile.total_trades > 0:
            d = profile_to_dict(profile)
            # API contract: overall_win_rate and pattern win_rate as percentage (0-100); engine stores 0-1
            overall_wr = profile.overall_win_rate
            if overall_wr <= 1.0:
                overall_wr = overall_wr * 100.0
            patterns_for_response = []
            for p in d['patterns']:
                wr = p.get('win_rate', 0)
                if wr <= 1.0:
                    wr = wr * 100.0
                patterns_for_response.append({
                    **p,
                    'win_rate': round(wr, 2)
                })
            best = d.get('best_pattern')
            if best and best.get('win_rate', 0) <= 1.0:
                best = {**best, 'win_rate': round(best['win_rate'] * 100.0, 2)}
            worst = d.get('worst_pattern')
            if worst and worst.get('win_rate', 0) <= 1.0:
                worst = {**worst, 'win_rate': round(worst['win_rate'] * 100.0, 2)}
            recommendations = _build_meme_recommendations(
                total_trades=profile.total_trades,
                overall_win_rate=overall_wr,
                avg_pnl_per_trade=profile.avg_pnl_per_trade,
                total_realized_pnl=profile.total_realized_pnl,
                trader_type=profile.trader_type.value,
                patterns=patterns_for_response,
                best_pattern=best,
                worst_pattern=worst,
                fomo_tendency=d.get('fomo_tendency'),
                panic_sell_tendency=d.get('panic_sell_tendency'),
                dip_buy_tendency=d.get('dip_buy_tendency'),
                whale_follow_tendency=d.get('whale_follow_tendency'),
                is_sniper=profile.is_sniper,
                is_flipper=profile.is_flipper,
                is_holder=profile.is_holder,
                current_market_context=current_market_context,
                observation_window=observation_window,
                max_drawdown_pct=profile.max_drawdown_pct,
            )
            risk_rating = None
            if overall_wr > 0:
                if overall_wr < 30:
                    risk_rating = {"rating": "RED", "reason": f"win rate below 30% ({overall_wr:.1f}%)"}
                elif overall_wr < 50:
                    risk_rating = {"rating": "YELLOW", "reason": f"win rate below 50% ({overall_wr:.1f}%)"}
                else:
                    risk_rating = {"rating": "GREEN", "reason": f"healthy win rate ({overall_wr:.1f}%)"}
            if risk_rating and profile.max_drawdown_pct <= -50:
                if risk_rating["rating"] != "RED":
                    risk_rating["rating"] = "RED"
                risk_rating["reason"] += f"; severe drawdown ({profile.max_drawdown_pct:.1f}%)"
            elif risk_rating and profile.max_drawdown_pct <= -30:
                if risk_rating["rating"] == "GREEN":
                    risk_rating["rating"] = "YELLOW"
                risk_rating["reason"] += f"; significant drawdown ({profile.max_drawdown_pct:.1f}%)"
            return UserProfileResponse(
                wallet_address=wallet_address,
                total_trades=profile.total_trades,
                total_mints_traded=profile.total_mints_traded,
                overall_win_rate=round(overall_wr, 2),
                avg_pnl_per_trade=round(profile.avg_pnl_per_trade, 6),
                total_realized_pnl=round(profile.total_realized_pnl, 6),
                lifetime_pnl=round(profile.lifetime_pnl, 2),
                total_volume_usd=round(profile.total_volume_usd, 2),
                max_drawdown_pct=round(profile.max_drawdown_pct, 4),
                avg_r_multiple=round(profile.avg_r_multiple, 4),
                avg_holding_time_minutes=round(profile.avg_holding_time_minutes, 2),
                avg_hold_time_hours=round(profile.avg_hold_time_hours, 2),
                typical_entry_phase=profile.typical_entry_phase,
                trader_type=_derive_trader_types(profile.is_sniper, profile.is_flipper, profile.is_holder),
                is_sniper=profile.is_sniper,
                is_holder=profile.is_holder,
                is_flipper=profile.is_flipper,
                fomo_tendency=d.get('fomo_tendency'),
                panic_sell_tendency=d.get('panic_sell_tendency'),
                dip_buy_tendency=d.get('dip_buy_tendency'),
                whale_follow_tendency=d.get('whale_follow_tendency'),
                is_valid=profile.is_valid,
                patterns=patterns_for_response,
                best_pattern=best,
                worst_pattern=worst,
                profile_description=None,
                risk_rating=risk_rating,
                recommendations=recommendations,
                current_market_context=market_ctx_dict,
                behavioral_biases_status=d.get('behavioral_biases_status'),
                behavioral_biases_message=d.get('behavioral_biases_message'),
                observation_window=observation_window,
                win_rate_round_trip=profile.win_rate_round_trip,
                win_rate_execution=profile.win_rate_execution,
                total_closed_pnl=round(profile.total_closed_pnl, 2) if profile.total_closed_pnl is not None else None,
                total_fees=round(profile.total_fees, 4) if profile.total_fees is not None else None,
                completed_round_trips=profile.completed_round_trips,
                trades_long=profile.trades_long,
                trades_short=profile.trades_short,
                avg_leverage=round(profile.avg_leverage, 2) if profile.avg_leverage is not None else None,
                bot_detected=profile.bot_detected,
                token_type="meme"
            )
    except Exception as profiler_error:
        logger.warning(f"UserProfiler path failed, falling back to simplified profile: {profiler_error}")
        import traceback
        traceback.print_exc()

    # Fallback: simplified profile from internal API (no pattern detection).
    # Uses the *same* from_date / to_date as the primary path and waits for
    # the user-profiling response (up to USER_PROFILING_TIMEOUT) before
    # returning — never returns an empty profile without waiting.
    # Normalize to_date so internal API does not return 400 (it rejects 23:59:59).
    to_date_internal = _normalize_to_date_for_internal_api(to_date)
    base_url = config.api.INTERNAL_BASE_URL.rstrip('/')
    fallback_timeout = config.api.USER_PROFILING_TIMEOUT  # same timeout as primary path (90s)
    summary_url = f"{base_url}/user-profiling"
    summary_params = {
        'address': wallet_address,
        'from': from_date,
        'to': to_date_internal,
        'type': 'spot'
    }
    trades_url = f"{base_url}/user-profiling/trades"
    trades_params = {
        'address': wallet_address,
        'from': from_date,
        'to': to_date_internal
    }
    logger.info(f"Fetching meme user profile from: {summary_url}")
    logger.info(f"Fetching meme user trades from: {trades_url}")
    try:
        summary_response = await asyncio.to_thread(requests.get, summary_url, params=summary_params, timeout=fallback_timeout)
        summary_raw = summary_response.json() if summary_response.status_code == 200 else {}
        summary_data = normalize_user_profiling_summary(summary_raw) if isinstance(summary_raw, dict) else {}
        trades_response = await asyncio.to_thread(requests.get, trades_url, params=trades_params, timeout=fallback_timeout)
        trades_data = trades_response.json() if trades_response.status_code == 200 else {}
        trades = trades_data.get('data', [])
        total_trades = len(trades)
        
        lifetime_pnl = summary_data.get('lifetime_pnl', 0.0)
        total_volume_usd = summary_data.get('total_volume_usd', 0.0)
        max_drawdown_pct = summary_data.get('max_drawdown_pct', 0.0)
        avg_r_multiple = summary_data.get('avg_r_multiple', 0.0)
        avg_holding_time_minutes = summary_data.get('avg_holding_time_minutes', 0.0)
        avg_hold_time_hours = avg_holding_time_minutes / 60.0 if avg_holding_time_minutes > 0 else 0.0
        
        win_rate = summary_data.get('win_rate', 0.0)
        if not win_rate and trades:
            profitable_trades = [t for t in trades if t.get('realized_pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('realized_pnl', 0) < 0]
            total_closed = len(profitable_trades) + len(losing_trades)
            if total_closed > 0:
                win_rate = (len(profitable_trades) / total_closed * 100)
        
        total_mints_traded = 0
        if trades:
            unique_mints = set(t.get('mint', '') for t in trades if t.get('mint'))
            total_mints_traded = len(unique_mints)
        
        avg_pnl_per_trade = lifetime_pnl / total_trades if total_trades > 0 else 0.0
        
        # trader_type (simplified path): from avg hold time — SNIPER <1min, FLIPPER <24h, HOLDER >168h, else MIXED
        trader_type = "MIXED"
        if avg_hold_time_hours > 0:
            if avg_holding_time_minutes < 1:
                trader_type = "SNIPER"
            elif avg_hold_time_hours < 24:
                trader_type = "FLIPPER"
            elif avg_hold_time_hours > 168:
                trader_type = "HOLDER"
        
        patterns = [{
            "pattern_type": "UNKNOWN",
            "occurrences": total_mints_traded,
            "win_rate": win_rate,
            "avg_pnl_percent": 0.0
        }]
        
        worst_pattern = {
            "pattern_type": "UNKNOWN",
            "win_rate": win_rate,
            "occurrences": total_mints_traded
        } if total_mints_traded > 0 else None
        
        risk_rating = None
        if win_rate > 0:
            if win_rate < 30:
                risk_rating = {"rating": "RED", "reason": f"win rate below 30% ({win_rate:.1f}%)"}
            elif win_rate < 50:
                risk_rating = {"rating": "YELLOW", "reason": f"win rate below 50% ({win_rate:.1f}%)"}
            else:
                risk_rating = {"rating": "GREEN", "reason": f"healthy win rate ({win_rate:.1f}%)"}
        if risk_rating and max_drawdown_pct <= -50:
            if risk_rating["rating"] != "RED":
                risk_rating["rating"] = "RED"
            risk_rating["reason"] += f"; severe drawdown ({max_drawdown_pct:.1f}%)"
        elif risk_rating and max_drawdown_pct <= -30:
            if risk_rating["rating"] == "GREEN":
                risk_rating["rating"] = "YELLOW"
            risk_rating["reason"] += f"; significant drawdown ({max_drawdown_pct:.1f}%)"
        
        fomo, panic, dip, whale, bias_status, bias_message = _behavioral_tendencies_for_response(
            total_trades, 0.0, 0.0, 0.0, 0.0
        )

        is_sniper_flag = trader_type == "SNIPER"
        is_holder_flag = trader_type == "HOLDER"
        is_flipper_flag = trader_type == "FLIPPER"

        recommendations = _build_meme_recommendations(
            total_trades=total_trades,
            overall_win_rate=win_rate,
            avg_pnl_per_trade=avg_pnl_per_trade,
            total_realized_pnl=lifetime_pnl,
            trader_type=trader_type,
            patterns=patterns,
            best_pattern=None,
            worst_pattern=worst_pattern,
            fomo_tendency=fomo,
            panic_sell_tendency=panic,
            dip_buy_tendency=dip,
            whale_follow_tendency=whale,
            is_sniper=is_sniper_flag,
            is_flipper=is_flipper_flag,
            is_holder=is_holder_flag,
            current_market_context=current_market_context,
            observation_window=observation_window,
            max_drawdown_pct=max_drawdown_pct,
        )

        typical_entry_phase = _compute_fallback_entry_phase(trades)

        logger.info(f"Meme profile built: {total_trades} trades, {win_rate:.1f}% win rate, ${lifetime_pnl:.2f} PnL")

        return UserProfileResponse(
            wallet_address=wallet_address,
            total_trades=total_trades,
            total_mints_traded=total_mints_traded,
            overall_win_rate=win_rate,
            avg_pnl_per_trade=avg_pnl_per_trade,
            total_realized_pnl=lifetime_pnl,
            lifetime_pnl=lifetime_pnl,
            total_volume_usd=total_volume_usd,
            max_drawdown_pct=max_drawdown_pct,
            avg_r_multiple=avg_r_multiple,
            avg_holding_time_minutes=avg_holding_time_minutes,
            avg_hold_time_hours=avg_hold_time_hours,
            typical_entry_phase=typical_entry_phase,
            trader_type=_derive_trader_types(is_sniper_flag, is_flipper_flag, is_holder_flag),
            is_sniper=is_sniper_flag,
            is_holder=is_holder_flag,
            is_flipper=is_flipper_flag,
            fomo_tendency=fomo,
            panic_sell_tendency=panic,
            dip_buy_tendency=dip,
            whale_follow_tendency=whale,
            is_valid=total_trades >= 10,
            patterns=patterns,
            best_pattern=None,
            worst_pattern=worst_pattern,
            profile_description=None,
            risk_rating=risk_rating,
            recommendations=recommendations,
            current_market_context=market_ctx_dict,
            behavioral_biases_status=bias_status,
            behavioral_biases_message=bias_message,
            observation_window=observation_window,
            win_rate_round_trip=summary_data.get('win_rate'),
            win_rate_execution=summary_data.get('win_rate_execution_based'),
            total_closed_pnl=summary_data.get('total_closed_pnl'),
            total_fees=summary_data.get('total_fees'),
            completed_round_trips=summary_data.get('completed_round_trips'),
            trades_long=summary_data.get('trades_long'),
            trades_short=summary_data.get('trades_short'),
            avg_leverage=summary_data.get('avg_leverage'),
            bot_detected=summary_data.get('bot_detected'),
            token_type="meme"
        )
        
    except Exception as fetch_error:
        logger.warning(f"Failed to process meme user profile: {fetch_error}")
        import traceback
        traceback.print_exc()
    
    fomo, panic, dip, whale, bias_status, bias_message = _behavioral_tendencies_for_response(
        0, 0.0, 0.0, 0.0, 0.0
    )
    return UserProfileResponse(
        wallet_address=wallet_address,
        total_trades=0,
        total_mints_traded=0,
        overall_win_rate=0.0,
        avg_pnl_per_trade=0.0,
        total_realized_pnl=0.0,
        lifetime_pnl=0.0,
        total_volume_usd=0.0,
        max_drawdown_pct=0.0,
        avg_r_multiple=0.0,
        avg_holding_time_minutes=0.0,
        avg_hold_time_hours=0.0,
        typical_entry_phase="UNKNOWN",
        trader_type=["MIXED"],
        is_sniper=False,
        is_holder=False,
        is_flipper=False,
        fomo_tendency=fomo,
        panic_sell_tendency=panic,
        dip_buy_tendency=dip,
        whale_follow_tendency=whale,
        is_valid=False,
        patterns=[],
        best_pattern=None,
        worst_pattern=None,
        profile_description=None,
        risk_rating=None,
        recommendations=["Insufficient trading history to generate recommendations."],
        current_market_context=market_ctx_dict,
        behavioral_biases_status=bias_status,
        behavioral_biases_message=bias_message,
        observation_window=observation_window,
        token_type="meme"
    )


async def _build_perps_profile(
    wallet_address: str,
    from_date: str,
    to_date: str,
    current_market_context,
    market_ctx_dict: Optional[Dict],
    observation_window: Optional[Dict[str, str]] = None,
) -> UserProfileResponse:
    """Build profile from perps/HyperEVM data."""
    from core.data_fetcher import PerpsDataFetcher
    
    perps_fetcher = PerpsDataFetcher()
    
    # Strip time portion for perps API (expects YYYY-MM-DD) and lag to_date by 1 day
    # because perps data for the current day is typically incomplete.
    perps_from = from_date[:10] if from_date else None
    if to_date:
        perps_to = (datetime.strptime(to_date[:10], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        perps_to = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching perps profile for {wallet_address[:10]}... from {perps_from} to {perps_to}")
    
    try:
        # Fetch trades and open positions concurrently
        trades_df, open_positions_raw = await asyncio.gather(
            asyncio.to_thread(
                perps_fetcher.fetch_user_perps_trades,
                wallet_address, perps_from, perps_to
            ),
            asyncio.to_thread(
                perps_fetcher.fetch_user_open_positions,
                wallet_address
            )
        )
        
        # Analyze the trades into a structured profile
        perps_profile = perps_fetcher.analyze_user_perps_profile(trades_df)
        
        total_trades = perps_profile.get('total_trades', 0)
        win_rate = perps_profile.get('win_rate', 0.0)
        total_pnl = perps_profile.get('total_pnl', 0.0)
        total_volume = perps_profile.get('total_volume', 0.0)
        # trader_type from PerpsDataFetcher.analyze_user_perps_profile: SCALPER / DAY_TRADER / SWING_TRADER / POSITION_TRADER by trades_per_day
        trader_type = perps_profile.get('trader_type', 'UNKNOWN')
        
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Normalize open positions into clean dicts
        positions_list = []
        max_leverage = 0.0
        for pos_entry in (open_positions_raw or []):
            pos = pos_entry.get('position', {})
            if not pos or not pos.get('coin'):
                continue
            leverage_val = 0.0
            lev_data = pos.get('leverage', {})
            if isinstance(lev_data, dict):
                leverage_val = float(lev_data.get('value', 0))
            elif lev_data:
                leverage_val = float(lev_data)
            
            entry_px = float(pos.get('entryPx', 0))
            liq_px = float(pos.get('liquidationPx', 0)) if pos.get('liquidationPx') else None
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))
            position_value = float(pos.get('positionValue', 0))
            size = pos.get('szi', '0')
            
            max_leverage = max(max_leverage, leverage_val)
            
            positions_list.append({
                "coin": pos.get('coin'),
                "size": size,
                "leverage": round(leverage_val, 2),
                "entry_price": round(entry_px, 4),
                "liquidation_price": round(liq_px, 4) if liq_px else None,
                "position_value": round(position_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 4)
            })
        
        # Risk rating (win rate + leverage/position risk)
        risk_rating = None
        if total_trades > 0:
            if win_rate < 30:
                risk_rating = {"rating": "RED", "reason": f"win rate below 30% ({win_rate:.1f}%)"}
            elif win_rate < 50 or max_leverage > 10:
                reason = f"win rate {win_rate:.1f}%"
                if max_leverage > 10:
                    reason += f", high leverage ({max_leverage:.1f}x)"
                risk_rating = {"rating": "YELLOW", "reason": reason}
            else:
                risk_rating = {"rating": "GREEN", "reason": f"healthy win rate ({win_rate:.1f}%)"}
        
        recommendations = _perps_recommendations(
            perps_profile, positions_list, max_leverage, current_market_context,
            total_realized_pnl=total_pnl,
            trader_type=trader_type,
            observation_window=observation_window,
        )
        recommendations.extend(_market_context_recommendations(current_market_context))
        
        # Highlight directional overexposure in risk_rating when BEAR + extreme fear + long_pct > 90%
        if current_market_context and not getattr(current_market_context, 'is_default', True):
            regime = getattr(current_market_context, 'market_regime', None)
            is_extreme_fear = getattr(current_market_context, 'is_extreme_fear', False)
            if regime == "BEAR" and is_extreme_fear and perps_profile.get('long_pct', 0) > 90:
                if risk_rating:
                    risk_rating["reason"] = (
                        risk_rating.get("reason", "") + "; directional overexposure in fearful bear market."
                    )
                else:
                    risk_rating = {
                        "rating": "YELLOW",
                        "reason": "Long bias in BEAR regime with extreme fear — directional overexposure risk.",
                    }
        
        # Profile description
        profit_factor = perps_profile.get('profit_factor', 0)
        direction_bias = perps_profile.get('direction_bias', 'UNKNOWN')
        trades_per_day = perps_profile.get('trades_per_day', 0)
        unique_coins = perps_profile.get('unique_coins', 0)
        
        profitable_str = "Profitable" if total_pnl > 0 else "Unprofitable"
        direction_str = direction_bias.lower().replace('_', ' ')
        if not direction_str.endswith('bias'):
            direction_str += ' bias'
        profile_desc = (
            f"{profitable_str} {trader_type.lower().replace('_', ' ')} with "
            f"{direction_str}. "
            f"Trades {trades_per_day:.1f} times/day across {unique_coins} coin(s). "
            f"Profit factor: {profit_factor:.2f}, win rate: {win_rate:.1f}%."
        )
        
        logger.info(f"Perps profile built: {total_trades} trades, {win_rate:.1f}% win rate, ${total_pnl:.2f} PnL")
        
        fomo, panic, dip, whale, bias_status, bias_message = _behavioral_tendencies_for_response(
            total_trades, 0.0, 0.0, 0.0, 0.0
        )
        # Override behavioral section when perps shows high leverage or overtrading
        from config import user_profile_config
        overtrading_threshold = user_profile_config.PERPS_OVERTRADING_TRADES_PER_DAY
        if max_leverage > 10 or (trades_per_day or 0) >= overtrading_threshold:
            bias_status = "perps_behavioral_concerns"
            if max_leverage > 10 and (trades_per_day or 0) >= overtrading_threshold:
                bias_message = "Overtrading and/or high leverage detected — consider reducing trade frequency and position size."
            elif max_leverage > 10:
                bias_message = "High leverage detected — consider reducing position size."
            else:
                bias_message = "Overtrading detected — consider reducing trade frequency."
        return UserProfileResponse(
            wallet_address=wallet_address,
            total_trades=total_trades,
            total_mints_traded=perps_profile.get('unique_coins', 0),
            overall_win_rate=win_rate,
            avg_pnl_per_trade=round(avg_pnl_per_trade, 4),
            total_realized_pnl=total_pnl,
            lifetime_pnl=total_pnl,
            total_volume_usd=total_volume,
            max_drawdown_pct=0.0,
            avg_r_multiple=0.0,
            avg_holding_time_minutes=0.0,
            avg_hold_time_hours=0.0,
            typical_entry_phase="N/A",
            trader_type=[trader_type] if trader_type != "UNKNOWN" else ["UNKNOWN"],
            is_sniper=trader_type == "SCALPER",
            is_holder=trader_type == "POSITION_TRADER",
            is_flipper=trader_type == "DAY_TRADER",
            fomo_tendency=fomo,
            panic_sell_tendency=panic,
            dip_buy_tendency=dip,
            whale_follow_tendency=whale,
            is_valid=total_trades >= 10,
            patterns=[],
            best_pattern=None,
            worst_pattern=None,
            profile_description=profile_desc,
            risk_rating=risk_rating,
            recommendations=recommendations,
            current_market_context=market_ctx_dict,
            behavioral_biases_status=bias_status,
            behavioral_biases_message=bias_message,
            observation_window=observation_window,
            token_type="perps",
            perps_profile=_sanitize_for_json(perps_profile),
        )
        
    except Exception as fetch_error:
        logger.warning(f"Failed to process perps user profile: {fetch_error}")
        import traceback
        traceback.print_exc()
    
    fomo, panic, dip, whale, bias_status, bias_message = _behavioral_tendencies_for_response(
        0, 0.0, 0.0, 0.0, 0.0
    )
    return UserProfileResponse(
        wallet_address=wallet_address,
        total_trades=0,
        total_mints_traded=0,
        overall_win_rate=0.0,
        avg_pnl_per_trade=0.0,
        total_realized_pnl=0.0,
        lifetime_pnl=0.0,
        total_volume_usd=0.0,
        max_drawdown_pct=0.0,
        avg_r_multiple=0.0,
        avg_holding_time_minutes=0.0,
        avg_hold_time_hours=0.0,
        typical_entry_phase="N/A",
        trader_type=["UNKNOWN"],
        is_sniper=False,
        is_holder=False,
        is_flipper=False,
        fomo_tendency=fomo,
        panic_sell_tendency=panic,
        dip_buy_tendency=dip,
        whale_follow_tendency=whale,
        is_valid=False,
        patterns=[],
        best_pattern=None,
        worst_pattern=None,
        profile_description=None,
        risk_rating=None,
        recommendations=["Insufficient perps trading history to generate recommendations."],
        current_market_context=market_ctx_dict,
        behavioral_biases_status=bias_status,
        behavioral_biases_message=bias_message,
        observation_window=observation_window,
        token_type="perps"
    )


def _perps_recommendations(
    perps_profile: Dict[str, Any],
    positions: List[Dict[str, Any]],
    max_leverage: float,
    current_market_context: Any = None,
    *,
    total_realized_pnl: float = 0.0,
    trader_type: str = "UNKNOWN",
    observation_window: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Generate perps-specific recommendations."""
    from config import user_profile_config
    recs = []

    total_trades = perps_profile.get('total_trades', 0)
    win_rate = perps_profile.get('win_rate', 0)
    profit_factor = perps_profile.get('profit_factor', 0)
    long_pct = perps_profile.get('long_pct', 50)
    short_pct = perps_profile.get('short_pct', 50)

    _has_market = (
        current_market_context
        and not getattr(current_market_context, 'is_default', True)
    )

    # --- Directional overexposure in BEAR + extreme fear ---
    if _has_market:
        regime = getattr(current_market_context, 'market_regime', None)
        is_extreme_fear = getattr(current_market_context, 'is_extreme_fear', False)
        fg = getattr(current_market_context, 'fear_greed', 50)
        if regime == "BEAR" and is_extreme_fear and long_pct > 90:
            recs.append(
                f"BEAR regime with extreme fear (F&G: {fg}) and strong long bias ({long_pct:.0f}% long) — "
                f"risk of directional overexposure; consider reducing long exposure or adding hedges."
            )

    # --- Market favorability vs trader style ---
    if _has_market:
        regime = getattr(current_market_context, 'market_regime', 'SIDEWAYS')
        fg = getattr(current_market_context, 'fear_greed', 50)
        tt_upper = trader_type.upper()
        if tt_upper == "SCALPER" and regime == "SIDEWAYS":
            recs.append(
                f"Sideways market (F&G: {fg}) — scalping opportunities exist but watch for "
                f"false breakouts; tighten stops."
            )
        if tt_upper == "POSITION_TRADER" and regime == "BULL":
            recs.append(
                "Bull regime aligns with longer holds. Consider trailing stops to lock gains."
            )
        if tt_upper in ("DAY_TRADER", "SCALPER") and regime == "BEAR":
            recs.append(
                f"Bear market (F&G: {fg}) — short-term trades carry higher whipsaw risk. "
                f"Reduce size or sit out low-conviction setups."
            )

    # --- Direction bias ---
    if long_pct > 70:
        recs.append(
            f"Direction bias is {long_pct:.0f}% LONG — consider diversifying with "
            f"short hedges during bearish macro conditions."
        )
    elif short_pct > 70:
        recs.append(
            f"Direction bias is {short_pct:.0f}% SHORT — may miss upside moves. "
            f"Consider balanced exposure during bullish trends."
        )

    # --- Win rate ---
    if win_rate < 40:
        recs.append(
            f"Win rate of {win_rate:.1f}% is below average — review entry criteria "
            f"and consider tighter stop-losses."
        )
    elif win_rate > 60:
        recs.append(f"Strong win rate of {win_rate:.1f}% — maintain current strategy discipline.")

    # --- Profit factor ---
    if profit_factor > 0:
        if profit_factor < 1.0:
            recs.append(
                f"Profit factor of {profit_factor:.2f} is below breakeven — "
                f"average losses exceed average wins, tighten risk management."
            )
        elif profit_factor > 2.0:
            recs.append(
                f"Excellent profit factor of {profit_factor:.2f} — "
                f"risk/reward discipline is strong."
            )

    # --- Total realized PnL (account growth) ---
    if total_realized_pnl > 0:
        recs.append(f"Account is net profitable (${total_realized_pnl:,.2f}) over the observation window.")
    elif total_realized_pnl < 0:
        recs.append(
            f"Account is net negative (${total_realized_pnl:,.2f}). "
            f"Review strategy and tighten risk management."
        )

    # --- High PnL + low win rate ---
    if total_realized_pnl > 0 and win_rate < 50:
        recs.append(
            "Positive PnL despite a sub-50% win rate — a few big wins are carrying results. "
            "Tighten stops on losers to improve consistency."
        )

    # --- Leverage from open positions ---
    if max_leverage > 10:
        recs.append(
            f"High leverage detected ({max_leverage:.1f}x) — consider reducing "
            f"position sizes to lower liquidation risk."
        )

    # --- Overtrading ---
    trades_per_day = perps_profile.get('trades_per_day', 0)
    if (trades_per_day or 0) >= user_profile_config.PERPS_OVERTRADING_TRADES_PER_DAY:
        recs.append(
            f"Very high trade count ({trades_per_day:.0f}/day) — possible overtrading; "
            f"consider fewer, higher-conviction trades."
        )

    # --- Liquidation proximity ---
    for pos in positions:
        liq_px = pos.get('liquidation_price')
        entry_px = pos.get('entry_price', 0)
        if liq_px and entry_px > 0:
            distance_pct = abs(entry_px - liq_px) / entry_px * 100
            if distance_pct < 5:
                recs.append(
                    f"Position in {pos['coin']} is only {distance_pct:.1f}% from liquidation — "
                    f"consider adding margin or reducing size."
                )

    # --- Observation window + total trades ---
    if observation_window:
        fd = observation_window.get('from_date', '?')[:10]
        td = observation_window.get('to_date', '?')[:10]
        recs.append(f"Based on {total_trades} trades from {fd} to {td}.")
    elif total_trades:
        recs.append(f"Based on {total_trades} trades in the observation window.")

    if not recs:
        recs.append("Trading metrics look healthy — continue current approach.")

    return recs[:_MAX_RECOMMENDATIONS]


@router.post("/user/assess", summary="Assess User Risk")
async def assess_user_risk(request: UserRiskAssessmentRequest):
    """
    Assess trade risk for specific user and token.
    
    Response shape depends on ``token_type``: perps-only keys in
    ``profile_summary`` are omitted for meme and vice-versa.
    
    Returns GREEN/YELLOW/RED rating based on:
    - **Meme**: whale state, holder concentration, token phase
    - **Perps**: leverage, liquidation proximity, funding rate, user trading history
    """
    try:
        if request.token_type == "perps":
            assessment = await _assess_perps_risk(request)
        else:
            assessment = _assess_meme_risk(request)
        
        return filter_response_by_token_type(
            assessment.model_dump(exclude_none=True),
            request.token_type,
            recurse_keys={"profile_summary"},
        )
        
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("risk_assessment_request", {
                "wallet_address": request.wallet_address,
                "token_address": request.token_address,
                "token_type": request.token_type,
                "trade_type": getattr(request, "trade_type", None),
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Error assessing user risk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assess user risk: {e}")


def _assess_meme_risk(request: UserRiskAssessmentRequest) -> RiskAssessmentResponse:
    """Meme/spot risk assessment (original logic)."""
    confidence = 0.75
    
    if request.whale_state.lower() in ["accumulation", "accumulating"]:
        rating = "GREEN"
        signal = "BUY"
        message = f"Positive signal: Whales are accumulating in {request.phase}"
    elif request.whale_state.lower() in ["distribution", "distributing"]:
        rating = "RED"
        signal = "SELL"
        message = f"Warning: Whales are distributing in {request.phase}"
    else:
        rating = "YELLOW"
        signal = "HOLD"
        message = f"Neutral: Whale activity is stable in {request.phase}"
    
    if request.top10_concentration > 80:
        if rating == "GREEN":
            rating = "YELLOW"
        elif rating == "YELLOW":
            rating = "RED"
        message += " (high concentration risk)"
        confidence *= 0.8
    
    signal_weight = confidence if rating == "GREEN" else -confidence if rating == "RED" else 0
    
    risk_factors = []
    if request.top10_concentration > 80:
        risk_factors.append(f"High concentration: {request.top10_concentration:.1f}%")
    if request.phase == "P1":
        risk_factors.append("Early launch phase - high volatility")
    if request.whale_state.lower() == "distribution":
        risk_factors.append("Whales are selling")
    
    if not risk_factors:
        risk_factors.append("Normal risk conditions")
    
    return RiskAssessmentResponse(
        rating=rating,
        confidence=round(confidence, 4),
        signal=signal,
        signal_weight=round(signal_weight, 4),
        message=message,
        risk_factors=risk_factors,
        matching_pattern={
            "type": "UNKNOWN",
            "win_rate": 0.5,
            "occurrences": 0
        },
        profile_summary={
            "wallet_address": request.wallet_address,
            "total_trades": 0,
            "win_rate": 0,
            "is_valid": False
        }
    )


async def _assess_perps_risk(request: UserRiskAssessmentRequest) -> RiskAssessmentResponse:
    """Perps risk assessment using leverage, liquidation, funding rate, and trade history."""
    from core.data_fetcher import PerpsDataFetcher
    
    perps_fetcher = PerpsDataFetcher()
    confidence = 0.75
    risk_factors = []
    
    # Fetch open positions and (optionally) market data concurrently
    open_positions_raw = await asyncio.to_thread(
        perps_fetcher.fetch_user_open_positions, request.wallet_address
    )
    
    market_data = None
    if request.coin:
        try:
            market_data = await asyncio.to_thread(
                perps_fetcher.fetch_market_data, request.coin
            )
        except Exception as e:
            logger.warning(f"Failed to fetch market data for {request.coin}: {e}")
    
    # Fetch user profile for trade history context
    user_profile = None
    try:
        trades_df = await asyncio.to_thread(
            perps_fetcher.fetch_user_perps_trades, request.wallet_address
        )
        user_profile = perps_fetcher.analyze_user_perps_profile(trades_df)
    except Exception as e:
        logger.warning(f"Failed to fetch perps trade history: {e}")
    
    # --- Analyze positions ---
    max_leverage = 0.0
    min_liq_distance_pct = 100.0
    has_position_in_coin = False
    
    for pos_entry in (open_positions_raw or []):
        pos = pos_entry.get('position', {})
        if not pos or not pos.get('coin'):
            continue
        
        leverage_val = 0.0
        lev_data = pos.get('leverage', {})
        if isinstance(lev_data, dict):
            leverage_val = float(lev_data.get('value', 0))
        elif lev_data:
            leverage_val = float(lev_data)
        
        max_leverage = max(max_leverage, leverage_val)
        
        entry_px = float(pos.get('entryPx', 0))
        liq_px = float(pos.get('liquidationPx', 0)) if pos.get('liquidationPx') else None
        
        if liq_px and entry_px > 0:
            distance = abs(entry_px - liq_px) / entry_px * 100
            min_liq_distance_pct = min(min_liq_distance_pct, distance)
        
        if request.coin and pos.get('coin', '').upper() == request.coin.upper():
            has_position_in_coin = True
    
    # --- Determine rating ---
    rating = "GREEN"
    signal = "HOLD"
    message = "Perps risk assessment"
    
    # Leverage check
    if max_leverage > 20:
        rating = "RED"
        risk_factors.append(f"Extremely high leverage: {max_leverage:.1f}x")
        confidence *= 0.6
    elif max_leverage > 10:
        if rating != "RED":
            rating = "YELLOW"
        risk_factors.append(f"High leverage: {max_leverage:.1f}x")
        confidence *= 0.8
    
    # Liquidation proximity check
    if min_liq_distance_pct < 3:
        rating = "RED"
        risk_factors.append(f"Critical: position only {min_liq_distance_pct:.1f}% from liquidation")
        confidence *= 0.6
    elif min_liq_distance_pct < 5:
        if rating != "RED":
            rating = "YELLOW"
        risk_factors.append(f"Warning: position {min_liq_distance_pct:.1f}% from liquidation")
        confidence *= 0.8
    
    # Funding rate check (from market data)
    if market_data is not None:
        funding_rate = None
        if hasattr(market_data, 'get'):
            funding_rate = market_data.get('funding_rate') or market_data.get('next_funding_rate')
        elif hasattr(market_data, 'funding_rate'):
            funding_rate = getattr(market_data, 'funding_rate', None)
        
        if funding_rate is not None:
            try:
                fr_val = abs(float(funding_rate))
                if fr_val > 0.0005:
                    if rating != "RED":
                        rating = "YELLOW"
                    risk_factors.append(f"Extreme funding rate: {float(funding_rate)*100:.4f}%")
                    confidence *= 0.85
                elif fr_val > 0.0003:
                    risk_factors.append(f"Elevated funding rate: {float(funding_rate)*100:.4f}%")
            except (ValueError, TypeError):
                pass
    
    # Whale state check (applicable to both meme and perps)
    if request.whale_state.lower() in ["distribution", "distributing"]:
        if rating == "GREEN":
            rating = "YELLOW"
        risk_factors.append("Large traders are distributing")
    elif request.whale_state.lower() in ["accumulation", "accumulating"]:
        risk_factors.append("Large traders are accumulating (positive)")
    
    # Determine signal from rating
    if rating == "GREEN":
        signal = "BUY"
        message = "Low risk: healthy leverage and position metrics"
    elif rating == "RED":
        signal = "SELL"
        message = "High risk: critical leverage or liquidation proximity detected"
    else:
        signal = "HOLD"
        message = "Moderate risk: review position sizing and leverage"
    
    if not risk_factors:
        risk_factors.append("Normal risk conditions")
    
    signal_weight = confidence if rating == "GREEN" else -confidence if rating == "RED" else 0
    
    # Build profile summary from trade history
    profile_total_trades = user_profile.get('total_trades', 0) if user_profile else 0
    profile_win_rate = user_profile.get('win_rate', 0) if user_profile else 0
    
    return RiskAssessmentResponse(
        rating=rating,
        confidence=round(confidence, 4),
        signal=signal,
        signal_weight=round(signal_weight, 4),
        message=message,
        risk_factors=risk_factors,
        matching_pattern=None,
        profile_summary={
            "wallet_address": request.wallet_address,
            "token_type": "perps",
            "total_trades": profile_total_trades,
            "win_rate": profile_win_rate,
            "is_valid": profile_total_trades >= 10,
            "max_leverage": round(max_leverage, 2),
            "min_liquidation_distance_pct": round(min_liq_distance_pct, 2) if min_liq_distance_pct < 100 else None,
            "open_positions_count": len(open_positions_raw) if open_positions_raw else 0,
            "has_position_in_coin": has_position_in_coin
        }
    )
