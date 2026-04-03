"""
AI-Powered User Profile Route

GET /user/ai-profile/{wallet_address}

Reuses the same raw-data fetching layer as the existing /user/profile endpoint
but delegates behavioral analysis (patterns, tendencies, risk rating, trader
type, recommendations) to an LLM via LLMAnalyzerService.

The existing /user/profile endpoint is completely untouched.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

import sentry_sdk
from fastapi import APIRouter, HTTPException, Query, Path as FastPath

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.response_schema import filter_response_by_token_type, strip_perps_redundant_top_level
from api.routes.users import UserProfileResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Profile"])

_TRADE_CAP_FOR_LLM = 200


# =============================================================================
# Endpoint
# =============================================================================

@router.get("/user/ai-profile/{wallet_address}", summary="AI-Powered Trader Profile")
async def get_ai_profile(
    wallet_address: str = FastPath(
        ..., description="User wallet address", min_length=32, max_length=64
    ),
    from_date: Optional[str] = Query(
        None, description="Start date (ISO format: 2025-01-01T00:00:00Z)"
    ),
    to_date: Optional[str] = Query(
        None, description="End date (ISO format: 2025-01-05T00:00:00Z)"
    ),
    token_type: str = Query("meme", description="Token type: 'meme' or 'perps'"),
):
    """
    AI-powered trader profile. Returns the **same response shape** as
    ``GET /user/profile/{wallet_address}`` but behavioral analysis
    (patterns, tendencies, risk rating, recommendations) is generated
    by Claude / GPT instead of the internal scoring engine.

    Raw metrics (trade counts, PnL, volume, hold times) are identical
    to the original endpoint — only the behavioral interpretation differs.
    """
    from config import get_config, llm_analyzer_config
    from core.market_context_fetcher import MarketContextFetcher

    if not llm_analyzer_config.ENABLED:
        raise HTTPException(status_code=503, detail="AI profile analysis is disabled")

    token_type = token_type.lower().strip()

    try:
        config = get_config()

        # -- Market context (shared) --
        current_market_context = None
        market_ctx_dict = None
        try:
            market_ctx_fetcher = MarketContextFetcher()
            current_market_context = await asyncio.to_thread(market_ctx_fetcher.fetch_latest)
            if current_market_context and hasattr(current_market_context, "to_dict"):
                market_ctx_dict = current_market_context.to_dict()
        except Exception as e:
            logger.warning(f"AI profile: failed to fetch market context: {e}")

        # -- Default date window (same logic as existing endpoint) --
        if from_date is None or to_date is None:
            api_cfg = config.api
            to_dt = datetime.utcnow()
            default_days = getattr(api_cfg, "USER_PROFILING_DEFAULT_DAYS", 80)
            from_dt = to_dt - timedelta(days=default_days)
            from_date = from_date or from_dt.strftime("%Y-%m-%dT00:00:00Z")
            to_date = to_date or to_dt.strftime("%Y-%m-%dT23:59:59Z")

        observation_window = {"from_date": from_date, "to_date": to_date}

        # -- Dispatch by token type --
        if token_type == "perps":
            profile = await _build_ai_perps_profile(
                wallet_address, from_date, to_date,
                market_ctx_dict, observation_window,
            )
        else:
            profile = await _build_ai_meme_profile(
                wallet_address, from_date, to_date,
                market_ctx_dict, observation_window, config,
            )

        # -- Same response filtering as existing endpoint --
        profile_dict = profile.model_dump(exclude_none=True)
        filtered = filter_response_by_token_type(
            profile_dict, token_type, recurse_keys=set()
        )
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
            scope.set_context("ai_profile_request", {
                "wallet_address": wallet_address,
                "from_date": from_date,
                "to_date": to_date,
                "token_type": token_type,
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"AI profile error: {e}")
        raise HTTPException(status_code=502, detail=f"AI profile analysis failed: {e}")


# =============================================================================
# Meme path
# =============================================================================

async def _build_ai_meme_profile(
    wallet_address: str,
    from_date: str,
    to_date: str,
    market_ctx_dict: Optional[Dict],
    observation_window: Dict[str, str],
    config,
) -> UserProfileResponse:
    """Fetch raw meme data, send to LLM, merge into UserProfileResponse."""
    from core.data_fetcher import DataFetcher
    from services.llm_analyzer_service import LLMAnalyzerService
    from config import user_profile_config

    data_fetcher = DataFetcher(
        base_url=config.api.INTERNAL_BASE_URL.rstrip("/"),
        limiter_name="users",
    )

    profile_data = await asyncio.to_thread(
        data_fetcher.fetch_user_complete_profile,
        wallet_address,
        from_date=from_date,
        to_date=to_date,
    )

    if not profile_data:
        raise HTTPException(status_code=404, detail="No trading data found for this wallet")

    summary = profile_data.get("summary", {})
    trades = profile_data.get("trades", [])

    total_trades = len(trades)
    is_valid = total_trades >= user_profile_config.MIN_TRADES_FOR_PROFILE

    # -- Aggregate raw metrics (pure math, same as existing endpoint) --
    lifetime_pnl = summary.get("lifetime_pnl", 0.0)
    total_volume_usd = summary.get("total_volume_usd", 0.0)
    max_drawdown_pct = summary.get("max_drawdown_pct", 0.0)
    avg_r_multiple = summary.get("avg_r_multiple", 0.0)
    avg_holding_time_minutes = summary.get("avg_holding_time_minutes", 0.0)
    avg_hold_time_hours = avg_holding_time_minutes / 60.0 if avg_holding_time_minutes else 0.0

    unique_mints = set()
    winning = 0
    total_pnl = 0.0
    for t in trades:
        mint = t.get("mint") or t.get("token") or t.get("token_address")
        if mint:
            unique_mints.add(mint)
        pnl = float(t.get("pnl") or t.get("realized_pnl") or t.get("profit") or 0)
        total_pnl += pnl
        if pnl > 0:
            winning += 1
    total_mints_traded = len(unique_mints)
    overall_win_rate = (winning / total_trades * 100) if total_trades > 0 else 0.0
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0

    typical_entry_phase = _compute_entry_phase(trades)

    raw_metrics = {
        "total_trades": total_trades,
        "total_mints_traded": total_mints_traded,
        "overall_win_rate": round(overall_win_rate, 2),
        "avg_pnl_per_trade": round(avg_pnl_per_trade, 6),
        "total_realized_pnl": round(total_pnl, 6),
        "lifetime_pnl": round(lifetime_pnl, 2),
        "total_volume_usd": round(total_volume_usd, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "avg_r_multiple": round(avg_r_multiple, 4),
        "avg_holding_time_minutes": round(avg_holding_time_minutes, 2),
        "avg_hold_time_hours": round(avg_hold_time_hours, 2),
        "typical_entry_phase": typical_entry_phase,
    }
    if summary.get('win_rate') is not None:
        raw_metrics["win_rate_round_trip"] = summary['win_rate']
    if summary.get('total_closed_pnl') is not None:
        raw_metrics["total_closed_pnl"] = round(summary['total_closed_pnl'], 2)
    if summary.get('total_fees') is not None:
        raw_metrics["total_fees"] = round(summary['total_fees'], 4)
    if summary.get('avg_leverage') is not None:
        raw_metrics["avg_leverage"] = round(summary['avg_leverage'], 2)
    if summary.get('bot_detected') is not None:
        raw_metrics["bot_detected"] = summary['bot_detected']

    trade_details = _slim_meme_trades(trades)

    raw_data = {
        "wallet_address": wallet_address,
        "token_type": "meme",
        "is_valid": is_valid,
        "observation_window": observation_window,
        "market_context": market_ctx_dict,
        "raw_metrics": raw_metrics,
        "trade_details": trade_details,
    }

    llm = LLMAnalyzerService()
    ai = await asyncio.to_thread(llm.analyze, raw_data, "meme")

    return UserProfileResponse(
        wallet_address=wallet_address,
        total_trades=total_trades,
        total_mints_traded=total_mints_traded,
        overall_win_rate=round(overall_win_rate, 2),
        avg_pnl_per_trade=round(avg_pnl_per_trade, 6),
        total_realized_pnl=round(total_pnl, 6),
        lifetime_pnl=round(lifetime_pnl, 2),
        total_volume_usd=round(total_volume_usd, 2),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        avg_r_multiple=round(avg_r_multiple, 4),
        avg_holding_time_minutes=round(avg_holding_time_minutes, 2),
        avg_hold_time_hours=round(avg_hold_time_hours, 2),
        typical_entry_phase=typical_entry_phase,
        trader_type=ai.get("trader_type", ["UNKNOWN"]),
        is_sniper=ai.get("is_sniper", False),
        is_holder=ai.get("is_holder", False),
        is_flipper=ai.get("is_flipper", False),
        fomo_tendency=ai.get("fomo_tendency", 0.0),
        panic_sell_tendency=ai.get("panic_sell_tendency", 0.0),
        dip_buy_tendency=ai.get("dip_buy_tendency", 0.0),
        whale_follow_tendency=ai.get("whale_follow_tendency", 0.0),
        is_valid=is_valid,
        patterns=ai.get("patterns", []),
        best_pattern=ai.get("best_pattern"),
        worst_pattern=ai.get("worst_pattern"),
        profile_description=ai.get("profile_description"),
        risk_rating=ai.get("risk_rating"),
        recommendations=ai.get("recommendations", []),
        current_market_context=market_ctx_dict,
        behavioral_biases_status="ai_computed",
        behavioral_biases_message="Behavioral analysis generated by AI",
        observation_window=observation_window,
        token_type="meme",
    )


# =============================================================================
# Perps path
# =============================================================================

async def _build_ai_perps_profile(
    wallet_address: str,
    from_date: str,
    to_date: str,
    market_ctx_dict: Optional[Dict],
    observation_window: Dict[str, str],
) -> UserProfileResponse:
    """Fetch raw perps data, send to LLM, merge into UserProfileResponse."""
    from core.data_fetcher import PerpsDataFetcher
    from services.llm_analyzer_service import LLMAnalyzerService

    perps_fetcher = PerpsDataFetcher()

    perps_from = from_date[:10] if from_date else None
    if to_date:
        perps_to = (
            datetime.strptime(to_date[:10], "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")
    else:
        perps_to = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    trades_df, open_positions_raw = await asyncio.gather(
        asyncio.to_thread(
            perps_fetcher.fetch_user_perps_trades,
            wallet_address, perps_from, perps_to,
        ),
        asyncio.to_thread(
            perps_fetcher.fetch_user_open_positions,
            wallet_address,
        ),
    )

    perps_profile = perps_fetcher.analyze_user_perps_profile(trades_df)

    total_trades = perps_profile.get("total_trades", 0)
    win_rate = perps_profile.get("win_rate", 0.0)
    total_pnl = perps_profile.get("total_pnl", 0.0)
    total_volume = perps_profile.get("total_volume", 0.0)
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
    is_valid = total_trades >= 10

    positions_list = _normalize_positions(open_positions_raw)
    max_leverage = max((p["leverage"] for p in positions_list), default=0.0)

    raw_metrics = {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_volume": round(total_volume, 2),
        "avg_pnl_per_trade": round(avg_pnl_per_trade, 4),
        "profit_factor": perps_profile.get("profit_factor", 0),
        "long_pct": perps_profile.get("long_pct", 50),
        "short_pct": perps_profile.get("short_pct", 50),
        "direction_bias": perps_profile.get("direction_bias", "UNKNOWN"),
        "trades_per_day": perps_profile.get("trades_per_day", 0),
        "unique_coins": perps_profile.get("unique_coins", 0),
        "top_coins": perps_profile.get("top_coins", {}),
        "favorite_coin": perps_profile.get("favorite_coin"),
        "max_leverage": round(max_leverage, 2),
        "avg_win": perps_profile.get("avg_win", 0),
        "avg_loss": perps_profile.get("avg_loss", 0),
        "total_fees": perps_profile.get("total_fees", 0),
    }

    trade_details = _slim_perps_trades(trades_df)

    raw_data = {
        "wallet_address": wallet_address,
        "token_type": "perps",
        "is_valid": is_valid,
        "observation_window": observation_window,
        "market_context": market_ctx_dict,
        "raw_metrics": raw_metrics,
        "trade_details": trade_details,
        "open_positions": positions_list,
    }

    llm = LLMAnalyzerService()
    ai = await asyncio.to_thread(llm.analyze, raw_data, "perps")

    return UserProfileResponse(
        wallet_address=wallet_address,
        total_trades=total_trades,
        total_mints_traded=perps_profile.get("unique_coins", 0),
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
        trader_type=ai.get("trader_type", ["UNKNOWN"]),
        is_sniper=ai.get("is_sniper", False),
        is_holder=ai.get("is_holder", False),
        is_flipper=ai.get("is_flipper", False),
        fomo_tendency=ai.get("fomo_tendency", 0.0),
        panic_sell_tendency=ai.get("panic_sell_tendency", 0.0),
        dip_buy_tendency=ai.get("dip_buy_tendency", 0.0),
        whale_follow_tendency=ai.get("whale_follow_tendency", 0.0),
        is_valid=is_valid,
        patterns=ai.get("patterns", []),
        best_pattern=ai.get("best_pattern"),
        worst_pattern=ai.get("worst_pattern"),
        profile_description=ai.get("profile_description"),
        risk_rating=ai.get("risk_rating"),
        recommendations=ai.get("recommendations", []),
        current_market_context=market_ctx_dict,
        behavioral_biases_status="ai_computed",
        behavioral_biases_message="Behavioral analysis generated by AI",
        observation_window=observation_window,
        token_type="perps",
        perps_profile=_sanitize_for_json(perps_profile),
        open_positions=positions_list if positions_list else None,
    )


# =============================================================================
# Helpers
# =============================================================================

def _compute_entry_phase(trades: List[Dict]) -> str:
    """Compute typical entry phase from trade timestamps and token creation time."""
    try:
        import pandas as pd
    except ImportError:
        return "UNKNOWN"

    phase_counts: Dict[str, int] = {}
    for trade in trades:
        is_buy = (
            trade.get("is_buy") or trade.get("isBuy")
            or trade.get("type", "").upper() == "BUY"
            or trade.get("side", "").upper() == "BUY"
        )
        if not is_buy:
            continue
        creation_ts = trade.get("creation_timestamp")
        if not creation_ts or creation_ts <= 0:
            continue
        executed_at_str = (
            trade.get("executed_at") or trade.get("timestamp")
            or trade.get("time") or trade.get("created_at")
        )
        if not executed_at_str:
            continue
        try:
            executed_at = pd.to_datetime(str(executed_at_str), utc=True)
            token_created_at = pd.to_datetime(int(creation_ts), unit="s", utc=True)
            age_days = (executed_at - token_created_at).total_seconds() / 86400.0
        except Exception:
            continue
        if age_days < 0:
            phase = "UNKNOWN"
        elif age_days <= 3:
            phase = "P1"
        elif age_days <= 14:
            phase = "P2"
        elif age_days <= 45:
            phase = "P3"
        else:
            phase = "P4"
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    return max(phase_counts, key=phase_counts.get) if phase_counts else "UNKNOWN"


def _slim_meme_trades(trades: List[Dict]) -> List[Dict]:
    """Extract essential fields from meme trades, capped for LLM context."""
    sorted_trades = sorted(
        trades,
        key=lambda t: t.get("executed_at") or t.get("timestamp") or "",
        reverse=True,
    )[:_TRADE_CAP_FOR_LLM]

    result = []
    for t in sorted_trades:
        pnl = t.get("pnl") or t.get("realized_pnl") or t.get("profit") or 0
        result.append({
            "token": t.get("token_symbol") or t.get("symbol") or t.get("mint", "")[:12],
            "side": "buy" if (
                t.get("is_buy") or t.get("isBuy")
                or t.get("type", "").upper() == "BUY"
                or t.get("side", "").upper() == "BUY"
            ) else "sell",
            "pnl_usd": round(float(pnl), 4) if pnl else 0,
            "timestamp": str(
                t.get("executed_at") or t.get("timestamp") or t.get("time") or ""
            ),
            "hold_duration_minutes": t.get("hold_duration_minutes"),
            "token_phase_at_entry": t.get("token_phase_at_entry") or t.get("phase"),
        })
    return result


def _slim_perps_trades(trades_df) -> List[Dict]:
    """Extract essential fields from perps trades DataFrame, capped for LLM context."""
    import pandas as pd
    if trades_df is None or (isinstance(trades_df, pd.DataFrame) and trades_df.empty):
        return []

    df = trades_df.copy()
    if "timestamp" in df.columns:
        try:
            df = df.sort_values("timestamp", ascending=False)
        except Exception:
            pass
    df = df.head(_TRADE_CAP_FOR_LLM)

    result = []
    for _, row in df.iterrows():
        result.append({
            "coin": str(row.get("coin", "")),
            "direction": str(row.get("direction", "")),
            "closed_pnl": round(float(row.get("closed_pnl", 0) or 0), 4),
            "trade_value": round(float(row.get("trade_value", 0) or 0), 2),
            "timestamp": str(row.get("timestamp", "")),
        })
    return result


def _normalize_positions(open_positions_raw) -> List[Dict]:
    """Clean open position data into simple dicts."""
    positions = []
    for pos_entry in (open_positions_raw or []):
        pos = pos_entry.get("position", {})
        if not pos or not pos.get("coin"):
            continue
        leverage_val = 0.0
        lev_data = pos.get("leverage", {})
        if isinstance(lev_data, dict):
            leverage_val = float(lev_data.get("value", 0))
        elif lev_data:
            leverage_val = float(lev_data)
        entry_px = float(pos.get("entryPx", 0))
        liq_px = float(pos.get("liquidationPx", 0)) if pos.get("liquidationPx") else None
        unrealized_pnl = float(pos.get("unrealizedPnl", 0))
        position_value = float(pos.get("positionValue", 0))
        positions.append({
            "coin": pos.get("coin"),
            "size": pos.get("szi", "0"),
            "leverage": round(leverage_val, 2),
            "entry_price": round(entry_px, 4),
            "liquidation_price": round(liq_px, 4) if liq_px else None,
            "position_value": round(position_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 4),
        })
    return positions


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
