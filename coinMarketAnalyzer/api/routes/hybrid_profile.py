"""
Hybrid User Profile Route

GET /user/hybrid-profile/{wallet_address}

Runs the existing internal calculation engine first (UserProfiler for meme,
PerpsDataFetcher for perps), then feeds BOTH the raw trade data AND the
engine's computed analysis to an LLM for refinement.

The LLM acts as a reviewer: it validates, corrects, and enhances the
rule-based outputs while staying anchored to real calculations.

The existing /user/profile and /user/ai-profile endpoints are completely
untouched.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

import sentry_sdk
from fastapi import APIRouter, HTTPException, Query, Path as FastPath

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.response_schema import filter_response_by_token_type, strip_perps_redundant_top_level
from api.routes.users import (
    UserProfileResponse,
    _build_meme_recommendations,
    _derive_trader_types,
    _market_context_recommendations,
    _perps_recommendations,
    _behavioral_tendencies_for_response,
    _sanitize_for_json,
)
from api.routes.ai_profile import (
    _slim_meme_trades,
    _slim_perps_trades,
    _normalize_positions,
    _compute_entry_phase,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Hybrid Profile"])

_TRADE_CAP_FOR_LLM = 200


# =============================================================================
# Endpoint
# =============================================================================

@router.get("/user/hybrid-profile/{wallet_address}", summary="Hybrid Trader Profile")
async def get_hybrid_profile(
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
    Hybrid trader profile combining internal engine analysis with LLM
    refinement. Returns the **same response shape** as
    ``GET /user/profile/{wallet_address}``.

    The internal engine computes patterns, tendencies, risk rating, and
    recommendations first. These are then sent alongside the raw trade data
    to the LLM, which validates, corrects, and enhances the analysis.

    If the LLM call fails, the endpoint gracefully falls back to the pure
    internal engine output (identical to ``/user/profile``).
    """
    from config import get_config, llm_analyzer_config
    from core.market_context_fetcher import MarketContextFetcher

    if not llm_analyzer_config.ENABLED:
        raise HTTPException(status_code=503, detail="AI profile analysis is disabled")

    token_type = token_type.lower().strip()

    try:
        config = get_config()

        current_market_context = None
        market_ctx_dict = None
        try:
            market_ctx_fetcher = MarketContextFetcher()
            current_market_context = await asyncio.to_thread(market_ctx_fetcher.fetch_latest)
            if current_market_context and hasattr(current_market_context, "to_dict"):
                market_ctx_dict = current_market_context.to_dict()
        except Exception as e:
            logger.warning(f"Hybrid profile: failed to fetch market context: {e}")

        if from_date is None or to_date is None:
            api_cfg = config.api
            to_dt = datetime.utcnow()
            default_days = getattr(api_cfg, "USER_PROFILING_DEFAULT_DAYS", 80)
            from_dt = to_dt - timedelta(days=default_days)
            from_date = from_date or from_dt.strftime("%Y-%m-%dT00:00:00Z")
            to_date = to_date or to_dt.strftime("%Y-%m-%dT23:59:59Z")

        observation_window = {"from_date": from_date, "to_date": to_date}

        if token_type == "perps":
            profile = await _build_hybrid_perps_profile(
                wallet_address, from_date, to_date,
                current_market_context, market_ctx_dict,
                observation_window,
            )
        else:
            profile = await _build_hybrid_meme_profile(
                wallet_address, from_date, to_date,
                current_market_context, market_ctx_dict,
                observation_window, config,
            )

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
            scope.set_context("hybrid_profile_request", {
                "wallet_address": wallet_address,
                "from_date": from_date,
                "to_date": to_date,
                "token_type": token_type,
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Hybrid profile error: {e}")
        raise HTTPException(status_code=502, detail=f"Hybrid profile analysis failed: {e}")


# =============================================================================
# Meme path
# =============================================================================

async def _build_hybrid_meme_profile(
    wallet_address: str,
    from_date: str,
    to_date: str,
    current_market_context,
    market_ctx_dict: Optional[Dict],
    observation_window: Dict[str, str],
    config,
) -> UserProfileResponse:
    """
    Run the internal UserProfiler engine, compute rule-based analysis,
    then send both raw data and engine output to LLM for refinement.
    Falls back to pure engine output if the LLM call fails.
    """
    from core.data_fetcher import DataFetcher
    from engines.user_profiler import UserProfiler, profile_to_dict
    from services.llm_analyzer_service import LLMAnalyzerService
    from config import user_profile_config

    data_fetcher = DataFetcher(
        base_url=config.api.INTERNAL_BASE_URL.rstrip("/"),
        limiter_name="users",
    )

    # ── Step 1: Run the internal engine (same as /user/profile) ──

    profiler = UserProfiler(data_fetcher=data_fetcher)
    profiler.invalidate_cache(wallet_address)

    profile = await asyncio.to_thread(
        profiler.get_profile,
        wallet_address,
        use_birdeye_fallback=False,
        from_date=from_date,
        to_date=to_date,
    )

    if profile is None or profile.total_trades == 0:
        raise HTTPException(status_code=404, detail="No trading data found for this wallet")

    d = profile_to_dict(profile)

    overall_wr = profile.overall_win_rate
    if overall_wr <= 1.0:
        overall_wr = overall_wr * 100.0

    patterns_for_response = []
    for p in d["patterns"]:
        wr = p.get("win_rate", 0)
        if wr <= 1.0:
            wr = wr * 100.0
        patterns_for_response.append({**p, "win_rate": round(wr, 2)})

    best = d.get("best_pattern")
    if best and best.get("win_rate", 0) <= 1.0:
        best = {**best, "win_rate": round(best["win_rate"] * 100.0, 2)}
    worst = d.get("worst_pattern")
    if worst and worst.get("win_rate", 0) <= 1.0:
        worst = {**worst, "win_rate": round(worst["win_rate"] * 100.0, 2)}

    # ── Step 2: Compute rule-based risk rating and recommendations ──

    recommendations = _build_meme_recommendations(
        total_trades=profile.total_trades,
        overall_win_rate=overall_wr,
        avg_pnl_per_trade=profile.avg_pnl_per_trade,
        total_realized_pnl=profile.total_realized_pnl,
        trader_type=profile.trader_type.value,
        patterns=patterns_for_response,
        best_pattern=best,
        worst_pattern=worst,
        fomo_tendency=d.get("fomo_tendency"),
        panic_sell_tendency=d.get("panic_sell_tendency"),
        dip_buy_tendency=d.get("dip_buy_tendency"),
        whale_follow_tendency=d.get("whale_follow_tendency"),
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

    trader_type_list = _derive_trader_types(profile.is_sniper, profile.is_flipper, profile.is_holder)

    # ── Step 3: Build the engine output dict for the LLM ──

    engine_output = {
        "trader_type": trader_type_list,
        "is_sniper": profile.is_sniper,
        "is_flipper": profile.is_flipper,
        "is_holder": profile.is_holder,
        "fomo_tendency": d.get("fomo_tendency"),
        "panic_sell_tendency": d.get("panic_sell_tendency"),
        "dip_buy_tendency": d.get("dip_buy_tendency"),
        "whale_follow_tendency": d.get("whale_follow_tendency"),
        "patterns": patterns_for_response,
        "best_pattern": best,
        "worst_pattern": worst,
        "risk_rating": risk_rating,
        "recommendations": recommendations,
        "profile_description": None,
    }

    # ── Step 4: Fetch raw trade details for LLM context ──

    profile_data = await asyncio.to_thread(
        data_fetcher.fetch_user_complete_profile,
        wallet_address,
        from_date=from_date,
        to_date=to_date,
    )

    trades = profile_data.get("trades", []) if profile_data else []
    summary = profile_data.get("summary", {}) if profile_data else {}

    total_trades = profile.total_trades
    is_valid = total_trades >= user_profile_config.MIN_TRADES_FOR_PROFILE

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

    raw_metrics = {
        "total_trades": total_trades,
        "total_mints_traded": profile.total_mints_traded,
        "overall_win_rate": round(overall_wr, 2),
        "avg_pnl_per_trade": round(profile.avg_pnl_per_trade, 6),
        "total_realized_pnl": round(profile.total_realized_pnl, 6),
        "lifetime_pnl": round(profile.lifetime_pnl, 2),
        "total_volume_usd": round(profile.total_volume_usd, 2),
        "max_drawdown_pct": round(profile.max_drawdown_pct, 4),
        "avg_r_multiple": round(profile.avg_r_multiple, 4),
        "avg_holding_time_minutes": round(profile.avg_holding_time_minutes, 2),
        "avg_hold_time_hours": round(profile.avg_hold_time_hours, 2),
        "typical_entry_phase": profile.typical_entry_phase,
    }
    if profile.win_rate_round_trip is not None:
        raw_metrics["win_rate_round_trip"] = profile.win_rate_round_trip
    if profile.total_closed_pnl is not None:
        raw_metrics["total_closed_pnl"] = round(profile.total_closed_pnl, 2)
    if profile.total_fees is not None:
        raw_metrics["total_fees"] = round(profile.total_fees, 4)
    if profile.avg_leverage is not None:
        raw_metrics["avg_leverage"] = round(profile.avg_leverage, 2)
    if profile.bot_detected is not None:
        raw_metrics["bot_detected"] = profile.bot_detected

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

    # ── Step 5: Send to LLM for hybrid refinement ──

    try:
        llm = LLMAnalyzerService()
        ai = await asyncio.to_thread(
            llm.analyze_hybrid, raw_data, engine_output, "meme"
        )

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
            trader_type=ai.get("trader_type", trader_type_list),
            is_sniper=ai.get("is_sniper", profile.is_sniper),
            is_holder=ai.get("is_holder", profile.is_holder),
            is_flipper=ai.get("is_flipper", profile.is_flipper),
            fomo_tendency=ai.get("fomo_tendency", d.get("fomo_tendency")),
            panic_sell_tendency=ai.get("panic_sell_tendency", d.get("panic_sell_tendency")),
            dip_buy_tendency=ai.get("dip_buy_tendency", d.get("dip_buy_tendency")),
            whale_follow_tendency=ai.get("whale_follow_tendency", d.get("whale_follow_tendency")),
            is_valid=profile.is_valid,
            patterns=ai.get("patterns", patterns_for_response),
            best_pattern=ai.get("best_pattern", best),
            worst_pattern=ai.get("worst_pattern", worst),
            profile_description=ai.get("profile_description"),
            risk_rating=ai.get("risk_rating", risk_rating),
            recommendations=ai.get("recommendations", recommendations),
            current_market_context=market_ctx_dict,
            behavioral_biases_status="hybrid_computed",
            behavioral_biases_message="Internal analysis refined by AI",
            observation_window=observation_window,
            token_type="meme",
        )

    except Exception as llm_err:
        logger.warning(f"Hybrid meme: LLM refinement failed, falling back to engine output: {llm_err}")

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
            trader_type=trader_type_list,
            is_sniper=profile.is_sniper,
            is_holder=profile.is_holder,
            is_flipper=profile.is_flipper,
            fomo_tendency=d.get("fomo_tendency"),
            panic_sell_tendency=d.get("panic_sell_tendency"),
            dip_buy_tendency=d.get("dip_buy_tendency"),
            whale_follow_tendency=d.get("whale_follow_tendency"),
            is_valid=profile.is_valid,
            patterns=patterns_for_response,
            best_pattern=best,
            worst_pattern=worst,
            profile_description=None,
            risk_rating=risk_rating,
            recommendations=recommendations,
            current_market_context=market_ctx_dict,
            behavioral_biases_status=d.get("behavioral_biases_status"),
            behavioral_biases_message=d.get("behavioral_biases_message"),
            observation_window=observation_window,
            token_type="meme",
        )


# =============================================================================
# Perps path
# =============================================================================

async def _build_hybrid_perps_profile(
    wallet_address: str,
    from_date: str,
    to_date: str,
    current_market_context,
    market_ctx_dict: Optional[Dict],
    observation_window: Dict[str, str],
) -> UserProfileResponse:
    """
    Run the internal perps engine, compute rule-based analysis, then send
    both raw data and engine output to LLM for refinement.
    Falls back to pure engine output if the LLM call fails.
    """
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

    logger.info(
        f"Hybrid: fetching perps profile for {wallet_address[:10]}... "
        f"from {perps_from} to {perps_to}"
    )

    try:
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
        trader_type = perps_profile.get("trader_type", "UNKNOWN")
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        is_valid = total_trades >= 10

        # ── Normalize positions and compute max leverage ──

        positions_list = []
        max_leverage = 0.0
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

            max_leverage = max(max_leverage, leverage_val)

            positions_list.append({
                "coin": pos.get("coin"),
                "size": pos.get("szi", "0"),
                "leverage": round(leverage_val, 2),
                "entry_price": round(entry_px, 4),
                "liquidation_price": round(liq_px, 4) if liq_px else None,
                "position_value": round(position_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 4),
            })

        # ── Compute rule-based risk rating ──

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

        # ── Compute rule-based recommendations ──

        recommendations = _perps_recommendations(
            perps_profile, positions_list, max_leverage, current_market_context,
            total_realized_pnl=total_pnl,
            trader_type=trader_type,
            observation_window=observation_window,
        )
        recommendations.extend(_market_context_recommendations(current_market_context))

        # Market overlay on risk rating
        if current_market_context and not getattr(current_market_context, "is_default", True):
            regime = getattr(current_market_context, "market_regime", None)
            is_extreme_fear = getattr(current_market_context, "is_extreme_fear", False)
            if regime == "BEAR" and is_extreme_fear and perps_profile.get("long_pct", 0) > 90:
                if risk_rating:
                    risk_rating["reason"] = (
                        risk_rating.get("reason", "") +
                        "; directional overexposure in fearful bear market."
                    )
                else:
                    risk_rating = {
                        "rating": "YELLOW",
                        "reason": "Long bias in BEAR regime with extreme fear — directional overexposure risk.",
                    }

        # ── Profile description (engine template) ──

        profit_factor = perps_profile.get("profit_factor", 0)
        direction_bias = perps_profile.get("direction_bias", "UNKNOWN")
        trades_per_day = perps_profile.get("trades_per_day", 0)
        unique_coins = perps_profile.get("unique_coins", 0)

        profitable_str = "Profitable" if total_pnl > 0 else "Unprofitable"
        direction_str = direction_bias.lower().replace("_", " ")
        if not direction_str.endswith("bias"):
            direction_str += " bias"
        profile_desc = (
            f"{profitable_str} {trader_type.lower().replace('_', ' ')} with "
            f"{direction_str}. "
            f"Trades {trades_per_day:.1f} times/day across {unique_coins} coin(s). "
            f"Profit factor: {profit_factor:.2f}, win rate: {win_rate:.1f}%."
        )

        # ── Behavioral tendencies (engine) ──

        fomo, panic, dip, whale, bias_status, bias_message = _behavioral_tendencies_for_response(
            total_trades, 0.0, 0.0, 0.0, 0.0
        )
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

        trader_type_list = [trader_type] if trader_type != "UNKNOWN" else ["UNKNOWN"]

        # ── Build engine output dict for LLM ──

        engine_output = {
            "trader_type": trader_type_list,
            "is_sniper": trader_type == "SCALPER",
            "is_holder": trader_type == "POSITION_TRADER",
            "is_flipper": trader_type == "DAY_TRADER",
            "fomo_tendency": fomo,
            "panic_sell_tendency": panic,
            "dip_buy_tendency": dip,
            "whale_follow_tendency": whale,
            "patterns": [],
            "best_pattern": None,
            "worst_pattern": None,
            "risk_rating": risk_rating,
            "recommendations": recommendations,
            "profile_description": profile_desc,
        }

        # ── Build raw data for LLM context ──

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

        # ── Send to LLM for hybrid refinement ──

        try:
            llm = LLMAnalyzerService()
            ai = await asyncio.to_thread(
                llm.analyze_hybrid, raw_data, engine_output, "perps"
            )

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
                trader_type=ai.get("trader_type", trader_type_list),
                is_sniper=ai.get("is_sniper", trader_type == "SCALPER"),
                is_holder=ai.get("is_holder", trader_type == "POSITION_TRADER"),
                is_flipper=ai.get("is_flipper", trader_type == "DAY_TRADER"),
                fomo_tendency=ai.get("fomo_tendency", fomo),
                panic_sell_tendency=ai.get("panic_sell_tendency", panic),
                dip_buy_tendency=ai.get("dip_buy_tendency", dip),
                whale_follow_tendency=ai.get("whale_follow_tendency", whale),
                is_valid=is_valid,
                patterns=ai.get("patterns", []),
                best_pattern=ai.get("best_pattern"),
                worst_pattern=ai.get("worst_pattern"),
                profile_description=ai.get("profile_description", profile_desc),
                risk_rating=ai.get("risk_rating", risk_rating),
                recommendations=ai.get("recommendations", recommendations),
                current_market_context=market_ctx_dict,
                behavioral_biases_status="hybrid_computed",
                behavioral_biases_message="Internal analysis refined by AI",
                observation_window=observation_window,
                token_type="perps",
                perps_profile=_sanitize_for_json(perps_profile),
                open_positions=positions_list if positions_list else None,
            )

        except Exception as llm_err:
            logger.warning(
                f"Hybrid perps: LLM refinement failed, falling back to engine output: {llm_err}"
            )

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
                trader_type=trader_type_list,
                is_sniper=trader_type == "SCALPER",
                is_holder=trader_type == "POSITION_TRADER",
                is_flipper=trader_type == "DAY_TRADER",
                fomo_tendency=fomo,
                panic_sell_tendency=panic,
                dip_buy_tendency=dip,
                whale_follow_tendency=whale,
                is_valid=is_valid,
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
        logger.warning(f"Hybrid: failed to process perps user profile: {fetch_error}")
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
        token_type="perps",
    )
