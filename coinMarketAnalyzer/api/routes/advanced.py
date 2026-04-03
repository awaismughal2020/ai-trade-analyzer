"""
Advanced API Routes
Endpoints for batch predictions, user context predictions, post-trade reviews, and data operations
Enhanced with rate limiting and throttled batch processing (v2.1)
"""

import os
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timedelta
from pathlib import Path

import sentry_sdk
from fastapi import APIRouter, HTTPException, Query

from core.circuit_breaker import sentry_fallback_warning
from pydantic import BaseModel, Field, model_validator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MINTS_DIR, CANDLES_DIR, HOLDERS_DIR, TRADES_DIR, api_config, post_trade_review_config, safety_config, ALGO_VERSION, PROMPT_VERSION
from api.validators import validate_token_address, validate_user_address, validate_date_field, validate_iso_datetime

# Feature 4 imports (Post-Trade Review)
try:
    from engines.post_trade_reviewer import (
        PostTradeReviewer, PostTradeAnalysis, TradeInput, UserContext,
        PricePoint, Mistake, MistakeType, Severity,
        parse_trade_input, parse_price_history
    )
    POST_TRADE_REVIEW_AVAILABLE = True
except ImportError as e:
    POST_TRADE_REVIEW_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Post-Trade Review not available: {e}")

# Layer 2 imports (User Profile - optional)
try:
    from engines.user_profiler import UserProfiler
    LAYER2_AVAILABLE = True
except ImportError:
    LAYER2_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Batch Processing Configuration
# =============================================================================

BATCH_SIZE = 10  # Process mints in batches of 10
BATCH_DELAY = 0.5  # 500ms delay between batches
MINT_DELAY = 0.1  # 100ms delay between mints within a batch

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    token_addresses: List[str] = Field(..., description="List of token addresses/tickers", max_length=50)
    token_type: Literal["meme", "perps"] = Field("meme", description="Token type: 'meme' or 'perps'")

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, values):
        if isinstance(values, dict):
            if "token_type" in values and isinstance(values["token_type"], str):
                values["token_type"] = values["token_type"].lower().strip()
        return values

    @model_validator(mode="after")
    def check_formats(self) -> "BatchPredictionRequest":
        self.token_addresses = [
            validate_token_address(addr, self.token_type)
            for addr in self.token_addresses
        ]
        return self


class PredictionWithUserRequest(BaseModel):
    """Prediction with user context request"""
    token_address: str = Field(..., min_length=2, max_length=44, description="Token address or ticker")
    token_type: Literal["meme", "perps"] = Field("meme", description="Token type: 'meme' or 'perps'")
    wallet_address: str = Field(..., min_length=32, max_length=44, description="User wallet address")
    user_from_date: Optional[str] = Field(None, description="User trade history start date")
    user_to_date: Optional[str] = Field(None, description="User trade history end date")

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, values):
        if isinstance(values, dict):
            if "token_type" in values and isinstance(values["token_type"], str):
                values["token_type"] = values["token_type"].lower().strip()
        return values

    @model_validator(mode="after")
    def check_formats(self) -> "PredictionWithUserRequest":
        self.token_address = validate_token_address(self.token_address, self.token_type)
        self.wallet_address = validate_user_address(self.wallet_address, self.token_type)
        if self.user_from_date:
            self.user_from_date = validate_date_field(self.user_from_date)
        if self.user_to_date:
            self.user_to_date = validate_date_field(self.user_to_date)
        return self


class PostTradeReviewRequest(BaseModel):
    """Request model for post-trade review analysis"""
    trade_id: str = Field(..., min_length=1, max_length=200, description="Unique trade identifier/signature")
    symbol: str = Field(..., min_length=1, max_length=20, description="Token symbol (e.g., 'Triad' for meme, 'BTC' for perps)")
    qty: float = Field(..., gt=0, description="Quantity of tokens traded")
    price: float = Field(..., gt=0, description="Execution price per token")
    executed_at: str = Field(..., description="Trade execution timestamp (ISO format)")
    realized_pnl: float = Field(0, description="Realized profit/loss from the trade")
    mint: str = Field(..., min_length=2, max_length=44, description="Token mint address (meme) or ticker (perps, e.g. 'BTC-USD')")
    is_buy: bool = Field(..., description="True if this was a BUY/LONG trade, False for SELL/SHORT")
    creation_timestamp: Optional[int] = Field(None, description="Token creation timestamp (Unix, meme only)")
    user_address: str = Field(..., min_length=32, max_length=44, description="User wallet address (Solana for meme, 0x for perps)")
    token_type: Literal["meme", "perps"] = Field("meme", description="Token type: 'meme' or 'perps'")

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, values):
        if isinstance(values, dict):
            if "token_type" in values and isinstance(values["token_type"], str):
                values["token_type"] = values["token_type"].lower().strip()
        return values

    @model_validator(mode="after")
    def check_formats(self) -> "PostTradeReviewRequest":
        self.mint = validate_token_address(self.mint, self.token_type)
        self.user_address = validate_user_address(self.user_address, self.token_type)
        self.executed_at = validate_iso_datetime(self.executed_at)
        return self

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Meme coin trade",
                    "value": {
                        "trade_id": "5TJn6QWDRY5VHAzqaMYiSQJQE2GuQnGb8bNZKtA7C1j9L4PMr2PepfeeCjQVYq9YNExSenpjW4rnTrxSBaLdSFtv",
                        "symbol": "Triad",
                        "qty": 1383136.587472,
                        "price": 0.0001314912,
                        "executed_at": "2025-11-01T19:46:48.000Z",
                        "realized_pnl": 0,
                        "mint": "DaWLFeW3mm7URTdqYuLNZQw7TyL7HSztBRH7fzggpump",
                        "is_buy": True,
                        "creation_timestamp": 1760743819,
                        "user_address": "F5ZCPAmRzG2Y8ReSMZL5nsHGpzPjTTe36iqgToLXGLAz",
                        "token_type": "meme"
                    }
                },
                {
                    "summary": "Perps trade (BTC-USD long)",
                    "value": {
                        "trade_id": "perps-btc-20260201-001",
                        "symbol": "BTC",
                        "qty": 0.1,
                        "price": 102500.0,
                        "executed_at": "2026-02-01T14:30:00.000Z",
                        "realized_pnl": 250.0,
                        "mint": "BTC-USD",
                        "is_buy": True,
                        "user_address": "0x1234567890abcdef1234567890abcdef12345678",
                        "token_type": "perps"
                    }
                }
            ]
        }


class MistakeResponse(BaseModel):
    """Response model for a single identified mistake"""
    type: str = Field(..., description="Type of mistake (EARLY_EXIT, LATE_ENTRY, FOMO_ENTRY, etc.)")
    severity: str = Field(..., description="Severity level (HIGH, MEDIUM, LOW)")
    impact_pct: float = Field(..., description="Estimated impact percentage")
    description: str = Field(..., description="Human-readable description of the mistake")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence for the mistake")


class OutcomeResponse(BaseModel):
    """Response model for outcome comparison"""
    price_at_trade: float = Field(..., description="Price at trade execution")
    price_after_1h: Optional[float] = Field(None, description="Price 1 hour after trade")
    price_after_4h: Optional[float] = Field(None, description="Price 4 hours after trade")
    price_after_24h: Optional[float] = Field(None, description="Price 24 hours after trade")
    max_price_after_exit: Optional[float] = Field(None, description="Maximum price after exit (for SELL)")
    min_price_after_exit: Optional[float] = Field(None, description="Minimum price after exit (for SELL)")
    optimal_exit_price: Optional[float] = Field(None, description="Optimal exit price in window")
    optimal_entry_price: Optional[float] = Field(None, description="Optimal entry price in window")
    time_to_max: Optional[str] = Field(None, description="Time until maximum price")
    missed_gain_pct: float = Field(0, description="Percentage of missed gains (SELL)")
    missed_dip_pct: float = Field(0, description="Percentage of missed dip (BUY)")


class PostTradeReviewResponse(BaseModel):
    """Response model for post-trade review analysis"""
    trade_id: str = Field(..., description="Trade identifier")
    trade_type: str = Field(..., description="BUY or SELL (meme) / LONG or SHORT (perps)")
    symbol: str = Field(..., description="Token symbol")
    mint: str = Field(..., description="Token mint address (meme) or ticker (perps)")
    token_type: str = Field("meme", description="Token type: 'meme' or 'perps'")
    
    # Outcome Analysis
    outcome: Dict[str, Any] = Field(..., description="Outcome comparison data")
    missed_opportunity_pct: float = Field(..., description="Total missed opportunity percentage")
    
    # Mistakes
    mistakes: List[Dict[str, Any]] = Field(default_factory=list, description="List of identified mistakes")
    primary_mistake: Optional[str] = Field(None, description="Primary/most impactful mistake type")
    
    # Impact Estimation
    monthly_impact_estimate_pct: float = Field(0, description="Estimated monthly impact percentage")
    monthly_impact_estimate_usd: float = Field(0, description="Estimated monthly impact in USD")
    
    # AI Coaching
    coaching_message: str = Field("", description="AI-generated coaching feedback")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    
    # Metadata
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence score")
    analysis_timestamp: str = Field(..., description="When the analysis was performed")
    
    # Data quality indicators
    price_data_available: bool = Field(True, description="Whether price history was available")
    user_context_available: bool = Field(False, description="Whether user trading history was successfully loaded")

    # Versioning
    algo_version: str = Field(default=ALGO_VERSION, description="Algorithm version")
    prompt_version: str = Field(default=PROMPT_VERSION, description="Prompt version")

    # Execution time
    execution_time_seconds: float = Field(0.0, description="Request execution time in seconds")


# =============================================================================
# Advanced Prediction Endpoints
# =============================================================================

@router.post("/predict", tags=["Prediction"], summary="Predict (POST)")
async def predict_post(request: Dict[str, Any]):
    """
    Generate trading signal prediction (POST without trailing slash)
    
    This is an alias for POST /predict/ for compatibility.
    Redirects to the main prediction endpoint.
    """
    try:
        start = time.perf_counter()
        from .predict import run_prediction
        from config import meme_config
        
        raw_token_type = request.get('token_type', 'meme')
        token_type = raw_token_type.lower().strip() if isinstance(raw_token_type, str) else 'meme'
        result = await run_prediction(
            token_address=request.get('token_address'),
            token_type=token_type,
            candle_days=meme_config.DEFAULT_CANDLE_DAYS,
            holder_limit=meme_config.DEFAULT_HOLDER_LIMIT,
            user_address=request.get('user_address'),
            user_from_date=request.get('user_from_date'),
            user_to_date=request.get('user_to_date')
        )
        result["execution_time_seconds"] = round(time.perf_counter() - start, 4)
        return result
        
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("prediction_request", {
                "token_address": request.get("token_address"),
                "token_type": request.get("token_type"),
                "user_address": request.get("user_address"),
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.post("/predict/batch", tags=["Prediction"], summary="Batch Predictions")
async def predict_batch(request: BatchPredictionRequest):
    """
    Generate predictions for multiple tokens
    
    Processes up to 50 tokens in a single request.
    Returns array of predictions with same structure as single prediction.
    """
    try:
        if len(request.token_addresses) == 0:
            raise HTTPException(status_code=400, detail="No tokens provided")
        
        if len(request.token_addresses) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 tokens allowed per request")
        
        from .predict import run_prediction
        from config import meme_config
        
        predictions = []
        errors = []
        
        for token_address in request.token_addresses:
            try:
                result = await run_prediction(
                    token_address=token_address,
                    token_type=request.token_type,
                    candle_days=meme_config.DEFAULT_CANDLE_DAYS,
                    holder_limit=meme_config.DEFAULT_HOLDER_LIMIT
                )
                predictions.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict {token_address}: {e}")
                errors.append({
                    "token_address": token_address,
                    "error": str(e)
                })
        
        return {
            "total_requested": len(request.token_addresses),
            "successful": len(predictions),
            "failed": len(errors),
            "predictions": predictions,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("batch_request", {
                "token_type": request.token_type,
                "token_count": len(request.token_addresses),
                "tokens": request.token_addresses[:10],
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


async def check_user_holds_token(
    token_address: str,
    wallet_address: str,
    token_type: str
) -> Dict[str, Any]:
    """
    Verify the user currently holds a token (meme) or has an open position (perps).

    Returns:
        Dict with keys:
          - holds (bool): True if user has the token
          - verified (bool): True if the check completed without errors
          - detail (str|None): Human-readable context on failure or skip
    """
    is_perps = token_type.lower() == "perps"

    if is_perps:
        # --- Perps: check open positions via HyperLiquid clearinghouse ---
        try:
            from core.data_fetcher import PerpsDataFetcher

            fetcher = PerpsDataFetcher()
            positions = await asyncio.to_thread(
                fetcher.fetch_user_open_positions, wallet_address
            )

            # Normalize the requested ticker: "BTC-USD" / "btc" -> "BTC"
            coin = token_address.split('-')[0].upper() if '-' in token_address else token_address.upper()

            for pos in positions:
                pos_data = pos.get('position', {})
                if pos_data.get('coin', '').upper() == coin:
                    szi = float(pos_data.get('szi', 0))
                    if szi != 0:
                        leverage_obj = pos_data.get('leverage', {})
                        leverage_val = leverage_obj.get('value', 0) if isinstance(leverage_obj, dict) else 0
                        try:
                            leverage_val = float(leverage_val)
                        except (ValueError, TypeError):
                            leverage_val = 0
                        position_risk = {
                            "leverage": leverage_val,
                            "entry_price": float(pos_data.get('entryPx', 0)),
                            "liquidation_price": float(pos_data.get('liquidationPx', 0)),
                            "position_value": float(pos_data.get('positionValue', 0)),
                            "unrealized_pnl": float(pos_data.get('unrealizedPnl', 0)),
                            "size": float(szi),
                        }
                        return {
                            "holds": True,
                            "verified": True,
                            "detail": None,
                            "position_risk": position_risk,
                        }

            return {
                "holds": False,
                "verified": True,
                "detail": f"User {wallet_address} has no open position for {coin}.",
            }

        except Exception as e:
            logger.warning(f"Perps ownership check failed (fail-open): {e}")
            return {
                "holds": True,
                "verified": False,
                "detail": f"Could not verify open positions: {e}",
            }

    else:
        # --- Meme: check on-chain balance via mint-timeline ---
        try:
            from core.data_fetcher import DataFetcher

            fetcher = DataFetcher(limiter_name="ownership_check")
            holders_df = await asyncio.to_thread(
                fetcher.fetch_user_holdings, token_address
            )

            if holders_df.empty:
                return {
                    "holds": False,
                    "verified": True,
                    "detail": f"No holder data found for token {token_address}.",
                }

            wallet_lower = wallet_address.lower()
            match = holders_df[
                holders_df['wallet'].str.lower() == wallet_lower
            ]

            if not match.empty and float(match.iloc[0].get('finalHolding', 0)) > 0:
                return {"holds": True, "verified": True, "detail": None}

            return {
                "holds": False,
                "verified": True,
                "detail": f"User {wallet_address} does not currently hold token {token_address}.",
            }

        except Exception as e:
            logger.warning(f"Meme ownership check failed (fail-open): {e}")
            return {
                "holds": True,
                "verified": False,
                "detail": f"Could not verify token holdings: {e}",
            }


@router.post("/predict/with-user", tags=["Prediction"], summary="Predict With User Context")
async def predict_with_user_context(request: PredictionWithUserRequest):
    """
    Generate trading signal with personalized user context.

    The signal is personalized using the user's past trade patterns (similar
    setups and optional same-token history).  Current token ownership is
    reported for informational purposes but does **not** gate the analysis —
    the full prediction pipeline always runs regardless of whether the user
    currently holds the requested token.
    """
    start = time.perf_counter()

    # --- Full prediction pipeline (always runs) ---
    try:
        from .predict import run_prediction
        from config import meme_config

        # Run prediction and ownership check concurrently
        prediction_task = run_prediction(
            token_address=request.token_address,
            token_type=request.token_type,
            candle_days=meme_config.DEFAULT_CANDLE_DAYS,
            holder_limit=meme_config.DEFAULT_HOLDER_LIMIT,
            user_address=request.wallet_address,
            user_from_date=request.user_from_date,
            user_to_date=request.user_to_date,
        )

        ownership_task = check_user_holds_token(
            token_address=request.token_address,
            wallet_address=request.wallet_address,
            token_type=request.token_type,
        )

        result, ownership = await asyncio.gather(
            prediction_task,
            ownership_task,
            return_exceptions=False,
        )

        result["user_context"] = {
            "wallet_address": request.wallet_address,
            "analysis_period": {
                "from": request.user_from_date,
                "to": request.user_to_date,
            },
            "personalized": True,
        }

        # Informational ownership status (never blocks the response)
        result["current_hold"] = {
            "holds": ownership["holds"],
            "verified": ownership["verified"],
            "detail": ownership.get("detail"),
        }

        # Attach position risk data for perps users (leverage, liquidation distance)
        pos_risk = ownership.get("position_risk")
        if pos_risk:
            current_price = 0
            sig_eff = result.get("signal_effectiveness") or {}
            r24 = sig_eff.get("recent_24h_analysis") or {}
            current_price = r24.get("price_current", 0)

            liq_price = pos_risk.get("liquidation_price", 0)
            leverage = pos_risk.get("leverage", 0)
            risk_warnings: List[str] = []

            liq_distance_pct = 0.0
            if current_price > 0 and liq_price > 0:
                liq_distance_pct = abs(current_price - liq_price) / current_price * 100

            if liq_distance_pct > 0 and liq_distance_pct < safety_config.LIQUIDATION_CRITICAL_DISTANCE_PCT:
                risk_warnings.append(
                    f"CRITICAL: Position within {liq_distance_pct:.1f}% of liquidation"
                )
            if leverage > safety_config.HIGH_LEVERAGE_THRESHOLD:
                risk_warnings.append(
                    f"HIGH LEVERAGE: {leverage}x — elevated liquidation risk"
                )

            result["position_risk"] = {
                **pos_risk,
                "liquidation_distance_pct": round(liq_distance_pct, 2),
                "warnings": risk_warnings if risk_warnings else None,
            }

        result["execution_time_seconds"] = round(time.perf_counter() - start, 4)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("prediction_request", {
                "token_address": request.token_address,
                "token_type": request.token_type,
                "wallet_address": request.wallet_address,
                "user_from_date": getattr(request, "user_from_date", None),
                "user_to_date": getattr(request, "user_to_date", None),
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"User context prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"User context prediction failed: {e}")


# =============================================================================
# Post-Trade Review Endpoint
# =============================================================================

@router.post("/post-trade-review", response_model=PostTradeReviewResponse, tags=["Review"], summary="Post-Trade Review")
async def post_trade_review(request: PostTradeReviewRequest):
    """
    Analyze a completed trade and provide AI-powered coaching feedback.
    
    This endpoint analyzes what happened after your trade to identify mistakes
    and provide actionable coaching to improve your trading.
    
    **Mistake Types Detected:**
    - `EARLY_EXIT`: Sold too early, price continued higher
    - `LATE_ENTRY`: Bought at higher price than available in window
    - `FOMO_ENTRY`: Bought after significant pump (>20% in 4h)
    - `PANIC_SELL`: Sold at loss with short hold time
    - `REVENGE_TRADE`: Trade made shortly after a loss
    - `OVER_TRADING`: Multiple trades on same token in short window
    - `BAD_RISK_REWARD`: Poor risk-reward ratio
    
    **Output Example:**
    ```
    "You closed too early. After your exit, price ran 12% more. 
    This behavior costs you ~14% monthly."
    ```
    
    **Supports both meme coins (Solana/Birdeye) and perps (HyperLiquid/HyperEVM).**
    Set `token_type` to `"meme"` or `"perps"` accordingly.
    
    **Example Meme Request:**
    ```json
    {
        "trade_id": "5TJn6QWDRY5VHAzqaMYiSQJQE2GuQnGb8bNZKtA7C1j9L4PMr2PepfeeCjQVYq9YNExSenpjW4rnTrxSBaLdSFtv",
        "symbol": "Triad",
        "qty": 1383136.587472,
        "price": 0.0001314912,
        "executed_at": "2025-11-01T19:46:48.000Z",
        "realized_pnl": 0,
        "mint": "DaWLFeW3mm7URTdqYuLNZQw7TyL7HSztBRH7fzggpump",
        "is_buy": True,
        "creation_timestamp": 1760743819,
        "user_address": "F5ZCPAmRzG2Y8ReSMZL5nsHGpzPjTTe36iqgToLXGLAz",
        "token_type": "meme"
    }
    ```
    
    **Example Perps Request:**
    ```json
    {
        "trade_id": "perps-btc-20260201-001",
        "symbol": "BTC",
        "qty": 0.1,
        "price": 102500.0,
        "executed_at": "2026-02-01T14:30:00.000Z",
        "realized_pnl": 250.0,
        "mint": "BTC-USD",
        "is_buy": True,
        "user_address": "0x1234567890abcdef1234567890abcdef12345678",
        "token_type": "perps"
    }
    ```
    """
    if not POST_TRADE_REVIEW_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Post-Trade Review feature not available. Check server logs for details."
        )
    start = time.perf_counter()
    is_perps = request.token_type.lower() == "perps"
    trade_direction = "LONG" if request.is_buy else "SHORT" if is_perps else ("BUY" if request.is_buy else "SELL")
    
    logger.info(f"Post-trade review request for trade {request.trade_id[:16]}... "
               f"({request.symbol}, {trade_direction}, type={request.token_type})")
    
    try:
        from .predict import get_components
        components = get_components()
        
        # Parse trade input
        trade_data = {
            'trade_id': request.trade_id,
            'symbol': request.symbol,
            'mint': request.mint,
            'qty': request.qty,
            'price': request.price,
            'executed_at': request.executed_at,
            'realized_pnl': request.realized_pnl,
            'is_buy': request.is_buy,
            'creation_timestamp': request.creation_timestamp,
            'user_address': request.user_address
        }
        
        trade = parse_trade_input(trade_data)
        
        # Fetch historical market context at trade execution time
        trade_market_context = None
        try:
            market_ctx_fetcher = components.get("market_context_fetcher")
            if market_ctx_fetcher:
                trade_market_context = await asyncio.to_thread(
                    market_ctx_fetcher.fetch_at,
                    time=request.executed_at
                )
        except Exception as e:
            logger.warning(f"Failed to fetch historical market context for post-trade review: {e}")
        
        # Initialize post-trade reviewer if not already done
        if "post_trade_reviewer" not in components:
            components["post_trade_reviewer"] = PostTradeReviewer(
                birdeye_fetcher=components.get("birdeye_fetcher"),
                data_fetcher=components.get("data_fetcher"),
                user_profiler=components["signal_generator"].user_profiler if hasattr(components["signal_generator"], "user_profiler") else None
            )
        
        reviewer = components["post_trade_reviewer"]
        
        # =====================================================================
        # Fetch price history (branching on token_type)
        # =====================================================================
        price_history_before = []
        price_history_after = []
        price_data_available = False
        
        if is_perps:
            # --- PERPS: Use PerpsDataFetcher (internal API /candles endpoint) ---
            perps_fetcher = components.get("perps_fetcher")
            if perps_fetcher:
                try:
                    # Resolve ticker (e.g., "BTC" -> "BTC-USD", or use mint directly if it's already "BTC-USD")
                    ticker = request.mint if '-' in request.mint else perps_fetcher.get_ticker_for_token(request.mint)
                    if not ticker:
                        ticker = f"{request.symbol}-USD"
                    
                    ohlcv_before, ohlcv_after = await asyncio.to_thread(
                        perps_fetcher.fetch_price_history_for_trade_review,
                        ticker=ticker,
                        trade_time=trade.executed_at,
                        hours_before=post_trade_review_config.FOMO_LOOKBACK_HOURS,
                        hours_after=post_trade_review_config.EARLY_EXIT_WINDOW_HOURS,
                        resolution="15MINS"
                    )
                    
                    # Convert to PricePoint objects (same format as Birdeye)
                    price_history_before = parse_price_history(ohlcv_before)
                    price_history_after = parse_price_history(ohlcv_after)
                    
                    price_data_available = len(price_history_before) > 0 or len(price_history_after) > 0
                    logger.info(f"Perps price history: {len(price_history_before)} before, {len(price_history_after)} after")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch perps price history: {e}")
            else:
                logger.info("PerpsDataFetcher not available, proceeding without price history")
        else:
            # --- MEME: Use BirdeyeFetcher (Birdeye API) ---
            birdeye_fetcher = components.get("birdeye_fetcher")
            if birdeye_fetcher and birdeye_fetcher.is_configured():
                try:
                    ohlcv_before, ohlcv_after = await asyncio.to_thread(
                        birdeye_fetcher.fetch_price_history_for_trade_review,
                        token_address=request.mint,
                        trade_time=trade.executed_at,
                        hours_before=post_trade_review_config.FOMO_LOOKBACK_HOURS,
                        hours_after=post_trade_review_config.EARLY_EXIT_WINDOW_HOURS,
                        interval=post_trade_review_config.OHLCV_INTERVAL
                    )
                    
                    price_history_before = parse_price_history(ohlcv_before)
                    price_history_after = parse_price_history(ohlcv_after)
                    
                    price_data_available = len(price_history_before) > 0 or len(price_history_after) > 0
                    logger.info(f"Meme price history: {len(price_history_before)} before, {len(price_history_after)} after")
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch price history from Birdeye: {e}")
                    sentry_fallback_warning("birdeye", "Failed to fetch price history from Birdeye for trade review", extra={"mint": request.mint[:16], "error": str(e)})
            else:
                logger.info("Birdeye not configured, proceeding without price history")
                sentry_fallback_warning("birdeye", "Birdeye not configured — post-trade review without price history")
        
        # =====================================================================
        # Build user context (branching on token_type)
        # =====================================================================
        user_context = None
        user_context_available = False
        
        if request.user_address:
            try:
                if is_perps:
                    # --- PERPS: Fetch user's perps trades via PerpsDataFetcher ---
                    perps_fetcher = components.get("perps_fetcher")
                    if perps_fetcher:
                        trades_df = await asyncio.to_thread(
                            perps_fetcher.fetch_user_perps_trades,
                            user_address=request.user_address,
                            from_date=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                            to_date=datetime.now().strftime('%Y-%m-%d')
                        )
                        
                        if not trades_df.empty:
                            trades_list = trades_df.to_dict('records')
                            
                            # Analyze trades to build user context
                            total_trades = len(trades_list)
                            wins = sum(1 for t in trades_list if t.get('closedPnl', t.get('realized_pnl', 0)) > 0)
                            win_rate = wins / total_trades if total_trades > 0 else 0
                            
                            # Calculate average PnL
                            pnls = [t.get('closedPnl', t.get('realized_pnl', 0)) for t in trades_list]
                            avg_pnl = sum(pnls) / len(pnls) if pnls else 0
                            
                            # Find recent losses
                            recent_losses = []
                            trades_on_same_ticker = []
                            for t in trades_list:
                                pnl = t.get('closedPnl', t.get('realized_pnl', 0))
                                if pnl < 0:
                                    recent_losses.append({
                                        'executed_at': t.get('time', t.get('executed_at')),
                                        'pnl_pct': pnl / (t.get('px', t.get('price', 1)) * t.get('sz', t.get('qty', 1))) if t.get('px') or t.get('price') else 0,
                                        'symbol': t.get('coin', t.get('symbol', 'unknown'))
                                    })
                                # Match trades on same ticker/coin
                                trade_coin = t.get('coin', t.get('symbol', ''))
                                if trade_coin and (trade_coin == request.symbol or trade_coin in request.mint):
                                    trades_on_same_ticker.append(t)
                            
                            user_context = UserContext(
                                total_trades=total_trades,
                                win_rate=win_rate,
                                avg_hold_time_hours=0,  # Not easily derived from perps trade data
                                avg_pnl_per_trade=avg_pnl,
                                fomo_tendency=None,
                                panic_sell_tendency=None,
                                trader_type='PERPS_TRADER',
                                recent_losses=recent_losses[-10:],
                                trades_on_same_mint=trades_on_same_ticker
                            )
                            user_context_available = True
                            logger.info(f"Perps user context loaded: {total_trades} trades, {win_rate:.1%} win rate")
                
                elif LAYER2_AVAILABLE:
                    # --- MEME: Fetch user profile via DataFetcher ---
                    data_fetcher = components.get("data_fetcher")
                    if data_fetcher:
                        user_profile_data = await asyncio.to_thread(
                            data_fetcher.fetch_user_complete_profile,
                            wallet_address=request.user_address
                        )
                        
                        if user_profile_data and user_profile_data.get('summary'):
                            summary = user_profile_data['summary']
                            trades = user_profile_data.get('trades', [])
                            
                            # Find recent losses (for revenge trade detection)
                            recent_losses = []
                            trades_on_same_mint = []
                            for t in trades:
                                pnl = t.get('realized_pnl', 0)
                                if pnl < 0:
                                    recent_losses.append({
                                        'executed_at': t.get('executed_at'),
                                        'pnl_pct': pnl / (t.get('price', 1) * t.get('qty', 1)) if t.get('price') and t.get('qty') else 0,
                                        'symbol': t.get('symbol', 'unknown')
                                    })
                                if t.get('mint') == request.mint:
                                    trades_on_same_mint.append(t)
                            
                            _hold_min = summary.get('avg_holding_time_minutes', 0)
                            _api_wr = summary.get('win_rate', 0)
                            _wr = _api_wr / 100.0 if _api_wr > 1 else _api_wr
                            user_context = UserContext(
                                total_trades=len(trades),
                                win_rate=_wr,
                                avg_hold_time_hours=_hold_min / 60 if _hold_min else 0,
                                avg_pnl_per_trade=summary.get('avg_pnl_per_trade', 0),
                                fomo_tendency=summary.get('fomo_tendency'),
                                panic_sell_tendency=summary.get('panic_sell_tendency'),
                                trader_type=summary.get('trader_type', 'UNKNOWN'),
                                recent_losses=recent_losses[-10:],
                                trades_on_same_mint=trades_on_same_mint
                            )
                            user_context_available = True
                            logger.info(f"Meme user context loaded: {user_context.total_trades} trades, {user_context.win_rate:.1%} win rate")
                        
            except Exception as e:
                logger.warning(f"Failed to fetch user context: {e}")
        
        # Run the analysis
        analysis = reviewer.analyze_trade(
            trade=trade,
            price_history_before=price_history_before,
            price_history_after=price_history_after,
            user_context=user_context
        )
        
        # Generate AI coaching message
        openai_service = components.get("openai_service")
        if openai_service and post_trade_review_config.COACHING_ENABLED:
            try:
                # Prepare analysis data for OpenAI
                analysis_dict = analysis.to_dict()
                
                # Enrich with market conditions at trade time
                if trade_market_context and not getattr(trade_market_context, 'is_default', True):
                    analysis_dict['market_conditions_at_trade'] = {
                        'fear_greed': trade_market_context.fear_greed,
                        'market_regime': trade_market_context.market_regime,
                        'btc_dominance': trade_market_context.btc_dominance,
                        'total_market_cap_change': trade_market_context.total_market_cap_change,
                    }
                
                # Add user context if available
                user_context_dict = None
                if user_context:
                    user_context_dict = {
                        'total_trades': user_context.total_trades,
                        'win_rate': user_context.win_rate,
                        'avg_hold_time_hours': user_context.avg_hold_time_hours,
                        'avg_pnl_per_trade': user_context.avg_pnl_per_trade,
                        'fomo_tendency': user_context.fomo_tendency,
                        'panic_sell_tendency': user_context.panic_sell_tendency,
                        'trader_type': user_context.trader_type
                    }
                
                coaching_message, recommendations = openai_service.generate_post_trade_review(
                    analysis_data=analysis_dict,
                    user_context=user_context_dict
                )
                
                analysis.coaching_message = coaching_message
                analysis.recommendations = recommendations
                logger.info(f"Coaching generated: {len(coaching_message)} chars, {len(recommendations)} recommendations")
                
            except Exception as e:
                logger.warning(f"Failed to generate AI coaching: {e}")
        
        # Ensure we always have coaching if mistakes were found (fallback)
        if analysis.mistakes and not analysis.coaching_message:
            logger.info("Generating fallback coaching for identified mistakes")
            primary = analysis.mistakes[0]
            mistake_type_str = primary.type.value if hasattr(primary.type, "value") else str(primary.type)
            sentry_fallback_warning(
                "openai_coaching",
                "AI coaching failed — using fallback coaching from first mistake",
                {"mistake_type": mistake_type_str},
            )
            analysis.coaching_message = f"Issue identified: {primary.description}. This pattern may be impacting your performance."
            
            # Add basic recommendations based on mistake type
            if primary.type == MistakeType.EARLY_EXIT:
                analysis.recommendations = [
                    "Consider using trailing stop-losses instead of manual exits.",
                    "Wait for momentum indicators to confirm exit timing.",
                    "Review your exit strategy for similar setups."
                ]
            elif primary.type == MistakeType.LATE_ENTRY:
                analysis.recommendations = [
                    "Set limit orders at support levels instead of market orders.",
                    "Wait for pullbacks after pumps before entering.",
                    "Use RSI oversold signals to time entries."
                ]
            elif primary.type == MistakeType.FOMO_ENTRY:
                fomo_pct = int(post_trade_review_config.FOMO_PUMP_THRESHOLD * 100)
                analysis.recommendations = [
                    "Avoid chasing green candles - wait for consolidation.",
                    "Set price alerts at your target entry instead of reacting to pumps.",
                    f"If a token has pumped >{fomo_pct}%, wait for a pullback before entering."
                ]
            elif primary.type == MistakeType.PANIC_SELL:
                analysis.recommendations = [
                    "Set stop-losses in advance instead of manual panic exits.",
                    "Define your maximum acceptable loss before entering a trade.",
                    "Take a break after a loss before making your next trade."
                ]
            elif primary.type == MistakeType.REVENGE_TRADE:
                revenge_hours = post_trade_review_config.REVENGE_TRADE_WINDOW_HOURS
                analysis.recommendations = [
                    f"Wait at least {revenge_hours} hours after a significant loss before trading again.",
                    "Review your trading journal before making emotionally-driven trades.",
                    "Set a daily loss limit and stop trading when reached."
                ]
            elif primary.type == MistakeType.OVER_TRADING:
                overtrade_min = post_trade_review_config.OVERTRADE_MIN_TRADES
                analysis.recommendations = [
                    f"Limit yourself to fewer than {overtrade_min} trades per token per day.",
                    "Each trade should have a clear thesis - avoid impulsive entries.",
                    "Transaction fees add up - factor them into your expected returns."
                ]
            else:
                analysis.recommendations = [
                    "Review this trade type in your history.",
                    "Look for similar setups to identify patterns.",
                    "Consider adjusting your entry/exit criteria."
                ]
        
        # Build response
        response_dict = analysis.to_dict()
        response_dict['price_data_available'] = price_data_available
        response_dict['user_context_available'] = user_context_available
        response_dict['token_type'] = request.token_type
        if trade_market_context and not getattr(trade_market_context, 'is_default', True):
            response_dict['market_context_at_trade'] = trade_market_context.to_dict()
        response_dict['execution_time_seconds'] = round(time.perf_counter() - start, 4)
        logger.info(f"Post-trade review complete: {len(analysis.mistakes)} mistakes, "
                   f"primary: {analysis.primary_mistake.value if analysis.primary_mistake else 'None'}, "
                   f"type={request.token_type}")
        return PostTradeReviewResponse(**response_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("post_trade_request", {
                "trade_type": request.trade_type,
                "symbol": request.symbol,
                "mint": request.mint,
                "token_type": request.token_type,
                "wallet_address": getattr(request, "wallet_address", None),
                "entry_price": request.entry_price,
                "exit_price": request.exit_price,
                "entry_time": request.entry_time,
                "exit_time": request.exit_time,
            })
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Post-trade review failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Post-trade review failed: {str(e)}")


# =============================================================================
# Data Operations Endpoint
# =============================================================================

def _get_data_fetcher():
    """Get the data fetcher component"""
    from core.data_fetcher import DataFetcher
    return DataFetcher(limiter_name="advanced")


async def _process_mint_batch(
    data_fetcher,
    mints: List[str],
    candle_days: int,
    holder_days: int,
    trade_days: int,
    trade_limit: int,
    timestamp: str,
    batch_num: int,
    total_batches: int
) -> tuple:
    """
    Process a batch of mints with throttling.
    Returns (processed_mints, errors) tuple.
    """
    processed_mints = []
    errors = []
    
    for idx, mint in enumerate(mints):
        try:
            logger.info(f"Batch {batch_num}/{total_batches} - Processing mint {idx + 1}/{len(mints)}: {mint[:8]}...")
            
            mint_result = {
                "mint": mint,
                "candles": None,
                "holders": None,
                "trades": None,
                "status": "success"
            }
            
            # Fetch and save candles
            try:
                candles_df = await asyncio.to_thread(data_fetcher.fetch_candles, mint, days=candle_days, limit=10000)
                if len(candles_df) > 0:
                    candles_filename = f"candles_{mint[:8]}_{timestamp}.csv"
                    candles_filepath = os.path.join(str(CANDLES_DIR), candles_filename)
                    candles_df.to_csv(candles_filepath, index=False)
                    mint_result["candles"] = {
                        "count": len(candles_df),
                        "filename": candles_filename,
                        "filepath": candles_filepath
                    }
                    logger.info(f"  ✓ Saved {len(candles_df)} candles")
                else:
                    logger.warning(f"  ✗ No candles found for {mint[:8]}...")
                    mint_result["candles"] = {"count": 0, "error": "No data returned"}
            except Exception as e:
                logger.warning(f"  ✗ Failed to fetch candles for {mint[:8]}...: {e}")
                mint_result["candles"] = {"error": str(e)}
            
            # Small delay between API calls within a mint
            await asyncio.sleep(MINT_DELAY)
            
            # Fetch and save holders
            try:
                holders_df = await asyncio.to_thread(data_fetcher.fetch_holders, mint, days=holder_days)
                if len(holders_df) > 0:
                    holders_filename = f"holders_{mint[:8]}_{timestamp}.csv"
                    holders_filepath = os.path.join(str(HOLDERS_DIR), holders_filename)
                    holders_df.to_csv(holders_filepath, index=False)
                    mint_result["holders"] = {
                        "count": len(holders_df),
                        "filename": holders_filename,
                        "filepath": holders_filepath
                    }
                    logger.info(f"  ✓ Saved {len(holders_df)} holders")
                else:
                    logger.warning(f"  ✗ No holders found for {mint[:8]}...")
                    mint_result["holders"] = {"count": 0, "error": "No data returned"}
            except Exception as e:
                logger.warning(f"  ✗ Failed to fetch holders for {mint[:8]}...: {e}")
                mint_result["holders"] = {"error": str(e)}
            
            # Small delay between API calls within a mint
            await asyncio.sleep(MINT_DELAY)
            
            # Fetch and save trades
            try:
                trades_df = await asyncio.to_thread(data_fetcher.fetch_trades, mint, days=trade_days, limit=trade_limit)
                if len(trades_df) > 0:
                    trades_filename = f"trades_{mint[:8]}_{timestamp}.csv"
                    trades_filepath = os.path.join(str(TRADES_DIR), trades_filename)
                    trades_df.to_csv(trades_filepath, index=False)
                    mint_result["trades"] = {
                        "count": len(trades_df),
                        "filename": trades_filename,
                        "filepath": trades_filepath
                    }
                    logger.info(f"  ✓ Saved {len(trades_df)} trades")
                else:
                    logger.warning(f"  ✗ No trades found for {mint[:8]}...")
                    mint_result["trades"] = {"count": 0, "error": "No data returned"}
            except Exception as e:
                logger.warning(f"  ✗ Failed to fetch trades for {mint[:8]}...: {e}")
                mint_result["trades"] = {"error": str(e)}
            
            processed_mints.append(mint_result)
            
            # Small delay between mints
            await asyncio.sleep(MINT_DELAY)
            
        except Exception as e:
            error_msg = f"Error processing mint {mint[:8]}...: {str(e)}"
            logger.error(error_msg)
            errors.append({"mint": mint, "error": str(e)})
    
    return processed_mints, errors


@router.post("/data/fetch-all", tags=["Data"], summary="Fetch All Data")
async def fetch_all_data(
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD). Default: 3 months ago."),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD). Default: today."),
    candle_days: int = Query(90, ge=1, le=365, description="Days of historical candle data"),
    holder_days: int = Query(90, ge=1, le=365, description="Days of historical holder data (minimum 90 for API to return data)"),
    trade_days: int = Query(90, ge=1, le=365, description="Days of historical trade data"),
    trade_limit: int = Query(1000, ge=1, le=10000, description="Maximum trade records per mint"),
    max_mints: int = Query(500, ge=1, le=10000, description="Maximum number of mints to process"),
    max_pages: int = Query(10, ge=1, le=100, description="Maximum pages to fetch for mints (1000 mints per page)"),
    batch_size: int = Query(10, ge=1, le=50, description="Number of mints to process per batch (default 10)")
):
    """
    Fetch all data for model training: mints, then automatically fetch candles, holders, and trades for each mint.

    When from_date or to_date are omitted, the date range defaults to the last 3 months (e.g. 15 Mar -> 15 Dec to 15 Mar).
    
    This endpoint:
    1. Fetches all mints in the date range (with pagination support)
    2. For each mint, automatically fetches and saves:
       - Candlestick data (OHLCV)
       - Holder snapshots
       - Trade history
    
    All data is saved to CSV files in the respective data/ subdirectories.
    
    **Rate Limiting:**
    - Processes mints in batches to prevent overwhelming the API
    - Uses exponential backoff on failures
    - Circuit breaker protection against cascading failures
    
    **Pagination Notes:**
    - The API returns up to 1000 mints per page
    - Use max_pages to control how many pages to fetch (max_pages=10 = up to 10,000 mints)
    - Use max_mints to limit total mints processed (useful for testing)
    - Use batch_size to control how many mints are processed at once
    
    **Use this endpoint to collect training data before running model training.**
    """
    # Default date range: last 3 months when not provided (e.g. 15 Mar -> 15 Dec to 15 Mar)
    if not from_date or not str(from_date).strip() or not to_date or not str(to_date).strip():
        to_dt = datetime.utcnow()
        from_dt = to_dt - timedelta(days=90)
        from_date = from_date if (from_date and str(from_date).strip()) else from_dt.strftime('%Y-%m-%d')
        to_date = to_date if (to_date and str(to_date).strip()) else to_dt.strftime('%Y-%m-%d')
    from_date = str(from_date)[:10]
    to_date = str(to_date)[:10]
    if not (len(from_date) == 10 and from_date.replace('-', '').isdigit()):
        raise HTTPException(status_code=400, detail="from_date must be YYYY-MM-DD when provided")
    if not (len(to_date) == 10 and to_date.replace('-', '').isdigit()):
        raise HTTPException(status_code=400, detail="to_date must be YYYY-MM-DD when provided")

    data_fetcher = _get_data_fetcher()
    start_time = time.time()
    
    # Ensure minimum 90 days for holder_days (API requires longer range to return data)
    holder_days = max(holder_days, 90)
    
    # Ensure directories exist
    os.makedirs(str(MINTS_DIR), exist_ok=True)
    os.makedirs(str(CANDLES_DIR), exist_ok=True)
    os.makedirs(str(HOLDERS_DIR), exist_ok=True)
    os.makedirs(str(TRADES_DIR), exist_ok=True)
    
    try:
        # Step 1: Fetch mints with pagination
        logger.info(f"Fetching mints from {from_date} to {to_date} (max_pages={max_pages})...")
        mints_df = await asyncio.to_thread(data_fetcher.fetch_mints_range, from_date, to_date, limit=1000, max_pages=max_pages)
        
        if len(mints_df) == 0:
            return {
                "status": "success",
                "message": "No mints found for the specified date range",
                "mints_count": 0,
                "mints_file": None,
                "processed_mints": [],
                "from_date": from_date,
                "to_date": to_date
            }
        
        # Log DataFrame structure for debugging
        logger.info(f"Fetched {len(mints_df)} mints. Columns: {list(mints_df.columns)}")
        
        # Save mints file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mints_filename = f"mints_extracted_{timestamp}.csv"
        mints_filepath = os.path.join(str(MINTS_DIR), mints_filename)
        mints_df.to_csv(mints_filepath, index=False)
        logger.info(f"Saved {len(mints_df)} mints to {mints_filepath}")
        
        # Step 2: Find mint column
        mint_column = None
        possible_columns = [
            'Mint', 'mint', 'MINT',
            'address', 'Address', 'ADDRESS',
            'mint_address', 'MintAddress', 'mintAddress',
            'token', 'Token', 'TOKEN',
            'token_address', 'TokenAddress', 'tokenAddress'
        ]
        
        # First, try exact matches
        for col in possible_columns:
            if col in mints_df.columns:
                mint_column = col
                logger.info(f"Found mint column: {col}")
                break
        
        # If not found, try case-insensitive search
        if mint_column is None:
            df_columns_lower = [c.lower() for c in mints_df.columns]
            for col in possible_columns:
                if col.lower() in df_columns_lower:
                    mint_column = mints_df.columns[df_columns_lower.index(col.lower())]
                    logger.info(f"Found mint column (case-insensitive): {mint_column}")
                    break
        
        # If still not found, try to use the first column if it looks like an address
        if mint_column is None and len(mints_df.columns) > 0:
            first_col = mints_df.columns[0]
            sample_value = str(mints_df[first_col].iloc[0]) if len(mints_df) > 0 else ""
            if len(sample_value) >= 32 and len(sample_value) <= 44:
                mint_column = first_col
                logger.info(f"Using first column as mint address: {mint_column}")
        
        if mint_column is None:
            available_columns = list(mints_df.columns)
            error_msg = f"Could not find mint address column in mints data. Available columns: {available_columns}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Get unique mint addresses
        all_mint_addresses = mints_df[mint_column].dropna().unique().tolist()
        
        if len(all_mint_addresses) == 0:
            raise HTTPException(status_code=500, detail=f"No valid mint addresses found in column '{mint_column}'")
        
        # Apply max_mints limit
        mint_addresses = all_mint_addresses[:max_mints]
        
        logger.info(f"Found {len(all_mint_addresses)} unique mints, processing {len(mint_addresses)} (max_mints={max_mints})")
        
        # Step 3: Process mints in batches with throttling
        all_processed_mints = []
        all_errors = []
        
        # Create batches
        batches = [mint_addresses[i:i + batch_size] for i in range(0, len(mint_addresses), batch_size)]
        total_batches = len(batches)
        
        logger.info(f"Processing {len(mint_addresses)} mints in {total_batches} batches of {batch_size}")
        
        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"=== Starting batch {batch_idx}/{total_batches} ({len(batch)} mints) ===")
            
            # Process this batch
            processed, errors = await _process_mint_batch(
                data_fetcher=data_fetcher,
                mints=batch,
                candle_days=candle_days,
                holder_days=holder_days,
                trade_days=trade_days,
                trade_limit=trade_limit,
                timestamp=timestamp,
                batch_num=batch_idx,
                total_batches=total_batches
            )
            
            all_processed_mints.extend(processed)
            all_errors.extend(errors)
            
            # Log batch completion
            batch_candles = sum(m["candles"].get("count", 0) for m in processed if isinstance(m.get("candles"), dict))
            batch_holders = sum(m["holders"].get("count", 0) for m in processed if isinstance(m.get("holders"), dict))
            batch_trades = sum(m["trades"].get("count", 0) for m in processed if isinstance(m.get("trades"), dict))
            
            logger.info(f"=== Batch {batch_idx}/{total_batches} complete: {len(processed)} processed, "
                       f"{len(errors)} errors, {batch_candles} candles, {batch_holders} holders, {batch_trades} trades ===")
            
            # Delay between batches (except for the last batch)
            if batch_idx < total_batches:
                logger.info(f"Waiting {BATCH_DELAY}s before next batch...")
                await asyncio.sleep(BATCH_DELAY)
        
        # Calculate summary statistics
        total_candles = sum(m["candles"].get("count", 0) for m in all_processed_mints if isinstance(m.get("candles"), dict))
        total_holders = sum(m["holders"].get("count", 0) for m in all_processed_mints if isinstance(m.get("holders"), dict))
        total_trades = sum(m["trades"].get("count", 0) for m in all_processed_mints if isinstance(m.get("trades"), dict))
        
        elapsed_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(all_processed_mints)} mints in {elapsed_time:.1f}s",
            "mints_fetched": len(mints_df),
            "mints_processed": len(all_processed_mints),
            "mints_available": len(all_mint_addresses),
            "mints_file": {
                "filename": mints_filename,
                "filepath": mints_filepath,
                "count": len(mints_df)
            },
            "summary": {
                "total_candles": total_candles,
                "total_holders": total_holders,
                "total_trades": total_trades,
                "mints_with_candles": sum(1 for m in all_processed_mints if isinstance(m.get("candles"), dict) and m["candles"].get("count", 0) > 0),
                "mints_with_holders": sum(1 for m in all_processed_mints if isinstance(m.get("holders"), dict) and m["holders"].get("count", 0) > 0),
                "mints_with_trades": sum(1 for m in all_processed_mints if isinstance(m.get("trades"), dict) and m["trades"].get("count", 0) > 0)
            },
            "processed_mints": all_processed_mints,
            "errors": all_errors,
            "from_date": from_date,
            "to_date": to_date,
            "pagination": {
                "max_pages": max_pages,
                "max_mints": max_mints
            },
            "batch_config": {
                "batch_size": batch_size,
                "total_batches": total_batches,
                "batch_delay_seconds": BATCH_DELAY,
                "mint_delay_seconds": MINT_DELAY
            },
            "performance": {
                "elapsed_seconds": round(elapsed_time, 1),
                "mints_per_second": round(len(all_processed_mints) / elapsed_time, 2) if elapsed_time > 0 else 0
            },
            "timestamp": timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        with sentry_sdk.push_scope() as scope:
            scope.set_context("error_details", {"traceback": traceback.format_exc()})
            sentry_sdk.capture_exception(e)
        logger.error(f"Error in fetch_all_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch all data: {str(e)}")
