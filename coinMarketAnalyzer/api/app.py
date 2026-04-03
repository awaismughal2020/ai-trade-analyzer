"""
Coin Market Analyzer - Main FastAPI Application
Unified API for meme tokens and perpetual futures analysis
"""

import json
import os
import re
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import time

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests as _requests

from config import get_config, MODELS_DIR
from api.routes.predict import router as predict_router
from api.routes.training import router as training_router
from api.routes.users import router as users_router
from api.routes.system import router as system_router
from api.routes.advanced import router as advanced_router
from api.routes.legacy import router as legacy_router
from api.routes.ai_profile import router as ai_profile_router
from api.routes.hybrid_profile import router as hybrid_profile_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    meme_model_loaded: bool
    perps_model_loaded: bool
    execution_time_seconds: float = 0.0

    class Config:
        protected_namespaces = ()


class InfoResponse(BaseModel):
    name: str
    version: str
    description: str
    supported_token_types: list
    endpoints: dict


# =============================================================================
# Request Timeout Middleware
# =============================================================================

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))


class GlobalExceptionMiddleware(BaseHTTPMiddleware):
    """Ultimate safety net — catches any exception that escapes all other handlers."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("http.route", request.url.path)
                scope.set_context("request_details", {
                    "method": request.method,
                    "path": request.url.path,
                    "query_string": str(request.query_params),
                })
                sentry_sdk.capture_exception(exc)
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path}: {exc}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )


class SentryRequestContextMiddleware(BaseHTTPMiddleware):
    """Attach full request context (method, path, query, body) to every Sentry event."""

    MAX_BODY_SIZE = 16_384  # 16 KB cap to avoid bloating Sentry events

    async def dispatch(self, request: Request, call_next):
        path = request.url.path or request.scope.get("path", "")
        method = request.method or "GET"
        query_string = str(request.query_params) if request.query_params else ""

        # Read request body for POST/PUT/PATCH (safe — Starlette caches it)
        request_body_str = None
        if method in ("POST", "PUT", "PATCH"):
            try:
                raw = await request.body()
                if raw and len(raw) <= self.MAX_BODY_SIZE:
                    request_body_str = raw.decode("utf-8", errors="replace")
            except Exception:
                request_body_str = "<unreadable>"

        with sentry_sdk.push_scope() as scope:
            scope.set_tag("http.method", method)
            scope.set_tag("http.route", path)
            scope.set_context("request_details", {
                "method": method,
                "path": path,
                "query_string": query_string,
                "body": request_body_str,
            })
            sentry_sdk.add_breadcrumb(
                category="http",
                message="request_started",
                data={"method": method, "path": path, "query": query_string},
            )
            try:
                return await call_next(request)
            finally:
                pass


class JsonSanitizingMiddleware(BaseHTTPMiddleware):
    """Auto-correct Python-style literals (True/False/None) in JSON bodies.

    Only activates when the raw body fails standard JSON parsing, so
    valid requests pay zero overhead.
    """

    _FIXES = [
        (re.compile(r'(?<!")True(?!")'), "true"),
        (re.compile(r'(?<!")False(?!")'), "false"),
        (re.compile(r'(?<!")None(?!")'), "null"),
    ]

    async def dispatch(self, request: Request, call_next):
        content_type = request.headers.get("content-type", "")
        if request.method in ("POST", "PUT", "PATCH") and "application/json" in content_type:
            body = await request.body()
            body_str = body.decode("utf-8")
            try:
                json.loads(body_str)
            except (json.JSONDecodeError, ValueError):
                fixed = body_str
                for pattern, replacement in self._FIXES:
                    fixed = pattern.sub(replacement, fixed)
                if fixed != body_str:
                    try:
                        json.loads(fixed)
                        logger.warning(
                            "Auto-corrected Python-style literals in request to %s",
                            request.url.path,
                        )
                        body = fixed.encode("utf-8")
                    except (json.JSONDecodeError, ValueError):
                        pass

            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive
            request._body = body

        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce a maximum request duration.
    
    Prevents any single request from running indefinitely. Returns a
    504 Gateway Timeout if the handler doesn't complete within the limit.
    """
    
    def __init__(self, app, timeout: int = REQUEST_TIMEOUT):
        super().__init__(app)
        self.timeout = timeout
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            sentry_sdk.capture_message(
                f"Request timed out after {self.timeout}s: {request.method} {request.url.path}",
                level="warning",
            )
            logger.warning(f"Request timed out after {self.timeout}s: {request.method} {request.url.path}")
            return JSONResponse(
                status_code=504,
                content={"detail": f"Request timed out after {self.timeout} seconds"}
            )


# =============================================================================
# Secret Redacting Filter
# =============================================================================

class SecretRedactingFilter(logging.Filter):
    """Scrubs known secret patterns from log output."""
    _PATTERNS = [
        re.compile(r"(sk-[a-zA-Z0-9]{20,})"),
        re.compile(r"(eyJ[a-zA-Z0-9_\-]{20,})"),
        re.compile(r"(https://[^@\s]+@[^/\s]+\.ingest\.sentry\.io\S*)"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pat in self._PATTERNS:
            msg = pat.sub("***REDACTED***", msg)
        record.msg = msg
        record.args = ()
        return True


def _redacted_url(url: str) -> str:
    """Return only the scheme + host of a URL so secrets in paths/params are hidden."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.hostname}" if parsed.hostname else url


# =============================================================================
# Health-check probe helpers (run in thread pool, never trip circuit breakers)
# =============================================================================

_HEALTH_PROBE_TIMEOUT = 5

def _check_openai(config) -> bool:
    try:
        from openai import OpenAI
        if not config.openai.API_KEY:
            return False
        client = OpenAI(api_key=config.openai.API_KEY)
        client.models.list()
        return True
    except Exception:
        return False


def _check_birdeye(config) -> bool:
    try:
        if not config.api.BIRDEYE_API_KEY:
            return False
        resp = _requests.get(
            f"{config.api.BIRDEYE_BASE_URL}/defi/token_overview",
            headers={"X-API-KEY": config.api.BIRDEYE_API_KEY, "x-chain": "solana"},
            params={"address": "So11111111111111111111111111111111111111112"},
            timeout=_HEALTH_PROBE_TIMEOUT,
        )
        return resp.status_code in (200, 400, 401)
    except Exception:
        return False


def _check_api_server(config) -> bool:
    try:
        resp = _requests.get(
            f"{config.api.INTERNAL_BASE_URL}/",
            timeout=_HEALTH_PROBE_TIMEOUT,
        )
        return resp.status_code < 500
    except Exception:
        return False


# =============================================================================
# Lifespan Context Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("=" * 60)
    logger.info("Coin Market Analyzer - Starting Up")
    logger.info("=" * 60)

    config = get_config()

    # Fail fast if critical env vars are missing
    config.validate()

    # Attach secret-redacting filter to root logger
    _redact_filter = SecretRedactingFilter()
    logging.getLogger().addFilter(_redact_filter)

    # Initialize Sentry
    if config.api.SENTRY_DSN:
        sentry_sdk.init(
            dsn=config.api.SENTRY_DSN,
            environment=config.api.SENTRY_ENVIRONMENT,
            traces_sample_rate=0,
            sample_rate=1.0,
            send_default_pii=False,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
            ],
        )
        sentry_sdk.set_tag("server", config.api.SENTRY_SERVER_TAG)
        logger.info(f"Sentry initialized (environment={config.api.SENTRY_ENVIRONMENT})")
    else:
        logger.warning("SENTRY_DSN not set — Sentry error tracking disabled")

    # Check model files
    meme_model_exists = (MODELS_DIR / "meme" / "xgboost_hybrid_model.pkl").exists()
    perps_model_exists = (MODELS_DIR / "perps" / "perps_model.pkl").exists()

    logger.info(f"Meme model: {'Found' if meme_model_exists else 'Not found'}")
    logger.info(f"Perps model: {'Found' if perps_model_exists else 'Not found'}")
    logger.info(f"API Server: {_redacted_url(config.api.INTERNAL_BASE_URL)}")

    logger.info("System ready!")
    logger.info("=" * 60)

    yield

    # Shutdown
    logging.getLogger().removeFilter(_redact_filter)
    logger.info("Coin Market Analyzer - Shutting Down")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Coin Market Analyzer",
    description="""
## Unified Trading Signal API

A comprehensive API for analyzing both **Meme Tokens** and **Perpetual Futures (Perps)**.

### Features

- **Multi-Layer Analysis**: ML Model, Whale Engine, Technical Indicators, Holder Metrics, User Profile
- **Dual Token Support**: Meme tokens (Solana) and Perps (HyperLiquid)
- **Training Endpoints**: Train and manage ML models via API
- **Data Pipeline**: Fetch and process training data

### Token Types

| Type | Description |
|------|-------------|
| `meme` | Solana meme tokens (default) |
| `perps` | Perpetual futures on HyperLiquid |

### Quick Start

1. **Get a prediction**: `GET /predict/{token_address}?token_type=meme`
2. **Train a model**: `POST /training/train/perps`
3. **Check model info**: `GET /training/info/perps`
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware stack (Starlette processes bottom-to-top, so last-added = outermost):
# 1. CORS — outermost, handles preflight
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 2. JSON sanitizer — fix Python-style literals before anything else parses the body
app.add_middleware(JsonSanitizingMiddleware)

# 3. Sentry request context — tag every request with method and route for error attribution
app.add_middleware(SentryRequestContextMiddleware)

# 4. Global exception handler — catches anything that escapes all other handlers
app.add_middleware(GlobalExceptionMiddleware)

# 5. Request timeout — innermost, closest to route handlers
app.add_middleware(TimeoutMiddleware, timeout=REQUEST_TIMEOUT)

# Include routers
app.include_router(predict_router)
app.include_router(training_router)
app.include_router(users_router)
app.include_router(system_router)
app.include_router(advanced_router)
app.include_router(legacy_router)
app.include_router(ai_profile_router)
app.include_router(hybrid_profile_router)


# =============================================================================
# Global Exception Handlers
# =============================================================================

async def _read_request_body_safe(request: Request, max_size: int = 16_384) -> str:
    """Read request body for Sentry context, returns truncated string."""
    try:
        raw = await request.body()
        if raw and len(raw) <= max_size:
            return raw.decode("utf-8", errors="replace")
        elif raw:
            return raw[:max_size].decode("utf-8", errors="replace") + "...<truncated>"
    except Exception:
        pass
    return None


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code >= 500:
        with sentry_sdk.push_scope() as scope:
            body = await _read_request_body_safe(request)
            scope.set_context("request_details", {
                "method": request.method,
                "path": request.url.path,
                "query_string": str(request.query_params),
                "body": body,
                "status_code": exc.status_code,
            })
            sentry_sdk.capture_exception(exc)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return str(obj)

    errors = [_make_serializable(err) for err in exc.errors()]

    with sentry_sdk.push_scope() as scope:
        body = await _read_request_body_safe(request)
        scope.set_context("request_details", {
            "method": request.method,
            "path": request.url.path,
            "query_string": str(request.query_params),
            "body": body,
        })
        scope.set_context("validation_errors", {"errors": errors})
        sentry_sdk.capture_exception(exc)

    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    with sentry_sdk.push_scope() as scope:
        body = await _read_request_body_safe(request)
        scope.set_context("request_details", {
            "method": request.method,
            "path": request.url.path,
            "query_string": str(request.query_params),
            "body": body,
        })
        sentry_sdk.capture_exception(exc)
    logger.error(f"Unhandled: {request.method} {request.url.path}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/", response_model=InfoResponse, tags=["Info"])
async def root():
    """Get API information"""
    return InfoResponse(
        name="Coin Market Analyzer",
        version="2.0.0",
        description="Unified API for meme tokens and perpetual futures analysis",
        supported_token_types=["meme", "perps"],
        endpoints={
            "prediction": {
                "POST /predict": "Get trading signal (POST)",
                "POST /predict/": "Get trading signal",
                "GET /predict/{token_address}": "Get trading signal (GET method)",
                "POST /predict/batch": "Batch predictions",
                "POST /predict/with-user": "Pattern-based prediction personalized with user trade history"
            },
            "training": {
                "GET /training/info/meme": "Meme model info",
                "GET /training/info/perps": "Perps model info",
                "POST /training/train/meme": "Train meme model",
                "POST /training/train/perps": "Train perps model",
                "POST /training/data-pipeline/perps": "Fetch perps training data",
                "GET /training/data-info": "Training data info",
                "GET /training/jobs/{job_id}": "Get training job status",
                "GET /training/jobs": "List training jobs"
            },
            "users": {
                "GET /users": "List users",
                "GET /user/profile/{wallet_address}": "Get user profile",
                "POST /user/assess": "Assess user risk"
            },
            "system": {
                "GET /model/info": "Model information",
                "GET /outputs": "List output files",
                "GET /outputs/{filename}": "Download output file"
            },
            "review": {
                "POST /post-trade-review": "Post-trade analysis"
            },
            "data": {
                "POST /data/fetch-all": "Fetch all data"
            },
            "health": {
                "GET /health": "Health check",
                "GET /health/live": "Liveness probe",
                "GET /health/ready": "Readiness probe"
            }
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    start = time.perf_counter()
    meme_model_exists = (MODELS_DIR / "meme" / "xgboost_hybrid_model.pkl").exists()
    perps_model_exists = (MODELS_DIR / "perps" / "perps_model.pkl").exists()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="2.0.0",
        meme_model_loaded=meme_model_exists,
        perps_model_loaded=perps_model_exists,
        execution_time_seconds=round(time.perf_counter() - start, 4),
    )


@app.get("/health/live", tags=["Health"])
async def health_live():
    """Liveness probe -- is the process alive and are models present?"""
    start = time.perf_counter()
    meme_ok = (MODELS_DIR / "meme" / "xgboost_hybrid_model.pkl").exists()
    perps_ok = (MODELS_DIR / "perps" / "perps_model.pkl").exists()
    status = "live" if (meme_ok and perps_ok) else "degraded"
    return {
        "status": status,
        "meme_model_loaded": meme_ok,
        "perps_model_loaded": perps_ok,
        "server_available": True,
        "execution_time_seconds": round(time.perf_counter() - start, 4),
    }


@app.get("/health/ready", tags=["Health"])
async def health_ready():
    """Readiness probe -- can we reach all external dependencies?"""
    start = time.perf_counter()
    config = get_config()
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=3) as pool:
        openai_fut = loop.run_in_executor(pool, _check_openai, config)
        birdeye_fut = loop.run_in_executor(pool, _check_birdeye, config)
        api_fut = loop.run_in_executor(pool, _check_api_server, config)

        checks = {
            "openai": await openai_fut,
            "birdeye": await birdeye_fut,
            "api_server": await api_fut,
        }

    checks["meme_model_loaded"] = (MODELS_DIR / "meme" / "xgboost_hybrid_model.pkl").exists()
    checks["perps_model_loaded"] = (MODELS_DIR / "perps" / "perps_model.pkl").exists()

    all_ok = all(checks.values())
    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "execution_time_seconds": round(time.perf_counter() - start, 4),
    }


@app.get("/api", tags=["Info"])
async def api_info():
    """API documentation redirect"""
    return {
        "message": "Coin Market Analyzer API",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
