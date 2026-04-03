"""
System API Routes
Endpoints for system information, model info, file outputs, and rate limiter status
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

import sentry_sdk
from fastapi import APIRouter, HTTPException, Path as FastPath
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import MODELS_DIR, DATA_DIR, BASE_DIR, api_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])

# Outputs directory
OUTPUTS_DIR = BASE_DIR / "outputs"


# =============================================================================
# Response Models
# =============================================================================

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    feature_count: int
    model_path: str
    model_exists: bool
    last_trained: Optional[str] = None
    meme_model_exists: bool = False
    perps_model_exists: bool = False
    
    class Config:
        protected_namespaces = ()


# =============================================================================
# System Endpoints
# =============================================================================

@router.get("/model/info", response_model=ModelInfoResponse, summary="Get Model Info")
async def model_info():
    """
    Get information about trained models
    
    Returns status, paths, and metadata for both meme and perps models.
    """
    try:
        # Check meme model
        meme_model_path = MODELS_DIR / "meme" / "xgboost_hybrid_model.pkl"
        meme_exists = meme_model_path.exists()
        meme_last_trained = None
        
        if meme_exists:
            meme_last_trained = datetime.fromtimestamp(meme_model_path.stat().st_mtime).isoformat()
        
        # Check perps model
        perps_model_path = MODELS_DIR / "perps" / "perps_model.pkl"
        perps_exists = perps_model_path.exists()
        perps_last_trained = None
        
        if perps_exists:
            perps_last_trained = datetime.fromtimestamp(perps_model_path.stat().st_mtime).isoformat()
        
        # Count features for perps model
        perps_feature_count = 0
        if perps_exists:
            try:
                import pickle
                features_path = MODELS_DIR / "perps" / "perps_features.pkl"
                if features_path.exists():
                    with open(features_path, 'rb') as f:
                        features = pickle.load(f)
                        perps_feature_count = len(features) if isinstance(features, (list, tuple)) else 0
            except:
                pass
        
        # Return info about primary model (perps if exists, else meme)
        if perps_exists:
            return ModelInfoResponse(
                model_type="XGBoost Binary Classifier (Perps)",
                feature_count=perps_feature_count,
                model_path=str(perps_model_path),
                model_exists=True,
                last_trained=perps_last_trained,
                meme_model_exists=meme_exists,
                perps_model_exists=True
            )
        elif meme_exists:
            return ModelInfoResponse(
                model_type="XGBoost Hybrid Model (Meme)",
                feature_count=0,
                model_path=str(meme_model_path),
                model_exists=True,
                last_trained=meme_last_trained,
                meme_model_exists=True,
                perps_model_exists=False
            )
        else:
            return ModelInfoResponse(
                model_type="No model trained",
                feature_count=0,
                model_path="N/A",
                model_exists=False,
                meme_model_exists=False,
                perps_model_exists=False
            )
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")


@router.get("/outputs", summary="List Output Files")
async def list_outputs():
    """
    List available output files (results, reports, etc.)
    
    Returns list of downloadable files from the outputs directory.
    """
    try:
        if not OUTPUTS_DIR.exists():
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            return {
                "outputs_dir": str(OUTPUTS_DIR),
                "files": [],
                "total": 0
            }
        
        files = []
        for file in OUTPUTS_DIR.iterdir():
            if file.is_file():
                files.append({
                    "filename": file.name,
                    "size_bytes": file.stat().st_size,
                    "size_mb": round(file.stat().st_size / 1024 / 1024, 2),
                    "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                })
        
        return {
            "outputs_dir": str(OUTPUTS_DIR),
            "files": sorted(files, key=lambda x: x['modified'], reverse=True),
            "total": len(files)
        }
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error listing outputs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list outputs: {e}")


@router.get("/outputs/{filename}", summary="Download Output File")
async def download_output(filename: str = FastPath(..., description="Output filename")):
    """
    Download output file
    
    Returns the requested file from the outputs directory.
    """
    try:
        filepath = OUTPUTS_DIR / filename
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        if not filepath.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {filename}")
        
        # Security check: ensure file is within outputs directory
        if not str(filepath.resolve()).startswith(str(OUTPUTS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return FileResponse(
            path=str(filepath),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error downloading output: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download output: {e}")


# =============================================================================
# Rate Limiter Endpoints
# =============================================================================

@router.get("/rate-limiter/status", summary="Get Rate Limiter Status")
async def get_rate_limiter_status():
    """
    Get the current status of all rate limiters.
    
    Returns per-limiter stats:
    - Current request count in the window
    - Consecutive failure count
    - Circuit breaker state
    - Configuration settings
    """
    try:
        from core.data_fetcher import _rate_limiters
        
        limiters_status = {}
        for name, limiter in _rate_limiters.items():
            stats = limiter.get_stats()
            limiters_status[name] = {
                **stats,
                "circuit_status": "OPEN" if stats["circuit_open"] else "CLOSED",
                "health": "degraded" if stats["consecutive_failures"] > 0 else "healthy"
            }
        
        return {
            "status": "ok",
            "rate_limiters": limiters_status,
            "active_limiters": list(_rate_limiters.keys()),
            "config": {
                "requests_per_second": api_config.REQUESTS_PER_SECOND,
                "max_concurrent_requests": api_config.MAX_CONCURRENT_REQUESTS,
                "min_request_interval": api_config.MIN_REQUEST_INTERVAL,
                "backoff_factor": api_config.BACKOFF_FACTOR,
                "max_backoff": api_config.MAX_BACKOFF,
                "circuit_breaker_threshold": api_config.CIRCUIT_BREAKER_THRESHOLD,
                "circuit_breaker_reset": api_config.CIRCUIT_BREAKER_RESET
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error getting rate limiter status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rate limiter status: {e}")


@router.post("/rate-limiter/reset", summary="Reset Rate Limiter")
async def reset_rate_limiter():
    """
    Reset all rate limiter states.
    
    This will:
    - Reset all circuit breakers if any were open
    - Clear all failure counters
    - Reset request tracking
    
    Use this if rate limiters are stuck in a degraded state.
    """
    try:
        from core.data_fetcher import _rate_limiters
        
        before_stats = {}
        after_stats = {}
        
        for name, limiter in _rate_limiters.items():
            before_stats[name] = limiter.get_stats()
            limiter._reset_circuit_breaker()
            after_stats[name] = limiter.get_stats()
        
        return {
            "status": "success",
            "message": f"Reset {len(_rate_limiters)} rate limiter(s): {list(_rate_limiters.keys())}",
            "before": before_stats,
            "after": after_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error resetting rate limiter: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset rate limiter: {e}")
