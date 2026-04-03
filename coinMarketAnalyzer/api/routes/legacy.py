"""
Legacy API Routes
Backward compatibility endpoints for older API versions
"""

import logging
from typing import Optional
from datetime import datetime

import sentry_sdk
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Legacy Training"])


# =============================================================================
# Legacy Training Endpoints (for backward compatibility)
# =============================================================================

@router.post("/train", summary="Start Training (Legacy)")
async def start_training_legacy(background_tasks: BackgroundTasks):
    """
    Legacy training endpoint
    
    This endpoint is deprecated. Please use:
    - POST /training/train/meme for meme tokens
    - POST /training/train/perps for perps tokens
    """
    return {
        "status": "deprecated",
        "message": "This endpoint is deprecated. Use POST /training/train/meme or POST /training/train/perps instead.",
        "new_endpoints": {
            "meme": "POST /training/train/meme",
            "perps": "POST /training/train/perps"
        }
    }


@router.get("/train/{job_id}", summary="Get Training Status (Legacy)")
async def get_training_status_legacy(job_id: str):
    """
    Legacy training status endpoint
    
    This endpoint is deprecated. Please use GET /training/jobs/{job_id}
    """
    try:
        from .training import _get_job

        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return job
    except HTTPException:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Legacy get training status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.get("/train", summary="List Training Jobs (Legacy)")
async def list_training_jobs_legacy(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Max jobs to return")
):
    """
    Legacy training jobs list endpoint

    This endpoint is deprecated. Please use GET /training/jobs
    """
    try:
        from .training import _load_jobs

        jobs = list(_load_jobs().values())

        if status:
            jobs = [j for j in jobs if j.get("status") == status]

        jobs.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)

        return {
            "total": len(jobs),
            "jobs": jobs[:limit],
            "deprecated": True,
            "message": "This endpoint is deprecated. Use GET /training/jobs instead."
        }
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Legacy list training jobs failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.delete("/train/{job_id}", summary="Cancel Training Job (Legacy)")
async def cancel_training_job_legacy(job_id: str):
    """
    Legacy cancel training endpoint

    This endpoint is deprecated. Please use DELETE /training/jobs/{job_id}
    """
    try:
        from .training import _get_job, _update_job

        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.get("status") in ["completed", "failed", "cancelled"]:
            return {
                "job_id": job_id,
                "status": job.get("status"),
                "message": f"Job has already {job.get('status')}"
            }

        # Mark as cancelled
        _update_job(job_id, {
            "status": "cancelled",
            "completed_at": datetime.now().isoformat(),
            "error": "Cancelled by user"
        })

        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully",
            "deprecated": True,
            "note": "This endpoint is deprecated. Use DELETE /training/jobs/{job_id} instead."
        }
    except HTTPException:
        raise
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Legacy cancel training job failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")
