"""
Training API Routes
Endpoints for training meme and perps models with automatic data extraction
"""

import logging
import uuid
import json
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from enum import Enum

import sentry_sdk
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.meme_trainer import MemeModelTrainer
from training.perps_trainer import PerpsModelTrainer
from training.data_pipeline.perps_pipeline import PerpsDataPipeline
from training.data_pipeline.meme_pipeline import MemeDataPipeline
from config import TokenType, DATA_DIR, MODELS_DIR, get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])


# =============================================================================
# Job Status Enum
# =============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Job Storage (File-Based)
# =============================================================================

JOBS_FILE = DATA_DIR / "training_jobs.json"

def _load_jobs() -> Dict[str, Dict]:
    """Load jobs from file"""
    if JOBS_FILE.exists():
        try:
            with open(JOBS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_jobs(jobs: Dict[str, Dict]):
    """Save jobs to file"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(JOBS_FILE, 'w') as f:
        json.dump(jobs, f, indent=2)

def _get_job(job_id: str) -> Optional[Dict]:
    """Get job by ID"""
    jobs = _load_jobs()
    return jobs.get(job_id)

def _update_job(job_id: str, updates: Dict):
    """Update job data"""
    jobs = _load_jobs()
    if job_id in jobs:
        jobs[job_id].update(updates)
        _save_jobs(jobs)

def _create_job(job_id: str, job_data: Dict):
    """Create new job"""
    jobs = _load_jobs()
    jobs[job_id] = job_data
    _save_jobs(jobs)


# =============================================================================
# Request/Response Models
# =============================================================================

class MemeTrainingRequest(BaseModel):
    """Request model for meme training"""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)", pattern=r'^\d{4}-\d{2}-\d{2}$')
    end_date: str = Field(..., description="End date (YYYY-MM-DD)", pattern=r'^\d{4}-\d{2}-\d{2}$')
    use_csv_data: bool = Field(True, description="Use pre-collected CSV data from /data/fetch-all (recommended). Set to False to fetch fresh from API.")
    max_mints: int = Field(10000, ge=10, le=50000, description="Maximum number of mints to process for training. Set higher to use all available data.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "use_csv_data": True,
                "max_mints": 10000
            }
        }


class PerpsTrainingRequest(BaseModel):
    """Request model for perps training"""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)", pattern=r'^\d{4}-\d{2}-\d{2}$')
    end_date: str = Field(..., description="End date (YYYY-MM-DD)", pattern=r'^\d{4}-\d{2}-\d{2}$')
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-31"
            }
        }


class TrainingJobResponse(BaseModel):
    """Response for training job submission"""
    job_id: str
    status: str
    message: str
    submitted_at: str


class TrainingStatusResponse(BaseModel):
    """Response for training job status"""
    job_id: str
    status: str
    progress: Optional[float] = None
    current_step: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    token_type: str
    model_exists: bool
    model_path: str
    last_trained: Optional[str] = None
    feature_count: int = 0
    metrics: Optional[Dict[str, Any]] = None

    class Config:
        protected_namespaces = ()


class DataInfoResponse(BaseModel):
    """Training data info response"""
    meme_data: Dict[str, Any]
    perps_data: Dict[str, Any]
    timestamp: str


# =============================================================================
# Background Job Handlers
# =============================================================================

async def run_meme_training_job(job_id: str, request: MemeTrainingRequest):
    """Background task for meme training with data pipeline"""
    try:
        _update_job(job_id, {
            "status": JobStatus.RUNNING,
            "started_at": datetime.now().isoformat(),
            "progress": 10,
            "current_step": "Starting meme training"
        })
        
        logger.info(f"Meme training job {job_id}: {request.start_date} to {request.end_date}")
        logger.info(f"Using CSV data: {request.use_csv_data}, max_mints: {request.max_mints}")
        
        # Step 1: Run data pipeline
        data_source = "pre-collected CSV data" if request.use_csv_data else "fresh API fetch"
        _update_job(job_id, {"progress": 20, "current_step": f"Extracting meme token data from {data_source} (up to {request.max_mints} mints)"})
        
        pipeline = MemeDataPipeline()
        pipeline_result = pipeline.run_pipeline(
            from_date=request.start_date,
            to_date=request.end_date,
            max_mints=request.max_mints,
            candle_days=90,
            use_csv_data=request.use_csv_data
        )
        
        if not pipeline_result.get('success'):
            raise Exception(f"Data pipeline failed: {pipeline_result.get('error')}")
        
        logger.info(f"Pipeline extracted {pipeline_result.get('total_records', 0)} training samples")
        
        # Step 2: Train model
        _update_job(job_id, {"progress": 60, "current_step": "Training model"})
        
        trainer = MemeModelTrainer()
        results = trainer.train()
        
        _update_job(job_id, {"progress": 90, "current_step": "Saving model"})
        
        saved = trainer.save_model()
        
        _update_job(job_id, {
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "current_step": "Complete",
            "completed_at": datetime.now().isoformat(),
            "result": {
                "model_type": "meme",
                "date_range": {"start": request.start_date, "end": request.end_date},
                "max_mints_requested": request.max_mints,
                "train_accuracy": results.get('train_accuracy', 0),
                "test_accuracy": results.get('test_accuracy', 0),
                "train_test_gap": results.get('train_test_gap', 0),
                "test_f1_score": results.get('test_f1_score', 0),
                "test_precision": results.get('test_precision', 0),
                "test_recall": results.get('test_recall', 0),
                "test_auc_roc": results.get('test_auc_roc', 0),
                "n_train_samples": results.get('n_train_samples', 0),
                "n_test_samples": results.get('n_test_samples', 0),
                "n_features": results.get('n_features', 0),
                "class_distribution": results.get('class_distribution', {}),
                "mints_processed": pipeline_result.get('mints_processed', 0),
                "mints_attempted": pipeline_result.get('mints_attempted', 0),
                "samples_per_mint": pipeline_result.get('samples_per_mint', 1),
                "target_distribution": pipeline_result.get('target_distribution', {}),
                "labeling_method": pipeline_result.get('labeling_method', 'unknown'),
                "data_source": pipeline_result.get('data_source', 'unknown'),
                "model_path": saved.get('model_path'),
                "message": f"Meme model trained: Accuracy={results.get('test_accuracy', 0):.1f}%, F1={results.get('test_f1_score', 0):.1f}%, AUC-ROC={results.get('test_auc_roc', 0):.1f}%, Precision={results.get('test_precision', 0):.1f}%, Recall={results.get('test_recall', 0):.1f}%"
            }
        })
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Meme training failed: {e}", exc_info=True)
        _update_job(job_id, {
            "status": JobStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


async def run_perps_training_job(job_id: str, request: PerpsTrainingRequest):
    """Background task for perps training with data pipeline"""
    try:
        _update_job(job_id, {
            "status": JobStatus.RUNNING,
            "started_at": datetime.now().isoformat(),
            "progress": 10,
            "current_step": "Starting perps training"
        })
        
        logger.info(f"Perps training job {job_id}: {request.start_date} to {request.end_date}")
        
        # Configured tickers (from config)
        config = get_config()
        tickers = config.perps.TRAINING_TICKERS
        
        # Step 1: Run data pipeline
        _update_job(job_id, {"progress": 20, "current_step": "Extracting perps data"})
        
        pipeline = PerpsDataPipeline()
        pipeline_result = pipeline.run_pipeline(
            tickers=tickers,
            from_date=request.start_date,
            to_date=request.end_date
        )
        
        if not pipeline_result.get('success'):
            raise Exception(f"Data pipeline failed: {pipeline_result.get('error')}")
        
        # Step 2: Train model
        _update_job(job_id, {"progress": 60, "current_step": "Training model"})
        
        trainer = PerpsModelTrainer()
        results = trainer.train()
        
        _update_job(job_id, {"progress": 90, "current_step": "Saving model"})
        
        saved = trainer.save_model()
        
        _update_job(job_id, {
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "current_step": "Complete",
            "completed_at": datetime.now().isoformat(),
            "result": {
                "model_type": "perps",
                "tickers_used": tickers,
                "date_range": {"start": request.start_date, "end": request.end_date},
                "train_accuracy": results.get('train_accuracy', 0),
                "test_accuracy": results.get('test_accuracy', 0),
                "train_test_gap": results.get('train_test_gap', 0),
                "n_train_samples": results.get('n_train_samples', 0),
                "n_test_samples": results.get('n_test_samples', 0),
                "n_features": results.get('n_features', 0),
                "class_distribution": results.get('class_distribution', {}),
                "model_path": saved.get('model_path'),
                "message": f"Perps model trained with {results.get('test_accuracy', 0):.2f}% accuracy (train: {results.get('train_accuracy', 0):.2f}%, gap: {results.get('train_test_gap', 0):.2f}%)"
            }
        })
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Perps training failed: {e}", exc_info=True)
        _update_job(job_id, {
            "status": JobStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


# =============================================================================
# Training Endpoints
# =============================================================================

@router.get("/info/meme", response_model=ModelInfoResponse, summary="Get Meme Model Info")
async def get_meme_model_info():
    """
    Get information about the meme token prediction model
    
    Returns model status, training metrics, and feature information.
    """
    model_path = MODELS_DIR / "meme" / "xgboost_hybrid_model.pkl"
    model_exists = model_path.exists()
    
    last_trained = None
    if model_exists:
        last_trained = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
    
    return ModelInfoResponse(
        model_type="XGBoost Classifier",
        token_type="meme",
        model_exists=model_exists,
        model_path=str(model_path),
        last_trained=last_trained,
        feature_count=0,
        metrics=None
    )


@router.get("/info/perps", response_model=ModelInfoResponse, summary="Get Perps Model Info")
async def get_perps_model_info():
    """
    Get information about the perps prediction model
    
    Returns model status, training metrics, and feature information.
    """
    import pickle
    
    model_path = MODELS_DIR / "perps" / "perps_model.pkl"
    model_exists = model_path.exists()
    
    last_trained = None
    feature_count = 0
    metrics = None
    
    if model_exists:
        last_trained = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        
        # Load features
        features_path = MODELS_DIR / "perps" / "perps_features.pkl"
        if features_path.exists():
            try:
                with open(features_path, 'rb') as f:
                    features = pickle.load(f)
                    feature_count = len(features) if isinstance(features, (list, tuple)) else 0
            except:
                pass
    
    return ModelInfoResponse(
        model_type="XGBoost Binary Classifier (LONG vs NOT_LONG)",
        token_type="perps",
        model_exists=model_exists,
        model_path=str(model_path),
        last_trained=last_trained,
        feature_count=feature_count,
        metrics=metrics
    )


@router.post("/train/meme", response_model=TrainingJobResponse, summary="Train Meme Model")
async def train_meme_model(request: MemeTrainingRequest, background_tasks: BackgroundTasks):
    """
    Train the Meme coin ML model with automatic data extraction.
    
    **Input:**
    - `start_date`: Start date for mint extraction (YYYY-MM-DD)
    - `end_date`: End date for mint extraction (YYYY-MM-DD)
    
    **Training Details:**
    - Uses XGBoost classifier for BUY/SELL/HOLD signals
    - Multi-layer system: ML Model, Whale Engine, Technical, Holder Metrics
    - Automatically extracts candles, holders, and trades data
    
    Training runs as a background task. Use GET /training/jobs/{job_id} to check status.
    """
    job_id = f"meme-{str(uuid.uuid4())[:8]}"
    
    job_data = {
        "job_id": job_id,
        "model_type": "meme",
        "status": JobStatus.PENDING,
        "progress": 0,
        "current_step": "Queued",
        "submitted_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    _create_job(job_id, job_data)
    background_tasks.add_task(run_meme_training_job, job_id, request)
    
    logger.info(f"Meme training job {job_id} submitted")
    
    return TrainingJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Meme model training started. Check status at /training/jobs/{job_id}",
        submitted_at=job_data["submitted_at"]
    )


@router.post("/train/perps", response_model=TrainingJobResponse, summary="Train Perps Model")
async def train_perps_model(request: PerpsTrainingRequest, background_tasks: BackgroundTasks):
    """
    Train the Perps ML model with automatic data extraction.
    
    **Input:**
    - `start_date`: Start date for data extraction (YYYY-MM-DD)
    - `end_date`: End date for data extraction (YYYY-MM-DD)
    
    **Configured Tickers (from PerpsConfig.TRAINING_TICKERS):**
    Default: BTC-USD, ETH-USD, SOL-USD, APE-USD, AVAX-USD, ATOM-USD, BNB-USD, DYDX-USD, OP-USD
    Override via env: PERPS_TRAINING_TICKERS="BTC-USD,ETH-USD,SOL-USD"
    
    **Training Details:**
    - Uses XGBoost with binary classification (LONG vs NOT_LONG)
    - Target accuracy: >65%
    - Uses 56 engineered features including technical indicators, funding rate, and OI metrics
    
    Training runs as a background task. Use GET /training/jobs/{job_id} to check status.
    """
    job_id = f"perps-{str(uuid.uuid4())[:8]}"
    
    job_data = {
        "job_id": job_id,
        "model_type": "perps",
        "status": JobStatus.PENDING,
        "progress": 0,
        "current_step": "Queued",
        "submitted_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    _create_job(job_id, job_data)
    background_tasks.add_task(run_perps_training_job, job_id, request)
    
    logger.info(f"Perps training job {job_id} submitted")
    
    return TrainingJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Perps model training started. Check status at /training/jobs/{job_id}",
        submitted_at=job_data["submitted_at"]
    )


@router.get("/jobs/{job_id}", response_model=TrainingStatusResponse, summary="Get Training Job Status")
async def get_training_job_status(job_id: str):
    """
    Check status of a training job.
    
    Returns current progress, step name, and results when complete.
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return TrainingStatusResponse(**job)


@router.get("/jobs", summary="List Training Jobs")
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Max jobs to return")
):
    """List all training jobs with optional filtering"""
    jobs = list(_load_jobs().values())
    
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    
    jobs.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
    
    return {
        "total": len(jobs),
        "jobs": jobs[:limit]
    }


@router.delete("/jobs/{job_id}", summary="Cancel Training Job")
async def cancel_training_job(job_id: str):
    """
    Cancel a pending or running training job
    
    Note: Running jobs may not be immediately cancelled.
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.get("status") == JobStatus.COMPLETED:
        return {
            "job_id": job_id,
            "status": "already_completed",
            "message": "Job has already completed"
        }
    
    if job.get("status") == JobStatus.FAILED:
        return {
            "job_id": job_id,
            "status": "already_failed",
            "message": "Job has already failed"
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
        "message": "Job cancelled successfully"
    }


@router.get("/data-info", response_model=DataInfoResponse, summary="Get Training Data Info")
async def get_training_data_info():
    """
    Get information about available training data for both meme and perps models.
    
    Returns file paths, record counts, and date ranges.
    """
    import pandas as pd
    
    meme_data_path = DATA_DIR / "meme" / "train_data.csv"
    perps_data_path = DATA_DIR / "perps" / "train_data.csv"
    
    meme_data = {
        "exists": meme_data_path.exists(),
        "path": str(meme_data_path),
        "size_mb": round(meme_data_path.stat().st_size / 1024 / 1024, 2) if meme_data_path.exists() else 0,
        "total_records": 0
    }
    
    perps_data = {
        "exists": perps_data_path.exists(),
        "path": str(perps_data_path),
        "size_mb": round(perps_data_path.stat().st_size / 1024 / 1024, 2) if perps_data_path.exists() else 0,
        "total_records": 0,
        "tickers": []
    }
    
    if perps_data_path.exists():
        try:
            df = pd.read_csv(perps_data_path, nrows=1000)
            with open(perps_data_path, 'r') as f:
                perps_data["total_records"] = sum(1 for _ in f) - 1
            if 'ticker' in df.columns:
                perps_data["tickers"] = df['ticker'].unique().tolist()
        except:
            pass
    
    return DataInfoResponse(
        meme_data=meme_data,
        perps_data=perps_data,
        timestamp=datetime.now().isoformat()
    )


# =============================================================================
# Data Pipeline Endpoints
# =============================================================================

class DataPipelineJobRequest(BaseModel):
    """Request for data pipeline job"""
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2025-01-01",
                "end_date": "2025-01-31"
            }
        }


async def run_perps_data_pipeline_job(job_id: str, request: DataPipelineJobRequest):
    """Background task for perps data pipeline"""
    try:
        _update_job(job_id, {
            "status": JobStatus.RUNNING,
            "started_at": datetime.now().isoformat(),
            "progress": 10,
            "current_step": "Starting data extraction"
        })
        
        config = get_config()
        tickers = config.perps.TRAINING_TICKERS
        
        logger.info(f"Data pipeline job {job_id}: extracting {len(tickers)} tickers")
        
        _update_job(job_id, {"progress": 30, "current_step": "Fetching OHLCV and funding data"})
        
        pipeline = PerpsDataPipeline()
        results = pipeline.run_pipeline(tickers=tickers)
        
        if not results.get('success'):
            raise Exception(f"Pipeline failed: {results.get('error')}")
        
        _update_job(job_id, {
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "current_step": "Complete",
            "completed_at": datetime.now().isoformat(),
            "result": {
                "output_path": results.get('output_path'),
                "total_records": results.get('total_records', 0),
                "tickers": results.get('tickers', []),
                "message": f"Data pipeline completed with {results.get('total_records', 0)} records"
            }
        })
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Data pipeline failed: {e}", exc_info=True)
        _update_job(job_id, {
            "status": JobStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


@router.post("/data-pipeline/perps", response_model=TrainingJobResponse, summary="Run Perps Data Pipeline")
async def run_perps_data_pipeline(request: DataPipelineJobRequest, background_tasks: BackgroundTasks):
    """
    Run the Perps data pipeline to extract and prepare training data.
    
    This endpoint:
    1. Fetches OHLCV data from perps API for all configured tickers
    2. Fetches funding rate data
    3. Merges and prepares data for training
    4. Saves to data/perps/train_data.csv
    
    **Configured Tickers:**
    BTC-USD, ETH-USD, SOL-USD, APE-USD, AVAX-USD, ATOM-USD, BNB-USD, DYDX-USD, OP-USD
    
    Pipeline runs as a background task. Use GET /training/jobs/{job_id} to check status.
    """
    job_id = f"perps-pipeline-{str(uuid.uuid4())[:8]}"
    
    job_data = {
        "job_id": job_id,
        "job_type": "data_pipeline",
        "model_type": "perps",
        "status": JobStatus.PENDING,
        "progress": 0,
        "current_step": "Queued",
        "submitted_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    _create_job(job_id, job_data)
    background_tasks.add_task(run_perps_data_pipeline_job, job_id, request)
    
    logger.info(f"Perps data pipeline job {job_id} submitted")
    
    return TrainingJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Perps data pipeline started. Check status at /training/jobs/{job_id}",
        submitted_at=job_data["submitted_at"]
    )


async def run_meme_data_pipeline_job(job_id: str, request: DataPipelineJobRequest):
    """Background task for meme data pipeline"""
    try:
        _update_job(job_id, {
            "status": JobStatus.RUNNING,
            "started_at": datetime.now().isoformat(),
            "progress": 10,
            "current_step": "Starting data extraction"
        })
        
        # Set defaults
        from_date = request.start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        to_date = request.end_date or datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Data pipeline job {job_id}: extracting meme tokens from {from_date} to {to_date}")
        
        _update_job(job_id, {"progress": 30, "current_step": "Fetching mints and token data"})
        
        pipeline = MemeDataPipeline()
        results = pipeline.run_pipeline(
            from_date=from_date,
            to_date=to_date,
            max_mints=100,
            candle_days=90
        )
        
        if not results.get('success'):
            raise Exception(f"Pipeline failed: {results.get('error')}")
        
        _update_job(job_id, {
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "current_step": "Complete",
            "completed_at": datetime.now().isoformat(),
            "result": {
                "output_path": results.get('output_path'),
                "total_records": results.get('total_records', 0),
                "mints_processed": results.get('mints_processed', 0),
                "mints_attempted": results.get('mints_attempted', 0),
                "target_distribution": results.get('target_distribution', {}),
                "date_range": results.get('date_range', {}),
                "message": f"Data pipeline completed with {results.get('total_records', 0)} records from {results.get('mints_processed', 0)} mints"
            }
        })
        
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Data pipeline failed: {e}", exc_info=True)
        _update_job(job_id, {
            "status": JobStatus.FAILED,
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })


@router.post("/data-pipeline/meme", response_model=TrainingJobResponse, summary="Run Meme Data Pipeline")
async def run_meme_data_pipeline(request: DataPipelineJobRequest, background_tasks: BackgroundTasks):
    """
    Run the Meme token data pipeline to extract and prepare training data.
    
    This endpoint:
    1. Fetches mints within the specified date range
    2. For each mint, fetches candles, holders, trades
    3. Extracts technical indicators, holder metrics, whale behavior
    4. Creates training labels based on price movement
    5. Saves to data/meme/train_data.csv
    
    **Parameters:**
    - `start_date`: Start date for mint discovery (YYYY-MM-DD)
    - `end_date`: End date for mint discovery (YYYY-MM-DD)
    
    Pipeline runs as a background task. Use GET /training/jobs/{job_id} to check status.
    """
    job_id = f"meme-pipeline-{str(uuid.uuid4())[:8]}"
    
    job_data = {
        "job_id": job_id,
        "job_type": "data_pipeline",
        "model_type": "meme",
        "status": JobStatus.PENDING,
        "progress": 0,
        "current_step": "Queued",
        "submitted_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    _create_job(job_id, job_data)
    background_tasks.add_task(run_meme_data_pipeline_job, job_id, request)
    
    logger.info(f"Meme data pipeline job {job_id} submitted")
    
    return TrainingJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Meme data pipeline started. Check status at /training/jobs/{job_id}",
        submitted_at=job_data["submitted_at"]
    )
