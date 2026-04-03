"""
API Routes
"""
from .predict import router as predict_router
from .training import router as training_router

__all__ = ['predict_router', 'training_router']
