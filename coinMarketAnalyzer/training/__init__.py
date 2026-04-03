"""
Coin Market Analyzer - Training Module
Model training for both Meme and Perps tokens
"""
from .meme_trainer import MemeModelTrainer
from .perps_trainer import PerpsModelTrainer

__all__ = [
    'MemeModelTrainer',
    'PerpsModelTrainer'
]
