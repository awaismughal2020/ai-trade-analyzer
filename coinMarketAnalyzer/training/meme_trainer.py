"""
Meme Token Model Trainer
XGBoost binary classifier for meme token trading signals
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_config, MODELS_DIR, DATA_DIR

logger = logging.getLogger(__name__)


class MemeModelTrainer:
    """
    Trainer for meme token prediction model
    
    Binary classification: 0 = SELL/HOLD, 1 = BUY
    Uses 36 features from technical indicators, holder metrics, and whale analysis
    """
    
    def __init__(self):
        self.config = get_config()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        # Model output paths
        self.model_dir = MODELS_DIR / "meme"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.n_estimators = self.config.model.MEME_N_ESTIMATORS
        self.max_depth = self.config.model.MEME_MAX_DEPTH
        self.learning_rate = self.config.model.MEME_LEARNING_RATE
        
        logger.info("MemeModelTrainer initialized")
    
    def load_training_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load training data from CSV"""
        if data_path is None:
            data_path = DATA_DIR / "meme" / "train_data.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} training samples from {data_path}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare features for training
        
        Returns:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature column names
        """
        # Define feature columns (36 features)
        feature_columns = [
            # Technical indicators
            'rsi', 'rsi_signal', 'ema_20', 'ema_50', 'ema_cross',
            'macd_line', 'macd_signal', 'macd_histogram', 'macd_trend',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
            
            # Volume analysis
            'volume', 'volume_ma', 'volume_ratio', 'buy_sell_ratio',
            
            # Price momentum
            'price_momentum', 'volatility', 'atr',
            
            # Holder metrics
            'gini_coefficient', 'top10_concentration', 'holder_count',
            'unique_wallets', 'holder_growth_rate',
            
            # Whale metrics
            'whale_buy_volume', 'whale_sell_volume', 'whale_net_volume',
            'whale_state_encoded', 'dominant_whale_pct',
            
            # Market context
            'market_cap', 'liquidity', 'price_change_24h',
            
            # User profile (if available)
            'user_pnl', 'user_win_rate', 'user_trade_count'
        ]
        
        # Filter to available columns
        available_columns = [c for c in feature_columns if c in df.columns]
        
        if len(available_columns) < 10:
            raise ValueError(f"Insufficient features: only {len(available_columns)} available")
        
        logger.info(f"Using {len(available_columns)} features for training")
        
        # Get target column
        if 'target' not in df.columns and 'label' not in df.columns:
            raise ValueError("Target column ('target' or 'label') not found")
        
        target_col = 'target' if 'target' in df.columns else 'label'
        
        X = df[available_columns].values
        y = df[target_col].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_columns = available_columns
        
        return X, y, available_columns
    
    def train(
        self,
        data_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the meme token prediction model
        
        Args:
            data_path: Path to training data CSV
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting meme model training...")
        start_time = datetime.now()
        
        # Load data
        df = self.load_training_data(data_path)
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df)
        
        # Validate class distribution
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        total_samples = len(y)
        
        logger.info(f"Class distribution - Positive (BUY): {n_pos}, Negative (SELL/HOLD): {n_neg}, Total: {total_samples}")
        
        # Check if we have both classes
        if n_pos == 0:
            raise ValueError(f"No positive class (BUY) samples found in training data. "
                           f"All {total_samples} samples are negative (SELL/HOLD). "
                           f"Try adjusting the target threshold or collecting more data with price increases.")
        
        if n_neg == 0:
            raise ValueError(f"No negative class (SELL/HOLD) samples found in training data. "
                           f"All {total_samples} samples are positive (BUY). "
                           f"This is unusual - check your target labeling logic.")
        
        # Minimum samples per class for reliable training
        # Lower threshold for testing (2), but warn if below recommended (10)
        min_samples_per_class = 2
        recommended_samples = 10
        
        if n_pos < min_samples_per_class or n_neg < min_samples_per_class:
            raise ValueError(f"Insufficient samples per class. Need at least {min_samples_per_class} samples each. "
                           f"Found: Positive={n_pos}, Negative={n_neg}. "
                           f"Try extending the date range or processing more mints.")
        
        if n_pos < recommended_samples or n_neg < recommended_samples:
            logger.warning(f"Low sample count may result in poor model quality. "
                          f"Recommended: {recommended_samples}+ samples per class. "
                          f"Found: Positive={n_pos}, Negative={n_neg}")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(f"Using scale_pos_weight={scale_pos_weight:.2f} to handle class imbalance")
        
        # Split data - use stratify only if we have enough samples per class
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            # Fall back to non-stratified split if stratification fails
            logger.warning(f"Stratified split failed ({e}), using regular split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model with class imbalance handling
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        y_proba_test = self.model.predict_proba(X_test_scaled)[:, 1]
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Balanced metrics (critical for imbalanced data)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        
        # AUC-ROC (measures ranking quality regardless of threshold)
        try:
            test_auc_roc = roc_auc_score(y_test, y_proba_test)
        except ValueError:
            test_auc_roc = 0.0  # Only one class in test set
        
        # Classification report
        report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        results = {
            'train_accuracy': round(train_accuracy * 100, 2),
            'test_accuracy': round(test_accuracy * 100, 2),
            'train_test_gap': round((train_accuracy - test_accuracy) * 100, 2),
            # Balanced metrics
            'test_f1_score': round(test_f1 * 100, 2),
            'test_precision': round(test_precision * 100, 2),
            'test_recall': round(test_recall * 100, 2),
            'test_auc_roc': round(test_auc_roc * 100, 2),
            # Dataset info
            'n_features': len(feature_names),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'class_distribution': {
                'train_positive': int(np.sum(y_train == 1)),
                'train_negative': int(np.sum(y_train == 0)),
                'test_positive': int(np.sum(y_test == 1)),
                'test_negative': int(np.sum(y_test == 0)),
                'scale_pos_weight': round(scale_pos_weight, 2)
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'training_time_seconds': round(elapsed, 2),
            'feature_names': feature_names
        }
        
        logger.info(f"Training complete in {elapsed:.1f}s")
        logger.info(f"  Accuracy: {test_accuracy*100:.2f}%")
        logger.info(f"  F1 Score: {test_f1*100:.2f}%")
        logger.info(f"  Precision: {test_precision*100:.2f}% | Recall: {test_recall*100:.2f}%")
        logger.info(f"  AUC-ROC: {test_auc_roc*100:.2f}%")
        
        return results
    
    def save_model(self) -> Dict[str, str]:
        """Save trained model and artifacts"""
        if self.model is None:
            raise ValueError("No model to save - train first")
        
        model_path = self.model_dir / "xgboost_hybrid_model.pkl"
        scaler_path = self.model_dir / "feature_scaler.pkl"
        features_path = self.model_dir / "feature_columns.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        logger.info(f"Model saved to {model_path}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'features_path': str(features_path)
        }
    
    def load_model(self) -> bool:
        """Load existing trained model"""
        model_path = self.model_dir / "xgboost_hybrid_model.pkl"
        scaler_path = self.model_dir / "feature_scaler.pkl"
        features_path = self.model_dir / "feature_columns.pkl"
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        model_path = self.model_dir / "xgboost_hybrid_model.pkl"
        
        return {
            'model_type': 'XGBoost Binary Classifier',
            'token_type': 'meme',
            'model_exists': model_path.exists(),
            'model_path': str(model_path),
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trainer = MemeModelTrainer()
    print(trainer.get_model_info())
