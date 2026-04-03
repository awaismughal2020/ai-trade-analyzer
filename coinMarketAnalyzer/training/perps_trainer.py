"""
Perpetual Futures Model Trainer
XGBoost binary classifier for perps trading signals
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_config, MODELS_DIR, DATA_DIR

logger = logging.getLogger(__name__)


class PerpsModelTrainer:
    """
    Trainer for perpetual futures prediction model
    
    Binary classification: 0 = NOT_LONG, 1 = LONG
    Uses 56 features from OHLCV, funding rates, and technical indicators
    """
    
    def __init__(self):
        self.config = get_config()
        self.model = None
        self.scaler = StandardScaler()
        self.ticker_encoder = LabelEncoder()
        self.feature_columns = None
        
        # Model output paths
        self.model_dir = MODELS_DIR / "perps"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.n_estimators = self.config.model.PERPS_N_ESTIMATORS
        self.max_depth = self.config.model.PERPS_MAX_DEPTH
        self.learning_rate = self.config.model.PERPS_LEARNING_RATE
        
        logger.info("PerpsModelTrainer initialized")
    
    def load_training_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load training data from CSV"""
        if data_path is None:
            data_path = DATA_DIR / "perps" / "train_data.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} training samples from {data_path}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Define the 56 features used for perps prediction
        """
        # Core OHLCV features
        ohlcv_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Funding rate features
        funding_features = [
            'funding_rate', 'premium', 'next_funding_rate',
            'funding_rate_ma_8', 'funding_rate_std'
        ]
        
        # Market data features
        market_features = [
            'mark_price', 'index_price', 'open_interest',
            'volume_24h', 'oi_change_pct'
        ]
        
        # Technical indicator features
        tech_features = [
            'rsi', 'ema_12', 'ema_26', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
            'atr', 'adx', 'obv', 'obv_ema'
        ]
        
        # Momentum features
        momentum_features = [
            'returns_1h', 'returns_4h', 'returns_24h',
            'volatility_24h', 'high_low_range',
            'close_to_high', 'close_to_low'
        ]
        
        # Volume features
        volume_features = [
            'volume_ma_20', 'volume_ratio', 'volume_trend'
        ]
        
        # Derived features
        derived_features = [
            'price_momentum', 'trend_strength', 'mean_reversion_signal',
            'oi_momentum', 'funding_trend', 'spread',
            'hour_of_day', 'day_of_week', 'ticker_encoded'
        ]
        
        # Whale flow features (market-wide whale positioning)
        whale_flow_features = [
            'whale_flow_net', 'whale_flow_long_ratio', 'whale_flow_count_whales'
        ]
        
        # Combine all features
        all_features = (
            ohlcv_features + funding_features + market_features +
            tech_features + momentum_features + volume_features +
            derived_features + whale_flow_features
        )
        
        # Filter to available columns
        available = [f for f in all_features if f in df.columns]
        
        logger.info(f"Available features: {len(available)} / {len(all_features)}")
        
        return available
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from raw data"""
        df = df.copy()
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'funding_rate', 
                       'open_interest', 'mark_price', 'index_price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Price momentum
        if 'close' in df.columns:
            df['returns_1h'] = df['close'].pct_change(1)
            df['returns_4h'] = df['close'].pct_change(4)
            df['returns_24h'] = df['close'].pct_change(24)
            df['volatility_24h'] = df['returns_1h'].rolling(24).std()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['price_momentum'] = df['close'] / df['close'].shift(24) - 1
        
        # High/Low analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_to_high'] = (df['high'] - df['close']) / df['close']
            df['close_to_low'] = (df['close'] - df['low']) / df['close']
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_trend'] = df['volume'].diff(4)
        
        # Funding rate features
        if 'funding_rate' in df.columns:
            df['funding_rate_ma_8'] = df['funding_rate'].rolling(8).mean()
            df['funding_rate_std'] = df['funding_rate'].rolling(24).std()
            df['funding_trend'] = df['funding_rate'].diff(8)
        
        # Open interest features
        if 'open_interest' in df.columns:
            df['oi_change_pct'] = df['open_interest'].pct_change(1)
            df['oi_momentum'] = df['open_interest'].pct_change(24)
        
        # Spread
        if 'mark_price' in df.columns and 'index_price' in df.columns:
            df['spread'] = (df['mark_price'] - df['index_price']) / df['index_price']
        
        # RSI
        if 'close' in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if 'close' in df.columns:
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR
        if all(c in df.columns for c in ['high', 'low', 'close']):
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
        
        # Trend strength
        if 'ema_12' in df.columns and 'ema_50' in df.columns:
            df['trend_strength'] = (df['ema_12'] - df['ema_50']) / df['ema_50']
        
        # Mean reversion signal
        if 'close' in df.columns and 'bb_middle' in df.columns:
            df['mean_reversion_signal'] = (df['close'] - df['bb_middle']) / df['bb_middle']
        
        # Time features
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour_of_day'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
            except:
                df['hour_of_day'] = 12
                df['day_of_week'] = 3
        else:
            df['hour_of_day'] = 12
            df['day_of_week'] = 3
        
        # Encode ticker
        if 'ticker' in df.columns:
            df['ticker_encoded'] = self.ticker_encoder.fit_transform(df['ticker'].astype(str))
        else:
            df['ticker_encoded'] = 0
        
        return df
    
    def create_target(self, df: pd.DataFrame, lookahead: int = 4, threshold: float = 0.005) -> pd.DataFrame:
        """
        Create binary target for prediction
        
        Args:
            df: DataFrame with close prices
            lookahead: Number of periods to look ahead
            threshold: Minimum return to classify as LONG
            
        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()
        
        if 'close' not in df.columns:
            raise ValueError("'close' column required for target creation")
        
        # Future return
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        
        # Binary target: 1 = LONG (positive return above threshold), 0 = NOT_LONG
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Drop rows with NaN target
        df = df.dropna(subset=['target'])
        
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for training
        
        Returns:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature column names
        """
        # Engineer features
        df = self.engineer_features(df)
        
        # Create target if not present
        if 'target' not in df.columns:
            df = self.create_target(df)
        
        # Get feature columns
        feature_columns = self.get_feature_columns(df)
        
        if len(feature_columns) < 20:
            raise ValueError(f"Insufficient features: only {len(feature_columns)} available")
        
        X = df[feature_columns].values.astype(np.float32)
        y = df['target'].values.astype(np.int32)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_columns = feature_columns
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
        
        return X, y, feature_columns
    
    def train(
        self,
        data_path: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the perps prediction model
        
        Args:
            data_path: Path to training data CSV
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting perps model training...")
        start_time = datetime.now()
        
        # Load data
        df = self.load_training_data(data_path)
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df)
        
        # Time-based split (preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weight for imbalanced data
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        logger.info(f"Class distribution - Positive: {n_pos}, Negative: {n_neg}, Weight: {scale_pos_weight:.2f}")
        
        # Train XGBoost model
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
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Classification report
        report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        results = {
            'train_accuracy': round(train_accuracy * 100, 2),
            'test_accuracy': round(test_accuracy * 100, 2),
            'train_test_gap': round((train_accuracy - test_accuracy) * 100, 2),
            'n_features': len(feature_names),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'class_distribution': {
                'train_positive': int(n_pos),
                'train_negative': int(n_neg),
                'scale_pos_weight': round(scale_pos_weight, 2)
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'training_time_seconds': round(elapsed, 2),
            'feature_names': feature_names
        }
        
        logger.info(f"Training complete: {test_accuracy*100:.2f}% test accuracy in {elapsed:.1f}s")
        
        return results
    
    def save_model(self) -> Dict[str, str]:
        """Save trained model and artifacts"""
        if self.model is None:
            raise ValueError("No model to save - train first")
        
        model_path = self.model_dir / "perps_model.pkl"
        scaler_path = self.model_dir / "perps_scaler.pkl"
        features_path = self.model_dir / "perps_features.pkl"
        encoder_path = self.model_dir / "perps_ticker_encoder.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.ticker_encoder, f)
        
        logger.info(f"Model saved to {model_path}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'features_path': str(features_path),
            'encoder_path': str(encoder_path)
        }
    
    def load_model(self) -> bool:
        """Load existing trained model"""
        model_path = self.model_dir / "perps_model.pkl"
        scaler_path = self.model_dir / "perps_scaler.pkl"
        features_path = self.model_dir / "perps_features.pkl"
        encoder_path = self.model_dir / "perps_ticker_encoder.pkl"
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.ticker_encoder = pickle.load(f)
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        model_path = self.model_dir / "perps_model.pkl"
        
        return {
            'model_type': 'XGBoost Binary Classifier',
            'token_type': 'perps',
            'model_exists': model_path.exists(),
            'model_path': str(model_path),
            'n_features': len(self.feature_columns) if self.feature_columns else 56,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'target_description': 'LONG (1) vs NOT_LONG (0)'
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trainer = PerpsModelTrainer()
    print(trainer.get_model_info())
    
    # Test training if data exists
    try:
        results = trainer.train()
        print(f"\nTraining Results:")
        print(f"  Test Accuracy: {results['test_accuracy']}%")
        print(f"  Train-Test Gap: {results['train_test_gap']}%")
        
        trainer.save_model()
    except FileNotFoundError as e:
        print(f"Training data not found: {e}")
