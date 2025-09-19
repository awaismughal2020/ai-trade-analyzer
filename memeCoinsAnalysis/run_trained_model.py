#!/usr/bin/env python3
"""
Run Trained LSTM Model - Inference Script

This script loads a trained robust LSTM model and generates trading signals
for new cryptocurrency data.

Usage:
    python3 run_trained_model.py --data path/to/new_data.csv
    python3 run_trained_model.py --data path/to/new_data.csv --model_timestamp 20250919_143053
"""

import os
import sys
import json
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import ML libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TrainedModelRunner:
    """
    Load and run trained robust LSTM model for inference
    """
    
    def __init__(self, model_timestamp: str = None):
        """Initialize with trained model"""
        
        self.model_timestamp = model_timestamp
        self.models = []
        self.scaler = None
        self.metadata = {}
        self.feature_columns = []
        self.ensemble_weights = []
        
        # Setup logging
        self.setup_logging()
        
        # Load the trained model
        self.load_trained_model()
        
        self.logger.info("TrainedModelRunner initialized successfully")
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def find_latest_model(self) -> str:
        """Find the latest trained model timestamp"""
        
        models_dir = Path('models')
        metadata_files = list(models_dir.glob('robust_lstm_metadata_*.json'))
        
        if not metadata_files:
            raise FileNotFoundError("No trained models found in models/ directory")
        
        # Extract timestamps and find latest
        timestamps = []
        for file in metadata_files:
            # Extract timestamp from filename like "robust_lstm_metadata_20250919_143053.json"
            parts = file.stem.split('_')
            if len(parts) >= 4:
                timestamp = f"{parts[-2]}_{parts[-1]}"  # "20250919_143053"
                timestamps.append(timestamp)
        
        latest_timestamp = max(timestamps)
        self.logger.info(f"Found latest model timestamp: {latest_timestamp}")
        
        return latest_timestamp
    
    def load_trained_model(self):
        """Load the trained model and metadata"""
        
        if not self.model_timestamp:
            self.model_timestamp = self.find_latest_model()
        
        models_dir = Path('models')
        
        # Load metadata
        metadata_file = models_dir / f'robust_lstm_metadata_{self.model_timestamp}.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract configuration and feature columns
        self.config = self.metadata['config']
        self.feature_columns = self.metadata['feature_columns']
        self.ensemble_weights = self.metadata.get('ensemble_weights', [1.0])
        
        self.logger.info(f"Loaded metadata for model {self.model_timestamp}")
        self.logger.info(f"Feature columns: {self.feature_columns}")
        
        # Load scaler
        scaler_file = models_dir / f'robust_scaler_{self.model_timestamp}.pkl'
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        
        self.scaler = joblib.load(scaler_file)
        self.logger.info("Loaded scaler successfully")
        
        # Load ensemble models
        model_files = list(models_dir.glob(f'robust_lstm_model_{self.model_timestamp}_*.h5'))
        if not model_files:
            raise FileNotFoundError(f"No model files found for timestamp {self.model_timestamp}")
        
        for model_file in sorted(model_files):
            model = load_model(model_file)
            self.models.append(model)
            self.logger.info(f"Loaded model: {model_file.name}")
        
        self.logger.info(f"Loaded {len(self.models)} ensemble models")
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess new data for prediction"""
        
        self.logger.info(f"Preprocessing {len(df)} records")
        
        # Validate feature columns exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Clean data
        df_clean = df.dropna(subset=self.feature_columns)
        self.logger.info(f"After cleaning: {len(df_clean)} records")
        
        # Extract features
        X = df_clean[self.feature_columns].values
        
        # Normalize features using trained scaler
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_sequences = self.create_sequences(X_scaled)
        
        self.logger.info(f"Created {len(X_sequences)} sequences")
        
        return X_sequences, df_clean
    
    def create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM prediction"""
        
        sequences = []
        sequence_length = self.config['sequence_length']
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
        
        return np.array(sequences)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using ensemble models"""
        
        if not self.models:
            raise ValueError("No trained models available")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Ensemble prediction
        if len(self.models) > 1:
            # Weighted average of predictions
            ensemble_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.ensemble_weights):
                ensemble_pred += pred * weight
        else:
            ensemble_pred = predictions[0]
        
        # Get predicted classes and confidence
        predicted_classes = np.argmax(ensemble_pred, axis=1)
        confidence_scores = np.max(ensemble_pred, axis=1)
        
        return predicted_classes, confidence_scores
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for new data"""
        
        self.logger.info("Generating trading signals")
        
        # Preprocess data
        X_sequences, df_clean = self.preprocess_data(df)
        
        if len(X_sequences) == 0:
            self.logger.warning("No sequences created - insufficient data")
            return df_clean
        
        # Make predictions
        predictions, confidence_scores = self.predict(X_sequences)
        
        # Align predictions with original data
        df_results = df_clean.copy()
        df_results = df_results.iloc[self.config['sequence_length']:]  # Align with sequences
        
        # Add predictions
        df_results['predicted_label'] = predictions
        df_results['confidence'] = confidence_scores
        
        # Map labels to signal names
        signal_names = ['SELL', 'HOLD', 'BUY']
        df_results['signal'] = df_results['predicted_label'].map(lambda x: signal_names[x])
        
        # Add confidence-based recommendations
        df_results['recommendation'] = df_results.apply(
            lambda row: row['signal'] if row['confidence'] >= self.config['confidence_threshold'] else 'HOLD',
            axis=1
        )
        
        # Add trading action
        df_results['action'] = df_results.apply(
            lambda row: self.get_trading_action(row['recommendation'], row['confidence']),
            axis=1
        )
        
        self.logger.info("Signal generation completed")
        
        return df_results
    
    def get_trading_action(self, recommendation: str, confidence: float) -> str:
        """Get trading action based on recommendation and confidence"""
        
        if confidence < self.config['confidence_threshold']:
            return "NO_ACTION"
        
        action_map = {
            'SELL': 'SELL',
            'BUY': 'BUY',
            'HOLD': 'NO_ACTION'
        }
        
        return action_map.get(recommendation, 'NO_ACTION')
    
    def print_summary(self, df_results: pd.DataFrame):
        """Print prediction summary"""
        
        total_signals = len(df_results)
        high_confidence = len(df_results[df_results['confidence'] >= self.config['confidence_threshold']])
        
        print(f"\nPREDICTION SUMMARY")
        print(f"Total predictions: {total_signals}")
        print(f"High confidence signals: {high_confidence} ({high_confidence/total_signals*100:.1f}%)")
        print(f"Mean confidence: {df_results['confidence'].mean():.4f}")
        
        print(f"\nSIGNAL DISTRIBUTION:")
        signal_counts = df_results['signal'].value_counts()
        for signal, count in signal_counts.items():
            percentage = count / total_signals * 100
            print(f"  {signal}: {count} ({percentage:.1f}%)")
        
        print(f"\n� TRADING ACTIONS:")
        action_counts = df_results['action'].value_counts()
        for action, count in action_counts.items():
            percentage = count / total_signals * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")
        
        print(f"\n TOP 10 HIGH CONFIDENCE SIGNALS:")
        top_signals = df_results.nlargest(10, 'confidence')[['timestamp', 'coin_id', 'signal', 'confidence', 'action']]
        print(top_signals.to_string(index=False))
    
    def save_results(self, df_results: pd.DataFrame, output_file: str = None):
        """Save prediction results to CSV"""
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"predictions_{timestamp}.csv"
        
        df_results.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to {output_file}")
        
        return output_file

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Run trained LSTM model for cryptocurrency trading signals')
    parser.add_argument('--data', required=True, help='Path to CSV file with new data')
    parser.add_argument('--model_timestamp', help='Model timestamp (default: latest)')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--summary', action='store_true', help='Print detailed summary')
    
    args = parser.parse_args()
    
    try:
        # Load new data
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} records")
        
        # Initialize model runner
        print("Loading trained model...")
        runner = TrainedModelRunner(model_timestamp=args.model_timestamp)
        
        # Generate signals
        print("Generating trading signals...")
        results = runner.generate_signals(df)
        
        # Print summary
        if args.summary:
            runner.print_summary(results)
        
        # Save results
        output_file = runner.save_results(results, args.output)
        
        print(f"\nSUCCESS!")
        print(f"Generated signals for {len(results)} records")
        print(f"Results saved to: {output_file}")
        
        # Quick stats
        high_conf = len(results[results['confidence'] >= runner.config['confidence_threshold']])
        print(f"High confidence signals: {high_conf} ({high_conf/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
