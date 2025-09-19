#!/usr/bin/env python3
"""
Simple Coin Analyzer - Individual Coin Trading Signal Generator

This script takes individual coin OHLC data and generates a trading signal
without processing the entire training dataset.

Usage:
    python3 simple_coin_analyzer.py
    # Then input coin data when prompted
"""

import os
import sys
import json
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

class SimpleCoinAnalyzer:
    """
    Simple coin analyzer for individual coin trading signals
    """
    
    def __init__(self):
        """Initialize with trained model"""
        
        self.models = []
        self.scaler = None
        self.metadata = {}
        self.feature_columns = []
        self.ensemble_weights = []
        
        # Load the trained model
        self.load_trained_model()
        
        print("Simple Coin Analyzer initialized successfully!")
    
    def find_latest_model(self) -> str:
        """Find the latest trained model timestamp"""
        
        models_dir = Path('models')
        metadata_files = list(models_dir.glob('robust_lstm_metadata_*.json'))
        
        if not metadata_files:
            raise FileNotFoundError("No trained models found in models/ directory")
        
        # Extract timestamps and find latest
        timestamps = []
        for file in metadata_files:
            parts = file.stem.split('_')
            if len(parts) >= 4:
                timestamp = f"{parts[-2]}_{parts[-1]}"
                timestamps.append(timestamp)
        
        return max(timestamps)
    
    def load_trained_model(self):
        """Load the trained model and metadata"""
        
        try:
            model_timestamp = self.find_latest_model()
            models_dir = Path('models')
            
            # Load metadata
            metadata_file = models_dir / f'robust_lstm_metadata_{model_timestamp}.json'
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            # Extract configuration and feature columns
            self.config = self.metadata['config']
            self.feature_columns = self.metadata['feature_columns']
            self.ensemble_weights = self.metadata.get('ensemble_weights', [1.0])
            
            # Load scaler
            scaler_file = models_dir / f'robust_scaler_{model_timestamp}.pkl'
            self.scaler = joblib.load(scaler_file)
            
            # Load ensemble models
            model_files = list(models_dir.glob(f'robust_lstm_model_{model_timestamp}_*.h5'))
            for model_file in sorted(model_files):
                model = load_model(model_file)
                self.models.append(model)
            
            print(f"Loaded model from {model_timestamp}")
            print(f"Features: {', '.join(self.feature_columns)}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please run 'python3 robust_lstm_model.py' first to train a model")
            sys.exit(1)
    
    def calculate_technical_indicators(self, ohlc_data: Dict) -> Dict:
        """Calculate technical indicators from OHLC data"""
        
        # Extract OHLC values
        open_price = float(ohlc_data['open'])
        high_price = float(ohlc_data['high'])
        low_price = float(ohlc_data['low'])
        close_price = float(ohlc_data['close'])
        volume = float(ohlc_data.get('volume', 1000000))  # Default volume if not provided
        
        # Calculate basic indicators
        price_change_pct = ((close_price - open_price) / open_price) * 100
        high_low_ratio = (high_price - low_price) / low_price
        
        # Simple EMA calculation (20-period approximation)
        ema_20 = close_price * 0.95 + open_price * 0.05  # Simplified EMA
        
        # RSI calculation (simplified)
        rsi_14 = 50.0  # Default neutral RSI
        
        # Volume ratio (simplified)
        volume_ratio = 1.0  # Default volume ratio
        
        # Volatility (simplified)
        volatility_20 = abs(price_change_pct) / 100
        
        # Price above EMA
        price_above_ema = 1 if close_price > ema_20 else 0
        
        # Volume above average (simplified)
        volume_above_avg = 1 if volume > 1000000 else 0
        
        # Close EMA ratio
        close_ema_ratio = close_price / ema_20
        
        return {
            'rsi_14': rsi_14,
            'ema_20': ema_20,
            'volume_ratio': volume_ratio,
            'price_change_pct': price_change_pct,
            'high_low_ratio': high_low_ratio,
            'close_ema_ratio': close_ema_ratio,
            'volatility_20': volatility_20,
            'price_above_ema': price_above_ema,
            'volume_above_avg': volume_above_avg
        }
    
    def create_sequence(self, indicators: Dict) -> np.ndarray:
        """Create a sequence for LSTM prediction"""
        
        # Create a simple sequence by repeating the indicators
        sequence_length = self.config['sequence_length']
        
        # Convert indicators to array
        feature_values = [indicators[col] for col in self.feature_columns]
        
        # Create sequence by repeating the same values (simplified approach)
        sequence = np.array([feature_values] * sequence_length)
        
        return sequence.reshape(1, sequence_length, len(self.feature_columns))
    
    def predict_signal(self, coin_name: str, ohlc_data: Dict) -> Dict:
        """Generate trading signal for individual coin"""
        
        try:
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(ohlc_data)
            
            # Create sequence
            X_sequence = self.create_sequence(indicators)
            
            # Normalize features
            X_scaled = self.scaler.transform(X_sequence.reshape(-1, len(self.feature_columns)))
            X_sequence_scaled = X_scaled.reshape(1, self.config['sequence_length'], len(self.feature_columns))
            
            # Get predictions from all models
            predictions = []
            for model in self.models:
                pred = model.predict(X_sequence_scaled, verbose=0)
                predictions.append(pred)
            
            # Ensemble prediction
            if len(self.models) > 1:
                ensemble_pred = np.zeros_like(predictions[0])
                for pred, weight in zip(predictions, self.ensemble_weights):
                    ensemble_pred += pred * weight
            else:
                ensemble_pred = predictions[0]
            
            # Get predicted class and confidence
            predicted_class = np.argmax(ensemble_pred[0])
            confidence = np.max(ensemble_pred[0])
            
            # Map to signal names
            signal_names = ['SELL', 'HOLD', 'BUY']
            signal = signal_names[predicted_class]
            
            # Determine trading action
            confidence_threshold = self.config['confidence_threshold']
            if confidence >= confidence_threshold:
                if signal == 'SELL':
                    action = 'SELL'
                elif signal == 'BUY':
                    action = 'BUY'
                else:
                    action = 'NO_ACTION'
            else:
                action = 'NO_ACTION'
            
            return {
                'coin_name': coin_name,
                'signal': signal,
                'confidence': confidence,
                'action': action,
                'indicators': indicators
            }
            
        except Exception as e:
            return {
                'coin_name': coin_name,
                'signal': 'ERROR',
                'confidence': 0.0,
                'action': 'ERROR',
                'error': str(e)
            }
    
    def format_output(self, result: Dict, ohlc_data: Dict = None) -> str:
        """Format the output in the requested format"""
        
        if 'error' in result:
            return f"Error: {result['error']}"
        
        ohlc_str = ""
        if ohlc_data:
            ohlc_str = f"Open={ohlc_data['open']}, High={ohlc_data['high']}, Low={ohlc_data['low']}, Close={ohlc_data['close']}"
        
        return f"""
ANALYSIS RESULT:
Coin Name: {result['coin_name']}
OHLC Data: {ohlc_str}
Signal: {result['signal']} (Confidence: {result['confidence']:.3f})
Action: {result['action']}
"""

def get_user_input():
    """Get coin data from user input"""
    
    print("\n" + "="*50)
    print("SIMPLE COIN ANALYZER")
    print("="*50)
    
    coin_name = input("Enter Coin Name: ").strip()
    if not coin_name:
        coin_name = "unknown-coin"
    
    print("\nEnter OHLC Data:")
    try:
        open_price = float(input("Open Price: "))
        high_price = float(input("High Price: "))
        low_price = float(input("Low Price: "))
        close_price = float(input("Close Price: "))
        
        volume_input = input("Volume (optional, press Enter for default): ").strip()
        volume = float(volume_input) if volume_input else 1000000
        
        return {
            'coin_name': coin_name,
            'ohlc_data': {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
        }
        
    except ValueError:
        print(" Invalid input. Please enter numeric values.")
        return None

def main():
    """Main function"""
    
    try:
        # Initialize analyzer
        analyzer = SimpleCoinAnalyzer()
        
        while True:
            # Get user input
            user_input = get_user_input()
            if user_input is None:
                continue
            
            # Generate signal
            result = analyzer.predict_signal(
                user_input['coin_name'], 
                user_input['ohlc_data']
            )
            
            # Display result
            print(analyzer.format_output(result, user_input['ohlc_data']))
            
            # Ask if user wants to continue
            continue_analysis = input("\nAnalyze another coin? (y/n): ").strip().lower()
            if continue_analysis not in ['y', 'yes']:
                break
        
        print("\nThanks for using Simple Coin Analyzer!")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
