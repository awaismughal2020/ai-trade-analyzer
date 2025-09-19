#!/usr/bin/env python3
"""
CoinGecko OHLC Data Fetcher

This script fetches OHLC data from CoinGecko API and analyzes it
using our trained LSTM model.

Usage:
    python3 coingecko_analyzer.py --coin bitcoin
    python3 coingecko_analyzer.py --coin ethereum --days 7
"""

import os
import sys
import json
import argparse
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

class CoinGeckoAnalyzer:
    """
    CoinGecko data fetcher and analyzer using trained LSTM model
    """
    
    def __init__(self):
        """Initialize with trained model"""
        
        self.models = []
        self.scaler = None
        self.metadata = {}
        self.feature_columns = []
        self.ensemble_weights = []
        
        # CoinGecko API base URL
        self.api_base = "https://api.coingecko.com/api/v3"
        
        # Load the trained model
        self.load_trained_model()
        
        print("CoinGecko Analyzer initialized successfully!")
    
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
    
    def search_coin_id(self, coin_name: str) -> str:
        """Search for coin ID by name"""
        
        try:
            url = f"{self.api_base}/search"
            params = {"query": coin_name}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['coins']:
                # Return the first match
                coin_id = data['coins'][0]['id']
                coin_symbol = data['coins'][0]['symbol']
                print(f"Found: {coin_name} -> {coin_id} ({coin_symbol})")
                return coin_id
            else:
                print(f"No coin found for: {coin_name}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"API Error: {str(e)}")
            return None
    
    def get_ohlc_data(self, coin_id: str, days: int = 1) -> List[Dict]:
        """Get OHLC data from CoinGecko"""
        
        try:
            url = f"{self.api_base}/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": days}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                print(f"No OHLC data available for {coin_id}")
                return []
            
            # Convert to our format
            ohlc_data = []
            for item in data:
                timestamp, open_price, high_price, low_price, close_price = item
                
                ohlc_data.append({
                    'timestamp': datetime.fromtimestamp(timestamp / 1000),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': 1000000  # Default volume (CoinGecko OHLC doesn't include volume)
                })
            
            print(f"Retrieved {len(ohlc_data)} OHLC data points for {coin_id}")
            return ohlc_data
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {str(e)}")
            return []
    
    def calculate_technical_indicators(self, ohlc_data: List[Dict]) -> List[Dict]:
        """Calculate technical indicators for OHLC data"""
        
        if len(ohlc_data) < 2:
            return []
        
        df = pd.DataFrame(ohlc_data)
        
        # Calculate basic indicators
        df['price_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
        
        # Simple EMA calculation (20-period approximation)
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Simple RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50)  # Default neutral RSI
        
        # Volume ratio (simplified)
        df['volume_ratio'] = 1.0
        
        # Volatility
        df['volatility_20'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        df['volatility_20'] = df['volatility_20'].fillna(0.01)
        
        # Price above EMA
        df['price_above_ema'] = (df['close'] > df['ema_20']).astype(int)
        
        # Volume above average (simplified)
        df['volume_above_avg'] = 1
        
        # Close EMA ratio
        df['close_ema_ratio'] = df['close'] / df['ema_20']
        
        return df.to_dict('records')
    
    def create_sequence(self, indicators: Dict) -> np.ndarray:
        """Create a sequence for LSTM prediction"""
        
        sequence_length = self.config['sequence_length']
        
        # Convert indicators to array
        feature_values = [indicators[col] for col in self.feature_columns]
        
        # Create sequence by repeating the same values (simplified approach)
        sequence = np.array([feature_values] * sequence_length)
        
        return sequence.reshape(1, sequence_length, len(self.feature_columns))
    
    def predict_signal(self, coin_name: str, indicators: Dict) -> Dict:
        """Generate trading signal for coin data"""
        
        try:
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
    
    def analyze_coin(self, coin_name: str, days: int = 1) -> Dict:
        """Complete analysis pipeline for a coin"""
        
        print(f"\nAnalyzing {coin_name}...")
        
        # Search for coin ID
        coin_id = self.search_coin_id(coin_name)
        if not coin_id:
            return {'error': f'Coin {coin_name} not found'}
        
        # Get OHLC data
        ohlc_data = self.get_ohlc_data(coin_id, days)
        if not ohlc_data:
            return {'error': f'No OHLC data available for {coin_name}'}
        
        # Calculate technical indicators
        indicators_data = self.calculate_technical_indicators(ohlc_data)
        if not indicators_data:
            return {'error': f'Insufficient data for {coin_name}'}
        
        # Use the latest data point for prediction
        latest_indicators = indicators_data[-1]
        
        # Generate signal
        result = self.predict_signal(coin_name, latest_indicators)
        
        # Add OHLC data to result
        latest_ohlc = ohlc_data[-1]
        result['ohlc_data'] = latest_ohlc
        result['ohlc_history'] = ohlc_data
        
        return result
    
    def format_output(self, result: Dict) -> str:
        """Format the output in the requested format"""
        
        if 'error' in result:
            return f"Error: {result['error']}"
        
        ohlc = result['ohlc_data']
        
        return f"""
ANALYSIS RESULT:
Coin Name: {result['coin_name']}
OHLC Data: Open=${ohlc['open']:.6f}, High=${ohlc['high']:.6f}, Low=${ohlc['low']:.6f}, Close=${ohlc['close']:.6f}
Signal: {result['signal']} (Confidence: {result['confidence']:.3f})
Action: {result['action']}
Timestamp: {ohlc['timestamp']}
"""

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Analyze cryptocurrency using CoinGecko data and trained LSTM model')
    parser.add_argument('--coin', required=True, help='Coin name to analyze (e.g., bitcoin, ethereum)')
    parser.add_argument('--days', type=int, default=1, help='Number of days of data to fetch (default: 1)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = CoinGeckoAnalyzer()
        
        # Analyze the coin
        result = analyzer.analyze_coin(args.coin, args.days)
        
        # Display result
        print(analyzer.format_output(result))
        
        # Show additional info if available
        if 'ohlc_history' in result and len(result['ohlc_history']) > 1:
            print(f"\nData Points Analyzed: {len(result['ohlc_history'])}")
            print(f"Price Range: ${min(h['low'] for h in result['ohlc_history']):.6f} - ${max(h['high'] for h in result['ohlc_history']):.6f}")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
