"""
Main execution file for Meme Coin Market Analysis Model
Orchestrates data collection, processing, training, and evaluation
"""
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import our custom modules
from data_collector import CoinGeckoDataCollector
from data_processor import DataProcessor
from market_model import MarketAnalysisLSTM

# Load environment variables
load_dotenv()


def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'plots', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}/")


def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = [
        'COINGECKO_BASE_URL',
        'TARGET_COINS',
        'HISTORICAL_DAYS',
        'SEQUENCE_LENGTH'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        print("Please check your .env file")
        return False

    print("Environment variables validated")
    return True


def collect_data(use_cached=False):
    """
    Collect market data from CoinGecko API

    Args:
        use_cached (bool): Whether to use cached data if available

    Returns:
        pd.DataFrame: Raw market data
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)

    cached_file = "data/cached_raw_data.csv"

    # Check for cached data
    if use_cached and os.path.exists(cached_file):
        print(f"Loading cached data from {cached_file}...")
        raw_data = pd.read_csv(cached_file)
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
        print(f"Loaded {len(raw_data):,} cached records")
        return raw_data

    # Collect fresh data
    collector = CoinGeckoDataCollector()
    raw_data = collector.collect_all_coins()

    if raw_data.empty:
        print("No data collected!")
        return None

    # Cache the data
    collector.save_data(raw_data, "cached_raw_data.csv")
    return raw_data


def process_data(raw_data):
    """
    Process raw data and create training sequences

    Args:
        raw_data (pd.DataFrame): Raw market data

    Returns:
        tuple: (X, y) training data
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA PROCESSING")
    print("=" * 60)

    processor = DataProcessor()

    # Clean data
    print("\n--- Data Cleaning ---")
    clean_data = processor.clean_data(raw_data)
    if clean_data.empty:
        print("No data remaining after cleaning!")
        return None, None

    # Calculate technical indicators
    print("\n--- Technical Indicators ---")
    processed_data = processor.calculate_technical_indicators(clean_data)
    if processed_data.empty:
        print("No data remaining after indicator calculation!")
        return None, None

    # Create labels
    print("\n--- Label Creation ---")
    labeled_data = processor.create_labels(processed_data)

    # Save processed data
    processor.save_processed_data(labeled_data, "processed_data_with_labels.csv")

    # Create sequences
    print("\n--- Sequence Creation ---")
    X, y = processor.create_sequences(labeled_data)

    if len(X) == 0:
        print("No sequences created!")
        return None, None

    return X, y


def train_model(X, y):
    """
    Train the LSTM model

    Args:
        X, y: Training data

    Returns:
        MarketAnalysisLSTM: Trained model
    """
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)

    # Initialize model
    model = MarketAnalysisLSTM(n_features=X.shape[2])

    # Build architecture
    print("\n--- Building Model ---")
    model.build_model(
        lstm_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    # Prepare data
    print("\n--- Preparing Data ---")
    X_train, X_test, y_train, y_test = model.prepare_data(X, y, test_size=0.2)

    # Train model
    print("\n--- Training Model ---")
    history = model.train(X_train, y_train, X_test, y_test, verbose=1)

    # Evaluate model
    print("\n--- Evaluating Model ---")
    results = model.evaluate(X_test, y_test, plot_results=True)

    # Save model
    print("\n--- Saving Model ---")
    save_path = model.save_model()

    return model, results


def test_prediction(model, X):
    """
    Test prediction functionality

    Args:
        model: Trained model
        X: Input data for testing
    """
    print("\n" + "=" * 60)
    print("STEP 4: PREDICTION TESTING")
    print("=" * 60)

    # Test with a random sample
    sample_idx = np.random.randint(0, len(X))
    sample_sequence = X[sample_idx]

    print(f"Testing prediction with sample {sample_idx}...")
    prediction = model.predict(sample_sequence)

    print(f"Prediction Result:")
    print(f"- Signal: {prediction['signal']}")
    print(f"- Confidence: {prediction['confidence']:.3f}")
    print(f"- Probabilities:")
    for signal, prob in prediction['probabilities'].items():
        print(f"  {signal}: {prob:.3f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Meme Coin Market Analysis Model')
    parser.add_argument('--use-cached', action='store_true',
                        help='Use cached data instead of fetching new data')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and load existing model')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run prediction test with existing model')

    args = parser.parse_args()

    print("MEME COIN MARKET ANALYSIS MODEL")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    setup_directories()
    if not validate_environment():
        sys.exit(1)

    # Test-only mode
    if args.test_only:
        print("\nTEST-ONLY MODE")
        model = MarketAnalysisLSTM()
        if model.load_model():
            # Create dummy data for testing
            dummy_X = np.random.random((100, model.sequence_length, model.n_features))
            test_prediction(model, dummy_X)
        else:
            print("No trained model found!")
        return

    try:
        # Step 1: Data Collection
        raw_data = collect_data(use_cached=args.use_cached)
        if raw_data is None:
            print("Data collection failed!")
            return

        # Step 2: Data Processing
        X, y = process_data(raw_data)
        if X is None:
            print("Data processing failed!")
            return

        # Step 3: Model Training
        if args.skip_training:
            print("\nSKIPPING TRAINING - Loading existing model")
            model = MarketAnalysisLSTM(n_features=X.shape[2])
            if not model.load_model():
                print("Failed to load existing model!")
                return
        else:
            model, results = train_model(X, y)

        # Step 4: Test Prediction
        test_prediction(model, X)

        # Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total data points: {len(X):,}")
        print(f"Model ready for production use!")

        print("\nNext steps:")
        print("1. Test the model with: python main.py --test-only")
        print("2. Integrate into your trading application")
        print("3. Set up regular retraining schedule")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
