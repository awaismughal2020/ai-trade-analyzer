"""
Quick Start Script for Meme Coin Market Analysis
Tests all components and runs a sample analysis
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'requests': 'requests',
        'dotenv': 'python-dotenv',
        'ta': 'ta',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'joblib': 'joblib'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} missing")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("All dependencies satisfied!\n")
    return True


def setup_environment():
    """Setup directories and check environment"""
    print("Setting up environment...")

    # Create necessary directories
    directories = ['data', 'models', 'plots', 'logs', 'data/cache']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory ready: {directory}/")

    # Check environment variables
    required_vars = ['COINGECKO_BASE_URL', 'TARGET_COINS']
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {missing_vars}")
        print("Creating default T1.env file...")

        # Create minimal T1.env if it doesn't exist
        if not os.path.exists('T1.env'):
            default_env = """# Minimal configuration for testing
COINGECKO_BASE_URL=https://api.coingecko.com/api/v3
TARGET_COINS=dogecoin,shiba-inu,pepe
HISTORICAL_DAYS=30
SEQUENCE_LENGTH=20
PREDICTION_HORIZON=1
TRENDING_THRESHOLD=0.001
BATCH_SIZE=32
EPOCHS=10
VALIDATION_SPLIT=0.2
MODEL_SAVE_PATH=./models/meme_coin_market_model
API_RATE_LIMIT_SECONDS=1
USE_COOKING=false
"""
            with open('T1.env', 'w') as f:
                f.write(default_env)
            print("✅ Created default T1.env file")

            # Reload environment
            load_dotenv(override=True)

    print("Environment setup complete!\n")
    return True


def test_data_collection():
    """Test data collection module"""
    print("=" * 60)
    print("TESTING DATA COLLECTION")
    print("=" * 60)

    try:
        from data_collector import DataCollector

        collector = DataCollector()

        # Test connection
        print("Testing API connection...")
        if collector.test_connection():
            print("✅ API connection successful!")
        else:
            print("❌ API connection failed!")
            return False

        # Test fetching small amount of data
        print("\nFetching sample data for dogecoin (7 days)...")
        sample_data = collector.get_coin_data('dogecoin', days=7)

        if sample_data is not None and not sample_data.empty:
            print(f"✅ Fetched {len(sample_data)} records")
            print(f"   Columns: {list(sample_data.columns)}")
            print(f"   Date range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
            return True
        else:
            print("❌ Failed to fetch data")
            return False

    except Exception as e:
        print(f"❌ Error in data collection: {e}")
        return False


def test_data_processing():
    """Test data processing module"""
    print("\n" + "=" * 60)
    print("TESTING DATA PROCESSING")
    print("=" * 60)

    try:
        from data_processor import DataProcessor
        from data_collector import DataCollector

        processor = DataProcessor()
        collector = DataCollector()

        # Get sample data
        print("Fetching data for processing test...")
        raw_data = collector.get_coin_data('dogecoin', days=30)

        if raw_data is None or raw_data.empty:
            print("❌ No data to process")
            return False

        # Test cleaning
        print("Testing data cleaning...")
        clean_data = processor.clean_data(raw_data)
        print(f"✅ Cleaned {len(clean_data)} records")

        # Test technical indicators
        print("Testing technical indicator calculation...")
        processed_data = processor.calculate_technical_indicators(clean_data)

        if not processed_data.empty:
            print(f"✅ Calculated indicators for {len(processed_data)} records")

            # Check if indicators were calculated
            indicator_cols = ['rsi_14', 'ema_20', 'volume_ratio']
            available = [col for col in indicator_cols if col in processed_data.columns]
            print(f"   Available indicators: {available}")

            return True
        else:
            print("❌ Failed to calculate indicators")
            return False

    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False


def test_model_creation():
    """Test model creation and basic functionality"""
    print("\n" + "=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)

    try:
        from market_model import MarketAnalysisLSTM

        # Create model
        print("Creating LSTM model...")
        model = MarketAnalysisLSTM(n_features=9)

        # Build architecture
        print("Building model architecture...")
        model.build_model(
            lstm_units=[32, 16],  # Smaller for testing
            dropout_rate=0.2,
            learning_rate=0.001
        )

        print("✅ Model created successfully!")
        print(f"   Total parameters: {model.model.count_params():,}")

        # Test with dummy data
        print("\nTesting with dummy data...")
        dummy_X = np.random.random((100, 20, 9))  # 100 samples, 20 timesteps, 9 features
        dummy_y = np.random.randint(0, 3, 100)  # Random labels

        # Prepare data
        X_train, X_test, y_train, y_test = model.prepare_data(dummy_X, dummy_y)
        print("✅ Data preparation successful!")

        return True

    except Exception as e:
        print(f"❌ Error in model creation: {e}")
        return False


def test_meme_analyzer():
    """Test the meme coin analyzer wrapper"""
    print("\n" + "=" * 60)
    print("TESTING MEME COIN ANALYZER")
    print("=" * 60)

    try:
        from meme_coin_analyzer import MemeCoinAnalyzer

        # Initialize analyzer
        print("Initializing Meme Coin Analyzer...")
        analyzer = MemeCoinAnalyzer()

        # Test getting top meme coins
        print("\nFetching top 5 meme coins...")
        meme_coins = analyzer.get_top_meme_coins(limit=5)

        if meme_coins:
            print(f"✅ Found {len(meme_coins)} meme coins:")
            for coin in meme_coins[:3]:
                print(f"   - {coin['symbol']}: {coin['name']}")

            # Test analyzing a single coin
            print(f"\nTesting analysis on {meme_coins[0]['id']}...")
            analysis = analyzer.analyze_coin(
                coin_id=meme_coins[0]['id'],
                days=7,  # Short period for testing
                use_cached=False
            )

            if 'error' not in analysis:
                print(f"✅ Analysis completed successfully!")
                print(f"   Latest price: ${analysis['latest_price']:.6f}")
                print(f"   Market strength: {analysis['market_strength']}")
                if analysis['technical_indicators']['rsi']:
                    print(f"   RSI: {analysis['technical_indicators']['rsi']:.1f}")
                return True
            else:
                print(f"❌ Analysis failed: {analysis['error']}")
                return False
        else:
            print("❌ Failed to fetch meme coins")
            return False

    except Exception as e:
        print(f"❌ Error in meme analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_sample_analysis():
    """Run a complete sample analysis"""
    print("\n" + "=" * 60)
    print("RUNNING SAMPLE ANALYSIS")
    print("=" * 60)

    try:
        from meme_coin_analyzer import MemeCoinAnalyzer

        analyzer = MemeCoinAnalyzer()

        # Run analysis on top 3 meme coins
        print("Analyzing top 3 meme coins (7 days of data)...")
        results = analyzer.run_full_analysis(
            top_n=3,
            min_volume=50000,
            min_market_cap=500000,
            analysis_days=7
        )

        if not results.empty:
            print("\n✅ Sample analysis completed!")
            print(f"Analyzed {len(results)} coins")

            # Show summary
            print("\nTop opportunities found:")
            for i, opp in enumerate(analyzer.top_opportunities[:2], 1):
                print(f"{i}. {opp['coin']} (Score: {opp['score']})")

            return True
        else:
            print("❌ No results from analysis")
            return False

    except Exception as e:
        print(f"❌ Error in sample analysis: {e}")
        return False


def main():
    """Main testing function"""
    print("=" * 60)
    print("MEME COIN MARKET ANALYSIS - SYSTEM TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first!")
        return False

    # Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed!")
        return False

    # Run tests
    tests = [
        ("Data Collection", test_data_collection),
        ("Data Processing", test_data_processing),
        ("Model Creation", test_model_creation),
        ("Meme Analyzer", test_meme_analyzer),
        ("Sample Analysis", run_sample_analysis)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            time.sleep(1)  # Small delay between tests
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        print("\nNext steps:")
        print("1. Review and update the T1.env file with your API keys")
        print("2. Run full training: python main.py")
        print("3. Analyze meme coins: python meme_coin_analyzer.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Missing API key in T1.env file")
        print("- Rate limiting from API")
        print("- Network connectivity issues")
        print("- Insufficient data for analysis")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
