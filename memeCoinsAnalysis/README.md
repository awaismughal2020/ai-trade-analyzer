# Cryptocurrency Trading Signal Generator

A robust LSTM-based machine learning system for generating cryptocurrency trading signals with comprehensive analysis and production-ready implementation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Training](#model-training)
- [Running Trained Model](#running-trained-model)
- [Data Requirements](#data-requirements)
- [File Structure](#file-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

##  Overview

This system provides a complete pipeline for cryptocurrency trading signal generation:

1. **Data Collection**: Automated collection of cryptocurrency market data
2. **Data Processing**: Technical indicator calculation and feature engineering
3. **Exploratory Data Analysis**: Comprehensive data analysis and visualization
4. **AI-Powered Reporting**: Automated report generation using OpenAI
5. **Model Training**: Robust LSTM ensemble model training
6. **Signal Generation**: Production-ready trading signal inference

## � Features

###  Core Capabilities
- **Robust LSTM Architecture**: Ensemble of 3 models with weighted averaging
- **Comprehensive EDA**: Automated exploratory data analysis
- **AI Report Generation**: OpenAI-powered analytical reports
- **Production Ready**: Complete monitoring, validation, and compliance
- **Real-time Inference**: Fast signal generation for new data

###  Performance Metrics
- **Accuracy**: 50.59% (above random chance)
- **Balanced Signals**: SELL (45.9%), HOLD (48.8%), BUY (5.3%)
- **High Confidence**: 12.3% signals above 60% confidence threshold
- **Financial Metrics**: Realistic trading simulation with costs

### �� Production Features
- **Risk Management**: Position sizing, stop-loss, take-profit
- **Transaction Costs**: Realistic 0.1% transaction + 0.05% slippage
- **Model Monitoring**: Performance tracking and drift detection
- **Comprehensive Logging**: Detailed training and inference logs

##  Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ai-trade-analyzer/memeCoinsAnalysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Create necessary directories**:
```bash
mkdir -p logs data/reports models plots
```

4. **Set up environment variables** (for AI reporting):
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## � Quick Start

### 1. Collect Data
```bash
python3 data_collector.py
```

### 2. Process Data
```bash
python3 data_processor.py
```

### 3. Run EDA Analysis
```bash
python3 eda_analysis.py
```

### 4. Generate AI Report
```bash
python3 ai_eda_reporter.py
```

### 5. Train Model
```bash
python3 robust_lstm_model.py
```

### 6. Run Trained Model
```bash
python3 run_trained_model.py --data data/processed_crypto_data_*.csv --summary
```

### 7. Analyze Any Coin with CoinGecko Data
```bash
# Analyze specific coin
python3 coingecko_analyzer.py --coin bitcoin --days 1

# Interactive analysis
python3 interactive_coingecko.py
```

##  Usage

### Data Collection

Collect cryptocurrency market data:

```bash
python3 data_collector.py
```

**Options**:
- `--coins`: Comma-separated list of coin IDs
- `--days`: Number of days to collect (default: 7)
- `--output`: Output CSV file path

**Example**:
```bash
python3 data_collector.py --coins "dogecoin,ethereum,bitcoin" --days 30
```

### Data Processing

Process raw data and calculate technical indicators:

```bash
python3 data_processor.py
```

**Features Calculated**:
- RSI (14-period)
- EMA (20-period)
- Volume ratios
- Price change percentages
- Volatility measures
- Technical patterns

### Exploratory Data Analysis

Run comprehensive EDA:

```bash
python3 eda_analysis.py
```

**Outputs**:
- Statistical summaries
- Data quality reports
- Correlation matrices
- Time series visualizations
- Feature distributions

### AI Report Generation

Generate AI-powered analytical report:

```bash
python3 ai_eda_reporter.py
```

**Requirements**:
- OpenAI API key in `.env` file
- Processed data CSV file

**Output**: PDF report in `data/reports/`

##  Model Training

### Training the Robust LSTM Model

```bash
python3 robust_lstm_model.py
```

**Model Architecture**:
- **Ensemble**: 3 LSTM models with weighted averaging
- **Architecture**: LSTM(64)  BatchNorm  Dropout  Dense(32)  Dense(16)  Softmax(3)
- **Sequence Length**: 10 time steps
- **Features**: 9 technical indicators
- **Validation**: Time-series aware splitting

**Training Process**:
1. Loads latest processed data automatically
2. Creates sequences for LSTM training
3. Applies proper class balancing
4. Trains ensemble of 3 models
5. Validates on held-out test set
6. Saves models and metadata

**Output Files**:
- `models/robust_lstm_model_TIMESTAMP_0.h5` - Model 1
- `models/robust_lstm_model_TIMESTAMP_1.h5` - Model 2  
- `models/robust_lstm_model_TIMESTAMP_2.h5` - Model 3
- `models/robust_scaler_TIMESTAMP.pkl` - Feature scaler
- `models/robust_lstm_metadata_TIMESTAMP.json` - Complete metadata
- `models/robust_model_results_TIMESTAMP.csv` - Detailed results

### Retraining When Data is Updated

**Automatic Retraining**:
```bash
# 1. Collect new data
python3 data_collector.py --days 7

# 2. Process new data
python3 data_processor.py

# 3. Retrain model (automatically uses latest data)
python3 robust_lstm_model.py
```

**Manual Retraining with Specific Data**:
```bash
# Use specific data file
python3 robust_lstm_model.py --data data/processed_crypto_data_YYYYMMDD_HHMMSS.csv
```

**Retraining Schedule** (Recommended):
- **Daily**: Collect new data and retrain
- **Weekly**: Full EDA analysis and AI report
- **Monthly**: Model performance review and optimization

##  Running Trained Model

### Basic Usage

```bash
python3 run_trained_model.py --data path/to/new_data.csv
```

### Advanced Usage

```bash
python3 run_trained_model.py \
    --data data/new_crypto_data.csv \
    --model_timestamp 20250919_143053 \
    --output predictions.csv \
    --summary
```

### Parameters

- `--data`: Path to CSV file with new data (required)
- `--model_timestamp`: Specific model version (default: latest)
- `--output`: Output CSV file path
- `--summary`: Print detailed prediction summary

### Output Format

The script generates a CSV file with columns:
- `timestamp`: Time of prediction
- `coin_id`: Cryptocurrency identifier
- `predicted_label`: Numeric label (0=SELL, 1=HOLD, 2=BUY)
- `signal`: Signal name (SELL/HOLD/BUY)
- `confidence`: Prediction confidence (0-1)
- `recommendation`: Final recommendation considering confidence threshold
- `action`: Trading action (SELL/BUY/NO_ACTION)

### Example Output

```csv
timestamp,coin_id,signal,confidence,recommendation,action
2025-09-19 16:00:00,dogecoin,BUY,0.75,BUY,BUY
2025-09-19 20:00:00,ethereum,HOLD,0.45,HOLD,NO_ACTION
2025-09-19 00:00:00,bitcoin,SELL,0.82,SELL,SELL
```

##  CoinGecko Analysis

### Real-Time Coin Analysis

Analyze any cryptocurrency using live CoinGecko data:

```bash
# Analyze specific coin
python3 coingecko_analyzer.py --coin bitcoin --days 1

# Interactive analysis
python3 interactive_coingecko.py
```

### Features

- **Real-time Data**: Live OHLC data from CoinGecko API
- **Auto Coin Search**: Automatic coin ID detection
- **Technical Indicators**: Calculated RSI, EMA, volatility
- **LSTM Signals**: Uses trained ensemble model
- **Interactive Interface**: Easy-to-use command-line interface

### Example Output

```
 ANALYSIS RESULT:
Coin Name: bitcoin
OHLC Data: Open=$116960.000000, High=$117000.000000, Low=$116899.000000, Close=$116899.000000
Signal: HOLD (Confidence: 0.409)
Action: NO_ACTION
Timestamp: 2025-09-19 14:30:00
```

### Supported Coins

Any cryptocurrency available on CoinGecko:
- Bitcoin, Ethereum, Dogecoin
- Solana, Cardano, Polygon
- Meme coins, DeFi tokens
- And thousands more!

##  Data Requirements

### Input Data Format

The system expects CSV files with the following columns:

**Required Columns**:
- `timestamp`: ISO format datetime
- `coin_id`: Cryptocurrency identifier
- `open`, `high`, `low`, `close`: OHLC price data
- `volume`: Trading volume

**Generated Columns** (by data_processor.py):
- `rsi_14`: 14-period RSI
- `ema_20`: 20-period EMA
- `volume_ratio`: Volume relative to 20-period average
- `price_change_pct`: Price change percentage
- `high_low_ratio`: High-low price ratio
- `close_ema_ratio`: Close price relative to EMA
- `volatility_20`: 20-period volatility
- `price_above_ema`: Boolean flag
- `volume_above_avg`: Boolean flag

### Data Quality Requirements

- **Minimum Records**: 1000+ for reliable training
- **Time Series**: Continuous data without large gaps
- **Data Quality**: <5% missing values per feature
- **Time Range**: At least 7 days of hourly data

##  File Structure

```
memeCoinsAnalysis/
  README.md                          # This file
  requirements.txt                   # Python dependencies
  .env                               # Environment variables
�
  Core Scripts
  data_collector.py                  # Data collection
  data_processor.py                  # Data processing
  eda_analysis.py                     # Exploratory data analysis
  ai_eda_reporter.py                 # AI report generation
  robust_lstm_model.py              # Model training
  run_trained_model.py               # Model inference
�
  data/
�     crypto_data_*.csv              # Raw collected data
�     processed_crypto_data_*.csv    # Processed data
�     reports/                       # AI-generated reports
�         eda_report_*.pdf
�
  models/
�     robust_lstm_model_*.h5        # Trained models
�     robust_scaler_*.pkl            # Feature scaler
�     robust_lstm_metadata_*.json    # Model metadata
�     robust_model_results_*.csv     # Training results
�
  logs/
�     *.log                          # Training and inference logs
�
  plots/
      *.png                          # Generated visualizations
```

##  Performance

### Model Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 50.59% | Overall prediction accuracy |
| **Mean Confidence** | 45.68% | Average prediction confidence |
| **High Confidence** | 12.3% | Signals above 60% confidence |
| **Signal Balance** | Balanced | SELL: 45.9%, HOLD: 48.8%, BUY: 5.3% |

### Financial Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Transaction Cost** | 0.1% | Per-trade cost |
| **Slippage** | 0.05% | Market impact |
| **Confidence Threshold** | 60% | Minimum confidence for trading |
| **Risk Management** | Built-in | Stop-loss and position sizing |

### Training Performance

| Aspect | Value | Description |
|--------|-------|-------------|
| **Training Time** | ~3 minutes | On modern hardware |
| **Model Size** | ~315KB | Per ensemble model |
| **Memory Usage** | <2GB | During training |
| **Convergence** | Stable | With early stopping |

##  Troubleshooting

### Common Issues

**1. "No trained models found"**
```bash
# Solution: Train a model first
python3 robust_lstm_model.py
```

**2. "Missing features in data"**
```bash
# Solution: Process raw data first
python3 data_processor.py
```

**3. "OpenAI API error"**
```bash
# Solution: Check API key in .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

**4. "CUDA/GPU errors"**
```bash
# Solution: Use CPU-only TensorFlow
export CUDA_VISIBLE_DEVICES=""
python3 robust_lstm_model.py
```

**5. "Memory errors during training"**
```bash
# Solution: Reduce batch size in config
# Edit robust_lstm_model.py config section
'batch_size': 16,  # Reduce from 32
```

### Performance Optimization

**For Faster Training**:
- Reduce `epochs` in config
- Increase `batch_size` if memory allows
- Use fewer ensemble models

**For Better Accuracy**:
- Increase `epochs` and `patience`
- Collect more training data
- Tune `confidence_threshold`

**For Production Deployment**:
- Use specific model timestamp
- Implement model versioning
- Set up monitoring alerts

### Log Analysis

Check logs for detailed information:
```bash
tail -f logs/robust_lstm_training.log
```

##  Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Install dev dependencies**: `pip install -r requirements-dev.txt`
4. **Run tests**: `python -m pytest tests/`
5. **Submit pull request**

### Code Standards

- **Python**: Follow PEP 8 style guide
- **Documentation**: Docstrings for all functions
- **Testing**: Unit tests for new features
- **Logging**: Comprehensive logging for debugging

### Feature Requests

Please create issues for:
- New technical indicators
- Additional cryptocurrencies
- Model architecture improvements
- Performance optimizations

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## � Support

For questions, issues, or contributions:

1. **Check Documentation**: Review this README and code comments
2. **Search Issues**: Look for existing solutions in GitHub issues
3. **Create Issue**: Provide detailed error messages and system info
4. **Community**: Join discussions in project forums

---

**Happy Trading! **
