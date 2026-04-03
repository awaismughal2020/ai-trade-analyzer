# Coin Market Analyzer

Unified trading signal API for **Meme Tokens** and **Perpetual Futures (Perps)**.

## Features

- **Multi-Layer Analysis**: ML Model, Whale Engine, Technical Indicators, Holder Metrics, User Profile
- **Dual Token Support**: Meme tokens (Solana) and Perps (HyperLiquid)
- **Training Endpoints**: Train and manage ML models via API
- **Data Pipeline**: Fetch and process training data
- **Swagger Documentation**: Full API documentation at `/docs`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp config/env.example .env
# Edit .env with your API keys
```

### 3. Run the API

```bash
chmod +x run_api.sh
./run_api.sh
```

Or manually:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/` | Get trading signal |
| GET | `/predict/{token_address}` | Get trading signal (GET method) |

**Example:**

```bash
# Meme token prediction
curl "http://localhost:8000/predict/TOKEN_ADDRESS?token_type=meme"

# Perps prediction
curl "http://localhost:8000/predict/BTC-USD?token_type=perps"
```

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/training/info/meme` | Get meme model info |
| GET | `/training/info/perps` | Get perps model info |
| POST | `/training/train/meme` | Train meme model |
| POST | `/training/train/perps` | Train perps model |
| POST | `/training/data-pipeline/perps` | Fetch perps training data |
| GET | `/training/data-info` | Get training data info |

**Example - Train Perps Model:**

```bash
curl -X POST "http://localhost:8000/training/train/perps" \
  -H "Content-Type: application/json" \
  -d '{"test_size": 0.2, "save_model": true}'
```

**Example - Fetch Training Data:**

```bash
curl -X POST "http://localhost:8000/training/data-pipeline/perps" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["BTC-USD", "ETH-USD"], "candle_limit": 2000}'
```

## Project Structure

```
coinMarketAnalyzer/
├── api/                    # FastAPI application
│   ├── app.py             # Main application
│   └── routes/            # API routes
│       ├── predict.py     # Prediction endpoints
│       └── training.py    # Training endpoints
├── config/                 # Configuration
│   ├── settings.py        # All settings
│   └── env.example        # Environment template
├── core/                   # Core data fetching
│   ├── data_fetcher.py    # Main data fetcher
│   └── data_fetcher_birdeye.py
├── engines/                # Analysis engines
│   ├── whale_engine.py    # Whale analysis
│   ├── technical_engine.py # Technical indicators
│   ├── holder_metrics.py  # Holder metrics
│   ├── entry_timing.py    # Entry/exit timing
│   └── ...
├── generators/             # Signal generators
│   ├── signal_generator.py
│   ├── summary_generator.py
│   └── layer_aggregator.py
├── training/               # Model training
│   ├── meme_trainer.py    # Meme model trainer
│   ├── perps_trainer.py   # Perps model trainer
│   └── data_pipeline/     # Data pipelines
├── models/                 # Trained models
│   ├── meme/              # Meme token models
│   └── perps/             # Perps models
├── data/                   # Training data
│   ├── meme/
│   └── perps/
├── services/               # External services
│   └── openai_service.py
├── tests/                  # Test files
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── run_api.sh
```

## Token Types

| Type | Description | Data Source |
|------|-------------|-------------|
| `meme` | Solana meme tokens | Internal API + Birdeye (see below) |
| `perps` | Perpetual futures | Internal API (HyperLiquid) |

### Meme prediction data sources

For meme tokens, data can be sourced with **Birdeye first** or **internal API first**:

- **`MEME_PRIMARY_DATA_SOURCE=birdeye`** (default): Uses Birdeye first (candles, trade data, holders, liquidity). If Birdeye is unavailable, times out, or returns no candles, the request falls back to the internal API. A time budget and per-call timeouts keep latency bounded (see `MEME_BIRDEYE_PRIMARY_TIMEOUT_SECONDS`, `MEME_BIRDEYE_PER_CALL_TIMEOUT_SECONDS`).
- **`MEME_PRIMARY_DATA_SOURCE=internal`**: Uses the internal mint API first; Birdeye is used only for candle fallback when internal data is stale and for liquidity.

The response `data_quality` and `candle_data_source` / `whale_data_source` fields indicate which source was used.

## Analysis Layers

| Layer | Weight | Description |
|-------|--------|-------------|
| ML Model | 30% | XGBoost binary classifier |
| Whale Engine | 25% | Large holder behavior |
| Technical | 10% | RSI, MACD, Bollinger Bands |
| Holder Metrics | 15% | Gini coefficient, concentration |
| User Profile | 20% | User trading history |

## ML Models

### Meme Model
- **Type**: XGBoost Binary Classifier
- **Features**: 36
- **Output**: BUY (1) / SELL (0)

### Perps Model
- **Type**: XGBoost Binary Classifier
- **Features**: 56
- **Output**: LONG (1) / NOT_LONG (0)
- **Accuracy**: 68%+ (target)

## Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INTERNAL_API_BASE_URL` | Internal API URL | http://52.3.148.51:3000 |
| `BIRDEYE_API_KEY` | Birdeye API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `PORT` | Server port | 8000 |
| `MEME_PRIMARY_DATA_SOURCE` | Meme data source order: `internal` or `birdeye` | `birdeye` |
| `MEME_BIRDEYE_PRIMARY_TIMEOUT_SECONDS` | Total time budget for Birdeye-primary fetch (seconds) | 28 |
| `MEME_BIRDEYE_PER_CALL_TIMEOUT_SECONDS` | Per-request timeout for each Birdeye call (seconds) | 12 |

## License

Proprietary - DroxLab
