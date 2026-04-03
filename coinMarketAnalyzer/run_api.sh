#!/bin/bash

# Coin Market Analyzer - Run Script

# Change to script directory
cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set defaults
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "=========================================="
echo "Coin Market Analyzer"
echo "=========================================="
echo "Starting server on $HOST:$PORT"
echo ""

# Run the API
python3 -m uvicorn api.app:app --host $HOST --port $PORT --reload

