#!/bin/bash
# Docker entrypoint script for Coin Market Analyzer
# Determines port based on ENVIRONMENT variable

set -e

# Determine port: use APP_PORT if set (e.g. from docker-compose), else by ENVIRONMENT
if [ -z "$APP_PORT" ]; then
    if [ "$ENVIRONMENT" = "development" ]; then
        APP_PORT=8900
        echo "🔧 Running in DEVELOPMENT mode on port $APP_PORT"
    else
        APP_PORT=7800
        echo "🚀 Running in PRODUCTION mode on port $APP_PORT"
    fi
else
    echo "🚀 Using APP_PORT from environment: $APP_PORT"
fi

# Export the port for any child processes
export APP_PORT

# Log startup info
echo "=================================================="
echo "Coin Market Analyzer"
echo "Environment: $ENVIRONMENT"
echo "Port: $APP_PORT"
echo "=================================================="

# Start the application
exec uvicorn api.app:app --host 0.0.0.0 --port $APP_PORT
