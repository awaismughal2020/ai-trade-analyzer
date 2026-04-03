# Coin Market Analyzer - Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p models/meme models/perps data/meme data/perps data/candles data/holders data/trades data/mints logs outputs

# Environment variable for port (default: 8000 for production)
# Set ENVIRONMENT=development for port 8900
ENV ENVIRONMENT=production
ENV APP_PORT=8000

# Expose both possible ports
EXPOSE 8000
EXPOSE 8900

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Run the application via entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
