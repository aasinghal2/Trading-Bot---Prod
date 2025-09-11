# AI-Powered Trading Bot - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache  
# Force rebuild - minimal deps 2025-09-11
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data recommendations

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Performance optimizations for Railway deployment
ENV DEPLOYMENT_MODE=production
ENV RAILWAY_ENVIRONMENT=true
ENV SENTIMENT_FAST_MODE=true
ENV DISABLE_ML_MODELS=true
ENV CACHE_SENTIMENT_RESULTS=true
ENV PYTHON_GC_THRESHOLD=100
ENV MALLOC_ARENA_MAX=2

# Notification settings (Discord/SendGrid for Railway compatibility)
ENV EMAIL_NOTIFICATIONS=true
# Add DISCORD_WEBHOOK_URL in Railway variables for notifications

# Expose ports for web UI and health checks
EXPOSE 5000 8081

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start health check server in background\n\
python -c "from http.server import HTTPServer, BaseHTTPRequestHandler; import threading; import time\n\
class HealthHandler(BaseHTTPRequestHandler):\n\
    def do_GET(self):\n\
        if self.path == \"/health\":\n\
            self.send_response(200)\n\
            self.end_headers()\n\
            self.wfile.write(b\"OK\")\n\
        else:\n\
            self.send_response(404)\n\
            self.end_headers()\n\
server = HTTPServer((\"0.0.0.0\", 8081), HealthHandler)\n\
threading.Thread(target=server.serve_forever, daemon=True).start()\n\
print(\"Health check server started on port 8081\")\n\
exec(open(\"/app/main.py\").read())" &\n\
\n\
# Start the Smart Trader (SWING TRADER MODE)\n\
exec python scripts/smart_trader.py --run\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]