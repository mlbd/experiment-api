FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables for model caching
ENV U2NET_HOME=/app/.u2net
ENV HOME=/app

# Create cache directory for rembg models
RUN mkdir -p /app/.u2net

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the isnet-general-use model (recommended for remove.bg quality)
# This prevents download on first request and speeds up cold starts
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

COPY app.py .

EXPOSE 5000

# Increased timeout for model loading and processing
# Workers=1 to prevent multiple model loads (saves memory)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--graceful-timeout", "120"]