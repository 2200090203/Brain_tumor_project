FROM python:3.11-slim

# Install system dependencies required by pillow/opencv and tflite-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . /app

# Ensure models and static dirs exist (they may be populated in image or mounted)
RUN mkdir -p /app/models /app/static/uploads /app/static/results || true

# Expose the port Cloud Run expects
ENV PORT 8080
EXPOSE 8080

# Run with gunicorn for production
# Use 1 worker to keep memory usage predictable; tune as needed
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2"]
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache usage
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads static/results

# Set environment variables
ENV FLASK_APP=wsgi.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run gunicorn server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 wsgi:app