FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=config/default.yaml

# Expose the API port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"] 