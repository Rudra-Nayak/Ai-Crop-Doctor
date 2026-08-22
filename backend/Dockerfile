FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY knowledge_base/ knowledge_base/
COPY evaluation/ evaluation/
COPY .env.example .env.example

# Create directories
RUN mkdir -p uploads logs knowledge_base/faiss_index

# Environment variables for 512MB RAM and dynamic PORT binding

ENV PORT=8000
EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
