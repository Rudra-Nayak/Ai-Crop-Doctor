FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into Docker image cache during build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code and assets
COPY backend/app/ app/
COPY backend/knowledge_base/ knowledge_base/
COPY backend/evaluation/ evaluation/
COPY frontend/ frontend/
COPY backend/.env.example .env.example

# Create runtime directories
RUN mkdir -p uploads logs knowledge_base/faiss_index

# Environment variables for 512MB RAM and dynamic PORT binding
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
