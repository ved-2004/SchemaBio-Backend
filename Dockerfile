FROM python:3.11-slim

# System deps for ChromaDB (ONNX runtime), psycopg2, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the rest of the repo
COPY . .

# Cloud Run injects PORT; default to 8080
ENV PORT=8080

EXPOSE ${PORT}

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
