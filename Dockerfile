FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
# CPU-only PyTorch first (sentence-transformers needs it): the default wheel drags in ~2 GB of GPU libraries that are never used here
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the models at build time so the container starts ready to use
RUN python -m app.ml.train_classifier && python -m app.ml.train_anomaly && python -m app.ml.train_extractor

# Pre-download the free embedding model into the image so document search needs no internet at run time. The build does not
# fail if Hugging Face is unreachable: the model is then fetched on first use (or set OPSPILOT_RAG_EMBEDDING_LOCAL_FILES_ONLY).
ENV HF_HOME=/opt/hf-cache
RUN python -m scripts.download_embedding_model || echo "embedding model not pre-downloaded; it will be fetched on first use"

# The database is MySQL, configured at run time through OPSPILOT_DB_* environment variables (see docker-compose.yml
# and .env.example). No credentials are baked into the image.

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
