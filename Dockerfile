FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the models at build time so the container starts ready to use
RUN python -m app.ml.train_classifier && python -m app.ml.train_anomaly && python -m app.ml.train_extractor

# The database is MySQL, configured at run time through OPSPILOT_DB_* environment variables (see docker-compose.yml
# and .env.example). No credentials are baked into the image.

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
