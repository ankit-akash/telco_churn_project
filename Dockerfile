# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy dependency file first (better Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the complete project
COPY . .

# 6. Copy the trained mo+del artifacts
# NOTE: Replace the run ID below whenever you promote a new MLflow model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# 7. Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Start FastAPI
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]