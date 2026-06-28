# syntax=docker/dockerfile:1
# Container image for the diabetes risk screening app.
# Used for local Docker, Hugging Face Spaces (Docker SDK), and Google Cloud Run.
FROM python:3.11-slim

# Avoid interactive prompts, keep Python output unbuffered for clean logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source (includes the committed model artifacts).
COPY . .

# Validate that the artifacts load in this image; train only if missing. This
# fails the build early if the model can't be loaded, rather than at runtime.
RUN python -c "from src.inference import load_model, load_metrics; load_model(); load_metrics()"

# Hugging Face Spaces and Cloud Run both inject $PORT; default to 8501 locally.
ENV PORT=8501
EXPOSE 8501

# Run as a non-root user (good practice; required by HF Spaces).
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Use shell form so $PORT is expanded at runtime.
CMD streamlit run streamlit_app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
