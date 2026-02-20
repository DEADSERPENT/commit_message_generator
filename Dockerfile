# Multi-stage Dockerfile for the commit message generator API server.
#
# Build:
#   docker build -t commit-suggest-api .
#
# Run (mount your trained checkpoint):
#   docker run -p 8000:8000 \
#     -v $(pwd)/runs:/app/runs:ro \
#     -v $(pwd)/data:/app/data:ro \
#     commit-suggest-api
#
# Optional env vars:
#   CHECKPOINT   path to .pt file inside the container  (default: /app/runs/best.pt)
#   DEVICE       torch device string                     (default: cpu)
#   API_KEY      bearer token to protect the endpoint   (default: none)

FROM python:3.11-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install Python dependencies in two layers so the heavy ML layer is cached
# when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy application source
COPY src/      ./src/
COPY configs/  ./configs/
COPY server/   ./server/

# The model checkpoint and tokenizer are expected to be mounted at runtime.
# Create stub directories so the container starts without them (server logs
# a clear error if CHECKPOINT is missing).
RUN mkdir -p runs data && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app
ENV CHECKPOINT=/app/runs/best.pt
ENV DEVICE=cpu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
