FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src CLAIMGATE_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && python -m nltk.downloader -q -d /usr/local/share/nltk_data punkt punkt_tab

COPY src/ src/
COPY scripts/fetch_models.py scripts/

# model weights are not baked in: mount them (docker run -v "$PWD/models:/app/models") or fetch with scripts/fetch_models.py
EXPOSE 8060
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8060/health || exit 1
CMD ["uvicorn", "claimgate.serve:app", "--host", "0.0.0.0", "--port", "8060"]
