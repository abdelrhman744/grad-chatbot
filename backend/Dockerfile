# syntax=docker/dockerfile:1

# ── Builder ──────────────────────────────────────────────────────────────
# Compiles/installs Python deps into an isolated venv so the compiler
# toolchain (build-essential) and pip's cache never end up in the final
# image.
FROM python:3.11.10-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
# `pip install torch` on Linux resolves to the default PyPI wheel, which
# is CUDA-enabled and drags in several GB of nvidia_* wheels
# (nvidia-cudnn, nvidia-cusparselt, ...) even though this image has no
# GPU access. Installing the CPU-only build from PyTorch's own index
# FIRST satisfies requirements.txt's unpinned `torch` line, so the
# later full install leaves it alone instead of pulling the CUDA build.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

# ── Runtime ──────────────────────────────────────────────────────────────
FROM python:3.11.10-slim-bookworm AS runtime

# ffmpeg          - audio ingestion (Whisper STT)
# tesseract-ocr   - OCR fallback for scanned PDFs/images
# poppler-utils   - pdf2image (PDF -> image rasterization for OCR)
# libgl1/libglib2.0-0 - required by opencv-python at import time
# curl            - used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY . .

# Model weights (sentence-transformers, cross-encoder, Whisper) are
# intentionally NOT baked into the image — they're downloaded on first
# run into /root/.cache, which docker-compose.yml mounts as a named
# volume so the download only happens once, not on every rebuild.

EXPOSE 8000

# Generous start_period: first boot loads the embedding + cross-encoder
# models (and downloads them if the model cache volume is empty).
# 127.0.0.1, not localhost — see frontend/Dockerfile for why (IPv6 ::1
# resolves first on some resolvers while the app only binds IPv4).
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -f http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
