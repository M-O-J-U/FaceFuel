# FaceFuel API + web frontend, GPU image.
#
#   python scripts/collect_weights.py          # fills ./weights (7 files, ~135 MB)
#   docker build -t facefuel .
#   docker run --gpus all -p 8000:8000 facefuel
#
# CPU-only build:  docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t facefuel:cpu .
# See docs/DEPLOYMENT.md for hosts, env vars and mounting weights instead of baking them in.

FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    FACEFUEL_WEIGHTS_DIR=/app/weights \
    FACEFUEL_PORT=8000

# OpenCV (pulled in by ultralytics) and MediaPipe need these shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake DINOv2 ViT-S/14 into the image so the container starts without internet
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small')"
ENV HF_HUB_OFFLINE=1

COPY facefuel/ facefuel/
COPY static/ static/
COPY server.py .
COPY weights/ weights/

RUN useradd -m -u 1000 app && chown -R app /app /opt/hf
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:${FACEFUEL_PORT}/health || exit 1
CMD ["python", "server.py"]
