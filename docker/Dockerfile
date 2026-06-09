# Dockerfile 

# Build strategy: multi-stage 
#   stage 1: "builder" - installs all heavy build-time deps
#   stage 2: "runtime" - copies only what is need to run

# Result: a lean image that work on both arches.


# Stage 1: install python packages that need compilation (numpy, opencv, tflite)

FROM python:3.11-slim AS builder

# gcc/g++ : needed to compile C extensions of some pip packages
# libgomp1 : OpenMP runtime - required by numpy on ARM
# libglib2.0-0 : base system library required by opencv
# --no-install-recommends keeps the layer as small as possible
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    libglib2.0-0t64 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libwebp-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*
# work inside /install
WORKDIR /install

# Copy only requirement.txt
COPY docker/requirements.txt ./requirements.txt

# install all python deps into install/packages

RUN pip install --no-cache-dir --prefix=/install/packages \
    --extra-index-url https://www.piwheels.org/simple \
    -r requirements.txt 
# Stage 2: the actual image that runs on both arches

FROM python:3.11-slim AS runtime

# Runtime-only system libraries 
# libgomp1 : OpenMP - same reason as builder, needed at runtime
# libglib2.0-0 : opencv runtime dependency
# libgl1 : OpenGL - opencv needs this even in headless mode
# curl : useful for health checks inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0t64 \
    libopenblas0 \
    libjpeg62-turbo \
    libpng16-16t64 \
    libwebp7 \
    libavcodec61 \
    libavformat61 \
    libswscale8 \
    libavutil59 \
    libblas3 \
    python3-opencv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && cp -r /usr/lib/python3/dist-packages/cv2 /usr/local/lib/python3.11/site-packages/cv2

# Copy compiled packages from the builder stage
COPY --from=builder /install/packages /usr/local

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app
USER appuser

# Copy application code
COPY --chown=appuser:appuser model/train/edge_cctv_v2/weights/best.pt ./model/best.pt
COPY --chown=appuser:appuser app/ ./

# EXPOSE — catches missing
# RUN python -c "import cv2; import numpy; print('runtime libs OK')"

# Expose the inference API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Entry point # using only CMD to avoid entry point concat issue
CMD ["python", "inference.py", "--model", "./model/best.pt", "--port", "5000"]
