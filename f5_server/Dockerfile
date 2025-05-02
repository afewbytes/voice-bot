# f5_server/Dockerfile.cuda  ───────────────────────────────────────────────
#  • Uses the official PyTorch image that already bundles CUDA 12.1 + cuDNN 8
#  • Torch/Torchaudio are GPU builds out-of-the-box, no extra wheels needed
#  • Everything else is identical to your CPU image
#  • Build       :  docker build -f f5_server/Dockerfile.cuda -t f5-tts-gpu .
#  • Run (Linux) :  docker run --gpus all -v $PWD/sockets:/app/sockets f5-tts-gpu
#                   ## the container creates f5-tts.sock inside /app/sockets

# ────────── 0. Base image ────────────────────────────────────────────────
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ────────── 1. Environment ───────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    F5_SOCKET=/app/sockets/f5-tts.sock

# ────────── 2. System libs ───────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git build-essential protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# ────────── 3. Python deps ───────────────────────────────────────────────
# torch + torchaudio are already installed in the base image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        xformers==0.0.24            \
        huggingface_hub grpcio grpcio-tools soundfile

# ---- F5-TTS -------------------------------------------------------------
ARG F5_COMMIT=09b478b            # tag 0.6.2  (2025-02-24)
RUN git clone --depth 1 --branch 0.6.2 \
        https://github.com/SWivid/F5-TTS.git /tmp/f5-tts && \
    pip install --no-cache-dir -e /tmp/f5-tts[eval] && \
    pip show f5-tts && echo "✓ F5-TTS 0.6.2 installed"

# ────────── 4. gRPC stubs ────────────────────────────────────────────────
WORKDIR /app
COPY ../proto/ ./proto/
RUN python -m grpc_tools.protoc \
        -I./proto \
        --python_out=. \
        --grpc_python_out=. \
        ./proto/voice.proto

# ────────── 5. Checkpoint + vocab ────────────────────────────────────────
ENV F5_CHECKPOINT_DIR=/app/checkpoints/F5TTS_v1_Base
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="SWivid/F5-TTS",
    allow_patterns=[
        "F5TTS_v1_Base/model_1250000.safetensors",
        "F5TTS_v1_Base/vocab.txt",
    ],
    local_dir="/app/checkpoints",
    local_dir_use_symlinks=False)
PY
ENV F5_VOCAB=${F5_CHECKPOINT_DIR}/vocab.txt

# ────────── 6. Server code ───────────────────────────────────────────────
COPY f5_server/f5_tts_server.py .

RUN mkdir -p /app/sockets

# ────────── 7. Entrypoint ────────────────────────────────────────────────
RUN cat <<'BASH' >/usr/local/bin/entrypoint && chmod +x /usr/local/bin/entrypoint
#!/usr/bin/env bash
set -e
echo "Starting **GPU** F5-TTS gRPC server on ${F5_SOCKET}"
exec python3 /app/f5_tts_server.py \
     --checkpoint "${F5_CHECKPOINT_DIR}" \
     --vocab      "${F5_VOCAB}" \
     --socket     "${F5_SOCKET}" \
     --device     cuda            # <-- only change needed
BASH

ENTRYPOINT ["/usr/local/bin/entrypoint"]