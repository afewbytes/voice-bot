#############################################################################
#  f5_server/Dockerfile.cuda
#  --------------------------------------------------------------------------
#  • Base image already bundles CUDA 12.1, cuDNN 8, torch 2.3.0+cu121,
#    torchaudio 2.3.0+cu121 and torchvision 0.18.0+cu121
#  • We add our deps, build F5-TTS from source, then *re-pin* torchaudio
#    to make sure no CPU-only wheel sneaks in.
#############################################################################

# ────────── 0. Base image (CUDA 12.1 runtime + torch 2.3) ─────────────────
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ────────── 1. Environment ────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    F5_SOCKET=/app/f5-sockets/f5-tts.sock

# ────────── 2. System libs ────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git build-essential protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# ────────── 3. Python deps (first pass) ───────────────────────────────────
#     *Leave* torchaudio alone for now – the correct CUDA wheel is already
#     in the base image.  We install everything else we need.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        xformers==0.0.24 \
        huggingface_hub grpcio grpcio-tools soundfile

# ────────── 4. F5-TTS source build ---------------------------------------
ARG F5_COMMIT=09b478b            # tag v0.6.2  (2025-02-24)
RUN git clone --depth 1 --branch 0.6.2 \
        https://github.com/SWivid/F5-TTS.git /tmp/f5-tts && \
    pip install --no-cache-dir -e /tmp/f5-tts[eval] && \
    pip show f5-tts && echo "✓ F5-TTS 0.6.2 installed"

# ────────── 5. ***** RE-PIN TORCHAUDIO ***** ------------------------------
# Some of the previous installs may have *upgraded* torchaudio to a CPU wheel.
# Replace it with the exact CUDA-matched build:
RUN pip uninstall -y torchaudio && \
    pip install --no-cache-dir --force-reinstall \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torchaudio==2.3.0+cu121

# ────────── 6. gRPC stubs ─────────────────────────────────────────────────
WORKDIR /app
COPY ../proto/ ./proto/
RUN python -m grpc_tools.protoc \
        -I./proto \
        --python_out=. \
        --grpc_python_out=. \
        ./proto/voice.proto

# ────────── 7. Checkpoint + vocab ────────────────────────────────────────
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

# ────────── 8. Server code ────────────────────────────────────────────────
COPY f5_server/f5_tts_server.py .
RUN mkdir -p /app/f5-sockets

# ────────── 9. Entrypoint ────────────────────────────────────────────────
RUN cat <<'BASH' >/usr/local/bin/entrypoint && chmod +x /usr/local/bin/entrypoint
#!/usr/bin/env bash
set -e
echo "Starting **GPU** F5-TTS gRPC server on ${F5_SOCKET}"
exec python3 /app/f5_tts_server.py \
     --checkpoint "${F5_CHECKPOINT_DIR}" \
     --vocab      "${F5_VOCAB}" \
     --socket     "${F5_SOCKET}" \
     --device     cuda
BASH

ENTRYPOINT ["/usr/local/bin/entrypoint"]