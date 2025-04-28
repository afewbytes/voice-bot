# f5_server/Dockerfile.cpu ────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# ────────── Environment ───────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    F5_SOCKET=/app/sockets/f5-tts.sock

# ────────── 1. System libs ────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git build-essential protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# ────────── 2. Python deps ────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 torchaudio==2.3.0

# ---- F5-TTS (old API with .tts) ------------------------------------------
ARG F5_COMMIT=09b478b            # tag 0.6.2  (2025-02-24)
RUN git clone --depth 1 --branch 0.6.2 \
        https://github.com/SWivid/F5-TTS.git /tmp/f5-tts && \
    pip install --no-cache-dir -e /tmp/f5-tts[eval] && \
    pip show f5-tts && echo "✓ F5-TTS 0.6.2 installed"

RUN pip install --no-cache-dir \
        huggingface_hub grpcio grpcio-tools soundfile

# ────────── 3. gRPC stubs ─────────────────────────────────────────────────
WORKDIR /app
COPY ../proto/ ./proto/
RUN python -m grpc_tools.protoc \
        -I./proto \
        --python_out=. \
        --grpc_python_out=. \
        ./proto/voice.proto

# ────────── 4. Checkpoint + vocab ─────────────────────────────────────────
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

# ────────── 5. Server code ────────────────────────────────────────────────
COPY f5_server/f5_tts_server.py .

RUN mkdir -p /app/sockets

# ────────── 6. Entrypoint ─────────────────────────────────────────────────
RUN cat <<'BASH' >/usr/local/bin/entrypoint && chmod +x /usr/local/bin/entrypoint
#!/usr/bin/env bash
set -e
echo "Starting CPU F5-TTS gRPC server on ${F5_SOCKET}"
exec python3 /app/f5_tts_server.py \
     --checkpoint "${F5_CHECKPOINT_DIR}" \
     --vocab      "${F5_VOCAB}" \
     --socket     "${F5_SOCKET}" \
     --device     cpu
BASH

ENTRYPOINT ["/usr/local/bin/entrypoint"]