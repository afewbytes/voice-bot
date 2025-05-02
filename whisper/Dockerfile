###############################################################################
# whisper/Dockerfile – GPU build + runtime (CUDA 12.4, Ubuntu 22.04)
###############################################################################

########################
# 1. BUILD STAGE
########################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG CUDA_ARCHS="89"                     # Ada-Lovelace (RTX 4000 SFF Ada)
ARG DEBIAN_FRONTEND=noninteractive

# ── development tool-chain & headers ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libcurl4-openssl-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── clone whisper.cpp (CUDA-enabled) ─────────────────────────────────────────
RUN git clone --depth 1 --branch v1.7.5 https://github.com/ggml-org/whisper.cpp.git

# ── server sources & proto schema ────────────────────────────────────────────
COPY whisper/ /app/
COPY proto/   /app/proto/
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

# ── link-time CUDA stubs (libcuda.so → stubs) ────────────────────────────────
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so.1

# ── optional: bake a Swedish Whisper model into the image ────────────────────
#RUN mkdir -p /app/models && \
#    wget -q -O /app/models/kb-ggml-model.bin \
#         https://huggingface.co/KBLab/kb-whisper-base/resolve/main/ggml-model.bin

RUN mkdir -p /app/models && \
    wget -q -O /app/models/base.en-ggml.bin \
         https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# ── configure & build ────────────────────────────────────────────────────────
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j"$(nproc)"

# ── collect runtime DSOs (flat) ──────────────────────────────────────────────
RUN mkdir /deps && \
    # 1) everything built by whisper.cpp / ggml (whatever the names are)
    find /app/build -maxdepth 3 -type f -name '*.so*' -exec cp -v {} /deps/ \; && \
    # 2) every library ldd says whisper_server will need
    ldd /app/build/whisper_server | awk '/=> \// {print $3}' \
      | xargs -r -I{} cp -v {} /deps/

########################
# 2. RUNTIME STAGE
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ── server binary ────────────────────────────────────────────────────────────
COPY --from=build /app/build/whisper_server /app/

# ── all required DSOs straight into /usr/local/lib ───────────────────────────
COPY --from=build /deps/ /usr/local/lib/
RUN ldconfig                # refresh loader cache

# ── baked-in model (remove if you mount a volume instead) ────────────────────
COPY --from=build /app/models /app/models

# ── runtime directories ──────────────────────────────────────────────────────
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

CMD ["/app/whisper_server"]