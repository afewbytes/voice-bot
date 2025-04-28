###############################################################################
# whisper/Dockerfile – GPU build + runtime (CUDA 12.4, Ubuntu 22.04)
###############################################################################

########################
# 1. BUILD STAGE
########################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG CUDA_ARCHS="89"                     # Ada Lovelace (RTX 4000 SFF Ada)
ARG DEBIAN_FRONTEND=noninteractive

# ── development tool-chain & headers ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libcurl4-openssl-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# ── clone whisper.cpp ────────────────────────────────────────────────────────
RUN git clone --depth 1 --branch v1.7.5 https://github.com/ggml-org/whisper.cpp.git

# ── server sources & proto schema ────────────────────────────────────────────
COPY whisper/ /app/
COPY proto/   /app/proto/
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

# ── link-time CUDA stub (both libcuda.so & libcuda.so.1) ─────────────────────
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so.1

# ── (optional) bake a Swedish base model in the image ───────────────────────
RUN mkdir -p /app/models && \
    wget -q -O /app/models/kb-ggml-model.bin \
         https://huggingface.co/KBLab/kb-whisper-base/resolve/main/ggml-model.bin

# ── configure & build ────────────────────────────────────────────────────────
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

########################
# 2. RUNTIME STAGE
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# -- copy server binary
WORKDIR /app
COPY --from=build /app/build/whisper_server /app/

# -- copy whisper / ggml libs
COPY --from=build /app/build/whisper_cpp_build/src/libwhisper.so*           /usr/local/lib/
COPY --from=build /app/build/whisper_cpp_build/ggml/src/libggml*.so*        /usr/local/lib/
COPY --from=build /app/build/whisper_cpp_build/ggml/src/libggml-cuda*.so*   /usr/local/lib/

# -- copy gRPC & Protobuf runtimes from build stage (exact matching versions)
COPY --from=build /usr/lib/x86_64-linux-gnu/libgrpc++.so*   /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libprotobuf.so* /usr/local/lib/
RUN ldconfig

# -- baked-in model (remove if you mount a volume)
COPY --from=build /app/models /app/models

# -- runtime directories
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

CMD ["/app/whisper_server"]