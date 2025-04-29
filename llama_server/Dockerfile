###############################################################################
# llama_server/Dockerfile – one stage, CUDA enabled
###############################################################################

# CUDA 12.4 developer image (compiler + runtime in one)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ---- optional: pick your GPU’s compute capability (89 = Ada Lovelace) ------
ARG CUDA_ARCHS="89"
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. build dependencies (gRPC, protobuf, cmake, etc.)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libcurl4-openssl-dev libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. source code
# -----------------------------------------------------------------------------
WORKDIR /app
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
COPY llama_server/ /app/llama_server/
COPY proto/        /app/proto/

# -----------------------------------------------------------------------------
# 3. generate gRPC / Protobuf stubs
# -----------------------------------------------------------------------------
RUN protoc -Iproto \
           --cpp_out=proto \
           --grpc_out=proto \
           --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
           proto/voice.proto

# -----------------------------------------------------------------------------
# 4. allow linking against the CUDA stub (build works even if host driver
#    isn’t present at build-time; harmless if it is)
# -----------------------------------------------------------------------------
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so.1

# -----------------------------------------------------------------------------
# 5. configure & build (GPU enabled)
#    -DCMAKE_CUDA_ARCHITECTURES guarantees the binary only contains the CC you
#      need, which keeps the build fast and the binary smaller.
#    BUILD_SHARED_LIBS is ON by default; because we stay in the *same* image
#    there’s no need to switch to static linking.
# -----------------------------------------------------------------------------
RUN mkdir -p /app/llama_server/build && \
    cd /app/llama_server/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

# -----------------------------------------------------------------------------
# 6. place for the UNIX-domain socket that docker-compose mounts
# -----------------------------------------------------------------------------
RUN mkdir -p /app/llama-sockets && chmod 777 /app/llama-sockets

# -----------------------------------------------------------------------------
# 7. run
# -----------------------------------------------------------------------------
CMD ["/app/llama_server/build/llama_server", "/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]