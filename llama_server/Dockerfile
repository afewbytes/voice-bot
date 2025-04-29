###############################################################################
# llama_server/Dockerfile – GPU build + runtime (self-verifying, all deps)
###############################################################################

########################
# 1. BUILD STAGE
########################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ARG CUDA_ARCHS="89"
ARG DEBIAN_FRONTEND=noninteractive

# ---- system tool-chain -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libcurl4-openssl-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- source -----------------------------------------------------------------
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
COPY llama_server/ /app/llama_server/
COPY proto/        /app/proto/

# ---- protobuf / gRPC stubs ---------------------------------------------------
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
    --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

# ---- CUDA stub links (builds without host driver) ---------------------------
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so.1

# ---- configure, build, install ---------------------------------------------
RUN mkdir build && cd build && \
    cmake ../llama_server \
          -DCMAKE_BUILD_TYPE=Release \
          -DGGML_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
          -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    # until CMakeLists has an install(TARGETS llama_server ...) line:
    install -Dm755 llama_server /usr/local/bin/llama_server

# ---- capture *all* runtime dependencies -------------------------------------
RUN ldd /usr/local/bin/llama_server | awk '/=>/ {print $3}' | sort -u > /tmp/needed.txt

########################
# 2. RUNTIME STAGE
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ---- copy executable + libllama + CUDA stub ---------------------------------
COPY --from=build /usr/local /usr/local

# ---- copy every remaining shared library the binary expects -----------------
RUN mkdir -p /usr/local/lib/deps
COPY --from=build /tmp/needed.txt /tmp/needed.txt
RUN set -e; \
    for so in $(cat /tmp/needed.txt); do \
        case "$so" in \
            /usr/local/*)          ;;  # already inside image                    \
            /lib/*|/usr/lib/*) cp -v "$so" /usr/local/lib/deps/ ;;               \
        esac; \
    done && \
    echo -e "/usr/local/lib\n/usr/local/lib/deps" > /etc/ld.so.conf.d/llama.conf && \
    ldconfig

# ---- socket directory -------------------------------------------------------
WORKDIR /app
RUN mkdir -p /app/llama-sockets && chmod 777 /app/llama-sockets

# ---- launch -----------------------------------------------------------------
ENTRYPOINT ["/usr/local/bin/llama_server"]
CMD ["/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]