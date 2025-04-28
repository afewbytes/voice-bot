###############################################################################
# Build stage –   CUDA tool-chain + dev packages
###############################################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

# -------- system deps --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config wget \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libgomp1 libomp-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# -------- source -------------------------------------------------------------
WORKDIR /app
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

COPY llama_server/ /app/llama_server/
COPY proto/        /app/proto/

# gRPC code-gen
RUN cd /app/proto && \
    protoc -I. --cpp_out=. --grpc_out=. \
           --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) voice.proto

###############################################################################
# Build llama.cpp + server
###############################################################################
# Which GPU archs to build for – override on the CLI if needed
ARG CUDA_ARCHS="89"
ENV CUDA_ARCHS=${CUDA_ARCHS}

# make the CUDA *stub* visible so linking works without a real driver
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig

RUN mkdir -p /app/llama_server/build && cd /app/llama_server/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        -DLLAMA_CURL=OFF && \
    make -j$(nproc)

###############################################################################
# Runtime stage –   tiny image with the CUDA libraries only
###############################################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app
COPY --from=build /app/llama_server/build/llama_server /app/
COPY --from=build /app/proto /app/proto
COPY --from=build /app/llama.cpp/common /app/llama.cpp/common
COPY --from=build /app/llama.cpp/libllama.so* /usr/local/lib/

# create socket dir at start-up
RUN mkdir -p /app/llama-sockets && chmod 777 /app/llama-sockets

# model is provided by a bind-mount or another volume
CMD ["/app/llama_server", "/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]