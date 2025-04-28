FROM ubuntu:22.04

# 1) Install build deps (gRPC, protobuf, CMake, curl, etc)
RUN apt-get update && apt-get install -y \
      build-essential \
      cmake \
      git \
      libgrpc++-dev \
      libprotobuf-dev \
      protobuf-compiler \
      protobuf-compiler-grpc \
      pkg-config \
      libgomp1 \
      libomp-dev \
      libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Clone llama.cpp into /app/llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git /app/llama.cpp

# 3) Copy your server code & proto files
COPY llama_server/   /app/llama_server/
COPY proto/          /app/proto/

# 4) Generate C++ gRPC bindings for llama.proto
RUN cd /app/proto && \
    protoc \
      --proto_path=. \
      --cpp_out=. \
      --grpc_out=. \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
      voice.proto

# 5) Build llama_server (this will also add_subdirectory(/app/llama.cpp))
RUN mkdir -p /app/llama_server/build && \
    cd /app/llama_server/build && \
    cmake .. && \
    make -j$(nproc)

# 6) Pass model path directly as the first argument (no --model flag)
CMD ["/app/llama_server/build/llama_server", "/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]