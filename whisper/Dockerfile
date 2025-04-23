FROM ubuntu:22.04

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pkg-config \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone ggml-org whisper.cpp and checkout a stable tag (optional)
RUN git clone https://github.com/ggml-org/whisper.cpp.git /app/whisper.cpp && \
    cd /app/whisper.cpp && \
    git checkout v1.7.5

# Download model weights
RUN mkdir -p /app/models && \
    wget -O /app/models/ggml-base.en.bin \
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Copy server sources & proto definitions
COPY whisper/ /app/
COPY proto/ /app/proto/

# Generate gRPC bindings
RUN protoc --proto_path=/app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` /app/proto/whisper.proto

# Enable core dumps (optional)
RUN ulimit -c unlimited

# Build both whisper.cpp and the server in one step
RUN mkdir -p /app/build && cd /app/build && \
    cmake /app \
      -DWHISPER_NO_ACCEL=ON \
      -DGGML_DISABLE_CUDA=ON \
      -DGGML_DISABLE_METAL=ON \
      -DGGML_SIMD=NONE \
      -DWHISPER_USE_OPENMP=OFF && \
    make -j$(nproc)

# Prepare socket directory
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

# Launch the server
CMD ["/app/build/whisper_server"]