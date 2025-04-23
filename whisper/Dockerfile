FROM ubuntu:22.04

# Install dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone whisper.cpp
RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    make

# Create models directory and download a model
RUN mkdir -p /app/models
RUN wget -O /app/models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# Copy whisper source code
COPY whisper/ .

# Copy proto files from the root-level proto directory
COPY proto/ /app/proto/

# Generate proto files
RUN protoc --proto_path=/app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` /app/proto/whisper.proto

# Build
RUN mkdir -p build && cd build && \
    cmake .. && \
    make

# Create required directory for socket
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

CMD ["./build/whisper_server"]