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
    pkg-config

# Set working directory
WORKDIR /app

# Copy whisper source code
COPY whisper/ .

# Copy proto files from the root-level proto directory
COPY proto/ /app/proto/

# Regenerate proto files with the installed protobuf version
RUN protoc --proto_path=/app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
    --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` /app/proto/whisper.proto

# Build
RUN mkdir -p build && cd build && cmake .. && make

RUN cp build/whisper_server .

CMD ["./whisper_server"]