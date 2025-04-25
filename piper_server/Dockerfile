# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
  FROM ubuntu:22.04

  # -----------------------------------------------------------------------------
  # Install system dependencies
  # -----------------------------------------------------------------------------
  RUN apt-get update && apt-get install -y \
      build-essential cmake git \
      libgrpc++-dev \
      libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
      pkg-config libgomp1 \
      libsndfile1-dev libespeak-ng-dev \
      libopus-dev libopusfile-dev libsamplerate0-dev \
      libspdlog-dev \
      && rm -rf /var/lib/apt/lists/*
  
  # -----------------------------------------------------------------------------
  # Working directory
  # -----------------------------------------------------------------------------
  WORKDIR /app
  
  # -----------------------------------------------------------------------------
  # Clone Piper
  # -----------------------------------------------------------------------------
  RUN git clone --depth 1 https://github.com/rhasspy/piper.git /app/piper
  
  # -----------------------------------------------------------------------------
  # Copy piper_server and proto files
  # -----------------------------------------------------------------------------
  COPY piper_server/ /app/piper_server/
  COPY proto/        /app/proto/
  
  # -----------------------------------------------------------------------------
  # Generate gRPC C++ bindings for our proto file
  # -----------------------------------------------------------------------------
  RUN protoc --proto_path=/app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) /app/proto/voice.proto
  
  # -----------------------------------------------------------------------------
  # Build Piper for its dependencies (like ONNX Runtime)
  # -----------------------------------------------------------------------------
  WORKDIR /app/piper
  RUN cmake -Bbuild -DCMAKE_INSTALL_PREFIX=build/install && \
      cmake --build build --config Release && \
      cmake --install build
  
  # Check what was built
  RUN echo "Listing Piper build files:" && \
      find /app/piper/build -name "*.so" | grep -v "/temp/" && \
      echo "Listing Piper install files:" && \
      find /app/piper/build/install -type f | sort
  
  # -----------------------------------------------------------------------------
  # Build piper_server (directly including Piper's source files)
  # -----------------------------------------------------------------------------
  WORKDIR /app/piper_server
  
  # Copy all necessary Piper source files
  RUN mkdir -p src/piper && \
      cp /app/piper/src/cpp/piper.cpp src/piper/ && \
      cp /app/piper/src/cpp/piper.hpp src/piper/ && \
      # Copy any additional required headers
      cp -r /app/piper/src/cpp/phonemize src/piper/ || true
  
  # Copy the updated CMakeLists.txt
  COPY piper_server/CMakeLists.txt /app/piper_server/
  
  # Build piper_server
  RUN mkdir -p build && cd build && \
      cmake .. && \
      make -j$(nproc) VERBOSE=1
  
  # -----------------------------------------------------------------------------
  # Runtime setup
  # -----------------------------------------------------------------------------
  RUN mkdir -p /app/piper-sockets && chmod 777 /app/piper-sockets
  
  # -----------------------------------------------------------------------------
  # Command to start piper_server
  # -----------------------------------------------------------------------------
  CMD ["/app/piper_server/build/piper_server", \
       "/app/models/en_GB-alba-medium.onnx", \
       "/app/models/en_GB-alba-medium.onnx.json"]