# -----------------------------------------------------------------------------
# Base image
# -----------------------------------------------------------------------------
  FROM ubuntu:22.04

  # -----------------------------------------------------------------------------
  # Install system dependencies
  # -----------------------------------------------------------------------------
  RUN apt-get update && apt-get install -y \
      build-essential cmake git wget \
      libgrpc++-dev \
      libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
      pkg-config libgomp1 \
      libsndfile1-dev libespeak-ng-dev \
      libopus-dev libopusfile-dev libsamplerate0-dev \
      libspdlog-dev \
      && rm -rf /var/lib/apt/lists/*
  
  # -----------------------------------------------------------------------------
  # Download and install piper_phonemize with its data files
  # -----------------------------------------------------------------------------
  RUN mkdir -p /tmp/piper_phonemize && \
      cd /tmp/piper_phonemize && \
      wget https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/libpiper_phonemize-amd64.tar.gz && \
      tar -xf libpiper_phonemize-amd64.tar.gz && \
      cp -r /tmp/piper_phonemize/lib/espeak-ng-data /usr/share/ && \
      rm -rf /tmp/piper_phonemize
  
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
      find /app/piper/build -name "*.so" | grep -v "/temp/" || true
  
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
      make -j$(nproc)
  
  # -----------------------------------------------------------------------------
  # Verify espeak-ng data files are installed correctly
  # -----------------------------------------------------------------------------
  RUN ls -la /usr/share/espeak-ng-data/ && \
      ls -la /usr/share/espeak-ng-data/phontab || echo "phontab file is missing"
  
  # -----------------------------------------------------------------------------
  # Runtime setup
  # -----------------------------------------------------------------------------
  RUN mkdir -p /app/piper-sockets && chmod 777 /app/piper-sockets
  RUN mkdir -p /app/models
  
  # -----------------------------------------------------------------------------
  # Command to start piper_server
  # -----------------------------------------------------------------------------
  CMD ["/app/piper_server/build/piper_server", \
       "/app/models/en_GB-alba-medium.onnx", \
       "/app/models/en_GB-alba-medium.onnx.json"]