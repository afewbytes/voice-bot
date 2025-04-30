###############################################################################
# Stage 1: Build ONNX Runtime with CUDA support
###############################################################################
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 AS onnxruntime-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="80;86;89"
ARG ORT_VER=1.21.1

# Install build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget pkg-config ninja-build \
        python3 python3-dev python3-pip \
        g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    rm -rf /var/lib/apt/lists/*

# Install modern CMake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.tar.gz \
 && tar -xzf cmake-3.29.0-linux-x86_64.tar.gz --strip-components=1 -C /usr/local \
 && rm cmake-3.29.0-linux-x86_64.tar.gz \
 && ln -sf /usr/local/bin/cmake /usr/bin/cmake \
 && python3 -m pip install --no-cache-dir psutil

# Build ONNX Runtime from source
WORKDIR /onnxruntime-build
RUN git clone --branch v${ORT_VER} --depth 1 https://github.com/microsoft/onnxruntime && \
    cd onnxruntime && \
    ./build.sh \
        --config Release \
        --skip_tests \
        --parallel $(nproc) \
        --build_shared_lib \
        --use_cuda \
        --cuda_home=/usr/local/cuda \
        --cudnn_home=/usr/lib/x86_64-linux-gnu \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        --allow_running_as_root

# Organize ONNX Runtime files for the next stage
RUN mkdir -p /onnxruntime-dist/lib /onnxruntime-dist/include && \
    cp /onnxruntime-build/onnxruntime/build/Linux/Release/libonnxruntime*.so* /onnxruntime-dist/lib/ && \
    cp -r /onnxruntime-build/onnxruntime/include/* /onnxruntime-dist/include/

# Create flat-compat headers for projects that still do #include <onnxruntime_*.h>
RUN for h in onnxruntime_cxx_api.h onnxruntime_c_api.h; do \
        real=$(find /onnxruntime-dist/include -name $h | head -n1); \
        ln -sf "${real}" "/onnxruntime-dist/include/${h}"; \
    done

###############################################################################
# Stage 2: Build piper-phonemize with the built ONNX Runtime
###############################################################################
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 AS phonemize-builder

# Copy ONNX Runtime files from the previous stage
COPY --from=onnxruntime-builder /onnxruntime-dist /usr/local

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget pkg-config ninja-build \
        g++-12 \
        libespeak-ng-dev && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    rm -rf /var/lib/apt/lists/*

# Install modern CMake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.tar.gz \
 && tar -xzf cmake-3.29.0-linux-x86_64.tar.gz --strip-components=1 -C /usr/local \
 && rm cmake-3.29.0-linux-x86_64.tar.gz \
 && ln -sf /usr/local/bin/cmake /usr/bin/cmake

# Build piper-phonemize
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/rhasspy/piper-phonemize.git && \
    cmake -S piper-phonemize -B piper-phonemize/build \
        -DONNXRUNTIME_DIR=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DPIPER_PHONEMIZE_BUILD_EXE=OFF && \
    cmake --build piper-phonemize/build -j$(nproc) && \
    cmake --install piper-phonemize/build && \
    ldconfig

# Organize piper-phonemize files for the next stage
RUN mkdir -p /phonemize-dist/lib /phonemize-dist/include && \
    cp /usr/local/lib/libpiper_phonemize* /phonemize-dist/lib/ && \
    cp -r /usr/local/include/piper_phonemize /phonemize-dist/include/

###############################################################################
# Stage 3: Final image - build Piper and piper_server
###############################################################################
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="80;86;89"

# Copy built libraries from previous stages
COPY --from=onnxruntime-builder /onnxruntime-dist /usr/local
COPY --from=phonemize-builder /phonemize-dist /usr/local

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget pkg-config ninja-build \
        python3 python3-dev python3-pip \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libsndfile1-dev libespeak-ng-dev \
        libopus-dev libopusfile-dev libsamplerate0-dev \
        libspdlog-dev libfmt-dev \
        g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig

# Install modern CMake
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.tar.gz \
 && tar -xzf cmake-3.29.0-linux-x86_64.tar.gz --strip-components=1 -C /usr/local \
 && rm cmake-3.29.0-linux-x86_64.tar.gz \
 && ln -sf /usr/local/bin/cmake /usr/bin/cmake \
 && python3 -m pip install --no-cache-dir psutil

# Clone Piper + server sources
WORKDIR /app
RUN git clone --depth 1 https://github.com/rhasspy/piper.git
COPY piper_server/ /app/piper_server/
COPY proto/        /app/proto/

# Generate gRPC stubs
RUN protoc -I /app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
    --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) /app/proto/voice.proto

# Build Piper
RUN cmake -S /app/piper -B /app/piper/build \
        -DONNXRUNTIME_PREFIX=/usr/local \
        -DCMAKE_INSTALL_PREFIX=/app/piper/build/install \
        -DPIPER_USE_SYSTEM_FMT=ON \
        -DPIPER_USE_SYSTEM_SPDLOG=ON \
        -DPIPER_USE_SYSTEM_PHONEMIZE=ON && \
    cmake --build /app/piper/build --config Release -j$(nproc) && \
    cmake --install /app/piper/build

# Build piper_server
WORKDIR /app/piper_server
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        -DONNXRUNTIME_PREFIX=/usr/local && \
    make -j$(nproc)

# Runtime housekeeping
RUN mkdir -p /app/piper-sockets /app/models && chmod 777 /app/piper-sockets
RUN ldconfig

# Entry-point
CMD ["/app/piper_server/build/piper_server", \
     "/app/models/sv_SE-nst-medium.onnx", \
     "/app/models/sv_SE-nst-medium.onnx.json"]