###############################################################################
# piper_server – CUDA 12.3 / cuDNN 9 runtime with Piper + ONNX Runtime 1.22.0
###############################################################################
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

#######################################################################
# 0. build/run-time prerequisites
#######################################################################
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="80;86;89"
ARG ORT_VER=1.22.0
ENV ORT_PREFIX=onnxruntime-linux-x64-gpu-${ORT_VER}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget pkg-config ninja-build \
        python3 python3-dev python3-pip \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libsndfile1-dev libespeak-ng-dev \
        libopus-dev libopusfile-dev libsamplerate0-dev \
        libspdlog-dev libfmt-dev \
    && rm -rf /var/lib/apt/lists/*

# ── modern CMake + psutil (for Piper build) ──────────────────────────
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.tar.gz \
 && tar -xzf cmake-3.29.0-linux-x86_64.tar.gz --strip-components=1 -C /usr/local \
 && rm cmake-3.29.0-linux-x86_64.tar.gz \
 && ln -sf /usr/local/bin/cmake /usr/bin/cmake \
 && python3 -m pip install --no-cache-dir psutil
# ─────────────────────────────────────────────────────────────────────

#######################################################################
# 1. runtime libs: piper-phonemize + eSpeak-NG voice data
#######################################################################
RUN mkdir -p /tmp/pp && cd /tmp/pp && \
    wget -q https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/libpiper_phonemize-amd64.tar.gz && \
    tar -xf libpiper_phonemize-amd64.tar.gz && \
    cp -a lib/. /usr/local/lib/ && \
    cp -a lib/espeak-ng-data /usr/share/ && ldconfig && \
    cd / && rm -rf /tmp/pp

#######################################################################
# 2. Piper + server sources
#######################################################################
WORKDIR /app
RUN git clone --depth 1 https://github.com/rhasspy/piper.git
COPY piper_server/ /app/piper_server/
COPY proto/        /app/proto/

#######################################################################
# 3. protobuf / gRPC stubs
#######################################################################
RUN protoc -I /app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
    --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) /app/proto/voice.proto

#######################################################################
# 4. drop-in ONNX Runtime – no compile needed
#######################################################################
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PREFIX}.tgz \
 && tar -xzf ${ORT_PREFIX}.tgz \
 && mv ${ORT_PREFIX} /opt/onnxruntime && \
    ln -s /opt/onnxruntime/lib/libonnxruntime.so /usr/local/lib/ && ldconfig

#######################################################################
# 5. build Piper (uses system fmt/spdlog, drop-in ORT)
#######################################################################
RUN cmake -S /app/piper -B /app/piper/build \
        -DONNXRUNTIME_PREFIX=/opt/onnxruntime \
        -DCMAKE_INSTALL_PREFIX=/app/piper/build/install \
        -DPIPER_USE_SYSTEM_FMT=ON \
        -DPIPER_USE_SYSTEM_SPDLOG=ON \
        -DPIPER_USE_SYSTEM_PHONEMIZE=ON && \
    cmake --build /app/piper/build --config Release -j$(nproc) && \
    cmake --install /app/piper/build

#######################################################################
# 6. build piper_server (CUDA enabled)
#######################################################################
WORKDIR /app/piper_server
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        -DONNXRUNTIME_PREFIX=/opt/onnxruntime && \
    make -j$(nproc)

#######################################################################
# 7. runtime housekeeping
#######################################################################
RUN mkdir -p /app/piper-sockets /app/models && chmod 777 /app/piper-sockets
RUN ldconfig

#######################################################################
# 8. entry-point
#######################################################################
CMD ["/app/piper_server/build/piper_server", \
     "/app/models/sv_SE-nst-medium.onnx", \
     "/app/models/sv_SE-nst-medium.onnx.json"]