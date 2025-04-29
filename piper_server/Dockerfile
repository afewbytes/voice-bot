###############################################################################
# piper_server – one-stage CUDA build (Ubuntu 22.04 + CUDA 12.4.1)
###############################################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

#######################################################################
# 0. build/run-time prerequisites
#######################################################################
ARG  DEBIAN_FRONTEND=noninteractive
ARG  CUDA_ARCHS=89
ARG  ORT_VER=1.21.1

# → pre-built ONNX Runtime GPU tarball that matches CUDA 12
ENV  ORT_PREFIX=onnxruntime-linux-x64-gpu-cuda12-${ORT_VER}
ENV  ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PREFIX}.tgz

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libsndfile1-dev libespeak-ng-dev \
        libopus-dev libopusfile-dev libsamplerate0-dev \
        libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

#######################################################################
# 1. extra runtime libs: piper-phonemize + espeak-data
#######################################################################
RUN mkdir -p /tmp/pp && \
    cd /tmp/pp && \
    wget -q https://github.com/rhasspy/piper-phonemize/releases/download/v1.1.0/libpiper_phonemize-amd64.tar.gz && \
    tar -xf libpiper_phonemize-amd64.tar.gz && \
    cp -a lib/. /usr/local/lib/ && \
    cp -a lib/espeak-ng-data /usr/share/ && \
    ldconfig && cd / && rm -rf /tmp/pp

#######################################################################
# 2. source code
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
# 4. ONNX Runtime GPU download (no build needed)
#######################################################################
RUN wget -q ${ORT_URL} && tar -xzf ${ORT_PREFIX}.tgz && \
    mv ${ORT_PREFIX} /opt/onnxruntime

#######################################################################
# 5. build Piper (downloads nothing now – we already have ORT)
#######################################################################
RUN cmake -S /app/piper -B /app/piper/build \
        -DONNXRUNTIME_PREFIX=/opt/onnxruntime \
        -DCMAKE_INSTALL_PREFIX=/app/piper/build/install && \
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