###############################################################################
# piper_server – single-stage CUDA build (Ubuntu 22.04 + CUDA 12.3.2 / cuDNN 9)
###############################################################################
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

#######################################################################
# 0. build / run-time prerequisites
#######################################################################
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="80;86;89"
ARG ORT_VER=1.21.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget pkg-config ninja-build \
        python3 python3-dev python3-pip \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libsndfile1-dev libespeak-ng-dev \
        libopus-dev libopusfile-dev libsamplerate0-dev \
        libspdlog-dev \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

#######################################################################
# 1. extra runtime libs: piper-phonemize + eSpeak-NG data
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
# 4. build ONNX Runtime GPU **from source**
#######################################################################
RUN git clone --branch v${ORT_VER} --depth 1 https://github.com/microsoft/onnxruntime && \
    cd onnxruntime && \
    ./build.sh \
        --config Release \
        --skip_tests \
        --parallel \
        --build_shared_lib \
        --build_cuda \
        --cuda_home=/usr/local/cuda \
        --cudnn_home=/usr/lib/x86_64-linux-gnu \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=\"${CUDA_ARCHS}\" && \
    cp build/Linux/Release/libonnxruntime*.so* /usr/local/lib/ && \
    ldconfig && \
    cd /app && rm -rf onnxruntime

#######################################################################
# 5. build Piper (links to the freshly-built ORT)
#######################################################################
RUN cmake -S /app/piper -B /app/piper/build \
        -DONNXRUNTIME_PREFIX=/usr/local \
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
        -DONNXRUNTIME_PREFIX=/usr/local && \
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