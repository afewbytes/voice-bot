###############################################################################
# piper_server – single-stage CUDA build (Ubuntu 22.04 + CUDA 12.3.2 / cuDNN 9)
# Option B: build *piper-phonemize* from source so it links to ORT ${ORT_VER}
###############################################################################
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

#######################################################################
# 0. build / run-time prerequisites
#######################################################################
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="80;86;89"
ARG ORT_VER=1.21.1                # keep this in sync with the block below

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
    rm -rf /var/lib/apt/lists/*

# ── modern CMake + psutil (Piper needs ≥3.27) ────────────────────────
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.29.0/cmake-3.29.0-linux-x86_64.tar.gz \
 && tar -xzf cmake-3.29.0-linux-x86_64.tar.gz --strip-components=1 -C /usr/local \
 && rm cmake-3.29.0-linux-x86_64.tar.gz \
 && ln -sf /usr/local/bin/cmake /usr/bin/cmake \
 && python3 -m pip install --no-cache-dir psutil
# ─────────────────────────────────────────────────────────────────────

#######################################################################
# 1. clone Piper + server sources
#######################################################################
WORKDIR /app
RUN git clone --depth 1 https://github.com/rhasspy/piper.git
COPY piper_server/ /app/piper_server/
COPY proto/        /app/proto/

#######################################################################
# 2. generate gRPC stubs
#######################################################################
RUN protoc -I /app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
    --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) /app/proto/voice.proto

#######################################################################
# 3. build ONNX Runtime GPU **from source**
#######################################################################
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
        --allow_running_as_root && \
    # copy libs and headers for system-wide use
    cp build/Linux/Release/libonnxruntime*.so* /usr/local/lib/ && \
    cp -r include /usr/local && \
    # ── create flat-compat headers for projects that still do #include <onnxruntime_*.h> ──
    for h in onnxruntime_cxx_api.h onnxruntime_c_api.h; do \
        real=$(find /usr/local/include -name $h | head -n1); \
        ln -sf "${real}" "/usr/local/include/${h}"; \
    done && \
    # ────────────────────────────────────────────────────────────────────────────────
    ldconfig && \
    cd /app && rm -rf onnxruntime

#######################################################################
# 4. build piper-phonemize **from source** – now links to ORT ${ORT_VER}
#######################################################################
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/rhasspy/piper-phonemize.git && \
    cmake -S piper-phonemize -B piper-phonemize/build \
        -DONNXRUNTIME_DIR=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DPIPER_PHONEMIZE_BUILD_EXE=OFF && \
    cmake --build piper-phonemize/build -j$(nproc) && \
    cmake --install piper-phonemize/build && \
    ldconfig && \
    rm -rf piper-phonemize

#######################################################################
# 5. build Piper (uses the freshly-built ORT + phonemizer)
#######################################################################
RUN cmake -S /app/piper -B /app/piper/build \
        -DONNXRUNTIME_PREFIX=/usr/local \
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