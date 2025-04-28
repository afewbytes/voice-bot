###############################################################################
# BUILD STAGE – CUDA tool-chain, Piper, ONNX Runtime-GPU
###############################################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG DEBIAN_FRONTEND=noninteractive

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config wget \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libgomp1 libsndfile1-dev libespeak-ng-dev libopus-dev libopusfile-dev \
        libsamplerate0-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

# ── CUDA stub so the linker finds libcuda.so during image build ──────────────
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig

# ── sources ─────────────────────────────────────────────────────────────────
WORKDIR /app
RUN git clone --depth 1 https://github.com/rhasspy/piper.git

COPY piper_server/ /app/piper_server/
COPY proto/        /app/proto/

# ── gRPC bindings ───────────────────────────────────────────────────────────
RUN cd /app/proto && \
    protoc -I. --cpp_out=. --grpc_out=. \
          --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) voice.proto

# ── GPU-enabled ONNX Runtime -------------------------------------------------
ARG ORT_VER=1.21.1                     # match CUDA 12 build provided by ORT
ENV ORT_PREFIX=onnxruntime-linux-x64-gpu-cuda12-${ORT_VER}
ENV ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PREFIX}.tgz  [oai_citation_attribution:0‡GitHub](https://github.com/microsoft/onnxruntime/releases?utm_source=chatgpt.com)

# ── CUDA arch list (override at build-time if you move to another GPU) ──────
ARG CUDA_ARCHS="89"
ENV CUDA_ARCHS=${CUDA_ARCHS}

# ── build Piper (downloads ORT-GPU via the variables above) ─────────────────
WORKDIR /app/piper
RUN cmake -Bbuild \
          -DONNXRUNTIME_PREFIX=${ORT_PREFIX} \
          -DONNXRUNTIME_URL=${ORT_URL} \
          -DCMAKE_INSTALL_PREFIX=build/install && \
    cmake --build build --config Release -j$(nproc) && \
    cmake --install build

# ── build the gRPC server ----------------------------------------------------
WORKDIR /app/piper_server
RUN mkdir build && cd build && \
    cmake .. \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

###############################################################################
# RUNTIME STAGE – tiny image + CUDA runtime libs
###############################################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app
COPY --from=build /app/piper_server/build/piper_server /app/
COPY --from=build /app/piper/src/cpp/piper.hpp      /app/include/
COPY --from=build /app/piper/build/install          /usr/local/
COPY --from=build /usr/local/cuda/lib64/libcudart.so* /usr/local/lib/

RUN mkdir -p /app/piper-sockets /app/models && chmod 777 /app/piper-sockets

# Swedish medium voice (example)
# COPY or bind-mount your own model + json at runtime
CMD ["/app/piper_server", \
     "/app/models/sv_SE-nst-medium.onnx", \
     "/app/models/sv_SE-nst-medium.onnx.json"]