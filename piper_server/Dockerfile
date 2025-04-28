###############################################################################
# piper_server/Dockerfile – GPU Piper TTS + ONNX Runtime
###############################################################################

########################
# 1. BUILD STAGE
########################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG CUDA_ARCHS="89"
ARG ORT_VER=1.21.1
ENV ORT_PREFIX=onnxruntime-linux-x64-gpu-cuda12-${ORT_VER}
ENV ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PREFIX}.tgz
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libsndfile1-dev libespeak-ng-dev \
        libopus-dev libopusfile-dev libsamplerate0-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/rhasspy/piper.git

COPY piper_server/ /app/piper_server/
COPY proto/        /app/proto/
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so.1

# compile Piper (downloads ORT-GPU tarball)
RUN cmake -S /app/piper -B /app/piper/build \
        -DONNXRUNTIME_PREFIX=${ORT_PREFIX} \
        -DONNXRUNTIME_URL=${ORT_URL} \
        -DCMAKE_INSTALL_PREFIX=/app/piper/build/install && \
    cmake --build /app/piper/build --config Release -j$(nproc)

# compile piper_server
RUN mkdir /app/piper_server/build && cd /app/piper_server/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

########################
# 2. RUNTIME STAGE
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app

COPY --from=build /app/piper_server/build/piper_server /app/
COPY --from=build /app/piper/build/pi/lib/libpiper_phonemize.so* /usr/local/lib/
COPY --from=build /app/piper/build/pi/lib/libonnxruntime.so*     /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libgrpc++.so*        /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libprotobuf.so*      /usr/local/lib/
RUN ldconfig

RUN mkdir -p /app/piper-sockets /app/models && chmod 777 /app/piper-sockets
CMD ["/app/piper_server", "/app/models/sv_SE-nst-medium.onnx", "/app/models/sv_SE-nst-medium.onnx.json"]