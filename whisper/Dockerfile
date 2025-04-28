###############################################################################
# BUILD – compile whisper.cpp + gRPC server
###############################################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="89"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config wget \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 --branch v1.7.5 https://github.com/ggml-org/whisper.cpp.git

# ── server source + proto
COPY whisper/ /app/
COPY proto/   /app/proto/
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

# ── CUDA stub for linking
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && ldconfig

# ── optional: bake model into the image
RUN mkdir -p /app/models && \
    wget -q -O /app/models/kb-ggml-model.bin \
        https://huggingface.co/KBLab/kb-whisper-base/resolve/main/ggml-model.bin

# ── build
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

###############################################################################
# RUNTIME
###############################################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgrpc++1 libprotobuf32 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/build/whisper_server /app/
COPY --from=build /app/build/whisper_cpp_build/src/libwhisper.so*         /usr/local/lib/
COPY --from=build /app/build/whisper_cpp_build/ggml/src/libggml*.so*       /usr/local/lib/
COPY --from=build /app/build/whisper_cpp_build/ggml/src/libggml-cuda*.so*  /usr/local/lib/
RUN ldconfig

COPY --from=build /app/models /app/models
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

CMD ["/app/whisper_server"]