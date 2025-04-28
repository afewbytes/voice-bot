###############################################################################
# BUILD – whisper.cpp + gRPC server (CUDA)
###############################################################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG CUDA_ARCHS="89"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config wget \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 --branch v1.7.5 https://github.com/ggml-org/whisper.cpp.git

# server source + proto
COPY whisper/ /app/
COPY proto/   /app/proto/
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

# CUDA stub so linking works without host driver
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && ldconfig

# (optional) bake a Swedish model into the image
RUN mkdir -p /app/models && \
    wget -q -O /app/models/kb-ggml-model.bin \
        https://huggingface.co/KBLab/kb-whisper-base/resolve/main/ggml-model.bin

# build whisper + server
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

###############################################################################
# RUNTIME – tiny CUDA image + copied libs
###############################################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app

# ── copy server binary
COPY --from=build /app/build/whisper_server /app/

# ── copy whisper / ggml shared libs
COPY --from=build /app/build/whisper_cpp_build/src/libwhisper.so*               /usr/local/lib/
COPY --from=build /app/build/whisper_cpp_build/ggml/src/libggml*.so*            /usr/local/lib/
COPY --from=build /app/build/whisper_cpp_build/ggml/src/libggml-cuda*.so*       /usr/local/lib/

# ── copy gRPC + Protobuf runtime libs from the build stage
COPY --from=build /usr/lib/x86_64-linux-gnu/libgrpc++.so*   /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libprotobuf.so* /usr/local/lib/

RUN ldconfig

# baked-in model (delete if you mount your own)
COPY --from=build /app/models /app/models

# sockets dir
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

CMD ["/app/whisper_server"]