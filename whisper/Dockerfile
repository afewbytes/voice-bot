###############################################################################
# BUILD stage – compile whisper.cpp + server
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

# server sources + proto
COPY whisper/ /app/
COPY proto/   /app/proto/

# gRPC bindings
RUN protoc -I/app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) /app/proto/voice.proto

# build
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig

RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

###############################################################################
# RUNTIME stage
###############################################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app

# binary
COPY --from=build /app/build/whisper_server /app/

# libraries built in the first stage
COPY --from=build /app/build/libwhisper.so*        /usr/local/lib/
COPY --from=build /app/build/libggml*.so*          /usr/local/lib/
COPY --from=build /app/build/libggml-cuda*.so*     /usr/local/lib/
RUN ldconfig

# model + sockets
COPY --from=build /app/models /app/models
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

CMD ["/app/whisper_server"]