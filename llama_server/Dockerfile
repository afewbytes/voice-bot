###############################################################################
# llama_server/Dockerfile – GPU build + runtime
###############################################################################

########################
# 1. BUILD STAGE
########################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG CUDA_ARCHS="89"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libcurl4-openssl-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

COPY llama_server/ /app/llama_server/
COPY proto/        /app/proto/
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so.1

RUN mkdir /app/llama_server/build && cd /app/llama_server/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

########################
# 2. RUNTIME STAGE
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app

COPY --from=build /app/llama_server/build/llama_server /app/
COPY --from=build /app/llama_server/build/llama_cpp_build/libllama.so*      /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libgrpc++.so*                   /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libprotobuf.so*                 /usr/local/lib/
RUN ldconfig

RUN mkdir -p /app/llama-sockets && chmod 777 /app/llama-sockets
CMD ["/app/llama_server", "/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]