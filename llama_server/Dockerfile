###############################################################################
# BUILD stage – compile llama.cpp + server
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
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

COPY llama_server/ /app/llama_server/
COPY proto/        /app/proto/

RUN protoc -I/app/proto --cpp_out=/app/proto --grpc_out=/app/proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) /app/proto/voice.proto

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so && \
    echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig

RUN mkdir /app/llama_server/build && cd /app/llama_server/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} && \
    make -j$(nproc)

###############################################################################
# RUNTIME stage
###############################################################################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app

COPY --from=build /app/llama_server/build/llama_server /app/
COPY --from=build /app/llama.cpp/libllama.so*          /usr/local/lib/
RUN ldconfig

RUN mkdir -p /app/llama-sockets && chmod 777 /app/llama-sockets
CMD ["/app/llama_server", "/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]