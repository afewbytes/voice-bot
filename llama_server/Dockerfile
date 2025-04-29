###############################################################################
# llama_server/Dockerfile – GPU build + runtime
###############################################################################

########################
# 1. BUILD STAGE
########################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ARG CUDA_ARCHS="89"
ARG DEBIAN_FRONTEND=noninteractive

# ---- system dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git wget pkg-config \
        libgomp1 \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
        libcurl4-openssl-dev libspdlog-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- source code ------------------------------------------------------------
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
COPY llama_server/ /app/llama_server/
COPY proto/        /app/proto/

# ---- generate protobuf / gRPC stubs ----------------------------------------
RUN protoc -Iproto --cpp_out=proto --grpc_out=proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) proto/voice.proto

# ---- allow CUDA linking without the driver present --------------------------
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so      /usr/local/cuda/lib64/libcuda.so.1

# ---- configure, build, install libs, copy binary ---------------------------
RUN mkdir build && cd build && \
    cmake ../llama_server \
          -DCMAKE_BUILD_TYPE=Release \
          -DGGML_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
          -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    # ----- temporarily install the executable by hand (until your CMakeLists.txt has an install() line)
    install -Dm755 llama_server /usr/local/bin/llama_server

########################
# 2. RUNTIME STAGE
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# copy everything that was installed in /usr/local during the build stage
COPY --from=build /usr/local /usr/local

# ensure the dynamic loader knows that directory
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama.conf && ldconfig

# directory for the UNIX-domain socket
WORKDIR /app
RUN mkdir -p /app/llama-sockets && chmod 777 /app/llama-sockets

# launch the server
ENTRYPOINT ["/usr/local/bin/llama_server"]
CMD ["/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]