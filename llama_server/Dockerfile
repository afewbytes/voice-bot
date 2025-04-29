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

# ---- configure, build, install libs, copy executable -----------------------
RUN mkdir build && cd build && \
    cmake ../llama_server \
          -DCMAKE_BUILD_TYPE=Release \
          -DGGML_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
          -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    # until CMakeLists.txt has an install() line for the server binary:
    install -Dm755 llama_server /usr/local/bin/llama_server

########################
# 2. RUNTIME STAGE
########################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ---- bring in everything installed in /usr/local ---------------------------
COPY --from=build /usr/local /usr/local

# ---- copy shared libs that came from apt in the build stage ----------------
# (grpc++, protobuf, absl, etc.); keep them in /usr/local/lib to stay tidy
COPY --from=build /usr/lib/x86_64-linux-gnu/libgrpc++.so*          /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libgrpc++_reflection.so* /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libgrpc.so*            /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libprotobuf.so*        /usr/local/lib/
COPY --from=build /usr/lib/x86_64-linux-gnu/libabsl_*.so*          /usr/local/lib/

# ---- make the loader aware of /usr/local/lib -------------------------------
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama.conf && ldconfig

# ---- directory for the UNIX-domain socket ----------------------------------
WORKDIR /app
RUN mkdir -p /app/llama-sockets && chmod 777 /app/llama-sockets

# ---- launch the server ------------------------------------------------------
ENTRYPOINT ["/usr/local/bin/llama_server"]
CMD ["/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"]