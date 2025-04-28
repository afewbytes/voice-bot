# CUDA toolkit + build tools
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}

# -------- dependencies -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config wget \
        libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc \
    && rm -rf /var/lib/apt/lists/*

# -------- sources ------------------------------------------------------------
WORKDIR /app

# whisper.cpp (tag v1.7.5 ≙ server API used here)
RUN git clone --depth 1 --branch v1.7.5 https://github.com/ggml-org/whisper.cpp.git

# Swedish KB-Lab base model (change if you need another one)
RUN mkdir -p /app/models && \
    wget -q -O /app/models/kb-ggml-model.bin \
        https://huggingface.co/KBLab/kb-whisper-base/resolve/main/ggml-model.bin

# voice-bot gRPC server sources
COPY whisper/ /app/
COPY proto/   /app/proto/

# -------- gRPC code-gen ------------------------------------------------------
RUN protoc -I/app/proto \
      --cpp_out=/app/proto \
      --grpc_out=/app/proto \
      --plugin=protoc-gen-grpc=$(which grpc_cpp_plugin) \
      /app/proto/voice.proto

# -------- build --------------------------------------------------------------
RUN mkdir -p /app/build && cd /app/build && \
    cmake /app \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \           # <<< enable CUDA backend
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DWHISPER_USE_OPENMP=OFF && \
    make -j$(nproc)

# -------- runtime image ------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# small runtime deps
RUN apt-get update && apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /app/build/whisper_server /app/
COPY --from=build /app/models          /app/models
RUN mkdir -p /app/sockets && chmod 777 /app/sockets

EXPOSE 50051
CMD ["/app/whisper_server"]