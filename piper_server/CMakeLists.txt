cmake_minimum_required(VERSION 3.10)
project(piper_server)

# ── dependencies ────────────────────────────────────────────────────────────
find_package(Protobuf      REQUIRED)
find_package(PkgConfig     REQUIRED)
find_package(Threads       REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
find_package(CUDAToolkit   REQUIRED)

message(STATUS "Protobuf : ${Protobuf_VERSION}")
message(STATUS "gRPC     : ${GRPC_VERSION}")
message(STATUS "CUDA     : ${CUDAToolkit_VERSION}")

# ── build executable ────────────────────────────────────────────────────────
add_executable(piper_server
    piper_server.cpp
    ../proto/voice.pb.cc
    ../proto/voice.grpc.pb.cc
)

target_compile_options(piper_server PRIVATE -O3 -fPIC)
set_target_properties(piper_server PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

target_include_directories(piper_server PRIVATE
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto
)

target_link_libraries(piper_server PRIVATE
    onnxruntime
    piper_phonemize
    spdlog::spdlog
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    Threads::Threads
    CUDA::cudart
)