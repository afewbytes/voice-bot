cmake_minimum_required(VERSION 3.10)
project(whisper_server)

# -------------------------------------------------------------------
#  Dependencies
# -------------------------------------------------------------------
find_package(Protobuf      REQUIRED)
find_package(PkgConfig     REQUIRED)
find_package(Threads       REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
find_package(CUDAToolkit   REQUIRED)          # <- new

message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC     version: ${GRPC_VERSION}")
message(STATUS "CUDA toolkit   version: ${CUDAToolkit_VERSION}")

# -------------------------------------------------------------------
#  whisper.cpp / ggml (built as sub-project)
# -------------------------------------------------------------------
set(GGML_CUDA ON CACHE BOOL "Enable CUDA backend in ggml" FORCE)  # <- new
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp whisper_cpp_build)

# -------------------------------------------------------------------
#  Server target
# -------------------------------------------------------------------
add_executable(whisper_server
    whisper_server.cpp
    proto/voice.pb.cc
    proto/voice.grpc.pb.cc
)

target_compile_options(whisper_server PRIVATE -O3 -fPIC)
target_include_directories(whisper_server PRIVATE
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
)

target_link_libraries(whisper_server PRIVATE
    whisper           # from whisper.cpp
    ggml              # from whisper.cpp
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    Threads::Threads
    CUDA::cudart      # <- makes CUDA runtime symbols visible
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)