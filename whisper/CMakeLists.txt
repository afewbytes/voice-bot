cmake_minimum_required(VERSION 3.10)
project(whisper_server)

# Find dependencies
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC version: ${GRPC_VERSION}")

# Add ggml-org whisper.cpp as a subproject (will build libggml & libwhisper)
set(WHISPER_CPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp")
add_subdirectory(${WHISPER_CPP_PATH} whisper_cpp_build)

# Disable GPU and SIMD in both whisper and ggml
add_compile_definitions(
    WHISPER_NO_ACCEL    # whisper.cpp flag
    GGML_DISABLE_CUDA   # disable CUDA backend
    GGML_DISABLE_METAL  # disable Metal backend
)

# Build our server
add_executable(whisper_server
    whisper_server.cpp
    proto/whisper.pb.cc
    proto/whisper.grpc.pb.cc
)

target_compile_options(whisper_server PRIVATE -O3 -fPIC)

target_include_directories(whisper_server PRIVATE
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
)

# Link against whisper and ggml libraries, plus gRPC/Protobuf
target_link_libraries(whisper_server PRIVATE
    whisper    # the libwhisper from add_subdirectory
    ggml       # the libggml from add_subdirectory
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    Threads::Threads
)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)