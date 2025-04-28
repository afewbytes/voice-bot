cmake_minimum_required(VERSION 3.10)
project(llama_server)

# -------------------------------------------------------------------
#  Dependencies
# -------------------------------------------------------------------
find_package(Protobuf      REQUIRED)
find_package(PkgConfig     REQUIRED)
find_package(Threads       REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
find_package(CUDAToolkit   REQUIRED)

message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC     version: ${GRPC_VERSION}")
message(STATUS "CUDA toolkit   version: ${CUDAToolkit_VERSION}")

# -------------------------------------------------------------------
#  llama.cpp sub-project (GPU enabled)
# -------------------------------------------------------------------
set(GGML_CUDA ON CACHE BOOL "Enable CUDA backend in ggml" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp llama_cpp_build)

# -------------------------------------------------------------------
#  Server executable
# -------------------------------------------------------------------
add_executable(llama_server
    llama_server.cpp
    ../proto/voice.pb.cc
    ../proto/voice.grpc.pb.cc
)

target_compile_options(llama_server PRIVATE -O3 -fPIC)
set_target_properties(llama_server PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

target_include_directories(llama_server PRIVATE
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/common
)

target_link_libraries(llama_server PRIVATE
    llama
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    Threads::Threads
    CUDA::cudart
)