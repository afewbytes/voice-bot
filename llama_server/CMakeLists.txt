# llama_server/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(llama_server)

# 1) find dependencies
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC version: ${GRPC_VERSION}")

# 2) add llama.cpp subproject
set(LLAMA_CPP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")
add_subdirectory(${LLAMA_CPP_PATH} llama_cpp_build)

# disable unwanted backends
add_compile_definitions(
    LLAMA_CURL=OFF
    LLAMA_CUBLAS=OFF
)

# 3) build the server executable
add_executable(llama_server
    llama_server.cpp
    ../proto/voice.pb.cc
    ../proto/voice.grpc.pb.cc
)

target_compile_options(llama_server PRIVATE -O3 -fPIC)
set_target_properties(llama_server PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# include dirs for proto headers and llama.cpp
target_include_directories(llama_server PRIVATE
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto
    ${LLAMA_CPP_PATH}
    ${LLAMA_CPP_PATH}/common 
)

# 4) link libraries
target_link_libraries(llama_server PRIVATE
    llama
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    Threads::Threads
)