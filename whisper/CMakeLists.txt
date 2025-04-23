cmake_minimum_required(VERSION 3.10)
project(whisper_server)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# Print versions for debugging
message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC version: ${GRPC_VERSION}")

include_directories(
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
)

link_directories(${GRPC_LIBRARY_DIRS})

add_executable(whisper_server
    whisper_server.cpp
    proto/whisper.pb.cc
    proto/whisper.grpc.pb.cc
)

target_link_libraries(whisper_server
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
)