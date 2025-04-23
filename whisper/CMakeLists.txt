cmake_minimum_required(VERSION 3.10)
project(whisper_server)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# Print versions for debugging
message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC version: ${GRPC_VERSION}")

# Set whisper.cpp directory - adjust path if needed
set(WHISPER_CPP_PATH "/app/whisper.cpp")
message(STATUS "Using whisper.cpp from: ${WHISPER_CPP_PATH}")

# Include directories
include_directories(
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
    ${WHISPER_CPP_PATH}/include
    ${WHISPER_CPP_PATH}/ggml/include
    ${WHISPER_CPP_PATH}/ggml/src
    ${WHISPER_CPP_PATH}/ggml/src/ggml-cpu
)

link_directories(
    ${GRPC_LIBRARY_DIRS}
    ${WHISPER_CPP_PATH}
)

# Include all .cpp and .c sources from whisper.cpp and ggml
file(GLOB WHISPER_SOURCES
    "${WHISPER_CPP_PATH}/src/*.cpp"
    "${WHISPER_CPP_PATH}/ggml/src/*.cpp"
    "${WHISPER_CPP_PATH}/ggml/src/*.c"
)

# Debug output to verify files found
message(STATUS "Whisper sources: ${WHISPER_SOURCES}")

# Define executable
add_executable(whisper_server
    whisper_server.cpp
    proto/whisper.pb.cc
    proto/whisper.grpc.pb.cc
    ${WHISPER_SOURCES}
)

# Link libraries
target_link_libraries(whisper_server
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    Threads::Threads
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)