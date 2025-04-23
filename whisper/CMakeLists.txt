cmake_minimum_required(VERSION 3.10)
project(whisper_server)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# ───────────────────────────────────────────────────────
#  Versions (for your build logs)
# ───────────────────────────────────────────────────────
message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC version: ${GRPC_VERSION}")

# ───────────────────────────────────────────────────────
#  whisper.cpp location
# ───────────────────────────────────────────────────────
set(WHISPER_CPP_PATH "/app/whisper.cpp")
message(STATUS "Using whisper.cpp from: ${WHISPER_CPP_PATH}")

# ───────────────────────────────────────────────────────
#  Global compile flags to disable all ggml back-ends
# ───────────────────────────────────────────────────────
add_compile_definitions(
    WHISPER_NO_ACCEL=1     # whisper.cpp flag
    GGML_DISABLE_DEV=1          # ggml flag (safe even if ggml is older)
)

# Optional: extra optimisation & PIC
add_compile_options(-O3 -fPIC)

# ───────────────────────────────────────────────────────
#  Include / link paths
# ───────────────────────────────────────────────────────
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
)

# ───────────────────────────────────────────────────────
#  Whisper / ggml source files (built here in-tree)
# ───────────────────────────────────────────────────────
file(GLOB WHISPER_SOURCES
    "${WHISPER_CPP_PATH}/src/*.cpp"
    "${WHISPER_CPP_PATH}/ggml/src/*.cpp"
    "${WHISPER_CPP_PATH}/ggml/src/*.c"
)

# ───────────────────────────────────────────────────────
#  Executable
# ───────────────────────────────────────────────────────
add_executable(whisper_server
    whisper_server.cpp
    proto/whisper.pb.cc
    proto/whisper.grpc.pb.cc
    ${WHISPER_SOURCES}
)

target_link_libraries(whisper_server
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    Threads::Threads
)

# ───────────────────────────────────────────────────────
#  C++ standard
# ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)