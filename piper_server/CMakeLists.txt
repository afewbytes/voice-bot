###############################################################################
# CMakeLists.txt – piper_server (CUDA-enabled, local ORT + phonemize)
###############################################################################

cmake_minimum_required(VERSION 3.10)
project(piper_server)

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
find_package(Protobuf      REQUIRED)
find_package(PkgConfig     REQUIRED)
find_package(Threads       REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
find_package(CUDAToolkit   REQUIRED)
find_package(spdlog        REQUIRED)

message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC      version: ${GRPC_VERSION}")
message(STATUS "Using CUDA      version: ${CUDAToolkit_VERSION}")
message(STATUS "Using spdlog    version: ${spdlog_VERSION}")

# ---------------------------------------------------------------------------
# 2. Paths – Piper checkout + ORT headers staged by Dockerfile
# ---------------------------------------------------------------------------
set(PIPER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../piper")
set(ORT_INC_DIR "/usr/local/include")      # copied in Docker step 4

# ---------------------------------------------------------------------------
# 3. Libraries built by Piper
# ---------------------------------------------------------------------------
find_library(PIPER_PHONEMIZE_LIBRARY piper_phonemize
    PATHS ${PIPER_PATH}/build/pi/lib
    NO_DEFAULT_PATH            # ← use *only* the fresh build
    REQUIRED)
message(STATUS "Found Piper Phonemize library: ${PIPER_PHONEMIZE_LIBRARY}")

find_library(ONNX_RUNTIME_LIBRARY onnxruntime REQUIRED)
message(STATUS "Found ONNX Runtime library: ${ONNX_RUNTIME_LIBRARY}")

# ---------------------------------------------------------------------------
# 4. Include / link search paths
# ---------------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${PIPER_PATH}/src/cpp
    ${ORT_INC_DIR}
    /usr/local/include               # fmt / spdlog
    ${spdlog_INCLUDE_DIRS}
)

link_directories(
    ${PIPER_PATH}/build/pi/lib
)

# ---------------------------------------------------------------------------
# 5. Executable
# ---------------------------------------------------------------------------
add_executable(piper_server
    piper_server.cpp
    ../proto/voice.pb.cc
    ../proto/voice.grpc.pb.cc
    ${PIPER_PATH}/src/cpp/piper.cpp
)

target_compile_options(piper_server PRIVATE -O3 -fPIC)

target_link_libraries(piper_server
    ${PIPER_PHONEMIZE_LIBRARY}
    ${ONNX_RUNTIME_LIBRARY}
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    spdlog::spdlog
    Threads::Threads
    CUDA::cudart
    sndfile opusfile opus samplerate espeak-ng
)

# ---------------------------------------------------------------------------
# 6. Language standard
# ---------------------------------------------------------------------------
set_property(TARGET piper_server PROPERTY CXX_STANDARD 17)
set_property(TARGET piper_server PROPERTY CXX_STANDARD_REQUIRED ON)