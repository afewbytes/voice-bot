###############################################################################
# CMakeLists.txt – piper_server (CUDA-enabled)
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
# 2. Paths – Piper checkout and vendored ORT-GPU
# ---------------------------------------------------------------------------
set(PIPER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../piper")

set(ORT_ROOT     "${PIPER_PATH}/build/pi")
set(ORT_INC_DIR  "${ORT_ROOT}/include")
set(ORT_LIB_DIR  "${ORT_ROOT}/lib")

# ---------------------------------------------------------------------------
# 3. Find libs built by Piper
# ---------------------------------------------------------------------------
find_library(PIPER_PHONEMIZE_LIBRARY piper_phonemize
    PATHS ${PIPER_PATH}/build/pi/lib
    REQUIRED)
message(STATUS "Found Piper Phonemize library: ${PIPER_PHONEMIZE_LIBRARY}")

find_library(ONNX_RUNTIME_LIBRARY onnxruntime
    PATHS  ${ORT_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED)
message(STATUS "Found ONNX Runtime library: ${ONNX_RUNTIME_LIBRARY}")

# ---------------------------------------------------------------------------
# 4. Include / link search paths
# ---------------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${PIPER_PATH}/src/cpp            # piper.hpp + helpers
    ${ORT_INC_DIR}                   # onnxruntime_cxx_api.h
    ${spdlog_INCLUDE_DIRS}
)

link_directories(
    ${PIPER_PATH}/build/pi/lib       # ORT + phonemize
)

# ---------------------------------------------------------------------------
# 5. Build the executable
# ---------------------------------------------------------------------------
add_executable(piper_server
    piper_server.cpp
    ../proto/voice.pb.cc
    ../proto/voice.grpc.pb.cc
    ${PIPER_PATH}/src/cpp/piper.cpp      # ★ correct location of piper.cpp
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