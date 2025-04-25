###############################################################################
# CMakeLists.txt – piper_server
# Builds the gRPC Unix-socket server re-using the ONNX Runtime copy that Piper
# downloads/compiles inside its own build tree (no system libonnxruntime-dev).
###############################################################################

cmake_minimum_required(VERSION 3.10)
project(piper_server)

# ---------------------------------------------------------------------------
# 1. Dependencies available in Ubuntu 22.04
# ---------------------------------------------------------------------------
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads  REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# Find spdlog package
find_package(spdlog REQUIRED)

message(STATUS "Using Protobuf version: ${Protobuf_VERSION}")
message(STATUS "Using gRPC      version: ${GRPC_VERSION}")
message(STATUS "Using spdlog    version: ${spdlog_VERSION}")

# ---------------------------------------------------------------------------
# 2. Paths – Piper source tree and its vendored ONNX Runtime
# ---------------------------------------------------------------------------
set(PIPER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../piper")  # cloned in Dockerfile

# After Piper's build:
#   ${PIPER_PATH}/build/pi/include        ← public headers
#   ${PIPER_PATH}/build/pi/lib/libonnxruntime.so
set(ORT_ROOT     "${PIPER_PATH}/build/pi")
set(ORT_INC_DIR  "${ORT_ROOT}/include")
set(ORT_LIB_DIR  "${ORT_ROOT}/lib")
set(ORT_LIB      "${ORT_LIB_DIR}/libonnxruntime.so")

# ---------------------------------------------------------------------------
# 3. Find the phonemize library and ONNX Runtime from Piper's build
# ---------------------------------------------------------------------------
find_library(PIPER_PHONEMIZE_LIBRARY piper_phonemize
    PATHS 
        ${PIPER_PATH}/build/install/lib
        ${PIPER_PATH}/build/pi/lib
        /usr/local/lib
    REQUIRED
)
message(STATUS "Found Piper Phonemize library: ${PIPER_PHONEMIZE_LIBRARY}")

find_library(ONNX_RUNTIME_LIBRARY onnxruntime
    PATHS
        ${PIPER_PATH}/build/install/lib
        ${PIPER_PATH}/build/pi/lib
        ${ORT_LIB_DIR}
    REQUIRED
)
message(STATUS "Found ONNX Runtime library: ${ONNX_RUNTIME_LIBRARY}")

# ---------------------------------------------------------------------------
# 4. Compiler & linker search paths
# ---------------------------------------------------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto
    ${CMAKE_CURRENT_SOURCE_DIR}/src      # Local copy of Piper source
    ${GRPC_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${PIPER_PATH}/src                    # Piper's public C API
    ${PIPER_PATH}/src/cpp                # Piper's C++ helpers
    ${PIPER_PATH}/build/install/include  # Installed headers
    ${ORT_INC_DIR}                       # ONNX Runtime headers
    ${spdlog_INCLUDE_DIRS}               # spdlog headers
)

link_directories(
    ${PIPER_PATH}/build/install/lib      # Installed libraries
    ${PIPER_PATH}/build/pi/lib           # Additional libraries
)

# ---------------------------------------------------------------------------
# 5. Build the executable
# ---------------------------------------------------------------------------
add_executable(piper_server
    # Main application sources
    piper_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto/voice.pb.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../proto/voice.grpc.pb.cc
    
    # Piper source files (direct compilation)
    ${CMAKE_CURRENT_SOURCE_DIR}/src/piper/piper.cpp
)

target_compile_options(piper_server PRIVATE -O3 -fPIC)

target_link_libraries(piper_server
    ${PIPER_PHONEMIZE_LIBRARY}          # Phonemization library
    ${ONNX_RUNTIME_LIBRARY}             # ONNX Runtime
    ${GRPC_LIBRARIES}
    ${Protobuf_LIBRARIES}
    spdlog::spdlog                      # spdlog library
    Threads::Threads
    sndfile opusfile opus samplerate espeak-ng
)

# ---------------------------------------------------------------------------
# 6. Language standard
# ---------------------------------------------------------------------------
set_property(TARGET piper_server PROPERTY CXX_STANDARD 17)
set_property(TARGET piper_server PROPERTY CXX_STANDARD_REQUIRED ON)