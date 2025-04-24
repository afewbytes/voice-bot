// whisper_server.cpp
#include "whisper_server.h"
#include <fstream>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <deque>
#include <iostream>
#include <iomanip>

static const std::string SOCKET_PATH = "/app/sockets/whisper.sock";
static const char* MODEL_PATH        = "/app/models/ggml-base.en.bin";

// Buffer parameters
static const size_t MAX_AUDIO_BUFFER = 30 * 16000 * sizeof(int16_t);
static const size_t MIN_PROCESS_SIZE = 0.5 * 16000 * sizeof(int16_t);

// Logging helper
#define LOG_INFO(msg) do { std::cout << "[INFO] " << msg << std::endl; } while(0)
#define LOG_ERROR(msg) do { std::cerr << "[ERROR] " << msg << std::endl; } while(0)
#define LOG_DEBUG(msg) do { if (VERBOSE_LOGGING) std::cout << "[DEBUG] " << msg << std::endl; } while(0)

ContextGuard::ContextGuard(whisper_context* ctx, WhisperContextPool* pool)
    : ctx_(ctx), pool_(pool) {}
ContextGuard::~ContextGuard() { pool_->release(ctx_); }

WhisperContextPool::WhisperContextPool(size_t count) {
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    for (size_t i = 0; i < count; ++i) {
        auto* ctx = whisper_init_from_file_with_params(MODEL_PATH, params);
        if (!ctx) exit(1);
        pool_.push(ctx);
    }
}
WhisperContextPool::~WhisperContextPool() {
    while (!pool_.empty()) {
        whisper_free(pool_.front());
        pool_.pop();
    }
}
ContextGuardPtr WhisperContextPool::acquire() {
    std::unique_lock<std::mutex> lk(mux_);
    cond_.wait(lk, [&]{ return !pool_.empty(); });
    auto* ctx = pool_.front(); pool_.pop();
    return std::make_unique<ContextGuard>(ctx, this);
}
void WhisperContextPool::release(whisper_context* ctx) {
    std::lock_guard<std::mutex> lk(mux_);
    pool_.push(ctx);
    cond_.notify_one();
}

WhisperServiceImpl::WhisperServiceImpl(WhisperContextPool& p)
    : pool_(p) {}

grpc::Status WhisperServiceImpl::StreamAudio(
    grpc::ServerContext*,
    grpc::ServerReaderWriter<voice::StreamAudioResponse, voice::AudioChunk>* stream
) {
    static int session_counter = 0;
    int session_id = ++session_counter;
    LOG_INFO("Session " << session_id << ": New audio stream connection started");

    auto guard = pool_.acquire();
    auto* ctx = guard->get();

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.translate        = false;
    wparams.language         = "en";
    wparams.n_threads        = int(std::thread::hardware_concurrency());
    wparams.no_context       = false;
    wparams.single_segment   = false;
    wparams.max_tokens       = 0;
    wparams.audio_ctx        = 0;

    std::vector<whisper_token> prompt_tokens;
    const int N_CTX = whisper_n_text_ctx(ctx);

    std::vector<float> audio_buffer;
    audio_buffer.reserve(MAX_AUDIO_BUFFER / sizeof(float));

    bool in_speech_segment = false;
    bool processed_segment = false;

    auto to_f32 = [&](const std::string& in) {
        size_t n = in.size() / sizeof(int16_t);
        std::vector<float> out(n);
        auto d = reinterpret_cast<const int16_t*>(in.data());
        for (size_t i = 0; i < n; ++i) out[i] = d[i] / 32768.0f;
        return out;
    };

    auto process_buffer = [&](bool is_final) -> bool {
        if (audio_buffer.size() * sizeof(float) < MIN_PROCESS_SIZE) return false;
        std::string out_text;
        int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char* seg = whisper_full_get_segment_text(ctx, i);
            if (seg && *seg) out_text += seg;
        }
        if (!out_text.empty()) {
            voice::StreamAudioResponse resp;
            resp.set_text(std::move(out_text));
            resp.set_done(false);
            resp.set_source(voice::StreamAudioResponse::WHISPER);
            stream->Write(resp);
        }
        // manage prompt context tokens...
        return true;
    };

    voice::AudioChunk chunk;
    while (stream->Read(&chunk)) {
        if (chunk.speech_start()) {
            in_speech_segment = true;
            prompt_tokens.clear(); audio_buffer.clear(); processed_segment = false;
            continue;
        }
        if (chunk.speech_end()) {
            if (!audio_buffer.empty()) process_buffer(true);
            in_speech_segment = false; audio_buffer.clear();
            continue;
        }
        auto data = to_f32(chunk.data());
        if (!data.empty() && in_speech_segment) {
            audio_buffer.insert(audio_buffer.end(), data.begin(), data.end());
            if (audio_buffer.size() * sizeof(float) > MAX_AUDIO_BUFFER/4) {
                process_buffer(false);
                size_t keep = std::min(audio_buffer.size(), (size_t)(0.5 * WHISPER_SAMPLE_RATE));
                audio_buffer = std::vector<float>(audio_buffer.end()-keep, audio_buffer.end());
            }
        }
    }
    if (!audio_buffer.empty()) process_buffer(true);

    LOG_INFO("Session " << session_id << ": Stream closed");
    return grpc::Status::OK;
}

void cleanup_socket() {
    if (access(SOCKET_PATH.c_str(), F_OK) == 0) unlink(SOCKET_PATH.c_str());
}
void RunServer() {
    mkdir("/app/sockets", 0777);
    cleanup_socket();
    std::string addr = "unix:" + SOCKET_PATH;
    WhisperContextPool pool(std::max(1, int(std::thread::hardware_concurrency())));
    WhisperServiceImpl svc(pool);
    grpc::ServerBuilder b;
    b.AddListeningPort(addr, grpc::InsecureServerCredentials());
    b.RegisterService(&svc);
    auto server = b.BuildAndStart();
    std::ofstream("/app/sockets/ready").close();
    LOG_INFO("Server ready to accept connections");
    server->Wait();
}
int main() {
    std::signal(SIGINT,  [](int){ cleanup_socket(); exit(0); });
    std::signal(SIGTERM, [](int){ cleanup_socket(); exit(0); });
    setpriority(PRIO_PROCESS, 0, -10);
    RunServer();
    return 0;
}