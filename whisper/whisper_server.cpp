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

static const std::string SOCKET_PATH = "/app/sockets/whisper.sock";
static const char* MODEL_PATH        = "/app/models/ggml-base.en.bin";

// Buffer parameters
static const size_t MAX_AUDIO_BUFFER = 30 * 16000 * sizeof(int16_t); // 30 seconds max buffer
static const size_t MIN_PROCESS_SIZE = 0.5 * 16000 * sizeof(int16_t); // Min 0.5 seconds for processing

// ContextGuard (unchanged)
ContextGuard::ContextGuard(whisper_context* ctx, WhisperContextPool* pool)
    : ctx_(ctx), pool_(pool) {}
ContextGuard::~ContextGuard() { pool_->release(ctx_); }

// WhisperContextPool (unchanged)
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

// Updated Service Implementation
WhisperServiceImpl::WhisperServiceImpl(WhisperContextPool& p) : pool_(p) {}

grpc::Status WhisperServiceImpl::StreamAudio(
    grpc::ServerContext*,
    grpc::ServerReaderWriter<whisper::Transcription, whisper::AudioChunk>* stream
) {
    auto guard = pool_.acquire();
    auto* ctx = guard->get();

    // Prepare whisper parameters 
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.translate        = false;
    wparams.language         = "en";
    wparams.n_threads        = int(std::thread::hardware_concurrency());
    wparams.no_context       = false;
    wparams.single_segment   = false; // Process as a whole utterance
    wparams.max_tokens       = 0;     // Use full segment output
    wparams.audio_ctx        = 0;

    // Token context storage
    std::vector<whisper_token> prompt_tokens;
    const int N_CTX = whisper_n_text_ctx(ctx);

    // Audio buffer for complete utterances
    std::vector<float> audio_buffer;
    audio_buffer.reserve(MAX_AUDIO_BUFFER / sizeof(float));
    
    // Time tracking
    auto last_process_time = std::chrono::steady_clock::now();
    bool processing_active = false;
    
    // Helper: convert incoming int16 data to float
    auto to_f32 = [&](const std::string& in) {
        size_t n = in.size() / sizeof(int16_t);
        std::vector<float> out(n);
        auto d = reinterpret_cast<const int16_t*>(in.data());
        for (size_t i = 0; i < n; ++i) {
            out[i] = d[i] / 32768.0f;
        }
        return out;
    };

    // Helper: process current audio buffer
    auto process_buffer = [&]() -> bool {
        if (audio_buffer.empty() || audio_buffer.size() < MIN_PROCESS_SIZE / sizeof(float)) {
            return false;
        }

        // Process the entire buffer as a complete utterance
        wparams.prompt_n_tokens = prompt_tokens.size();
        wparams.prompt_tokens   = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
        
        if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) != 0) {
            return false;
        }

        // Prepare output text - using a proper line break instead of ANSI clear
        std::string out_text = "";
        int n_segments = whisper_full_n_segments(ctx);
        
        for (int i = 0; i < n_segments; ++i) {
            const char* seg = whisper_full_get_segment_text(ctx, i);
            if (seg && *seg) out_text += seg;
        }

        // If we have text, send it to client with a proper line ending
        if (!out_text.empty()) {
            whisper::Transcription out;
            out.set_text(out_text + "\n");
            stream->Write(out);
        }

        // Update prompt tokens for context in next processing
        prompt_tokens.clear();
        for (int i = 0; i < n_segments; ++i) {
            int nt = whisper_full_n_tokens(ctx, i);
            for (int j = 0; j < nt; ++j) {
                prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
            }
        }
        if ((int)prompt_tokens.size() > N_CTX) {
            prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.end() - N_CTX);
        }

        return true;
    };

    whisper::AudioChunk chunk;
    size_t silence_count = 0;
    const size_t silence_threshold = 10; // Arbitrary number to detect silence sections
    
    while (stream->Read(&chunk)) {
        auto data = to_f32(chunk.data());
        
        // Check if this is silence (VAD off period)
        bool is_silence = false;
        if (data.size() > 0) {
            // Simple energy-based silence detection as a backup
            float energy = 0.0f;
            for (float sample : data) {
                energy += sample * sample;
            }
            energy /= data.size();
            
            // Extremely low energy usually indicates silence or VAD off section
            if (energy < 1e-6) {
                silence_count++;
                is_silence = (silence_count >= silence_threshold);
            } else {
                silence_count = 0;
            }
        }
        
        // If this chunk has data, add it to our buffer
        if (!data.empty()) {
            // Append new audio data
            audio_buffer.insert(audio_buffer.end(), data.begin(), data.end());
            
            // Limit buffer size to prevent memory issues
            if (audio_buffer.size() * sizeof(float) > MAX_AUDIO_BUFFER) {
                audio_buffer.erase(audio_buffer.begin(), 
                    audio_buffer.begin() + (audio_buffer.size() - MAX_AUDIO_BUFFER / sizeof(float)));
            }
            
            processing_active = true;
        }
        
        // Determine if we should process the buffer
        bool should_process = false;
        
        // Process if we have a silence section after activity (VAD "off" period)
        if (is_silence && processing_active) {
            should_process = true;
        }
        
        // Process if we haven't processed in a while and have enough data
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_process_time).count();
            
        if (elapsed > 2000 && audio_buffer.size() * sizeof(float) >= MIN_PROCESS_SIZE) {
            should_process = true;
        }
        
        // Process the buffer if needed
        if (should_process) {
            if (process_buffer()) {
                last_process_time = now;
                audio_buffer.clear();
                processing_active = false;
                silence_count = 0;
            }
        }
    }
    
    // Process any remaining audio at the end
    if (!audio_buffer.empty()) {
        process_buffer();
    }

    return grpc::Status::OK;
}

// socket setup / main() (unchanged)
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
    auto s = b.BuildAndStart();
    std::ofstream("/app/sockets/ready").close();
    s->Wait();
}
int main() {
    std::signal(SIGINT,  [](int){ cleanup_socket(); exit(0); });
    std::signal(SIGTERM, [](int){ cleanup_socket(); exit(0); });
    setpriority(PRIO_PROCESS, 0, -10);
    RunServer();
    return 0;
}