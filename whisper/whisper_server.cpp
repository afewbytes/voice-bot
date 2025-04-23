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

static const std::string SOCKET_PATH = "/app/sockets/whisper.sock";
static const char* MODEL_PATH = "/app/models/ggml-base.en.bin";

// ContextGuard implementation
ContextGuard::ContextGuard(whisper_context* ctx, WhisperContextPool* pool)
    : ctx_(ctx), pool_(pool) {}

ContextGuard::~ContextGuard() { 
    pool_->release(ctx_); 
}

// WhisperContextPool implementation
WhisperContextPool::WhisperContextPool(size_t count) {
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    for (size_t i = 0; i < count; ++i) {
        auto* ctx = whisper_init_from_file_with_params(MODEL_PATH, params);
        if (!ctx) { std::cerr << "Failed loading model for pool instance " << i << std::endl; exit(1); }
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
    auto* ctx = pool_.front(); 
    pool_.pop();
    return std::make_unique<ContextGuard>(ctx, this);
}

void WhisperContextPool::release(whisper_context* ctx) {
    std::lock_guard<std::mutex> lk(mux_);
    pool_.push(ctx);
    cond_.notify_one();
}

// WhisperServiceImpl implementation
WhisperServiceImpl::WhisperServiceImpl(WhisperContextPool& p) : pool_(p) {}

grpc::Status WhisperServiceImpl::StreamAudio(grpc::ServerContext* /*ctx*/,
                                      grpc::ServerReaderWriter<whisper::Transcription, whisper::AudioChunk>* stream) {
    auto guard = pool_.acquire();
    auto* ctx = guard->get();

    // whisper params - optimized for faster transcription
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.translate        = false;
    wparams.language         = "en";
    wparams.n_threads        = static_cast<int>(std::thread::hardware_concurrency());
    wparams.no_context       = true;    // Faster processing with context disabled
    wparams.single_segment   = true;    // Better for real-time transcription
    wparams.max_tokens       = 32;      // Limit token generation for faster response
    
    // Removed speed_up parameter as it's not available in this version

    const int sample_rate = 16000;
    
    // OPTIMIZED: Smaller time windows for faster processing
    const int step_ms = 1000;      // 1 second step (was 5000ms)
    const int len_ms  = 3000;      // 3 second window (was 10000ms)
    const int keep_ms = 500;       // 0.5 second overlap (was 2000ms)
    
    const int n_step  = step_ms * sample_rate / 1000;
    const int n_len   = len_ms  * sample_rate / 1000;
    const int n_keep  = keep_ms * sample_rate / 1000;

    if (VERBOSE_LOGGING) {
        std::cerr << "Optimized parameters: step=" << step_ms << "ms, len=" << len_ms 
                  << "ms, keep=" << keep_ms << "ms" << std::endl;
    }

    std::vector<float> buf_len(n_len, 0.0f);
    std::vector<float> buf_old(n_keep, 0.0f);
    std::vector<float> incoming;
    incoming.reserve(n_step * 10);

    float last_end_ts = 0.0f;
    float window_offset = 0.0f;  // Track the absolute position of each window
    whisper::AudioChunk chunk;

    auto convert = [&](const std::string& bytes) {
        size_t n = bytes.size() / sizeof(int16_t);
        std::vector<float> out(n);
        const int16_t* data = reinterpret_cast<const int16_t*>(bytes.data());
        for (size_t i = 0; i < n; ++i) out[i] = data[i] / 32768.0f;
        return out;
    };

    // streaming loop
    bool first_chunk = true;
    uint64_t start_time = 0;

    while (stream->Read(&chunk)) {
        if (first_chunk) {
            first_chunk = false;
            start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            if (VERBOSE_LOGGING) {
                std::cerr << "First audio chunk received" << std::endl;
            }
        }

        auto pcm = convert(chunk.data());
        incoming.insert(incoming.end(), pcm.begin(), pcm.end());

        if (VERBOSE_LOGGING) {
            std::cerr << "Audio buffer size: " << incoming.size() 
                      << " samples (need " << n_step << " to process)" << std::endl;
        }

        // Process as soon as we have enough samples
        while ((int)incoming.size() >= n_step) {
            if (VERBOSE_LOGGING) {
                uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                std::cerr << "Processing window at t+" << (current_time - start_time) 
                          << "ms, buffer: " << incoming.size() << " samples" << std::endl;
            }

            // build window with overlap
            std::fill(buf_len.begin(), buf_len.end(), 0.0f);
            std::copy(buf_old.begin(), buf_old.end(), buf_len.begin());
            
            // Copy the appropriate amount of new data
            size_t to_copy = std::min((size_t)(n_len - n_keep), incoming.size());
            std::copy(incoming.begin(), incoming.begin() + to_copy,
                      buf_len.begin() + n_keep);

            uint64_t before_whisper = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            // Run inference
            if (whisper_full(ctx, wparams, buf_len.data(), buf_len.size()) != 0) {
                std::cerr << "Inference failed" << std::endl;
                return grpc::Status::OK;
            }

            uint64_t after_whisper = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            if (VERBOSE_LOGGING) {
                std::cerr << "Whisper processing took " 
                          << (after_whisper - before_whisper) << "ms" << std::endl;
            }

            // emit new segments by timestamp
            int nseg = whisper_full_n_segments(ctx);
            
            if (VERBOSE_LOGGING) {
                std::cerr << "Found " << nseg << " segment(s)" << std::endl;
            }
            
            for (int i = 0; i < nseg; ++i) {
                // Get relative timestamps within the window
                float relative_start = whisper_full_get_segment_t0(ctx, i) * 0.02f;
                float relative_end = whisper_full_get_segment_t1(ctx, i) * 0.02f;
                
                // Calculate absolute timestamps based on window position
                float absolute_start = window_offset + relative_start;
                float absolute_end = window_offset + relative_end;
                
                // Get the text and send it
                const char* txt = whisper_full_get_segment_text(ctx, i);
                std::string text = txt ? txt : "";
                
                // Skip empty segments
                if (!text.empty()) {
                    whisper::Transcription out;
                    out.set_text(text);
                    stream->Write(out);
                    
                    if (VERBOSE_LOGGING) {
                        std::cerr << "Sent transcription: '" << text << "'" << std::endl;
                    }
                    
                    last_end_ts = absolute_end;
                }
            }

            // slide buffers
            std::copy(buf_len.begin() + (n_len - n_keep), buf_len.end(), buf_old.begin());
            incoming.erase(incoming.begin(), incoming.begin() + n_step);
            
            // Update the window offset for the next window
            window_offset += (float)n_step / sample_rate;
        }
    }

    // process any leftover audio
    if (!incoming.empty()) {
        if (VERBOSE_LOGGING) {
            std::cerr << "Processing final " << incoming.size() << " samples" << std::endl;
        }
        
        std::fill(buf_len.begin(), buf_len.end(), 0.0f);
        std::copy(buf_old.begin(), buf_old.end(), buf_len.begin());
        size_t copy_count = std::min((int)incoming.size(), n_len - n_keep);
        std::copy(incoming.begin(), incoming.begin() + copy_count,
                  buf_len.begin() + n_keep);

        if (whisper_full(ctx, wparams, buf_len.data(), buf_len.size()) == 0) {
            int nseg = whisper_full_n_segments(ctx);
            for (int i = 0; i < nseg; ++i) {
                float relative_start = whisper_full_get_segment_t0(ctx, i) * 0.02f;
                float relative_end = whisper_full_get_segment_t1(ctx, i) * 0.02f;
                
                // Calculate absolute timestamps for final window
                float absolute_start = window_offset + relative_start;
                float absolute_end = window_offset + relative_end;
                
                const char* txt = whisper_full_get_segment_text(ctx, i);
                std::string text = txt ? txt : "";
                
                if (!text.empty()) {
                    whisper::Transcription out;
                    out.set_text(text);
                    stream->Write(out);
                    last_end_ts = absolute_end;
                }
            }
        }
    }

    if (VERBOSE_LOGGING) {
        std::cerr << "Stream ended, transcription complete" << std::endl;
    }

    return grpc::Status::OK;
}

void cleanup_socket() {
    if (access(SOCKET_PATH.c_str(), F_OK) == 0) unlink(SOCKET_PATH.c_str());
}

void RunServer() {
    mkdir("/app/sockets", 0777);
    cleanup_socket();
    std::string addr = "unix:" + SOCKET_PATH;

    // Increase the number of contexts in the pool if you have more CPU cores
    int num_threads = std::max(1, (int)std::thread::hardware_concurrency());
    std::cerr << "Starting WhisperContextPool with " << num_threads << " threads" << std::endl;
    
    WhisperContextPool pool(num_threads);
    WhisperServiceImpl svc(pool);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&svc);

    auto server = builder.BuildAndStart();
    if (!server) { std::cerr << "gRPC start failed" << std::endl; exit(1); }
    std::ofstream("/app/sockets/ready").close();
    std::cerr << "Server started and ready" << std::endl;
    server->Wait();
}

int main() {
    std::signal(SIGINT,  [](int){ cleanup_socket(); exit(0); });
    std::signal(SIGTERM, [](int){ cleanup_socket(); exit(0); });
    
    // Set high priority for the process
    setpriority(PRIO_PROCESS, 0, -10);
    
    RunServer();
    return 0;
}