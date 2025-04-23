#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <csignal>
#include <unistd.h>
#include <grpcpp/grpcpp.h>
#include "whisper.grpc.pb.h"
#include "whisper.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <cmath> // For std::ceil

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerReaderWriter;
using whisper::WhisperService;
using whisper::AudioChunk;
using whisper::Transcription;

static const std::string SOCKET_PATH = "/app/sockets/whisper.sock";
static const char* MODEL_PATH = "/app/models/ggml-base.en.bin";

class WhisperContextPool;

class ContextGuard {
public:
    ContextGuard(whisper_context* ctx, WhisperContextPool* pool)
        : ctx_(ctx), pool_(pool) {}
    ~ContextGuard();
    whisper_context* get() const { return ctx_; }
private:
    whisper_context* ctx_;
    WhisperContextPool* pool_;
};

typedef std::unique_ptr<ContextGuard> ContextGuardPtr;

class WhisperContextPool {
public:
    WhisperContextPool(size_t count) {
        auto params = whisper_context_default_params();
        params.use_gpu = false;
        for (size_t i = 0; i < count; ++i) {
            auto* ctx = whisper_init_from_file_with_params(MODEL_PATH, params);
            if (!ctx) { std::cerr << "Failed loading model for pool instance " << i << std::endl; exit(1); }
            pool_.push(ctx);
        }
    }
    ~WhisperContextPool() {
        while (!pool_.empty()) { whisper_free(pool_.front()); pool_.pop(); }
    }
    ContextGuardPtr acquire() {
        std::unique_lock<std::mutex> lk(mux_);
        cond_.wait(lk, [&]{ return !pool_.empty(); });
        auto* ctx = pool_.front(); pool_.pop();
        return std::make_unique<ContextGuard>(ctx, this);
    }
private:
    friend class ContextGuard;
    void release(whisper_context* ctx) {
        std::lock_guard<std::mutex> lk(mux_);
        pool_.push(ctx);
        cond_.notify_one();
    }
    std::mutex mux_;
    std::condition_variable cond_;
    std::queue<whisper_context*> pool_;
};

ContextGuard::~ContextGuard() { pool_->release(ctx_); }

class WhisperServiceImpl final : public WhisperService::Service {
public:
    explicit WhisperServiceImpl(WhisperContextPool& p) : pool_(p) {}

    Status StreamAudio(ServerContext* /*ctx*/,
                       ServerReaderWriter<Transcription, AudioChunk>* stream) override {
        auto guard = pool_.acquire();
        auto* ctx = guard->get();

        // whisper params
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = true;
        wparams.translate        = false;
        wparams.language         = "en";
        wparams.n_threads        = static_cast<int>(std::thread::hardware_concurrency());
        wparams.no_context       = true;
        wparams.single_segment   = false;

        const int sample_rate = 16000;
        const int step_ms = 5000;    // Larger step for file processing
        const int len_ms  = 10000;   // Longer window to capture more context
        const int keep_ms = 2000;    // Less overlap needed for file processing
        const int n_step  = step_ms * sample_rate / 1000;
        const int n_len   = len_ms  * sample_rate / 1000;
        const int n_keep  = keep_ms * sample_rate / 1000;

        std::vector<float> buf_len(n_len, 0.0f);
        std::vector<float> buf_old(n_keep, 0.0f);
        std::vector<float> incoming;
        incoming.reserve(n_step * 10);

        float last_end_ts = 0.0f;
        float window_offset = 0.0f;  // Track the absolute position of each window
        AudioChunk chunk;

        auto convert = [&](const std::string& bytes) {
            size_t n = bytes.size() / sizeof(int16_t);
            std::vector<float> out(n);
            const int16_t* data = reinterpret_cast<const int16_t*>(bytes.data());
            for (size_t i = 0; i < n; ++i) out[i] = data[i] / 32768.0f;
            return out;
        };

        // streaming loop
        while (stream->Read(&chunk)) {
            auto pcm = convert(chunk.data());
            incoming.insert(incoming.end(), pcm.begin(), pcm.end());

            while ((int)incoming.size() >= n_step) {
                // build window with overlap
                std::fill(buf_len.begin(), buf_len.end(), 0.0f);
                std::copy(buf_old.begin(), buf_old.end(), buf_len.begin());
                std::copy(incoming.begin(), incoming.begin() + (n_len - n_keep),
                          buf_len.begin() + n_keep);

                if (whisper_full(ctx, wparams, buf_len.data(), buf_len.size()) != 0) {
                    std::cerr << "Inference failed" << std::endl;
                    return Status::OK;
                }

                // emit new segments by timestamp
                int nseg = whisper_full_n_segments(ctx);
                for (int i = 0; i < nseg; ++i) {
                    // Get relative timestamps within the window
                    float relative_start = whisper_full_get_segment_t0(ctx, i) * 0.02f;
                    float relative_end = whisper_full_get_segment_t1(ctx, i) * 0.02f;
                    
                    // Calculate absolute timestamps based on window position
                    float absolute_start = window_offset + relative_start;
                    float absolute_end = window_offset + relative_end;
                    
                    const char* txt = whisper_full_get_segment_text(ctx, i);
                    Transcription out;
                    out.set_text(txt ? txt : "");
                    stream->Write(out);
                    last_end_ts = absolute_end;
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
                    Transcription out;
                    out.set_text(txt ? txt : "");
                    stream->Write(out);
                    last_end_ts = absolute_end;
                }
            }
        }

        return Status::OK;
    }

private:
    WhisperContextPool& pool_;
};

void cleanup_socket() {
    if (access(SOCKET_PATH.c_str(), F_OK) == 0) unlink(SOCKET_PATH.c_str());
}

void RunServer() {
    mkdir("/app/sockets", 0777);
    cleanup_socket();
    std::string addr = "unix:" + SOCKET_PATH;

    WhisperContextPool pool(1);
    WhisperServiceImpl svc(pool);

    ServerBuilder builder;
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
    RunServer();
    return 0;
}