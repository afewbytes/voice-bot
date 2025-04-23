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

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerReader;
using whisper::WhisperService;
using whisper::AudioChunk;
using whisper::Transcription;

// Socket path
static const std::string SOCKET_PATH = "/app/sockets/whisper.sock";
// Model file
static const char* MODEL_PATH = "/app/models/ggml-base.en.bin";

// Forward declaration
class WhisperContextPool;

// RAII guard for a whisper_context pulled from the pool
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

// A simple thread-safe pool of whisper_context objects
class WhisperContextPool {
public:
    WhisperContextPool(size_t size) {
        whisper_context_params params = whisper_context_default_params();
        params.use_gpu = false;

        for (size_t i = 0; i < size; ++i) {
            whisper_context* ctx = whisper_init_from_file_with_params(MODEL_PATH, params);
            if (!ctx) {
                std::cerr << "Failed to load whisper model for pool instance " << i << std::endl;
                exit(1);
            }
            pool_.push(ctx);
        }
    }

    ~WhisperContextPool() {
        // Release and free all contexts in the pool
        while (!pool_.empty()) {
            whisper_context* ctx = pool_.front();
            pool_.pop();
            whisper_free(ctx);
        }
    }

    ContextGuardPtr acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        cond_.wait(lock, [&]{ return !pool_.empty(); });
        whisper_context* ctx = pool_.front();
        pool_.pop();
        return std::make_unique<ContextGuard>(ctx, this);
    }

private:
    friend class ContextGuard;

    void release(whisper_context* ctx) {
        std::lock_guard<std::mutex> lock(mtx_);
        pool_.push(ctx);
        cond_.notify_one();
    }

    std::mutex mtx_;
    std::condition_variable cond_;
    std::queue<whisper_context*> pool_;
};

// Return context to the pool on guard destruction
ContextGuard::~ContextGuard() {
    pool_->release(ctx_);
}

// The gRPC service implementation
class WhisperServiceImpl final : public WhisperService::Service {
public:
    explicit WhisperServiceImpl(WhisperContextPool& pool)
        : pool_(pool) {}

    Status StreamAudio(ServerContext* context,
                       ServerReader<AudioChunk>* reader,
                       Transcription* response) override {
        // Acquire a context for this RPC
        auto guard = pool_.acquire();
        whisper_context* ctx = guard->get();

        AudioChunk chunk;
        std::vector<float> audio_buffer;
        std::string transcription;
        const int min_samples = 16000;

        // Whisper parameters
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_realtime = false;
        wparams.print_progress = false;
        wparams.print_timestamps = false;
        wparams.print_special = false;
        wparams.translate = false;
        wparams.language = "en";
        wparams.n_threads = static_cast<int>(std::thread::hardware_concurrency());
        wparams.offset_ms = 0;
        wparams.no_context = true;
        wparams.single_segment = true;

        auto convert = [&](const std::string& bytes) {
            std::vector<float> out;
            size_t n = bytes.size() / sizeof(int16_t);
            out.reserve(n);
            const int16_t* data = reinterpret_cast<const int16_t*>(bytes.data());
            for (size_t i = 0; i < n; ++i) {
                out.push_back(data[i] / 32768.0f);
            }
            return out;
        };

        auto pad = [&](std::vector<float>& buf) {
            if (buf.size() < static_cast<size_t>(min_samples)) {
                buf.resize(min_samples, 0.0f);
            }
        };

        while (reader->Read(&chunk)) {
            auto pcm = convert(chunk.data());
            audio_buffer.insert(audio_buffer.end(), pcm.begin(), pcm.end());

            if (audio_buffer.size() >= static_cast<size_t>(min_samples)) {
                pad(audio_buffer);
                if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) == 0) {
                    int nseg = whisper_full_n_segments(ctx);
                    for (int i = 0; i < nseg; ++i) {
                        const char* txt = whisper_full_get_segment_text(ctx, i);
                        if (txt && *txt) {
                            transcription += txt;
                            transcription += ' ';
                        }
                    }
                }
                audio_buffer.clear();
            }
        }

        // Process leftover audio
        if (!audio_buffer.empty()) {
            pad(audio_buffer);
            if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) == 0) {
                int nseg = whisper_full_n_segments(ctx);
                for (int i = 0; i < nseg; ++i) {
                    const char* txt = whisper_full_get_segment_text(ctx, i);
                    if (txt && *txt) {
                        transcription += txt;
                        transcription += ' ';
                    }
                }
            }
        }

        response->set_text(transcription);
        return Status::OK;
    }

private:
    WhisperContextPool& pool_;
};

void cleanup_socket() {
    if (access(SOCKET_PATH.c_str(), F_OK) != -1) {
        unlink(SOCKET_PATH.c_str());
    }
}

void RunServer() {
    std::string addr = "unix:" + SOCKET_PATH;

    mkdir("/app/sockets", 0777);
    cleanup_socket();

    // Preload a pool of contexts
    size_t pool_size = std::thread::hardware_concurrency();
    WhisperContextPool pool(pool_size);

    WhisperServiceImpl service(pool);

    ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "Failed to start server" << std::endl;
        exit(1);
    }

    // Signal readiness
    std::ofstream ready("/app/sockets/ready");
    ready.close();

    server->Wait();
}

int main(int, char**) {
    std::signal(SIGINT, [](int){ cleanup_socket(); exit(0); });
    std::signal(SIGTERM, [](int){ cleanup_socket(); exit(0); });
    RunServer();
    return 0;
}
