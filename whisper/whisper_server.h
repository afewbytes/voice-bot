#ifndef WHISPER_SERVICE_H
#define WHISPER_SERVICE_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <grpcpp/grpcpp.h>
#include "whisper.grpc.pb.h"
#include "whisper.h"

// Flag to enable more verbose logging for debugging
static const bool VERBOSE_LOGGING = false;

class WhisperContextPool;

class ContextGuard {
public:
    ContextGuard(whisper_context* ctx, WhisperContextPool* pool);
    ~ContextGuard();
    whisper_context* get() const { return ctx_; }
private:
    whisper_context* ctx_;
    WhisperContextPool* pool_;
};

typedef std::unique_ptr<ContextGuard> ContextGuardPtr;

class WhisperContextPool {
public:
    WhisperContextPool(size_t count);
    ~WhisperContextPool();
    ContextGuardPtr acquire();
private:
    friend class ContextGuard;
    void release(whisper_context* ctx);
    std::mutex mux_;
    std::condition_variable cond_;
    std::queue<whisper_context*> pool_;
};

class WhisperServiceImpl final : public whisper::WhisperService::Service {
public:
    explicit WhisperServiceImpl(WhisperContextPool& p);
    grpc::Status StreamAudio(grpc::ServerContext* ctx,
                       grpc::ServerReaderWriter<whisper::Transcription, whisper::AudioChunk>* stream) override;
private:
    WhisperContextPool& pool_;
};

void cleanup_socket();
void RunServer();

#endif // WHISPER_SERVICE_H