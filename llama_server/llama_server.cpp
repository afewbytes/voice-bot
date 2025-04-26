#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

#include <grpcpp/grpcpp.h>
#include "voice.grpc.pb.h"
#include "llama.h"  // adjust include path if needed

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerWriter;

using voice::LlamaService;
using voice::GenerateRequest;
using voice::GenerateResponse;

// ───────────────────────────────────────── context‑pool helpers ──────────────
class LlamaContextPool; // fwd
class ContextGuard {
public:
    ContextGuard(llama_context *ctx, LlamaContextPool *pool) : ctx_(ctx), pool_(pool) {}
    ~ContextGuard();
    llama_context *get() { return ctx_; }
private:
    llama_context *ctx_;
    LlamaContextPool *pool_;
};
using ContextGuardPtr = std::unique_ptr<ContextGuard>;

class LlamaContextPool {
public:
    LlamaContextPool(llama_model *model, size_t count, int n_ctx);
    ~LlamaContextPool();
    ContextGuardPtr acquire();
    void            release(llama_context *ctx);
private:
    llama_model              *model_;
    std::queue<llama_context*> pool_;
    std::mutex                mux_;
    std::condition_variable   cond_;
};

ContextGuard::~ContextGuard() { pool_->release(ctx_); }

LlamaContextPool::LlamaContextPool(llama_model *model, size_t count, int n_ctx) : model_(model) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx;
    cp.n_batch = 24;

    std::cout << "Creating " << count << " context(s) – n_ctx=" << n_ctx << std::endl;
    for (size_t i = 0; i < count; ++i) {
        auto *ctx = llama_init_from_model(model_, cp);
        if (!ctx) {
            std::cerr << "Failed to create context " << i << std::endl;
            std::exit(1);
        }
        pool_.push(ctx);
    }
}

LlamaContextPool::~LlamaContextPool() {
    while (!pool_.empty()) { llama_free(pool_.front()); pool_.pop(); }
}

ContextGuardPtr LlamaContextPool::acquire() {
    std::unique_lock<std::mutex> lk(mux_);
    cond_.wait(lk, [&]{ return !pool_.empty(); });
    auto *ctx = pool_.front(); pool_.pop();
    return std::make_unique<ContextGuard>(ctx, this);
}
void LlamaContextPool::release(llama_context *ctx) {
    { std::lock_guard<std::mutex> lk(mux_); pool_.push(ctx); }
    cond_.notify_one();
}

// ───────────────────────────────────────── startup smoke‑test ────────────────
static bool run_startup_test(llama_model *model, const llama_vocab *vocab, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return false;

    const std::string probe = "Hello";
    std::vector<llama_token> toks(-llama_tokenize(vocab, probe.c_str(), probe.size(), nullptr, 0, true, true));
    llama_tokenize(vocab, probe.c_str(), probe.size(), toks.data(), toks.size(), true, true);
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size())) != 0) { llama_free(ctx); return false; }
    llama_free(ctx);
    return true;
}

// ───────────────────────────────────────── gRPC service impl ────────────────
class LlamaServiceImpl : public LlamaService::Service {
public:
    explicit LlamaServiceImpl(const std::string &model_path) {
        // read n_ctx from env (default 2048)
        int n_ctx = 2048;
        if (const char *env = std::getenv("LLAMA_N_CTX")) n_ctx = std::atoi(env);
        if (n_ctx < 512) n_ctx = 512; // safety floor
        std::cout << "LLAMA_N_CTX=" << n_ctx << std::endl;

        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        model_ = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model_) { std::cerr << "Cannot load model " << model_path << std::endl; std::exit(1);}    
        vocab_ = llama_model_get_vocab(model_);
        if (!vocab_) { std::cerr << "Cannot get vocab" << std::endl; std::exit(1);}        

        if (!run_startup_test(model_, vocab_, n_ctx)) { std::cerr << "startup test failed" << std::endl; std::exit(1);}    

        size_t ctx_count = std::max(1u, std::thread::hardware_concurrency() / 2u);
        context_pool_ = std::make_unique<LlamaContextPool>(model_, ctx_count, n_ctx);
        std::cout << "Pool ready: " << ctx_count << " × context(s)" << std::endl;
    }
    ~LlamaServiceImpl() override {
        context_pool_.reset(); if (model_) llama_model_free(model_);
    }

    Status Generate(ServerContext *, const GenerateRequest *req, ServerWriter<GenerateResponse> *w) override {
        auto guard = context_pool_->acquire();
        llama_context *ctx = guard->get();

        // tokenise prompt
        const std::string &prompt = req->prompt();
        int n   = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
        if (n <= 0) return Status(grpc::StatusCode::INTERNAL, "tokenise_fail");
        std::vector<llama_token> toks(n);
        llama_tokenize(vocab_, prompt.c_str(), prompt.size(), toks.data(), n, true, true);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0)
            return Status(grpc::StatusCode::INTERNAL, "prompt_eval_fail");

        // sampler
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        llama_sampler *sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(req->top_p() > 0 ? req->top_p() : 0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(req->temperature() > 0 ? req->temperature() : 0.7f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        // stream tokens
        const int max_tok = req->max_tokens() > 0 ? req->max_tokens() : 256;
        GenerateResponse out;
        char buf[64]; // was 16 – avoid overflow

        for (int i = 0; i < max_tok; ++i) {
            llama_token next = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab_, next)) break;
            int sz = llama_token_to_piece(vocab_, next, buf, sizeof(buf), 0, true);
            out.set_text(sz > 0 ? std::string(buf, sz) : "");
            out.set_done(false);
            w->Write(out);
            if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) {
                llama_sampler_free(sampler);
                return Status(grpc::StatusCode::INTERNAL, "decode_fail");
            }
        }
        out.set_text(""); out.set_done(true); w->Write(out);
        llama_sampler_free(sampler);
        return Status::OK;
    }
private:
    llama_model *model_{nullptr};
    const llama_vocab *vocab_{nullptr};
    std::unique_ptr<LlamaContextPool> context_pool_;
};

// ───────────────────────────────────────── main ──────────────────────────────
int main(int argc, char **argv) {
    std::string model_path = argc > 1 ? argv[1] : "/app/models/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf";
    const std::string sock = "/app/llama-sockets/llama.sock";
    const std::string addr = "unix://" + sock;
    mkdir("/app/llama-sockets", 0777);
    unlink(sock.c_str());

    LlamaServiceImpl svc(model_path);

    grpc::ServerBuilder b; b.AddListeningPort(addr, grpc::InsecureServerCredentials()); b.RegisterService(&svc);
    std::unique_ptr<Server> server(b.BuildAndStart());
    std::ofstream("/app/llama-sockets/ready").close();
    std::cout << "llama‑cpp server ready on " << addr << std::endl;
    server->Wait();
    return 0;
}
