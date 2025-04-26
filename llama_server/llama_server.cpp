// llama_server.cpp – llama.cpp gRPC micro‑service
// Streamlined for the **latest** llama.cpp (≥ 2025‑03)
// – one llama_context per conversation
// – 1024‑token sliding KV‑window using llama_kv_cache_seq_shift()
// Build example:
//   g++ -std=c++17 -O3 -pthread llama_server.cpp \
//       -lgrpc++ -lprotobuf -lllama -lggml -o llama_server

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#include <grpcpp/grpcpp.h>
#include "voice.grpc.pb.h"
#include "llama.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

using voice::LlamaService;
using voice::GenerateRequest;
using voice::GenerateResponse;

// ───────────────────────────────────────── constants ────────────────────────
namespace {
constexpr int WINDOW_TOKENS = 1024;   // keep last 1 k‑tokens in cache

// ---------------------------------------------------------------------------
// SessionStore – manages one llama_context per live conv‑id
class SessionStore {
public:
    SessionStore(llama_model *model, int n_ctx) : model_(model), n_ctx_(n_ctx) {}

    llama_context *get(const std::string &cid) {
        std::lock_guard<std::mutex> lk(mux_);
        auto it = sessions_.find(cid);
        if (it != sessions_.end()) return it->second;
        // first time → create context
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx   = n_ctx_;
        cp.n_batch = 512;
        llama_context *ctx = llama_init_from_model(model_, cp);
        sessions_.emplace(cid, ctx);
        return ctx;
    }

    void drop(const std::string &cid) {
        std::lock_guard<std::mutex> lk(mux_);
        auto it = sessions_.find(cid);
        if (it != sessions_.end()) {
            llama_free(it->second);
            sessions_.erase(it);
        }
    }

private:
    llama_model *model_;
    int n_ctx_;
    std::unordered_map<std::string, llama_context*> sessions_;
    std::mutex mux_;
};

bool startup_test(llama_model *model, const llama_vocab *vocab, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return false;
    const std::string probe = "Hello";
    std::vector<llama_token> toks(-llama_tokenize(vocab, probe.c_str(), probe.size(), nullptr, 0, true, true));
    llama_tokenize(vocab, probe.c_str(), probe.size(), toks.data(), toks.size(), true, true);
    bool ok = llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size())) == 0;
    llama_free(ctx);
    return ok;
}
} // namespace

// ───────────────────────────────────────── service ─────────────────────────
class LlamaServiceImpl : public LlamaService::Service {
public:
    explicit LlamaServiceImpl(const std::string &model_path) {
        int n_ctx = 2048;
        if (const char *env = std::getenv("LLAMA_N_CTX")) n_ctx = std::atoi(env);
        n_ctx = std::max(n_ctx, 512);
        std::cout << "LLAMA_N_CTX=" << n_ctx << "\n";

        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        model_ = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model_) {
            std::cerr << "Cannot load model " << model_path << "\n";
            std::exit(1);
        }
        vocab_ = llama_model_get_vocab(model_);
        if (!startup_test(model_, vocab_, n_ctx)) {
            std::cerr << "startup test failed" << std::endl;
            std::exit(1);
        }
        sessions_ = std::make_unique<SessionStore>(model_, n_ctx);
        std::cout << "llama server ready (sliding window active)" << std::endl;
    }

    ~LlamaServiceImpl() override {
        sessions_.reset();
        if (model_) llama_model_free(model_);
    }

    Status Generate(ServerContext *ctx,
                    const GenerateRequest *req,
                    ServerWriter<GenerateResponse> *w) override {
        // conv‑id header
        std::string cid = "default";
        auto md = ctx->client_metadata().find("conv-id");
        if (md != ctx->client_metadata().end())
            cid.assign(md->second.data(), md->second.length());

        llama_context *llctx = sessions_->get(cid);

        // ── feed incremental user text ──
        if (!req->prompt().empty()) {
            int n_tok = -llama_tokenize(vocab_, req->prompt().c_str(), req->prompt().size(), nullptr, 0, true, true);
            std::vector<llama_token> toks(n_tok);
            llama_tokenize(vocab_, req->prompt().c_str(), req->prompt().size(), toks.data(), n_tok, true, true);
            if (llama_decode(llctx, llama_batch_get_one(toks.data(), n_tok)) != 0)
                return Status(grpc::StatusCode::INTERNAL, "prompt_eval_fail");
        }

        #ifdef llama_sampling_stop_strings
        // ── set stop strings (new API, Apr‑25) ──
        llama_sampling_stop_strings stop;
        stop.add("assistant:");
        llama_set_stop_strings(llctx, &stop);
#endif

        // ── sampling chain ── ──
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        llama_sampler *sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(req->top_p() > 0 ? req->top_p() : 0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(req->temperature() > 0 ? req->temperature() : 0.7f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const int max_tok = req->max_tokens() > 0 ? req->max_tokens() : 256;
        GenerateResponse out;
        char buf[128];
        for (int i = 0; i < max_tok; ++i) {
            llama_token next = llama_sampler_sample(sampler, llctx, -1);
            if (llama_vocab_is_eog(vocab_, next)) break;
            int sz = llama_token_to_piece(vocab_, next, buf, sizeof(buf), 0, true);
            out.set_text(sz > 0 ? std::string(buf, sz) : "");
            out.set_done(false);
            w->Write(out);
            if (llama_decode(llctx, llama_batch_get_one(&next, 1)) != 0) {
                llama_sampler_free(sampler);
                return Status(grpc::StatusCode::INTERNAL, "decode_fail");
            }
        }
        out.set_text(""); out.set_done(true); w->Write(out);
        llama_sampler_free(sampler);

        // ── sliding-window pruning ──
        #if defined(llama_kv_cache_seq_shift)
        // preferred fast path (needs llama.cpp ≥ 2025‑04‑22)
        int total = llama_kv_self_n_tokens(llctx);
        if (total > WINDOW_TOKENS)
            llama_kv_cache_seq_shift(llctx, total - WINDOW_TOKENS);
#else
        // compatibility – clear everything once window exceeded
        if (llama_kv_self_n_tokens(llctx) > WINDOW_TOKENS)
            llama_kv_cache_clear(llctx);
#endif

        if (ctx->IsCancelled()) sessions_->drop(cid);
        return Status::OK;
    }

private:
    llama_model *model_{nullptr};
    const llama_vocab *vocab_{nullptr};
    std::unique_ptr<SessionStore> sessions_;
};

// ───────────────────────────────────────── main ─────────────────────────────
int main(int argc, char **argv) {
    std::string model_path = argc > 1 ? argv[1] : "/app/models/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf";
    std::string dir   = "/app/llama-sockets";
    std::string sock  = dir + "/llama.sock";
    mkdir(dir.c_str(), 0777);
    unlink(sock.c_str());

    LlamaServiceImpl svc(model_path);

    grpc::ServerBuilder b;
    b.AddListeningPort("unix://" + sock, grpc::InsecureServerCredentials());
    b.RegisterService(&svc);
    std::unique_ptr<Server> server(b.BuildAndStart());
    std::ofstream(dir + "/ready").close();
    std::cout << "llama server ready on unix://" << sock << std::endl;
    server->Wait();
    return 0;
}
