// llama_server.cpp – llama.cpp gRPC micro‑service (incremental, template‑based)
// Keeps a growing chat history but **really** decodes only the new tokens each turn.
// Compatible with llama.cpp v0.1.x → v0.2.x (no token‑level chat helpers required).
// – one llama_context per conversation
// – 1 024‑token sliding KV‑window
// Build:
//   g++ -std=c++17 -O3 -pthread llama_server.cpp \
//       -lgrpc++ -lprotobuf -llama -ldl -o llama_server

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

#include <grpcpp/grpcpp.h>
#include "voice.grpc.pb.h"
#include "llama.h"
#include "common/sampling.h"
#include "common/chat.h"   // string‑based helpers (always present)

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

using voice::LlamaService;
using voice::GenerateRequest;
using voice::GenerateResponse;

// ─────────────────────────────────────────────────────────────────────────────
namespace {
constexpr int WINDOW_TOKENS = 1024;
constexpr char SYSTEM_PROMPT[] =
    "You are a concise conversational voice assistant. "
    "DO NOT invent user turns or ask follow‑up questions. Just answer briefly.";

// helper: format ONE chat message → tokenize → append to dest
static void append_msg_tokens(llama_model* model,
                              const llama_vocab* vocab,
                              const char* role,
                              const std::string& content,
                              std::vector<llama_token>& dest)
{
    llama_chat_message msg{role, content.c_str()};
    const char* tmpl = llama_model_chat_template(model, nullptr);

    std::string buf(4096, '\0');
    int n = llama_chat_apply_template(tmpl, &msg, 1, /*add_bos=*/true,
                                      buf.data(), buf.size());
    if(n <= 0) throw std::runtime_error("chat template apply failed");
    buf.resize(n);

    int needed = -llama_tokenize(vocab, buf.c_str(), buf.size(), nullptr, 0, true, true);
    size_t old = dest.size();
    dest.resize(old + needed);
    llama_tokenize(vocab, buf.c_str(), buf.size(), dest.data() + old, needed, true, true);
}

struct Session {
    llama_context*             ctx   {nullptr};
    std::vector<llama_token>   prompt_tokens;   // everything already in KV‑cache
};
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
class SessionStore {
public:
    SessionStore(llama_model* m, int n_ctx): model_(m), n_ctx_(n_ctx){}

    Session& get(const std::string& id){
        std::lock_guard<std::mutex> lock(mu_);
        if(auto it = map_.find(id); it != map_.end()) return it->second;

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx   = n_ctx_;
        cp.n_batch = 512;

        Session s;
        s.ctx = llama_init_from_model(model_, cp);

        // seed with the SYSTEM prompt
        append_msg_tokens(model_, llama_model_get_vocab(model_),
                          "system", SYSTEM_PROMPT, s.prompt_tokens);
        llama_decode(s.ctx, llama_batch_get_one(s.prompt_tokens.data(), (int)s.prompt_tokens.size()));

        return map_.emplace(id, std::move(s)).first->second;
    }

    void drop(const std::string& id){
        std::lock_guard<std::mutex> lock(mu_);
        if(auto it = map_.find(id); it != map_.end()){
            llama_free(it->second.ctx);
            map_.erase(it);
        }
    }
private:
    llama_model*                              model_ {nullptr};
    int                                       n_ctx_ {0};
    std::unordered_map<std::string, Session>  map_;
    std::mutex                                mu_;
};

// ─────────────────────────────────────────────────────────────────────────────
class LlamaServiceImpl : public LlamaService::Service {
public:
    explicit LlamaServiceImpl(const std::string& model_path){
        int n_ctx = 2048;
        if(const char* env = std::getenv("LLAMA_N_CTX")) n_ctx = std::atoi(env);

        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        model_ = llama_model_load_from_file(model_path.c_str(), mp);
        if(!model_) throw std::runtime_error("failed to load model");

        vocab_ = llama_model_get_vocab(model_);
        sessions_ = std::make_unique<SessionStore>(model_, n_ctx);
    }

    // gRPC entry‑point -------------------------------------------------------
    Status Generate(ServerContext* ctx,
                    const GenerateRequest* req,
                    ServerWriter<GenerateResponse>* stream) override
    {
        // ── which conversation? ────────────────────────────────────────────
        std::string cid = "default";
        if(auto md = ctx->client_metadata().find("conv-id"); md != ctx->client_metadata().end())
            cid.assign(md->second.data(), md->second.length());

        Session&       ses  = sessions_->get(cid);
        llama_context* lctx = ses.ctx;

        // ── KV‑window trim BEFORE adding new tokens ─────────────────────────
        int total = llama_kv_self_n_tokens(lctx);
        if(total > WINDOW_TOKENS){
            int shift = total - WINDOW_TOKENS;            // how many oldest tokens to drop
            // remove [0, shift) of sequence‑id 0 (single sequence chat)
            llama_kv_self_seq_rm (lctx, 0, 0, shift);
            // shift remaining positions back so next token lands at = WINDOW_TOKENS‑1
            llama_kv_self_seq_add(lctx, 0, shift, -1, -shift);
            if((size_t)shift <= ses.prompt_tokens.size())
                ses.prompt_tokens.erase(ses.prompt_tokens.begin(), ses.prompt_tokens.begin() + shift);
        }

        // ── add new user message, prefill its tokens ───────────────────────
        std::vector<llama_token> new_toks;
        if(!req->prompt().empty()){
            append_msg_tokens(model_, vocab_, "user", req->prompt(), new_toks);
            ses.prompt_tokens.insert(ses.prompt_tokens.end(), new_toks.begin(), new_toks.end());
            llama_decode(lctx, llama_batch_get_one(new_toks.data(), (int)new_toks.size()));
        }

        // ── sampler chain setup ────────────────────────────────────────────
        llama_sampler_chain_params scp = llama_sampler_chain_default_params();
        llama_sampler* sampler = llama_sampler_chain_init(scp);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(req->top_p()>0?req->top_p():0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(req->temperature()>0?req->temperature():0.7f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const llama_token eos = llama_vocab_eos(vocab_);
        const int max_tok = req->max_tokens()>0 ? req->max_tokens() : 256;

        std::string assistant_reply;
        char        buf[128];

        for(int i=0; i<max_tok; ++i){
            llama_token next = llama_sampler_sample(sampler, lctx, -1);
            if(next == eos) break;

            int sz = llama_token_to_piece(vocab_, next, buf, sizeof(buf), 0, true);
            assistant_reply.append(buf, sz);

            GenerateResponse chunk; chunk.set_text(sz>0?std::string(buf,sz):""); chunk.set_done(false);
            stream->Write(chunk);

            llama_decode(lctx, llama_batch_get_one(&next, 1));
            ses.prompt_tokens.push_back(next);
        }

        GenerateResponse end_msg; end_msg.set_text(""); end_msg.set_done(true); stream->Write(end_msg);
        llama_sampler_free(sampler);

        // ── append assistant message to history ────────────────────────────
        std::vector<llama_token> asst_toks;
        append_msg_tokens(model_, vocab_, "assistant", assistant_reply, asst_toks);
        ses.prompt_tokens.insert(ses.prompt_tokens.end(), asst_toks.begin(), asst_toks.end());
        llama_decode(lctx, llama_batch_get_one(asst_toks.data(), (int)asst_toks.size()));

        if(ctx->IsCancelled()) sessions_->drop(cid);
        return Status::OK;
    }

private:
    llama_model*                                model_   {nullptr};
    const llama_vocab*                          vocab_   {nullptr};
    std::unique_ptr<SessionStore>               sessions_;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv){
    std::string model_path = argc>1 ? argv[1] : "/app/models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf";

    mkdir("/app/llama-sockets", 0777);
    std::string sock = "/app/llama-sockets/llama.sock";
    unlink(sock.c_str());

    LlamaServiceImpl svc(model_path);

    grpc::ServerBuilder builder;
    builder.AddListeningPort("unix://" + sock, grpc::InsecureServerCredentials());
    builder.RegisterService(&svc);

    auto server = builder.BuildAndStart();
    std::ofstream("/app/llama-sockets/ready").close();
    std::cout << "llama server ready on unix://" << sock << std::endl;
    server->Wait();
    return 0;
}
