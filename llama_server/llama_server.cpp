// llama_server.cpp – llama.cpp gRPC micro‑service (Chat‑template version)
// Uses llama_chat_apply_template() so we no longer hand‑roll "user:/assistant:" tags.
// – one llama_context + message history per conversation
// – 1024‑token sliding KV‑window
// Build:
//   g++ -std=c++17 -O3 -pthread llama_server.cpp \
//       -lgrpc++ -lprotobuf -lllama -ldl -o llama_server

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
#include "common/sampling.h"
#include "common/chat.h"              // <-- chat helpers + struct

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;

using voice::LlamaService;
using voice::GenerateRequest;
using voice::GenerateResponse;

// ───────────────────────────────── constants ───────────────────────────────
namespace {
constexpr int WINDOW_TOKENS = 1024; // keep last 1k tokens in KV‑cache
constexpr char SYSTEM_PROMPT[] =
    "You are a concise conversational voice assistant. "
    "DO NOT invent user turns or ask follow‑up questions. Just answer briefly.";

struct ChatHist {
    std::string role;
    std::string content;
    ChatHist() = default;
    ChatHist(std::string r, std::string c) : role(std::move(r)), content(std::move(c)) {}
};

struct Session {
    llama_context* ctx{nullptr};
    std::vector<ChatHist> msgs;   // user / assistant history with owned strings
};
}

// ---------------------------------------------------------------------------
class SessionStore {
public:
    SessionStore(llama_model* model, int n_ctx): model_(model), n_ctx_(n_ctx) {}

    Session& get(const std::string& cid) {
        std::lock_guard<std::mutex> lk(mux_);
        auto it = map_.find(cid);
        if (it != map_.end()) return it->second;

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_;
        cp.n_batch = 512;
        Session s;
        s.ctx = llama_init_from_model(model_, cp);
        s.msgs = {{"system", SYSTEM_PROMPT}};   // first system message
        map_.emplace(cid, std::move(s));
        return map_.at(cid);
    }

    void drop(const std::string& cid) {
        std::lock_guard<std::mutex> lk(mux_);
        auto it = map_.find(cid);
        if (it != map_.end()) {
            llama_free(it->second.ctx);
            map_.erase(it);
        }
    }
private:
    llama_model* model_;
    int n_ctx_;
    std::unordered_map<std::string,Session> map_;
    std::mutex mux_;
};

// ---------------------------------------------------------------------------
class LlamaServiceImpl : public LlamaService::Service {
public:
    explicit LlamaServiceImpl(const std::string& model_path) {
        int n_ctx = 2048;
        if (const char* e = std::getenv("LLAMA_N_CTX")) n_ctx = std::atoi(e);
        n_ctx = std::max(n_ctx, 512);
        ggml_backend_load_all();
        llama_model_params mp = llama_model_default_params();
        model_ = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model_) throw std::runtime_error("cannot load model");
        vocab_ = llama_model_get_vocab(model_);
        sessions_ = std::make_unique<SessionStore>(model_, n_ctx);
    }

    Status Generate(ServerContext* ctx,
                    const GenerateRequest* req,
                    ServerWriter<GenerateResponse>* w) override {
        // conversation id
        std::string cid = "default";
        auto md = ctx->client_metadata().find("conv-id");
        if (md!=ctx->client_metadata().end())
            cid.assign(md->second.data(), md->second.length());

        Session& s = sessions_->get(cid);
        llama_context* llctx = s.ctx;

        // push user message from API caller
        if (!req->prompt().empty())
            s.msgs.emplace_back("user", req->prompt());

        // ---- format with chat template ----------------------------------
        // ---- build transient view for llama_chat_apply_template --------
        std::vector<llama_chat_message> view;
        view.reserve(s.msgs.size());
        for (auto &m : s.msgs)
            view.push_back({m.role.c_str(), m.content.c_str()});

        const char* tmpl = llama_model_chat_template(model_, nullptr); // default template
        std::string prompt(32*1024,'\0');
        // --- build prompt --------------------------------------------------
int n = llama_chat_apply_template(
            tmpl,
            view.data(),
            view.size(),
            /*add_assistant_tag=*/true,
            prompt.data(),
            static_cast<int32_t>(prompt.size()));
        if (n < 0) return Status(grpc::StatusCode::INTERNAL,"template_fail");
        prompt.resize(n);

        // tokenize & evaluate the whole prompt (assistant tag included)
        std::vector<llama_token> inp(-llama_tokenize(vocab_, prompt.c_str(), prompt.size(), nullptr,0,true,true));
        llama_tokenize(vocab_, prompt.c_str(), prompt.size(), inp.data(), inp.size(), true, true);
        if (llama_decode(llctx, llama_batch_get_one(inp.data(), inp.size())) != 0)
            return Status(grpc::StatusCode::INTERNAL, "prompt_eval_fail");

        // ---- sampler chain ---------------------------------------------
        llama_sampler_chain_params scp = llama_sampler_chain_default_params();
        llama_sampler* sampler = llama_sampler_chain_init(scp);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(req->top_p()>0?req->top_p():0.95f,1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(req->temperature()>0?req->temperature():0.7f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const llama_token eos = llama_vocab_eos(vocab_);
        const int max_tok = req->max_tokens()>0?req->max_tokens():256;
        std::string assistant_reply;
        char buf[128];

        for (int i=0;i<max_tok;++i){
            llama_token next = llama_sampler_sample(sampler,llctx,-1);
            if (next==eos) break;
            int sz = llama_token_to_piece(vocab_,next,buf,sizeof(buf),0,true);
            assistant_reply.append(buf,sz);
            GenerateResponse chunk; chunk.set_text(sz>0?std::string(buf,sz):""); chunk.set_done(false);
            w->Write(chunk);
            if (llama_decode(llctx, llama_batch_get_one(&next,1))!=0){
                llama_sampler_free(sampler); return Status(grpc::StatusCode::INTERNAL,"decode_fail"); }
        }
        GenerateResponse fin; fin.set_text(""); fin.set_done(true); w->Write(fin);
        llama_sampler_free(sampler);

        // store assistant message back into history for next turn
        s.msgs.emplace_back("assistant", assistant_reply);

        // KV‑cache sliding window
        #ifdef llama_kv_cache_seq_shift
        int total = llama_kv_self_n_tokens(llctx);
        if (total>WINDOW_TOKENS) llama_kv_cache_seq_shift(llctx,total-WINDOW_TOKENS);
        #else
        if (llama_kv_self_n_tokens(llctx)>WINDOW_TOKENS) llama_kv_self_clear(llctx);
        #endif

        if (ctx->IsCancelled()) sessions_->drop(cid);
        return Status::OK;
    }
private:
    llama_model* model_{nullptr};
    const llama_vocab* vocab_{nullptr};
    std::unique_ptr<SessionStore> sessions_;
};

// ---------------------------------------------------------------------------
int main(int argc,char** argv){
    std::string model_path = argc>1?argv[1]:"/app/models/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf";
    std::string dir="/app/llama-sockets"; std::string sock=dir+"/llama.sock";
    mkdir(dir.c_str(),0777); unlink(sock.c_str());
    LlamaServiceImpl svc(model_path);
    grpc::ServerBuilder b; b.AddListeningPort("unix://"+sock,grpc::InsecureServerCredentials());
    b.RegisterService(&svc);
    auto server=b.BuildAndStart(); std::ofstream(dir+"/ready").close();
    std::cout<<"llama server ready on unix://"<<sock<<std::endl; server->Wait(); return 0;
}
